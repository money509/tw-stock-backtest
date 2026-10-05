"""
daily_squeeze_signals.py —— 擠壓+KDJ(--squeeze-kdj-fixed選定設定)每日訊號掃描器。

用途：每個交易日收盤後(GitHub Actions排程，台北時間19:00)掃一次全部股票期貨標的，
列出「明天開盤要不要掛單、掛哪幾檔、掛多少錢、成交後停損/停利掛多少」。使用者隔天開盤前
打開 signals/latest.md：有訊號就照做，沒訊號就什麼都不做。

跟回測(compare_breakout.py --squeeze-kdj-fixed，SQUEEZE_KDJ_FIXED_SETTING + execution_model=
"limit_1tick")用的是「同一套」程式碼與規則，不另外重寫一份：
  - 進場訊號：squeeze_kdj_signal.compute_squeeze_kdj_features()的EntryFlag(同一個狀態機)，
    「最後一根K棒(=as_of那天)EntryFlag=True」就是訊號，隔天(下一個交易日)進場。
  - 排名：觸發K棒當天漲幅(close_t/close_(t-1)-1)由大到小，同分時維持universe順序(跟回測的
    穩定排序一致)。
  - 限價：觸發K棒收盤 + 1檔(squeeze_kdj_signal.taiwan_tick_size，tick用收盤價的級距)，
    開盤高於限價不追(回測的skipped_limit_not_filled)。
  - ATR：mean_reversion_engine.compute_atr_correct(df, period=14)在t日(觸發K棒)的值——回測的
    precompute把ATR shift(1)對齊到進場日(t+1)那一列，查到的就是t日的ATR，同一個數字。
  - 停損 = 成交價 - 1.0 x ATR；停利 = 成交價 + 3.0 x ATR；最長持有20個交易日(進場日算第1天，
    第20天收盤平倉)；同時最多3檔(使用者實際打算1~2檔)；單筆保證金不超過資金35%。
  一致性由 test_daily_squeeze_signals.py 用長段合成資料逐日比對回測的事件表來保證。

唯一刻意的差異(誠實聲明)：價格欄位(OHLC)有NaN的列，這裡直接丟掉再算指標；回測的
precompute是「保留這些列算指標、只是當天不交易」(NaN會讓BB/KC的rolling視窗在之後約20天
都是NaN，等於那檔股票有一段時間不會出訊號，觸發K棒前一天是NaN時漲幅也是NaN、事件被丟掉)。
丟掉NaN列比較貼近「那天其實有交易、只是資料源漏了」的真實情況；沒有NaN列的股票(絕大多數)
兩者完全相同。

使用：
  python daily_squeeze_signals.py                      # 今天(台北時間)
  python daily_squeeze_signals.py --as-of 2026-10-02   # 回補/測試某一天
"""
import argparse
import datetime
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from data_loader import load_price_data, STOCK_FUTURES_WHITELIST
from mean_reversion_engine import compute_atr_correct, DEFAULT_MARGIN_CAP_RATIO, STOP_LOSS_COOLDOWN_DAYS
from squeeze_kdj_signal import compute_squeeze_kdj_features, taiwan_tick_size
from taifex_universe import STOCK_FUTURES_UNIVERSE, estimate_margin, get_contract_multiplier
from compare_breakout import SQUEEZE_KDJ_FIXED_SETTING

FIXED = SQUEEZE_KDJ_FIXED_SETTING
ATR_PERIOD = FIXED["atr_period"]            # 14
ATR_STOP_MULT = FIXED["atr_stop_mult"]      # 1.0
ATR_TARGET_MULT = FIXED["atr_target_mult"]  # 3.0
MAX_HOLD_DAYS = FIXED["max_hold_days"]      # 20
TOP_N = FIXED["top_n"]                      # 3
MAX_CONCURRENT = FIXED["max_concurrent_positions"]  # 3
COOLDOWN_CALENDAR_DAYS = STOP_LOSS_COOLDOWN_DAYS * 2  # 回測：停損出場日 + 10個日曆天之前不再進場
TOTAL_MARGIN_CAP_RATIO = min(DEFAULT_MARGIN_CAP_RATIO * MAX_CONCURRENT, 0.9)  # 回測的整體保證金上限
MIN_BARS = 60  # 跟precompute_squeeze_kdj_features_by_code()同一個門檻：資料太短的股票不算
REFERENCE_CODE = "2330"
STALE_SHARE_WARN = 0.10  # 超過10%的股票最新一根不是as_of，額外提醒

DEFAULT_CAPITAL = 200_000
DEFAULT_LOOKBACK_DAYS = 400
# 為什麼400個日曆天(約270個交易日)：
#  - 硬性暖身：BB20/KC20(ATR20 SMA)要20根、KDJ用25日高低點、ATR14要14根、擠壓回看10根、
#    武裝最多等15根 → 大約60根之後狀態機就跟「從更早開始算」的結果無關(武裝狀態最多延續15根)。
#  - 有無限記憶的兩個量：KC的EMA20(adjust=False，起始值=第一根收盤)誤差每根乘(19/21)，
#    KDJ的K(K_0=50)誤差每根乘(2/3)。約270根之後EMA20起始誤差剩(19/21)^270 ≈ 2e-12(相對)，
#    K剩(2/3)^270 ≈ 0，都遠小於會改變「BB是否在KC內/K是否<20」這種比較結果的量級。
#  - 合成資料測試(test_daily_squeeze_signals.py)用這個預設回看窗口逐日比對「用完整歷史算」的
#    回測事件表，代號、排名、限價、ATR全部一致。
#  - 多抓一點完全不花成本(一檔一年的日K只有幾KB)，所以取比60根寬很多的值。

TAIPEI_TZ = "Asia/Taipei"
WEEKDAY_ZH = "一二三四五六日"


# ----------------------------------------------------------------------------
# 價格/檔位工具
# ----------------------------------------------------------------------------
def floor_to_tick(price: float) -> float:
    """往下取到「這個價位級距」的合法檔位(台股/股票期貨升降單位表，見taiwan_tick_size)。
    級距邊界(10/50/100/500/1000)本身都是下面每一級tick的整數倍，所以用price自己的tick往下取，
    結果一定是 <= price 的最大合法價位。"""
    tick = taiwan_tick_size(price)
    k = math.floor(price / tick + 1e-9)
    return round(k * tick, 2)


def round_to_tick(price: float) -> float:
    """四捨五入到最接近的合法檔位(只用在「收盤+1檔」這種本來就該落在檔位上、只是有浮點雜訊的價格)。"""
    tick = taiwan_tick_size(price)
    return round(round(price / tick) * tick, 2)


def fmt_price(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "-"
    s = f"{p:,.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def fmt_date(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.date().isoformat()}({WEEKDAY_ZH[d.weekday()]})"


# ----------------------------------------------------------------------------
# 交易日(近似)：repo裡沒有台股休市行事曆，用週一~週五近似，不扣國定假日
# ----------------------------------------------------------------------------
def next_business_day(d) -> pd.Timestamp:
    return pd.Timestamp(d) + pd.offsets.BDay(1)


def time_exit_date(entry_date) -> pd.Timestamp:
    """最長持有到期日：進場日算第1天，第MAX_HOLD_DAYS(20)個交易日收盤平倉(回測hold_days>=20的那天)。
    用週一~週五近似，沒扣國定假日——遇到休市要往後順延，以實際第20個交易日為準。"""
    return pd.Timestamp(entry_date) + pd.offsets.BDay(MAX_HOLD_DAYS - 1)


# ----------------------------------------------------------------------------
# 核心：掃描(純函式，不碰網路/檔案，測試直接呼叫)
# ----------------------------------------------------------------------------
def build_universe(max_stocks: int = 0) -> dict:
    """跟compare_breakout.main()同一個做法：前max_stocks檔 + 一定包含2330(參考日曆)。
    順序很重要：同分時的排名先後跟回測一樣取決於universe順序。"""
    universe = dict(STOCK_FUTURES_UNIVERSE)
    if max_stocks > 0:
        universe = dict(list(universe.items())[:max_stocks])
        if REFERENCE_CODE not in universe:
            universe[REFERENCE_CODE] = STOCK_FUTURES_UNIVERSE[REFERENCE_CODE]
    return universe


def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """丟掉OHLC任一欄是NaN的列(見模組docstring「唯一刻意的差異」)，日期排序。"""
    if df is None or df.empty:
        return df
    df = df.sort_index()
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def compute_signal_row(code: str, df: pd.DataFrame, capital: float) -> dict:
    """df的最後一根K棒是觸發K棒(EntryFlag=True)時，算出這一檔的全部下單資訊；不是訊號回傳None。
    df必須已經clean_price_df()過。"""
    if df is None or len(df) < MIN_BARS:
        return None
    features = compute_squeeze_kdj_features(df)
    if not bool(features["EntryFlag"].iloc[-1]):
        return None

    close_t = float(df["Close"].iloc[-1])
    close_prev = float(df["Close"].iloc[-2])
    trigger_return = close_t / close_prev - 1
    if not np.isfinite(trigger_return):
        return None  # 回測：觸發K棒漲幅是NaN的事件會被丟掉
    atr = float(compute_atr_correct(df, period=ATR_PERIOD).iloc[-1])

    tick = taiwan_tick_size(close_t)
    limit_raw = close_t + tick                 # 回測的limit_price(未取整，用來比對一致性)
    limit = round_to_tick(limit_raw)           # 實際掛單價

    signal_date = pd.Timestamp(df.index[-1])
    entry_date = next_business_day(signal_date)
    mult = get_contract_multiplier(code, limit)
    margin = estimate_margin(code, limit, 1)
    margin_cap = capital * DEFAULT_MARGIN_CAP_RATIO
    atr_ok = np.isfinite(atr) and atr > 0

    row = {
        "code": code,
        "name": STOCK_FUTURES_WHITELIST.get(code, ""),
        "signal_date": signal_date.date().isoformat(),
        "close": close_t,
        "prev_close": close_prev,
        "trigger_return": trigger_return,
        "tick": tick,
        "limit_raw": limit_raw,
        "limit_price": limit,
        "atr": atr,
        "stop_dist": ATR_STOP_MULT * atr if atr_ok else np.nan,
        "target_dist": ATR_TARGET_MULT * atr if atr_ok else np.nan,
        "stop_if_fill_at_limit": floor_to_tick(limit - ATR_STOP_MULT * atr) if atr_ok and limit - ATR_STOP_MULT * atr > 0 else np.nan,
        "target_if_fill_at_limit": floor_to_tick(limit + ATR_TARGET_MULT * atr) if atr_ok else np.nan,
        "contract": "小型" if mult == 100 else "標準",
        "multiplier": mult,
        "stop_dist_ntd_per_lot": ATR_STOP_MULT * atr * mult if atr_ok else np.nan,
        "target_dist_ntd_per_lot": ATR_TARGET_MULT * atr * mult if atr_ok else np.nan,
        "margin_est_1lot": margin,
        "margin_over_cap": bool(margin > margin_cap),
        "atr_invalid": not atr_ok,
        "entry_date": entry_date.date().isoformat(),
        "time_exit_date": time_exit_date(entry_date).date().isoformat(),
    }
    # 回測會直接略過的：ATR算不出來(skipped_invalid_stop)、單筆保證金超過35%(skipped_single_margin_cap)
    row["backtest_would_skip"] = bool(row["margin_over_cap"] or row["atr_invalid"])
    return row


def scan(price_data: dict, universe: dict, as_of, capital: float = DEFAULT_CAPITAL) -> dict:
    """掃描全部標的，回傳結果dict(markdown/csv都從這裡產生)。as_of：訊號日(收盤那天)。"""
    as_of = pd.Timestamp(as_of).normalize()
    cleaned = {}
    for code in universe:
        df = price_data.get(code)
        if df is None or df.empty:
            continue
        df = clean_price_df(df)
        df = df[df.index <= as_of]  # 保險：就算資料源多回傳了as_of之後的列也不偷看
        if not df.empty:
            cleaned[code] = df

    if REFERENCE_CODE in cleaned:
        latest = pd.Timestamp(cleaned[REFERENCE_CODE].index[-1])
        latest_source = REFERENCE_CODE
    elif cleaned:
        latest = max(pd.Timestamp(df.index[-1]) for df in cleaned.values())
        latest_source = "全部標的最大值(2330沒有資料)"
    else:
        latest = None
        latest_source = "無"
    data_fresh = latest is not None and latest.normalize() == as_of

    stale_codes = [c for c, df in cleaned.items() if pd.Timestamp(df.index[-1]).normalize() != as_of]
    signals = []
    for code, df in cleaned.items():
        if pd.Timestamp(df.index[-1]).normalize() != as_of:
            continue  # 最後一根不是as_of(停牌/資料沒更新)：訊號只看「as_of那根」，舊的訊號不算
        row = compute_signal_row(code, df, capital)
        if row is not None:
            signals.append(row)

    # 回測：Python穩定排序、reverse=True，同分維持universe順序
    signals.sort(key=lambda r: r["trigger_return"], reverse=True)
    for i, r in enumerate(signals, start=1):
        r["rank"] = i

    return {
        "as_of": as_of,
        "capital": capital,
        "universe_size": len(universe),
        "downloaded": len([c for c in universe if price_data.get(c) is not None and not price_data[c].empty]),
        "failed_codes": [c for c in universe if price_data.get(c) is None or price_data[c].empty],
        "usable": len(cleaned),
        "stale_codes": stale_codes,
        "latest_date": latest,
        "latest_source": latest_source,
        "data_fresh": data_fresh,
        "signals": signals,
        "entry_date": next_business_day(as_of),
    }


# ----------------------------------------------------------------------------
# 輸出
# ----------------------------------------------------------------------------
CSV_COLUMNS = [
    "rank", "code", "name", "signal_date", "close", "prev_close", "trigger_return", "tick", "limit_price",
    "atr", "stop_dist", "target_dist", "stop_if_fill_at_limit", "target_if_fill_at_limit", "contract",
    "multiplier", "stop_dist_ntd_per_lot", "target_dist_ntd_per_lot", "margin_est_1lot", "margin_over_cap",
    "atr_invalid", "backtest_would_skip", "entry_date", "time_exit_date", "data_fresh",
]


def signals_frame(result: dict) -> pd.DataFrame:
    rows = [{**r, "data_fresh": result["data_fresh"]} for r in result["signals"]]
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def _rules_section() -> list:
    cd = COOLDOWN_CALENDAR_DAYS
    return [
        "## 每日操作規則",
        "",
        "**掛單(開盤前)**",
        "1. 每檔只做1口，價格用上面的「限價買進」，**開盤沒成交就不追**，等下一個訊號。",
        f"2. 空位有N個，就從第1名往下取N檔；跳過：標⛔的、你已經持有的、停損出場後{cd}個日曆天內的"
        f"(例：10/1停損，10/{1 + cd}起才能再進同一檔)。",
        f"3. 回測最多同時{MAX_CONCURRENT}檔、你打算1~2檔：空位數用你自己的上限算。全部持倉保證金合計不超過資金的{TOTAL_MARGIN_CAP_RATIO:.0%}。",
        f"4. 回測裡如果排名較前的那檔開盤高於限價沒成交，會改試下一名(最多試到第{TOP_N}名，不含已持有/冷卻中)。"
        "實盤只能近似：開盤後下一名的價格如果還≤它的限價，可以改掛它；已經漲過限價就放棄。",
        "",
        "**成交後(馬上做)**",
        f"5. 用**實際成交價**重算並立刻掛條件單：停損 = 成交價 − {ATR_STOP_MULT:g}×ATR，停利 = 成交價 + {ATR_TARGET_MULT:g}×ATR，"
        "兩個都往下取到合法檔位。成交當天就生效(回測進場當天就會檢查停損/停利)。",
        f"6. 停損、停利都沒碰到：持有到「最長持有到期日」(進場日算第1天的第{MAX_HOLD_DAYS}個交易日)收盤前平倉。"
        "到期日是用週一~週五估的，遇到國定假日要往後順延。",
        "7. 回測細節：明天收盤就要到期平倉的部位，回測把它的名額算成明天可用。",
        "",
        "**你自己定的停損規則(不是回測規則，是你選的風控)**",
        "8. 帳戶從開始實盤累計虧損達 **NT$30,000** → 全部停止。",
        "9. 做滿 **30筆** 交易後，如果這30筆的獲利因子 **PF<1** → 全部停止。",
        "",
        "⚠️ 這個策略**沒有通過回測驗證**(2018–2023 未見過區段、真實成交情境 PF≈0.67)，"
        "實盤是實驗，部位保持最小。",
    ]


def _signal_block(r: dict, data_fresh: bool) -> list:
    name = f" {r['name']}" if r["name"] else ""
    head = f"### {r['rank']}. {r['code']}{name}"
    if r["backtest_would_skip"]:
        head += " ⛔"
    lines = [head, ""]
    if r["margin_over_cap"]:
        lines.append(f"- ⛔ **超過單筆保證金上限，回測會略過，不建議下單**"
                     f"(保證金約 NT${r['margin_est_1lot']:,.0f} > 資金35%)")
    if r["atr_invalid"]:
        lines.append("- ⛔ ATR算不出來(資料異常)，回測會略過，不建議下單")
    lines.append(f"- 觸發K棒漲幅 **{r['trigger_return']:+.2%}**｜收盤 {fmt_price(r['close'])}")
    if not data_fresh:
        buy = f"~~{fmt_price(r['limit_price'])}~~(資料不可信，不可下單)"
    elif r["backtest_would_skip"]:
        buy = f"{fmt_price(r['limit_price'])} ⛔"
    else:
        buy = f"**{fmt_price(r['limit_price'])}**"
    lines.append(f"- 限價買進 {buy}(收盤+1檔{fmt_price(r['tick'])})，開盤高於此價不追")
    if not r["atr_invalid"]:
        lines.append(f"- 若成交在 {fmt_price(r['limit_price'])}：停損 **{fmt_price(r['stop_if_fill_at_limit'])}**"
                     f"｜停利 **{fmt_price(r['target_if_fill_at_limit'])}**")
        lines.append(f"- ATR14 = {r['atr']:.2f} → 停損 = 成交價 − {r['stop_dist']:.2f}，"
                     f"停利 = 成交價 + {r['target_dist']:.2f}")
        lines.append(f"- 每口：停損距離約 NT${r['stop_dist_ntd_per_lot']:,.0f}｜停利距離約 NT${r['target_dist_ntd_per_lot']:,.0f}")
    lines.append(f"- {r['contract']}契約 {r['multiplier']:,}股｜保證金約 NT${r['margin_est_1lot']:,.0f}")
    lines.append(f"- 最長持有到 {fmt_date(r['time_exit_date'])} 收盤(約略，遇假日順延)")
    lines.append("")
    return lines


def render_markdown(result: dict) -> str:
    as_of = result["as_of"]
    sigs = result["signals"]
    fresh = result["data_fresh"]
    latest = result["latest_date"]
    lines = [f"# 擠壓+KDJ 每日訊號 {fmt_date(as_of)}", ""]

    if not fresh:
        lines += [
            "> ## ⚠️ 資料還沒更新到今天，這份清單不可信，請晚點重跑",
            f"> 最新資料日期是 {fmt_date(latest) if latest is not None else '無資料'}"
            f"(參考：{result['latest_source']})，不是 {fmt_date(as_of)}。"
            "可能是資料源還沒更新，或今天休市。**下面不列可下單的訊號。**",
            "",
        ]

    n_actionable = sum(1 for r in sigs if not r["backtest_would_skip"])
    lines += [
        f"- 訊號日 {as_of.date().isoformat()}｜資料最新 {latest.date().isoformat() if latest is not None else '無'}"
        f" {'✅' if fresh else '❌'}",
        f"- 下單日 {fmt_date(result['entry_date'])} 開盤前掛單",
        f"- 下載 {result['downloaded']}/{result['universe_size']} 檔"
        + (f"(**{len(result['failed_codes'])} 檔失敗**)" if result["failed_codes"] else ""),
    ]
    n_stale = len(result["stale_codes"])
    if fresh and result["usable"] and n_stale / result["usable"] > STALE_SHARE_WARN:
        lines.append(f"- ⚠️ {n_stale} 檔的最新資料不是今天(停牌或沒更新)，這些股票今天就算有訊號也看不到")
    elif n_stale and fresh:
        lines.append(f"- {n_stale} 檔最新資料不是今天(多半是停牌)，不列入")
    lines.append(f"- 資金假設 NT${result['capital']:,.0f}")

    if fresh:
        lines.append(f"- **今天訊號：{len(sigs)} 檔**" + (f"(可下單 {n_actionable} 檔)" if len(sigs) != n_actionable else ""))
    lines.append("")

    if fresh and not sigs:
        lines += ["## ✅ 今天沒有訊號，明天不用下單", ""]
    elif fresh and n_actionable == 0:
        lines += ["## ✅ 今天的訊號回測都會略過(⛔)，明天不用下單", ""]
    if sigs:
        lines += ["---", ""]
        if not fresh:
            lines += ["(以下僅供參考，不可據此下單)", ""]
        for r in sigs:
            lines += _signal_block(r, data_fresh=fresh)

    lines += ["---", ""] + _rules_section() + [""]
    return "\n".join(lines)


def write_outputs(result: dict, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    md = render_markdown(result)
    day = result["as_of"].date().isoformat()
    paths = {
        "latest": os.path.join(output_dir, "latest.md"),
        "md": os.path.join(output_dir, f"{day}.md"),
        "csv": os.path.join(output_dir, f"{day}.csv"),
    }
    for key in ("latest", "md"):
        with open(paths[key], "w", encoding="utf-8") as f:
            f.write(md)
    signals_frame(result).to_csv(paths["csv"], index=False, encoding="utf-8-sig")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    return {"markdown": md, **paths}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def today_in_taipei() -> datetime.date:
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo(TAIPEI_TZ)).date()
    except Exception:
        # 沒有時區資料庫時退回UTC+8(台灣沒有夏令時間，固定+8)
        return (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).date()


def download_window(as_of: datetime.date, lookback_days: int):
    """yfinance的end是「不含」(exclusive)：要拿到as_of那根K棒，end必須是as_of+1天。"""
    start = (as_of - datetime.timedelta(days=lookback_days)).isoformat()
    end = (as_of + datetime.timedelta(days=1)).isoformat()
    return start, end


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="擠壓+KDJ(固定設定)每日訊號掃描")
    p.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, help="帳戶資金(新台幣)，用來判斷單筆保證金35%%上限")
    p.add_argument("--as-of", default=None, help="訊號日YYYY-MM-DD，預設今天(台北時間)")
    p.add_argument("--max-stocks", type=int, default=0, help="只掃前N檔(0=全部)，測試用")
    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                   help="往前抓幾個日曆天的資料當指標暖身(預設400，見程式碼註解)")
    p.add_argument("--output-dir", default="signals", help="輸出資料夾")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    as_of = datetime.date.fromisoformat(args.as_of) if args.as_of else today_in_taipei()
    universe = build_universe(args.max_stocks)
    start, end = download_window(as_of, args.lookback_days)
    print(f"掃描 {len(universe)} 檔，資料區間 {start} ~ {as_of.isoformat()}(yfinance end={end}，不含)", flush=True)

    # refresh=True + 暫存快取資料夾：保證一定重新下載(不會用到同一天稍早、資料還沒出來時存下的快取)，
    # 也不會在data_cache/每天留下一批之後用不到的CSV。
    tmp_cache = tempfile.mkdtemp(prefix="daily_squeeze_cache_")
    try:
        price_data = load_price_data(universe, start, end, refresh=True, cache_dir=tmp_cache)
    finally:
        shutil.rmtree(tmp_cache, ignore_errors=True)

    result = scan(price_data, universe, as_of, capital=args.capital)
    out = write_outputs(result, args.output_dir)
    print(out["markdown"], flush=True)
    print(f"已寫入：{out['latest']}、{out['md']}、{out['csv']}", flush=True)
    return {"result": result, **out}


if __name__ == "__main__":
    main()
    sys.exit(0)
