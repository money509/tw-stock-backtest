"""
daily_squeeze_signals.py —— 擠壓+KDJ(實盤設定SQUEEZE_KDJ_LIVE_SETTINGS)每日訊號掃描器。

用途：每個交易日收盤後(GitHub Actions排程，台北時間19:00)掃一次全部股票期貨標的，
列出「明天開盤要不要掛單、掛哪幾檔、掛多少錢、成交後停損/停利掛多少」。使用者隔天開盤前
打開 signals/latest.md：有訊號就照做，沒訊號就什麼都不做。

設定來自compare_breakout.SQUEEZE_KDJ_LIVE_SETTINGS，依--positions(同時最多持有幾檔，1或2)二選一：
  --positions 1(預設)：停損1.5倍ATR、停利4.0倍ATR、同時最多1檔；
  --positions 2      ：停損1.5倍ATR、停利3.0倍ATR、同時最多2檔；
兩組都是限價 = 訊號日收盤+2檔(回測execution_model="limit_ticks"、entry_limit_ticks=2)。
跟回測(run_squeeze_kdj_capital_constrained_backtest + 上面的設定)用的是「同一套」程式碼與規則，不另外重寫一份：
  - 進場訊號：squeeze_kdj_signal.compute_squeeze_kdj_features()的EntryFlag(同一個狀態機)，
    「最後一根K棒(=as_of那天)EntryFlag=True」就是訊號，隔天(下一個交易日)進場。
  - 排名：觸發K棒當天漲幅(close_t/close_(t-1)-1)由大到小，同分時維持universe順序(跟回測的
    穩定排序一致)。
  - 限價：觸發K棒收盤往上走2檔(squeeze_kdj_signal.squeeze_kdj_entry_limit_price(..., "limit_ticks",
    entry_limit_ticks=2)，跟回測同一個函式；逐檔走，跨價位級距時tick跟著變，例如49.95+2檔=50.1)，
    開盤高於限價不追(回測的skipped_limit_not_filled)。
  - ATR：mean_reversion_engine.compute_atr_correct(df, period=14)在t日(觸發K棒)的值——回測的
    precompute把ATR shift(1)對齊到進場日(t+1)那一列，查到的就是t日的ATR，同一個數字。
  - 停損 = 成交價 - 1.5 x ATR；停利 = 成交價 + 4.0(1檔)/3.0(2檔) x ATR；最長持有20個交易日
    (進場日算第1天，第20天收盤平倉)；同時最多1或2檔；單筆保證金不超過資金35%。
  一致性由 test_daily_squeeze_signals.py 用長段合成資料逐日比對回測的事件表來保證。

唯一刻意的差異(誠實聲明)：價格欄位(OHLC)有NaN的列，這裡直接丟掉再算指標；回測的
precompute是「保留這些列算指標、只是當天不交易」(NaN會讓BB/KC的rolling視窗在之後約20天
都是NaN，等於那檔股票有一段時間不會出訊號，觸發K棒前一天是NaN時漲幅也是NaN、事件被丟掉)。
丟掉NaN列比較貼近「那天其實有交易、只是資料源漏了」的真實情況；沒有NaN列的股票(絕大多數)
兩者完全相同。

兩個清單(同一批訊號、同樣的進場/排名/限價/停損/停利，各自篩選、各自重新排名、各自算空位)：
  1. 個股期貨清單(主要)：實際交易的契約(有小型用小型，否則標準；同get_contract_multiplier)要夠流動：
     5個交易日平均成交量(全部月份、一般+盤後加總、不含價差委託) ≥ --fut-min-volume(預設100口)
     且 近月未平倉 ≥ --fut-min-oi(預設300口)。資料來自taifex_futures_loader.py(期交所)。
     **fail closed**：期交所資料抓不到/解析失敗 → 期貨清單一檔都不列，顯示大警告(不是「沒有訊號」)。
     不夠流動的訊號不逐檔列出，只寫一行「另有N檔訊號因期貨成交量不足未列出」(代碼只寫在latest.md)。
  2. 現股清單(參考)：股票20日平均成交值(收盤x成交量) ≥ --stock-min-turnover(預設5000萬)；
     每筆固定 --stock-amount(預設10萬，可零股)，股數=金額÷限價無條件捨去，<1股標⛔；
     附來回成本估計(手續費0.1425%x2 + 證交稅0.3%)。回測裡現股版含成本大約只是打平，所以只當參考。
  Telegram每天兩則：signals/telegram_futures.txt(期貨，主要)、signals/telegram_stock.txt(現股)；
  signals/telegram.txt = 期貨那一則(向下相容)。

使用：
  python daily_squeeze_signals.py                      # 今天(台北時間)
  python daily_squeeze_signals.py --as-of 2026-10-02   # 回補/測試某一天
  python daily_squeeze_signals.py --positions 2        # 同時最多2檔(停利3.0倍ATR)
  python daily_squeeze_signals.py --stock-amount 50000 --fut-min-volume 200 --fut-min-oi 500
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
from squeeze_kdj_signal import compute_squeeze_kdj_features, taiwan_tick_size, squeeze_kdj_entry_limit_price
from taifex_universe import STOCK_FUTURES_UNIVERSE, estimate_margin, get_contract_multiplier
import taifex_futures_loader as tfl
from taifex_futures_loader import load_futures_liquidity
from compare_breakout import (
    SQUEEZE_KDJ_LIVE_SETTINGS, SQUEEZE_KDJ_LIVE_DEFAULT_POSITIONS, SQUEEZE_KDJ_LIVE_REFERENCE_PF_2018_2025,
    SQUEEZE_KDJ_LIVE_CUMULATIVE_VARIANTS, squeeze_kdj_live_setting_label,
)

VALID_POSITIONS = tuple(sorted(SQUEEZE_KDJ_LIVE_SETTINGS))  # (1, 2)
DEFAULT_POSITIONS = SQUEEZE_KDJ_LIVE_DEFAULT_POSITIONS      # 1
# 兩組共用的欄位(變體B、ATR14、最長20天、top_n=3、限價+2檔)直接從設定讀；停利/持倉數依positions。
_COMMON = SQUEEZE_KDJ_LIVE_SETTINGS[DEFAULT_POSITIONS]
ATR_PERIOD = _COMMON["atr_period"]            # 14
ATR_STOP_MULT = _COMMON["atr_stop_mult"]      # 1.5(兩組相同)
MAX_HOLD_DAYS = _COMMON["max_hold_days"]      # 20
TOP_N = _COMMON["top_n"]                      # 3
EXECUTION_MODEL = _COMMON["execution_model"]  # "limit_ticks"
ENTRY_LIMIT_TICKS = _COMMON["entry_limit_ticks"]  # 2
assert all(s[k] == _COMMON[k] for s in SQUEEZE_KDJ_LIVE_SETTINGS.values()
           for k in ("atr_period", "atr_stop_mult", "max_hold_days", "top_n", "execution_model",
                     "entry_limit_ticks", "variant", "ranking_rule", "entry_filter", "lots"))
COOLDOWN_CALENDAR_DAYS = STOP_LOSS_COOLDOWN_DAYS * 2  # 回測：停損出場日 + 10個日曆天之前不再進場
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

# ----------------------------------------------------------------------------
# 兩個清單(同一批訊號)：個股期貨清單 / 現股清單
# ----------------------------------------------------------------------------
# 期貨清單：實際交易的那種契約(小型/標準，同get_contract_multiplier)流動性要夠，資料來自taifex_futures_loader。
DEFAULT_FUT_MIN_VOLUME = 100   # 5個交易日平均成交量(全部月份、一般+盤後加總)，口
DEFAULT_FUT_MIN_OI = 300       # 近月未平倉，口
# 現股清單：每筆固定金額買現股(可零股)，股票本身也要有基本流動性。
DEFAULT_STOCK_AMOUNT = 100_000
DEFAULT_STOCK_MIN_TURNOVER = 50_000_000  # 20日平均成交值(收盤x成交量)，新台幣
TURNOVER_DAYS = 20
STOCK_COMMISSION_RATE = 0.001425   # 手續費(買、賣各一次)，牌告；很多券商有折扣
STOCK_SELL_TAX_RATE = 0.003        # 證券交易稅(賣出時)
FUTURES_FAIL_MESSAGE = tfl.FAILURE_MESSAGE
STOCK_BACKTEST_WARNING = (
    "回測(每筆固定NT$100,000、含手續費0.1425%x2+證交稅0.3%)：現股版大約只是打平——"
    "最多1檔、停利4倍時 2018–2022 PF≈1.12、2023年起PF≈1.02；最多2檔、停利3倍約1.0。"
    "成本吃掉大部分優勢，這個清單請當參考。"
)
ODD_LOT_NOTE = "盤中零股的流動性比整張差很多(成交量小、買賣價差大)，零股限價單可能掛不到。"


def live_setting(positions: int) -> dict:
    """依「同時最多持有幾檔」取實盤設定；只接受1或2(其他值直接ValueError，不默默套用別的設定)。"""
    if isinstance(positions, bool) or positions not in SQUEEZE_KDJ_LIVE_SETTINGS:
        raise ValueError(f"positions必須是{VALID_POSITIONS}其中之一，收到{positions!r}")
    return SQUEEZE_KDJ_LIVE_SETTINGS[positions]


def setting_label(positions: int) -> str:
    """報告/Telegram最上面那一行：「設定：同時最多1檔｜停損1.5倍ATR｜停利4.0倍ATR｜限價收盤+2檔」。"""
    live_setting(positions)
    return "設定：" + squeeze_kdj_live_setting_label(positions)


def total_margin_cap_ratio(positions: int) -> float:
    """回測的整體保證金上限：min(35% x 最多持倉數, 90%)。"""
    return min(DEFAULT_MARGIN_CAP_RATIO * live_setting(positions)["max_concurrent_positions"], 0.9)
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
    """四捨五入到最接近的合法檔位(只用在「收盤+2檔」這種本來就該落在檔位上、只是有浮點雜訊的價格)。"""
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


def entry_limit_raw(close_t: float) -> float:
    """回測的限價(未取整)：跟run_squeeze_kdj_capital_constrained_backtest同一個函式、同一組參數。"""
    return squeeze_kdj_entry_limit_price(close_t, EXECUTION_MODEL, entry_limit_ticks=ENTRY_LIMIT_TICKS)


def average_turnover(df: pd.DataFrame, days: int = TURNOVER_DAYS) -> float:
    """最近days根K棒的平均成交值(收盤x成交量，新台幣；yfinance台股Volume單位是股)。
    沒有Volume欄或有效值不到一半 → NaN(現股清單會因此不列，fail closed)。"""
    if df is None or "Volume" not in df.columns or df.empty:
        return np.nan
    tail = df.tail(days)
    val = (tail["Close"].astype(float) * tail["Volume"].astype(float)).dropna()
    if len(val) < max(1, days // 2):
        return np.nan
    return float(val.mean())


def compute_signal_row(code: str, df: pd.DataFrame, capital: float,
                       positions: int = DEFAULT_POSITIONS) -> dict:
    """df的最後一根K棒是觸發K棒(EntryFlag=True)時，算出這一檔的全部下單資訊；不是訊號回傳None。
    df必須已經clean_price_df()過。positions：同時最多持有幾檔(1或2)，決定停利倍數。"""
    setting = live_setting(positions)
    stop_mult = setting["atr_stop_mult"]
    target_mult = setting["atr_target_mult"]
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

    tick = taiwan_tick_size(close_t)           # 收盤價那一級的tick(只顯示用；+2檔逐檔走，可能跨級距)
    limit_raw = entry_limit_raw(close_t)       # 回測的limit_price(未取整，用來比對一致性)
    limit = round_to_tick(limit_raw)           # 實際掛單價

    signal_date = pd.Timestamp(df.index[-1])
    entry_date = next_business_day(signal_date)
    mult = get_contract_multiplier(code, limit)
    margin = estimate_margin(code, limit, 1)
    margin_cap = capital * DEFAULT_MARGIN_CAP_RATIO
    atr_ok = np.isfinite(atr) and atr > 0
    turnover_20d = average_turnover(df)

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
        "positions": positions,
        "atr_stop_mult": stop_mult,
        "atr_target_mult": target_mult,
        "entry_limit_ticks": ENTRY_LIMIT_TICKS,
        "stop_dist": stop_mult * atr if atr_ok else np.nan,
        "target_dist": target_mult * atr if atr_ok else np.nan,
        "stop_if_fill_at_limit": floor_to_tick(limit - stop_mult * atr) if atr_ok and limit - stop_mult * atr > 0 else np.nan,
        "target_if_fill_at_limit": floor_to_tick(limit + target_mult * atr) if atr_ok else np.nan,
        "contract": "小型" if mult == 100 else "標準",
        "multiplier": mult,
        "stop_dist_ntd_per_lot": stop_mult * atr * mult if atr_ok else np.nan,
        "target_dist_ntd_per_lot": target_mult * atr * mult if atr_ok else np.nan,
        "margin_est_1lot": margin,
        "margin_over_cap": bool(margin > margin_cap),
        "atr_invalid": not atr_ok,
        "entry_date": entry_date.date().isoformat(),
        "time_exit_date": time_exit_date(entry_date).date().isoformat(),
        "turnover_20d": turnover_20d,
    }
    # 回測會直接略過的：ATR算不出來(skipped_invalid_stop)、單筆保證金超過35%(skipped_single_margin_cap)
    row["backtest_would_skip"] = bool(row["margin_over_cap"] or row["atr_invalid"])
    return row


def scan(price_data: dict, universe: dict, as_of, capital: float = DEFAULT_CAPITAL,
         positions: int = DEFAULT_POSITIONS, futures_liquidity: dict = None,
         stock_amount: float = DEFAULT_STOCK_AMOUNT, stock_min_turnover: float = DEFAULT_STOCK_MIN_TURNOVER,
         fut_min_volume: float = DEFAULT_FUT_MIN_VOLUME, fut_min_oi: float = DEFAULT_FUT_MIN_OI) -> dict:
    """掃描全部標的，回傳結果dict(markdown/csv都從這裡產生)。as_of：訊號日(收盤那天)。
    positions：同時最多持有幾檔(1或2)，決定用哪一組實盤設定。
    futures_liquidity：taifex_futures_loader.load_futures_liquidity()的結果；None或ok=False → 期貨清單fail closed。
    訊號本身(候選、排名、限價、停損、停利)跟兩個清單無關，兩個清單只是從同一批訊號各自篩選。"""
    live_setting(positions)  # 不合法的positions在下載/計算之前就擋掉
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
        row = compute_signal_row(code, df, capital, positions=positions)
        if row is not None:
            signals.append(row)

    # 回測：Python穩定排序、reverse=True，同分維持universe順序
    signals.sort(key=lambda r: r["trigger_return"], reverse=True)
    for i, r in enumerate(signals, start=1):
        r["rank"] = i

    result = {
        "as_of": as_of,
        "capital": capital,
        "positions": positions,
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
    return build_tracks(result, futures_liquidity, stock_amount=stock_amount,
                        stock_min_turnover=stock_min_turnover, fut_min_volume=fut_min_volume, fut_min_oi=fut_min_oi)


# ----------------------------------------------------------------------------
# 兩個清單
# ----------------------------------------------------------------------------
def fmt_shares(shares: int) -> str:
    """1張=1000股：5120股 → 「5張120股」；800股 → 「800股(零股)」；3000股 → 「3張」。"""
    lots, odd = divmod(int(shares), 1000)
    if lots and odd:
        return f"{lots}張{odd}股"
    if lots:
        return f"{lots}張"
    return f"{odd}股(零股)" if odd else "0股"


def fmt_turnover(t: float) -> str:
    if t is None or not np.isfinite(t):
        return "無資料"
    return f"NT${t / 1e8:,.2f}億" if t >= 1e8 else f"NT${t / 1e4:,.0f}萬"


def stock_sizing(r: dict, amount: float) -> dict:
    """現股：固定金額 → 股數(無條件捨去，可零股)、成本估計(以限價買、以限價賣估手續費與證交稅)。"""
    price = r["limit_price"]
    shares = int(math.floor(amount / price + 1e-9)) if price and price > 0 else 0
    cost = shares * price
    fee_buy = cost * STOCK_COMMISSION_RATE
    fee_sell = cost * STOCK_COMMISSION_RATE
    tax = cost * STOCK_SELL_TAX_RATE
    atr_ok = not r["atr_invalid"]
    blocked_reason = None
    if shares < 1:
        blocked_reason = f"NT${amount:,.0f}買不到1股"
    elif not atr_ok:
        blocked_reason = "ATR異常"
    return {
        "stock_amount": amount,
        "stock_shares": shares,
        "stock_lots": shares // 1000,
        "stock_odd_shares": shares % 1000,
        "stock_cost": cost,
        "stock_fee_buy": fee_buy,
        "stock_fee_sell": fee_sell,
        "stock_tax": tax,
        "stock_roundtrip_cost": fee_buy + fee_sell + tax,
        "stock_stop_ntd": r["stop_dist"] * shares if atr_ok else np.nan,
        "stock_target_ntd": r["target_dist"] * shares if atr_ok else np.nan,
        "stock_blocked": blocked_reason is not None,
        "stock_block_reason": blocked_reason,
    }


def build_stock_track(signals: list, amount: float, min_turnover: float) -> dict:
    rows, excluded = [], []
    for r in signals:
        t = r.get("turnover_20d", np.nan)
        if t is None or not np.isfinite(t) or t < min_turnover:
            excluded.append({"code": r["code"], "name": r["name"], "signal_rank": r["rank"], "turnover_20d": t,
                             "reason": f"20日均成交值{fmt_turnover(t)}<{fmt_turnover(min_turnover)}"})
            continue
        rows.append({**r, **stock_sizing(r, amount), "signal_rank": r["rank"]})
    for i, r in enumerate(rows, start=1):  # 清單內自己排名(順序=原本的觸發K棒漲幅排名)
        r["rank"] = i
    return {"rows": rows, "excluded": excluded, "amount": amount, "min_turnover": min_turnover}


def build_futures_track(signals: list, liq: dict, min_volume: float, min_oi: float) -> dict:
    ok = bool(liq and liq.get("ok"))
    rows, excluded = [], []
    if ok:
        for r in signals:
            ev = tfl.evaluate_code(liq, r["code"], min_volume, min_oi)
            info = {"fut_code": ev["fut_code"], "fut_avg_volume": ev["avg_volume"], "fut_near_oi": ev["near_oi"],
                    "fut_near_month": ev["near_month"], "fut_spread_ticks": ev["spread_ticks"],
                    "fut_bid": ev["bid"], "fut_ask": ev["ask"], "fut_reason": ev["reason"]}
            if ev["passed"]:
                rows.append({**r, **info, "signal_rank": r["rank"]})
            else:
                excluded.append({"code": r["code"], "name": r["name"], "signal_rank": r["rank"], **info})
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
    error = None if ok else ((liq or {}).get("error") or "沒有期貨流動性資料")
    return {"ok": ok, "error": error, "rows": rows, "excluded": excluded,
            "min_volume": min_volume, "min_oi": min_oi,
            "latest_date": (liq or {}).get("latest_date"), "trading_dates": (liq or {}).get("trading_dates", [])}


def build_tracks(result: dict, futures_liquidity: dict = None, stock_amount: float = DEFAULT_STOCK_AMOUNT,
                 stock_min_turnover: float = DEFAULT_STOCK_MIN_TURNOVER,
                 fut_min_volume: float = DEFAULT_FUT_MIN_VOLUME, fut_min_oi: float = DEFAULT_FUT_MIN_OI) -> dict:
    """在result上加 futures_track / stock_track(回傳新的dict，不改原本的)。"""
    if not (stock_amount > 0):
        raise ValueError(f"stock_amount必須>0，收到{stock_amount!r}")
    sigs = result["signals"]
    return {**result,
            "futures_track": build_futures_track(sigs, futures_liquidity, fut_min_volume, fut_min_oi),
            "stock_track": build_stock_track(sigs, stock_amount, stock_min_turnover)}


def with_tracks(result: dict) -> dict:
    """舊格式的result(沒有兩個清單)→ 用預設值補上；沒有期貨資料 → 期貨清單fail closed。"""
    if "futures_track" in result and "stock_track" in result:
        return result
    return build_tracks(result, None)


# ----------------------------------------------------------------------------
# 輸出
# ----------------------------------------------------------------------------
CSV_COLUMNS = [
    "rank", "code", "name", "signal_date", "positions", "atr_stop_mult", "atr_target_mult", "entry_limit_ticks",
    "close", "prev_close", "trigger_return", "tick", "limit_price",
    "atr", "stop_dist", "target_dist", "stop_if_fill_at_limit", "target_if_fill_at_limit", "contract",
    "multiplier", "stop_dist_ntd_per_lot", "target_dist_ntd_per_lot", "margin_est_1lot", "margin_over_cap",
    "atr_invalid", "backtest_would_skip", "entry_date", "time_exit_date", "data_fresh",
    "fut_listed", "fut_code", "fut_avg_volume", "fut_near_oi", "fut_spread_ticks", "fut_reason",
    "turnover_20d", "stock_listed", "stock_shares", "stock_cost", "stock_roundtrip_cost",
]


def signals_frame(result: dict) -> pd.DataFrame:
    result = with_tracks(result)
    ft, stt = result["futures_track"], result["stock_track"]
    fut_rows = {r["code"]: r for r in ft["rows"]}
    fut_excl = {r["code"]: r for r in ft["excluded"]}
    stock_rows = {r["code"]: r for r in stt["rows"]}
    rows = []
    for r in result["signals"]:
        f = fut_rows.get(r["code"]) or fut_excl.get(r["code"]) or {}
        s = stock_rows.get(r["code"], {})
        rows.append({
            **r, "data_fresh": result["data_fresh"],
            "fut_listed": r["code"] in fut_rows,
            "fut_code": f.get("fut_code"), "fut_avg_volume": f.get("fut_avg_volume"),
            "fut_near_oi": f.get("fut_near_oi"), "fut_spread_ticks": f.get("fut_spread_ticks"),
            "fut_reason": f.get("fut_reason") if f else (None if ft["ok"] else ft["error"]),
            "stock_listed": r["code"] in stock_rows,
            "stock_shares": s.get("stock_shares"), "stock_cost": s.get("stock_cost"),
            "stock_roundtrip_cost": s.get("stock_roundtrip_cost"),
        })
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def _rules_section(positions: int) -> list:
    st = live_setting(positions)
    n = st["max_concurrent_positions"]
    cd = COOLDOWN_CALENDAR_DAYS
    pf = SQUEEZE_KDJ_LIVE_REFERENCE_PF_2018_2025
    t1, t2 = SQUEEZE_KDJ_LIVE_SETTINGS[1]["atr_target_mult"], SQUEEZE_KDJ_LIVE_SETTINGS[2]["atr_target_mult"]
    return [
        "## 每日操作規則",
        "",
        f"目前{setting_label(positions)}(另一組：--positions {2 if positions == 1 else 1})",
        "",
        "**掛單(開盤前)**",
        f"1. 期貨清單每檔只做1口(現股清單用該檔算好的股數)，價格用上面的「限價買進」(=訊號日收盤**+{ENTRY_LIMIT_TICKS}檔**，"
        "逐檔往上走，跨價位級距時檔位跟著變)，**開盤高於限價就不追**，等下一個訊號。",
        f"2. 空位上限={n}檔(你設定的N檔)，**每個清單各自計算**；空位有幾個就從該清單第1名往下取幾檔；跳過：標⛔的、你已經持有的、"
        f"停損出場後{cd}個日曆天內的(例：10/1停損，10/{1 + cd}起才能再進同一檔)。",
        f"3. (期貨)全部持倉保證金合計不超過資金的{total_margin_cap_ratio(positions):.0%}，單筆不超過35%。",
        f"4. 回測裡如果排名較前的那檔開盤高於限價沒成交，會改試下一名(最多試到第{TOP_N}名，不含已持有/冷卻中)。"
        "實盤只能近似：開盤後下一名的價格如果還≤它的限價，可以改掛它；已經漲過限價就放棄。",
        "",
        "**成交後(馬上做)**",
        f"5. 用**實際成交價**重算並立刻掛條件單：停損 = 成交價 − {st['atr_stop_mult']:g}×ATR，"
        f"停利 = 成交價 + {st['atr_target_mult']:g}×ATR，兩個都往下取到合法檔位。成交當天就生效(回測進場當天就會檢查停損/停利)。",
        f"6. 停損、停利都沒碰到：持有到「最長持有到期日」(進場日算第1天的第{MAX_HOLD_DAYS}個交易日)收盤前平倉。"
        "到期日是用週一~週五估的，遇到國定假日要往後順延。",
        "7. 回測細節：明天收盤就要到期平倉的部位，回測把它的名額算成明天可用。",
        "",
        "**兩個清單**",
        "8. 個股期貨清單、現股清單是**同一批訊號**(同樣的進場、排名、限價、停損、停利)的兩種做法，各自排名、各自算空位。"
        "**兩個清單同時做 = 同一個訊號的曝險加倍**(虧損也加倍)，請先決定只做哪一個。",
        "",
        "**你自己定的停損規則(不是回測規則，是你選的風控)**",
        "9. 帳戶從開始實盤累計虧損達 **NT$30,000** → 全部停止。",
        "10. 做滿 **30筆** 交易後，如果這30筆的獲利因子 **PF<1** → 全部停止。",
        "",
        f"⚠️ 這組設定是在2018–2026的資料上比過約{SQUEEZE_KDJ_LIVE_CUMULATIVE_VARIANTS}個回測變體之後才挑的，"
        "**從來沒有通過專案的正式驗證**，數字偏樂觀。"
        f"限價+{ENTRY_LIMIT_TICKS}檔時，2018–2025的回測PF約{pf[1]:.2f}(最多1檔、停利{t1:g}倍)／"
        f"約{pf[2]:.2f}(最多2檔、停利{t2:g}倍)；最近的成績有一大部分是2026年撐起來的。"
        "部位保持最小，以實盤結果為準。",
    ]


def _fmt_count(x, unit="口") -> str:
    return f"{x:,.0f}{unit}" if x is not None and np.isfinite(x) else "無資料"


def _signal_block(r: dict, data_fresh: bool) -> list:
    """期貨清單的一檔(沿用原本的格式，多一行期貨流動性)。"""
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
    lines.append(f"- 限價買進 {buy}(收盤{fmt_price(r['close'])}+{r['entry_limit_ticks']}檔)，開盤高於此價不追")
    if not r["atr_invalid"]:
        lines.append(f"- 若成交在 {fmt_price(r['limit_price'])}：停損 **{fmt_price(r['stop_if_fill_at_limit'])}**"
                     f"｜停利 **{fmt_price(r['target_if_fill_at_limit'])}**")
        lines.append(f"- ATR14 = {r['atr']:.2f} → 停損 = 成交價 − {r['stop_dist']:.2f}({r['atr_stop_mult']:g}×ATR)，"
                     f"停利 = 成交價 + {r['target_dist']:.2f}({r['atr_target_mult']:g}×ATR)")
        lines.append(f"- 每口：停損距離約 NT${r['stop_dist_ntd_per_lot']:,.0f}｜停利距離約 NT${r['target_dist_ntd_per_lot']:,.0f}")
    lines.append(f"- {r['contract']}契約 {r['multiplier']:,}股｜保證金約 NT${r['margin_est_1lot']:,.0f}")
    if r.get("fut_code"):
        lines.append(f"- 期貨流動性：{r['fut_code']} 5日均量 {_fmt_count(r['fut_avg_volume'])}｜"
                     f"近月({r.get('fut_near_month') or '-'})未平倉 {_fmt_count(r['fut_near_oi'])}｜"
                     f"最後最佳買賣價差 {_fmt_count(r.get('fut_spread_ticks'), '檔')}")
    lines.append(f"- 最長持有到 {fmt_date(r['time_exit_date'])} 收盤(約略，遇假日順延)")
    lines.append("")
    return lines


def _stock_block(r: dict, data_fresh: bool) -> list:
    name = f" {r['name']}" if r["name"] else ""
    head = f"### {r['rank']}. {r['code']}{name}" + (" ⛔" if r["stock_blocked"] else "")
    lines = [head, ""]
    if r["stock_blocked"]:
        lines.append(f"- ⛔ **{r['stock_block_reason']}，不要下單**")
    lines.append(f"- 觸發K棒漲幅 **{r['trigger_return']:+.2%}**｜收盤 {fmt_price(r['close'])}")
    if not data_fresh:
        buy = f"~~{fmt_price(r['limit_price'])}~~(資料不可信，不可下單)"
    elif r["stock_blocked"]:
        buy = f"{fmt_price(r['limit_price'])} ⛔"
    else:
        buy = f"**{fmt_price(r['limit_price'])}**"
    lines.append(f"- 限價買進 {buy}(收盤{fmt_price(r['close'])}+{r['entry_limit_ticks']}檔)，開盤高於此價不追")
    lines.append(f"- 股數 **{fmt_shares(r['stock_shares'])}**(NT${r['stock_amount']:,.0f} ÷ 限價，無條件捨去)"
                 f"｜金額約 NT${r['stock_cost']:,.0f}")
    if not r["atr_invalid"]:
        lines.append(f"- 若成交在 {fmt_price(r['limit_price'])}：停損 **{fmt_price(r['stop_if_fill_at_limit'])}**"
                     f"｜停利 **{fmt_price(r['target_if_fill_at_limit'])}**")
        lines.append(f"- ATR14 = {r['atr']:.2f} → 停損 = 成交價 − {r['stop_dist']:.2f}({r['atr_stop_mult']:g}×ATR)，"
                     f"停利 = 成交價 + {r['target_dist']:.2f}({r['atr_target_mult']:g}×ATR)")
        lines.append(f"- 這個股數：停損約 −NT${r['stock_stop_ntd']:,.0f}｜停利約 +NT${r['stock_target_ntd']:,.0f}")
    lines.append(f"- 來回成本約 NT${r['stock_roundtrip_cost']:,.0f}(手續費 {r['stock_fee_buy'] + r['stock_fee_sell']:,.0f}"
                 f" + 證交稅 {r['stock_tax']:,.0f}，以限價估)")
    lines.append(f"- 20日均成交值 {fmt_turnover(r['turnover_20d'])}")
    lines.append(f"- 最長持有到 {fmt_date(r['time_exit_date'])} 收盤(約略，遇假日順延)")
    lines.append("")
    return lines


def _futures_excluded_line(ft: dict) -> str:
    n = len(ft["excluded"])
    return (f"另有{n}檔訊號因期貨成交量不足未列出(5日均量<{ft['min_volume']:,.0f}口"
            f"或近月未平倉<{ft['min_oi']:,.0f}口)")


def _futures_section(result: dict) -> list:
    ft = result["futures_track"]
    sigs, fresh, as_of = result["signals"], result["data_fresh"], result["as_of"]
    lines = ["## 📈 個股期貨清單(主要)", "",
             f"條件：實際交易的契約(有小型就用小型)5個交易日平均成交量≥{ft['min_volume']:,.0f}口"
             f"(全部月份、一般+盤後時段加總，不含價差委託) 且 近月未平倉≥{ft['min_oi']:,.0f}口。資料：期交所每日行情。", ""]
    if not ft["ok"]:
        return lines + [f"> ## ⚠️ {FUTURES_FAIL_MESSAGE}",
                        f"> 原因：{ft['error']}",
                        f"> 今天共 {len(sigs)} 檔訊號；現股清單不受影響。診斷檔在 signals/debug/。", ""]
    if ft["latest_date"] and ft["latest_date"] != as_of.date().isoformat():
        lines += [f"- ⚠️ 期貨資料最新日期是 {ft['latest_date']}，不是訊號日 {as_of.date().isoformat()}"
                  "(期交所資料可能還沒更新)，流動性數字是舊的", ""]
    rows = ft["rows"]
    if fresh and sigs and not rows:
        lines += ["**期貨清單今天沒有可下單的標的**(訊號的期貨流動性都不夠)", ""]
    elif fresh and rows and all(r["backtest_would_skip"] for r in rows):
        lines += ["**✅ 期貨清單的訊號回測都會略過(⛔)，明天不用下期貨單**", ""]
    if rows and not fresh:
        lines += ["(以下僅供參考，不可據此下單)", ""]
    for r in rows:
        lines += _signal_block(r, data_fresh=fresh)
    if ft["excluded"]:
        detail = "；".join(f"{e['code']}{(' ' + e['name']) if e['name'] else ''}({e['fut_reason']})"
                           for e in ft["excluded"])
        lines += [_futures_excluded_line(ft) + "：" + detail, ""]
    return lines


def _stock_section(result: dict) -> list:
    stt = result["stock_track"]
    sigs, fresh = result["signals"], result["data_fresh"]
    lines = ["## 🧾 現股清單(參考)", "",
             f"條件：股票20日平均成交值(收盤×成交量)≥{fmt_turnover(stt['min_turnover'])}。"
             f"每筆固定 NT${stt['amount']:,.0f}(可零股)，股數 = 金額 ÷ 限價 無條件捨去。",
             f"成本估計：手續費{STOCK_COMMISSION_RATE:.4%}(買、賣各一次；牌告價，很多券商有折扣，未計每筆最低手續費)"
             f" + 證券交易稅{STOCK_SELL_TAX_RATE:.1%}(賣出)。", "",
             f"> ⚠️ {STOCK_BACKTEST_WARNING}",
             f"> {ODD_LOT_NOTE}", ""]
    rows = stt["rows"]
    if fresh and sigs and not rows:
        lines += ["**現股清單今天沒有可下單的標的**(訊號股票的成交值都不夠)", ""]
    if rows and not fresh:
        lines += ["(以下僅供參考，不可據此下單)", ""]
    for r in rows:
        lines += _stock_block(r, data_fresh=fresh)
    if stt["excluded"]:
        detail = "；".join(f"{e['code']}{(' ' + e['name']) if e['name'] else ''}({e['reason']})" for e in stt["excluded"])
        lines += [f"另有{len(stt['excluded'])}檔訊號因現股20日均成交值不足未列出：{detail}", ""]
    return lines


def render_markdown(result: dict) -> str:
    result = with_tracks(result)
    as_of = result["as_of"]
    sigs = result["signals"]
    fresh = result["data_fresh"]
    latest = result["latest_date"]
    positions = result.get("positions", DEFAULT_POSITIONS)
    lines = [f"# 擠壓+KDJ 每日訊號 {fmt_date(as_of)}", "", f"**{setting_label(positions)}**", ""]

    if not fresh:
        lines += [
            "> ## ⚠️ 資料還沒更新到今天，這份清單不可信，請晚點重跑",
            f"> 最新資料日期是 {fmt_date(latest) if latest is not None else '無資料'}"
            f"(參考：{result['latest_source']})，不是 {fmt_date(as_of)}。"
            "可能是資料源還沒更新，或今天休市。**下面不列可下單的訊號。**",
            "",
        ]

    ft, stt = result["futures_track"], result["stock_track"]
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
        fut_part = f"期貨清單 {len(ft['rows'])} 檔" if ft["ok"] else "期貨清單：資料失敗、不列"
        lines.append(f"- **今天訊號：{len(sigs)} 檔**({fut_part}｜現股清單 {len(stt['rows'])} 檔)")
    lines.append("")

    if fresh and not sigs:
        lines += ["## ✅ 今天沒有訊號，明天不用下單", ""]

    lines += ["---", ""] + _futures_section(result)
    lines += ["---", ""] + _stock_section(result)
    lines += ["---", ""] + _rules_section(positions) + [""]
    return "\n".join(lines)


TELEGRAM_MAX_CHARS = 3900  # Telegram單則訊息上限4096字，留一點空間給結尾連結


def _finalize_telegram(lines: list, link: str = None) -> str:
    footer_text = f"\n\n完整說明與操作規則：{link}" if link else ""
    text = "\n".join(lines)
    if len(text) + len(footer_text) > TELEGRAM_MAX_CHARS:
        cut = TELEGRAM_MAX_CHARS - len(footer_text) - 40
        text = text[:cut].rsplit("\n", 1)[0] + "\n…(訊號太多，其餘請看完整版)"
    return text + footer_text


def _stale_telegram_lines(result: dict) -> list:
    latest = result["latest_date"]
    return [
        "⚠️ 資料還沒更新到今天，這份清單不可信，不要下單。",
        f"最新資料日期：{fmt_date(latest) if latest is not None else '無資料'}。"
        "可能是資料源還沒更新或今天休市，請晚點到Actions手動重跑。",
    ]


def render_telegram_futures(result: dict, link: str = None) -> str:
    """Telegram第一則(主要)：個股期貨清單。純文字(不用parse_mode)。數字跟render_markdown()同一份result。"""
    result = with_tracks(result)
    as_of, sigs, fresh = result["as_of"], result["signals"], result["data_fresh"]
    ft = result["futures_track"]
    positions = result.get("positions", DEFAULT_POSITIONS)
    n_slots = live_setting(positions)["max_concurrent_positions"]
    lines = [f"【個股期貨清單】擠壓+KDJ 每日訊號 {fmt_date(as_of)}", setting_label(positions)]
    if not fresh:
        return _finalize_telegram(lines + _stale_telegram_lines(result), link)
    if not ft["ok"]:
        err = str(ft["error"])
        lines += [f"⚠️ {FUTURES_FAIL_MESSAGE}",
                  f"原因：{err[:300]}",
                  f"今天共{len(sigs)}檔訊號；現股清單見另一則訊息。診斷檔在signals/debug/。"]
        return _finalize_telegram(lines, link)
    rows = ft["rows"]
    n_actionable = sum(1 for r in rows if not r["backtest_would_skip"])
    if not sigs:
        lines.append(f"✅ 今天沒有訊號，{fmt_date(result['entry_date'])} 不用下單。")
    elif not rows:
        lines.append(f"✅ 今天{len(sigs)}檔訊號的期貨流動性都不夠，{fmt_date(result['entry_date'])} 不用下期貨單。")
    elif n_actionable == 0:
        lines.append(f"✅ 今天{len(rows)}檔訊號都是⛔(回測會略過)，{fmt_date(result['entry_date'])} 不用下單。")
    else:
        lines.append(f"📌 {len(rows)}檔訊號(可下單{n_actionable}檔)，{fmt_date(result['entry_date'])} 開盤前掛單，"
                     f"依名次取到你的空位數為止(上限{n_slots}檔，期貨清單自己算)；"
                     f"已持有/停損後{COOLDOWN_CALENDAR_DAYS}天內的跳過。")
    if ft["latest_date"] and ft["latest_date"] != as_of.date().isoformat():
        lines.append(f"⚠️ 期貨流動性資料最新是{ft['latest_date']}，不是今天")
    for r in rows:
        name = f" {r['name']}" if r["name"] else ""
        lines.append("")
        if r["backtest_would_skip"]:
            why = "保證金超過資金35%" if r["margin_over_cap"] else "ATR異常"
            lines.append(f"{r['rank']}. {r['code']}{name} ⛔ 不要下({why})")
            continue
        lines.append(f"{r['rank']}. {r['code']}{name}｜{r['contract']}契約1口｜保證金約{r['margin_est_1lot']:,.0f}")
        lines.append(f"限價買 {fmt_price(r['limit_price'])}(收盤+{r['entry_limit_ticks']}檔，開盤高於此價不追)")
        lines.append(f"若成交在{fmt_price(r['limit_price'])}：停損 {fmt_price(r['stop_if_fill_at_limit'])}"
                     f"｜停利 {fmt_price(r['target_if_fill_at_limit'])}")
        lines.append(f"實際成交價不同就重算：停損=成交價−{r['stop_dist']:.2f}({r['atr_stop_mult']:g}×ATR)，"
                     f"停利=成交價+{r['target_dist']:.2f}({r['atr_target_mult']:g}×ATR)")
        lines.append(f"期貨{r['fut_code']}：5日均量{_fmt_count(r['fut_avg_volume'])}｜近月未平倉{_fmt_count(r['fut_near_oi'])}"
                     f"｜買賣價差{_fmt_count(r.get('fut_spread_ticks'), '檔')}")
        lines.append(f"最晚 {fmt_date(r['time_exit_date'])} 收盤前平倉(遇假日順延)")
    if ft["excluded"]:
        lines += ["", _futures_excluded_line(ft)]
    return _finalize_telegram(lines, link)


def render_telegram_stock(result: dict, link: str = None) -> str:
    """Telegram第二則：現股清單(參考)。"""
    result = with_tracks(result)
    as_of, sigs, fresh = result["as_of"], result["signals"], result["data_fresh"]
    stt = result["stock_track"]
    positions = result.get("positions", DEFAULT_POSITIONS)
    n_slots = live_setting(positions)["max_concurrent_positions"]
    lines = [f"【現股清單】擠壓+KDJ 每日訊號 {fmt_date(as_of)}", setting_label(positions),
             f"⚠️ 參考用：回測含成本只約打平(PF≈1.0–1.1)。每筆固定NT${stt['amount']:,.0f}，可零股。"]
    if not fresh:
        return _finalize_telegram(lines + _stale_telegram_lines(result), link)
    rows = stt["rows"]
    n_actionable = sum(1 for r in rows if not r["stock_blocked"])
    if not sigs:
        lines.append(f"✅ 今天沒有訊號，{fmt_date(result['entry_date'])} 不用下單。")
    elif not rows:
        lines.append(f"✅ 今天{len(sigs)}檔訊號的現股成交值都不夠，{fmt_date(result['entry_date'])} 不用下單。")
    elif n_actionable == 0:
        lines.append(f"✅ 今天{len(rows)}檔訊號都是⛔，{fmt_date(result['entry_date'])} 不用下單。")
    else:
        lines.append(f"📌 {len(rows)}檔訊號(可下單{n_actionable}檔)，{fmt_date(result['entry_date'])} 開盤前掛單，"
                     f"依名次取到你的空位數為止(上限{n_slots}檔，現股清單自己算)；"
                     f"已持有/停損後{COOLDOWN_CALENDAR_DAYS}天內的跳過。")
    for r in rows:
        name = f" {r['name']}" if r["name"] else ""
        lines.append("")
        if r["stock_blocked"]:
            lines.append(f"{r['rank']}. {r['code']}{name} ⛔ 不要下({r['stock_block_reason']})")
            continue
        lines.append(f"{r['rank']}. {r['code']}{name}｜{fmt_shares(r['stock_shares'])}｜約NT${r['stock_cost']:,.0f}")
        lines.append(f"限價買 {fmt_price(r['limit_price'])}(收盤+{r['entry_limit_ticks']}檔，開盤高於此價不追)")
        lines.append(f"若成交在{fmt_price(r['limit_price'])}：停損 {fmt_price(r['stop_if_fill_at_limit'])}"
                     f"｜停利 {fmt_price(r['target_if_fill_at_limit'])}")
        lines.append(f"這個股數：停損約−{r['stock_stop_ntd']:,.0f}｜停利約+{r['stock_target_ntd']:,.0f}"
                     f"｜來回成本約{r['stock_roundtrip_cost']:,.0f}")
        lines.append(f"實際成交價不同就重算：停損=成交價−{r['stop_dist']:.2f}({r['atr_stop_mult']:g}×ATR)，"
                     f"停利=成交價+{r['target_dist']:.2f}({r['atr_target_mult']:g}×ATR)")
        lines.append(f"最晚 {fmt_date(r['time_exit_date'])} 收盤前平倉(遇假日順延)")
    if any(r["stock_odd_shares"] for r in rows if not r["stock_blocked"]):
        lines += ["", "註：零股在盤中零股市場交易，流動性比整張差，限價單可能掛不到。"]
    if stt["excluded"]:
        lines += ["", f"另有{len(stt['excluded'])}檔訊號因現股20日均成交值不足未列出"]
    return _finalize_telegram(lines, link)


# 舊名稱：telegram.txt = 期貨清單那一則(向下相容)
render_telegram = render_telegram_futures


def write_outputs(result: dict, output_dir: str) -> dict:
    result = with_tracks(result)
    os.makedirs(output_dir, exist_ok=True)
    md = render_markdown(result)
    day = result["as_of"].date().isoformat()
    paths = {
        "latest": os.path.join(output_dir, "latest.md"),
        "md": os.path.join(output_dir, f"{day}.md"),
        "csv": os.path.join(output_dir, f"{day}.csv"),
        "telegram_futures": os.path.join(output_dir, "telegram_futures.txt"),
        "telegram_stock": os.path.join(output_dir, "telegram_stock.txt"),
        "telegram": os.path.join(output_dir, "telegram.txt"),  # 向下相容 = 期貨清單那一則
    }
    for key in ("latest", "md"):
        with open(paths[key], "w", encoding="utf-8") as f:
            f.write(md)
    link = os.environ.get("SIGNALS_LINK")
    tg_futures = render_telegram_futures(result, link=link)
    tg_stock = render_telegram_stock(result, link=link)
    for key, text in (("telegram_futures", tg_futures), ("telegram", tg_futures), ("telegram_stock", tg_stock)):
        with open(paths[key], "w", encoding="utf-8") as f:
            f.write(text)
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


def _positive_float(s):
    v = float(s)
    if not (v > 0):
        raise argparse.ArgumentTypeError(f"必須>0，收到{s}")
    return v


def _nonneg_float(s):
    v = float(s)
    if not (v >= 0):
        raise argparse.ArgumentTypeError(f"必須>=0，收到{s}")
    return v


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="擠壓+KDJ(實盤設定)每日訊號掃描")
    p.add_argument("--positions", type=int, choices=VALID_POSITIONS, default=DEFAULT_POSITIONS,
                   help="同時最多持有幾檔：1=停利4.0倍ATR(預設)、2=停利3.0倍ATR；停損都是1.5倍ATR、限價收盤+2檔")
    p.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, help="帳戶資金(新台幣)，用來判斷單筆保證金35%%上限")
    p.add_argument("--as-of", default=None, help="訊號日YYYY-MM-DD，預設今天(台北時間)")
    p.add_argument("--max-stocks", type=int, default=0, help="只掃前N檔(0=全部)，測試用")
    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                   help="往前抓幾個日曆天的資料當指標暖身(預設400，見程式碼註解)")
    p.add_argument("--output-dir", default="signals", help="輸出資料夾")
    p.add_argument("--stock-amount", type=_positive_float, default=DEFAULT_STOCK_AMOUNT,
                   help="現股清單：每筆固定金額(新台幣，預設100000，可零股)")
    p.add_argument("--stock-min-turnover", type=_nonneg_float, default=DEFAULT_STOCK_MIN_TURNOVER,
                   help="現股清單：股票20日平均成交值下限(新台幣，預設50000000)")
    p.add_argument("--fut-min-volume", type=_nonneg_float, default=DEFAULT_FUT_MIN_VOLUME,
                   help="期貨清單：5日平均成交量下限(口，全部月份+一般/盤後加總，預設100)")
    p.add_argument("--fut-min-oi", type=_nonneg_float, default=DEFAULT_FUT_MIN_OI,
                   help="期貨清單：近月未平倉下限(口，預設300)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    as_of = datetime.date.fromisoformat(args.as_of) if args.as_of else today_in_taipei()
    universe = build_universe(args.max_stocks)
    start, end = download_window(as_of, args.lookback_days)
    print(setting_label(args.positions), flush=True)
    print(f"掃描 {len(universe)} 檔，資料區間 {start} ~ {as_of.isoformat()}(yfinance end={end}，不含)", flush=True)

    # refresh=True + 暫存快取資料夾：保證一定重新下載(不會用到同一天稍早、資料還沒出來時存下的快取)，
    # 也不會在data_cache/每天留下一批之後用不到的CSV。期交所的原始回應也放同一個暫存資料夾。
    tmp_cache = tempfile.mkdtemp(prefix="daily_squeeze_cache_")
    try:
        price_data = load_price_data(universe, start, end, refresh=True, cache_dir=tmp_cache)
        try:
            liq = load_futures_liquidity(as_of, list(universe), cache_dir=tmp_cache, refresh=True,
                                         debug_dir=os.path.join(args.output_dir, "debug"))
        except Exception as e:  # loader本身不丟例外；這裡是最後一道保險：失敗就關閉期貨清單
            liq = tfl.failed_liquidity(f"未預期錯誤 {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp_cache, ignore_errors=True)

    result = scan(price_data, universe, as_of, capital=args.capital, positions=args.positions,
                  futures_liquidity=liq, stock_amount=args.stock_amount,
                  stock_min_turnover=args.stock_min_turnover,
                  fut_min_volume=args.fut_min_volume, fut_min_oi=args.fut_min_oi)
    out = write_outputs(result, args.output_dir)
    print(out["markdown"], flush=True)
    print(f"已寫入：{out['latest']}、{out['md']}、{out['csv']}、{out['telegram_futures']}、{out['telegram_stock']}",
          flush=True)
    return {"result": result, **out}


if __name__ == "__main__":
    main()
    sys.exit(0)
