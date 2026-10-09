"""
tx_daily_signal.py —— 台指期(小台MTX)日線策略的「每日實盤訊號」(一個凍結的變體，紙上交易用)。

用途：每個交易日收盤後(跟daily_squeeze_signals.py同一個GitHub Actions排程)算一次，告訴你明天
小台要做什麼：空手沒事 / 明天開盤買進1口 / 繼續持有(停損掛在哪) / 明天開盤平倉。

策略 = TX_LIVE_SETTINGS(下面那個常數，**唯一**定義的地方)：
  MA_5_20&DONCHIAN20｜只做多｜停損1.0/移動3.0
  - 進場規則：MA5上穿MA20 跟 收盤突破前20天最高價 兩個訊號在3天視窗內同方向出現
    (= tx_daily_engine.compose_pair，跟--grid/--refine完全相同)；只做多(空訊號不進場)。
  - 第t天收盤後出訊號 → 第t+1天開盤市價進場；初始停損 = 進場價 − 1.0×ATR14(ATR取訊號日t的值)。
  - 移動停損：每天收盤後 = max(目前停損, 進場後最高收盤 − 3.0×ATR14(當天))，只會往上收緊；無停利。
  - 盤中觸及停損 → 停損價成交；開盤就跳空跌破 → 開盤價成交(各扣1點滑價)。
  - 持倉中出現同一條規則的「原始」空訊號 → 隔天開盤平倉；持有滿60個交易日(進場日算第1天) → 隔天開盤平倉。
  - 無動能門檻。

資料：^TWII加權指數日線(yfinance；tx_daily_data.load_index_ohlc，從2023-01-01抓，暖身用)，
  跟回測--refine的「近期(2024~)指數代理」那一段同一種資料。訊號/價位都是「指數點數」。

狀態：**沒有**會被改寫的狀態檔。每次執行都用回測引擎(tx_daily_engine.run_tx_daily_grid_backtest，
  跟--refine同一支、同一組參數)從REPLAY_START(2024-01-01)重播到as_of；引擎的選配參數state_out
  讓最後一天不強制平倉，直接拿到「現在的部位、下一個交易日要用的停損、明天要不要進/出場」。
  所以這裡的停損/進出場跟回測逐筆相同(test_tx_daily_signal.py有驗證)。
  ⚠️ 重播是確定性的，但如果Yahoo事後修正了歷史K棒，過去的訊號可能跟當天報的不同；
  每天實際報出的內容另外記在signals/tx_paper_log.csv(每天一列，同一天重跑會覆蓋那一列)。

資料沒更新(fail closed，跟daily_squeeze_signals.py同一個慣例)：下載失敗、或最新一根K棒不是as_of那天
  → 照樣寫檔，但最上面大大的「⚠️ 資料還沒更新」，而且**不給任何新進場指示**(持倉的停損只列出最新
  資料日的值當參考)。

輸出(--output-dir，預設signals/)：tx_latest.md、telegram_tx.txt、tx_history/YYYY-MM-DD.md、tx_paper_log.csv。

使用：
  python tx_daily_signal.py --output-dir signals
  python tx_daily_signal.py --output-dir signals --as-of 2026-10-12
"""
import argparse
import datetime
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import tx_daily_data as tdd
import tx_daily_engine as eng
import tx_daily_refine as rf
from daily_squeeze_signals import fmt_date, next_business_day, today_in_taipei

# ---------------------------------------------------------------------------
# 凍結的實盤設定 —— 2026-10 由 compare_tx_daily.py --refine 的walk-forward選出(2026年被選中的變體)。
# 規則：只在每年1月重跑一次 --refine，看walk-forward當年選哪個變體再決定要不要換；其他時候不要改
# (看了最近的結果就改 = 又一次事後挑選)。name必須跟tx_daily_refine.refine_variant_list()裡的名稱一字不差。
# ---------------------------------------------------------------------------
TX_LIVE_SETTINGS = {
    "name": "MA_5_20&DONCHIAN20｜只做多｜停損1.0/移動3.0",
    "rule": "MA_5_20&DONCHIAN20",
    "components": ("MA_5_20", "DONCHIAN20"),
    "pair_window_days": eng.PAIR_WINDOW_DAYS,   # 3天視窗
    "direction": "long",                         # 只做多
    "stop_atr": 1.0,                             # 初始停損1.0×ATR14
    "exit_kind": "trail",                        # 移動停損、無停利
    "exit_mult": 3.0,                            # 最高收盤 − 3.0×ATR14
    "atr_period": eng.ATR_PERIOD,                # 14
    "max_hold_days": eng.MAX_HOLD_DAYS,          # 60
    "gate": False,                               # 無動能門檻
    "contract": "mini",                          # 小台MTX，每點NT$50
    "lots": 1,
    "frozen": "2026-10",
}

REPLAY_START = "2024-01-01"     # 重播起點(= 回測近期段的起點)
PAPER_START = "2026-10-12"      # 紙上交易起點：進場日>=這天的交易才算「實際紀錄」
DOWNLOAD_START = "2023-01-01"   # 多抓一年暖身(MA/唐奇安/ATR都是有限視窗，2024起的訊號跟用完整歷史算的相同)
CAPITAL_NOTE = "建議每口準備至少NT$400,000；回測walk-forward逐日市值最大回撤約NT$18萬。"
PRE_PAPER_NOTE = ("這筆部位是紙上交易開始({start})之前就進場的重播部位(回測參考)；"
                  "如果你沒有實際持有，不要追進，等下一個進場訊號。")
FUTURES_PRICE_NOTE = ("小台實際價格 = 指數 + 基差(近月合約與指數的價差，除息季7~8月常為負數百點)；"
                      "停損單請用「停損距離(點)」換算到期貨價格。")
TELEGRAM_MAX_CHARS = 1500

STATUS_LABELS = {
    "flat": "空手、今天沒訊號",
    "entry": "空手、今天出現進場訊號 → 明天開盤買進小台1口",
    "hold": "持有多單",
    "exit": "持有多單、今天出現出場條件 → 明天開盤平倉",
    "stale": "⚠️ 資料還沒更新",
}
LOG_STATUS = {"flat": "空手無訊號", "entry": "進場訊號", "hold": "持有多單", "exit": "出場訊號", "stale": "資料未更新"}
PAPER_LOG_COLUMNS = ["date", "status", "position", "stop", "close"]


def live_variant() -> dict:
    """--refine變體清單裡跟TX_LIVE_SETTINGS同名的那一個；欄位不一致直接ValueError(不默默用別的)。"""
    v = next((x for x in rf.refine_variant_list() if x["name"] == TX_LIVE_SETTINGS["name"]), None)
    if v is None:
        raise ValueError(f"--refine變體清單裡找不到 {TX_LIVE_SETTINGS['name']}")
    for k in ("rule", "components", "direction", "stop_atr", "exit_kind", "exit_mult"):
        if v[k] != TX_LIVE_SETTINGS[k]:
            raise ValueError(f"TX_LIVE_SETTINGS[{k!r}]={TX_LIVE_SETTINGS[k]!r} 跟--refine的{v[k]!r}不一致")
    return v


def settings_label() -> str:
    s = TX_LIVE_SETTINGS
    return (f"MA5上穿MA20＋突破20日高(3天內)｜只做多｜停損{s['stop_atr']:g}×ATR14｜"
            f"移動停損{s['exit_mult']:g}×ATR14｜最長{s['max_hold_days']}天")


# ---------------------------------------------------------------------------
# 重播(純函式，不碰網路/檔案)
# ---------------------------------------------------------------------------
def engine_frame(index_ohlc: pd.DataFrame) -> pd.DataFrame:
    """^TWII OHLC → 引擎格式(Offset=0、RollDay=曆法換月日)，跟回測的指數代理段同一個函式。"""
    return tdd._index_frame(index_ohlc)


def replay(frame: pd.DataFrame, start=REPLAY_START, end=None) -> dict:
    """用回測引擎重播凍結變體到end(含)。frame = engine_frame()的結果(只用<=end的列)。
    回傳{"trades": 已平倉交易, "state": 引擎的期末狀態, "arrs": ...}。"""
    v = live_variant()
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end)]
    if frame.empty:
        return {"trades": [], "state": {"position": None, "pending_entry": 0, "pending_exit": None,
                                        "last_index": None}, "arrs": None}
    arrs = eng.prepare_grid_arrays(frame, eng.compute_grid_indicators(frame))
    es, rs = rf.refine_entry_signals(arrs, v)
    state = {}
    trades = eng.run_tx_daily_grid_backtest(
        arrs, es, rs, v["stop_atr"], v["exit_kind"], v["exit_mult"], start=start, end=None,
        contract=TX_LIVE_SETTINGS["contract"], max_hold_days=TX_LIVE_SETTINGS["max_hold_days"],
        variant_name=v["name"], state_out=state)
    if not state:  # 區間內沒有任何K棒
        state = {"position": None, "pending_entry": 0, "pending_exit": None, "last_index": None}
    return {"trades": trades, "state": state, "arrs": arrs}


def _pf(pnl):
    return rf.profit_factor(pnl) if len(pnl) else float("nan")


def evaluate(index_ohlc: pd.DataFrame, as_of, index_info: dict = None) -> dict:
    """主要邏輯：回傳result dict(markdown/telegram/紙上紀錄都從這裡產生)。index_ohlc可以含as_of之後的列
    (不會被用到)。"""
    as_of = pd.Timestamp(as_of).normalize()
    mult = eng.CONTRACT_MULTIPLIER[TX_LIVE_SETTINGS["contract"]]
    res = {"as_of": as_of, "index_info": index_info or {}, "latest_date": None, "data_fresh": False,
           "status": "stale", "position": None, "entry_plan": None, "exit_reason": None,
           "exits_today": [], "paper_trades": [], "paper_open": None, "paper_realized_ntd": 0.0,
           "ref": None, "close": None, "atr": None, "next_day": next_business_day(as_of),
           "multiplier": mult, "stale_reason": None}
    df = pd.DataFrame() if index_ohlc is None else index_ohlc
    if not df.empty:
        df = df[pd.DatetimeIndex(df.index) <= as_of]
    if df.empty:
        reason = (index_info or {}).get("reason") or "沒有資料"
        res["stale_reason"] = f"加權指數下載失敗或沒有資料({reason})"
        return res

    frame = engine_frame(df)
    latest = pd.Timestamp(frame.index[-1]).normalize()
    res["latest_date"] = latest
    res["data_fresh"] = bool(latest == as_of)   # 跟daily_squeeze_signals.scan()同一個慣例
    if not res["data_fresh"]:
        res["stale_reason"] = (f"最新資料日期是 {fmt_date(latest)}，不是 {fmt_date(as_of)}"
                               "(資料源還沒更新，或今天休市)")

    rp = replay(frame)
    st, trades, arrs = rp["state"], rp["trades"], rp["arrs"]
    hi = len(frame) - 1
    c = arrs["c"]
    atr_today = arrs["atr"][hi]
    res["close"] = float(c[hi])
    res["atr"] = float(atr_today) if np.isfinite(atr_today) else float("nan")
    res["exits_today"] = [t for t in trades if pd.Timestamp(t["exit_date"]).normalize() == latest]

    pos = st["position"]
    if pos is not None:
        prev_stop = pos["initial_stop"]
        if pos["entry_idx"] < hi:
            prev = replay(frame.iloc[:hi])["state"]["position"]
            if prev is not None and prev["entry_date"] == pos["entry_date"]:
                prev_stop = prev["stop"]
        days = hi - pos["entry_idx"] + 1
        res["position"] = {
            "entry_date": pd.Timestamp(pos["entry_date"]), "signal_date": pd.Timestamp(pos["signal_date"]),
            "entry_price": float(pos["entry_price"]), "initial_stop": float(pos["initial_stop"]),
            "stop": float(pos["stop"]), "prev_stop": float(prev_stop),
            "stop_moved_up": bool(pos["stop"] > prev_stop + 1e-9),
            "best_close": float(pos["best_close"]), "days_held": int(days),
            "unrealized_points": float(c[hi] - pos["entry_price"]),
            "unrealized_ntd": float((c[hi] - pos["entry_price"]) * mult),
            "stop_distance_from_close": float(c[hi] - pos["stop"]),
            "trail_level": float(pos["best_close"] - TX_LIVE_SETTINGS["exit_mult"] * atr_today)
            if np.isfinite(atr_today) else float("nan"),
        }

    if res["data_fresh"]:
        if pos is not None:
            res["status"] = "exit" if st["pending_exit"] else "hold"
            res["exit_reason"] = st["pending_exit"]
        elif st["pending_entry"] == 1 and np.isfinite(atr_today) and atr_today > 0:  # 引擎ATR無效時也不進場
            res["status"] = "entry"
            dist = TX_LIVE_SETTINGS["stop_atr"] * atr_today
            res["entry_plan"] = {"stop_distance": float(dist), "stop_distance_ntd": float(dist * mult),
                                 "approx_stop": float(c[hi] - dist), "atr": float(atr_today),
                                 "trail_distance": float(TX_LIVE_SETTINGS["exit_mult"] * atr_today)}
        else:
            res["status"] = "flat"

    # 紙上交易(進場日>=PAPER_START) + 回測參考(2024~全部已平倉)
    paper_start = pd.Timestamp(PAPER_START)
    res["paper_trades"] = [t for t in trades if pd.Timestamp(t["entry_date"]) >= paper_start]
    res["paper_realized_ntd"] = float(sum(t["pnl_ntd"] for t in res["paper_trades"]))
    if res["position"] is not None and res["position"]["entry_date"] >= paper_start:
        res["paper_open"] = res["position"]
    pnl = [t["pnl_ntd"] for t in trades]
    res["ref"] = {"n": len(trades), "pf": _pf(pnl), "pnl": float(sum(pnl)),
                  "win_rate": (sum(1 for x in pnl if x > 0) / len(pnl) * 100) if pnl else float("nan"),
                  "start": REPLAY_START}
    return res


# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------
def fp(x, nd=1) -> str:
    """指數點位/點數：千分位、預設1位小數。"""
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:,.{nd}f}"


def fntd(x) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{'+' if x > 0 else ''}{x:,.0f}"


def _pf_text(pf) -> str:
    if pf is None or (isinstance(pf, float) and np.isnan(pf)):
        return "-"
    return "∞" if np.isinf(pf) else f"{pf:.2f}"


def status_line(res: dict) -> str:
    return STATUS_LABELS[res["status"]]


def _position_lines(p: dict, res: dict, md=True) -> list:
    b = "**" if md else ""
    mh = TX_LIVE_SETTINGS["max_hold_days"]
    moved = (f"今天上移 {fp(p['stop'] - p['prev_stop'])} 點(原{fp(p['prev_stop'])})" if p["stop_moved_up"]
             else "今天沒有變動")
    return [
        f"進場：{fmt_date(p['entry_date'])} 開盤，回測成交價 {fp(p['entry_price'])}(指數開盤+1點滑價)",
        f"停損單掛在這個價位：{b}{fp(p['stop'])}{b}(指數點；{moved})",
        f"停損距離：距今天收盤 {fp(p['stop_distance_from_close'])} 點 → 小台停損 ≈ 小台收盤 − {fp(p['stop_distance_from_close'])}",
        f"初始停損 {fp(p['initial_stop'])}｜進場後最高收盤 {fp(p['best_close'])}｜"
        f"最高收盤−{TX_LIVE_SETTINGS['exit_mult']:g}×ATR = {fp(p['trail_level'])}",
        f"持有第 {p['days_held']} 天 / {mh}(第{mh}天收盤後 → 隔天開盤平倉)",
        f"未實現損益：{fp(p['unrealized_points'], 1)} 點 ≈ NT${fntd(p['unrealized_ntd'])}(收盤價計，未扣手續費/稅)",
    ]


def _exit_today_lines(res: dict) -> list:
    out = []
    for t in res["exits_today"]:
        out.append(f"最近一次出場(今天)：{t['exit_reason']}，出場價 {fp(t['exit_price'])}"
                   f"(進場 {fmt_date(t['entry_date'])} @ {fp(t['entry_price'])})，"
                   f"{fp(t['points'])} 點，含成本 NT${fntd(t['pnl_ntd'])}")
    return out


def _paper_section(res: dict) -> list:
    lines = [f"## 紙上交易紀錄(進場日 ≥ {PAPER_START})", ""]
    trades = res["paper_trades"]
    op = res["paper_open"]
    if not trades and op is None:
        lines += [f"紙上交易從 {PAPER_START} 開始，目前還沒有交易。", ""]
    else:
        lines += ["| 進場日 | 進場價 | 出場日 | 出場價 | 原因 | 點數 | 損益(NT$) |",
                  "|---|---:|---|---:|---|---:|---:|"]
        for t in trades:
            lines.append(f"| {pd.Timestamp(t['entry_date']).date()} | {fp(t['entry_price'])} | "
                         f"{pd.Timestamp(t['exit_date']).date()} | {fp(t['exit_price'])} | {t['exit_reason']} | "
                         f"{fp(t['points'])} | {fntd(t['pnl_ntd'])} |")
        if op is not None:
            lines.append(f"| {op['entry_date'].date()} | {fp(op['entry_price'])} | 持有中 | - | - | "
                         f"{fp(op['unrealized_points'])} | {fntd(op['unrealized_ntd'])}(未實現) |")
        lines += ["", f"已平倉 {len(trades)} 筆，累計損益 **NT${fntd(res['paper_realized_ntd'])}**"
                  "(含手續費每邊NT$50、期交稅、1點滑價，跟回測同一套成本)", ""]
    ref = res["ref"]
    if ref is not None:
        lines += [f"回測參考，不是實際紀錄：{ref['start']} 起重播 {ref['n']} 筆已平倉，PF {_pf_text(ref['pf'])}，"
                  f"勝率 {fp(ref['win_rate'])}%，損益 NT${fntd(ref['pnl'])}(加權指數代理，含成本)", ""]
    return lines


def _rules_section() -> list:
    s = TX_LIVE_SETTINGS
    return [
        "## 規則重點(凍結，不要臨時改)",
        "",
        f"- 變體：{s['name']}(2026-10由 --refine 的walk-forward選出；每年1月重跑 --refine 才重新挑)",
        "- 進場：MA5上穿MA20 與 收盤突破前20天最高價，兩者在3個交易日內都出現 → 隔天開盤市價買進小台1口(只做多)",
        f"- 初始停損 = 成交價 − {s['stop_atr']:g}×ATR14(ATR取訊號日)；之後每天收盤後"
        f"停損 = max(目前停損, 進場後最高收盤 − {s['exit_mult']:g}×ATR14)，只上移不下移；無停利",
        "- 盤中觸及停損 → 停損出場；開盤就跳空跌破 → 開盤就出場",
        f"- 出現反向(空)訊號，或持有滿{s['max_hold_days']}個交易日 → 隔天開盤市價平倉；平倉後等新訊號才再進場",
        "- 訊號用加權指數(^TWII)日線計算，價位都是指數點數；成本假設：手續費每邊NT$50＋期交稅＋1點滑價",
        f"- {FUTURES_PRICE_NOTE}",
        f"- 資金：{CAPITAL_NOTE}",
        "- ⚠️ 這個變體是從312個候選裡依歷史挑出來的，回測數字偏樂觀；紙上交易的結果才是真正的檢驗。",
        "",
    ]


def render_markdown(res: dict) -> str:
    as_of = res["as_of"]
    lines = [f"# 台指期(小台)每日訊號 {fmt_date(as_of)}", "", f"**策略：{settings_label()}**", ""]
    latest = res["latest_date"]
    if res["status"] == "stale":
        lines += ["> ## ⚠️ 資料還沒更新，今天不給新的進場指示",
                  f"> {res['stale_reason']}。**不要依這份報告開新倉**，請晚點到Actions手動重跑。", ""]
    lines += [f"- 訊號日 {as_of.date().isoformat()}｜資料最新 {latest.date().isoformat() if latest is not None else '無'}"
              f" {'✅' if res['data_fresh'] else '❌'}",
              f"- 加權指數收盤 {fp(res['close'])}｜ATR14 {fp(res['atr'])}", ""]
    lines += [f"## 今天狀態：{status_line(res)}", ""]
    st = res["status"]
    p = res["position"]
    if st == "flat":
        lines += [f"✅ 明天({fmt_date(res['next_day'])})不用下單。", ""]
    elif st == "entry":
        e = res["entry_plan"]
        lines += [f"📌 **{fmt_date(res['next_day'])}(下一交易日，遇假日順延)開盤市價買進小台1口**", "",
                  f"- 停損距離 **{fp(e['stop_distance'])} 點**({TX_LIVE_SETTINGS['stop_atr']:g}×ATR14 = "
                  f"{TX_LIVE_SETTINGS['stop_atr']:g}×{fp(e['atr'])})，每口約 NT${e['stop_distance_ntd']:,.0f}",
                  f"- 以今天收盤估：停損約 **{fp(e['approx_stop'])}**(指數點)；實際停損 = 成交價 − {fp(e['stop_distance'])}"
                  "(用小台實際成交價減)",
                  f"- 移動停損：之後每天收盤後，停損 = max(目前停損, 進場後最高收盤 − {TX_LIVE_SETTINGS['exit_mult']:g}×ATR14)"
                  f"(目前3×ATR約 {fp(e['trail_distance'])} 點)，只上移；每天看這份報告更新停損單", ""]
    elif st in ("hold", "exit"):
        if st == "exit":
            lines += [f"🔔 **出場條件：{res['exit_reason']} → {fmt_date(res['next_day'])}(下一交易日)開盤市價平倉**", "",
                      "(平倉前停損單繼續掛著)", ""]
        lines += [f"- {x}" for x in _position_lines(p, res)] + [""]
        if p["entry_date"] < pd.Timestamp(PAPER_START):
            lines += [f"> ⚠️ {PRE_PAPER_NOTE.format(start=PAPER_START)}", ""]
    elif st == "stale" and p is not None:
        lines += [f"參考(截至 {latest.date().isoformat()} 的資料，僅供參考)：持有多單，停損 {fp(p['stop'])}"
                  f"(進場 {p['entry_date'].date()} @ {fp(p['entry_price'])})。停損只會上移，原本掛的停損單繼續有效。", ""]
    ex = _exit_today_lines(res)
    if ex:
        lines += [f"- {x}" for x in ex] + [""]
    lines += [f"> {FUTURES_PRICE_NOTE}", ""]
    lines += ["---", ""] + _paper_section(res)
    lines += ["---", ""] + _rules_section()
    return "\n".join(lines)


def render_telegram(res: dict, link: str = None) -> str:
    """純文字、<1500字。第一行【台指期訊號】日期，第二行狀態。"""
    as_of, st, p = res["as_of"], res["status"], res["position"]
    lines = [f"【台指期訊號】{fmt_date(as_of)}", f"狀態：{status_line(res)}"]
    if st == "stale":
        lines.append(f"{res['stale_reason']}。不要依此開新倉，請晚點重跑。")
        if p is not None:
            lines.append(f"參考(截至{res['latest_date'].date()})：持有多單，停損{fp(p['stop'])}，原停損單繼續有效。")
    elif st == "flat":
        lines.append(f"明天({fmt_date(res['next_day'])})不用下單。收盤{fp(res['close'])}")
    elif st == "entry":
        e = res["entry_plan"]
        lines += [f"{fmt_date(res['next_day'])}開盤市價買進小台1口",
                  f"停損距離{fp(e['stop_distance'])}點(1×ATR，每口約NT${e['stop_distance_ntd']:,.0f})",
                  f"以收盤{fp(res['close'])}估停損約{fp(e['approx_stop'])}；實際=成交價−{fp(e['stop_distance'])}",
                  f"之後每天收盤後停損=max(停損,最高收盤−3×ATR)，只上移"]
    else:
        if st == "exit":
            lines.append(f"出場條件：{res['exit_reason']} → {fmt_date(res['next_day'])}開盤市價平倉")
        moved = f"今天上移{fp(p['stop'] - p['prev_stop'])}點" if p["stop_moved_up"] else "今天沒變"
        lines += [f"進場{p['entry_date'].date()} @ {fp(p['entry_price'])}",
                  f"停損單：{fp(p['stop'])}({moved})；距收盤{fp(p['stop_distance_from_close'])}點",
                  f"最高收盤{fp(p['best_close'])}｜持有{p['days_held']}/{TX_LIVE_SETTINGS['max_hold_days']}天",
                  f"未實現{fp(p['unrealized_points'])}點≈NT${fntd(p['unrealized_ntd'])}"]
        if p["entry_date"] < pd.Timestamp(PAPER_START):
            lines.append(f"⚠️ 紙上交易開始({PAPER_START})前進場的重播部位；沒有實際持有就不要追，等下一個訊號。")
    for x in _exit_today_lines(res):
        lines.append(x)
    lines.append(f"紙上累計(≥{PAPER_START})：{len(res['paper_trades'])}筆 NT${fntd(res['paper_realized_ntd'])}")
    lines.append("價位為加權指數點；小台=指數+基差，停損請用停損距離換算。")
    text = "\n".join(lines)
    if link:
        footer = f"\n完整說明：{link}"
        if len(text) + len(footer) < TELEGRAM_MAX_CHARS:
            text += footer
    if len(text) >= TELEGRAM_MAX_CHARS:
        text = text[:TELEGRAM_MAX_CHARS - 20].rsplit("\n", 1)[0] + "\n…(其餘看完整版)"
    return text


def paper_log_row(res: dict) -> dict:
    p = res["position"]
    holding = p is not None and res["status"] in ("hold", "exit")
    if res["status"] == "stale":
        pos_txt = (f"資料未更新(截至{res['latest_date'].date()}持有多1口)" if p is not None else "資料未更新")
    else:
        pos_txt = f"多1口@{p['entry_price']:.1f}" if holding else "空手"
    return {"date": res["as_of"].date().isoformat(), "status": LOG_STATUS[res["status"]], "position": pos_txt,
            "stop": round(p["stop"], 1) if holding else "",
            "close": round(res["close"], 1) if res["data_fresh"] and res["close"] is not None else ""}


def update_paper_log(path: str, row: dict) -> pd.DataFrame:
    """每次執行一列；同一天重跑 → 取代那一列(冪等)。依日期排序。"""
    if os.path.exists(path):
        try:
            old = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            old = pd.DataFrame(columns=PAPER_LOG_COLUMNS)
    else:
        old = pd.DataFrame(columns=PAPER_LOG_COLUMNS)
    old = old.reindex(columns=PAPER_LOG_COLUMNS)
    old = old[old["date"] != row["date"]]
    new = pd.concat([old, pd.DataFrame([{k: str(row[k]) for k in PAPER_LOG_COLUMNS}])], ignore_index=True)
    new = new.sort_values("date", kind="stable").reset_index(drop=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")
    return new


def write_outputs(res: dict, output_dir: str) -> dict:
    os.makedirs(os.path.join(output_dir, "tx_history"), exist_ok=True)
    md = render_markdown(res)
    day = res["as_of"].date().isoformat()
    paths = {"latest": os.path.join(output_dir, "tx_latest.md"),
             "history": os.path.join(output_dir, "tx_history", f"{day}.md"),
             "telegram": os.path.join(output_dir, "telegram_tx.txt"),
             "paper_log": os.path.join(output_dir, "tx_paper_log.csv")}
    for k in ("latest", "history"):
        with open(paths[k], "w", encoding="utf-8") as f:
            f.write(md)
    tg = render_telegram(res, link=os.environ.get("TX_SIGNALS_LINK"))
    with open(paths["telegram"], "w", encoding="utf-8") as f:
        f.write(tg)
    update_paper_log(paths["paper_log"], paper_log_row(res))
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    return {"markdown": md, "telegram_text": tg, **paths}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def download_index(as_of: datetime.date):
    """^TWII日線(DOWNLOAD_START ~ as_of，yfinance end不含 → as_of+1)。強制重新下載(暫存快取資料夾，
    用完就刪)，避免讀到同一天稍早資料還沒出來時存的快取。失敗回傳(空df, info)，不丟例外。"""
    end = (as_of + datetime.timedelta(days=1)).isoformat()
    tmp = tempfile.mkdtemp(prefix="tx_signal_cache_")
    try:
        return tdd.load_index_ohlc(tdd.INDEX_SYMBOL, DOWNLOAD_START, end, refresh=True, cache_dir=tmp)
    except Exception as e:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"]), {"reason": f"exception:{e}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="台指期(小台)日線凍結變體的每日訊號")
    p.add_argument("--output-dir", default="signals", help="輸出資料夾")
    p.add_argument("--as-of", default=None, help="訊號日YYYY-MM-DD，預設今天(台北時間)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    as_of = datetime.date.fromisoformat(args.as_of) if args.as_of else today_in_taipei()
    print(f"台指期訊號：{TX_LIVE_SETTINGS['name']}，訊號日 {as_of.isoformat()}", flush=True)
    df, info = download_index(as_of)
    res = evaluate(df, as_of, info)
    out = write_outputs(res, args.output_dir)
    print(out["markdown"], flush=True)
    print(f"已寫入：{out['latest']}、{out['history']}、{out['telegram']}、{out['paper_log']}", flush=True)
    return {"result": res, **out}


if __name__ == "__main__":
    main()
    sys.exit(0)
