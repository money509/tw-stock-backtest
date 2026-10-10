"""
leader_signal.py —— 領頭股突破(leader_breakout)使用者選定變體的「每日實盤訊號」(紙上交易帳戶)。

用途：每個交易日收盤後(GitHub Actions .github/workflows/leader_daily_signal.yml，台北時間19:20)算一次，告訴你：
  1. 明天開盤要買哪幾檔(最多補滿空位)、買幾股(可零股)、初始停損大約在哪
  2. 目前持股、明天的停損單價位(移動停損只上移)
  3. 今天被停損出場的部位
  4. 紙上帳戶權益/現金/持股數、從 PAPER_START 起的累計損益與勝率

策略 = LEADER_LIVE_SETTINGS(下面那個常數，**唯一**定義的地方)：E_D｜X2｜K3｜無濾網
  - 進場 E_D：收盤創252日新高(前20個交易日都沒有這種新高) 且 營收3個月平均YoY(PIT) > 20% 且 RS ≥ 第80百分位
  - 出場 X2：移動停損 = 進場後最高收盤×(1−20%)；另有初始停損 = 進場價 − 2×ATR14(訊號日)；停損只上移
  - 最多同時持有3檔；每筆風險 = 權益×1.5%；可零股；空位不夠時 RS 高者優先、同分看20日成交金額
  - 無大盤濾網；開盤一價漲停買不到 → 跳過換下一順位；成本(手續費0.1425%最低20元、證交稅0.3%)同回測

狀態：**沒有**會被改寫的狀態檔。每次執行都用回測同一支函式 leader_breakout.simulate_portfolio，
  從 PAPER_START(空手、資金 PAPER_CAPITAL)重播到as_of；選配參數 state_out 讓最後一天不強制平倉，
  直接拿到「現在的持股、明天的停損、明天開盤要補的候選」。所以這裡的進出場/停損/股數跟回測逐筆相同
  (test_leader_signal.py 有驗證)。
  ⚠️ 價格是還原權值價：持股除權息後，過去的進場價/最高收盤會被等比例調整(停損觸發與否不受影響，
  但重播算出的股數/損益可能跟當天報的略有不同)；每天實際報出的內容另外記在 signals/leader_paper_log.csv。

資料(沿用 winners_study.py 的下載器)：ISIN股票清單(每月快取)、yfinance日K(從 min(as_of, PAPER_START)−420天起，
  每次重新下載)、MOPS月營收(從 min(as_of, PAPER_START)−15個月起；舊月份讀快取，最近2個月一律重抓)。

資料沒更新(fail closed)：最新K棒不是as_of那天、當天有K棒的股票太少、或月營收完全抓不到 → 照樣寫檔，
  最上面「⚠️」，而且**不給任何新買進指示**(持股與停損照列，以最新資料日為準)。

輸出(--output-dir，預設 signals/)：leader_latest.md、telegram_leader.txt、leader_history/YYYY-MM-DD.md、
  leader_paper_log.csv(每天一列，同一天重跑覆蓋)、leader_trades_paper.csv(PAPER_START 起已平倉交易)。

使用：
  python leader_signal.py --output-dir signals [--as-of YYYY-MM-DD] [--capital 200000]
"""
import argparse
import datetime
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import leader_breakout as lb
import winners_study as ws
from daily_squeeze_signals import fmt_date, next_business_day, today_in_taipei

# ---------------------------------------------------------------------------
# 凍結的實盤設定 —— 2026-10 使用者在 leader_breakout 回測後選定(事先登錄的4項PASS檢查全部通過；
# 2018–2026 回測 CAGR 16.1%、最大回撤 −31.4%，股票清單有存活者偏誤 → 偏樂觀)。
# 不要依最近的結果調整任何參數(看了結果再改 = 事後挑選)。name 必須跟 leader_breakout.build_variants() 一字不差。
# ---------------------------------------------------------------------------
LEADER_LIVE_SETTINGS = {
    "name": "E_D｜X2｜K3｜無濾網",
    "entry": "E_D",
    "exit": "X2",
    "K": 3,
    "filter": False,
    "rank": "RS",
    "risk_pct": lb.RISK_PCT,              # 1.5%
    "trail_pct": lb.X2_TRAIL_PCT,         # 20%
    "init_stop_atr": lb.INITIAL_STOP_ATR,  # 2×ATR14
    "rs_top": lb.RS_TOP,                  # 0.80
    "rev_yoy_min": lb.REV_YOY_MIN,        # 20%
    "frozen": "2026-10",
}

PAPER_START = "2026-10-12"        # 紙上帳戶第一天(空手)；第一批買進最早在下一個交易日開盤
PAPER_CAPITAL = 200_000
PRICE_LOOKBACK_DAYS = 420          # 252日新高+前20日無新高 需要約273根K棒暖機
REVENUE_LOOKBACK_MONTHS = 15
MIN_COVERAGE = 0.9                 # 最新一天有K棒的股票數 / 前一天 < 90% → 視為資料不完整
TELEGRAM_MAX_CHARS = 3500
RUNNER_UP_N = 5
DEFAULT_CACHE_DIR = "leader_signal_cache"
DEFAULT_DEBUG_DIR = "leader_signal_debug"
PAPER_LOG_COLUMNS = ["date", "equity", "cash", "positions", "new_buys", "exits"]
TRADE_COLUMNS = ["code", "name", "industry", "signal_date", "entry_date", "entry_price", "exit_date", "exit_price",
                 "shares", "buy_fee", "sell_cost", "pnl", "ret", "exit_reason", "hold_days"]
STATUS_LABELS = {
    "ok": "資料已更新",
    "stale": "⚠️ 資料還沒更新／不完整",
    "no_revenue": "⚠️ 月營收抓不到",
    "before_start": f"紙上交易尚未開始({PAPER_START}起)",
}
STOP_ORDER_NOTE = ("停損單：整股(1000股的倍數)可以掛觸價/停損單；零股多數券商不支援停損單 → 每天收盤後看這份報告，"
                   "收盤 ≤ 停損價就隔天開盤賣出。回測假設盤中碰到停損就以停損價成交(跳空開低以開盤價)，"
                   "零股收盤後才處理會跟回測有落差(可能更好也可能更差)。")
ENTRY_ORDER_NOTE = ("進場：明天開盤市價(零股用盤中零股最早一盤)買進；開盤一價鎖漲停買不到就放棄，改買下一順位候補。"
                    "如果持股開盤跳空跌破停損而出場，空出的位置也依序由候補遞補(跟回測相同)。")
ADJ_NOTE = ("價格為還原權值價(除權息後過去的價位會等比例下修)；最新一天的收盤 = 實際收盤。"
            "持股的進場價/最高收盤若遇除權息會跟你實際成交價不同，但停損價已經換算成今天的價位，直接照掛即可。")


def live_variant() -> dict:
    """leader_breakout 變體清單裡跟 LEADER_LIVE_SETTINGS 同名的那一個；欄位不一致直接 ValueError。"""
    v = next((x for x in lb.build_variants() if x["name"] == LEADER_LIVE_SETTINGS["name"]), None)
    if v is None:
        raise ValueError(f"leader_breakout 變體清單裡找不到 {LEADER_LIVE_SETTINGS['name']}")
    for k in ("entry", "exit", "K", "filter", "rank"):
        if v[k] != LEADER_LIVE_SETTINGS[k]:
            raise ValueError(f"LEADER_LIVE_SETTINGS[{k!r}]={LEADER_LIVE_SETTINGS[k]!r} 跟 leader_breakout 的 {v[k]!r} 不一致")
    for k, ref in (("risk_pct", lb.RISK_PCT), ("trail_pct", lb.X2_TRAIL_PCT), ("init_stop_atr", lb.INITIAL_STOP_ATR),
                   ("rs_top", lb.RS_TOP), ("rev_yoy_min", lb.REV_YOY_MIN)):
        if LEADER_LIVE_SETTINGS[k] != ref:
            raise ValueError(f"LEADER_LIVE_SETTINGS[{k!r}] 跟 leader_breakout 不一致")
    return v


def settings_label() -> str:
    s = LEADER_LIVE_SETTINGS
    return (f"{s['name']}：創252日新高(前20日無新高)＋營收3月均YoY>{s['rev_yoy_min']:g}%＋RS≥{s['rs_top'] * 100:g}｜"
            f"移動停損{s['trail_pct']:.0%}＋初始停損{s['init_stop_atr']:g}×ATR｜最多{s['K']}檔｜每筆風險{s['risk_pct']:.1%}")


# ---------------------------------------------------------------------------
# 格式
# ---------------------------------------------------------------------------
def lots_text(n: int) -> str:
    """股數 → 「零股N股」或「X張(+零股Y股)」。"""
    n = int(n)
    if n < 1000:
        return f"零股{n}股"
    z, r = divmod(n, 1000)
    return f"{z}張" + (f"+零股{r}股" if r else "")


def money(x, signed=False) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    s = f"{abs(x):,.0f}"
    if signed:
        return ("+" if x > 0 else "−" if x < 0 else "") + "NT$" + s
    return ("−" if x < 0 else "") + "NT$" + s


def px(x) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:,.2f}"


def pct(x, nd=1, signed=False) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x * 100:+.{nd}f}%" if signed else f"{x * 100:.{nd}f}%"


# ---------------------------------------------------------------------------
# 重播(純函式，不碰網路/檔案)
# ---------------------------------------------------------------------------
def truncate_prices(raw_prices: dict, as_of) -> dict:
    """只留 as_of(含)以前的K棒 → 之後的資料完全不會被用到。"""
    as_of = pd.Timestamp(as_of)
    out = {}
    for code, df in (raw_prices or {}).items():
        if df is None or len(df) == 0:
            continue
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        out[code] = df[np.asarray(idx.normalize() <= as_of)]
    return out


def replay(P: dict, capital: float = PAPER_CAPITAL, start=PAPER_START, i1: int = None) -> dict:
    """用回測同一支 simulate_portfolio 從 start(空手、資金capital) 重播到 i1(預設最後一天)。
    回傳 {started, i0, i1, sim(回測回傳值), state(state_out), sig}。start 之後沒有K棒 → started=False。"""
    v = live_variant()
    cal = P["cal"]
    sig = lb.variant_signal(lb.entry_signals(P), P, v)
    i1 = len(cal) - 1 if i1 is None else int(i1)
    i0 = max(int(cal.searchsorted(pd.Timestamp(start))), 1)
    if i1 < i0:
        A = P["A"]
        prim, sec = A["rs"], A["turnover20"]
        row = np.flatnonzero(sig[i1])
        state = {"positions": [], "cash": float(capital), "equity": float(capital), "i0": i0, "i1": i1, "k": v["K"],
                 "pending": lb.rank_order(row, prim[i1, row], sec[i1, row]) if len(row) else np.array([], dtype=int)}
        return {"started": False, "i0": i0, "i1": i1, "sim": {"trades": [], "open": []}, "state": state, "sig": sig}
    state = {}
    sim = lb.simulate_portfolio(P, sig, i0, i1, v["exit"], v["K"], capital, rank=v["rank"], state_out=state)
    return {"started": True, "i0": i0, "i1": i1, "sim": sim, "state": state, "sig": sig}


def plan_entries(P: dict, state: dict, t: int, k: int, runner_n: int = RUNNER_UP_N) -> dict:
    """明天開盤的買進計畫(以今天收盤估)：候選依回測排序(state['pending'])，跳過已持有的；依序估股數
    (position_shares，跟回測同一個公式，用今天收盤代替明天開盤)，股數<1 的跳過換下一檔(回測也是)；
    補滿 K−持股數 個空位。其餘前 runner_n 檔列為候補(僅供參考)。"""
    A = P["A"]
    held = {p["j"] for p in state["positions"]}
    free = k - len(state["positions"])
    cash = float(state["cash"])
    eq = float(state["equity"])
    buys, runners, too_small = [], [], []
    rank_no = 0
    for j in state["pending"]:
        j = int(j)
        if j in held:
            continue
        rank_no += 1
        close, atr = float(A["close"][t, j]), float(A["atr14"][t, j])
        info = {"j": j, "rank": rank_no, "close": close, "atr": atr,
                "rs": float(A["rs"][t, j]), "yoy3": float(A["yoy3"][t, j]),
                "turnover20": float(A["turnover20"][t, j]),
                "stop_dist": lb.INITIAL_STOP_ATR * atr, "approx_stop": close - lb.INITIAL_STOP_ATR * atr}
        if len(buys) < free:
            n = lb.position_shares(eq, cash, close, atr, k)
            if n < 1:
                too_small.append(info)
                continue
            notional = n * close
            fee = lb.buy_fee(notional)
            cash -= notional + fee
            info.update({"shares": int(n), "cost": notional + fee, "fee": fee,
                         "risk": n * lb.INITIAL_STOP_ATR * atr})
            buys.append(info)
        elif len(runners) < runner_n:
            runners.append(info)
    return {"free": free, "buys": buys, "runners": runners, "too_small": too_small, "cash_after": cash}


def evaluate(universe: pd.DataFrame, raw_prices: dict, revenue_long: pd.DataFrame, as_of,
             capital: float = PAPER_CAPITAL, info: dict = None) -> dict:
    """主要邏輯：回傳 result dict(markdown/telegram/紙上紀錄都從這裡產生)。raw_prices 可以含as_of之後的K棒(不會用到)。"""
    info = info or {}
    as_of = pd.Timestamp(as_of).normalize()
    k = LEADER_LIVE_SETTINGS["K"]
    res = {"as_of": as_of, "capital": float(capital), "status": "stale", "stale_reason": None,
           "latest_date": None, "data_fresh": False, "warnings": list(info.get("warnings", [])),
           "holdings": [], "exits_today": [], "plan": None, "candidates_info": [],
           "account": {"equity": float(capital), "cash": float(capital), "npos": 0, "k": k,
                       "cum_pnl": 0.0, "cum_ret": 0.0, "closed": 0, "wins": 0, "win_rate": np.nan,
                       "realized": 0.0},
           "closed_trades": [], "next_day": next_business_day(as_of), "coverage": np.nan,
           "n_stocks": 0, "n_elig": 0, "n_signals": 0}
    prices = truncate_prices(raw_prices, as_of)
    try:
        P = lb.build_panels(universe, prices, revenue_long, pd.Series(dtype=float))
    except ValueError as e:
        res["stale_reason"] = f"股價資料不足，無法計算({e})"
        return res
    cal, A = P["cal"], P["A"]
    codes, names, inds = P["codes"], P["names"], P["industries"]
    t = len(cal) - 1
    latest = pd.Timestamp(cal[t]).normalize()
    res["latest_date"] = latest
    res["next_day"] = next_business_day(latest if latest >= as_of else as_of)
    res["n_stocks"] = len(codes)
    res["n_elig"] = int(A["elig"][t].sum())
    cov = float(A["valid"][t].sum() / max(1, A["valid"][t - 1].sum())) if t >= 1 else 1.0
    res["coverage"] = cov
    fresh = bool(latest == as_of)
    if not fresh:
        res["stale_reason"] = (f"最新股價日期是 {fmt_date(latest)}，不是 {fmt_date(as_of)}(資料源還沒更新，或今天休市)")
    elif cov < MIN_COVERAGE:
        fresh = False
        res["stale_reason"] = (f"{fmt_date(latest)} 只有 {A['valid'][t].sum()} 檔有K棒(前一天 {A['valid'][t - 1].sum()} 檔，"
                               f"{cov:.0%})，資料可能還沒更新完")
    res["data_fresh"] = fresh

    rp = replay(P, capital)
    st, sim = rp["state"], rp["sim"]
    res["n_signals"] = int(rp["sig"][t].sum())
    cff = A["close_ff"]
    # ---- 持股
    for p in st["positions"]:
        j = p["j"]
        last = float(cff[t, j])
        has_bar = bool(A["valid"][t, j])
        moved = has_bar and p["stop"] > p.get("stop_before_close", p["stop"]) + 1e-9
        prev_stop = p.get("stop_before_close", p["stop"]) if moved else p["stop"]
        mv = p["shares"] * last
        res["holdings"].append({
            "code": codes[j], "name": names[j], "industry": inds[j],
            "signal_date": cal[p["sig_idx"]], "entry_date": cal[p["entry_idx"]], "entry_price": p["entry_px"],
            "shares": int(p["shares"]), "last_close": last, "has_bar": has_bar,
            "highest_close": float(p["hc"]) if np.isfinite(p["hc"]) else np.nan,
            "init_stop": float(p["init_stop"]), "stop": float(p["stop"]), "prev_stop": float(prev_stop),
            "trail_level": float(p["hc"] * (1 - lb.X2_TRAIL_PCT)) if np.isfinite(p["hc"]) else np.nan,
            "stop_moved_up": bool(moved), "dist_pct": (last - p["stop"]) / last if last > 0 else np.nan,
            "market_value": mv, "cost_basis": float(p["cost_basis"]),
            "unrealized": mv - p["cost_basis"], "unrealized_pct": (mv - p["cost_basis"]) / p["cost_basis"],
            "days_held": int(t - p["entry_idx"] + 1), "entered_today": p["entry_idx"] == t})
    # ---- 今天出場 / 已平倉
    closed = []
    for tr in sim["trades"]:
        j = tr["code_idx"]
        row = {"code": codes[j], "name": names[j], "industry": inds[j], "signal_date": cal[tr["signal_idx"]],
               "entry_date": cal[tr["entry_idx"]], "exit_date": cal[tr["exit_idx"]],
               **{c: tr[c] for c in ("entry_price", "exit_price", "shares", "buy_fee", "sell_cost", "pnl", "ret",
                                     "exit_reason", "hold_days")}}
        closed.append(row)
        if tr["exit_idx"] == t:
            res["exits_today"].append(row)
    res["closed_trades"] = closed
    # ---- 帳戶
    acc = res["account"]
    pnls = np.array([c["pnl"] for c in closed], dtype=float)
    acc.update({"equity": float(st["equity"]), "cash": float(st["cash"]), "npos": len(st["positions"]),
                "cum_pnl": float(st["equity"] - capital), "cum_ret": float(st["equity"] / capital - 1),
                "closed": len(closed), "wins": int((pnls > 0).sum()),
                "win_rate": float((pnls > 0).mean()) if len(pnls) else np.nan,
                "realized": float(pnls.sum()) if len(pnls) else 0.0})
    # ---- 明天的買進計畫
    plan = plan_entries(P, st, t, k)
    for lst in ("buys", "runners", "too_small"):
        for x in plan[lst]:
            x.update({"code": codes[x["j"]], "name": names[x["j"]], "industry": inds[x["j"]]})
    if not rp["started"]:
        res["status"] = "before_start"
        res["candidates_info"] = plan["buys"] + plan["runners"]
        plan = None
    elif not fresh:
        res["status"] = "stale"
        plan = None
    elif not info.get("revenue_ok", True):
        res["status"] = "no_revenue"
        res["stale_reason"] = "月營收(MOPS)完全抓不到，E_D 的營收條件無法判斷"
        plan = None
    else:
        res["status"] = "ok"
    res["plan"] = plan
    return res


# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------
def _buy_lines(b: dict, md: bool) -> list:
    bold = "**" if md else ""
    return [
        f"{b['code']} {b['name']}({b['industry'] or '-'})｜訊號收盤 {px(b['close'])}",
        f"買 {bold}{b['shares']:,}股({lots_text(b['shares'])}){bold}，約 {money(b['cost'])}(含手續費，以今天收盤估)",
        f"初始停損 ≈ {bold}{px(b['approx_stop'])}{bold}(收盤−2×ATR14；ATR {px(b['atr'])}，停損距離 {px(b['stop_dist'])}"
        f" = {pct(b['stop_dist'] / b['close'])})；實際停損 = 成交價 − {px(b['stop_dist'])}",
        f"RS {b['rs'] * 100:.1f}｜營收3月均YoY {b['yoy3']:.1f}%｜20日均成交金額 {b['turnover20'] / 1e8:.1f}億"
        f"｜排序第{b['rank']}(RS高者優先，同分看成交金額)",
    ]


def _holding_lines(h: dict, md: bool) -> list:
    bold = "**" if md else ""
    moved = (f"今天上移(原 {px(h['prev_stop'])})" if h["stop_moved_up"] else "今天沒變")
    odd = h["shares"] % 1000
    if h["shares"] < 1000:
        order = "零股：收盤後檢查，收盤 ≤ 停損 → 隔天開盤賣出"
    elif odd:
        order = f"整張部分掛停損單；零股{odd}股收盤後檢查"
    else:
        order = "可掛觸價/停損單"
    return [
        f"{h['code']} {h['name']}({h['industry'] or '-'})｜{h['shares']:,}股({lots_text(h['shares'])})",
        f"進場 {h['entry_date'].date()} @ {px(h['entry_price'])}｜收盤 {px(h['last_close'])}"
        f"{'' if h['has_bar'] else '(今天沒有成交資料)'}｜進場後最高收盤 {px(h['highest_close'])}",
        f"明天停損單價位 {bold}{px(h['stop'])}{bold}({moved}；= max(初始停損 {px(h['init_stop'])}, "
        f"最高收盤×0.8 = {px(h['trail_level'])}))｜距收盤 {pct(h['dist_pct'])}｜{order}",
        f"未實現 {money(h['unrealized'], True)}({pct(h['unrealized_pct'], 1, True)}，未扣賣出成本)｜持有第 {h['days_held']} 個交易日",
    ]


def _exit_line(x: dict) -> str:
    return (f"{x['code']} {x['name']}：{x['exit_reason']}，出場 {px(x['exit_price'])}(進場 {x['entry_date'].date()} @ "
            f"{px(x['entry_price'])}，{x['shares']:,}股)，損益 {money(x['pnl'], True)}({pct(x['ret'], 1, True)}，含成本)")


def _account_line(res: dict) -> str:
    a = res["account"]
    wr = f"{a['win_rate'] * 100:.0f}%" if np.isfinite(a["win_rate"]) else "-"
    return (f"權益 {money(a['equity'])}｜現金 {money(a['cash'])}｜持股 {a['npos']}/{a['k']}｜"
            f"累計 {money(a['cum_pnl'], True)}({pct(a['cum_ret'], 1, True)})｜已平倉 {a['closed']} 筆、勝率 {wr}")


def render_markdown(res: dict) -> str:
    as_of, st = res["as_of"], res["status"]
    L = [f"# 強勢股(領頭股突破)每日訊號 {fmt_date(as_of)}", "", f"**策略：{settings_label()}**", "",
         f"紙上交易帳戶：{PAPER_START} 起、起始資金 {money(res['capital'])}(每天從頭重播，跟回測同一支模擬函式)", ""]
    latest = res["latest_date"]
    if st in ("stale", "no_revenue"):
        L += ["> ## ⚠️ 資料還沒更新／不完整，今天不給新的買進指示",
              f"> {res['stale_reason']}。**不要依這份報告買新股票**，請晚點到Actions手動重跑。"
              "持股的停損以最新資料日為準(停損只會上移，原本的停損單繼續有效)。", ""]
    for w in res["warnings"]:
        L.append(f"> ⚠️ {w}")
    if res["warnings"]:
        L.append("")
    L += [f"- 訊號日 {as_of.date()}｜資料最新 {latest.date() if latest is not None else '無'} "
          f"{'✅' if res['data_fresh'] else '❌'}｜有股價 {res['n_stocks']} 檔、今天合格 {res['n_elig']} 檔、"
          f"E_D訊號 {res['n_signals']} 檔", ""]

    L += ["## 1. 明天開盤要買", ""]
    plan = res["plan"]
    if st == "before_start":
        L += [f"紙上交易從 {PAPER_START} 收盤後開始，第一批買進最早在 {PAPER_START} 的下一個交易日開盤。今天不用下單。", ""]
        if res["candidates_info"]:
            L += ["今天符合條件的股票(僅供參考，不是買進指示)："]
            L += [f"- {c['code']} {c['name']}({c['industry'] or '-'}) 收盤 {px(c['close'])}，RS {c['rs'] * 100:.1f}，"
                  f"營收YoY {c['yoy3']:.1f}%" for c in res["candidates_info"]]
            L.append("")
    elif plan is None:
        L += ["今天不給買進指示(見上方警告)。", ""]
    else:
        if not plan["buys"]:
            why = "沒有空位(已持有3檔)" if plan["free"] <= 0 else "今天沒有符合條件的新訊號"
            L += [f"明天({fmt_date(res['next_day'])})不用買。{why}。", ""]
        else:
            L += [f"📌 **{fmt_date(res['next_day'])}(下一交易日，遇假日順延)開盤買進 {len(plan['buys'])} 檔**"
                  f"(空位 {plan['free']} 個)", ""]
            for i, b in enumerate(plan["buys"], 1):
                lines = _buy_lines(b, True)
                L.append(f"{i}. {lines[0]}")
                L += [f"   - {x}" for x in lines[1:]]
            L.append("")
        if plan["too_small"]:
            L += ["資金/股數不足而跳過(回測也一樣跳過)：" + "、".join(f"{x['code']} {x['name']}" for x in plan["too_small"]), ""]
        if plan["runners"]:
            L += ["候補(沒有空位所以沒買，僅供參考；買進的股票開盤漲停買不到、或持股開盤跳空停損時依序遞補)："]
            L += [f"- {r['code']} {r['name']}({r['industry'] or '-'}) 收盤 {px(r['close'])}，RS {r['rs'] * 100:.1f}，"
                  f"營收YoY {r['yoy3']:.1f}%，停損≈{px(r['approx_stop'])}" for r in plan["runners"]]
            L.append("")

    L += ["## 2. 目前持股(明天的停損單)", ""]
    if not res["holdings"]:
        L += ["空手。", ""]
    else:
        if st in ("stale", "no_revenue") and latest is not None:
            L += [f"(截至 {latest.date()} 的資料)", ""]
        for h in res["holdings"]:
            lines = _holding_lines(h, True)
            L.append(f"- {lines[0]}" + ("　🆕今天開盤買進" if h["entered_today"] else ""))
            L += [f"  - {x}" for x in lines[1:]]
        L.append("")

    L += [f"## 3. 今天出場({latest.date() if latest is not None else '-'})", ""]
    if res["exits_today"]:
        L += [f"- {_exit_line(x)}" for x in res["exits_today"]] + [""]
    else:
        L += ["無。", ""]

    a = res["account"]
    L += ["## 4. 帳戶(紙上交易)", "", f"- {_account_line(res)}",
          f"- 已實現損益 {money(a['realized'], True)}；權益 = 現金 + 持股×收盤(未扣賣出成本)", ""]
    if res["closed_trades"]:
        L += ["| 代號 | 名稱 | 進場日 | 進場價 | 出場日 | 出場價 | 股數 | 原因 | 損益 |", "|---|---|---|---:|---|---:|---:|---|---:|"]
        for c in res["closed_trades"]:
            L.append(f"| {c['code']} | {c['name']} | {c['entry_date'].date()} | {px(c['entry_price'])} | "
                     f"{c['exit_date'].date()} | {px(c['exit_price'])} | {c['shares']:,} | {c['exit_reason']} | "
                     f"{money(c['pnl'], True)} |")
        L.append("")

    L += ["## 5. 注意事項", "", f"- {STOP_ORDER_NOTE}", f"- {ENTRY_ORDER_NOTE}",
          "- 股數 = floor(權益×1.5% ÷ (2×ATR14))，上限 權益/3 與 可用現金；這裡用今天收盤估，明天以開盤價重算差不多即可。",
          f"- {ADJ_NOTE}",
          "- ⚠️ 回測(2018–2026 CAGR 16.1%、最大回撤 −31.4%)有存活者偏誤且是從38個變體中選的，偏樂觀；紙上交易的結果才是真正的檢驗。",
          "- 規則凍結(2026-10)，不要依最近的結果改參數。", ""]
    return "\n".join(L)


def render_telegram(res: dict, link: str = None) -> str:
    """純文字、< TELEGRAM_MAX_CHARS。第一行【強勢股訊號】日期。太長時先刪候補，再截斷。"""
    as_of, st = res["as_of"], res["status"]
    head = [f"【強勢股訊號】{as_of.date().isoformat()}",
            f"{LEADER_LIVE_SETTINGS['name']}｜紙上交易(起始{res['capital'] / 10000:g}萬，{PAPER_START}起)",
            _account_line(res)]
    body = []
    if st in ("stale", "no_revenue"):
        body.append(f"⚠️ {res['stale_reason']}。今天不給買進指示，請晚點重跑。")
    for w in res["warnings"][:3]:
        body.append(f"⚠️ {w}")
    plan = res["plan"]
    body.append("")
    body.append(f"＝明天({fmt_date(res['next_day'])})開盤要買＝")
    if st == "before_start":
        body.append("紙上交易尚未開始，今天不用下單。")
    elif plan is None:
        body.append("無(資料問題)")
    elif not plan["buys"]:
        body.append("不用買(" + ("沒有空位" if plan["free"] <= 0 else "沒有新訊號") + ")")
    else:
        for i, b in enumerate(plan["buys"], 1):
            body.append(f"{i}. {b['code']} {b['name']}({b['industry'] or '-'}) 收{px(b['close'])}")
            body.append(f"  買{b['shares']:,}股({lots_text(b['shares'])}) 約{money(b['cost'])}")
            body.append(f"  停損≈{px(b['approx_stop'])}(成交價−{px(b['stop_dist'])}，2×ATR {px(b['atr'])}，"
                        f"距{pct(b['stop_dist'] / b['close'])})")
            body.append(f"  RS {b['rs'] * 100:.0f}｜營收YoY {b['yoy3']:.0f}%｜排序第{b['rank']}")
    body.append("")
    body.append("＝目前持股(明天停損單)＝")
    if not res["holdings"]:
        body.append("空手")
    for h in res["holdings"]:
        mv = "↑今天上移" if h["stop_moved_up"] else "沒變"
        new = "🆕" if h["entered_today"] else ""
        body.append(f"{new}{h['code']} {h['name']} {h['shares']:,}股({lots_text(h['shares'])})")
        body.append(f"  進場{h['entry_date'].strftime('%m-%d')}@{px(h['entry_price'])} 收{px(h['last_close'])} "
                    f"高{px(h['highest_close'])}")
        body.append(f"  停損{px(h['stop'])}({mv}) 距{pct(h['dist_pct'])} 未實現{money(h['unrealized'], True)}"
                    f" 第{h['days_held']}天")
    if res["exits_today"]:
        body.append("")
        body.append("＝今天出場＝")
        for x in res["exits_today"]:
            body.append(f"{x['code']} {x['name']} {x['exit_reason']} @{px(x['exit_price'])} {money(x['pnl'], True)}")
    tail = ["", "整股可掛停損單；零股多半不行 → 收盤跌破停損就隔天開盤賣。開盤一價漲停買不到就換候補。"]
    runners = []
    if plan is not None and plan["runners"]:
        runners = [f"{r['code']} {r['name']} 收{px(r['close'])} RS {r['rs'] * 100:.0f} 停損≈{px(r['approx_stop'])}"
                   for r in plan["runners"]]
    footer = [f"完整說明：{link}"] if link else []

    def join(n_runners, ft):
        rn = (["", "＝候補(沒空位，僅供參考)＝"] + runners[:n_runners]) if n_runners > 0 else []
        return "\n".join(head + body + rn + tail + ft)
    # 太長 → 先拿掉連結，再從最後一個候補開始刪；還是太長才硬截斷
    for n in range(len(runners), -1, -1):
        for ft in (footer, []):
            text = join(n, ft)
            if len(text) < TELEGRAM_MAX_CHARS:
                return text
    text = join(0, [])
    return text[:TELEGRAM_MAX_CHARS - 20].rsplit("\n", 1)[0] + "\n…(其餘看完整版)"


def paper_log_row(res: dict) -> dict:
    a = res["account"]
    pos = "、".join(f"{h['code']}×{h['shares']}" for h in res["holdings"])
    if res["status"] in ("stale", "no_revenue"):
        buys = f"資料未更新(截至{res['latest_date'].date()})" if res["latest_date"] is not None else "資料未更新"
    elif res["plan"] is not None:
        buys = "、".join(f"{b['code']}×{b['shares']}" for b in res["plan"]["buys"])
    else:
        buys = ""
    exits = "、".join(f"{x['code']}({x['exit_reason']}{x['pnl']:+.0f})" for x in res["exits_today"])
    return {"date": res["as_of"].date().isoformat(), "equity": round(a["equity"]), "cash": round(a["cash"]),
            "positions": pos, "new_buys": buys, "exits": exits}


def update_paper_log(path: str, row: dict) -> pd.DataFrame:
    """每次執行一列；同一天重跑 → 取代那一列(冪等)。依日期排序。"""
    old = pd.DataFrame(columns=PAPER_LOG_COLUMNS)
    if os.path.exists(path):
        try:
            old = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            pass
    old = old.reindex(columns=PAPER_LOG_COLUMNS)
    old = old[old["date"] != row["date"]]
    new = pd.concat([old, pd.DataFrame([{k: str(row[k]) for k in PAPER_LOG_COLUMNS}])], ignore_index=True)
    new = new.sort_values("date", kind="stable").reset_index(drop=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")
    return new


def trades_frame(res: dict) -> pd.DataFrame:
    """PAPER_START 起的已平倉交易(重播；進場日 ≥ PAPER_START)。"""
    start = pd.Timestamp(PAPER_START)
    rows = [{c: t[c] for c in TRADE_COLUMNS} for t in res["closed_trades"] if t["entry_date"] >= start]
    df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    for c in ("signal_date", "entry_date", "exit_date"):
        df[c] = pd.to_datetime(df[c]).dt.date
    return df.rename(columns=lb.TRADE_LABELS)


def write_outputs(res: dict, output_dir: str) -> dict:
    os.makedirs(os.path.join(output_dir, "leader_history"), exist_ok=True)
    md = render_markdown(res)
    day = res["as_of"].date().isoformat()
    paths = {"latest": os.path.join(output_dir, "leader_latest.md"),
             "history": os.path.join(output_dir, "leader_history", f"{day}.md"),
             "telegram": os.path.join(output_dir, "telegram_leader.txt"),
             "paper_log": os.path.join(output_dir, "leader_paper_log.csv"),
             "trades": os.path.join(output_dir, "leader_trades_paper.csv")}
    for k in ("latest", "history"):
        with open(paths[k], "w", encoding="utf-8") as f:
            f.write(md)
    tg = render_telegram(res, link=os.environ.get("LEADER_SIGNALS_LINK"))
    with open(paths["telegram"], "w", encoding="utf-8") as f:
        f.write(tg)
    update_paper_log(paths["paper_log"], paper_log_row(res))
    trades_frame(res).to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    return {"markdown": md, "telegram_text": tg, **paths}


# ---------------------------------------------------------------------------
# 下載(只在GitHub Actions上真正執行；測試monkeypatch ws.load_universe/load_prices/load_revenue)
# ---------------------------------------------------------------------------
def data_windows(as_of: datetime.date):
    """(股價起, 股價迄(不含), 月營收起始月, 月營收結束月)。從 min(as_of, PAPER_START) 往回推，
    確保紙上交易第一天也有完整暖機(之後每天重播的結果才會一樣)。"""
    base = min(pd.Timestamp(as_of), pd.Timestamp(PAPER_START))
    dl_start = (base - pd.Timedelta(days=PRICE_LOOKBACK_DAYS)).date().isoformat()
    dl_end = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).date().isoformat()
    rev_start = (base - pd.DateOffset(months=REVENUE_LOOKBACK_MONTHS)).normalize().replace(day=1)
    return dl_start, dl_end, rev_start, pd.Timestamp(as_of).normalize()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="領頭股突破 E_D｜X2｜K3｜無濾網 每日訊號(紙上交易)")
    p.add_argument("--output-dir", default="signals")
    p.add_argument("--as-of", default=None, help="訊號日YYYY-MM-DD，預設今天(台北時間)")
    p.add_argument("--capital", type=float, default=PAPER_CAPITAL)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="月營收/股票清單快取(股價每次重抓)")
    p.add_argument("--debug-dir", default=DEFAULT_DEBUG_DIR)
    return p.parse_args(argv)


def main(argv=None) -> int:
    t0 = time.time()
    args = parse_args(argv)
    as_of = datetime.date.fromisoformat(args.as_of) if args.as_of else today_in_taipei()
    dl_start, dl_end, rev_start, rev_end = data_windows(as_of)
    print(f"強勢股訊號：{LEADER_LIVE_SETTINGS['name']}，訊號日 {as_of}，資金 {args.capital:,.0f}；"
          f"股價 {dl_start} 起、月營收 {rev_start.date()} 起", flush=True)
    os.makedirs(args.debug_dir, exist_ok=True)
    timing = {}
    warnings = []
    t = time.time()
    universe, w = ws.load_universe(args.cache_dir, False, args.debug_dir, as_of.strftime("%Y-%m"))  # 每月一份快取
    warnings += w
    timing["股票清單"] = time.time() - t
    if universe is None or universe.empty:
        print("⚠️ 上市/上櫃股票清單都抓不到(ISIN頁面)，無法產生訊號", flush=True)
        return 1
    print(f"股票清單 {len(universe)} 檔", flush=True)
    t = time.time()
    tmp = tempfile.mkdtemp(prefix="leader_signal_px_")
    try:   # 股價每次重新下載(暫存資料夾)，避免讀到稍早資料還沒更新時存的快取
        prices, fail_df = ws.load_prices(universe, dl_start, dl_end, tmp, True, args.debug_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    timing["股價"] = time.time() - t
    if len(prices) == 0:
        print("⚠️ 股價全部下載失敗(yfinance)，無法產生訊號", flush=True)
        return 1
    fail_ratio = len(fail_df) / max(1, len(universe))
    if fail_ratio > 0.05:
        warnings.append(f"股價下載失敗 {len(fail_df)} 檔({fail_ratio:.0%})，持股/候選可能不完整")
    t = time.time()
    revenue, rev_ok = ws.load_revenue(rev_start, rev_end, args.cache_dir, False, args.debug_dir)
    timing["月營收"] = time.time() - t
    t = time.time()
    res = evaluate(universe, prices, revenue, as_of, args.capital,
                   {"warnings": warnings, "revenue_ok": rev_ok})
    timing["計算"] = time.time() - t
    out = write_outputs(res, args.output_dir)
    print(out["markdown"], flush=True)
    total = time.time() - t0
    print("執行時間：" + "、".join(f"{k} {v:.0f}秒" for k, v in timing.items()) + f"；總計 {total / 60:.1f} 分鐘", flush=True)
    print(f"已寫入：{out['latest']}、{out['history']}、{out['telegram']}、{out['paper_log']}、{out['trades']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
