"""
均值回歸雙向策略回測引擎。

跟 engine.py (動能突破策略) 是兩套獨立邏輯，共用「一次只能持有一個部位」、
「保證金查表」這些跟production行為一致的限制。

已知簡化/近似之處（誠實列出，不是bug，是資料/工程上的合理取捨）：
1. 結算日用「每月第三個星期三」近似計算，若該月剛好遇到國定假日順延，可能有1天誤差。
2. 大盤氛圍濾網用 0050.TW (元大台灣50) 的均線位置近似大盤趨勢，不是直接用加權指數，
   因為 0050 資料品質穩定、流動性極佳，跟大盤高度連動。
3. 假日已經隱含在價格資料本身裡 (yfinance不會回傳休市日資料)，不需要另外維護假日清單。
"""
import pandas as pd
import numpy as np

from taifex_universe import get_contract_multiplier, estimate_margin

COMMISSION_PER_LOT_PER_LEG = 200  # 新台幣，沿用Gemini那份回測的假設
STOP_LOSS_COOLDOWN_DAYS = 5       # 同一檔股票停損出場後，幾個交易日內不再進場
DEFAULT_MARGIN_CAP_RATIO = 0.35   # 單筆交易保證金不得超過總資金的比例


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 修正：avg_loss剛好是0代表這段期間完全沒有下跌日 (強勢上漲)，RSI該是100，
    # 不是資料不足，不能塞中性值50，否則會把真正的強勢訊號誤判成中性
    all_gain_mask = (avg_loss == 0) & (avg_gain > 0)
    rsi = rsi.where(~all_gain_mask, 100.0)

    # avg_gain跟avg_loss都是0代表完全走平盤，RSI才是中性值50
    flat_mask = (avg_loss == 0) & (avg_gain == 0)
    rsi = rsi.where(~flat_mask, 50.0)

    # 只有真正資料不足 (rolling window還沒填滿) 才用中性值50頂替
    return rsi.fillna(50.0)


def compute_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def compute_atr_correct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def is_near_settlement(date, days_before: int = 2) -> bool:
    """近似判斷是否在結算日前 N 天內 (結算日近似為當月第三個星期三)。"""
    d = pd.Timestamp(date)
    first_of_month = d.replace(day=1)
    wednesdays = [first_of_month + pd.Timedelta(days=i) for i in range(31)
                  if (first_of_month + pd.Timedelta(days=i)).month == first_of_month.month
                  and (first_of_month + pd.Timedelta(days=i)).weekday() == 2]
    if len(wednesdays) < 3:
        return False
    settlement = wednesdays[2]
    days_to_settlement = (settlement - d).days
    return 0 <= days_to_settlement <= days_before


def compute_regime(index_df: pd.DataFrame, as_of_date, ma_period: int = 120) -> str:
    """
    回傳 'bull' / 'bear'，依大盤代理指標 (預設0050) 相對長期均線的位置判斷。
    找不到足夠資料時回傳 'neutral'，此時多空訊號都正常放行(不縮小也不停用)。
    """
    hist = index_df[index_df.index < as_of_date]
    if len(hist) < ma_period:
        return "neutral"
    ma = hist["Close"].rolling(ma_period).mean().iloc[-1]
    last_close = hist["Close"].iloc[-1]
    if pd.isna(ma):
        return "neutral"
    return "bull" if last_close > ma else "bear"


def scan_mean_reversion_candidates(price_data: dict, universe: dict, as_of_date,
                                    regime: str, excluded_codes: set, allow_short: bool = True,
                                    rsi_long_threshold: float = 30, rsi_short_threshold: float = 70):
    """
    掃描全市場候選標的，回傳依「偏離程度」排序的前3名多方候選、前3名空方候選。
    regime == 'bull' 時停用空方訊號；regime == 'bear' 時停用多方訊號；'neutral' 兩者都放行。
    allow_short=False 時，不管regime是什麼，永遠不產生空方候選 (獨立於氛圍濾網的開關)。
    rsi_long_threshold/rsi_short_threshold 可以調鬆一點(例如35/65)來增加訊號出現頻率，
    預設30/70是教科書常見門檻。
    """
    long_candidates = []
    short_candidates = []

    for code in universe:
        if code in excluded_codes:
            continue
        df = price_data.get(code)
        if df is None:
            continue
        hist = df[df.index < as_of_date]
        if len(hist) < 60:
            continue

        close = hist["Close"]
        rsi = compute_rsi(close).iloc[-1]
        mid, upper, lower = compute_bollinger(close)
        mid_v, upper_v, lower_v = mid.iloc[-1], upper.iloc[-1], lower.iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        last_close = close.iloc[-1]
        atr = compute_atr_correct(hist).iloc[-1]

        if pd.isna(mid_v) or pd.isna(ma60) or pd.isna(atr) or atr <= 0:
            continue

        # 做多：RSI超賣 + 跌破布林下軌 + 仍在60日均線之上
        if regime != "bear":
            if rsi < rsi_long_threshold and last_close <= lower_v and last_close > ma60:
                score = rsi_long_threshold - rsi  # 越超賣分數越高
                long_candidates.append({
                    "code": code, "side": "long", "score": score,
                    "c_prev": last_close, "mid": mid_v, "lower": lower_v, "atr": atr,
                })

        # 做空：RSI超買 + 站上布林上軌 + 仍在60日均線之下 (allow_short=False時完全不產生空方候選)
        if allow_short and regime != "bull":
            if rsi > rsi_short_threshold and last_close >= upper_v and last_close < ma60:
                score = rsi - rsi_short_threshold
                short_candidates.append({
                    "code": code, "side": "short", "score": score,
                    "c_prev": last_close, "mid": mid_v, "upper": upper_v, "atr": atr,
                })

    long_candidates.sort(key=lambda x: x["score"], reverse=True)
    short_candidates.sort(key=lambda x: x["score"], reverse=True)
    return long_candidates[:3] + short_candidates[:3]


def try_enter_mean_reversion(price_data: dict, candidates: list, entry_date,
                              starting_capital: float, lots: int = 2):
    """依序檢查候選名單(已跳空風控+保證金上限過濾)，第一個通過的進場。"""
    for cand in candidates:
        code = cand["code"]
        df = price_data.get(code)
        if df is None or entry_date not in df.index:
            continue
        open_p = df.loc[entry_date, "Open"]
        c_prev = cand["c_prev"]
        if c_prev == 0:
            continue

        gap_pct = (open_p - c_prev) / c_prev
        # 跳空風控：不管多空，方向不利的跳空超過0.5%就放棄
        if cand["side"] == "long" and gap_pct <= -0.005:
            continue
        if cand["side"] == "short" and gap_pct >= 0.005:
            continue

        margin_needed = estimate_margin(code, open_p, lots)
        if margin_needed > starting_capital * DEFAULT_MARGIN_CAP_RATIO:
            continue  # 保證金超過上限，跳過遞補下一名

        atr = cand["atr"]
        e_price = open_p
        side = cand["side"]
        if side == "long":
            target_price = cand["mid"]
            stop_price = cand["lower"] - 1.0 * atr
        else:
            target_price = cand["mid"]
            stop_price = cand["upper"] + 1.0 * atr

        return {
            "code": code, "side": side, "entry_date": entry_date,
            "e_price": e_price, "target_price": target_price, "stop_price": stop_price,
            "lots": lots, "hold_days": 1,
        }
    return None


def check_exit(row, position):
    """回傳 (event, exit_price)，event 為 'target' / 'stop' / None。"""
    high, low = row["High"], row["Low"]
    if position["side"] == "long":
        # 保守假設：同一天內若停損停利都可能觸及，優先判定停損
        if low <= position["stop_price"]:
            return "stop", position["stop_price"]
        if high >= position["target_price"]:
            return "target", position["target_price"]
    else:  # short
        if high >= position["stop_price"]:
            return "stop", position["stop_price"]
        if low <= position["target_price"]:
            return "target", position["target_price"]
    return None, None


def run_mean_reversion_backtest(price_data: dict, index_df: pd.DataFrame, universe: dict,
                                 master_calendar: pd.DatetimeIndex, max_hold_days: int,
                                 starting_capital: float, allow_short: bool = True,
                                 lots: int = 2, rsi_long_threshold: float = 30,
                                 rsi_short_threshold: float = 70):
    """
    完整 day-by-day walk-forward 模擬。
    max_hold_days: 短線版建議3-5，中期版建議10-20 (交易日)。
    allow_short=False 時只做多，等同long-only版本。
    """
    trades = []
    position = None
    cooldown_until = {}  # code -> 最後可再次進場的日期之前都排除

    for date in master_calendar:
        excluded_codes = {c for c, until in cooldown_until.items() if date < until}

        if position is None:
            if is_near_settlement(date, days_before=2):
                continue  # 結算日前1-2天不開新倉

            regime = compute_regime(index_df, date)
            candidates = scan_mean_reversion_candidates(
                price_data, universe, date, regime, excluded_codes, allow_short=allow_short,
                rsi_long_threshold=rsi_long_threshold, rsi_short_threshold=rsi_short_threshold,
            )
            if candidates:
                position = try_enter_mean_reversion(price_data, candidates, date, starting_capital, lots)
                if position is not None:
                    df = price_data[position["code"]]
                    row = df.loc[date]
                    position = _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until)
            continue

        df = price_data.get(position["code"])
        if df is None or date not in df.index:
            continue

        if date != position["entry_date"]:
            position["hold_days"] += 1

        row = df.loc[date]
        position = _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until)

    return trades


def _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until):
    event, exit_price = check_exit(row, position)

    if event == "stop":
        _close_mr_trade(position, exit_price, "stop", date, trades)
        cooldown_until[position["code"]] = date + pd.Timedelta(days=STOP_LOSS_COOLDOWN_DAYS * 2)
        # *2 是粗略把交易日轉近似日曆天數，避免跨假日冷卻期被縮短；回測用途足夠精確
        return None

    if event == "target":
        _close_mr_trade(position, exit_price, "target", date, trades)
        return None

    if position["hold_days"] >= max_hold_days:
        _close_mr_trade(position, row["Close"], "forced_close", date, trades)
        return None

    return position


def _close_mr_trade(position, exit_price, reason, date, trades):
    code = position["code"]
    e_price = position["e_price"]
    side = position["side"]
    lots = position["lots"]
    mult = get_contract_multiplier(code, e_price)

    if side == "long":
        price_pnl = (exit_price - e_price) * mult * lots
    else:
        price_pnl = (e_price - exit_price) * mult * lots

    commission = COMMISSION_PER_LOT_PER_LEG * lots * 2  # 進場+出場各一次
    pnl_ntd = price_pnl - commission
    return_pct = pnl_ntd / (e_price * mult * lots)

    trades.append({
        "code": code, "side": side,
        "entry_date": position["entry_date"], "exit_date": date,
        "e_price": e_price, "exit_price": exit_price, "exit_reason": reason,
        "lots": lots, "pnl_ntd": pnl_ntd, "return_pct": return_pct,
        "hold_days": position["hold_days"],
    })


def summarize_mr(trades: list, starting_capital: float) -> dict:
    if not trades:
        return {
            "trade_count": 0, "win_rate": 0.0, "avg_return_pct": 0.0,
            "total_pnl_ntd": 0.0, "max_drawdown_ntd": 0.0, "ending_equity_ntd": starting_capital,
            "long_count": 0, "short_count": 0,
        }

    pnl_list = [t["pnl_ntd"] for t in trades]
    returns = [t["return_pct"] for t in trades]
    wins = [p for p in pnl_list if p > 0]

    equity = [starting_capital]
    for p in pnl_list:
        equity.append(equity[-1] + p)
    equity = np.array(equity)
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max

    return {
        "trade_count": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_return_pct": float(np.mean(returns)) * 100,
        "total_pnl_ntd": float(sum(pnl_list)),
        "max_drawdown_ntd": float(dd.min()),
        "ending_equity_ntd": float(equity[-1]),
        "long_count": sum(1 for t in trades if t["side"] == "long"),
        "short_count": sum(1 for t in trades if t["side"] == "short"),
    }
