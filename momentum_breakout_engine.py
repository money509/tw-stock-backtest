"""
右側順勢突破雙向策略回測引擎。

跟 mean_reversion_engine.py (均值回歸、逢低承接) 是相反方向的進場邏輯：
只在「真正突破」時才進場(有量、有相對大盤強弱、站上均線)，而不是逢超跌承接。

出場判定、大盤氛圍濾網、結算日過濾、停損冷卻期、保證金上限查表——這些跟「進場方向」
無關的機制，直接從 mean_reversion_engine 匯入重用，不重複實作，避免邏輯分岔。

選股邏輯分成兩層：
1. 結構性門檻(必要條件，不計分，只決定「算不算候選」)：
   多方 = 站上20日均線 + 突破前N日收盤新高
   空方 = 跌破20日均線 + 跌破前N日收盤新低
2. 訊號評分(在通過門檻的候選裡，依加權分數排名，取前top_n名)：
   score_volume_ratio  量比(當日量/20日均量)，量能有沒有配合，過濾無量假突破
   score_rel_strength  相對大盤強弱(個股N日報酬 - 大盤代理N日報酬)
   score_rsi_cross     RSI偏離50的程度(动能剛轉強 vs 已經很強，不是超買超賣濾網)
   score_macd          MACD柱狀圖(動能確認指標)
   score_chip_confirm  外資連續買超/賣超天數(可選，需額外提供籌碼資料)

跟均值回歸引擎的籌碼確認一樣，這裡查的都是「嚴格早於進場日」的資料(用
_lookup_prior_row)，不會有隔日衝那邊發現的「偷看當天籌碼/美股」的時間差問題。
"""
import pandas as pd
import numpy as np

from taifex_universe import estimate_margin
from mean_reversion_engine import (
    compute_rsi, compute_atr_correct, is_near_settlement,
    precompute_regime_series, compute_regime, _lookup_prior_row,
    _process_mr_day, summarize_mr,
    DEFAULT_MARGIN_CAP_RATIO,
)
from overnight_momentum_engine import percentile_score

BREAKOUT_SIGNAL_NAMES = [
    "score_volume_ratio", "score_rel_strength", "score_rsi_cross",
    "score_macd", "score_chip_confirm",
]


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def precompute_breakout_indicators(df: pd.DataFrame, index_close: pd.Series,
                                    breakout_lookback: int = 20, rel_strength_window: int = 5) -> pd.DataFrame:
    """
    針對一檔股票的完整價格序列，一次算好突破策略需要的全部指標。
    rolling_high/rolling_low 用 .shift(1)，確保「前N日新高/新低」不包含當天自己，
    否則當天收盤創新高這件事本身會讓rolling_high等於今天的收盤價，變成恆真的假突破判斷。
    """
    close = df["Close"]
    volume = df["Volume"]

    rsi = compute_rsi(close)
    ma20 = close.rolling(20).mean()
    atr = compute_atr_correct(df)

    vol_avg20 = volume.rolling(20).mean()
    volume_ratio = volume / vol_avg20.replace(0, np.nan)

    rolling_high = close.rolling(breakout_lookback).max().shift(1)
    rolling_low = close.rolling(breakout_lookback).min().shift(1)

    _, _, macd_hist = compute_macd(close)

    stock_ret = close.pct_change(rel_strength_window)
    idx_aligned = index_close.reindex(close.index).ffill()
    index_ret = idx_aligned.pct_change(rel_strength_window)
    rel_strength = stock_ret - index_ret

    return pd.DataFrame({
        "Close": close, "MA20": ma20, "ATR": atr,
        "VolumeRatio": volume_ratio, "RollingHigh": rolling_high, "RollingLow": rolling_low,
        "RSI": rsi, "MACDHist": macd_hist, "RelStrength": rel_strength,
    }, index=df.index)


def precompute_all_breakout_indicators(price_data: dict, universe: dict, index_code: str = "2330",
                                        breakout_lookback: int = 20, rel_strength_window: int = 5) -> dict:
    index_df = price_data.get(index_code)
    if index_df is None:
        raise RuntimeError(
            f"大盤氛圍/相對強弱代理指標 {index_code} 沒有成功下載，無法計算相對大盤強弱。"
        )
    index_close = index_df["Close"]

    result = {}
    for code in universe:
        df = price_data.get(code)
        if df is None:
            continue
        result[code] = precompute_breakout_indicators(df, index_close, breakout_lookback, rel_strength_window)
    return result


def _rank_candidates(rows: list, side: str, signal_weights: dict, top_n: int) -> list:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    long_dir = (side == "long")

    df["score_volume_ratio"] = percentile_score(df["volume_ratio"], higher_is_better=True)
    df["score_rel_strength"] = percentile_score(df["rel_strength"], higher_is_better=long_dir)

    rsi_signed = df["rsi"] - 50.0
    df["score_rsi_cross"] = percentile_score(rsi_signed if long_dir else -rsi_signed, higher_is_better=True)
    df["score_macd"] = percentile_score(df["macd_hist"] if long_dir else -df["macd_hist"], higher_is_better=True)

    if df["chip_streak"].notna().any():
        chip_signed = df["chip_streak"] if long_dir else -df["chip_streak"]
        df["score_chip_confirm"] = percentile_score(chip_signed, higher_is_better=True)
    else:
        # 沒有籌碼資料時，這個分量給中性分(不排除候選，只是這個訊號不貢獻排名資訊)
        df["score_chip_confirm"] = 0.0

    total_weight = sum(signal_weights.get(name, 0.0) for name in BREAKOUT_SIGNAL_NAMES)
    if total_weight <= 0:
        total_weight = 1.0
    df["total_score"] = sum(
        df[name] * signal_weights.get(name, 0.0) for name in BREAKOUT_SIGNAL_NAMES
    ) / total_weight

    df = df.sort_values("total_score", ascending=False)
    out = []
    for _, r in df.iterrows():
        out.append({
            "code": r["code"], "side": side, "score": float(r["total_score"]),
            "c_prev": float(r["close"]), "atr": float(r["atr"]),
        })
    return out[:top_n]


def scan_momentum_breakout_candidates(indicators_by_code: dict, as_of_date, regime: str,
                                       excluded_codes: set, allow_short: bool = True,
                                       signal_weights: dict = None,
                                       chip_streak_by_code: dict = None, top_n: int = 3) -> list:
    """
    掃描全市場候選標的：先套結構性突破門檻(必要條件)，通過門檻的候選再依加權分數排名，
    回傳前top_n名多方候選 + 前top_n名空方候選。
    regime == 'bull' 時停用空方訊號；regime == 'bear' 時停用多方訊號；'neutral' 兩者都放行。
    allow_short=False 時，不管regime是什麼，永遠不產生空方候選。
    signal_weights 為 None 時，5個訊號等權重。
    """
    if signal_weights is None:
        signal_weights = {name: 1.0 for name in BREAKOUT_SIGNAL_NAMES}

    long_rows, short_rows = [], []

    for code, ind_df in indicators_by_code.items():
        if code in excluded_codes:
            continue
        row, pos = _lookup_prior_row(ind_df, as_of_date)
        if row is None or pos < 60:
            continue

        close = row["Close"]
        ma20 = row["MA20"]
        atr = row["ATR"]
        rolling_high = row["RollingHigh"]
        rolling_low = row["RollingLow"]

        if pd.isna(ma20) or pd.isna(atr) or atr <= 0 or pd.isna(rolling_high) or pd.isna(rolling_low):
            continue

        chip_streak_val = None
        if chip_streak_by_code is not None and code in chip_streak_by_code:
            chip_row, _ = _lookup_prior_row(chip_streak_by_code[code], as_of_date)
            chip_streak_val = chip_row  # precompute_chip_streak() 回傳的是Series，取出的是純量

        base = {
            "code": code, "close": close, "atr": atr,
            "volume_ratio": row["VolumeRatio"], "rel_strength": row["RelStrength"],
            "rsi": row["RSI"], "macd_hist": row["MACDHist"], "chip_streak": chip_streak_val,
        }

        # 多方結構門檻：站上20日均線 + 突破前N日新高
        if regime != "bear" and close > ma20 and close > rolling_high:
            long_rows.append(dict(base))

        # 空方結構門檻(對稱)：跌破20日均線 + 跌破前N日新低
        if allow_short and regime != "bull" and close < ma20 and close < rolling_low:
            short_rows.append(dict(base))

    long_candidates = _rank_candidates(long_rows, "long", signal_weights, top_n)
    short_candidates = _rank_candidates(short_rows, "short", signal_weights, top_n)
    return long_candidates + short_candidates


def try_enter_breakout(price_data: dict, candidates: list, entry_date, starting_capital: float,
                        lots: int = 2, atr_stop_mult: float = 1.0, atr_target_mult: float = 2.0):
    """
    依序檢查候選名單(已跳空風控+保證金上限過濾)，第一個通過的進場。
    跟均值回歸引擎的跳空風控方向一致：不管多空，方向不利的跳空超過0.5%就放棄
    (突破後反向跳空代表隔天開盤可能已經是假突破被打回，不追價)。
    """
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
        if cand["side"] == "long" and gap_pct <= -0.005:
            continue
        if cand["side"] == "short" and gap_pct >= 0.005:
            continue

        margin_needed = estimate_margin(code, open_p, lots)
        if margin_needed > starting_capital * DEFAULT_MARGIN_CAP_RATIO:
            continue

        atr = cand["atr"]
        side = cand["side"]
        e_price = open_p
        if side == "long":
            stop_price = e_price - atr_stop_mult * atr
            target_price = e_price + atr_target_mult * atr
        else:
            stop_price = e_price + atr_stop_mult * atr
            target_price = e_price - atr_target_mult * atr

        return {
            "code": code, "side": side, "entry_date": entry_date,
            "e_price": e_price, "target_price": target_price, "stop_price": stop_price,
            "lots": lots, "hold_days": 1,
        }
    return None


def run_momentum_breakout_backtest(price_data: dict, indicators_by_code: dict, regime_series: pd.DataFrame,
                                    master_calendar: pd.DatetimeIndex, max_hold_days: int,
                                    starting_capital: float, allow_short: bool = True, lots: int = 2,
                                    signal_weights: dict = None, atr_stop_mult: float = 1.0,
                                    atr_target_mult: float = 2.0, chip_streak_by_code: dict = None,
                                    top_n: int = 3):
    """
    完整 day-by-day walk-forward 模擬。出場判定/強制平倉/停損冷卻期，重用
    mean_reversion_engine._process_mr_day()，跟均值回歸引擎共用同一套出場機制，
    差別只在進場端(scan_momentum_breakout_candidates / try_enter_breakout)。
    """
    trades = []
    position = None
    cooldown_until = {}

    for date in master_calendar:
        excluded_codes = {c for c, until in cooldown_until.items() if date < until}

        if position is None:
            if is_near_settlement(date, days_before=2):
                continue

            regime = compute_regime(regime_series, date)
            candidates = scan_momentum_breakout_candidates(
                indicators_by_code, date, regime, excluded_codes, allow_short=allow_short,
                signal_weights=signal_weights, chip_streak_by_code=chip_streak_by_code, top_n=top_n,
            )
            if candidates:
                position = try_enter_breakout(
                    price_data, candidates, date, starting_capital, lots,
                    atr_stop_mult=atr_stop_mult, atr_target_mult=atr_target_mult,
                )
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
