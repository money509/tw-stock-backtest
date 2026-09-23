"""
右側順勢突破雙向策略回測引擎。

跟 mean_reversion_engine.py (均值回歸、逢低承接) 是相反方向的進場邏輯：
只在「真正突破」時才進場(有量、有相對大盤強弱、站上均線)，而不是逢超跌承接。

出場判定、大盤氛圍濾網、結算日過濾、停損冷卻期、保證金上限查表——這些跟「進場方向」
無關的機制，直接從 mean_reversion_engine 匯入重用，不重複實作，避免邏輯分岔。

選股邏輯分成兩層：
1. 結構性門檻(必要條件，不計分，只決定「算不算候選」)：
   基本門檻(永遠套用)：
     多方 = 站上20日均線 + 突破前N日收盤新高
     空方 = 跌破20日均線 + 跌破前N日收盤新低
   可選的額外門檻(每個都可以單獨開關，用來做「哪種過濾條件比較有效」的比較)：
     require_above_ma60          站上/跌破季線(60日均線)，過濾「短線站上但大方向還在空頭」的假突破
     require_ma_bullish_alignment 均線多頭/空頭排列(MA5/MA20/MA60完整排列)，比單一均線更嚴謹
     require_dual_institutional_buy 外資+投信買超比重同步同方向(需要有籌碼資料)
     min_volume_ratio            量比門檻(例如>=1.5才算「爆量」，不是隨便超過1倍均量就算)
2. 訊號評分(在通過門檻的候選裡，依加權分數排名，取前top_n名)：
   score_volume_ratio          量比(當日量/20日均量)
   score_rel_strength          相對大盤強弱(個股N日報酬 - 大盤代理N日報酬)
   score_rsi_cross             RSI偏離50的程度(动能剛轉強，不是超買超賣濾網)
   score_macd                  MACD柱狀圖(動能確認指標)
   score_golden_cross          均線黃金交叉/死亡交叉(MA5剛穿越MA20，抓轉折發生的當下)
   score_price_volume_new_high 價量同步創高/創低(比量比更嚴格，要求價跟量同時創極值)
   score_gap_breakout          近期是否出現方向一致的跳空缺口(市場情緒強烈的確認)
   score_candle_body           K棒實體比例(買盤/賣盤主導意願強不強)
   score_foreign_ratio         外資買超金額佔成交金額比重(可選，需要籌碼資料)
   score_trust_ratio           投信買超金額佔成交金額比重(可選，需要籌碼資料)

籌碼相關的欄位(ForeignRatio/TrustRatio)查的都是「嚴格早於進場日」的資料(用
_lookup_prior_row)，不會有隔日衝那邊發現的「偷看當天籌碼」的時間差問題。
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
    "score_volume_ratio", "score_rel_strength", "score_rsi_cross", "score_macd",
    "score_golden_cross", "score_price_volume_new_high", "score_gap_breakout",
    "score_candle_body", "score_foreign_ratio", "score_trust_ratio",
]

# 這幾個訊號需要額外的籌碼資料才能算，沒有籌碼資料時會被自動跳過(不計分，不影響其他訊號)
CHIP_DEPENDENT_SIGNALS = {"score_foreign_ratio", "score_trust_ratio"}


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_institutional_ratio(chip_df: pd.DataFrame, price_df: pd.DataFrame, net_col: str) -> pd.Series:
    """
    把三大法人買賣超(股數)換算成「買超金額 / 當日成交金額」的比重序列，對齊到price_df的日期索引。
    成交金額用 收盤價 x 成交量 估算(近似值，不是交易所公布的真實成交均價金額)。
    沒有籌碼資料或成交金額<=0的日期回傳NaN，不會讓程式崩潰，也不會被誤判成0(0代表「有資料但買賣平衡」)。
    """
    close = price_df["Close"]
    volume = price_df["Volume"]
    turnover_value = (close * volume).replace(0, np.nan)
    net_shares = chip_df[net_col].reindex(price_df.index)
    amount = net_shares * close
    return amount / turnover_value


def precompute_breakout_indicators(df: pd.DataFrame, index_close: pd.Series, chip_df: pd.DataFrame = None,
                                    breakout_lookback: int = 20, rel_strength_window: int = 5,
                                    golden_cross_lookback: int = 3, gap_lookback: int = 5,
                                    gap_threshold: float = 0.02, candle_lookback: int = 3) -> pd.DataFrame:
    """
    針對一檔股票的完整價格序列，一次算好突破策略需要的全部指標(技術面10個訊號用得到的欄位 +
    結構性門檻用得到的欄位)。所有rolling/shift計算在完整序列上一次算好，跟均值回歸引擎的
    precompute_indicators()同樣精神：day-by-day迴圈之後只是查表，不重複計算。
    """
    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    rsi = compute_rsi(close)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    atr = compute_atr_correct(df)

    vol_avg20 = volume.rolling(20).mean()
    volume_ratio = volume / vol_avg20.replace(0, np.nan)

    # rolling_high/low/vol_high 用 shift(1)：不能把「當天自己」算進前高前低裡，
    # 否則突破判斷會變成恆真(股價/成交量永遠等於自己的新高)。
    rolling_high = close.rolling(breakout_lookback).max().shift(1)
    rolling_low = close.rolling(breakout_lookback).min().shift(1)
    rolling_vol_high = volume.rolling(breakout_lookback).max().shift(1)

    _, _, macd_hist = compute_macd(close)

    stock_ret = close.pct_change(rel_strength_window)
    idx_aligned = index_close.reindex(close.index).ffill()
    index_ret = idx_aligned.pct_change(rel_strength_window)
    rel_strength = stock_ret - index_ret

    # 均線黃金/死亡交叉：MA5剛穿越MA20(golden_cross_lookback天之內發生過穿越事件)，
    # 抓「轉折發生的當下」，不是「現在有沒有站上」(那是結構門檻在做的事)。
    # 用「穿越當天」為事件點，再用rolling.max()看事件是否落在最近lookback天窗口內，
    # 比單純比較「今天 vs N天前」更正確：後者只在事件剛好發生在N天前那一刻才抓得到，
    # 事件發生在窗口內其他天數就會漏抓。
    ma5_above = ma5 > ma20
    # shift(1)在第一列會產生NaN，把原本的bool dtype序列變成object dtype，
    # 這時候用 ~ 對object dtype裡的Python bool做的是「位元反轉」(~False=-1、~True=-2)，
    # 不是邏輯反轉，會整組判斷錯掉——用.astype(bool)先轉回正規bool dtype才能正確做「不是」的判斷。
    ma5_above_prev = ma5_above.shift(1).fillna(False).astype(bool)
    cross_up_event = (ma5_above & ~ma5_above_prev).astype(float)
    cross_down_event = ((~ma5_above) & ma5_above_prev).astype(float)
    golden_cross_flag = cross_up_event.rolling(golden_cross_lookback).max().fillna(0.0)
    death_cross_flag = cross_down_event.rolling(golden_cross_lookback).max().fillna(0.0)

    # 價量同步創高/創低：收盤突破前高的同時，成交量也突破前高量(比單純量比更嚴格)
    price_vol_new_high_long = ((close > rolling_high) & (volume > rolling_vol_high)).astype(float)
    price_vol_new_high_short = ((close < rolling_low) & (volume > rolling_vol_high)).astype(float)

    # 跳空缺口：近gap_lookback天內，是否曾出現方向一致、幅度超過門檻的跳空
    prev_close = close.shift(1)
    gap_pct = (open_ - prev_close) / prev_close.replace(0, np.nan)
    gap_up_flag = (gap_pct > gap_threshold).rolling(gap_lookback).max().fillna(0.0)
    gap_down_flag = (gap_pct < -gap_threshold).rolling(gap_lookback).max().fillna(0.0)

    # K棒實體比例：買盤/賣盤主導意願強不強，用近candle_lookback天平均，過濾單日雜訊
    candle_range = (high - low).replace(0, np.nan)
    body_ratio = (close - open_).abs() / candle_range
    bullish_body = body_ratio.where(close > open_, 0.0)
    bearish_body = body_ratio.where(close < open_, 0.0)
    bullish_candle_score = bullish_body.rolling(candle_lookback).mean()
    bearish_candle_score = bearish_body.rolling(candle_lookback).mean()

    result = pd.DataFrame({
        "Close": close, "MA5": ma5, "MA20": ma20, "MA60": ma60, "ATR": atr,
        "VolumeRatio": volume_ratio, "RollingHigh": rolling_high, "RollingLow": rolling_low,
        "RSI": rsi, "MACDHist": macd_hist, "RelStrength": rel_strength,
        "GoldenCrossFlag": golden_cross_flag, "DeathCrossFlag": death_cross_flag,
        "PriceVolNewHighLong": price_vol_new_high_long, "PriceVolNewHighShort": price_vol_new_high_short,
        "GapUpFlag": gap_up_flag, "GapDownFlag": gap_down_flag,
        "BullishCandleScore": bullish_candle_score, "BearishCandleScore": bearish_candle_score,
    }, index=df.index)

    if chip_df is not None:
        result["ForeignRatio"] = compute_institutional_ratio(chip_df, df, "foreign_net")
        result["TrustRatio"] = compute_institutional_ratio(chip_df, df, "trust_net")
    else:
        result["ForeignRatio"] = np.nan
        result["TrustRatio"] = np.nan

    return result


def precompute_all_breakout_indicators(price_data: dict, universe: dict, index_code: str = "2330",
                                        chip_data: dict = None, breakout_lookback: int = 20,
                                        rel_strength_window: int = 5) -> dict:
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
        chip_df = chip_data.get(code) if chip_data is not None else None
        result[code] = precompute_breakout_indicators(
            df, index_close, chip_df=chip_df,
            breakout_lookback=breakout_lookback, rel_strength_window=rel_strength_window,
        )
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

    cross_flag = df["golden_cross_flag"] if long_dir else df["death_cross_flag"]
    df["score_golden_cross"] = percentile_score(cross_flag, higher_is_better=True)

    pv_flag = df["price_vol_new_high_long"] if long_dir else df["price_vol_new_high_short"]
    df["score_price_volume_new_high"] = percentile_score(pv_flag, higher_is_better=True)

    gap_flag = df["gap_up_flag"] if long_dir else df["gap_down_flag"]
    df["score_gap_breakout"] = percentile_score(gap_flag, higher_is_better=True)

    candle_score = df["bullish_candle_score"] if long_dir else df["bearish_candle_score"]
    df["score_candle_body"] = percentile_score(candle_score, higher_is_better=True)

    if df["foreign_ratio"].notna().any():
        df["score_foreign_ratio"] = percentile_score(df["foreign_ratio"], higher_is_better=long_dir)
    else:
        df["score_foreign_ratio"] = 0.0
    if df["trust_ratio"].notna().any():
        df["score_trust_ratio"] = percentile_score(df["trust_ratio"], higher_is_better=long_dir)
    else:
        df["score_trust_ratio"] = 0.0

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
                                       signal_weights: dict = None, top_n: int = 3,
                                       require_above_ma60: bool = False,
                                       require_ma_bullish_alignment: bool = False,
                                       require_dual_institutional_buy: bool = False,
                                       min_volume_ratio: float = 0.0) -> list:
    """
    掃描全市場候選標的：先套結構性突破門檻(基本門檻永遠套用，其餘4個額外門檻可選)，
    通過門檻的候選再依加權分數排名，回傳前top_n名多方候選 + 前top_n名空方候選。
    regime == 'bull' 時停用空方訊號；regime == 'bear' 時停用多方訊號；'neutral' 兩者都放行。
    allow_short=False 時，不管regime是什麼，永遠不產生空方候選。
    signal_weights 為 None 時，10個訊號等權重(沒有籌碼資料的兩個訊號會自動變成0分，不影響排名)。

    require_dual_institutional_buy=True 但沒有籌碼資料時，該檔股票視為沒過門檻(保守排除，
    跟均值回歸引擎「開啟籌碼確認但沒有籌碼資料就排除」的處理方式一致)。
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

        if min_volume_ratio > 0:
            vr = row["VolumeRatio"]
            if pd.isna(vr) or vr < min_volume_ratio:
                continue

        base = {
            "code": code, "close": close, "atr": atr,
            "volume_ratio": row["VolumeRatio"], "rel_strength": row["RelStrength"],
            "rsi": row["RSI"], "macd_hist": row["MACDHist"],
            "golden_cross_flag": row["GoldenCrossFlag"], "death_cross_flag": row["DeathCrossFlag"],
            "price_vol_new_high_long": row["PriceVolNewHighLong"],
            "price_vol_new_high_short": row["PriceVolNewHighShort"],
            "gap_up_flag": row["GapUpFlag"], "gap_down_flag": row["GapDownFlag"],
            "bullish_candle_score": row["BullishCandleScore"], "bearish_candle_score": row["BearishCandleScore"],
            "foreign_ratio": row["ForeignRatio"], "trust_ratio": row["TrustRatio"],
        }

        ma5 = row["MA5"]
        ma60 = row["MA60"]

        # 多方
        if regime != "bear" and close > ma20 and close > rolling_high:
            ok = True
            if require_above_ma60:
                ok = ok and not pd.isna(ma60) and close > ma60
            if require_ma_bullish_alignment:
                ok = ok and not pd.isna(ma5) and not pd.isna(ma60) and ma5 > ma20 > ma60
            if require_dual_institutional_buy:
                fr, tr = row["ForeignRatio"], row["TrustRatio"]
                ok = ok and not pd.isna(fr) and not pd.isna(tr) and fr > 0 and tr > 0
            if ok:
                long_rows.append(dict(base))

        # 空方(對稱)
        if allow_short and regime != "bull" and close < ma20 and close < rolling_low:
            ok = True
            if require_above_ma60:
                ok = ok and not pd.isna(ma60) and close < ma60
            if require_ma_bullish_alignment:
                ok = ok and not pd.isna(ma5) and not pd.isna(ma60) and ma5 < ma20 < ma60
            if require_dual_institutional_buy:
                fr, tr = row["ForeignRatio"], row["TrustRatio"]
                ok = ok and not pd.isna(fr) and not pd.isna(tr) and fr < 0 and tr < 0
            if ok:
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
                                    atr_target_mult: float = 2.0, top_n: int = 3,
                                    require_above_ma60: bool = False,
                                    require_ma_bullish_alignment: bool = False,
                                    require_dual_institutional_buy: bool = False,
                                    min_volume_ratio: float = 0.0):
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
                signal_weights=signal_weights, top_n=top_n,
                require_above_ma60=require_above_ma60,
                require_ma_bullish_alignment=require_ma_bullish_alignment,
                require_dual_institutional_buy=require_dual_institutional_buy,
                min_volume_ratio=min_volume_ratio,
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
