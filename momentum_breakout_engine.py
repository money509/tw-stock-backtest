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
   score_breakout_strength     突破強度：離結構性突破窗口的高/低點還有多少ATR倍數的距離
                               (連續分數，不是二元判定；require_hard_breakout=False時，
                               這是唯一還在間接反映「有沒有突破」的分數，取代原本「過了就算，
                               沒過就不算」的硬門檻，見下方「軟性排名」說明)

軟性排名模式(require_hard_breakout=False，這輪新增)：多輪測試發現「250日創新高」這類
絕對值硬門檻，結果會隨標的池大小/期間劇烈變動、排名不穩定，改成只保留最基本的「站上20日
均線」當結構性資格要求，其餘完全交給訊號評分(相對排名)決定要不要進場，包含上面的
score_breakout_strength——概念是「不追求猜中絕對的突破時機，追求在候選裡相對排名夠前面」，
對標的池組成的變動理論上應該比絕對值門檻更穩定，但這是需要驗證的假設，不是先驗認定的答案，
見compare_breakout.py新增的「軟性排名 vs 硬性突破門檻」比較(階段0.7)。

籌碼相關的欄位(ForeignRatio/TrustRatio)查的都是「嚴格早於進場日」的資料(用
_lookup_prior_row)，不會有隔日衝那邊發現的「偷看當天籌碼」的時間差問題。
"""
import pandas as pd
import numpy as np

from taifex_universe import estimate_margin, get_contract_multiplier
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
    "score_breakout_strength",
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


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ADX(平均趨向指標)，用來過濾「真正有趨勢」跟「盤整區間反覆假突破」的環境——
    只有ADX夠高，代表現在是有方向性的行情，這時候的突破訊號才比較可信；ADX低的時候，
    「創新高」很可能只是盤整區間裡的雜訊。

    這裡用ewm(alpha=1/period)做Wilder平滑的近似實作，不是Wilder原始公式逐日遞迴的
    寫法，但ewm(alpha=1/period, adjust=False)在數學上等價於Wilder平滑的極限行為，
    業界常見的近似做法，跟嚴格照抄教科書遞迴公式的差異在最初幾筆會有些微落差，
    對回測用途影響可以忽略。
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_wilder = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_wilder.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_wilder.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def precompute_breakout_indicators(df: pd.DataFrame, index_close: pd.Series, chip_df: pd.DataFrame = None,
                                    breakout_lookback: int = 20, rel_strength_window: int = 5,
                                    golden_cross_lookback: int = 3, gap_lookback: int = 5,
                                    gap_threshold: float = 0.02, candle_lookback: int = 3,
                                    extra_breakout_windows: tuple = (5, 60, 250),
                                    vol_contraction_short: int = 10, vol_contraction_prior: int = 20) -> pd.DataFrame:
    """
    針對一檔股票的完整價格序列，一次算好突破策略需要的全部指標(技術面10個訊號用得到的欄位 +
    結構性門檻用得到的欄位)。所有rolling/shift計算在完整序列上一次算好，跟均值回歸引擎的
    precompute_indicators()同樣精神：day-by-day迴圈之後只是查表，不重複計算。

    extra_breakout_windows：除了原本的breakout_lookback(預設20日，存在"RollingHigh"/
    "RollingLow"欄位)，額外多算幾組不同窗口的前高/前低("RollingHigh_5"、"RollingHigh_60"、
    "RollingHigh_250"...)，讓scan_momentum_breakout_candidates()可以選擇用哪個窗口當
    「算不算突破」的基本門檻——這是為了回答「20日創新高這個進場時點是不是太晚」的問題：
    5日窗口進場更早、250日(約年線)窗口進場更晚但濾掉更多雜訊，都是同一組股價資料上
    可以直接算出來的候選比較對象，不需要另外抓資料。

    vol_contraction_short/vol_contraction_prior：波動收縮比率用的兩個窗口——
    「近vol_contraction_short天的平均波動度」跟「再往前vol_contraction_prior天的平均波動度」
    的比值，比值明顯小於1代表波動剛收縮完，通常對應盤整、籌碼沉澱的階段，此時如果接著突破，
    比「隨便哪天」的突破更可信(不是每天都在噴出雜訊的股票，剛好那天噴了一根)。
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

    extra_rolling_cols = {}
    for w in extra_breakout_windows:
        extra_rolling_cols[f"RollingHigh_{w}"] = close.rolling(w).max().shift(1)
        extra_rolling_cols[f"RollingLow_{w}"] = close.rolling(w).min().shift(1)

    adx = compute_adx(df)

    # 波動收縮比率：近vol_contraction_short天的平均ATR% ÷ 再往前vol_contraction_prior天
    # 的平均ATR%，明顯小於1代表波動剛收縮完，這時候的突破比隨便哪天都噴量的雜訊更可信。
    atr_pct = compute_atr_correct(df) / close.replace(0, np.nan)
    atr_pct_recent = atr_pct.rolling(vol_contraction_short).mean()
    atr_pct_prior = atr_pct.shift(vol_contraction_short).rolling(vol_contraction_prior).mean()
    vol_contraction_ratio = atr_pct_recent / atr_pct_prior.replace(0, np.nan)

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
        "ADX": adx, "VolContractionRatio": vol_contraction_ratio,
        **extra_rolling_cols,
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

    breakout_strength = df["breakout_strength_long"] if long_dir else df["breakout_strength_short"]
    df["score_breakout_strength"] = percentile_score(breakout_strength, higher_is_better=True)

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
                                       min_volume_ratio: float = 0.0,
                                       ex_dividend_dates_by_code: dict = None,
                                       breakout_window: int = 20, min_adx: float = 0.0,
                                       require_vol_contraction: bool = False,
                                       require_hard_breakout: bool = True) -> list:
    """
    掃描全市場候選標的：先套結構性突破門檻(基本門檻永遠套用，其餘6個額外門檻可選)，
    通過門檻的候選再依加權分數排名，回傳前top_n名多方候選 + 前top_n名空方候選。
    regime == 'bull' 時停用空方訊號；regime == 'bear' 時停用多方訊號；'neutral' 兩者都放行。
    allow_short=False 時，不管regime是什麼，永遠不產生空方候選。
    signal_weights 為 None 時，10個訊號等權重(沒有籌碼資料的兩個訊號會自動變成0分，不影響排名)。

    require_dual_institutional_buy=True 但沒有籌碼資料時，該檔股票視為沒過門檻(保守排除，
    跟均值回歸引擎「開啟籌碼確認但沒有籌碼資料就排除」的處理方式一致)。

    ex_dividend_dates_by_code：可選，來自 dividend_data_loader.load_dividend_events()，
    {code: set(日期字串YYYYMMDD)}。指標查的是「as_of_date之前最後一列」那一天的資料，
    如果那一天剛好是除權息日，股價會因為配股配息機械性跳空，量比/相對大盤強弱/突破前高
    這些訊號會被這個假跳空污染(不是真的動能轉強或轉弱)，所以這裡直接把那天當候選來源的
    股票跳過，不是修正股價，因為修正股價會讓進出場價位跟真實報價脫節。

    breakout_window：算「有沒有突破」用哪個窗口的前高/前低，預設20日(對應"RollingHigh"/
    "RollingLow"欄位)。給5/60/250之類的值時，改讀"RollingHigh_{breakout_window}"/
    "RollingLow_{breakout_window}"(需要precompute_breakout_indicators()的
    extra_breakout_windows有算過這個窗口，預設已包含5/60/250)——用來測試「20日創新高
    這個進場時點是不是太晚/太早」。

    min_adx：ADX(平均趨向指標)門檻，要求進場當下的趨勢強度夠高，過濾掉盤整區間反覆假突破
    的雜訊。0代表不套用這個門檻。

    require_vol_contraction：要求近期波動度明顯低於前一段時期(波動收縮比率<=0.85)才算
    候選，抓「盤整、籌碼沉澱後才發動」的突破，過濾「隨便哪天都在噴量」的雜訊股。

    require_hard_breakout：預設True，維持舊版行為(一定要「收盤價>前N日高/低點」才算候選)。
    設False時改用「軟性排名」模式：只保留最基本的「站上/跌破20日均線」當結構性資格，不再
    要求絕對值意義上的「有沒有突破」，改交給score_breakout_strength(離突破窗口高/低點的
    ATR倍數距離，連續分數)在訊號評分階段自然排出「誰比較接近/超過突破」的相對順序——
    這是回應多輪測試裡「絕對值硬門檻對標的池大小/期間變動極度敏感」的問題，用相對排名
    取代絕對值判定，理論上應該更穩定，但這是待驗證的假設。
    """
    if signal_weights is None:
        signal_weights = {name: 1.0 for name in BREAKOUT_SIGNAL_NAMES}

    high_col = "RollingHigh" if breakout_window == 20 else f"RollingHigh_{breakout_window}"
    low_col = "RollingLow" if breakout_window == 20 else f"RollingLow_{breakout_window}"

    long_rows, short_rows = [], []

    for code, ind_df in indicators_by_code.items():
        if code in excluded_codes:
            continue
        row, pos = _lookup_prior_row(ind_df, as_of_date)
        if row is None or pos < 60:
            continue

        if ex_dividend_dates_by_code is not None and code in ex_dividend_dates_by_code:
            row_date_str = ind_df.index[pos - 1].strftime("%Y%m%d")
            if row_date_str in ex_dividend_dates_by_code[code]:
                continue

        close = row["Close"]
        ma20 = row["MA20"]
        atr = row["ATR"]
        rolling_high = row[high_col]
        rolling_low = row[low_col]

        if pd.isna(ma20) or pd.isna(atr) or atr <= 0 or pd.isna(rolling_high) or pd.isna(rolling_low):
            continue

        if min_volume_ratio > 0:
            vr = row["VolumeRatio"]
            if pd.isna(vr) or vr < min_volume_ratio:
                continue

        if min_adx > 0:
            adx_val = row["ADX"]
            if pd.isna(adx_val) or adx_val < min_adx:
                continue

        if require_vol_contraction:
            vcr = row["VolContractionRatio"]
            if pd.isna(vcr) or vcr > 0.85:
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
            "breakout_strength_long": (close - rolling_high) / atr,
            "breakout_strength_short": (rolling_low - close) / atr,
        }

        ma5 = row["MA5"]
        ma60 = row["MA60"]

        # 多方
        if regime != "bear" and close > ma20 and (not require_hard_breakout or close > rolling_high):
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
        if allow_short and regime != "bull" and close < ma20 and (not require_hard_breakout or close < rolling_low):
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
                        lots: int = 2, atr_stop_mult: float = 1.0, atr_target_mult: float = 2.0,
                        use_trailing_stop: bool = False, trailing_atr_mult: float = None,
                        slippage_pct: float = 0.0, used_margin: float = 0.0,
                        total_margin_cap_ratio: float = None,
                        risk_pct_per_trade: float = None, account_equity: float = None,
                        max_gap_pct: float = None,
                        trailing_activation_days: int = 0, trailing_activation_profit_atr: float = 0.0):
    """
    依序檢查候選名單(已跳空風控+保證金上限過濾)，第一個通過的進場。
    跟均值回歸引擎的跳空風控方向一致：不管多空，方向不利的跳空超過0.5%就放棄
    (突破後反向跳空代表隔天開盤可能已經是假突破被打回，不追價)。

    use_trailing_stop=True 時，這筆倉位改用移動停利出場(見 mean_reversion_engine.
    update_trailing_stop())，不設固定停利目標價(target_price=None)——這才是右側/
    順勢交易「讓獲利奔跑」的精神：進場當下只設一個小的初始停損，出場完全交給
    「趨勢還在不在」決定，不是「碰到某個固定倍數就跑」。trailing_atr_mult 沒指定時，
    預設跟 atr_stop_mult 用同一個值(進場停損距離 = 移動停利距離)。

    slippage_pct：進場成交價位的滑價假設，多方成交價比開盤價再高一點(追價買貴)、
    空方成交價比開盤價再低一點(追價賣便宜)，模擬開盤瞬間流動性最差時的真實成交狀況。
    預設0.0，跟舊版行為完全一致。

    used_margin/total_margin_cap_ratio：多部位並行時，除了原本「單筆保證金不超過
    總資金35%」的個別上限，還要檢查「這筆 + 目前已經在用的保證金」有沒有超過帳戶
    整體的保證金上限，避免同時開太多部位讓槓桿疊加到不合理的程度。total_margin_cap_ratio
    為None時不做這個檢查(維持舊版單一部位的行為，因為單一部位下個別上限本身就等於整體上限)。

    risk_pct_per_trade：給定時改用「風險預算反推口數」，取代固定的lots參數——
    口數 = (account_equity x risk_pct_per_trade) / (初始停損距離 x 合約乘數)，讓每筆
    交易承擔的風險金額大致一致，不會因為標的波動度不同、用同樣口數卻扛完全不同的風險。
    算出來口數 < 1 時直接放棄這個候選(風險預算連1口都不夠，不該硬凹進場放大風險)。
    帳戶權益(account_equity)沒給時退回用starting_capital，讓現有呼叫方式(固定口數)
    完全不受影響。

    max_gap_pct：開盤跳空幅度的「上限」濾網(跟前面的-0.5%下限方向相反)，多單如果
    開盤跳空超過這個正值就放棄進場，用意是避免追在隔日沖主力已經拉高、獲利空間被
    追價盤吃乾抹淨的位置。None(預設)代表不檢查上限，維持舊版行為完全不變。

    trailing_activation_days/trailing_activation_profit_atr：移動停利延遲啟動，
    在進場後的前幾天(或還沒累積到一定倍數的ATR獲利之前)只用進場當下設定的固定
    初始停損防守，不提前啟動移動停利，避免「進場後正常的健康拉回」被貼太緊的
    移動停利提前洗出場、魚身都還沒吃到就出局。兩者都預設0，代表進場當天(第1天)
    就立即啟動移動停利，維持舊版行為完全不變(見mean_reversion_engine._process_mr_day
    的啟動判斷)。
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
        if max_gap_pct is not None:
            if cand["side"] == "long" and gap_pct > max_gap_pct:
                continue
            if cand["side"] == "short" and gap_pct < -max_gap_pct:
                continue

        atr = cand["atr"]
        side = cand["side"]

        lots_to_use = lots
        if risk_pct_per_trade is not None:
            equity = account_equity if account_equity is not None else starting_capital
            mult = get_contract_multiplier(code, open_p)
            stop_distance = atr_stop_mult * atr
            if stop_distance <= 0 or mult <= 0:
                continue
            risk_budget = equity * risk_pct_per_trade
            computed_lots = int(risk_budget // (stop_distance * mult))
            if computed_lots < 1:
                continue
            lots_to_use = computed_lots

        margin_needed = estimate_margin(code, open_p, lots_to_use)
        if margin_needed > starting_capital * DEFAULT_MARGIN_CAP_RATIO:
            continue
        if total_margin_cap_ratio is not None and \
                used_margin + margin_needed > starting_capital * total_margin_cap_ratio:
            continue

        if side == "long":
            e_price = open_p * (1 + slippage_pct)
            stop_price = e_price - atr_stop_mult * atr
            target_price = None if use_trailing_stop else e_price + atr_target_mult * atr
        else:
            e_price = open_p * (1 - slippage_pct)
            stop_price = e_price + atr_stop_mult * atr
            target_price = None if use_trailing_stop else e_price - atr_target_mult * atr

        position = {
            "code": code, "side": side, "entry_date": entry_date,
            "e_price": e_price, "target_price": target_price, "stop_price": stop_price,
            "lots": lots_to_use, "hold_days": 1, "margin_used": margin_needed,
        }
        if use_trailing_stop:
            position["trailing_stop"] = True
            position["trailing_atr_mult"] = trailing_atr_mult if trailing_atr_mult is not None else atr_stop_mult
            position["atr_entry"] = atr
            position["trailing_anchor"] = e_price
            position["trailing_activation_days"] = trailing_activation_days
            position["trailing_activation_profit_atr"] = trailing_activation_profit_atr
        return position
    return None


def run_momentum_breakout_backtest(price_data: dict, indicators_by_code: dict, regime_series: pd.DataFrame,
                                    master_calendar: pd.DatetimeIndex, max_hold_days: int,
                                    starting_capital: float, allow_short: bool = True, lots: int = 2,
                                    signal_weights: dict = None, atr_stop_mult: float = 1.0,
                                    atr_target_mult: float = 2.0, top_n: int = 3,
                                    require_above_ma60: bool = False,
                                    require_ma_bullish_alignment: bool = False,
                                    require_dual_institutional_buy: bool = False,
                                    min_volume_ratio: float = 0.0,
                                    ex_dividend_dates_by_code: dict = None,
                                    use_trailing_stop: bool = False, trailing_atr_mult: float = None,
                                    slippage_pct: float = 0.0, max_concurrent_positions: int = 1,
                                    total_margin_cap_ratio: float = None,
                                    risk_pct_per_trade: float = None,
                                    breakout_window: int = 20, min_adx: float = 0.0,
                                    require_vol_contraction: bool = False,
                                    max_gap_pct: float = None,
                                    trailing_activation_days: int = 0,
                                    trailing_activation_profit_atr: float = 0.0,
                                    require_hard_breakout: bool = True):
    """
    完整 day-by-day walk-forward 模擬。出場判定/強制平倉/停損冷卻期，重用
    mean_reversion_engine._process_mr_day()，跟均值回歸引擎共用同一套出場機制，
    差別只在進場端(scan_momentum_breakout_candidates / try_enter_breakout)。

    use_trailing_stop=True 時，建議 max_hold_days 給一個很大的值(例如250個交易日，
    約一年)當工程上的安全上限，不是真正的出場依據——出場交給移動停利本身，讓真正
    延續的趨勢可以抱得比原本的5天/15天硬上限長很多，這是右側/順勢交易「讓獲利奔跑」
    的核心精神，也是這裡跟均值回歸引擎最大的出場邏輯差異。

    max_concurrent_positions：預設1，維持跟舊版完全一樣的「一次只能持有一個部位」行為
    (原本大盤普遍上漲時，資金大部分時間閒置)。設>1時，允許同時持有多檔不同標的的部位
    (同一檔股票不會同時開兩個部位)，每天先處理所有既有部位的出場/移動停利，再依序
    補進新部位填滿空出來的名額。保證金檢查也跟著擴充：除了單筆不超過總資金35%的個別
    上限，還會檢查「所有持倉的保證金總和」有沒有超過total_margin_cap_ratio(預設None時
    自動抓 min(35% x max_concurrent_positions, 90%)，避免多部位疊加槓桿疊到不合理)。

    risk_pct_per_trade：給定時，每筆交易的口數改用風險預算反推(帳戶權益x risk_pct_per_trade
    ÷ 停損距離)，取代固定的lots，且帳戶權益會隨已實現損益動態調整(複利效果)，而不是
    永遠用starting_capital當基準。

    require_hard_breakout：見scan_momentum_breakout_candidates()說明，預設True維持舊版
    行為；False時改用軟性排名模式(只留MA20結構性資格，突破強度交給score_breakout_strength
    做相對排名)。
    """
    trades = []
    cooldown_until = {}
    open_positions = []  # 每個元素是一個 position dict(已含 margin_used 欄位)

    effective_total_margin_cap_ratio = total_margin_cap_ratio
    if max_concurrent_positions > 1 and effective_total_margin_cap_ratio is None:
        effective_total_margin_cap_ratio = min(DEFAULT_MARGIN_CAP_RATIO * max_concurrent_positions, 0.9)

    for date in master_calendar:
        # 1) 先處理所有既有部位的出場判定/移動停利/強制平倉(可能更新cooldown_until)
        still_open = []
        for position in open_positions:
            df = price_data.get(position["code"])
            if df is None or date not in df.index:
                still_open.append(position)
                continue
            if date != position["entry_date"]:
                position["hold_days"] += 1
            row = df.loc[date]
            updated = _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until,
                                       slippage_pct=slippage_pct)
            if updated is not None:
                still_open.append(updated)
        open_positions = still_open

        # 2) 補進新部位，填滿空出來的名額(max_concurrent_positions=1時，等同舊版的
        #    「position is None才進場」邏輯)。exclude集合要用「今天處理完出場之後」的
        #    cooldown_until狀態，不然今天才剛停損出場的標的，理論上今天就該進冷卻期，
        #    卻因為排除名單是用今天開始前的舊狀態算的而漏掉，變成同一天又立刻進場。
        held_codes = {p["code"] for p in open_positions}
        excluded_codes = {c for c, until in cooldown_until.items() if date < until} | held_codes

        slots_available = max_concurrent_positions - len(open_positions)
        if slots_available > 0 and not is_near_settlement(date, days_before=2):
            regime = compute_regime(regime_series, date)
            used_margin = sum(p["margin_used"] for p in open_positions)
            equity = starting_capital + sum(t["pnl_ntd"] for t in trades)

            candidates = scan_momentum_breakout_candidates(
                indicators_by_code, date, regime, excluded_codes, allow_short=allow_short,
                signal_weights=signal_weights, top_n=max(top_n, slots_available),
                require_above_ma60=require_above_ma60,
                require_ma_bullish_alignment=require_ma_bullish_alignment,
                require_dual_institutional_buy=require_dual_institutional_buy,
                min_volume_ratio=min_volume_ratio,
                ex_dividend_dates_by_code=ex_dividend_dates_by_code,
                breakout_window=breakout_window, min_adx=min_adx,
                require_vol_contraction=require_vol_contraction,
                require_hard_breakout=require_hard_breakout,
            )

            while slots_available > 0 and candidates:
                new_position = try_enter_breakout(
                    price_data, candidates, date, starting_capital, lots,
                    atr_stop_mult=atr_stop_mult, atr_target_mult=atr_target_mult,
                    use_trailing_stop=use_trailing_stop, trailing_atr_mult=trailing_atr_mult,
                    slippage_pct=slippage_pct, used_margin=used_margin,
                    total_margin_cap_ratio=effective_total_margin_cap_ratio,
                    risk_pct_per_trade=risk_pct_per_trade, account_equity=equity,
                    max_gap_pct=max_gap_pct,
                    trailing_activation_days=trailing_activation_days,
                    trailing_activation_profit_atr=trailing_activation_profit_atr,
                )
                if new_position is None:
                    break

                used_margin += new_position["margin_used"]
                slots_available -= 1
                candidates = [c for c in candidates if c["code"] != new_position["code"]]

                # 進場當天立刻檢查一次出場(跟原本單一部位邏輯一致：跳空穿越停損可能當天就出場)
                row = price_data[new_position["code"]].loc[date]
                updated = _process_mr_day(new_position, row, date, trades, max_hold_days, cooldown_until,
                                           slippage_pct=slippage_pct)
                if updated is not None:
                    open_positions.append(updated)
                else:
                    used_margin -= new_position["margin_used"]

    return trades
