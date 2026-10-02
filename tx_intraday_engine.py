"""
台指期(TX)15分鐘K棒 右側突破雙向策略回測引擎 —— 單一商品版本。

跟 momentum_breakout_engine.py 同樣的「右側/順勢突破」哲學(有結構性突破才算候選，
候選裡用加權訊號分數決定進場強度，可多可空)，但這裡**不是**照抄那個檔案，而是
針對「只有一檔商品(TX期貨本身)、沒有其他標的可以互相比較」這個根本差異，重新設計
了排名機制——見下方「橫斷面排名 vs 自身歷史滾動排名」說明。

⚠️⚠️ 樣本量極小的誠實聲明(這是這個模組存在的前提，不是事後補充的免責聲明)：
資料來源taifex_intraday_loader.py只提供TAIFEX免費的「前30個交易日」逐筆成交資料，
換算成15分鐘K棒、只算日盤(08:45-13:45，每天5小時=20根15分鐘K棒)，整個可用樣本
大約是 20~22個交易日 x 20根/天 ≈ 400~450根K棒。這個引擎沿用momentum_breakout_engine.py/
compare_breakout.py同一套「單一訊號拆解→自動篩選PF>1的訊號→組合成最終策略」方法論，
但**在這麼小的樣本上跑這整套方法論，本質上就是在對雜訊做多重比較**——每個訊號單獨
測出來的PF有很高機率只是這400多根K棒剛好某個訊號方向蒙對，不是真的有預測力，
compare_tx_intraday.py的每一個輸出區塊都會重複標示這一點，不是只在檔案開頭寫一次
就算了事。使用者已經明確要求「照原計畫跑全套單一訊號+自動組合，但報告裡強調樣本
太小不能信」，這裡是在誠實揭露前提下的刻意選擇，不是誤判風險。

橫斷面排名 vs 自身歷史滾動排名(這個模組跟momentum_breakout_engine.py最根本的設計差異)：
momentum_breakout_engine.py的percentile_score()是「今天這檔股票的量比，在今天所有
候選股票裡排第幾百分位」——這需要同一天有很多檔股票可以互相比較(cross-sectional)。
這裡只有TX一檔商品，沒有「同一根K棒裡其他商品」可以比較，所以改用
rolling_percentile_score()：「這根K棒的MACD柱狀圖值，在這個商品自己過去
ROLLING_PCT_WINDOW_BARS根K棒的歷史分佈裡排第幾百分位」——是同一個「相對排名、
不是絕對值判定」的精神，只是比較的對象從「同一時間點的其他商品」換成「同一個商品
的過去自己」，這是文獻上常見的cross-sectional relative scoring適應成單一商品
time series的標準做法(rolling percentile/rank transform)，不是這裡發明的新技巧。
每個時間點t的分數只用[t-window+1, t]這個窗口內(含自己，不含未來)的歷史值計算，
不會用到t之後的任何資料，避免look-ahead bias(見rolling_percentile_score()測試)。

參數選擇(以下全部是誠實承認的判斷值，不是實測調過的最佳參數，跟squeeze_kdj_signal.py
的SQUEEZE_LOOKBACK_DEFAULT同樣精神——「合理但未經實測調整的預設值」)：
  ROLLING_PCT_WINDOW_BARS_DEFAULT = 60(約3個交易日，20根/天 x 3天)：
    原本直覺會選「5個交易日=100根」(比照使用者原本100根的估計)，但整個可用樣本
    只有約400~450根，暖身期(rolling window不足時分數固定給0分、不會產生候選)
    用掉100根等於先燒掉將近1/4的本來就很少的樣本，在「樣本已經太小」這個前提下
    再把暖身期拉得更長並不合理，所以退而求其次選3個交易日(60根)，仍然是「跟自己
    過去一小段時間比」的精神，只是暖身期短一點，這是在「窗口要夠長才有統計意義」
    跟「窗口太長會吃掉本來就稀缺的可用樣本」兩者之間的權衡，沒有標準答案。
  BREAKOUT_WINDOW_BARS_DEFAULT = 20(約1個交易日)：
    結構性突破門檻用「前20根K棒(前一個完整交易日)的收盤新高/新低」，對應
    momentum_breakout_engine.py日線版本「20日創新高」在日內尺度上的類比——
    日線版本的20天大約是1個月，這裡選「前一個完整交易日」當突破基準，是等比例
    縮放到日內頻率的判斷，不是從資料裡估出來的。
  ATR_PERIOD_BARS_DEFAULT = 14：
    沿用這個repo其他引擎(日線)一貫的ATR(14)預設值，只是這裡的14代表14根K棒
    (約0.7個交易日)而不是14天，是「沿用既有預設值的數字，重新詮釋成bar-based」，
    不是針對15分鐘K棒重新估計過的參數。
  CANDLE_LOOKBACK_BARS_DEFAULT = 3、DIVERGENCE_LOOKBACK_BARS_DEFAULT = 20：
    同上，數字直接沿用momentum_breakout_engine.py的日線預設值，改用bars為單位。
  VOLUME_RATIO_WINDOW_BARS_DEFAULT = 20(約1個交易日)：跟BREAKOUT_WINDOW_BARS_DEFAULT
    用同一個窗口，「今天這根K棒的量，相對前一個完整交易日的均量」。
  SCORE_ENTRY_THRESHOLD_DEFAULT = 50.0：
    因為只有一檔商品，沒有其他候選可以「排名前N名」互相淘汰(momentum_breakout_engine.py
    的_rank_candidates()在多商品情境下用top_n篩選候選)，這裡改用「加權分數要贏過
    自己過去歷史的中位數(rolling_percentile_score的5~100分尺度中間值)」當門檻，
    是「結構性突破門檻決定候選資格，訊號分數決定進場強度」兩階段架構在單一商品下
    的類比實作——分數決定的不是「贏過誰」，是「這次的訊號支持強度算不算贏過這個
    商品自己過去的一般水準」，一樣是未經實測調整的判斷值。
  MAX_HOLD_BARS_DEFAULT = 80(約4個交易日)：
    移動停利模式下的工程安全上限(不是真正的出場依據，見momentum_breakout_engine.py
    run_momentum_breakout_backtest()同樣的說明)，80根約占整個30天免費樣本視窗的
    20%，避免單一部位吃掉太大比例的可用K棒數，是保守選擇，不是估出來的最佳值。

訊號清單(TX_INTRADAY_SIGNAL_NAMES，共6個，全部從純OHLCV算出來，不需要籌碼資料——
台指期貨沒有個股籌碼概念，所以momentum_breakout_engine.py裡跟外資/投信買賣超相關
的訊號在這裡完全不適用，不是遺漏)：
  score_macd             MACD柱狀圖，rolling_percentile_score(higher_is_better依方向)
  score_candle_body       K棒實體比例，重用momentum_breakout_engine.py同款計算邏輯
  score_rsi_cross         RSI偏離50的程度，重用compute_rsi()(mean_reversion_engine)
  score_volume_ratio      量比(當根量/前一交易日均量)，多空共用同一個分數(爆量不分方向)
  score_macd_divergence   MACD背離(近似實作，邏輯完全重用momentum_breakout_engine.py
                          precompute_breakout_indicators()裡的定義，只是lookback
                          單位從天換成根)
  score_squeeze_kdj       布林+Keltner擠壓+KDJ關鍵K棒(重用squeeze_kdj_signal.py的
                          compute_squeeze_kdj_features()/EntryFlag，不重新實作)——
                          這個訊號只支援多方(見squeeze_kdj_signal.py模組docstring
                          「本模組只實作多方」的說明)，空方候選這個訊號固定給0分
                          (中性，不影響空方排名)，EntryFlag是0/1旗標，這裡直接映射
                          成0分或100分(跟其他0~100分尺度的訊號對齊)，不是
                          rolling_percentile_score的輸出(狀態機旗標本身已經是
                          「有沒有觸發」的離散事件，不需要再做歷史排名)。

出場：ATR固定停損 + 可選ATR移動停利，直接重用 mean_reversion_engine.check_exit()/
update_trailing_stop()——這兩個函式只依賴row(Open/High/Low/Close)跟position dict
的欄位(side/stop_price/target_price/trailing_atr_mult/atr_entry/trailing_anchor)，
完全不假設「一天一根K棒」，所以可以原封不動套用在15分鐘K棒上，不需要另外實作一套
bar-based版本(這是刻意的重用選擇，不是偷懶——這個repo一貫的原則是「出場判定這種跟
K棒頻率無關的機制，直接重用，不要每個引擎各自重寫一份」，見momentum_breakout_engine.py
模組docstring開頭的說明)。

部位大小/契約乘數/成本假設：直接從squeeze_kdj_tx_hourly_preview.py匯入
CONTRACT_MULTIPLIER/DEFAULT_COMMISSION_PER_SIDE/DEFAULT_EXCHANGE_FEE_PER_SIDE，
不重新定義新數字，讓這兩支TX相關腳本的假設保持一致(見該檔案模組docstring的
契約規格/成本假設誠實聲明)。
"""
import numpy as np
import pandas as pd

from mean_reversion_engine import compute_rsi, compute_atr_correct, check_exit, update_trailing_stop
from momentum_breakout_engine import compute_macd
from squeeze_kdj_signal import compute_squeeze_kdj_features

ROLLING_PCT_WINDOW_BARS_DEFAULT = 60
BREAKOUT_WINDOW_BARS_DEFAULT = 20
ATR_PERIOD_BARS_DEFAULT = 14
CANDLE_LOOKBACK_BARS_DEFAULT = 3
DIVERGENCE_LOOKBACK_BARS_DEFAULT = 20
VOLUME_RATIO_WINDOW_BARS_DEFAULT = 20
SCORE_ENTRY_THRESHOLD_DEFAULT = 50.0
MAX_HOLD_BARS_DEFAULT = 80

TX_INTRADAY_SIGNAL_NAMES = [
    "score_macd", "score_candle_body", "score_rsi_cross", "score_volume_ratio",
    "score_macd_divergence", "score_squeeze_kdj",
]


def rolling_percentile_score(series: pd.Series, window: int = ROLLING_PCT_WINDOW_BARS_DEFAULT,
                              higher_is_better: bool = True, min_periods: int = None) -> pd.Series:
    """
    單一商品版本的percentile_score()：不是跟「同一時間點的其他商品」比，是跟
    「這個商品自己過去window根K棒的歷史分佈」比，回傳同樣0~100分尺度的分數
    (5~100，用法/意義完全比照overnight_momentum_engine.percentile_score()，
    只是比較對象換成自身歷史，見模組docstring「橫斷面排名 vs 自身歷史滾動排名」)。

    每個時間點t的分數只用pandas rolling()預設的向後看窗口[t-window+1, t](含t本身，
    不含t之後任何一筆)計算，天生不會看到未來資料——這是用pandas內建rolling機制
    保證的，不是額外加的防護(見test_tx_intraday_engine.py的no-lookahead測試，
    直接驗證「改動t之後的值不影響t當下的分數」)。

    min_periods：窗口內有效(非NaN)樣本數不到這個門檻時，分數給0分(暖身期/資料不足，
    視為「還沒有足夠歷史可以比較，不算特別強或特別弱」，不是NaN，避免下游判斷需要
    額外處理NaN)。預設 max(10, window // 3)，是「至少要有窗口1/3、且不少於10筆」
    的務實折衷，同樣是判斷值，不是實測調過的參數。
    """
    if min_periods is None:
        min_periods = max(10, window // 3)
    s = series.astype(float)

    def _pct_rank(arr: np.ndarray) -> float:
        cur = arr[-1]
        if np.isnan(cur):
            return 0.0
        valid = arr[~np.isnan(arr)]
        if len(valid) < min_periods:
            return 0.0
        if higher_is_better:
            rank = float(np.sum(valid <= cur))
        else:
            rank = float(np.sum(valid >= cur))
        pct = rank / len(valid)
        return 5.0 + pct * 95.0

    scores = s.rolling(window, min_periods=1).apply(_pct_rank, raw=True)
    return scores.fillna(0.0)


def precompute_tx_intraday_indicators(df: pd.DataFrame,
                                       rolling_window_bars: int = ROLLING_PCT_WINDOW_BARS_DEFAULT,
                                       breakout_window_bars: int = BREAKOUT_WINDOW_BARS_DEFAULT,
                                       atr_period_bars: int = ATR_PERIOD_BARS_DEFAULT,
                                       candle_lookback_bars: int = CANDLE_LOOKBACK_BARS_DEFAULT,
                                       divergence_lookback_bars: int = DIVERGENCE_LOOKBACK_BARS_DEFAULT,
                                       volume_ratio_window_bars: int = VOLUME_RATIO_WINDOW_BARS_DEFAULT) -> pd.DataFrame:
    """
    對TX的15分鐘(或任意K棒週期，函式本身不假設頻率)OHLCV序列，一次算好右側突破引擎
    需要的全部指標跟訊號分數，之後day-by-day(這裡是bar-by-bar)迴圈只查表，
    同一精神見momentum_breakout_engine.precompute_breakout_indicators()。
    df必須有Open/High/Low/Close欄位；Volume缺席時量比訊號會整段是0分(不影響其他訊號，
    因為15分鐘K棒的Volume是taifex_intraday_loader.py聚合時就會提供的欄位，這裡的
    容錯只是為了讓測試可以用不含Volume的合成資料練其他訊號)。
    """
    close, open_, high, low = df["Close"], df["Open"], df["High"], df["Low"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    atr = compute_atr_correct(df, period=atr_period_bars)
    rsi = compute_rsi(close)
    macd_line, _, macd_hist = compute_macd(close)

    # RollingHigh/RollingLow用shift(1)：不能把「這根K棒自己」算進前高前低裡，
    # 否則突破判斷會變成恆真，跟momentum_breakout_engine.py同樣的處理方式。
    rolling_high = close.rolling(breakout_window_bars).max().shift(1)
    rolling_low = close.rolling(breakout_window_bars).min().shift(1)

    vol_avg = volume.rolling(volume_ratio_window_bars).mean()
    volume_ratio = volume / vol_avg.replace(0, np.nan)

    candle_range = (high - low).replace(0, np.nan)
    body_ratio = (close - open_).abs() / candle_range
    bullish_body = body_ratio.where(close > open_, 0.0)
    bearish_body = body_ratio.where(close < open_, 0.0)
    bullish_candle_raw = bullish_body.rolling(candle_lookback_bars).mean()
    bearish_candle_raw = bearish_body.rolling(candle_lookback_bars).mean()

    # MACD背離(近似實作)：邏輯完全重用momentum_breakout_engine.precompute_breakout_indicators()
    # 的定義，見該函式docstring對這個近似做法侷限性的完整說明，這裡不重複。
    price_change_n = close.pct_change(divergence_lookback_bars)
    macd_change_n = macd_line.diff(divergence_lookback_bars)
    bullish_divergence_raw = (-price_change_n).clip(lower=0) * macd_change_n.clip(lower=0)
    bearish_divergence_raw = price_change_n.clip(lower=0) * (-macd_change_n).clip(lower=0)

    squeeze_kdj_features = compute_squeeze_kdj_features(df)
    squeeze_kdj_entry_long = squeeze_kdj_features["EntryFlag"].astype(float)

    rsi_signed = rsi - 50.0

    result = pd.DataFrame({
        "Close": close, "Open": open_, "High": high, "Low": low,
        "ATR": atr, "RollingHigh": rolling_high, "RollingLow": rolling_low,
        "VolumeRatio": volume_ratio,
        "ScoreMACDLong": rolling_percentile_score(macd_hist, rolling_window_bars, higher_is_better=True),
        "ScoreMACDShort": rolling_percentile_score(macd_hist, rolling_window_bars, higher_is_better=False),
        "ScoreRSILong": rolling_percentile_score(rsi_signed, rolling_window_bars, higher_is_better=True),
        "ScoreRSIShort": rolling_percentile_score(rsi_signed, rolling_window_bars, higher_is_better=False),
        "ScoreVolumeRatio": rolling_percentile_score(volume_ratio, rolling_window_bars, higher_is_better=True),
        "ScoreCandleLong": rolling_percentile_score(bullish_candle_raw, rolling_window_bars, higher_is_better=True),
        "ScoreCandleShort": rolling_percentile_score(bearish_candle_raw, rolling_window_bars, higher_is_better=True),
        "ScoreDivergenceLong": rolling_percentile_score(bullish_divergence_raw, rolling_window_bars, higher_is_better=True),
        "ScoreDivergenceShort": rolling_percentile_score(bearish_divergence_raw, rolling_window_bars, higher_is_better=True),
        "SqueezeKDJEntryLong": squeeze_kdj_entry_long,
    }, index=df.index)
    return result


def _weighted_score(row: pd.Series, side: str, signal_weights: dict, total_weight: float) -> float:
    parts = {
        "score_macd": row["ScoreMACDLong"] if side == "long" else row["ScoreMACDShort"],
        "score_candle_body": row["ScoreCandleLong"] if side == "long" else row["ScoreCandleShort"],
        "score_rsi_cross": row["ScoreRSILong"] if side == "long" else row["ScoreRSIShort"],
        "score_volume_ratio": row["ScoreVolumeRatio"],
        "score_macd_divergence": row["ScoreDivergenceLong"] if side == "long" else row["ScoreDivergenceShort"],
        # score_squeeze_kdj只支援多方(見模組docstring)，空方固定中性0分，不拖累/不幫助空方排名。
        "score_squeeze_kdj": (row["SqueezeKDJEntryLong"] * 100.0) if side == "long" else 0.0,
    }
    return sum(parts[name] * signal_weights.get(name, 0.0) for name in TX_INTRADAY_SIGNAL_NAMES) / total_weight


def scan_tx_intraday_candidate(indicators: pd.DataFrame, pos: int, signal_weights: dict = None,
                                allow_short: bool = True,
                                score_entry_threshold: float = SCORE_ENTRY_THRESHOLD_DEFAULT):
    """
    單一K棒(indicators第pos列，代表這根K棒收盤後的狀態)的兩階段判定：
    1) 結構性門檻(必要條件)：收盤突破前breakout_window_bars根的高/低點才算候選，
       跟momentum_breakout_engine.py的require_hard_breakout=True(硬突破門檻)同精神。
    2) 訊號分數(在通過門檻的方向上，加權分數要贏過score_entry_threshold才真的進場)——
       單一商品沒有「候選池排名」，這個門檻是排名機制在單一商品情境下的類比，
       見模組docstring SCORE_ENTRY_THRESHOLD_DEFAULT說明。
    多空互斥(close不可能同時大於前高又小於前低)，最多回傳一個候選(dict)或None。
    """
    if pos < 0 or pos >= len(indicators):
        return None
    row = indicators.iloc[pos]
    atr, rolling_high, rolling_low, close = row["ATR"], row["RollingHigh"], row["RollingLow"], row["Close"]
    if pd.isna(atr) or atr <= 0 or pd.isna(rolling_high) or pd.isna(rolling_low):
        return None

    if signal_weights is None:
        signal_weights = {name: 1.0 for name in TX_INTRADAY_SIGNAL_NAMES}
    total_weight = sum(signal_weights.get(name, 0.0) for name in TX_INTRADAY_SIGNAL_NAMES)
    if total_weight <= 0:
        total_weight = 1.0

    if close > rolling_high:
        score = _weighted_score(row, "long", signal_weights, total_weight)
        if score >= score_entry_threshold:
            return {"side": "long", "score": float(score), "atr": float(atr)}

    if allow_short and close < rolling_low:
        score = _weighted_score(row, "short", signal_weights, total_weight)
        if score >= score_entry_threshold:
            return {"side": "short", "score": float(score), "atr": float(atr)}

    return None


def try_enter_tx_intraday(df: pd.DataFrame, candidate: dict, entry_pos: int,
                           atr_stop_mult: float = 1.0, atr_target_mult: float = 2.0,
                           use_trailing_stop: bool = False, trailing_atr_mult: float = None,
                           lots: int = 1):
    """
    候選在bar(entry_pos-1)收盤後觸發，隔一根K棒(entry_pos)的開盤進場(比照這個repo
    「訊號當根收盤後決定，下一根開盤成交」的一貫慣例)。entry_pos超出資料範圍(候選是
    倒數第二根K棒觸發，已經沒有下一根可以進場)時回傳None，呼叫端(run_tx_intraday_backtest)
    要優雅處理，不能讓迴圈索引出界。
    """
    if entry_pos >= len(df):
        return None
    open_p = float(df["Open"].iloc[entry_pos])
    atr = candidate["atr"]
    side = candidate["side"]

    if side == "long":
        e_price = open_p
        stop_price = e_price - atr_stop_mult * atr
        target_price = None if use_trailing_stop else e_price + atr_target_mult * atr
    else:
        e_price = open_p
        stop_price = e_price + atr_stop_mult * atr
        target_price = None if use_trailing_stop else e_price - atr_target_mult * atr

    position = {
        "side": side, "e_price": e_price, "stop_price": stop_price, "target_price": target_price,
        "hold_days": 1, "lots": lots, "entry_date": df.index[entry_pos],
    }
    if use_trailing_stop:
        position["trailing_stop"] = True
        position["trailing_atr_mult"] = trailing_atr_mult if trailing_atr_mult is not None else atr_stop_mult
        position["atr_entry"] = atr
        position["trailing_anchor"] = e_price
    return position


def _close_tx_trade(position: dict, exit_price: float, exit_reason: str, exit_date,
                     contract_multiplier: int, commission_per_side: float, exchange_fee_per_side: float) -> dict:
    """把出場結果換算成含成本的損益，欄位比照mean_reversion_engine.summarize_mr()
    期待的格式(pnl_ntd/return_pct/hold_days/side)，這樣可以直接重用summarize_mr()
    彙總統計，不用另外寫一套。成本假設(手續費+期交所規費)重用
    squeeze_kdj_tx_hourly_preview.py既有的_cost_trades()同一套算法(進場+出場各收一次，
    乘上口數)，只是這裡在出場當下就地計算，不是先跑完模擬再事後批次套用——
    兩種寫法算出來的數字完全一樣，這裡選在close當下算是因為這個引擎用day-by-day
    (bar-by-bar)迴圈、有停損/停利/強制平倉三種出場路徑，事後批次處理反而要多一層
    轉換。"""
    side = position["side"]
    e_price = position["e_price"]
    lots = position["lots"]
    if side == "long":
        price_pnl = (exit_price - e_price) * contract_multiplier * lots
    else:
        price_pnl = (e_price - exit_price) * contract_multiplier * lots
    round_trip_cost = (commission_per_side + exchange_fee_per_side) * 2 * lots
    pnl_ntd = price_pnl - round_trip_cost
    notional = e_price * contract_multiplier * lots
    return_pct = pnl_ntd / notional if notional > 0 else 0.0
    return {
        "side": side, "entry_date": position.get("entry_date"), "exit_date": exit_date,
        "e_price": e_price, "exit_price": float(exit_price), "exit_reason": exit_reason,
        "lots": lots, "pnl_ntd": pnl_ntd, "return_pct": return_pct,
        "hold_days": position["hold_days"],
    }


def run_tx_intraday_backtest(df: pd.DataFrame, indicators: pd.DataFrame,
                              max_hold_bars: int = MAX_HOLD_BARS_DEFAULT,
                              allow_short: bool = True, lots: int = 1,
                              atr_stop_mult: float = 1.0, atr_target_mult: float = 2.0,
                              use_trailing_stop: bool = False, trailing_atr_mult: float = None,
                              signal_weights: dict = None,
                              score_entry_threshold: float = SCORE_ENTRY_THRESHOLD_DEFAULT,
                              contract_multiplier: int = 200,
                              commission_per_side: float = 60.0, exchange_fee_per_side: float = 20.0) -> list:
    """
    完整bar-by-bar模擬(單一商品，同一時間最多一個部位，不做多部位並行——TX只有一個
    商品可以交易，跟momentum_breakout_engine.py多商品版本max_concurrent_positions
    的概念不適用)。出場判定/移動停利直接重用mean_reversion_engine.check_exit()/
    update_trailing_stop()(見模組docstring)，進場判定用scan_tx_intraday_candidate()。

    索引處理：迴圈在bar i先處理「目前持有部位」在bar i的出場判定；沒有部位時，
    用indicators第i列(這根K棒收盤後的狀態)判斷要不要在下一根(i+1)開盤進場，進場後
    立刻在i+1這根做一次出場檢查(跳空穿越停損可能進場當根就出場)，這個立即檢查完
    之後迴圈主索引直接跳到i+1，避免i+1這根被重複處理兩次(這裡的處理方式跟
    momentum_breakout_engine.run_momentum_breakout_backtest()「進場當天立刻檢查
    一次出場」的邏輯同精神)。
    """
    n = len(df)
    trades = []
    position = None
    i = 0
    while i < n:
        row = df.iloc[i]
        if position is not None:
            event, exit_price = check_exit(row, position)
            if event in ("stop", "stop_gap"):
                trades.append(_close_tx_trade(position, exit_price, event, df.index[i],
                                               contract_multiplier, commission_per_side, exchange_fee_per_side))
                position = None
            elif event == "target":
                trades.append(_close_tx_trade(position, exit_price, "target", df.index[i],
                                               contract_multiplier, commission_per_side, exchange_fee_per_side))
                position = None
            elif position["hold_days"] >= max_hold_bars:
                trades.append(_close_tx_trade(position, float(row["Close"]), "forced_close", df.index[i],
                                               contract_multiplier, commission_per_side, exchange_fee_per_side))
                position = None
            else:
                if position.get("trailing_stop"):
                    update_trailing_stop(position, row)
                position["hold_days"] += 1

        if position is None and i + 1 < n:
            candidate = scan_tx_intraday_candidate(indicators, i, signal_weights, allow_short, score_entry_threshold)
            if candidate is not None:
                position = try_enter_tx_intraday(
                    df, candidate, i + 1, atr_stop_mult=atr_stop_mult, atr_target_mult=atr_target_mult,
                    use_trailing_stop=use_trailing_stop, trailing_atr_mult=trailing_atr_mult, lots=lots,
                )
                if position is not None:
                    entry_row = df.iloc[i + 1]
                    event, exit_price = check_exit(entry_row, position)
                    if event in ("stop", "stop_gap"):
                        trades.append(_close_tx_trade(position, exit_price, event, df.index[i + 1],
                                                        contract_multiplier, commission_per_side, exchange_fee_per_side))
                        position = None
                    elif event == "target":
                        trades.append(_close_tx_trade(position, exit_price, "target", df.index[i + 1],
                                                        contract_multiplier, commission_per_side, exchange_fee_per_side))
                        position = None
                    i += 1  # 這根(i+1)已經當進場+出場檢查處理過，主迴圈索引跳過去，避免重複處理

        i += 1

    if position is not None:
        # 資料結束時還有未平倉部位，用最後一根收盤價強制平倉，避免這筆交易憑空消失
        # (跟mean_reversion_engine/momentum_breakout_engine系列引擎的慣例一致，只是
        # 那些引擎是靠master_calendar走完，這裡靠資料本身走完)。
        trades.append(_close_tx_trade(position, float(df["Close"].iloc[-1]), "forced_close", df.index[-1],
                                       contract_multiplier, commission_per_side, exchange_fee_per_side))

    return trades
