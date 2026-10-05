"""
布林通道(BB) + Keltner通道(KC) 擠壓(squeeze) + KDJ 訊號 —— 共用模組。

來源與限制(誠實聲明，不誇大)：
這個訊號的原始構想來自一支介紹「加密貨幣1小時K線」的教學影片(使用者口述翻譯後轉交)，
描述的是 TTM squeeze 的經典變形：布林通道縮進Keltner通道內代表盤整蓄積(擠壓)，
配合KDJ動能指標找進場時機。這裡把它「翻譯」成台股日K線版本，純粹是把同樣的
指標定義套用在日線資料上，並沒有針對台股日線重新驗證過參數是否合理——尤其是
「擠壓後幾天內有效」的回看窗口(SQUEEZE_LOOKBACK)跟「進場後最多等幾天出現關鍵K棒」
(MAX_ARMED_BARS)，兩個都是把「1小時圖」的直覺(以K棒數為單位、不是以時間為單位)
直接套用到日線，屬於合理但未經實測調整的預設值，見本檔案函式docstring個別說明。

本模組只實作多方(long)。空方(做空)在两個既有引擎裡雖然都支援，但這個設定本身
(擠壓後跌破下軌+K<20才算「武裝」，K回升穿越20才觸發)是明顯偏多方抄底反彈的邏輯，
直接鏡射成「漲破上軌+K>80武裝，K回落穿越80觸發」雖然對稱，但原始影片的情境
(布林擠壓後方向不明、抓反轉)本身就沒有明確要求對稱鏡射，加上兩個引擎既有的
單一訊號消融(ablation)測試都是「每個訊號独立試多方/空方各自的候選」，貿然自行
發明一個沒有使用者確認過的空方鏡射規則，比誠實承認「這一版只做多方」風險更高。
之後如果要加空方，建議先跟使用者確認鏡射規則是否合理。

指標定義：
  布林通道(BB)：20日SMA收盤價 ± 2倍20日標準差 (重用 mean_reversion_engine.compute_bollinger)
  Keltner通道(KC)：20日EMA收盤價(basis) ± 1.5倍ATR(20)
    -- Keltner的basis業界常見用EMA(比SMA對新資訊反應快)，這裡沿用這個業界慣例，
       不是這個repo既有程式碼裡已經有的慣例(這個repo的均線多半用SMA)，這裡刻意
       選標準Keltner定義(EMA)而不是為了跟repo風格一致改用SMA，因為Keltner
       Channel的標準定義(包含最早Chester Keltner版本的後繼者、TTM squeeze的
       通用實作)幾乎都用EMA，改用SMA會讓這個指標不再是一般人認得的「Keltner Channel」。
  擠壓(Squeeze)：BB完全收在KC裡面 (BB上軌 < KC上軌 且 BB下軌 > KC上軌) —— 這是
    標準TTM squeeze定義。
  KDJ(只算K，J不用)：
    RSV_t = (close_t - 最近25日最低價) / (最近25日最高價 - 最近25日最低價) * 100
    K_t = (2/3)*K_(t-1) + (1/3)*RSV_t，K_0 = 50 (遞迴平滑，等價於一般教科書常見的
    "1/3平滑" KDJ，不是另一種較少見的"3日SMA-style"平滑版本——兩種業界都有人用，
    這裡選遞迴平滑，因為它是目前主流看盤軟體最常見的預設實作)。
    25日視窗不足時 (min_periods=1) 用已有資料算，早期幾筆RSV精確度較低，但不會整段是NaN。

進場規則(多方，狀態機，逐檔股票各自獨立)：
  1. 「擠壓窗口」：過去 SQUEEZE_LOOKBACK 天內(含當天)曾經出現過擠壓，才算「還在
     擠壓情境附近」，不要求進場當下這一刻仍在擠壓中(因為擠壓解除、行情開始噴出
     本身就是進場訊號的一部分)。
  2. 「武裝(armed)」：在擠壓窗口內，某天收盤跌破布林下軌 且 K<20，設定武裝狀態。
  3. 「關鍵K棒(觸發)」：武裝之後，某天收紅(收盤>前一天收盤) 且 K>20(已經回升)，
     這天就是關鍵K棒，訊號在這天收盤後觸發(比照這兩個引擎既有慣例，隔天開盤進場)。
     觸發後武裝狀態重置(不會同一輪擠壓連續觸發好幾次)。
  4. 為了避免「武裝」之後遙遙無期都等不到關鍵K棒(訊號變得跟原本的擠壓情境已經
     脫鉤)，超過 MAX_ARMED_BARS 天還沒觸發就自動解除武裝狀態，必須重新出現
     條件2才能再次武裝——這是這裡自行加上的、原始描述沒有明講的停損閥門，
     用意是讓訊號保持「跟這一輪擠壓有關」，不是無限期等待。

出場規則，兩個版本(見 compare_breakout.py / compare_equity_swing.py 怎麼各自接到
自己的成本模型/出場框架)：
  變體A「原規則」：停損 = 關鍵K棒「前一根」K棒的最低價(固定價位，不是ATR動態算);
    停利 = 進場後K只要曾經衝到>=80，之後第一次K跌破80的那根K棒觸發停利出場。
    這是這兩個引擎目前都沒有的出場型態(沒有「靠指標值出場」的既有機制)，這裡在
    本模組內自己實作day-by-day處理(見 simulate_variant_a_trades())。
  變體B「沿用ATR框架」：進場規則不變，出場完全交給呼叫端既有的ATR停損/移動停利
    機制(各引擎已經有的try_enter_*/check_exit/update_trailing_stop)，這裡不重新
    實作，只負責告訴呼叫端「這天要不要進場」(EntryFlag)。
"""
import pandas as pd
import numpy as np

from mean_reversion_engine import (
    compute_bollinger, compute_atr_correct, DEFAULT_MARGIN_CAP_RATIO, STOP_LOSS_COOLDOWN_DAYS,
    _process_mr_day, _close_mr_trade,
)
from taifex_universe import estimate_margin

SQUEEZE_LOOKBACK_DEFAULT = 10  # 「還在擠壓情境附近」的回看天數，日線上的假設值，未實測調整
MAX_ARMED_BARS_DEFAULT = 15    # 武裝之後最多等幾天，超過就重置，避免訊號跟擠壓情境脫鉤
OVERSOLD_K_DEFAULT = 20.0
TARGET_K_DEFAULT = 80.0
KDJ_PERIOD_DEFAULT = 25
BB_PERIOD_DEFAULT = 20
BB_STD_DEFAULT = 2.0
KC_PERIOD_DEFAULT = 20
KC_ATR_MULT_DEFAULT = 1.5


def compute_keltner_channel(df: pd.DataFrame, period: int = KC_PERIOD_DEFAULT,
                             atr_mult: float = KC_ATR_MULT_DEFAULT):
    """Keltner通道：EMA(period)收盤價當basis(業界慣例，見模組docstring說明)，
    ± atr_mult 倍 ATR(period，重用 mean_reversion_engine.compute_atr_correct)。"""
    close = df["Close"]
    basis = close.ewm(span=period, adjust=False).mean()
    atr = compute_atr_correct(df, period=period)
    upper = basis + atr_mult * atr
    lower = basis - atr_mult * atr
    return basis, upper, lower


def compute_squeeze_flag(bb_upper: pd.Series, bb_lower: pd.Series,
                          kc_upper: pd.Series, kc_lower: pd.Series) -> pd.Series:
    """標準TTM squeeze定義：布林通道完全收在Keltner通道裡面。
    任一邊界是NaN(暖身期)時視為False(不算擠壓)，不會誤判成True。"""
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    return squeeze.fillna(False)


def compute_kdj_k(df: pd.DataFrame, period: int = KDJ_PERIOD_DEFAULT) -> pd.Series:
    """KDJ的K值，遞迴平滑 K_t = (2/3)K_(t-1) + (1/3)RSV_t，K_0=50。
    只算K(J不用，見模組docstring)。"""
    close, high, low = df["Close"], df["High"], df["Low"]
    lowest_low = low.rolling(period, min_periods=1).min()
    highest_high = high.rolling(period, min_periods=1).max()
    rng = (highest_high - lowest_low)
    rsv = ((close - lowest_low) / rng.replace(0, np.nan) * 100).fillna(50.0)

    k_values = np.empty(len(df))
    k_prev = 50.0
    rsv_arr = rsv.to_numpy()
    for i in range(len(df)):
        k_prev = (2.0 / 3.0) * k_prev + (1.0 / 3.0) * rsv_arr[i]
        k_values[i] = k_prev
    return pd.Series(k_values, index=df.index)


def compute_squeeze_kdj_features(df: pd.DataFrame,
                                  squeeze_lookback: int = SQUEEZE_LOOKBACK_DEFAULT,
                                  oversold_k: float = OVERSOLD_K_DEFAULT,
                                  target_k: float = TARGET_K_DEFAULT,
                                  max_armed_bars: int = MAX_ARMED_BARS_DEFAULT,
                                  bb_period: int = BB_PERIOD_DEFAULT, bb_std: float = BB_STD_DEFAULT,
                                  kc_period: int = KC_PERIOD_DEFAULT,
                                  kc_atr_mult: float = KC_ATR_MULT_DEFAULT,
                                  kdj_period: int = KDJ_PERIOD_DEFAULT) -> pd.DataFrame:
    """
    一次算好一檔股票完整價格序列的擠壓+KDJ相關欄位，包含進場狀態機的結果。
    跟這個repo其他precompute_*函式同樣精神：day-by-day迴圈之後只查表(EntryFlag)，
    狀態機本身用一次性的正向逐列迴圈算好(擠壓的「武裝」狀態有時間先後相依性，
    沒辦法像其他訊號一樣純向量化，這是唯一一個需要逐列迴圈的訊號)。

    回傳欄位：
      Squeeze       這天是否處於擠壓(BB完全收在KC內)
      SqueezeRecent 過去squeeze_lookback天內(含當天)是否曾經擠壓過
      K             KDJ的K值
      EntryFlag     這天是否是「關鍵K棒」(訊號觸發，隔天開盤進場)
      PriorLow      只有EntryFlag=True的列有值：關鍵K棒前一根K棒的最低價
                    (變體A「原規則」停損價用)
    """
    close = df["Close"]
    bb_mid, bb_upper, bb_lower = compute_bollinger(close, period=bb_period, num_std=bb_std)
    _, kc_upper, kc_lower = compute_keltner_channel(df, period=kc_period, atr_mult=kc_atr_mult)
    squeeze = compute_squeeze_flag(bb_upper, bb_lower, kc_upper, kc_lower)
    squeeze_recent = squeeze.rolling(squeeze_lookback, min_periods=1).max().astype(bool)
    k = compute_kdj_k(df, period=kdj_period)

    entry_flag, prior_low = compute_entry_state_machine(
        close=close, low=df["Low"], bb_lower=bb_lower, k=k, squeeze_recent=squeeze_recent,
        oversold_k=oversold_k, max_armed_bars=max_armed_bars,
    )

    return pd.DataFrame({
        "Squeeze": squeeze, "SqueezeRecent": squeeze_recent, "K": k,
        "EntryFlag": entry_flag, "PriorLow": prior_low,
    }, index=df.index)


def compute_entry_state_machine(close: pd.Series, low: pd.Series, bb_lower: pd.Series,
                                 k: pd.Series, squeeze_recent: pd.Series,
                                 oversold_k: float = OVERSOLD_K_DEFAULT,
                                 max_armed_bars: int = MAX_ARMED_BARS_DEFAULT):
    """
    把進場狀態機拆成獨立、純陣列輸入輸出的函式，方便單元測試直接餵一組手算好的
    K/BB下軌/擠壓窗口序列去驗證「武裝→觸發」的邏輯，不用依賴完整的BB/KC/KDJ計算鏈路
    (compute_squeeze_kdj_features內部就是呼叫這個函式)。

    武裝(armed)：squeeze_recent為True 且 close跌破bb_lower 且 k<oversold_k 那天開始。
    觸發(entry)：武裝之後，收紅(close_t > close_(t-1)) 且 k > oversold_k 的第一天，
    觸發後武裝重置；超過max_armed_bars天沒觸發也會重置(見模組docstring)。
    第0天(沒有前一天可比較收盤)永遠不會觸發。

    回傳 (entry_flag: np.ndarray[bool], prior_low: np.ndarray[float])，
    prior_low只有entry_flag=True的位置有值(前一根K棒的最低價)，其餘為NaN。
    """
    close_arr = np.asarray(close, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    bb_lower_arr = np.asarray(bb_lower, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    squeeze_recent_arr = np.asarray(squeeze_recent, dtype=bool)
    n = len(close_arr)

    entry_flag = np.zeros(n, dtype=bool)
    prior_low = np.full(n, np.nan)

    armed = False
    armed_since = -1
    for t in range(1, n):
        if pd.isna(bb_lower_arr[t]) or pd.isna(k_arr[t]):
            continue

        if armed and (t - armed_since) > max_armed_bars:
            armed = False  # 太久沒觸發，跟這一輪擠壓情境已經脫鉤，重置

        if not armed:
            if squeeze_recent_arr[t] and close_arr[t] < bb_lower_arr[t] and k_arr[t] < oversold_k:
                armed = True
                armed_since = t
            continue

        # 已武裝：等待關鍵K棒(收紅 + K回升穿越oversold_k之上)
        if close_arr[t] > close_arr[t - 1] and k_arr[t] > oversold_k:
            entry_flag[t] = True
            prior_low[t] = low_arr[t - 1]
            armed = False

    return entry_flag, prior_low


def simulate_variant_a_trades(df: pd.DataFrame, features: pd.DataFrame,
                               target_k: float = TARGET_K_DEFAULT) -> list:
    """
    變體A「原規則」的day-by-day出場模擬(多方，單一部位、單一股票，不做資金/部位管理，
    純粹用來跟變體B做「同一個進場訊號、不同出場方式」的隔離比較，不是完整的投資組合回測)。

    進場：EntryFlag觸發那天的隔天開盤價進場(比照repo既有慣例，訊號當天收盤後決定，
    隔天開盤成交)。
    停損：features裡記錄的PriorLow(固定價位，一路到出場都不會動)。
    停利：進場後追蹤K，一旦K>=target_k(預設80)過，之後第一次K跌破target_k的那根
    K棒觸發停利，用「跌破那天的收盤價」當成交價(這是K值本身算出來的訊號，不是碰到
    某個價位，沒有對應的盤中價位可用，用收盤價是最貼近「訊號在那天收盤後確認」的
    務實選擇)。
    同一天內若停損、停利理論上都可能發生，跟這個repo mean_reversion_engine.check_exit()
    一樣保守優先判定停損。

    回傳每一筆交易的dict：entry_date/entry_price/exit_date/exit_price/exit_reason
    (stop/target/forced_close)/hold_days。forced_close代表回測資料結束時部位還沒出場，
    用最後一筆收盤價強制平倉，避免這筆交易憑空消失。
    """
    trades = []
    n = len(df)
    dates = df.index
    open_arr = df["Open"].to_numpy()
    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    close_arr = df["Close"].to_numpy()
    entry_flag_arr = features["EntryFlag"].to_numpy()
    prior_low_arr = features["PriorLow"].to_numpy()
    k_arr = features["K"].to_numpy()

    t = 0
    while t < n - 1:
        if not entry_flag_arr[t]:
            t += 1
            continue

        entry_idx = t + 1
        entry_price = open_arr[entry_idx]
        stop_price = prior_low_arr[t]
        entry_date = dates[entry_idx]

        if pd.isna(stop_price) or pd.isna(entry_price):
            t += 1
            continue

        target_armed = False
        exit_idx = None
        exit_price = None
        exit_reason = None
        for u in range(entry_idx, n):
            if low_arr[u] <= stop_price:
                exit_idx, exit_price, exit_reason = u, stop_price, "stop"
                break
            if k_arr[u] >= target_k:
                target_armed = True
            elif target_armed and k_arr[u] < target_k:
                exit_idx, exit_price, exit_reason = u, close_arr[u], "target"
                break
        else:
            pass

        if exit_idx is None:
            exit_idx, exit_price, exit_reason = n - 1, close_arr[n - 1], "forced_close"

        trades.append({
            "entry_date": entry_date, "entry_price": float(entry_price),
            "exit_date": dates[exit_idx], "exit_price": float(exit_price),
            "exit_reason": exit_reason, "hold_days": int(exit_idx - entry_idx + 1),
        })

        t = exit_idx + 1  # 下一輪掃描從出場之後開始，避免同一段期間內重疊進場

    return trades


def simulate_variant_b_trades(df: pd.DataFrame, features: pd.DataFrame,
                               atr_period: int = 20, atr_stop_mult: float = 1.0,
                               atr_target_mult: float = None, max_hold_days: int = None) -> list:
    """
    變體B「沿用ATR框架」的day-by-day出場模擬(多方，單一部位、單一股票，跟
    simulate_variant_a_trades()一樣不做資金/部位管理，純粹隔離比較用)。

    進場規則跟變體A完全相同(見features["EntryFlag"])，差別只在出場：
    停損 = 進場當下ATR的atr_stop_mult倍(固定距離，不像變體A用前一根K棒的低點)；
    atr_target_mult給定時額外設固定停利目標價(進場價+atr_target_mult倍ATR)，
    這是momentum_breakout_engine.py非移動停利模式的預設出場框架；
    max_hold_days給定時，超過這個天數還沒出場就強制平倉，這是equity_swing_engine.py
    用「持有時間到了」當主要出場依據的預設框架。兩個參數哪個engine要用哪個，
    由呼叫端(compare_breakout.py / compare_equity_swing.py)決定，這個函式本身
    兩種模式都支援，好讓兩個引擎的「沿用ATR框架」定義能各自貼近自己既有的做法。

    停損/停利同一天都可能觸及時，優先判定停損(跟mean_reversion_engine.check_exit()
    一樣保守)。
    """
    trades = []
    n = len(df)
    dates = df.index
    open_arr = df["Open"].to_numpy()
    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    close_arr = df["Close"].to_numpy()
    entry_flag_arr = features["EntryFlag"].to_numpy()
    atr_series = compute_atr_correct(df, period=atr_period)
    atr_arr = atr_series.to_numpy()

    t = 0
    while t < n - 1:
        if not entry_flag_arr[t]:
            t += 1
            continue

        entry_idx = t + 1
        entry_price = open_arr[entry_idx]
        atr_at_signal = atr_arr[t]

        if pd.isna(atr_at_signal) or atr_at_signal <= 0 or pd.isna(entry_price):
            t += 1
            continue

        stop_price = entry_price - atr_stop_mult * atr_at_signal
        target_price = entry_price + atr_target_mult * atr_at_signal if atr_target_mult else None

        exit_idx = exit_price = exit_reason = None
        for u in range(entry_idx, n):
            if low_arr[u] <= stop_price:
                exit_idx, exit_price, exit_reason = u, stop_price, "stop"
                break
            if target_price is not None and high_arr[u] >= target_price:
                exit_idx, exit_price, exit_reason = u, target_price, "target"
                break
            if max_hold_days is not None and (u - entry_idx + 1) >= max_hold_days:
                exit_idx, exit_price, exit_reason = u, close_arr[u], "hold_days_reached"
                break

        if exit_idx is None:
            exit_idx, exit_price, exit_reason = n - 1, close_arr[n - 1], "forced_close"

        trades.append({
            "entry_date": dates[entry_idx], "entry_price": float(entry_price),
            "exit_date": dates[exit_idx], "exit_price": float(exit_price),
            "exit_reason": exit_reason, "hold_days": int(exit_idx - entry_idx + 1),
        })

        t = exit_idx + 1

    return trades


# ============================================================================
# 資金/部位受限版本(capital-constrained)——把上面已驗證過的進場訊號接到這個repo
# 「真正會拿去模擬實戰」的資金/部位管理框架(top_n排名 + max_concurrent_positions
# 持倉上限 + 保證金查表)，回答「拿掉『每個訊號都能無限制同時成交』這個樂觀假設之後，
# 實際帳戶能拿到的PF還剩多少」。
#
# 跟上面simulate_variant_a_trades/simulate_variant_b_trades的關係：這裡完全不修改、
# 也不呼叫那兩個函式——那兩個函式是「單一股票、不做資金管理」的隔離測試，跟這裡
# 「全市場逐日競爭有限的持倉名額」是不同的回測迴圈結構(前者是對每檔股票各自獨立
# 用while迴圈往前掃描，後者是對整個市場逐日walk-forward)，沒辦法簡單重用，但
# 「進場規則」「變體A/B的出場規則」這兩件事的定義完全跟上面一致，只是改成在
# day-by-day迴圈裡逐日查表/逐日判斷。
# ============================================================================

MAX_HOLD_DAYS_CAPITAL_CONSTRAINED_DEFAULT = 60
# 原始的simulate_variant_a_trades/simulate_variant_b_trades完全沒有max_hold_days這個
# 安全閥(停損/停利沒觸發就一路等到資料結束，forced_close)，這在「資金無限、每個訊號都能
# 獨立成交」的情境下沒問題——反正不會卡到別的訊號。但這裡要跟其他標的搶有限的
# max_concurrent_positions名額，一旦有一個部位卡住遲遲不出場(例如變體A等不到K衝上80，
# 或變體B兩個方向都沒碰到)，等於永久佔用一個名額，排擠掉後面真正該進場的訊號——
# 所以這裡新增一個這兩個原始模擬函式都沒有的安全閥：60個交易日還沒出場就強制平倉。
# 這是這支函式自己新增的工程判斷，不是原始訊號定義的一部分，60天選得比較寬鬆
# (這個repo其他引擎常見的max_hold_days多半落在5~20天)，用意是盡量不要讓這個安全閥
# 本身干擾到變體A/B原本「等停損/停利條件觸發」的精神，只在真的卡太久時才介入。


def precompute_squeeze_kdj_features_by_code(price_data: dict, universe: dict) -> dict:
    """
    對universe裡每一檔股票各自呼叫一次compute_squeeze_kdj_features()，回傳
    {code: features_df}，供run_squeeze_kdj_capital_constrained_backtest()重複使用。

    跟這個repo其他precompute_*_by_code/precompute_all_indicators()同樣的精神：
    擠壓/KDJ狀態機的計算本身不受「資金夠不夠、要不要排隊搶名額」這些資金管理層面的
    參數影響，應該只算一次、在IS/OOS兩次回測之間共用，不要每次呼叫backtest函式都
    重新算一次全市場的BB/KC/KDJ。

    len(df) < 60 的股票直接跳過，不計算(資料太短，連暖身期都撐不過，跟
    run_squeeze_kdj_exit_style_comparison()既有的篩選門檻一致)。
    """
    result = {}
    for code in universe:
        df = price_data.get(code)
        if df is None or len(df) < 60:
            continue
        result[code] = compute_squeeze_kdj_features(df)
    return result


def _process_squeeze_kdj_variant_a_day(position: dict, row, k_today: float, date, trades: list,
                                        max_hold_days: int, cooldown_until: dict,
                                        slippage_pct: float = 0.0):
    """
    變體A「原規則」出場邏輯的day-by-day處理，專門給資金受限版的逐日迴圈用。

    為什麼不能直接重用mean_reversion_engine._process_mr_day()：變體A的停利條件
    (「K曾經衝到>=80，之後第一次跌破80」)需要「這個部位進場後K有沒有達到過80」這個
    跨日的狀態(target_armed)，這不是check_exit()看得懂的「碰到固定target_price」
    判定方式，check_exit()完全不知道K值這件事——所以這裡寫一個新函式，但刻意盡量
    模仿_process_mr_day()/check_exit()既有的慣例，讓兩套出場函式的「形狀」一致、
    好對照、好維護：
      - 停損優先於停利(同一天兩者理論上都可能發生時，保守先判定停損)，跟check_exit()
        同一個保守假設。
      - 跳空穿越停損(今天開盤價本身已經低於停損價)用開盤價出場(stop_gap)，不是
        假裝用停損價精準出場，跟check_exit()的'stop_gap'事件同一個精神——這是比
        simulate_variant_a_trades()原本只有單一種"stop"判定更保守寫實的處理，算是
        這裡為了跟其他引擎的既有慣例一致而做的額外加強，不是原始變體A規則的一部分。
      - 停損出場後的冷卻期一樣用STOP_LOSS_COOLDOWN_DAYS * 2天(近似日曆天數)。
      - 出場的trade dict直接重用_close_mr_trade()，保持跟summarize_mr()/bootstrap
        流程的格式相容。
      - max_hold_days強制平倉(用Close價)，這是資金受限版才有的安全閥(見上方
        MAX_HOLD_DAYS_CAPITAL_CONSTRAINED_DEFAULT說明)，原本的simulate_variant_a_trades
        完全沒有這個機制。

    position必須已經有"target_armed"欄位(進場時由呼叫端初始化為False)；
    k_today是「今天」這個交易日的K值(NaN代表這天查不到K，例如資料缺口，此時只做
    停損/停利判定，不更新target_armed狀態，避免用缺值誤判)。
    """
    open_, high, low, close = row["Open"], row["High"], row["Low"], row["Close"]
    stop_price = position["stop_price"]

    if open_ <= stop_price:
        exit_price = open_ * (1 - slippage_pct)
        _close_mr_trade(position, exit_price, "stop_gap", date, trades)
        cooldown_until[position["code"]] = date + pd.Timedelta(days=STOP_LOSS_COOLDOWN_DAYS * 2)
        return None

    if low <= stop_price:
        exit_price = stop_price * (1 - slippage_pct)
        _close_mr_trade(position, exit_price, "stop", date, trades)
        cooldown_until[position["code"]] = date + pd.Timedelta(days=STOP_LOSS_COOLDOWN_DAYS * 2)
        return None

    if not pd.isna(k_today):
        if k_today >= TARGET_K_DEFAULT:
            position["target_armed"] = True
        elif position.get("target_armed") and k_today < TARGET_K_DEFAULT:
            _close_mr_trade(position, close, "target", date, trades)
            return None

    if position["hold_days"] >= max_hold_days:
        _close_mr_trade(position, close, "forced_close", date, trades)
        return None

    return position


def run_squeeze_kdj_capital_constrained_backtest(price_data: dict, universe: dict,
                                                   master_calendar: pd.DatetimeIndex,
                                                   starting_capital: float, variant: str = "B",
                                                   lots: int = 2, top_n: int = 3,
                                                   max_concurrent_positions: int = 3,
                                                   atr_stop_mult: float = 1.0, atr_target_mult: float = 2.0,
                                                   atr_period: int = 14,
                                                   max_hold_days: int = MAX_HOLD_DAYS_CAPITAL_CONSTRAINED_DEFAULT,
                                                   slippage_pct: float = 0.0,
                                                   features_by_code: dict = None) -> list:
    """
    擠壓+KDJ訊號的「資金/部位受限版」完整day-by-day walk-forward回測，只做多方
    (見模組docstring)。跟run_squeeze_kdj_exit_style_comparison()/
    run_squeeze_kdj_exit_style_comparison_is_oos()最根本的差異：那兩個函式對universe
    裡「每一檔」股票的「每一個」觸發訊號都視為獨立成交(資金/持倉無限)；這裡改成
    跟momentum_breakout_engine.run_momentum_breakout_backtest()一樣的逐日競爭結構——
    同一天如果有超過max_concurrent_positions個空位的候選同時觸發，只有排名前面的
    才進得去，其餘的訊號就這樣錯過，這才是真實帳戶會發生的事。

    進場時機(EntryFlag對齊，無lookahead)：
    compute_squeeze_kdj_features()算出的EntryFlag/PriorLow定義是「訊號在t日收盤後
    觸發、t+1日開盤進場」(見simulate_variant_a_trades/simulate_variant_b_trades的
    entry_idx = t + 1)。這裡的逐日迴圈是「站在日期d，決定d這天要不要做事」，所以
    改成預先把EntryFlag/PriorLow/「觸發K棒的收紅幅度」整個序列往後位移一天
    (.shift(1))，位移後第d列存的就是「d的前一個交易日(=t)有沒有觸發/觸發時的
    PriorLow/觸發K棒的漲幅」，查表時直接查第d列，不必在迴圈裡手動做t/t+1的日期運算，
    也不會有lookahead(位移後的資料本來就只包含「昨天」的資訊)。

    同一天多檔股票搶有限名額時怎麼排名(這是這個函式新增的判斷，不是原始驗證過的
    進場規則的一部分)：
    原始訊號(EntryFlag)是純0/1的旗標，不像momentum_breakout_engine那樣有一組加權
    訊號分數可以排名。這裡刻意不發明一個跟原始訊號無關的排名公式，改用一個最貼近
    「進場規則本身」、不需要新假設的簡單代理指標：觸發K棒當天(t日)本身的漲幅
    (close_t / close_t-1 - 1)——進場規則第3條本來就要求「t日收紅」，這裡只是把
    「收得有多紅」這個本來就已經算出來的資訊拿來排序，漲幅越大的候選排越前面，
    不是引入新的、沒驗證過的邏輯。誠實聲明：這個排名規則本身完全沒有被驗證過
    (不知道「觸發當天漲幅大」是不是真的代表訊號品質比較好)，只是在「資金有限、
    必須選一個」的前提下，矮子裡挑將軍的務實選擇；原始驗證(PF=5.01/3.44)裡
    每個訊號都被視為獨立可成交，根本不存在「選誰」這個問題，所以這個排名規則
    在原始驗證結果裡完全沒有被驗證過，使用這個函式的人應該把這一點放在心上。

    出場規則(跟上面變體A/B完全一致，只是逐日判斷)：
      variant="B"：進場當下記錄的ATR(atr_period天，預設14天——對齊
        compare_breakout.py實際呼叫simulate_variant_b_trades時用的atr_period=14，
        不是這個模組simulate_variant_b_trades()函式簽名本身的預設值20，見本函式
        atr_period參數說明/呼叫端docstring)算出固定停損/停利價位，之後逐日
        重用mean_reversion_engine._process_mr_day()處理停損/停利/強制平倉/冷卻期，
        跟momentum_breakout_engine對非移動停利倉位的處理完全同構，不重新實作。
      variant="A"：停損=觸發K棒前一根K棒的最低價(PriorLow，固定不動)；停利=K值
        曾經衝到>=80之後第一次跌破80那天的收盤價，這個repo兩個既有引擎都沒有這種
        「靠指標狀態出場」的機制，所以寫了_process_squeeze_kdj_variant_a_day()
        專門處理(見該函式docstring)。

    資金/部位管理(逐字鏡射momentum_breakout_engine.run_momentum_breakout_backtest()，
    不重新發明)：
      - max_concurrent_positions：同時最多持有幾個部位(不同標的)，預設3。
      - 單筆保證金上限：跟其他引擎一樣，不得超過starting_capital的
        DEFAULT_MARGIN_CAP_RATIO(35%)。
      - 整體保證金上限：max_concurrent_positions > 1時，自動抓
        min(DEFAULT_MARGIN_CAP_RATIO * max_concurrent_positions, 0.9)當整體上限，
        跟run_momentum_breakout_backtest()的total_margin_cap_ratio預設計算方式
        完全一樣(這裡沒有另外開放total_margin_cap_ratio參數給呼叫端覆寫，因為
        目前沒有需要覆寫的使用情境，保持介面單純；需要的話之後再加)。
      - estimate_margin()/get_contract_multiplier()(經由_close_mr_trade()內部呼叫)
        直接沿用既有的保證金/合約乘數查表，不重新計算。
      - 保證金檢查沒過、或min(top_n, slots_available)排名之後還是沒位置的候選，
        當天直接錯過，不會遞延到隔天用同一個「今天」的訊號補進場(明天會是明天的
        EntryFlag，不是今天沒排到的訊號延後生效)。
      - 跟run_momentum_breakout_backtest()同一個(略保守的)小瑕疵：候選成功進場但
        當天就立刻出場(例如開盤跳空直接跌破停損)時，這個部位佔用的「名額」在
        當天的迴圈裡還是算用掉了，不會在同一天的迴圈裡釋出重新遞補下一個候選——
        這是刻意保持跟既有引擎一致的行為，不是這裡才有的新瑕疵，也不影響正確性
        (隔天的迴圈一開始就會看到這個部位已經不在open_positions裡，正常遞補)。

    max_hold_days：見MAX_HOLD_DAYS_CAPITAL_CONSTRAINED_DEFAULT說明——原始變體A/B
    模擬沒有這個安全閥，這裡新增是因為「卡住的部位會永久佔用有限名額」在資金受限
    情境下是真實問題，在資金無限的情境下不是。

    沒有is_near_settlement()結算日濾網：那是台指期貨選擇權的結算日強制平倉機制，
    這個函式交易的是個股(透過股票期貨)，不是台指期貨本身，沒有對應的結算日強制
    平倉需求，加這個濾網是張冠李戴。

    features_by_code：外部預先算好時直接傳入(見precompute_squeeze_kdj_features_by_code()，
    IS/OOS兩次呼叫共用同一份)，沒給時這裡自己算一次。

    回傳：trades list，每筆trade dict的形狀跟mean_reversion_engine._close_mr_trade()
    產生的完全一樣(code/side/entry_date/exit_date/e_price/exit_price/exit_reason/
    lots/pnl_ntd/return_pct/hold_days)，可以直接餵summarize_mr()/bootstrap_resample_pnl()。
    """
    if variant not in ("A", "B"):
        raise ValueError(f"variant必須是'A'或'B'，收到{variant!r}")

    if features_by_code is None:
        features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)

    # 逐碼預先算好「查表版」的進場訊號(已經shift(1)對齊成「今天能不能進場」)，
    # 避免在day-by-day迴圈裡對每個日期重複做t/t+1的日期運算。
    entry_by_code = {}
    atr_by_code = {}
    for code, features in features_by_code.items():
        df = price_data.get(code)
        if df is None:
            continue
        close = df["Close"]
        trigger_return = (close / close.shift(1) - 1)  # t日(觸發K棒)本身的漲幅，排名用
        entry_by_code[code] = pd.DataFrame({
            "EntryToday": features["EntryFlag"].shift(1).fillna(False).astype(bool),
            "StopPriceA": features["PriorLow"].shift(1),
            "TriggerStrength": trigger_return.shift(1),
        }, index=df.index)
        if variant == "B":
            atr_by_code[code] = compute_atr_correct(df, period=atr_period).shift(1)

    effective_total_margin_cap_ratio = None
    if max_concurrent_positions > 1:
        effective_total_margin_cap_ratio = min(DEFAULT_MARGIN_CAP_RATIO * max_concurrent_positions, 0.9)

    trades = []
    cooldown_until = {}
    open_positions = []

    for date in master_calendar:
        # 1) 先處理既有部位的出場判定
        still_open = []
        for position in open_positions:
            df = price_data.get(position["code"])
            if df is None or date not in df.index:
                still_open.append(position)
                continue
            if date != position["entry_date"]:
                position["hold_days"] += 1
            row = df.loc[date]
            if variant == "B":
                updated = _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until,
                                           slippage_pct=slippage_pct)
            else:
                k_series = features_by_code[position["code"]]["K"]
                k_today = k_series.loc[date] if date in k_series.index else np.nan
                updated = _process_squeeze_kdj_variant_a_day(position, row, k_today, date, trades,
                                                               max_hold_days, cooldown_until,
                                                               slippage_pct=slippage_pct)
            if updated is not None:
                still_open.append(updated)
        open_positions = still_open

        # 2) 收集今天觸發進場的候選，依「觸發K棒當天漲幅」排名，依序補進空出來的名額
        held_codes = {p["code"] for p in open_positions}
        excluded_codes = {c for c, until in cooldown_until.items() if date < until} | held_codes
        slots_available = max_concurrent_positions - len(open_positions)
        if slots_available <= 0:
            continue

        candidates = []
        for code, sig_df in entry_by_code.items():
            if code in excluded_codes:
                continue
            if date not in sig_df.index:
                continue
            sig_row = sig_df.loc[date]
            if not bool(sig_row["EntryToday"]):
                continue
            if pd.isna(sig_row["TriggerStrength"]):
                continue
            df = price_data.get(code)
            if df is None or date not in df.index:
                continue
            candidates.append({
                "code": code,
                "stop_price_a": sig_row["StopPriceA"],
                "trigger_strength": float(sig_row["TriggerStrength"]),
            })

        candidates.sort(key=lambda c: c["trigger_strength"], reverse=True)
        candidates = candidates[: max(top_n, slots_available)]

        used_margin = sum(p["margin_used"] for p in open_positions)
        while slots_available > 0 and candidates:
            cand = candidates.pop(0)
            code = cand["code"]
            df = price_data[code]
            open_p = df.loc[date, "Open"]
            e_price = open_p * (1 + slippage_pct)

            if variant == "B":
                atr_at_signal = atr_by_code[code].loc[date] if date in atr_by_code[code].index else np.nan
                if pd.isna(atr_at_signal) or atr_at_signal <= 0:
                    continue
                stop_price = e_price - atr_stop_mult * atr_at_signal
                target_price = e_price + atr_target_mult * atr_at_signal
            else:
                stop_price = cand["stop_price_a"]
                if pd.isna(stop_price):
                    continue
                target_price = None  # 變體A的停利靠target_armed狀態判定，不是固定價位

            margin_needed = estimate_margin(code, open_p, lots)
            if margin_needed > starting_capital * DEFAULT_MARGIN_CAP_RATIO:
                continue
            if effective_total_margin_cap_ratio is not None and \
                    used_margin + margin_needed > starting_capital * effective_total_margin_cap_ratio:
                continue

            position = {
                "code": code, "side": "long", "entry_date": date,
                "e_price": e_price, "target_price": target_price, "stop_price": stop_price,
                "lots": lots, "hold_days": 1, "margin_used": margin_needed,
            }
            if variant == "A":
                position["target_armed"] = False

            used_margin += margin_needed
            slots_available -= 1

            row = df.loc[date]
            if variant == "B":
                updated = _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until,
                                           slippage_pct=slippage_pct)
            else:
                k_series = features_by_code[code]["K"]
                k_today = k_series.loc[date] if date in k_series.index else np.nan
                updated = _process_squeeze_kdj_variant_a_day(position, row, k_today, date, trades,
                                                               max_hold_days, cooldown_until,
                                                               slippage_pct=slippage_pct)
            if updated is not None:
                open_positions.append(updated)
            else:
                used_margin -= margin_needed

    return trades
