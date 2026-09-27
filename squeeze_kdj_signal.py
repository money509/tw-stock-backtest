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

from mean_reversion_engine import compute_bollinger, compute_atr_correct

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
