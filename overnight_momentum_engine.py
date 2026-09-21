"""
overnight_momentum_engine.py
=============================
「隔日衝」策略引擎：今天收盤前用技術面+籌碼面評分選股，明天開盤先檢查跳空，
用 ATR 停利停損管理，收盤前強制平倉（不留倉過第二晚）。

設計依討論結果：
  技術面評分（5項，取平均 -> technical_score, 0~100）：
    - 收盤位置（收在當天高低點的哪個位置）
    - 量比（今天量 / 過去20日均量）
    - 相對大盤強弱（個股漲幅 - 大盤漲幅）
    - 漲幅%（鐘型評分，太小太誇張都扣分，中間區間最高分）
    - 股價位階（離季線 MA60 太遠要扣分，防止追高）

  籌碼面評分（3項，取平均 -> chip_score, 0~100）：
    - 當沖比例（越低分越高——比例太高代表沒人要留倉）
    - 外資買超金額佔成交金額比重
    - 投信買超金額佔成交金額比重

  最終分數 = technical_score * TECH_WEIGHT + chip_score * CHIP_WEIGHT
  （預設 35% / 65%，籌碼面權重高於均值回歸策略，因為隔日衝賭的是「明天還有沒有人買」，
   這件事技術線圖看不出來，只有籌碼資料看得出來）

  硬門檻（不算分，沒過直接刷掉，不進入候選名單）：
    - 成交金額門檻（流動性）
    - 站上 5 日均線 且 站上 20 日均線（方向濾網）

進出場邏輯：
  T日收盤前：依最終分數選出前 N 檔，以收盤價（近似）進場
  T+1日開盤：跳空檢查——開盤跌幅超過 GAP_STOP_THRESHOLD 直接停損出場
  T+1日盤中：ATR 停利/停損（1.2 ATR / 0.8 ATR，可調）
  T+1日收盤前：以上都沒觸發則強制平倉，不留倉過第二晚

輸入資料的介面約定（避免耦合到特定資料源的欄位命名）：
  price_df（每檔股票一份，dict[code] -> DataFrame）：
      index 或 'date' 欄位為交易日字串 YYYYMMDD，欄位至少要有
      ['open','high','low','close','volume']

  market_df：大盤/指標股（例如加權指數或2330）的 DataFrame，
      欄位 ['date','close']，用來算「相對大盤強弱」

  foreign_ratio_df / trust_ratio_df：
      欄位 ['date','code','ratio']，ratio = 買超金額 / 當天成交金額
      （由呼叫端拿 chip_data_loader.py 的買賣超資料 + 成交金額算好後傳入，
        這裡不假設 chip_data_loader.py 內部欄位名稱，介面在此對齊）

  day_trading_ratio_df：
      day_trading_loader.compute_day_trading_ratio() 的輸出，
      欄位至少要有 ['date','code','day_trading_ratio']
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 可調參數
# ---------------------------------------------------------------------------

TECH_WEIGHT = 0.35
CHIP_WEIGHT = 0.65

MIN_TURNOVER_VALUE = 20_000_000  # 硬門檻：當天成交金額至少 2000 萬（避免冷門股）

GAIN_PCT_MIN = 1.0       # 漲幅% 評分下限
GAIN_PCT_MAX = 9.5       # 漲幅% 評分上限（接近漲停就快速扣分）
GAIN_PCT_IDEAL_LOW = 3.0   # 理想區間下界（滿分區間）
GAIN_PCT_IDEAL_HIGH = 6.0  # 理想區間上界（滿分區間）

PRICE_LEVEL_COMFORT_PCT = 8.0   # 離季線在這個百分比之內不扣分
PRICE_LEVEL_PENALTY_SCALE = 4.0  # 超過comfort後，每超過1%扣多少分

GAP_STOP_THRESHOLD = -0.015   # 開盤跳空跌幅超過 -1.5% 直接停損出場
US_MARKET_DROP_THRESHOLD_PCT = -1.5  # 隔夜美股跌幅超過這個%，當天完全不進場（硬門檻，不是扣分）
ATR_STOP_MULT = 0.8
ATR_TARGET_MULT = 1.2

TOP_N = 5

# 8 個子訊號的名稱，順序跟 scan_candidates_for_date() 裡算出來的欄位名稱一致。
# 給 signal_weights 這種「每個訊號各自可調權重」的用法用（例如單一訊號拆解測試），
# 跟 tech_weight/chip_weight 那種「兩組各自取平均」的舊用法是兩條平行路徑。
SIGNAL_NAMES = [
    "score_close_position", "score_volume_ratio", "score_rel_strength",
    "score_gain_pct", "score_price_level",
    "score_day_trading", "score_foreign", "score_trust",
]

MA_SHORT = 5
MA_MID = 20
MA_LONG = 60
VOL_AVG_WINDOW = 20
ATR_WINDOW = 14


# ---------------------------------------------------------------------------
# 通用評分工具
# ---------------------------------------------------------------------------

def percentile_score(series, higher_is_better=True):
    """
    把一個 Series 換算成同一批候選者裡的 0~100 百分位分數。
    higher_is_better=False 時（例如當沖比例，越低越好），先反轉再排名。
    NaN 值給分數 0（视为最不利，不會被優先選中，但不會讓程式崩潰）。
    """
    s = series.copy().astype(float)
    valid = s.notna()
    if valid.sum() == 0:
        return pd.Series(0.0, index=series.index)

    ranked = s[valid].rank(pct=True, ascending=higher_is_better)
    # rank pct 範圍是 (0,1]，換算成 5~100 分（避免最低分變成0造成鐘型/其他分數相減時失真）
    scores = pd.Series(0.0, index=series.index)
    scores.loc[valid] = 5 + ranked * 95
    return scores


def bell_score(series, lo, hi, ideal_lo, ideal_hi):
    """
    鐘型評分：落在 [ideal_lo, ideal_hi] 給 100 分；
    往 lo/hi 兩端遞減，超出 [lo, hi] 範圍給 0 分。
    用於「漲幅%」這種有明確合理區間、不是排名越前面越好的訊號。
    """
    s = series.astype(float)
    scores = pd.Series(0.0, index=series.index)

    in_ideal = (s >= ideal_lo) & (s <= ideal_hi)
    scores.loc[in_ideal] = 100.0

    below = (s >= lo) & (s < ideal_lo)
    if below.any():
        scores.loc[below] = (s.loc[below] - lo) / (ideal_lo - lo) * 100.0

    above = (s > ideal_hi) & (s <= hi)
    if above.any():
        scores.loc[above] = (hi - s.loc[above]) / (hi - ideal_hi) * 100.0

    return scores.clip(lower=0.0, upper=100.0)


def price_level_score(deviation_pct, comfort_pct=PRICE_LEVEL_COMFORT_PCT,
                       penalty_scale=PRICE_LEVEL_PENALTY_SCALE):
    """
    股價位階評分：deviation_pct = (close - MA60) / MA60 * 100。
    在 comfort_pct 之內給滿分 100；超過之後，每超過 1% 扣 penalty_scale 分。
    低於 -comfort_pct（跌破季線太多）也扣分，因為不符合順勢邏輯。
    """
    s = deviation_pct.astype(float)
    abs_dev = s.abs()
    scores = pd.Series(100.0, index=deviation_pct.index)

    over = abs_dev > comfort_pct
    if over.any():
        penalty = (abs_dev.loc[over] - comfort_pct) * penalty_scale
        scores.loc[over] = (100.0 - penalty).clip(lower=0.0)

    scores.loc[s.isna()] = 0.0
    return scores


# ---------------------------------------------------------------------------
# 技術指標預計算（沿用專案既有的「一次算好，不在迴圈裡重算」原則）
# ---------------------------------------------------------------------------

def compute_true_range(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def precompute_overnight_indicators(df):
    """
    對單一股票的價量 DataFrame（需有 open/high/low/close/volume，按日期升冪排序）
    一次算好隔日衝評分所需的所有技術欄位。
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["prev_close"] = out["close"].shift(1)
    out["gain_pct"] = (out["close"] / out["prev_close"] - 1.0) * 100.0

    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_position"] = (out["close"] - out["low"]) / rng  # 0~1

    out["vol_avg20"] = out["volume"].rolling(VOL_AVG_WINDOW, min_periods=5).mean().shift(1)
    out["volume_ratio"] = out["volume"] / out["vol_avg20"]

    out["ma5"] = out["close"].rolling(MA_SHORT, min_periods=MA_SHORT).mean()
    out["ma20"] = out["close"].rolling(MA_MID, min_periods=MA_MID).mean()
    out["ma60"] = out["close"].rolling(MA_LONG, min_periods=MA_LONG).mean()

    out["price_level_dev"] = (out["close"] - out["ma60"]) / out["ma60"] * 100.0

    tr = compute_true_range(out)
    out["atr14"] = tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean()

    out["turnover_value"] = out["close"] * out["volume"]

    return out


def precompute_market_returns(market_df):
    """大盤/指標股的每日報酬率，用於計算相對強弱。"""
    m = market_df.sort_values("date").reset_index(drop=True).copy()
    m["market_return_pct"] = (m["close"] / m["close"].shift(1) - 1.0) * 100.0
    return m[["date", "market_return_pct"]]


# ---------------------------------------------------------------------------
# 候選股掃描（單一交易日）
# ---------------------------------------------------------------------------

def scan_candidates_for_date(date_str, indicators_by_code, market_returns_df,
                              foreign_ratio_df, trust_ratio_df, day_trading_ratio_df,
                              universe_codes=None, top_n=TOP_N,
                              tech_weight=TECH_WEIGHT, chip_weight=CHIP_WEIGHT,
                              us_market_returns_df=None,
                              us_market_drop_threshold=US_MARKET_DROP_THRESHOLD_PCT,
                              signal_weights=None,
                              ex_dividend_dates_by_code=None,
                              chip_date_str=None,
                              us_market_date_str=None,
                              min_candidates=None,
                              min_trust_ratio=None):
    """
    對指定日期，回傳依最終分數排序的候選股 DataFrame（已套用硬門檻與評分）。
    若當天沒有任何股票通過硬門檻，回傳空 DataFrame。

    min_candidates：可選，當天通過硬門檻的候選股數量門檻。不傳（維持None）
    就完全不啟用，向後相容，跟改之前行為一模一樣。

    背景：final_score用的是percentile_score，是「當天通過硬門檻的這批候選股
    裡面排第幾名」的相對名次分數，不是絕對品質分數——代表就算今天只有2、3檔
    股票通過硬門檻(盤面很弱、能買的標的很少)，裡面排名最高的那檔一樣會被算出
    接近滿分，跟它放到歷史上、放到全市場比是不是真的強完全無關。這會讓策略
    在盤面很弱的日子，還是被迫矬子裡拔將軍選一檔出來交易。

    設min_candidates=N，代表「當天通過硬門檻的候選股數量 < N」時，直接視為
    今天盤面太弱、候選池太窄，回傳空DataFrame（今天完全不交易），跟
    us_market_drop_threshold那個「隔夜美股大跌就不進場」的開關是同一種概念：
    用「今天的市場廣度不夠」當作額外的硬門檻，而不是靠分數本身去反映
    （分數本身反映不出來，見上段說明）。

    us_market_returns_df：可選，us_market_loader.load_us_market_returns() 的輸出
    （欄位 date/close/return_pct）。這是隔夜跳空風險的源頭濾網——如果當天(date_str)
    對應的美股報酬率低於 us_market_drop_threshold(預設-1.5%)，代表隔夜美股大跌，
    直接跳過整天不進場，回傳空 DataFrame，而不是等進場後被動被跳空停損打到。
    不傳這個參數（維持 None）就完全不啟用這個濾網，行為跟改之前一模一樣，
    向後相容，不影響任何既有呼叫方式或測試。

    ⚠️ 這個濾網原本查的是「date_str當天」的美股報酬率，但根據us_market_loader.py
    自己註解的時區換算：「美股日期X」的交易時段，是落在「台股日期X收盤後」到
    「台股日期X+1開盤前」這段空窗期——也就是說，在date_str當天13:30做進場決策
    的當下，date_str自己這天的美股行情其實還沒發生(要等到當天晚上9點半後才開盤)，
    直接查date_str當天等於是用了未來才會出現的資料，跟三大法人資料是同一種
    時間軸兜不起來的問題。要跑「真的能在收盤前執行」的版本，呼叫端應該把
    us_market_date_str傳成T-1的日期字串（見下方us_market_date_str參數，以及
    run_overnight_backtest的use_prior_day_us_market_data參數）。

    signal_weights：可選，dict[str, float]，key 是 SIGNAL_NAMES 裡的欄位名稱
    （8個子訊號各自的分數欄位名）。傳入的話會**取代**掉 tech_weight/chip_weight
    那種「兩組各自取平均再加權」的算法，改成對全部8個子訊號直接做加權平均
    （沒列在 dict 裡的訊號權重視為0，不會影響最終分數）。
    用途：單一訊號拆解測試（sweep_signal_ablation.py）——只給某一個訊號
    weight=1，其餘不列出，就能看這個訊號單獨拿來選股的效果。
    維持 None（預設）就完全不影響既有行為，向後相容。

    ex_dividend_dates_by_code：可選，dict[str, set(str)]，
    dividend_data_loader.load_dividend_events() 的輸出。如果某檔股票在
    date_str 這天剛好是除權息/減資/分割日，直接把它從候選名單剔除——因為
    data_loader.py 抓的是未還原股價，除權息當天收盤價會機械性跳空下跌，
    這個跳動會被漲幅%評分、相對大盤強弱評分誤判成「轉弱了」，也會汙染
    atr14，甚至可能誤觸隔天的跳空停損。不傳這個參數（維持 None）就完全
    不啟用這個濾網，行為跟改之前一模一樣，向後相容。

    chip_date_str：可選，三大法人/當沖比例資料實際要用「哪一天」的日期去查，
    跟 date_str（技術面用的日期，也就是進場當天）分開處理。不傳（維持 None）
    就沿用舊行為，直接用 date_str 當天的籌碼資料——但這件事在實務上其實不可能
    做到：證交所的三大法人買賣超日報(T86)是當天收盤後才公布，不可能在
    date_str 當天收盤前選股當下，就已經知道 date_str「自己當天」的三大法人
    買賣超數字。真正能在收盤前拿到的，只有前一個交易日（T-1）收盤後公布的
    資料。要跑「真的能在實務上執行」的版本，呼叫端應該傳入 T-1 的日期字串
    （見 run_overnight_backtest 的 use_prior_day_chip_data 參數）。

    us_market_date_str：可選，美股濾網實際要用「哪一天」的日期去查，跟
    date_str 分開處理，理由跟 chip_date_str 一模一樣（見上方⚠️說明）。
    不傳（維持 None）就沿用舊行為，直接查 date_str 當天。

    min_trust_ratio：可選，投信買超金額佔成交金額比重（trust_ratio，原始數值，
    不是分數）的絕對門檻。不傳（維持None）就完全不啟用，向後相容。

    背景：score_trust用的也是percentile_score，是「當天候選股裡投信買超比重
    排第幾名」的相對名次分數，只知道誰是當天第一名，不知道差距有多大——今天
    最高的trust_ratio可能是5%（積極買超），也可能只是0.1%（幾乎沒買，但還是
    矬子裡拔將軍變成當天第一名），percentile_score算出來的分數可能很接近。
    這是min_candidates要解決的「候選股數量太少」問題以外，另一個獨立的問題：
    就算候選股數量夠多，分數本身也看不出「今天最強的訊號，是不是連歷史上/
    跟其他天比都算真的強」。

    設min_trust_ratio=X，代表trust_ratio缺值(NaN)或低於X的候選股，
    直接從候選名單剔除（不是扣分，是硬門檻），只有原始trust_ratio真的
    達到這個絕對門檻的股票，才有資格被納入候選池繼續評分排名。

    這個門檻是加在min_candidates的檢查「之後」（不影響min_candidates原本
    「通過技術面硬門檻的候選股數量」這個判定基準，兩者是各自獨立、可疊加
    的門檻）、技術面/籌碼面評分「之前」——被這個門檻剔除的候選股，不會進入
    percentile_score排名，也就不會拉低/墊高其他候選股的相對名次分數。

    ⚠️ 這個參數如果是新調的，應該先在原本的IS/OOS資料(2023-09-15~2026-09-14)
    上調好、鎖定下來，再拿來跨期驗證區間做最終確認，不要直接在跨期驗證資料上
    試調參數——否則跨期驗證就失去「獨立樣本外測試」的意義了。
    """
    us_market_lookup_date = us_market_date_str if us_market_date_str is not None else date_str
    if us_market_returns_df is not None and not us_market_returns_df.empty:
        us_row = us_market_returns_df[us_market_returns_df["date"] == us_market_lookup_date]
        if len(us_row):
            us_return = us_row["return_pct"].iloc[0]
            if pd.notna(us_return) and us_return <= us_market_drop_threshold:
                return pd.DataFrame()  # 隔夜美股大跌，今天完全不進場

    market_row = market_returns_df[market_returns_df["date"] == date_str]
    market_ret = float(market_row["market_return_pct"].iloc[0]) if len(market_row) else np.nan

    rows = []
    codes = universe_codes if universe_codes is not None else indicators_by_code.keys()
    for code in codes:
        df = indicators_by_code.get(code)
        if df is None:
            continue
        row = df[df["date"] == date_str]
        if row.empty:
            continue
        r = row.iloc[0]

        # 硬門檻
        if pd.isna(r["turnover_value"]) or r["turnover_value"] < MIN_TURNOVER_VALUE:
            continue
        if pd.isna(r["ma5"]) or pd.isna(r["ma20"]):
            continue
        if not (r["close"] > r["ma5"] and r["close"] > r["ma20"]):
            continue
        if pd.isna(r["atr14"]):
            continue
        if ex_dividend_dates_by_code is not None and \
                date_str in ex_dividend_dates_by_code.get(code, set()):
            continue  # 除權息/減資/分割日，未還原股價會機械性跳空，訊號不可靠，直接跳過

        rel_strength = r["gain_pct"] - market_ret if not pd.isna(market_ret) else np.nan

        rows.append({
            "date": date_str,
            "code": code,
            "close": r["close"],
            "atr14": r["atr14"],
            "gain_pct": r["gain_pct"],
            "close_position": r["close_position"],
            "volume_ratio": r["volume_ratio"],
            "rel_strength": rel_strength,
            "price_level_dev": r["price_level_dev"],
        })

    if not rows:
        return pd.DataFrame()

    if min_candidates is not None and len(rows) < min_candidates:
        # 今天通過硬門檻的候選股數量太少(盤面太窄)，直接視為今天不交易，
        # 不進入評分/排名階段——見上方docstring關於percentile_score相對名次的說明。
        return pd.DataFrame()

    cand = pd.DataFrame(rows)

    # 併入籌碼面資料（當天沒有資料的股票，ratio 視為 NaN，percentile_score 會給0分）
    # chip_lookup_date：預設等於date_str(舊行為)，傳入chip_date_str的話改用那一天
    # (通常是T-1，見上面docstring關於「T86收盤後才公布」的說明)
    chip_lookup_date = chip_date_str if chip_date_str is not None else date_str
    dtr = day_trading_ratio_df[day_trading_ratio_df["date"] == chip_lookup_date][["code", "day_trading_ratio"]]
    fr = foreign_ratio_df[foreign_ratio_df["date"] == chip_lookup_date][["code", "ratio"]].rename(
        columns={"ratio": "foreign_ratio"})
    tr = trust_ratio_df[trust_ratio_df["date"] == chip_lookup_date][["code", "ratio"]].rename(
        columns={"ratio": "trust_ratio"})

    cand = cand.merge(dtr, on="code", how="left")
    cand = cand.merge(fr, on="code", how="left")
    cand = cand.merge(tr, on="code", how="left")

    if min_trust_ratio is not None:
        # 投信買超比重絕對門檻：缺值或低於門檻直接剔除，不進入排名分數計算，
        # 見上方docstring關於percentile_score排名分數看不出絕對強弱的說明。
        cand = cand[cand["trust_ratio"].notna() & (cand["trust_ratio"] >= min_trust_ratio)]
        if cand.empty:
            return pd.DataFrame()

    # 技術面評分
    cand["score_close_position"] = percentile_score(cand["close_position"], higher_is_better=True)
    cand["score_volume_ratio"] = percentile_score(cand["volume_ratio"], higher_is_better=True)
    cand["score_rel_strength"] = percentile_score(cand["rel_strength"], higher_is_better=True)
    cand["score_gain_pct"] = bell_score(
        cand["gain_pct"], GAIN_PCT_MIN, GAIN_PCT_MAX, GAIN_PCT_IDEAL_LOW, GAIN_PCT_IDEAL_HIGH
    )
    cand["score_price_level"] = price_level_score(cand["price_level_dev"])

    cand["technical_score"] = cand[[
        "score_close_position", "score_volume_ratio", "score_rel_strength",
        "score_gain_pct", "score_price_level",
    ]].mean(axis=1)

    # 籌碼面評分
    cand["score_day_trading"] = percentile_score(cand["day_trading_ratio"], higher_is_better=False)
    cand["score_foreign"] = percentile_score(cand["foreign_ratio"], higher_is_better=True)
    cand["score_trust"] = percentile_score(cand["trust_ratio"], higher_is_better=True)

    cand["chip_score"] = cand[["score_day_trading", "score_foreign", "score_trust"]].mean(axis=1)

    if signal_weights is not None:
        total_weight = sum(signal_weights.values())
        if total_weight <= 0:
            raise ValueError("signal_weights 的權重總和必須大於0")
        cand["final_score"] = sum(
            cand[name] * signal_weights.get(name, 0.0) for name in SIGNAL_NAMES
        ) / total_weight
    else:
        cand["final_score"] = cand["technical_score"] * tech_weight + cand["chip_score"] * chip_weight

    cand = cand.sort_values("final_score", ascending=False).reset_index(drop=True)
    return cand.head(top_n)


# ---------------------------------------------------------------------------
# 單筆交易模擬（T日進場 -> T+1日出場）
# ---------------------------------------------------------------------------

def simulate_next_day_exit(entry_price, atr, next_day_bar,
                            gap_stop_threshold=GAP_STOP_THRESHOLD,
                            atr_stop_mult=ATR_STOP_MULT,
                            atr_target_mult=ATR_TARGET_MULT):
    """
    給定進場價、ATR，以及 T+1 日的 OHLC，模擬出場價與出場原因。
    優先序：跳空停損 > 盤中停損 > 盤中停利 > 收盤前強制平倉。
    （若同一天內停損停利同時被觸及，保守起見優先判定為停損，
      因為無法從日K得知盤中觸價的先後順序，寧可低估績效）
    """
    open_p = next_day_bar["open"]
    high_p = next_day_bar["high"]
    low_p = next_day_bar["low"]
    close_p = next_day_bar["close"]

    gap_pct = (open_p - entry_price) / entry_price
    if gap_pct <= gap_stop_threshold:
        return open_p, "gap_stop"

    stop_price = entry_price - atr_stop_mult * atr
    target_price = entry_price + atr_target_mult * atr

    if low_p <= stop_price:
        return stop_price, "stop"
    if high_p >= target_price:
        return target_price, "target"

    return close_p, "forced_close"


def get_contract_multiplier(price):
    """沿用專案既有假設：股價 >= 500 用小型合約(100股)，否則標準合約(2000股)。"""
    return 100 if price >= 500 else 2000


# ---------------------------------------------------------------------------
# 完整回測迴圈
# ---------------------------------------------------------------------------

def run_overnight_backtest(indicators_by_code, market_returns_df,
                            foreign_ratio_df, trust_ratio_df, day_trading_ratio_df,
                            trading_days, universe_codes=None, top_n=TOP_N,
                            tech_weight=TECH_WEIGHT, chip_weight=CHIP_WEIGHT,
                            fee_per_trade=200,
                            gap_stop_threshold=GAP_STOP_THRESHOLD,
                            atr_stop_mult=ATR_STOP_MULT,
                            atr_target_mult=ATR_TARGET_MULT,
                            us_market_returns_df=None,
                            us_market_drop_threshold=US_MARKET_DROP_THRESHOLD_PCT,
                            signal_weights=None,
                            ex_dividend_dates_by_code=None,
                            use_prior_day_chip_data=False,
                            use_prior_day_us_market_data=False,
                            slippage_pct=0.0,
                            min_candidates=None,
                            min_trust_ratio=None):
    """
    對 trading_days（已排序的 YYYYMMDD 字串 list）逐日跑隔日衝策略。
    第 i 天收盤選股、進場；用第 i+1 天的 K 棒模擬出場。
    回傳交易紀錄 list[dict]。

    gap_stop_threshold / atr_stop_mult / atr_target_mult 開放給呼叫端覆寫，
    是給參數掃描（sweep_overnight_params.py）用的——沒有指定的話就是引擎預設值，
    行為跟修改前完全一樣，不影響既有呼叫方式。

    us_market_returns_df：傳入的話會啟用「隔夜美股大跌就整天不進場」的硬門檻濾網
    （見 scan_candidates_for_date 的說明），不傳就完全不影響既有行為。

    signal_weights：見 scan_candidates_for_date 的說明，往下傳給它。

    ex_dividend_dates_by_code：見 scan_candidates_for_date 的說明，往下傳給它。

    use_prior_day_chip_data：預設False，維持舊行為（用進場當天T自己的三大法人
    資料選股）——這是回測裡的簡化，實務上不可能做到，因為T86是T日收盤後才公布，
    你不可能在T日收盤前選股時就已經知道T日自己的三大法人買賣超。
    設成True，會改成用T-1日（前一個交易日，已經公布過的）三大法人/當沖比例資料
    來選股，技術面訊號仍然用T日當天自己的資料——這樣才是「真的能在收盤前執行」
    的版本。因為訊號時間點不一樣，跑出來的結果不會跟False版本一樣，需要重新驗證。

    use_prior_day_us_market_data：預設False，維持舊行為（查T日當天對應的美股
    報酬率）——這也是回測裡的簡化，實務上不可能做到，因為T日當天的美股行情
    要等到台北時間當晚9點半後才開盤，不可能在T日13:30進場決策時就已經知道。
    設成True，會改成查T-1日的美股報酬率（前一晚已經收盤、已知的資料，當作
    「昨晚已經大跌，今晚接續大跌風險較高」的保守替代訊號），才是真的能在
    收盤前執行的版本。

    slippage_pct：預設0.0，維持舊行為（假設進出場都能剛好成交在理論價位）。
    回測用的entry_price/exit_price都是K棒上的理論價位（收盤價、停損價、
    停利價...），實際下單時，尤其是期貨流動性沒有現貨好、隔日衝這種策略
    又偏好挑近期強勢股（更容易有人搶進搶出、價差變大），成交價幾乎不可能
    剛好等於理論價——買進通常要多付一點、賣出通常要少拿一點，統稱「滑價」。
    設成例如0.1，代表假設買進要多付0.1%、賣出要少拿0.1%（用「對自己不利」
    的方向調整，屬於保守估計）；只影響pnl的計算，不影響trades紀錄裡
    entry_price/exit_price本身（這兩欄仍然回報理論價，方便跟既有分析程式相容），
    也不影響simulate_next_day_exit()本身的判斷邏輯（停損/停利/出場理由的
    判斷依據仍然是理論價，滑價只在最後換算成pnl時才套用）。

    min_candidates：可選，往下傳給scan_candidates_for_date()，見它的docstring
    說明。不傳（維持None）就完全不啟用，向後相容。

    min_trust_ratio：可選，往下傳給scan_candidates_for_date()，見它的docstring
    說明。不傳（維持None）就完全不啟用，向後相容。
    """
    trades = []

    for i in range(len(trading_days) - 1):
        t_date = trading_days[i]
        t1_date = trading_days[i + 1]

        if use_prior_day_chip_data:
            # 用前一個交易日的籌碼資料；如果t_date已經是這段資料裡最早的一天
            # (i==0，沒有更早一天可查)，故意給一個保證查不到的日期字串，
            # 讓那天的籌碼分數自然變成0，而不是誤用成T日自己當天的資料。
            chip_date_str = trading_days[i - 1] if i >= 1 else ""
        else:
            chip_date_str = None  # 沿用舊行為：內部會自動退回用t_date當天

        if use_prior_day_us_market_data:
            us_market_date_str = trading_days[i - 1] if i >= 1 else ""
        else:
            us_market_date_str = None  # 沿用舊行為：內部會自動退回用t_date當天

        candidates = scan_candidates_for_date(
            t_date, indicators_by_code, market_returns_df,
            foreign_ratio_df, trust_ratio_df, day_trading_ratio_df,
            universe_codes=universe_codes, top_n=top_n,
            tech_weight=tech_weight, chip_weight=chip_weight,
            us_market_returns_df=us_market_returns_df,
            us_market_drop_threshold=us_market_drop_threshold,
            signal_weights=signal_weights,
            ex_dividend_dates_by_code=ex_dividend_dates_by_code,
            chip_date_str=chip_date_str,
            us_market_date_str=us_market_date_str,
            min_candidates=min_candidates,
            min_trust_ratio=min_trust_ratio,
        )
        if candidates.empty:
            continue

        for _, cand in candidates.iterrows():
            code = cand["code"]
            df = indicators_by_code.get(code)
            if df is None:
                continue
            next_row = df[df["date"] == t1_date]
            if next_row.empty:
                continue
            next_bar = next_row.iloc[0]

            entry_price = cand["close"]
            atr = cand["atr14"]

            exit_price, reason = simulate_next_day_exit(
                entry_price, atr, next_bar,
                gap_stop_threshold=gap_stop_threshold,
                atr_stop_mult=atr_stop_mult,
                atr_target_mult=atr_target_mult,
            )

            mult = get_contract_multiplier(entry_price)

            # 滑價：買進多付一點、賣出少拿一點（對自己不利的方向），只用來算pnl，
            # 不覆蓋trades紀錄裡回報的entry_price/exit_price理論價。
            actual_entry_price = entry_price * (1 + slippage_pct / 100.0)
            actual_exit_price = exit_price * (1 - slippage_pct / 100.0)
            pnl = (actual_exit_price - actual_entry_price) * mult - fee_per_trade

            trades.append({
                "entry_date": t_date,
                "exit_date": t1_date,
                "code": code,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": reason,
                "final_score": cand["final_score"],
                "technical_score": cand["technical_score"],
                "chip_score": cand["chip_score"],
                "pnl": pnl,
            })

    return trades


def summarize_overnight(trades):
    """交易紀錄的績效彙總，格式比照專案既有的 summarize_mr() 風格。"""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "total_pnl": 0.0, "avg_pnl": 0.0,
        }

    df = pd.DataFrame(trades)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]

    win_pnl = wins["pnl"].sum()
    loss_pnl = abs(losses["pnl"].sum())
    profit_factor = (win_pnl / loss_pnl) if loss_pnl > 0 else (win_pnl if win_pnl > 0 else 0.0)

    return {
        "total_trades": len(df),
        "win_rate": len(wins) / len(df) * 100.0,
        "profit_factor": profit_factor,
        "total_pnl": df["pnl"].sum(),
        "avg_pnl": df["pnl"].mean(),
        "by_exit_reason": df.groupby("exit_reason")["pnl"].agg(["count", "sum", "mean"]).to_dict("index"),
    }
