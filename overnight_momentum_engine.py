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
ATR_STOP_MULT = 0.8
ATR_TARGET_MULT = 1.2

TOP_N = 5

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
                              tech_weight=TECH_WEIGHT, chip_weight=CHIP_WEIGHT):
    """
    對指定日期，回傳依最終分數排序的候選股 DataFrame（已套用硬門檻與評分）。
    若當天沒有任何股票通過硬門檻，回傳空 DataFrame。
    """
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

    cand = pd.DataFrame(rows)

    # 併入籌碼面資料（當天沒有資料的股票，ratio 視為 NaN，percentile_score 會給0分）
    dtr = day_trading_ratio_df[day_trading_ratio_df["date"] == date_str][["code", "day_trading_ratio"]]
    fr = foreign_ratio_df[foreign_ratio_df["date"] == date_str][["code", "ratio"]].rename(
        columns={"ratio": "foreign_ratio"})
    tr = trust_ratio_df[trust_ratio_df["date"] == date_str][["code", "ratio"]].rename(
        columns={"ratio": "trust_ratio"})

    cand = cand.merge(dtr, on="code", how="left")
    cand = cand.merge(fr, on="code", how="left")
    cand = cand.merge(tr, on="code", how="left")

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
                            atr_target_mult=ATR_TARGET_MULT):
    """
    對 trading_days（已排序的 YYYYMMDD 字串 list）逐日跑隔日衝策略。
    第 i 天收盤選股、進場；用第 i+1 天的 K 棒模擬出場。
    回傳交易紀錄 list[dict]。

    gap_stop_threshold / atr_stop_mult / atr_target_mult 開放給呼叫端覆寫，
    是給參數掃描（sweep_overnight_params.py）用的——沒有指定的話就是引擎預設值，
    行為跟修改前完全一樣，不影響既有呼叫方式。
    """
    trades = []

    for i in range(len(trading_days) - 1):
        t_date = trading_days[i]
        t1_date = trading_days[i + 1]

        candidates = scan_candidates_for_date(
            t_date, indicators_by_code, market_returns_df,
            foreign_ratio_df, trust_ratio_df, day_trading_ratio_df,
            universe_codes=universe_codes, top_n=top_n,
            tech_weight=tech_weight, chip_weight=chip_weight,
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
            pnl = (exit_price - entry_price) * mult - fee_per_trade

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
