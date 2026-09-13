"""
回測核心引擎：與 main.py 的策略邏輯保持一致，唯一可切換的變因是 ATR 計算方式（buggy / correct）。
刻意把「進出場邏輯」跟「資料來源」拆開，方便先用合成資料驗證邏輯正確性，再接上真實歷史股價。
"""
import pandas as pd
import numpy as np


def compute_atr(df: pd.DataFrame, mode: str, period: int = 14) -> pd.Series:
    """
    mode == "buggy":   舊版錯誤公式，只算當日High-Low，沒有納入跳空缺口 (等同原始main.py的bug)
    mode == "correct": 標準 True Range 公式，納入前一日收盤價
    """
    if mode == "buggy":
        tr = df["High"] - df["Low"]
    elif mode == "correct":
        prev_close = df["Close"].shift(1)
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - prev_close).abs()
        tr3 = (df["Low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        raise ValueError(f"未知的 atr mode: {mode}")
    return tr.rolling(period).mean()


def compute_score(hist: pd.DataFrame, mode: str = "original") -> float:
    """
    mode == "original":        原版，只看現價相對5日均線的偏離程度
    mode == "volume_weighted": 加入成交量權重，量能配合(當日量/20日均量)才給更高分數，
                                用 min(倍數, 2.0) 封頂，避免單一天爆量把分數推得不合理的高
    """
    c_prev = hist["Close"].iloc[-1]
    ma5 = hist["Close"].rolling(5).mean().iloc[-1]
    base_score = (c_prev / ma5) * 10

    if mode == "original":
        return base_score
    elif mode == "volume_weighted":
        if "Volume" not in hist.columns:
            return base_score  # 沒有量能資料時退回原版，避免直接壞掉
        vol_now = hist["Volume"].iloc[-1]
        vol_ma20 = hist["Volume"].rolling(20).mean().iloc[-1]
        if pd.isna(vol_ma20) or vol_ma20 <= 0:
            vol_ratio = 1.0
        else:
            vol_ratio = vol_now / vol_ma20
        vol_ratio_capped = min(vol_ratio, 2.0)
        return base_score * vol_ratio_capped
    else:
        raise ValueError(f"未知的 score mode: {mode}")


def scan_candidates(price_data: dict, whitelist: dict, as_of_date, atr_mode: str, score_mode: str = "original"):
    """
    複製 main.py run_engine() 裡的選股邏輯：
    用「as_of_date 之前」的資料 (不含當天，模擬開盤前掃描)，找出符合多頭排列條件的候選股，
    依動能分數排序回傳前3名。
    """
    candidates = []
    for code, name in whitelist.items():
        df = price_data.get(code)
        if df is None:
            continue
        hist = df[df.index < as_of_date]
        if len(hist) < 20:
            continue

        c_prev = hist["Close"].iloc[-1]
        h_prev2 = hist["High"].iloc[-2]
        ma5 = hist["Close"].rolling(5).mean().iloc[-1]
        ma20 = hist["Close"].rolling(20).mean().iloc[-1]

        if not (c_prev > ma20 and c_prev > ma5 and c_prev > h_prev2):
            continue

        atr_series = compute_atr(hist, atr_mode)
        atr = atr_series.iloc[-1]
        if pd.isna(atr) or atr <= 0:
            continue

        score = compute_score(hist, score_mode)
        candidates.append({
            "code": code,
            "name": name,
            "score": score,
            "c_prev": c_prev,
            "atr": atr,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:3]


def try_enter(price_data: dict, candidates: list, entry_date):
    """
    複製 main.py 的順位遞補邏輯：依序檢查前3名候選股的「今日開盤跳空」是否通過風控 (> -0.5%)，
    第一個通過的就進場。回測用「當日開盤價」取代「即時報價」作為跳空判斷基準。
    """
    for cand in candidates:
        df = price_data.get(cand["code"])
        if df is None or entry_date not in df.index:
            continue
        open_p = df.loc[entry_date, "Open"]
        c_prev = cand["c_prev"]
        if c_prev == 0:
            continue
        gap_pct = (open_p - c_prev) / c_prev
        if gap_pct > -0.005:
            e_price = open_p
            atr = cand["atr"]
            sl_price = e_price - 0.8 * atr
            return {
                "code": cand["code"],
                "name": cand["name"],
                "entry_date": entry_date,
                "e_price": e_price,
                "sl_price": sl_price,
                "sl_price_initial": sl_price,  # 保留最初的停損價，t1觸發後sl_price會被移到保本價，這欄不會變動
                "t1_price": e_price + 1.0 * atr,
                "t2_price": e_price + 1.6 * atr,
                "pos_stage": 1,
                "hold_days": 1,
                "leg1_return": None,
            }
    return None


def check_day_exit(row, position):
    """
    回傳 (event, exit_price)。event 可能是 'stop' / 't1' / 't2' / 'stop_be' / None。
    同一天內若停損跟停利同時被觸及，保守起見優先判定停損先發生（較不利假設，
    對「原版」和「新版」一視同仁，不影響兩者的相對比較）。
    """
    high, low = row["High"], row["Low"]
    if position["pos_stage"] == 1:
        if low <= position["sl_price"]:
            return "stop", position["sl_price"]
        if high >= position["t1_price"]:
            return "t1", position["t1_price"]
    else:  # stage 2 (已移動停損至保本價)
        if low <= position["sl_price"]:
            return "stop_be", position["sl_price"]
        if high >= position["t2_price"]:
            return "t2", position["t2_price"]
    return None, None


def run_backtest(price_data: dict, whitelist: dict, master_calendar: pd.DatetimeIndex,
                  atr_mode: str, score_mode: str = "original"):
    """
    完整 day-by-day walk-forward 模擬，一次只持有一個部位（跟production一致）。
    回傳交易紀錄列表，每筆交易包含進出場日期、標的、總報酬率(以%表示，已加權平均2口)，
    以及供後續套用不同部位大小規則重新計算損益用的分腿明細。
    """
    trades = []
    position = None

    for date in master_calendar:
        if position is None:
            candidates = scan_candidates(price_data, whitelist, date, atr_mode, score_mode)
            if candidates:
                position = try_enter(price_data, candidates, date)
                if position is not None:
                    # 進場當天也要立刻檢查是否當天就觸價 (跟production同時啟動監控一致)
                    df = price_data[position["code"]]
                    row = df.loc[date]
                    position = _process_day(position, row, date, trades)
            continue

        # 已有持倉：先看今天這檔股票有沒有交易資料
        df = price_data.get(position["code"])
        if df is None or date not in df.index:
            continue  # 該股當天無資料 (罕見，例如個股暫停交易)，跳過這天

        # 非進場當天，才需要遞增持倉天數 (與production的「隔日 run_engine 才 +1」一致)
        if date != position["entry_date"]:
            position["hold_days"] += 1

        row = df.loc[date]
        position = _process_day(position, row, date, trades)

    return trades


def _close_trade(position, exit_price, exit_reason, date, trades):
    """統一結算一筆交易並寫入 trades 清單。
    保留 return_pct (沿用原本固定2口、50/50加權的定義，維持跟舊版compare.py相容)，
    同時額外記錄 leg1/leg2 分腿明細跟最初停損價，供之後套用不同部位大小規則重新計算損益。
    """
    e_price = position["e_price"]
    leg1_hit = position["pos_stage"] == 2

    if not leg1_hit:
        # 從沒到過第一階段目標，2口在同一價位一起出場
        total_return = (exit_price - e_price) / e_price
        leg1_exit_price = None
    else:
        # 第一口已在 t1 出場 (leg1_return)，這裡結算第二口 (leg2)，兩口各佔一半權重
        leg1_return = position["leg1_return"]
        leg2_return = (exit_price - e_price) / e_price
        total_return = 0.5 * leg1_return + 0.5 * leg2_return
        leg1_exit_price = position["t1_price"]

    trades.append({
        "code": position["code"], "name": position["name"],
        "entry_date": position["entry_date"], "exit_date": date,
        "e_price": e_price, "exit_price": exit_price,
        "sl_price_initial": position["sl_price_initial"],
        "leg1_hit": leg1_hit, "leg1_exit_price": leg1_exit_price,
        "leg2_exit_price": exit_price,
        "exit_reason": exit_reason, "return_pct": total_return,
        "hold_days": position["hold_days"],
    })


def _process_day(position, row, date, trades):
    """
    處理單一交易日的持倉狀態變化。
    回傳更新後的 position（若已平倉則回傳 None）。
    """
    event, exit_price = check_day_exit(row, position)

    if event == "t1":
        # 命中第一目標：不平倉，轉入第二階段繼續持有剩餘1口
        position["leg1_return"] = (exit_price - position["e_price"]) / position["e_price"]
        position["pos_stage"] = 2
        position["sl_price"] = position["e_price"]  # 移動停損至保本價
        return position

    if event in ("stop", "t2", "stop_be"):
        _close_trade(position, exit_price, event, date, trades)
        return None

    # 當天沒有觸發任何事件，檢查是否已達5天期限，達到就強制以當天收盤價平倉
    if position["hold_days"] >= 5:
        _close_trade(position, row["Close"], "forced_close", date, trades)
        return None

    return position


def summarize(trades: list, starting_capital: float = None) -> dict:
    if not trades:
        base = {
            "trade_count": 0, "win_rate": 0.0, "avg_return_pct": 0.0,
            "cumulative_return_pct": 0.0, "max_drawdown_pct": 0.0,
        }
        if starting_capital is not None:
            base.update({"total_pnl_ntd": 0.0, "max_drawdown_ntd": 0.0})
        return base

    returns = [t["return_pct"] for t in trades]
    wins = [r for r in returns if r > 0]

    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = np.array(equity)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    result = {
        "trade_count": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_return_pct": float(np.mean(returns)) * 100,
        "cumulative_return_pct": (equity[-1] - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
    }

    # 如果交易紀錄有經過 sizing.apply_sizing() 處理過 (含 pnl_ntd)，額外算金額口徑的統計，
    # 這個口徑對「風險動態調整口數」的組合比較有意義，因為它反映了實際部位大小放大/縮小的效果
    if starting_capital is not None and "pnl_ntd" in trades[0]:
        pnl_list = [t["pnl_ntd"] for t in trades]
        cum_equity_ntd = [starting_capital]
        for p in pnl_list:
            cum_equity_ntd.append(cum_equity_ntd[-1] + p)
        cum_equity_ntd = np.array(cum_equity_ntd)
        running_max_ntd = np.maximum.accumulate(cum_equity_ntd)
        dd_ntd = (cum_equity_ntd - running_max_ntd)
        result["total_pnl_ntd"] = float(sum(pnl_list))
        result["max_drawdown_ntd"] = float(dd_ntd.min())
        result["ending_equity_ntd"] = float(cum_equity_ntd[-1])

    return result
