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


def precompute_regime_series(index_df: pd.DataFrame, ma_period: int = 120) -> pd.DataFrame:
    """把大盤代理指標的長期均線只算一次，避免day-by-day迴圈裡重複rolling運算。"""
    ma = index_df["Close"].rolling(ma_period).mean()
    return pd.DataFrame({"Close": index_df["Close"], "MA": ma}, index=index_df.index)


def compute_regime(regime_series: pd.DataFrame, as_of_date) -> str:
    """
    回傳 'bull' / 'bear'，依大盤代理指標 (預設2330) 相對長期均線的位置判斷。
    找不到足夠資料時回傳 'neutral'，此時多空訊號都正常放行(不縮小也不停用)。
    regime_series 是 precompute_regime_series() 預先算好的結果，這裡只做查表。
    """
    row, pos = _lookup_prior_row(regime_series, as_of_date)
    if row is None or pd.isna(row["MA"]):
        return "neutral"
    return "bull" if row["Close"] > row["MA"] else "bear"


def precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    針對一檔股票的完整價格序列，一次算好RSI/布林通道/60日均線/ATR。
    這些指標本質上都是「只看過去」的rolling計算，在完整序列上算一次，
    跟「每天只用當天以前的資料切片重新算一次」在數學上結果完全一樣，
    但前者是O(n)、後者等於在day-by-day迴圈裡對每檔股票重複算O(n)次、總共O(n²)，
    在全市場300多檔、跑3年資料、12種組合的情境下，會慢到不合理。
    這裡改成「先把每檔股票的指標一次算好存起來，之後day-by-day迴圈只是查表」。
    """
    close = df["Close"]
    rsi = compute_rsi(close)
    mid, upper, lower = compute_bollinger(close)
    ma60 = close.rolling(60).mean()
    atr = compute_atr_correct(df)
    return pd.DataFrame({
        "RSI": rsi, "BB_mid": mid, "BB_upper": upper, "BB_lower": lower,
        "MA60": ma60, "ATR": atr, "Close": close,
    }, index=df.index)


def _lookup_prior_row(ind_df: pd.DataFrame, as_of_date):
    """
    用searchsorted做O(log n)查詢，找出ind_df裡「日期嚴格早於as_of_date」的最後一列，
    等同於原本 df[df.index < as_of_date].iloc[-1]，但快很多，不用每次都重新切片整個序列。
    """
    pos = ind_df.index.searchsorted(as_of_date, side="left")
    if pos == 0:
        return None, pos
    return ind_df.iloc[pos - 1], pos


def scan_mean_reversion_candidates(indicators_by_code: dict, as_of_date,
                                    regime: str, excluded_codes: set, allow_short: bool = True,
                                    rsi_long_threshold: float = 30, rsi_short_threshold: float = 70,
                                    chip_streak_by_code: dict = None, min_chip_confirm_days: int = 0):
    """
    掃描全市場候選標的，回傳依「偏離程度」排序的前3名多方候選、前3名空方候選。
    regime == 'bull' 時停用空方訊號；regime == 'bear' 時停用多方訊號；'neutral' 兩者都放行。
    allow_short=False 時，不管regime是什麼，永遠不產生空方候選 (獨立於氛圍濾網的開關)。
    rsi_long_threshold/rsi_short_threshold 可以調鬆一點(例如35/65)來增加訊號出現頻率，
    預設30/70是教科書常見門檻。

    indicators_by_code: {code: precompute_indicators()的結果}，改用查表取代即時計算。

    chip_streak_by_code / min_chip_confirm_days：籌碼面確認條件(可選)。
    如果有提供 chip_streak_by_code (來自 chip_data_loader.precompute_chip_streak())，
    且 min_chip_confirm_days > 0，會額外要求：
    - 做多：外資連續買超天數 >= min_chip_confirm_days (機構籌碼認同這次超跌是機會，不是要崩了)
    - 做空：外資連續賣超天數 >= min_chip_confirm_days (機構籌碼認同這次超漲是過熱，不是要噴了)
    沒有籌碼資料的股票，在開啟這個條件時會被直接排除(視為沒有機構確認，保守處理)。
    """
    long_candidates = []
    short_candidates = []

    for code, ind_df in indicators_by_code.items():
        if code in excluded_codes:
            continue
        row, pos = _lookup_prior_row(ind_df, as_of_date)
        if row is None or pos < 60:
            continue

        rsi = row["RSI"]
        mid_v, upper_v, lower_v = row["BB_mid"], row["BB_upper"], row["BB_lower"]
        ma60 = row["MA60"]
        last_close = row["Close"]
        atr = row["ATR"]

        if pd.isna(mid_v) or pd.isna(ma60) or pd.isna(atr) or atr <= 0:
            continue

        chip_streak_val = None
        if min_chip_confirm_days > 0:
            if chip_streak_by_code is None or code not in chip_streak_by_code:
                continue  # 沒有籌碼資料，開啟籌碼確認時直接跳過這檔
            chip_row, chip_pos = _lookup_prior_row(chip_streak_by_code[code], as_of_date)
            # chip_streak_by_code[code] 是 precompute_chip_streak() 回傳的 Series (單一數值序列)，
            # 不是像 indicators_by_code 那樣的 DataFrame，_lookup_prior_row 對 Series 用 iloc 取出的
            # 直接就是純量數值，不能再呼叫 .iloc[0]，那是DataFrame欄位才需要的取法。
            chip_streak_val = None if chip_row is None else chip_row

        # 做多：RSI超賣 + 跌破布林下軌 + 仍在60日均線之上 (+可選：外資連續買超確認)
        if regime != "bear":
            if rsi < rsi_long_threshold and last_close <= lower_v and last_close > ma60:
                if min_chip_confirm_days > 0:
                    if chip_streak_val is None or chip_streak_val < min_chip_confirm_days:
                        pass  # 沒過籌碼確認，不加入候選
                    else:
                        score = rsi_long_threshold - rsi
                        long_candidates.append({
                            "code": code, "side": "long", "score": score,
                            "c_prev": last_close, "mid": mid_v, "lower": lower_v, "upper": upper_v, "atr": atr,
                        })
                else:
                    score = rsi_long_threshold - rsi
                    long_candidates.append({
                        "code": code, "side": "long", "score": score,
                        "c_prev": last_close, "mid": mid_v, "lower": lower_v, "upper": upper_v, "atr": atr,
                    })

        # 做空：RSI超買 + 站上布林上軌 + 仍在60日均線之下 (+可選：外資連續賣超確認)
        if allow_short and regime != "bull":
            if rsi > rsi_short_threshold and last_close >= upper_v and last_close < ma60:
                if min_chip_confirm_days > 0:
                    if chip_streak_val is None or chip_streak_val > -min_chip_confirm_days:
                        pass
                    else:
                        score = rsi - rsi_short_threshold
                        short_candidates.append({
                            "code": code, "side": "short", "score": score,
                            "c_prev": last_close, "mid": mid_v, "upper": upper_v, "lower": lower_v, "atr": atr,
                        })
                else:
                    score = rsi - rsi_short_threshold
                    short_candidates.append({
                        "code": code, "side": "short", "score": score,
                        "c_prev": last_close, "mid": mid_v, "upper": upper_v, "lower": lower_v, "atr": atr,
                    })

    long_candidates.sort(key=lambda x: x["score"], reverse=True)
    short_candidates.sort(key=lambda x: x["score"], reverse=True)
    return long_candidates[:3] + short_candidates[:3]


def try_enter_mean_reversion(price_data: dict, candidates: list, entry_date,
                              starting_capital: float, lots: int = 2, target_mode: str = "mid_band"):
    """
    依序檢查候選名單(已跳空風控+保證金上限過濾)，第一個通過的進場。
    target_mode == "mid_band":     出場目標是布林中軌(20日均線)，較保守，符合「小賺小賠」的均值回歸精神
    target_mode == "opposite_band": 出場目標是對側軌道(例如做多目標設在上軌)，更有企圖心，賺得多但達標機率通常較低
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
            target_price = cand["upper"] if target_mode == "opposite_band" else cand["mid"]
            stop_price = cand["lower"] - 1.0 * atr
        else:
            target_price = cand["lower"] if target_mode == "opposite_band" else cand["mid"]
            stop_price = cand["upper"] + 1.0 * atr

        return {
            "code": code, "side": side, "entry_date": entry_date,
            "e_price": e_price, "target_price": target_price, "stop_price": stop_price,
            "lots": lots, "hold_days": 1,
        }
    return None


def check_exit(row, position, slippage_pct: float = 0.0):
    """
    回傳 (event, exit_price)，event 為 'target' / 'stop' / 'stop_gap' / None。

    'stop_gap'：今天開盤價本身就已經跳空越過停損價(例如隔夜利空跳空低開，
    開盤已經比停損價還低)，這種情況下實際能成交的價位是開盤價，不是理論停損價——
    用理論停損價當出場價是過度樂觀的假設(等於假設一定能在停損價那個點精準出場，
    但跳空發生時那個價位根本沒有成交量)。用開盤價出場，是這裡對「跳空穿越停損」
    最務實的處理方式；真正的保證金追繳/強制斷頭機制比這個更複雜，這裡沒有完整模擬，
    但至少不會再假裝跳空跟沒跳空一樣可以用同一個價位出場。

    target_price 為 None 時代表這筆倉位用移動停利(trailing stop)出場，不看固定停利價。

    slippage_pct：只套用在停損/跳空停損這兩種「市價出場」的情境，不套用在停利(target)——
    停利通常是限價單，可以合理假設用設定的價位成交；停損是行情已經走到不利方向才觸發，
    實務上市場流動性通常比較差，用理論停損價/開盤價當成交價還是偏樂觀，這裡讓出場價位
    再往不利方向滑一點(多方停損成交價更低、空方停損成交價更高)，模擬真實的滑價成本。
    預設0.0(不模擬滑價)，跟舊版行為完全一致。
    """
    high, low, open_ = row["High"], row["Low"], row["Open"]
    target_price = position.get("target_price")
    if position["side"] == "long":
        if open_ <= position["stop_price"]:
            return "stop_gap", open_ * (1 - slippage_pct)
        # 保守假設：同一天內若停損停利都可能觸及，優先判定停損
        if low <= position["stop_price"]:
            return "stop", position["stop_price"] * (1 - slippage_pct)
        if target_price is not None and high >= target_price:
            return "target", target_price
    else:  # short
        if open_ >= position["stop_price"]:
            return "stop_gap", open_ * (1 + slippage_pct)
        if high >= position["stop_price"]:
            return "stop", position["stop_price"] * (1 + slippage_pct)
        if target_price is not None and low <= target_price:
            return "target", target_price
    return None, None


def update_trailing_stop(position, row):
    """
    移動停利：只在對持倉有利的方向移動停損價，絕不往回移(往不利方向移動等於
    自己放寬風險)。用「進場後最高/最低收盤價」當錨點，距離用進場當下固定的ATR
    (position['atr_entry'])換算，不隨每天的ATR重新變動——避免停損距離本身
    也跟著行情忽寬忽窄，讓移動停利的邏輯保持單純好懂。
    只有 position.get('trailing_stop') 為真的倉位才會被呼叫這個函式。
    """
    mult = position["trailing_atr_mult"]
    atr = position["atr_entry"]
    close = row["Close"]

    if position["side"] == "long":
        anchor = max(position.get("trailing_anchor", position["e_price"]), close)
        new_stop = anchor - mult * atr
        if new_stop > position["stop_price"]:
            position["stop_price"] = new_stop
        position["trailing_anchor"] = anchor
    else:
        anchor = min(position.get("trailing_anchor", position["e_price"]), close)
        new_stop = anchor + mult * atr
        if new_stop < position["stop_price"]:
            position["stop_price"] = new_stop
        position["trailing_anchor"] = anchor
    return position


def precompute_all_indicators(price_data: dict, universe: dict) -> dict:
    """
    對universe裡每一檔股票的指標只算一次，回傳 {code: precompute_indicators()的結果}。
    這份結果不受hold_days/rsi_threshold/target_mode影響，可以在12種組合、
    IS/OOS共24次回測之間直接共用，不用每次重算。
    """
    result = {}
    for code in universe:
        df = price_data.get(code)
        if df is None:
            continue
        result[code] = precompute_indicators(df)
    return result


def run_mean_reversion_backtest(price_data: dict, indicators_by_code: dict, regime_series: pd.DataFrame,
                                 master_calendar: pd.DatetimeIndex, max_hold_days: int,
                                 starting_capital: float, allow_short: bool = True,
                                 lots: int = 2, rsi_long_threshold: float = 30,
                                 rsi_short_threshold: float = 70, target_mode: str = "mid_band",
                                 chip_streak_by_code: dict = None, min_chip_confirm_days: int = 0):
    """
    完整 day-by-day walk-forward 模擬。
    max_hold_days: 短線版建議3-5，中期版建議10-20，長版可測30 (交易日)。
    allow_short=False 時只做多，等同long-only版本。
    target_mode: "mid_band"(回中軌，保守) 或 "opposite_band"(回對側軌道，更有企圖心)。
    chip_streak_by_code/min_chip_confirm_days: 可選的籌碼面確認條件，見 scan_mean_reversion_candidates 說明。

    indicators_by_code/regime_series 改成外部預先算好傳進來 (見 precompute_all_indicators /
    precompute_regime_series)，因為這兩者不受這次呼叫的參數影響，
    在多組合比較時應該只算一次、重複使用，而不是每次呼叫都重新算。
    """
    trades = []
    position = None
    cooldown_until = {}  # code -> 最後可再次進場的日期之前都排除

    for date in master_calendar:
        excluded_codes = {c for c, until in cooldown_until.items() if date < until}

        if position is None:
            if is_near_settlement(date, days_before=2):
                continue  # 結算日前1-2天不開新倉

            regime = compute_regime(regime_series, date)
            candidates = scan_mean_reversion_candidates(
                indicators_by_code, date, regime, excluded_codes, allow_short=allow_short,
                rsi_long_threshold=rsi_long_threshold, rsi_short_threshold=rsi_short_threshold,
                chip_streak_by_code=chip_streak_by_code, min_chip_confirm_days=min_chip_confirm_days,
            )
            if candidates:
                position = try_enter_mean_reversion(
                    price_data, candidates, date, starting_capital, lots, target_mode=target_mode
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


def _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until, slippage_pct: float = 0.0):
    event, exit_price = check_exit(row, position, slippage_pct=slippage_pct)

    if event in ("stop", "stop_gap"):
        _close_mr_trade(position, exit_price, event, date, trades)
        cooldown_until[position["code"]] = date + pd.Timedelta(days=STOP_LOSS_COOLDOWN_DAYS * 2)
        # *2 是粗略把交易日轉近似日曆天數，避免跨假日冷卻期被縮短；回測用途足夠精確
        return None

    if event == "target":
        _close_mr_trade(position, exit_price, "target", date, trades)
        return None

    if position["hold_days"] >= max_hold_days:
        _close_mr_trade(position, row["Close"], "forced_close", date, trades)
        return None

    if position.get("trailing_stop"):
        position = update_trailing_stop(position, row)

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


def _max_consecutive_losses(pnl_list: list) -> int:
    """連續虧損筆數的最長紀錄(交易依trades原本的順序，也就是出場日期先後順序)。
    這個數字比平均勝率更貼近「實際操作要撐過幾筆連虧才等得到下一筆賺錢」，
    勝率一樣的兩個策略，連續虧損次數可以差很多，心理/資金上能不能撐住是不同的事。"""
    longest = current = 0
    for p in pnl_list:
        if p < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_mr(trades: list, starting_capital: float) -> dict:
    if not trades:
        return {
            "trade_count": 0, "win_rate": 0.0, "avg_return_pct": 0.0,
            "total_pnl_ntd": 0.0, "max_drawdown_ntd": 0.0, "ending_equity_ntd": starting_capital,
            "long_count": 0, "short_count": 0,
            "profit_factor": 0.0, "max_consecutive_losses": 0, "sharpe_like": 0.0,
            "calmar_like": 0.0, "top_trade_pct_of_total_pnl": 0.0, "pnl_excluding_top3_ntd": 0.0,
        }

    pnl_list = [t["pnl_ntd"] for t in trades]
    returns = [t["return_pct"] for t in trades]
    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p < 0]
    total_pnl = float(sum(pnl_list))

    equity = [starting_capital]
    for p in pnl_list:
        equity.append(equity[-1] + p)
    equity = np.array(equity)
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    max_dd = float(dd.min())

    # Profit Factor = 總獲利 / 總虧損(取絕對值)。>1代表賺的比賠的多，<1代表反過來，
    # =1打平。只有賺沒有賠(losses為空)時視為無限大；一筆賺的都沒有時視為0，不用NaN，
    # 方便直接拿去排序比較，不用另外處理NaN。
    gross_win = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0

    # 用「每筆交易報酬率」當分布算的Sharpe-like比值(平均/標準差)，不是嚴謹的年化Sharpe
    # (交易之間間隔天數不固定，沒有真正做年化)，只拿來粗略比較「賺得穩不穩」，數字越高
    # 代表報酬相對於波動的效率越好；標準差為0(只有一筆交易或報酬完全一樣)時給0，避免除以0。
    returns_arr = np.array(returns)
    ret_std = float(returns_arr.std(ddof=0))
    sharpe_like = float(returns_arr.mean() / ret_std) if ret_std > 0 else 0.0

    # calmar-like = 總損益 / 最大回撤(絕對值)，不是嚴謹的年化Calmar比率(沒有年化報酬率)，
    # 只拿來粗略比較「賺的錢相對於曾經腰斬過的幅度划不划算」。
    calmar_like = float(total_pnl / abs(max_dd)) if max_dd != 0 else 0.0

    # 檢查獲利有沒有過度依賴少數幾筆極端交易：最大的一筆賺了多少佔總損益的比重，
    # 以及拿掉獲利最大的3筆之後，剩下的還賺不賺錢——如果拿掉3筆就轉虧，代表這個
    # 策略的優勢可能只是少數幾筆運氣好的交易撐出來的，不是穩定可重複的優勢。
    top_trade_pct = float(max(pnl_list) / total_pnl * 100) if total_pnl > 0 and pnl_list else 0.0
    sorted_pnl_desc = sorted(pnl_list, reverse=True)
    pnl_excluding_top3 = float(sum(sorted_pnl_desc[3:]))

    return {
        "trade_count": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_return_pct": float(np.mean(returns)) * 100,
        "total_pnl_ntd": total_pnl,
        "max_drawdown_ntd": max_dd,
        "ending_equity_ntd": float(equity[-1]),
        "long_count": sum(1 for t in trades if t["side"] == "long"),
        "short_count": sum(1 for t in trades if t["side"] == "short"),
        "profit_factor": profit_factor,
        "max_consecutive_losses": _max_consecutive_losses(pnl_list),
        "sharpe_like": sharpe_like,
        "calmar_like": calmar_like,
        "top_trade_pct_of_total_pnl": top_trade_pct,
        "pnl_excluding_top3_ntd": pnl_excluding_top3,
    }
