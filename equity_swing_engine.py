"""
現股波段策略回測引擎。

跟其他既有引擎(momentum_breakout_engine.py 右側突破、mean_reversion_engine.py
均值回歸)是完全獨立的新典範，不是這兩者的變形：

  既有引擎(股票期貨 股期)                    這支引擎(現股 波段)
  --------------------------------------  --------------------------------------
  標的：股票期貨(有到期日、需展期)              標的：實際股票(現股，沒有到期日)
  持有：短則數日(3~20個交易日)                持有：2~8週(以交易週為單位)
  資金：保證金/槓桿(taifex_universe.py)       資金：現金全額交割，1張=1000股(整張進出)
  出場：結算日前強制平倉(is_near_settlement)   出場：沒有結算日限制
  訊號：技術面為主(量比/RSI/MACD/突破強度)     訊號：基本面(營收/評價/財報)+籌碼面為主
  方向：可多可空(期貨可以放空)                 方向：只做多(v1不做放空，見下方說明)

明確**不**重用的兩個既有元件(理由見上方對照表)：
  - mean_reversion_engine.is_near_settlement()：期貨結算日限制，現股沒有到期日，
    完全不適用。
  - sizing.py / taifex_universe.get_contract_multiplier() / estimate_margin()：
    期貨契約規格/保證金查表，現股沒有槓桿、沒有契約乘數，一張就是1000股現金交割。

明確**有**重用的既有元件(這些是通用的技術指標/統計工具，不是期貨專屬邏輯)：
  - mean_reversion_engine.compute_atr_correct/precompute_regime_series/
    compute_regime/_lookup_prior_row/summarize_mr
  - overnight_momentum_engine.percentile_score(訊號評分的排名轉換工具)
  - chip_data_loader.precompute_chip_streak(外資/投信連續買超天數，籌碼資料
    本身不是期貨專屬的，現股一樣適用)
  - robustness_analysis.py 的bootstrap工具(compare_equity_swing.py會用到)

⚠️ v1明確排除放空(Short)：現股放空在台灣市場需要券商信用交易額度、融券
標借券機制，本身有一整套跟現股做多完全不同的成本結構(融券手續費、標借費、
回補風險、平盤以下不能放空的規則)，這裡沒有模擬這些機制，所以v1是long-only，
這是刻意的範圍縮減，不是忘記做——之後如果要加，需要另外一支模組處理券源/
融券成本，不是在這裡隨便加個「反向」就能做對的事。

部位管理：v1支援「固定並行部位數、現金平均分配」，也支援「風險預算(%)反推股數」
兩種模式，見 compute_equity_shares() / try_enter_equity_position()。

出場邏輯：主要用「持有時間」(2~8週，可設定)出場，搭配ATR停損；這是全新寫的
day-by-day邏輯(見 _process_equity_day())，不是重用mean_reversion_engine.
_process_mr_day()——理由：_process_mr_day是為「日等級」移動停利/保本停損調校的，
這支引擎是「週等級」持有，停損/強制出場的判斷頻率跟語意不一樣(例如「最短持有
週數」這個概念在原本的引擎裡完全不存在)，硬套用_process_mr_day的邏輯反而會
不清楚哪些行為是刻意設計、哪些是意外繼承來的期貨邏輯。
"""
import pandas as pd
import numpy as np

from mean_reversion_engine import (
    compute_atr_correct, precompute_regime_series, compute_regime, _lookup_prior_row,
    summarize_mr,
)
from overnight_momentum_engine import percentile_score
from chip_data_loader import precompute_chip_streak

TRADING_DAYS_PER_WEEK = 5  # 粗略近似(不扣國定假日)，跟is_near_settlement一樣的精神：
                            # 假日已經隱含在價格資料本身裡(yfinance不會回傳休市日)，
                            # 這裡只是把「幾週」換算成「幾個交易日」方便跟day-by-day
                            # 迴圈對齊，不是精確的日曆計算。

# 現股交易成本(跟股票期貨COMMISSION_PER_LOT_PER_LEG的固定金額完全不同機制——
# 現股是「金額」比例課費，不是「口數」固定費用，這是資產類別本質不同，不是
# 隨便選一個數字)：
#   買進：手續費 BROKERAGE_FEE_RATE(牌告費率0.1425%，這裡用牌告費率，不假設
#         折扣，偏保守估計成本)
#   賣出：手續費 BROKERAGE_FEE_RATE + 證券交易稅 SECURITIES_TRANSACTION_TAX_RATE
#         (現股當沖以外，賣出現股課0.3%證交稅，這裡沒有模擬當沖減半課稅，
#         因為這支引擎本來就不是當沖策略)
BROKERAGE_FEE_RATE = 0.001425
SECURITIES_TRANSACTION_TAX_RATE = 0.003

SHARES_PER_LOT = 1000  # 一張 = 1000股，台股現股最小交易單位(不含零股)

# 這幾個訊號依賴新的三個基本面/籌碼面資料源，沒有資料時會自動退化成中性分數0.0，
# 不影響其他訊號的排名，精神跟momentum_breakout_engine.CHIP_DEPENDENT_SIGNALS完全一樣。
REVENUE_DEPENDENT_SIGNALS = {"score_revenue_yoy", "score_revenue_mom"}
VALUATION_DEPENDENT_SIGNALS = {"score_valuation_pe", "score_valuation_pb", "score_dividend_yield"}
# 財報訊號依賴信心程度最低、未經驗證的financial_statement_loader.py，特別強調
# 「可選、缺席時優雅降級」，跟另外兩組資料源的處理方式一致，但這裡的資料缺席
# 機率原本就被預期會更高(見financial_statement_loader.py docstring)。
FINANCIAL_STATEMENT_DEPENDENT_SIGNALS = {"score_eps_growth", "score_gross_margin", "score_roe"}
CHIP_DEPENDENT_SIGNALS = {"score_foreign_streak", "score_trust_streak"}

FUNDAMENTAL_DEPENDENT_SIGNALS = (
    REVENUE_DEPENDENT_SIGNALS | VALUATION_DEPENDENT_SIGNALS
    | FINANCIAL_STATEMENT_DEPENDENT_SIGNALS | CHIP_DEPENDENT_SIGNALS
)

EQUITY_SIGNAL_NAMES = [
    "score_revenue_yoy", "score_revenue_mom",
    "score_valuation_pe", "score_valuation_pb", "score_dividend_yield",
    "score_eps_growth", "score_gross_margin", "score_roe",
    "score_foreign_streak", "score_trust_streak",
]


def precompute_equity_indicators(price_df: pd.DataFrame, revenue_df: pd.DataFrame = None,
                                  valuation_df: pd.DataFrame = None, financial_df: pd.DataFrame = None,
                                  chip_df: pd.DataFrame = None, ma_gate_period: int = 60,
                                  atr_period: int = 14) -> pd.DataFrame:
    """
    針對一檔股票，把技術面(ATR/MA60)跟基本面/籌碼面資料(可選，任一個可以是None)
    對齊到同一份以價格日期為準的每日索引上，一次算好，day-by-day迴圈只查表。

    revenue_df/valuation_df/financial_df/chip_df 的index都是「這筆資料公開可見的
    日期」(公告日/交易日)，用 reindex(method="ffill") 對齊到price_df的每日索引——
    ffill只會往前補值(用「這天之前最後一次公開的數字」)，不會用到未來才公開的
    資料，這是避免look-ahead bias最基本的保證；再加上scan_equity_candidates()
    一律用_lookup_prior_row()只看「嚴格早於進場日」的那一列，等於有兩層時間
    保護(對齊時不偷看未來 + 使用時只看前一天收盤後的狀態)。

    任一個來源是None(對應的loader沒有資料、或這個範例根本沒有下載那個來源)，
    對應欄位整欄會是NaN，之後在_rank_equity_candidates()裡會被偵測到、該訊號
    自動退化成中性分數0.0，不會讓程式崩潰。
    """
    close = price_df["Close"]
    atr = compute_atr_correct(price_df, period=atr_period)
    ma_gate = close.rolling(ma_gate_period).mean()

    out = pd.DataFrame({"Close": close, "ATR": atr, "MA_GATE": ma_gate}, index=price_df.index)

    if revenue_df is not None and not revenue_df.empty:
        aligned = revenue_df.reindex(price_df.index, method="ffill")
        out["revenue_yoy_pct"] = aligned["revenue_yoy_pct"]
        out["revenue_mom_pct"] = aligned["revenue_mom_pct"]
    else:
        out["revenue_yoy_pct"] = np.nan
        out["revenue_mom_pct"] = np.nan

    if valuation_df is not None and not valuation_df.empty:
        aligned = valuation_df.reindex(price_df.index, method="ffill")
        out["pe"] = aligned["pe"]
        out["pb"] = aligned["pb"]
        out["dividend_yield"] = aligned["dividend_yield"]
    else:
        out["pe"] = np.nan
        out["pb"] = np.nan
        out["dividend_yield"] = np.nan

    if financial_df is not None and not financial_df.empty:
        fdf = financial_df.sort_index().copy()
        # EPS成長率：跟上一次公告的EPS比較(季對季，QoQ)，不是YoY——財報資料本身
        # 是稀疏的(一年只有4筆)，這裡取最直接可得的「上一期比較」，已經是這份
        # 稀疏資料能算出的最短週期成長率，跟revenue_data_loader算的YoY/MoM
        # (月頻率資料)不是同一個時間解析度，這裡誠實只做QoQ，不假裝算得出YoY
        # (算YoY需要至少4期前的資料，樣本數更少時會有更多NaN)。
        fdf["eps_growth_pct"] = fdf["eps"].pct_change() * 100
        aligned = fdf.reindex(price_df.index, method="ffill")
        out["eps_growth_pct"] = aligned["eps_growth_pct"]
        out["gross_margin_pct"] = aligned["gross_margin_pct"]
        out["roe_pct"] = aligned["roe_pct"]
    else:
        out["eps_growth_pct"] = np.nan
        out["gross_margin_pct"] = np.nan
        out["roe_pct"] = np.nan

    if chip_df is not None and not chip_df.empty:
        foreign_streak = precompute_chip_streak(chip_df, net_col="foreign_net")
        trust_streak = precompute_chip_streak(chip_df, net_col="trust_net")
        out["foreign_streak"] = foreign_streak.reindex(price_df.index, method="ffill")
        out["trust_streak"] = trust_streak.reindex(price_df.index, method="ffill")
    else:
        out["foreign_streak"] = np.nan
        out["trust_streak"] = np.nan

    return out


def precompute_all_equity_indicators(price_data: dict, universe: dict, revenue_data: dict = None,
                                      valuation_data: dict = None, financial_data: dict = None,
                                      chip_data: dict = None) -> dict:
    """對universe裡每一檔股票呼叫precompute_equity_indicators()，回傳{code: 結果}。
    revenue_data/valuation_data/financial_data/chip_data 任一個整個是None，代表
    這次回測完全沒有下載那個來源，所有股票該來源的訊號都會退化成中性分數。"""
    result = {}
    for code in universe:
        df = price_data.get(code)
        if df is None:
            continue
        result[code] = precompute_equity_indicators(
            df,
            revenue_df=(revenue_data.get(code) if revenue_data is not None else None),
            valuation_df=(valuation_data.get(code) if valuation_data is not None else None),
            financial_df=(financial_data.get(code) if financial_data is not None else None),
            chip_df=(chip_data.get(code) if chip_data is not None else None),
        )
    return result


def _rank_equity_candidates(rows: list, signal_weights: dict, top_n: int) -> list:
    """對通過結構性門檻的候選做加權排名。跟momentum_breakout_engine._rank_candidates()
    同樣的精神：每個訊號先在「當天全部候選」裡做百分位排名(percentile_score)，
    再依權重加總——這是相對排名，不是絕對值評分，理由跟原本引擎完全一樣：
    絕對值門檻對候選池大小/期間變動敏感，相對排名比較穩定。"""
    if not rows:
        return []
    df = pd.DataFrame(rows)

    df["score_revenue_yoy"] = (
        percentile_score(df["revenue_yoy_pct"], higher_is_better=True)
        if df["revenue_yoy_pct"].notna().any() else 0.0
    )
    df["score_revenue_mom"] = (
        percentile_score(df["revenue_mom_pct"], higher_is_better=True)
        if df["revenue_mom_pct"].notna().any() else 0.0
    )
    # 本益比/股價淨值比：越低越好，higher_is_better=False
    df["score_valuation_pe"] = (
        percentile_score(df["pe"], higher_is_better=False) if df["pe"].notna().any() else 0.0
    )
    df["score_valuation_pb"] = (
        percentile_score(df["pb"], higher_is_better=False) if df["pb"].notna().any() else 0.0
    )
    df["score_dividend_yield"] = (
        percentile_score(df["dividend_yield"], higher_is_better=True)
        if df["dividend_yield"].notna().any() else 0.0
    )
    df["score_eps_growth"] = (
        percentile_score(df["eps_growth_pct"], higher_is_better=True)
        if df["eps_growth_pct"].notna().any() else 0.0
    )
    df["score_gross_margin"] = (
        percentile_score(df["gross_margin_pct"], higher_is_better=True)
        if df["gross_margin_pct"].notna().any() else 0.0
    )
    df["score_roe"] = (
        percentile_score(df["roe_pct"], higher_is_better=True) if df["roe_pct"].notna().any() else 0.0
    )
    df["score_foreign_streak"] = (
        percentile_score(df["foreign_streak"], higher_is_better=True)
        if df["foreign_streak"].notna().any() else 0.0
    )
    df["score_trust_streak"] = (
        percentile_score(df["trust_streak"], higher_is_better=True)
        if df["trust_streak"].notna().any() else 0.0
    )

    total_weight = sum(signal_weights.get(name, 0.0) for name in EQUITY_SIGNAL_NAMES)
    if total_weight <= 0:
        total_weight = 1.0
    df["total_score"] = sum(
        df[name] * signal_weights.get(name, 0.0) for name in EQUITY_SIGNAL_NAMES
    ) / total_weight

    df = df.sort_values("total_score", ascending=False)
    out = []
    for _, r in df.iterrows():
        out.append({
            "code": r["code"], "side": "long", "score": float(r["total_score"]),
            "c_prev": float(r["close"]), "atr": float(r["atr"]),
        })
    return out[:top_n]


def scan_equity_candidates(indicators_by_code: dict, as_of_date, regime: str, excluded_codes: set,
                            signal_weights: dict = None, top_n: int = 5) -> list:
    """
    掃描全市場候選標的(long-only)：
      結構性門檻(必要條件)：站上MA_GATE(預設60日均線的長版上升趨勢背景過濾，
        類比其他引擎「close > MA20」的基本門檻，但用更長的均線適配數週持有)，
        且ATR/指標資料齊全。
      regime == 'bear' 時完全停用(大盤氛圍濾網不利多方時不進場)，'bull'/'neutral'放行。
    """
    if signal_weights is None:
        signal_weights = {name: 1.0 for name in EQUITY_SIGNAL_NAMES}
    if regime == "bear":
        return []

    rows = []
    for code, ind_df in indicators_by_code.items():
        if code in excluded_codes:
            continue
        row, pos = _lookup_prior_row(ind_df, as_of_date)
        if row is None or pos < 60:
            continue

        close = row["Close"]
        ma_gate = row["MA_GATE"]
        atr = row["ATR"]
        if pd.isna(ma_gate) or pd.isna(atr) or atr <= 0:
            continue
        if close <= ma_gate:
            continue  # 結構性門檻：站上長均線(基本上升趨勢背景)

        rows.append({
            "code": code, "close": close, "atr": atr,
            "revenue_yoy_pct": row["revenue_yoy_pct"], "revenue_mom_pct": row["revenue_mom_pct"],
            "pe": row["pe"], "pb": row["pb"], "dividend_yield": row["dividend_yield"],
            "eps_growth_pct": row["eps_growth_pct"], "gross_margin_pct": row["gross_margin_pct"],
            "roe_pct": row["roe_pct"],
            "foreign_streak": row["foreign_streak"], "trust_streak": row["trust_streak"],
        })

    return _rank_equity_candidates(rows, signal_weights, top_n)


def compute_equity_shares(cash_allocated: float, entry_price: float) -> int:
    """
    現股整張(board lot)進場股數計算：floor(可用現金 / (進場價 x 1000)) x 1000。
    不買零股(不足一張的部分直接捨去，不是這支引擎不支援零股交易，是v1刻意
    只模擬整張進出，比較貼近多數波段操作者的實務下單方式)。
    entry_price<=0或算出來不足一張時回傳0(呼叫端據此判斷放棄這個候選)。
    """
    if entry_price <= 0 or cash_allocated <= 0:
        return 0
    lots = int(cash_allocated // (entry_price * SHARES_PER_LOT))
    return max(0, lots) * SHARES_PER_LOT


def try_enter_equity_position(price_data: dict, candidates: list, entry_date, cash_allocated: float,
                               atr_stop_mult: float = 2.0, risk_pct_per_trade: float = None,
                               account_equity: float = None, slippage_pct: float = 0.0):
    """
    依序檢查候選名單，第一個「算得出至少一張股數」的進場。long-only，沒有跳空
    風控(現股沒有股票期貨那種隔日跳空追繳的急迫性，波段持有本來就會跨過很多天
    的價格波動，這裡不做「跳空超過0.5%就放棄」這種短線引擎的處理)。

    risk_pct_per_trade：給定時改用「風險預算反推股數」，取代單純的現金平分——
    股數 = floor((account_equity x risk_pct_per_trade) / (ATR停損距離)) 股，
    再用cash_allocated當作這筆交易可動用現金的「上限」(不會因為風險預算算出來
    的股數超過這個資金槽位原本分配到的現金而放大部位，維持槽位彼此獨立、不互相
    排擠的簡化假設，見run_equity_swing_backtest()說明)。算出來不足一張時放棄
    這個候選、換下一名。
    """
    for cand in candidates:
        code = cand["code"]
        df = price_data.get(code)
        if df is None or entry_date not in df.index:
            continue
        open_p = df.loc[entry_date, "Open"]
        if open_p <= 0:
            continue

        atr = cand["atr"]
        e_price = open_p * (1 + slippage_pct)
        stop_price = e_price - atr_stop_mult * atr
        if stop_price <= 0:
            continue

        if risk_pct_per_trade is not None:
            equity = account_equity if account_equity is not None else cash_allocated
            stop_distance = e_price - stop_price
            if stop_distance <= 0:
                continue
            risk_budget = equity * risk_pct_per_trade
            shares_by_risk = int(risk_budget // stop_distance)
            shares_by_risk = (shares_by_risk // SHARES_PER_LOT) * SHARES_PER_LOT
            shares_by_cash = compute_equity_shares(cash_allocated, e_price)
            shares = min(shares_by_risk, shares_by_cash)
        else:
            shares = compute_equity_shares(cash_allocated, e_price)

        if shares < SHARES_PER_LOT:
            continue  # 連一張都買不起/風險預算不夠一張，放棄，換下一名候選

        return {
            "code": code, "side": "long", "entry_date": entry_date,
            "e_price": e_price, "stop_price": stop_price, "atr_entry": atr,
            "shares": shares, "hold_days": 1,
        }
    return None


def _check_equity_exit(row, position, slippage_pct: float = 0.0):
    """回傳(event, exit_price)。event為'stop'/'stop_gap'/None，跟mean_reversion_engine.
    check_exit()的跳空處理精神一致(開盤已經跳空穿越停損價時用開盤價，不是理論停損價)，
    但這裡沒有target(這支引擎不設固定停利價，出場主要靠持有週數，見模組docstring)。"""
    open_, low = row["Open"], row["Low"]
    stop_price = position["stop_price"]
    if open_ <= stop_price:
        return "stop_gap", open_ * (1 - slippage_pct)
    if low <= stop_price:
        return "stop", stop_price * (1 - slippage_pct)
    return None, None


def _close_equity_trade(position, exit_price, reason, date, trades):
    """現股損益：買進手續費 + 賣出(手續費+證交稅)，跟期貨那種固定口數手續費
    完全不同機制，見模組docstring頂部BROKERAGE_FEE_RATE/SECURITIES_TRANSACTION_TAX_RATE
    的說明。"""
    e_price = position["e_price"]
    shares = position["shares"]

    buy_cost = e_price * shares
    sell_proceeds = exit_price * shares
    buy_fee = buy_cost * BROKERAGE_FEE_RATE
    sell_fee = sell_proceeds * (BROKERAGE_FEE_RATE + SECURITIES_TRANSACTION_TAX_RATE)

    pnl_ntd = (sell_proceeds - sell_fee) - (buy_cost + buy_fee)
    return_pct = pnl_ntd / buy_cost if buy_cost > 0 else 0.0

    trades.append({
        "code": position["code"], "side": "long",
        "entry_date": position["entry_date"], "exit_date": date,
        "e_price": e_price, "exit_price": exit_price, "exit_reason": reason,
        "shares": shares, "lots": shares // SHARES_PER_LOT,
        "pnl_ntd": pnl_ntd, "return_pct": return_pct,
        "hold_days": position["hold_days"],
    })


def _process_equity_day(position, row, date, trades, min_hold_days, max_hold_days, slippage_pct: float = 0.0):
    """單一部位單一天的出場判斷。ATR停損隨時可以觸發(即使還沒到min_hold_days——
    停損是風險控管，不是「還沒到最短持有時間所以不管虧多少都不能出場」的規則，
    這兩者是不同目的，優先權停損比較高，見模組docstring)。
    只有「時間到期」(強制出場)才會被min/max_hold_days限制：max_hold_days到了
    一定強制出場；min_hold_days只是min_hold_days本身沒有單獨的出場動作，
    純粹是給呼叫端(進場邏輯或未來的訊號型出場)參考用的下限，v1版本沒有
    「訊號轉弱就提前出場」的機制，出場只有ATR停損+到期兩種，所以min_hold_days
    在目前版本不會產生額外行為，只保留參數位置給未來擴充(見engine模組docstring
    「出場邏輯」段落)。"""
    event, exit_price = _check_equity_exit(row, position, slippage_pct=slippage_pct)
    if event in ("stop", "stop_gap"):
        _close_equity_trade(position, exit_price, event, date, trades)
        return None

    if position["hold_days"] >= max_hold_days:
        _close_equity_trade(position, row["Close"], "time_exit", date, trades)
        return None

    return position


def run_equity_swing_backtest(price_data: dict, indicators_by_code: dict, regime_series: pd.DataFrame,
                               master_calendar: pd.DatetimeIndex, starting_capital: float,
                               min_hold_weeks: int = 2, max_hold_weeks: int = 8,
                               atr_stop_mult: float = 2.0, max_concurrent_positions: int = 5,
                               signal_weights: dict = None, top_n: int = 5,
                               risk_pct_per_trade: float = None, slippage_pct: float = 0.0) -> list:
    """
    完整day-by-day walk-forward模擬，long-only，最多同時持有max_concurrent_positions檔。

    資金模型(v1簡化，誠實列出)：起始資金平分成max_concurrent_positions個「槽位」，
    每個槽位固定分配到 starting_capital / max_concurrent_positions 的現金，槽位
    彼此獨立、不會因為某個槽位賺錢就把獲利挪去放大下一筆進場——這是為了讓
    「同時最多N個部位」這個限制在整個回測期間維持一致的資金紀律，不是模擬
    「總資金池動態重新分配」的複利效果(那樣的話賺錢會越滾越大部位、賠錢會
    越縮越小部位，讓PF這類指標的意義變得跟參數(槽位數/停損寬度)糾纏在一起，
    更難單純比較訊號本身的預測力)。risk_pct_per_trade模式下，槽位現金改成
    這筆交易可動用的「上限」，實際股數由風險預算跟槽位現金上限取較小值決定。

    min_hold_weeks目前只轉換成min_hold_days傳給_process_equity_day()備用(v1
    沒有額外的出場動作依賴它，見_process_equity_day()說明)；max_hold_weeks
    轉換成max_hold_days，到期強制出場(force close)。
    """
    min_hold_days = min_hold_weeks * TRADING_DAYS_PER_WEEK
    max_hold_days = max_hold_weeks * TRADING_DAYS_PER_WEEK
    cash_per_slot = starting_capital / max_concurrent_positions if max_concurrent_positions > 0 else starting_capital

    trades = []
    positions = {}  # code -> position

    for date in master_calendar:
        # 先處理所有已持倉部位的出場判斷
        for code in list(positions.keys()):
            df = price_data.get(code)
            if df is None or date not in df.index:
                continue
            position = positions[code]
            if date != position["entry_date"]:
                position["hold_days"] += 1
            row = df.loc[date]
            updated = _process_equity_day(
                position, row, date, trades, min_hold_days, max_hold_days, slippage_pct=slippage_pct)
            if updated is None:
                del positions[code]
            else:
                positions[code] = updated

        available_slots = max_concurrent_positions - len(positions)
        if available_slots <= 0:
            continue

        regime = compute_regime(regime_series, date)
        excluded_codes = set(positions.keys())
        candidates = scan_equity_candidates(
            indicators_by_code, date, regime, excluded_codes,
            signal_weights=signal_weights, top_n=max(top_n, available_slots),
        )
        if not candidates:
            continue

        for _ in range(available_slots):
            if not candidates:
                break
            new_position = try_enter_equity_position(
                price_data, candidates, date, cash_per_slot,
                atr_stop_mult=atr_stop_mult, risk_pct_per_trade=risk_pct_per_trade,
                account_equity=starting_capital, slippage_pct=slippage_pct,
            )
            if new_position is None:
                break
            code = new_position["code"]
            candidates = [c for c in candidates if c["code"] != code]
            if new_position["entry_date"] in price_data[code].index:
                row = price_data[code].loc[new_position["entry_date"]]
                new_position["hold_days"] = 1
                updated = _process_equity_day(
                    new_position, row, date, trades, min_hold_days, max_hold_days, slippage_pct=slippage_pct)
                if updated is not None:
                    positions[code] = updated
            else:
                positions[code] = new_position

    return trades
