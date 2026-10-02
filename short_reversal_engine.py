"""
短期反轉(Short-Term Reversal)雙向策略回測引擎。

策略假設(誠實揭露：這是外國/其他市場學術文獻支持的假說，還沒在台灣股票期貨上驗證過，
寫這支引擎的目的就是跑一次這個專案自己的IS/OOS + bootstrap驗證，誠實看結果，不是先驗
認定會賺錢)：

學術上「短期反轉(short-term reversal)」效應——每週重新平衡，買進上週表現最差的股票、
放空上週表現最好的股票——在「大型股、流動性佳」的子集合裡，扣掉交易成本後仍然存在
(de Groot/Huij/Zhou 2011；Quantpedia "Short-Term Reversal Effect in Stocks"整理：
限定在大型/高流動性股票時，淨成本後大約每週30~50bp，換到小型股/低流動性股票時，
週轉成本會把這個效應吃光甚至轉負)。

台灣股票期貨的標的池(taifex_universe.STOCK_FUTURES_UNIVERSE)剛好是這個效應需要的
「流動性子集合」：期交所本來就只替「夠流動」的標的掛牌股票期貨，等於已經先篩過一輪
流動性，不是隨便抓全市場1700多檔小型股。而且跟現股(需要融券才能放空，融券額度有限、
券源可能軋空、成本比做多高很多)不一樣，股票期貨放空跟做多一樣容易(同樣的保證金機制、
同樣的進出場流程)，才有辦法真的做「買最爛的、空最強的」這種真正的多空雙向配對，
不是只能做多方的半套實作。

這支引擎重用 mean_reversion_engine.py 共用的出場/停損/保證金查表機制(check_exit、
_process_mr_day、DEFAULT_MARGIN_CAP_RATIO)，進場端的候選排名重用
overnight_momentum_engine.percentile_score()的相對排名寫法(跟momentum_breakout_engine.py
一致：相對排名比絕對值門檻在標的池大小變動時更穩定，見momentum_breakout_engine.py模組
docstring的「軟性排名」討論)，不重新發明這幾個已經有、而且已經被其他引擎驗證過寫法的機制。

核心訊號：以掃描日為基準，用「嚴格早於掃描日」的資料(_lookup_prior_row，跟這個repo
其他引擎一致的防止偷看未來資料的寫法)算出每檔股票的「前N個交易日收盤對收盤報酬率」，
N(lookback_window)是可比較的參數(3/5/10日)，不是寫死單一值——見compare_short_reversal.py
的「階段1：回看窗口比較」，程式會實測哪個窗口比較好，不是先驗認定5日(文獻裡常見的
「一週」對應值)就是最佳選擇。

多方候選 = 這批報酬率裡最低(最差)的百分位排名；空方候選 = 最高(最強)的百分位排名——
用percentile_score()相對排名，不是用一個固定的絕對報酬率門檻去切，理由跟
momentum_breakout_engine.py的「軟性排名」討論一樣：絕對值門檻對標的池組成/期間長度
很敏感，相對排名比較穩定。

進場前的流動性/結構性門檻：當天成交金額(近似值，收盤價x成交量)至少
overnight_momentum_engine.MIN_TURNOVER_VALUE(2000萬新台幣)，重用既有引擎已經在用、
已經有名字的常數，不是另外發明一個新的magic number。

出場機制(這是這支引擎真正要測試的核心)：固定持有天數(3/5/10個交易日，對應文獻的
「一週後平倉重新平衡」基準，一樣是可比較的參數，不是寫死)當主要出場依據，底下另外疊一層
ATR保護性停損當安全網——跟這個repo其他引擎(動能突破的atr_stop_mult、均值回歸的
atr保護)一樣的雙重機制，不是只靠時間出場、完全不設停損風控。做法是把target_price設成
None(不設固定停利價，重用mean_reversion_engine.check_exit()既有的「target_price為None
代表不看固定停利」邏輯)，讓_process_mr_day()裡的「hold_days >= max_hold_days就強制平倉」
機制天然變成主要出場依據，ATR停損(stop_price)仍然正常生效、可能提早觸發。

⚠️ 跟其他引擎一樣的已知簡化：沒有大盤氛圍濾網(regime)、沒有除權息清洗——這個策略假設
本身是「橫斷面相對排名」，不是「順勢/逆勢」，加大盤氛圍濾網會引入「要不要在多頭市場停用
空方訊號」這種跟動能/均值回歸引擎不同性質的假設，這一輪先不加，留給之後驗證是否有必要。
"""
import pandas as pd
import numpy as np

from taifex_universe import estimate_margin
from mean_reversion_engine import (
    compute_atr_correct, is_near_settlement, _lookup_prior_row, _process_mr_day,
    summarize_mr, DEFAULT_MARGIN_CAP_RATIO,
)
from overnight_momentum_engine import percentile_score, MIN_TURNOVER_VALUE

# 回看窗口/固定持有天數：文獻的「一週重新平衡」基準是5個交易日，這裡額外加3日(更短、更貼近
# 雜訊)跟10日(兩週，更平滑)當比較對象，見compare_short_reversal.py階段1/階段2。
LOOKBACK_WINDOW_OPTIONS = (3, 5, 10)
HOLD_DAYS_OPTIONS = (3, 5, 10)

DEFAULT_TOP_N = 5
DEFAULT_ATR_STOP_MULT = 2.0  # 安全網停損，刻意設得比突破引擎的1.0寬，因為這裡出場主力是固定天數，
# ATR停損只是防止「持有天數還沒到、但已經大幅不利」時繼續扛著不管的保護機制，不是主要出場依據。


def precompute_reversal_indicators(df: pd.DataFrame, lookback_windows=LOOKBACK_WINDOW_OPTIONS) -> pd.DataFrame:
    """
    針對一檔股票的完整價格序列，一次算好短期反轉策略需要的欄位：ATR(安全網停損用)、
    成交金額(流動性門檻用)、以及每個回看窗口各自的N日收盤對收盤報酬率。跟這個repo其他
    引擎(precompute_indicators/precompute_breakout_indicators)同樣精神：day-by-day
    迴圈之後只是查表，不在迴圈裡重複計算。

    Return_{N} 用 close.pct_change(N)：第t列的值 = (close[t] - close[t-N]) / close[t-N]，
    只用到t跟t-N(含)之間、完全是「已經發生」的資料，不需要額外shift——配合
    _lookup_prior_row()找「嚴格早於掃描日」的那一列，兩層加起來才是完整的防止偷看未來
    資料保證：縱使掃描日當天這列已經不會被看到，查到的前一列本身的報酬率欄位也只用
    它自己那天為止的資料算出來的。
    """
    close = df["Close"]
    volume = df["Volume"]
    atr = compute_atr_correct(df)
    turnover_value = close * volume

    cols = {"Close": close, "ATR": atr, "TurnoverValue": turnover_value}
    for w in lookback_windows:
        cols[f"Return_{w}"] = close.pct_change(w)
    return pd.DataFrame(cols, index=df.index)


def precompute_all_reversal_indicators(price_data: dict, universe: dict,
                                        lookback_windows=LOOKBACK_WINDOW_OPTIONS) -> dict:
    result = {}
    for code in universe:
        df = price_data.get(code)
        if df is None:
            continue
        result[code] = precompute_reversal_indicators(df, lookback_windows=lookback_windows)
    return result


def scan_short_reversal_candidates(indicators_by_code: dict, as_of_date, excluded_codes: set,
                                    lookback_window: int = 5, top_n: int = DEFAULT_TOP_N,
                                    min_turnover_value: float = MIN_TURNOVER_VALUE,
                                    allow_short: bool = True) -> list:
    """
    掃描全市場候選標的：先套流動性門檻(成交金額 >= min_turnover_value)，通過門檻的
    候選再依「前lookback_window日報酬率」的相對百分位排名——多方取報酬率最低的前top_n名
    (買最爛的，賭均值回歸)、空方取報酬率最高的前top_n名(空最強的，賭漲多拉回)。

    allow_short=False時不產生空方候選(例如現股版本沒辦法真正放空時可以關掉，雖然這個
    repo的TAIFEX股票期貨本身多空一樣容易，保留這個開關純粹是跟其他引擎的allow_short
    參數介面一致，方便CLI做純多方 vs 多空雙向的比較)。
    """
    return_col = f"Return_{lookback_window}"
    rows = []
    for code, ind_df in indicators_by_code.items():
        if code in excluded_codes:
            continue
        if return_col not in ind_df.columns:
            raise KeyError(
                f"{return_col} 不在指標欄位裡，precompute_reversal_indicators()的"
                f"lookback_windows需要包含 {lookback_window}"
            )
        row, pos = _lookup_prior_row(ind_df, as_of_date)
        # pos < lookback_window+1：報酬率本身要有足夠的歷史才算得出來；
        # 另外要求至少20天暖身，跟這個repo其他引擎的"pos < 60"精神一致(但這裡用到的指標
        # 比較單純，不需要60天的均線/ATR暖身，20天已經足夠涵蓋最長的10日回看窗口+安全邊際)。
        if row is None or pos < max(lookback_window + 1, 20):
            continue

        ret = row[return_col]
        atr = row["ATR"]
        turnover = row["TurnoverValue"]
        close = row["Close"]
        if pd.isna(ret) or pd.isna(atr) or atr <= 0 or pd.isna(turnover) or pd.isna(close):
            continue
        if turnover < min_turnover_value:
            continue

        rows.append({"code": code, "close": close, "atr": atr, "ret": ret})

    if not rows:
        return []

    df = pd.DataFrame(rows)
    # 多方：報酬率越低分數越高(higher_is_better=False)；空方：報酬率越高分數越高。
    # 跟momentum_breakout_engine._rank_candidates()同樣的percentile_score()相對排名寫法。
    df["score_long"] = percentile_score(df["ret"], higher_is_better=False)
    df["score_short"] = percentile_score(df["ret"], higher_is_better=True)

    candidates = []
    long_df = df.sort_values("score_long", ascending=False).head(top_n)
    for _, r in long_df.iterrows():
        candidates.append({
            "code": r["code"], "side": "long", "score": float(r["score_long"]),
            "c_prev": float(r["close"]), "atr": float(r["atr"]),
        })

    if allow_short:
        short_df = df.sort_values("score_short", ascending=False).head(top_n)
        for _, r in short_df.iterrows():
            candidates.append({
                "code": r["code"], "side": "short", "score": float(r["score_short"]),
                "c_prev": float(r["close"]), "atr": float(r["atr"]),
            })

    return candidates


def try_enter_short_reversal(price_data: dict, candidates: list, entry_date, starting_capital: float,
                              lots: int = 2, atr_stop_mult: float = DEFAULT_ATR_STOP_MULT,
                              slippage_pct: float = 0.0, used_margin: float = 0.0,
                              total_margin_cap_ratio: float = None,
                              max_gap_pct: float = None):
    """
    依序檢查候選名單(已跳空風控+保證金上限過濾)，第一個通過的進場。
    跟mean_reversion_engine.try_enter_mean_reversion()/momentum_breakout_engine.
    try_enter_breakout()同樣的跳空風控方向：不管多空，方向不利的跳空超過0.5%就放棄。

    target_price固定給None——這支引擎的主要出場依據是固定持有天數(由呼叫端的
    max_hold_days控制，實際生效在mean_reversion_engine._process_mr_day()的
    「hold_days >= max_hold_days就強制平倉」判斷)，ATR停損(stop_price)只是安全網，
    不是「碰到目標價就跑」的均值回歸式出場。
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

        margin_needed = estimate_margin(code, open_p, lots)
        if margin_needed > starting_capital * DEFAULT_MARGIN_CAP_RATIO:
            continue
        if total_margin_cap_ratio is not None and \
                used_margin + margin_needed > starting_capital * total_margin_cap_ratio:
            continue

        if side == "long":
            e_price = open_p * (1 + slippage_pct)
            stop_price = e_price - atr_stop_mult * atr
        else:
            e_price = open_p * (1 - slippage_pct)
            stop_price = e_price + atr_stop_mult * atr

        return {
            "code": code, "side": side, "entry_date": entry_date,
            "e_price": e_price, "target_price": None, "stop_price": stop_price,
            "lots": lots, "hold_days": 1, "margin_used": margin_needed,
        }
    return None


def run_short_reversal_backtest(price_data: dict, indicators_by_code: dict,
                                 master_calendar: pd.DatetimeIndex, max_hold_days: int,
                                 starting_capital: float, lookback_window: int = 5,
                                 allow_short: bool = True, lots: int = 2,
                                 atr_stop_mult: float = DEFAULT_ATR_STOP_MULT,
                                 top_n: int = DEFAULT_TOP_N,
                                 min_turnover_value: float = MIN_TURNOVER_VALUE,
                                 slippage_pct: float = 0.0,
                                 max_concurrent_positions: int = 2 * DEFAULT_TOP_N,
                                 total_margin_cap_ratio: float = None,
                                 max_gap_pct: float = None):
    """
    完整day-by-day walk-forward模擬。出場判定/強制平倉/停損冷卻期，重用
    mean_reversion_engine._process_mr_day()，跟這個repo其他引擎共用同一套出場機制。

    跟mean_reversion_engine.run_mean_reversion_backtest()(一次只能一個部位)不同，
    這裡重用momentum_breakout_engine.run_momentum_breakout_backtest()的多部位並行
    迴圈結構(max_concurrent_positions)——因為短期反轉本質上是「橫斷面相對排名」策略，
    理論上就是要同時買一籃子最爛的、空一籃子最強的，不是找到一個候選就全押，單一部位的
    版本沒辦法呈現這個策略真正的設計精神。預設max_concurrent_positions=2*top_n，
    剛好可以同時容納多空雙向各top_n名。

    max_hold_days：固定持有天數(主要出場依據，見scan_short_reversal_candidates()/
    try_enter_short_reversal()的docstring說明)，不是「工程上的安全上限」，是這支引擎
    真正要測試的核心參數，由compare_short_reversal.py的階段2決定要用3/5/10天哪一個。
    """
    trades = []
    cooldown_until = {}
    open_positions = []

    effective_total_margin_cap_ratio = total_margin_cap_ratio
    if max_concurrent_positions > 1 and effective_total_margin_cap_ratio is None:
        effective_total_margin_cap_ratio = min(DEFAULT_MARGIN_CAP_RATIO * max_concurrent_positions, 0.9)

    for date in master_calendar:
        # 1) 先處理所有既有部位的出場判定/強制平倉(可能更新cooldown_until)
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

        # 2) 補進新部位，填滿空出來的名額
        held_codes = {p["code"] for p in open_positions}
        excluded_codes = {c for c, until in cooldown_until.items() if date < until} | held_codes

        slots_available = max_concurrent_positions - len(open_positions)
        if slots_available > 0 and not is_near_settlement(date, days_before=2):
            used_margin = sum(p["margin_used"] for p in open_positions)

            candidates = scan_short_reversal_candidates(
                indicators_by_code, date, excluded_codes, lookback_window=lookback_window,
                top_n=max(top_n, slots_available), min_turnover_value=min_turnover_value,
                allow_short=allow_short,
            )

            while slots_available > 0 and candidates:
                new_position = try_enter_short_reversal(
                    price_data, candidates, date, starting_capital, lots,
                    atr_stop_mult=atr_stop_mult, slippage_pct=slippage_pct,
                    used_margin=used_margin, total_margin_cap_ratio=effective_total_margin_cap_ratio,
                    max_gap_pct=max_gap_pct,
                )
                if new_position is None:
                    break

                used_margin += new_position["margin_used"]
                slots_available -= 1
                candidates = [c for c in candidates if c["code"] != new_position["code"]]

                # 進場當天立刻檢查一次出場(跟其他引擎一致：跳空穿越停損可能當天就出場)
                row = price_data[new_position["code"]].loc[date]
                updated = _process_mr_day(new_position, row, date, trades, max_hold_days, cooldown_until,
                                           slippage_pct=slippage_pct)
                if updated is not None:
                    open_positions.append(updated)
                else:
                    used_margin -= new_position["margin_used"]

    return trades
