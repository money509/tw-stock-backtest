"""
test_short_reversal_engine.py
===============================
short_reversal_engine.py的單元測試。全程用合成的小DataFrame，不碰網路(沒有任何
data_loader/yfinance呼叫)。涵蓋：回看窗口報酬率排名正確性(含不偷看未來資料)、
多方/空方percentile排名方向、流動性門檻過濾、放空的損益正負號是否正確、固定持有天數
出場是否在正確的那天觸發、ATR安全網停損仍然正常運作。
"""
import numpy as np
import pandas as pd
import pytest

from short_reversal_engine import (
    precompute_reversal_indicators, precompute_all_reversal_indicators,
    scan_short_reversal_candidates, try_enter_short_reversal, run_short_reversal_backtest,
)
from mean_reversion_engine import summarize_mr


def _make_price_df(closes, start="2024-01-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="B")
    closes = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
        "Volume": pd.Series(2_000_000.0, index=idx),  # close*volume遠大於MIN_TURNOVER_VALUE(2000萬)
    }, index=idx)


class TestPrecomputeReversalIndicators:
    def test_return_columns_present_for_each_lookback_window(self):
        df = _make_price_df([100 + i for i in range(30)])
        ind = precompute_reversal_indicators(df, lookback_windows=(3, 5, 10))
        assert {"Return_3", "Return_5", "Return_10", "ATR", "TurnoverValue", "Close"}.issubset(ind.columns)

    def test_return_n_matches_pct_change_definition(self):
        closes = [100, 102, 104, 103, 108, 110]
        df = _make_price_df(closes)
        ind = precompute_reversal_indicators(df, lookback_windows=(3,))
        # 第5列(index 4, 0-based)：(108-102)/102，驗證跟手動算的pct_change(3)一致
        expected = (closes[4] - closes[1]) / closes[1]
        assert ind["Return_3"].iloc[4] == pytest.approx(expected)

    def test_return_does_not_use_future_data(self):
        """驗證第t列的Return_N只用t跟t-N(含)之間的資料：把t之後的收盤價改掉，
        不該影響t列本身算出來的報酬率(沒有偷看未來)。"""
        closes_a = [100, 102, 104, 103, 108, 110, 50]
        closes_b = [100, 102, 104, 103, 108, 110, 999]  # 只改最後一天(未來)
        df_a = _make_price_df(closes_a)
        df_b = _make_price_df(closes_b)
        ind_a = precompute_reversal_indicators(df_a, lookback_windows=(3,))
        ind_b = precompute_reversal_indicators(df_b, lookback_windows=(3,))
        # 第5列(index 4)之前的所有Return_3都不該因為第7天(index 6)的資料被改而變動
        assert ind_a["Return_3"].iloc[:5].equals(ind_b["Return_3"].iloc[:5])


class TestScanShortReversalCandidates:
    def _build_universe(self, returns_by_code, lookback_window=5, warmup=25):
        """每檔股票走勢在暖身期(warmup天)都是平的，最後一段依returns_by_code指定的
        lookback_window日報酬率往上/下走，方便精確控制排名用的報酬率數值。"""
        universe, price_data = {}, {}
        for code, ret in returns_by_code.items():
            flat = [100.0] * warmup
            # 最後 lookback_window+1 天：從100走到 100*(1+ret)，製造出指定的N日報酬率
            tail = list(np.linspace(100.0, 100.0 * (1 + ret), lookback_window + 1))
            closes = flat + tail[1:]
            price_data[code] = _make_price_df(closes)
            universe[code] = {}
        indicators = precompute_all_reversal_indicators(price_data, universe, lookback_windows=(lookback_window,))
        as_of_date = price_data[next(iter(universe))].index[-1]
        return indicators, as_of_date

    def test_long_candidates_are_worst_performers(self):
        returns_by_code = {"1101": -0.10, "1102": -0.02, "1210": 0.05, "1216": 0.10}
        indicators, as_of_date = self._build_universe(returns_by_code, lookback_window=5)
        candidates = scan_short_reversal_candidates(
            indicators, as_of_date, excluded_codes=set(), lookback_window=5, top_n=2, allow_short=False,
        )
        long_codes = {c["code"] for c in candidates if c["side"] == "long"}
        assert long_codes == {"1101", "1102"}  # 報酬率最低的兩檔

    def test_short_candidates_are_best_performers(self):
        returns_by_code = {"1101": -0.10, "1102": -0.02, "1210": 0.05, "1216": 0.10}
        indicators, as_of_date = self._build_universe(returns_by_code, lookback_window=5)
        candidates = scan_short_reversal_candidates(
            indicators, as_of_date, excluded_codes=set(), lookback_window=5, top_n=2, allow_short=True,
        )
        short_codes = {c["code"] for c in candidates if c["side"] == "short"}
        assert short_codes == {"1210", "1216"}  # 報酬率最高的兩檔

    def test_allow_short_false_produces_no_short_candidates(self):
        returns_by_code = {"1101": -0.10, "1102": 0.10}
        indicators, as_of_date = self._build_universe(returns_by_code, lookback_window=5)
        candidates = scan_short_reversal_candidates(
            indicators, as_of_date, excluded_codes=set(), lookback_window=5, top_n=2, allow_short=False,
        )
        assert all(c["side"] == "long" for c in candidates)

    def test_excluded_codes_are_skipped(self):
        returns_by_code = {"1101": -0.10, "1102": -0.02}
        indicators, as_of_date = self._build_universe(returns_by_code, lookback_window=5)
        candidates = scan_short_reversal_candidates(
            indicators, as_of_date, excluded_codes={"1101"}, lookback_window=5, top_n=2, allow_short=False,
        )
        assert all(c["code"] != "1101" for c in candidates)

    def test_liquidity_gate_filters_low_turnover_stock(self):
        # 兩檔股票報酬率相同(都是最爛的)，但其中一檔成交金額(close*volume)低於門檻，應該被濾掉
        universe, price_data = {}, {}
        closes = [100.0] * 25 + list(np.linspace(100.0, 90.0, 6))[1:]
        price_data["1101"] = _make_price_df(closes)  # 預設Volume=2,000,000，成交金額夠高
        low_liquidity_df = _make_price_df(closes)
        low_liquidity_df["Volume"] = 100.0  # 成交金額遠低於MIN_TURNOVER_VALUE
        price_data["1102"] = low_liquidity_df
        universe = {"1101": {}, "1102": {}}
        indicators = precompute_all_reversal_indicators(price_data, universe, lookback_windows=(5,))
        as_of_date = price_data["1101"].index[-1]

        candidates = scan_short_reversal_candidates(
            indicators, as_of_date, excluded_codes=set(), lookback_window=5, top_n=5, allow_short=False,
        )
        codes = {c["code"] for c in candidates}
        assert "1102" not in codes
        assert "1101" in codes

    def test_unknown_lookback_window_raises_key_error(self):
        returns_by_code = {"1101": -0.05}
        indicators, as_of_date = self._build_universe(returns_by_code, lookback_window=5)
        with pytest.raises(KeyError):
            scan_short_reversal_candidates(
                indicators, as_of_date, excluded_codes=set(), lookback_window=20, allow_short=False,
            )


class TestTryEnterShortReversal:
    def _candidate(self, side="long", c_prev=100.0, atr=2.0, code="1101"):
        return {"code": code, "side": side, "score": 90.0, "c_prev": c_prev, "atr": atr}

    def test_long_stop_price_is_below_entry(self):
        df = _make_price_df([100.0, 101.0])
        price_data = {"1101": df}
        candidates = [self._candidate(side="long")]
        pos = try_enter_short_reversal(price_data, candidates, df.index[1], starting_capital=1_000_000,
                                        lots=2, atr_stop_mult=2.0)
        assert pos is not None
        assert pos["side"] == "long"
        assert pos["stop_price"] < pos["e_price"]
        assert pos["target_price"] is None  # 主要出場依據是固定持有天數，不是固定停利價

    def test_short_stop_price_is_above_entry(self):
        df = _make_price_df([100.0, 100.2])  # 小跳空(<0.5%)，不會被空方不利跳空風控擋掉
        price_data = {"1101": df}
        candidates = [self._candidate(side="short", c_prev=100.0)]
        pos = try_enter_short_reversal(price_data, candidates, df.index[1], starting_capital=1_000_000,
                                        lots=2, atr_stop_mult=2.0)
        assert pos is not None
        assert pos["side"] == "short"
        assert pos["stop_price"] > pos["e_price"]

    def test_long_blocked_by_unfavorable_gap_down(self):
        df = _make_price_df([100.0, 100.0])
        df.loc[df.index[1], "Open"] = 94.0  # 跳空跌破0.5%風控
        price_data = {"1101": df}
        candidates = [self._candidate(side="long", c_prev=100.0)]
        pos = try_enter_short_reversal(price_data, candidates, df.index[1], starting_capital=1_000_000)
        assert pos is None

    def test_short_blocked_by_unfavorable_gap_up(self):
        df = _make_price_df([100.0, 100.0])
        df.loc[df.index[1], "Open"] = 106.0  # 跳空漲破0.5%風控(對空方不利)
        price_data = {"1101": df}
        candidates = [self._candidate(side="short", c_prev=100.0)]
        pos = try_enter_short_reversal(price_data, candidates, df.index[1], starting_capital=1_000_000)
        assert pos is None

    def test_margin_cap_skips_to_next_candidate(self):
        df = _make_price_df([100.0, 101.0])
        price_data = {"1101": df, "1102": df.copy()}
        candidates = [
            self._candidate(side="long", c_prev=100.0, code="1101"),
            self._candidate(side="long", c_prev=100.0, code="1102"),
        ]
        # starting_capital小到第一筆保證金就超過35%上限，應該被跳過改選第二筆
        pos = try_enter_short_reversal(price_data, candidates, df.index[1], starting_capital=1.0, lots=2)
        assert pos is None  # 兩檔都因為保證金超過上限被跳過

    def test_slippage_makes_long_entry_price_worse(self):
        df = _make_price_df([100.0, 101.0])
        price_data = {"1101": df}
        candidates = [self._candidate(side="long", c_prev=100.0)]
        pos_no_slip = try_enter_short_reversal(price_data, candidates, df.index[1], starting_capital=1_000_000,
                                                slippage_pct=0.0)
        pos_slip = try_enter_short_reversal(price_data, candidates, df.index[1], starting_capital=1_000_000,
                                             slippage_pct=0.01)
        assert pos_slip["e_price"] > pos_no_slip["e_price"]


class TestRunShortReversalBacktestExitTiming:
    def _flat_then_drop_universe(self, n_tail_days=20):
        """造一檔「暖身期平盤、之後急跌」的股票當多方候選，另一檔「暖身期平盤、之後急漲」
        的股票當空方候選，確保掃描日一定會選到它們，之後專心檢查出場時機/損益正負號。"""
        warmup = 25
        long_closes = [100.0] * warmup + list(np.linspace(100.0, 80.0, 6))[1:] + [80.0] * n_tail_days
        short_closes = [100.0] * warmup + list(np.linspace(100.0, 125.0, 6))[1:] + [125.0] * n_tail_days
        price_data = {
            "1101": _make_price_df(long_closes),
            "1102": _make_price_df(short_closes),
        }
        universe = {"1101": {}, "1102": {}}
        indicators = precompute_all_reversal_indicators(price_data, universe, lookback_windows=(5,))
        return price_data, indicators

    def test_fixed_hold_days_forces_exit_on_the_right_day(self):
        price_data, indicators = self._flat_then_drop_universe()
        master_calendar = price_data["1101"].index[30:]  # 從掃描日之後開始跑
        max_hold_days = 3
        trades = run_short_reversal_backtest(
            price_data, indicators, master_calendar, max_hold_days=max_hold_days,
            starting_capital=1_000_000, lookback_window=5, allow_short=True,
            lots=2, atr_stop_mult=100.0,  # ATR停損設得極寬，確保不會提早被停損出場，只測固定天數出場
            top_n=2, max_concurrent_positions=4,
        )
        assert len(trades) > 0
        for t in trades:
            assert t["hold_days"] <= max_hold_days
        # 至少有一筆是因為撐滿天數被強制平倉出場(不是中途被停損/跳空停損出場)
        assert any(t["exit_reason"] == "forced_close" for t in trades)

    def test_short_side_profits_when_price_falls(self):
        """放空的股票之後持續下跌(在固定出場視窗內)，空單應該是賺錢，驗證放空的損益
        正負號沒有寫反——這是題目特別提醒容易出錯的地方。"""
        warmup = 25
        # 空方候選：暖身期平盤後急漲(被選中)，選中後(出場視窗內)又續跌，讓空單真的賺錢
        closes = [100.0] * warmup + list(np.linspace(100.0, 125.0, 6))[1:] + list(np.linspace(125.0, 90.0, 10))
        other_long_closes = [100.0] * warmup + list(np.linspace(100.0, 80.0, 6))[1:] + [80.0] * 10
        price_data = {"1101": _make_price_df(closes), "1102": _make_price_df(other_long_closes)}
        universe = {"1101": {}, "1102": {}}
        indicators = precompute_all_reversal_indicators(price_data, universe, lookback_windows=(5,))
        master_calendar = price_data["1101"].index[30:]

        trades = run_short_reversal_backtest(
            price_data, indicators, master_calendar, max_hold_days=5,
            starting_capital=1_000_000, lookback_window=5, allow_short=True,
            lots=2, atr_stop_mult=100.0, top_n=2, max_concurrent_positions=4,
        )
        short_trades = [t for t in trades if t["code"] == "1101" and t["side"] == "short"]
        assert len(short_trades) > 0
        for t in short_trades:
            # exit_price應該低於e_price(股價下跌)，空單的pnl應該是正的
            assert t["exit_price"] < t["e_price"]
            assert t["pnl_ntd"] > 0

    def test_long_side_loses_when_price_keeps_falling(self):
        """多方候選(被選中時報酬率最差)如果之後繼續跌，多單應該賠錢，同樣驗證
        做多方向的損益正負號正確(跟放空互為對照)。"""
        warmup = 25
        closes = [100.0] * warmup + list(np.linspace(100.0, 80.0, 6))[1:] + list(np.linspace(80.0, 60.0, 10))
        other_short_closes = [100.0] * warmup + list(np.linspace(100.0, 125.0, 6))[1:] + [125.0] * 10
        price_data = {"1101": _make_price_df(closes), "1102": _make_price_df(other_short_closes)}
        universe = {"1101": {}, "1102": {}}
        indicators = precompute_all_reversal_indicators(price_data, universe, lookback_windows=(5,))
        master_calendar = price_data["1101"].index[30:]

        trades = run_short_reversal_backtest(
            price_data, indicators, master_calendar, max_hold_days=5,
            starting_capital=1_000_000, lookback_window=5, allow_short=True,
            lots=2, atr_stop_mult=100.0, top_n=2, max_concurrent_positions=4,
        )
        long_trades = [t for t in trades if t["code"] == "1101" and t["side"] == "long"]
        assert len(long_trades) > 0
        for t in long_trades:
            assert t["exit_price"] < t["e_price"]
            assert t["pnl_ntd"] < 0

    def test_atr_safety_stop_exits_before_fixed_hold_days(self):
        """ATR停損設得很緊(0.1倍)，多方候選進場後立刻大跌，應該在撐滿max_hold_days之前
        就被停損出場，驗證ATR安全網仍然正常運作，不是完全被固定天數出場取代掉。"""
        warmup = 25
        # 暖身期平盤 -> 緩跌到80(被掃描選中) -> 進場當天開盤跟前一天收盤價差不多(不觸發
        # 跳空風控) -> 進場後才開始真正暴跌到50，製造「進場後才被ATR安全網停損」的情境
        closes = [100.0] * warmup + list(np.linspace(100.0, 80.0, 6))[1:] + [80.0] + \
            list(np.linspace(80.0, 50.0, 9))
        other_closes = [100.0] * warmup + list(np.linspace(100.0, 120.0, 6))[1:] + [120.0] * 10
        price_data = {"1101": _make_price_df(closes), "1102": _make_price_df(other_closes)}
        universe = {"1101": {}, "1102": {}}
        indicators = precompute_all_reversal_indicators(price_data, universe, lookback_windows=(5,))
        master_calendar = price_data["1101"].index[30:]

        trades = run_short_reversal_backtest(
            price_data, indicators, master_calendar, max_hold_days=10,
            starting_capital=1_000_000, lookback_window=5, allow_short=False,
            lots=2, atr_stop_mult=0.1,  # 極緊的ATR停損
            top_n=2, max_concurrent_positions=4,
        )
        long_trades_1101 = [t for t in trades if t["code"] == "1101"]
        assert len(long_trades_1101) > 0
        stop_trades = [t for t in long_trades_1101 if t["exit_reason"] in ("stop", "stop_gap")]
        assert len(stop_trades) > 0
        for t in stop_trades:
            assert t["hold_days"] < 10  # 比固定天數更早出場


class TestRunShortReversalBacktestSmoke:
    def test_runs_without_error_and_produces_summarizable_trades(self):
        np.random.seed(11)
        n = 120
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        price_data, universe = {}, {}
        for i, code in enumerate(["1101", "1102", "1210", "1216", "1301"]):
            base = 80 + i * 10
            noise = np.random.normal(0, 1.5, n)
            closes = np.maximum(base + np.cumsum(noise * 0.3), 1.0)
            df = pd.DataFrame({
                "Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
                "Volume": pd.Series(2_000_000.0, index=idx),
            }, index=idx)
            price_data[code] = df
            universe[code] = {}

        indicators = precompute_all_reversal_indicators(price_data, universe)
        trades = run_short_reversal_backtest(
            price_data, indicators, idx, max_hold_days=5, starting_capital=1_000_000,
            lookback_window=5, allow_short=True, lots=2, top_n=2, max_concurrent_positions=4,
        )
        stats = summarize_mr(trades, 1_000_000)
        assert "trade_count" in stats
        assert "profit_factor" in stats
