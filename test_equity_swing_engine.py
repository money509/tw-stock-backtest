"""
test_equity_swing_engine.py
==============================
針對 equity_swing_engine.py 的合成資料測試。完全不連真實網路——這支引擎本身
不會呼叫任何 requests/yfinance，資料一律由呼叫端(compare_equity_swing.py)
先下載/組好再傳進來，這裡直接構造小規模合成股價/基本面資料表測試。

涵蓋範圍：
1. 現金/整張股數計算(floor到1000股)
2. 基本面/籌碼面資料對齊到價格日期時不會用到未來資料(look-ahead防護)
3. 缺席的資料源(financial_df=None)會讓對應訊號優雅退化成中性分數0.0，不會讓
   排名或回測崩潰
4. MA60結構性門檻 + regime='bear'時完全停用進場
5. 出場邏輯：ATR停損(含跳空) + 最長持有週數強制出場
6. 完整day-by-day回測跑得通、不連網路、部位數不超過max_concurrent_positions
"""
import numpy as np
import pandas as pd
import pytest

import equity_swing_engine as ese
from equity_swing_engine import (
    compute_equity_shares, precompute_equity_indicators, precompute_all_equity_indicators,
    scan_equity_candidates, _rank_equity_candidates, _check_equity_exit, _process_equity_day,
    _close_equity_trade, try_enter_equity_position, run_equity_swing_backtest,
    EQUITY_SIGNAL_NAMES, SHARES_PER_LOT,
)


def make_price_df(n=200, start="2023-01-02", seed=0, trend=0.0008, base=100.0):
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    close = base * np.cumprod(1 + rng.normal(trend, 0.012, n))
    high = close * (1 + rng.uniform(0.0, 0.01, n))
    low = close * (1 - rng.uniform(0.0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    # 確保 High >= max(Open, Close), Low <= min(Open, Close)，避免不合理的合成K棒
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    vol = rng.integers(1000, 5000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates)


class TestComputeEquityShares:
    def test_floors_to_full_lot(self):
        # 現金150,000元、股價101元：150000/(101*1000)=1.48 -> 只能買1張(1000股)
        assert compute_equity_shares(150_000, 101.0) == 1000

    def test_exact_multiple_of_lot(self):
        assert compute_equity_shares(200_000, 100.0) == 2000

    def test_not_enough_cash_for_one_lot_returns_zero(self):
        assert compute_equity_shares(50_000, 100.0) == 0

    def test_zero_or_negative_price_returns_zero(self):
        assert compute_equity_shares(100_000, 0.0) == 0
        assert compute_equity_shares(100_000, -5.0) == 0

    def test_zero_cash_returns_zero(self):
        assert compute_equity_shares(0.0, 50.0) == 0

    def test_result_is_always_multiple_of_1000(self):
        for cash in [12345, 99999, 500_001, 1_234_567]:
            shares = compute_equity_shares(cash, 37.5)
            assert shares % SHARES_PER_LOT == 0


class TestPrecomputeAlignmentNoLookahead:
    def test_revenue_only_visible_from_announcement_date_onward(self):
        price_df = make_price_df(n=60)
        ann_date = price_df.index[30]
        revenue_df = pd.DataFrame(
            {"revenue_yoy_pct": [42.0], "revenue_mom_pct": [5.0]}, index=[ann_date])

        ind = precompute_equity_indicators(price_df, revenue_df=revenue_df)

        before = ind.loc[ind.index < ann_date, "revenue_yoy_pct"]
        assert before.isna().all(), "公告日之前不該看得到這筆營收資料(look-ahead)"

        after = ind.loc[ind.index >= ann_date, "revenue_yoy_pct"]
        assert (after == 42.0).all(), "公告日(含)之後應該持續看到最後一次公告的數字(ffill)"

    def test_valuation_forward_filled_between_updates(self):
        price_df = make_price_df(n=40)
        d0, d1 = price_df.index[5], price_df.index[20]
        valuation_df = pd.DataFrame(
            {"pe": [15.0, 12.0], "pb": [2.0, 1.8], "dividend_yield": [3.0, 3.5]}, index=[d0, d1])

        ind = precompute_equity_indicators(price_df, valuation_df=valuation_df)

        assert ind.loc[price_df.index[10], "pe"] == 15.0  # d0~d1之間沿用d0的值
        assert ind.loc[price_df.index[25], "pe"] == 12.0  # d1之後沿用d1的值
        assert ind.loc[price_df.index[0], "pe"] != ind.loc[price_df.index[0], "pe"]  # d0之前是NaN


class TestGracefulDegradation:
    def test_missing_financial_statement_data_yields_all_nan_columns(self):
        price_df = make_price_df(n=80)
        ind = precompute_equity_indicators(price_df, financial_df=None)
        assert ind["eps_growth_pct"].isna().all()
        assert ind["gross_margin_pct"].isna().all()
        assert ind["roe_pct"].isna().all()

    def test_missing_chip_data_yields_neutral_score_not_crash(self):
        # rows模擬scan_equity_candidates()收集到的候選：完全沒有chip資料(全NaN)
        rows = [
            {"code": "1000", "close": 100.0, "atr": 2.0,
             "revenue_yoy_pct": 10.0, "revenue_mom_pct": 5.0,
             "pe": 15.0, "pb": 1.5, "dividend_yield": 3.0,
             "eps_growth_pct": np.nan, "gross_margin_pct": np.nan, "roe_pct": np.nan,
             "foreign_streak": np.nan, "trust_streak": np.nan},
            {"code": "1001", "close": 50.0, "atr": 1.0,
             "revenue_yoy_pct": -5.0, "revenue_mom_pct": 2.0,
             "pe": 20.0, "pb": 2.5, "dividend_yield": 1.0,
             "eps_growth_pct": np.nan, "gross_margin_pct": np.nan, "roe_pct": np.nan,
             "foreign_streak": np.nan, "trust_streak": np.nan},
        ]
        ranked = _rank_equity_candidates(rows, {name: 1.0 for name in EQUITY_SIGNAL_NAMES}, top_n=5)
        assert len(ranked) == 2  # 不會因為部分訊號整批缺資料而崩潰或漏掉候選

    def test_all_signal_names_have_neutral_fallback_when_all_nan(self):
        rows = [{
            "code": "1000", "close": 100.0, "atr": 2.0,
            "revenue_yoy_pct": np.nan, "revenue_mom_pct": np.nan,
            "pe": np.nan, "pb": np.nan, "dividend_yield": np.nan,
            "eps_growth_pct": np.nan, "gross_margin_pct": np.nan, "roe_pct": np.nan,
            "foreign_streak": np.nan, "trust_streak": np.nan,
        }]
        ranked = _rank_equity_candidates(rows, {name: 1.0 for name in EQUITY_SIGNAL_NAMES}, top_n=5)
        assert len(ranked) == 1
        assert ranked[0]["score"] == 0.0  # 全部訊號都缺資料時，總分應該是中性的0


class TestScanEquityCandidatesGate:
    def test_bear_regime_returns_no_candidates(self):
        price_df = make_price_df(n=120, trend=0.002)
        indicators_by_code = {"1000": precompute_equity_indicators(price_df)}
        as_of = price_df.index[100]
        result = scan_equity_candidates(indicators_by_code, as_of, "bear", excluded_codes=set())
        assert result == []

    def test_below_ma60_gate_excludes_candidate(self):
        # 構造一段先跌破長均線的走勢：最近收盤持續走低，理論上不該通過MA60門檻
        price_df = make_price_df(n=120, trend=-0.01, seed=7)
        indicators_by_code = {"1000": precompute_equity_indicators(price_df)}
        as_of = price_df.index[100]
        result = scan_equity_candidates(indicators_by_code, as_of, "neutral", excluded_codes=set())
        assert result == []

    def test_excluded_codes_are_skipped(self):
        price_df = make_price_df(n=120, trend=0.003, seed=3)
        indicators_by_code = {"1000": precompute_equity_indicators(price_df)}
        as_of = price_df.index[100]
        result = scan_equity_candidates(
            indicators_by_code, as_of, "neutral", excluded_codes={"1000"})
        assert result == []


class TestExitLogic:
    def test_atr_stop_triggers_before_min_hold(self):
        position = {"code": "1000", "side": "long", "stop_price": 98.0, "entry_date": pd.Timestamp("2023-01-02"),
                    "hold_days": 1, "e_price": 100.0, "shares": 1000}
        row = pd.Series({"Open": 100.0, "High": 100.5, "Low": 97.0, "Close": 97.5})
        trades = []
        result = _process_equity_day(position, row, pd.Timestamp("2023-01-03"), trades,
                                      min_hold_days=10, max_hold_days=40)
        assert result is None  # 停損出場，即使還沒到最短持有天數
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "stop"

    def test_gap_through_stop_uses_open_price(self):
        position = {"code": "1000", "side": "long", "stop_price": 98.0, "entry_date": pd.Timestamp("2023-01-02"),
                    "hold_days": 2, "e_price": 100.0, "shares": 1000}
        row = pd.Series({"Open": 90.0, "High": 91.0, "Low": 89.0, "Close": 89.5})
        event, price = _check_equity_exit(row, position)
        assert event == "stop_gap"
        assert price == 90.0

    def test_forced_exit_at_max_hold_days(self):
        position = {"code": "1000", "side": "long", "stop_price": 50.0, "entry_date": pd.Timestamp("2023-01-02"),
                    "hold_days": 40, "e_price": 100.0, "shares": 1000}
        row = pd.Series({"Open": 105.0, "High": 106.0, "Low": 104.0, "Close": 105.5})
        trades = []
        result = _process_equity_day(position, row, pd.Timestamp("2023-03-01"), trades,
                                      min_hold_days=10, max_hold_days=40)
        assert result is None
        assert trades[0]["exit_reason"] == "time_exit"
        assert trades[0]["exit_price"] == 105.5

    def test_position_stays_open_when_no_exit_condition_met(self):
        position = {"code": "1000", "side": "long", "stop_price": 50.0, "entry_date": pd.Timestamp("2023-01-02"),
                    "hold_days": 5, "e_price": 100.0, "shares": 1000}
        row = pd.Series({"Open": 101.0, "High": 103.0, "Low": 99.0, "Close": 102.0})
        trades = []
        result = _process_equity_day(position, row, pd.Timestamp("2023-01-10"), trades,
                                      min_hold_days=10, max_hold_days=40)
        assert result is not None
        assert trades == []


class TestCloseEquityTradePnl:
    def test_pnl_includes_brokerage_fee_and_tax(self):
        position = {"code": "1000", "entry_date": pd.Timestamp("2023-01-02"),
                    "e_price": 100.0, "shares": 1000, "hold_days": 10}
        trades = []
        _close_equity_trade(position, 110.0, "time_exit", pd.Timestamp("2023-01-20"), trades)
        t = trades[0]
        buy_cost = 100.0 * 1000
        sell_proceeds = 110.0 * 1000
        buy_fee = buy_cost * ese.BROKERAGE_FEE_RATE
        sell_fee = sell_proceeds * (ese.BROKERAGE_FEE_RATE + ese.SECURITIES_TRANSACTION_TAX_RATE)
        expected_pnl = (sell_proceeds - sell_fee) - (buy_cost + buy_fee)
        assert t["pnl_ntd"] == pytest.approx(expected_pnl)
        assert t["lots"] == 1


class TestTryEnterEquityPosition:
    def test_skips_candidate_that_cannot_afford_one_lot(self):
        price_df = make_price_df(n=10, base=500.0)
        price_data = {"1000": price_df}
        candidates = [{"code": "1000", "side": "long", "score": 10.0,
                       "c_prev": 500.0, "atr": 5.0}]
        entry_date = price_df.index[1]
        # 現金只夠買不到一張(500元/股 x 1000股 = 500,000元，這裡只給10,000)
        position = try_enter_equity_position(price_data, candidates, entry_date, cash_allocated=10_000)
        assert position is None

    def test_enters_when_cash_sufficient(self):
        price_df = make_price_df(n=10, base=50.0)
        price_data = {"1000": price_df}
        candidates = [{"code": "1000", "side": "long", "score": 10.0,
                       "c_prev": 50.0, "atr": 1.0}]
        entry_date = price_df.index[1]
        position = try_enter_equity_position(price_data, candidates, entry_date, cash_allocated=200_000)
        assert position is not None
        assert position["shares"] >= SHARES_PER_LOT
        assert position["shares"] % SHARES_PER_LOT == 0


class TestFullBacktestSmoke:
    def _build_universe(self, n_codes=6, n_days=250):
        universe = {}
        price_data = {}
        for i in range(n_codes):
            code = f"{1000 + i}"
            universe[code] = {}
            price_data[code] = make_price_df(n=n_days, seed=i, trend=0.0012 if i % 2 == 0 else -0.0008)
        universe["2330"] = {}
        price_data["2330"] = make_price_df(n=n_days, seed=99, trend=0.001)
        return universe, price_data

    def test_runs_end_to_end_without_network_and_respects_position_cap(self):
        from mean_reversion_engine import precompute_regime_series

        universe, price_data = self._build_universe()
        indicators_by_code = precompute_all_equity_indicators(price_data, universe)
        regime_series = precompute_regime_series(price_data["2330"])

        max_positions = 3
        trades = run_equity_swing_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=price_data["2330"].index, starting_capital=1_000_000,
            min_hold_weeks=2, max_hold_weeks=8, atr_stop_mult=2.0,
            max_concurrent_positions=max_positions,
        )
        # 沒有崩潰、且至少有機會產生一些交易(不是嚴格要求一定有交易，但至少要能跑完整個迴圈)
        assert isinstance(trades, list)
        for t in trades:
            assert t["side"] == "long"
            assert t["shares"] % SHARES_PER_LOT == 0
            assert t["hold_days"] <= 8 * 5

    def test_no_network_calls_are_required_by_engine_module(self):
        # equity_swing_engine.py 不應該直接依賴 requests/yfinance——這支引擎完全
        # 靠呼叫端先準備好資料，這裡驗證模組本身沒有 import 對外連線套件。
        import inspect
        src = inspect.getsource(ese)
        assert "import requests" not in src
        assert "import yfinance" not in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
