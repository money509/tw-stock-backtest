"""
test_compare_equity_swing.py
===============================
針對 compare_equity_swing.py 的合成資料測試。完全不連真實網路——用小規模合成
股價資料直接呼叫這個檔案裡的函式，不透過main()/argparse，也不下載任何真實資料。

涵蓋範圍：
1. select_winning_signals()：PF>1篩選規則 vs 找不到時退回總損益前3名的fallback
2. run_fixed_combo_equity_walkforward()：切N折的輸出形狀(每折一列、欄位齊全)，
   以及某折交易筆數太少時會被跳過而不是硬塞進結果
3. run_signal_ablation()：small smoke test，確認能跑完整個訊號清單 + 基準列，
   且沒有開啟的資料源訊號會被排除(has_fundamentals過濾)
"""
import numpy as np
import pandas as pd
import pytest

from compare_equity_swing import (
    select_winning_signals, run_fixed_combo_equity_walkforward, run_signal_ablation, SIGNAL_LABELS,
)
from equity_swing_engine import EQUITY_SIGNAL_NAMES, precompute_all_equity_indicators
from mean_reversion_engine import precompute_regime_series


def make_price_df(n=250, seed=0, trend=0.001, base=100.0):
    dates = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(seed)
    close = base * np.cumprod(1 + rng.normal(trend, 0.012, n))
    high = close * (1 + rng.uniform(0.0, 0.01, n))
    low = close * (1 - rng.uniform(0.0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    vol = rng.integers(1000, 5000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates)


def build_synthetic_universe(n_codes=6, n_days=260):
    universe, price_data = {}, {}
    for i in range(n_codes):
        code = f"{1000 + i}"
        universe[code] = {}
        price_data[code] = make_price_df(n=n_days, seed=i, trend=0.0015 if i % 2 == 0 else -0.0008)
    universe["2330"] = {}
    price_data["2330"] = make_price_df(n=n_days, seed=99, trend=0.0012)
    return universe, price_data


class TestSelectWinningSignals:
    def test_picks_signals_with_pf_above_1_and_enough_trades(self):
        df = pd.DataFrame([
            {"signal": "score_revenue_yoy", "trade_count": 20, "profit_factor": 1.5, "total_pnl_ntd": 10000},
            {"signal": "score_valuation_pe", "trade_count": 20, "profit_factor": 0.8, "total_pnl_ntd": -5000},
            {"signal": "__baseline_equal_weight__", "trade_count": 20, "profit_factor": 1.1, "total_pnl_ntd": 2000},
        ])
        signals, reliable = select_winning_signals(df, min_trades=15)
        assert signals == ["score_revenue_yoy"]
        assert reliable is True

    def test_falls_back_to_top3_pnl_when_nothing_qualifies(self):
        df = pd.DataFrame([
            {"signal": "score_revenue_yoy", "trade_count": 3, "profit_factor": 1.5, "total_pnl_ntd": 500},
            {"signal": "score_valuation_pe", "trade_count": 3, "profit_factor": 0.8, "total_pnl_ntd": -300},
            {"signal": "score_roe", "trade_count": 3, "profit_factor": 0.9, "total_pnl_ntd": 100},
        ])
        signals, reliable = select_winning_signals(df, min_trades=15)
        assert reliable is False
        assert len(signals) <= 3
        assert set(signals).issubset({"score_revenue_yoy", "score_valuation_pe", "score_roe"})

    def test_zero_trade_rows_are_not_preferred_over_real_rows(self):
        df = pd.DataFrame([
            {"signal": "score_revenue_yoy", "trade_count": 0, "profit_factor": 0.0, "total_pnl_ntd": 0.0},
            {"signal": "score_valuation_pe", "trade_count": 20, "profit_factor": 1.2, "total_pnl_ntd": 8000},
        ])
        signals, reliable = select_winning_signals(df, min_trades=15)
        assert signals == ["score_valuation_pe"]


class TestFixedComboEquityWalkforward:
    def test_output_has_one_row_per_fold_with_expected_columns(self):
        universe, price_data = build_synthetic_universe(n_codes=8, n_days=300)
        indicators_by_code = precompute_all_equity_indicators(price_data, universe)
        regime_series = precompute_regime_series(price_data["2330"])
        master_calendar = price_data["2330"].index

        common_kwargs = dict(
            min_hold_weeks=2, max_hold_weeks=8, atr_stop_mult=2.0,
            max_concurrent_positions=3, risk_pct_per_trade=None, slippage_pct=0.0,
        )
        signal_weights = {name: 1.0 for name in EQUITY_SIGNAL_NAMES}

        df = run_fixed_combo_equity_walkforward(
            price_data, indicators_by_code, regime_series, master_calendar,
            starting_capital=1_000_000, common_kwargs=common_kwargs,
            signal_weights=signal_weights, n_folds=4,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 4  # 交易數太少的折會被跳過，所以是<=n_folds，不保證每折都有結果
        if not df.empty:
            expected_cols = {"fold", "period_start", "period_end", "trade_count",
                              "profit_factor", "win_rate", "total_pnl_ntd", "avg_hold_days"}
            assert expected_cols.issubset(set(df.columns))
            assert df["fold"].is_unique
            assert df["fold"].between(1, 4).all()

    def test_empty_master_calendar_chunk_is_handled_without_crash(self):
        universe, price_data = build_synthetic_universe(n_codes=3, n_days=20)
        indicators_by_code = precompute_all_equity_indicators(price_data, universe)
        regime_series = precompute_regime_series(price_data["2330"])
        master_calendar = price_data["2330"].index[:5]  # 刻意給很短的日曆，切10折會有很多空區塊

        common_kwargs = dict(
            min_hold_weeks=2, max_hold_weeks=8, atr_stop_mult=2.0,
            max_concurrent_positions=2, risk_pct_per_trade=None, slippage_pct=0.0,
        )
        df = run_fixed_combo_equity_walkforward(
            price_data, indicators_by_code, regime_series, master_calendar,
            starting_capital=500_000, common_kwargs=common_kwargs,
            signal_weights={name: 1.0 for name in EQUITY_SIGNAL_NAMES}, n_folds=10,
        )
        assert isinstance(df, pd.DataFrame)  # 不崩潰，回傳的DataFrame可能是空的或列數很少


class TestRunSignalAblation:
    def test_ablation_runs_and_excludes_signals_without_data_source(self):
        universe, price_data = build_synthetic_universe(n_codes=6, n_days=260)
        indicators_by_code = precompute_all_equity_indicators(price_data, universe)  # 沒有任何基本面/籌碼面資料
        regime_series = precompute_regime_series(price_data["2330"])
        master_calendar = price_data["2330"].index

        # 完全沒開任何資料源，has_fundamentals全部False，訊號拆解應該直接跳過全部訊號，
        # 只剩下基準列(0訊號等權重)
        has_fundamentals = {name: False for name in EQUITY_SIGNAL_NAMES}
        common_kwargs = dict(
            min_hold_weeks=2, max_hold_weeks=8, atr_stop_mult=2.0,
            max_concurrent_positions=3, risk_pct_per_trade=None, slippage_pct=0.0,
        )
        ablation_df, tested_signals = run_signal_ablation(
            price_data, indicators_by_code, regime_series, master_calendar, 1_000_000,
            common_kwargs, has_fundamentals,
        )
        assert tested_signals == []
        assert list(ablation_df["signal"]) == ["__baseline_equal_weight__"]

    def test_ablation_includes_signal_when_flagged_available(self):
        universe, price_data = build_synthetic_universe(n_codes=6, n_days=260)
        indicators_by_code = precompute_all_equity_indicators(price_data, universe)
        regime_series = precompute_regime_series(price_data["2330"])
        master_calendar = price_data["2330"].index

        has_fundamentals = {name: False for name in EQUITY_SIGNAL_NAMES}
        has_fundamentals["score_foreign_streak"] = True
        has_fundamentals["score_trust_streak"] = True
        common_kwargs = dict(
            min_hold_weeks=2, max_hold_weeks=8, atr_stop_mult=2.0,
            max_concurrent_positions=3, risk_pct_per_trade=None, slippage_pct=0.0,
        )
        ablation_df, tested_signals = run_signal_ablation(
            price_data, indicators_by_code, regime_series, master_calendar, 1_000_000,
            common_kwargs, has_fundamentals,
        )
        assert set(tested_signals) == {"score_foreign_streak", "score_trust_streak"}
        assert set(ablation_df["signal"]) == {
            "score_foreign_streak", "score_trust_streak", "__baseline_equal_weight__"}


class TestNoNetworkCallsInOrchestration:
    def test_module_does_not_directly_import_requests_or_yfinance(self):
        import inspect
        import compare_equity_swing as ces
        src = inspect.getsource(ces)
        assert "import requests" not in src
        assert "import yfinance" not in src
        # loader模組是延遲在main()裡面才import(避免測試/沒開資料源時的環境不需要它們)，
        # 這裡確認幾個loader的import語句確實只出現在function-local層級(縮排內)，不是頂層
        for loader_import in ("from revenue_data_loader", "from valuation_data_loader",
                               "from financial_statement_loader", "from chip_data_loader"):
            for line in src.splitlines():
                if line.strip().startswith(loader_import):
                    assert line.startswith("        ") or line.startswith("    "), \
                        f"{loader_import} 應該是函式內的延遲import，不是模組頂層import"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
