import pytest

from robustness_analysis import (
    bootstrap_resample_pnl, summarize_bootstrap, pnl_excluding_top_n_trades, bootstrap_p_value,
)


def make_trades(pnl_list):
    return [{"pnl_ntd": p} for p in pnl_list]


class TestBootstrapResamplePnl:
    def test_empty_trades_returns_empty_list(self):
        assert bootstrap_resample_pnl([]) == []

    def test_all_positive_pnl_always_resamples_positive(self):
        trades = make_trades([100, 200, 150, 300])
        results = bootstrap_resample_pnl(trades, n_resamples=200, seed=1)
        assert len(results) == 200
        assert all(r > 0 for r in results)

    def test_deterministic_with_fixed_seed(self):
        trades = make_trades([100, -50, 200, -30, 80])
        r1 = bootstrap_resample_pnl(trades, n_resamples=50, seed=7)
        r2 = bootstrap_resample_pnl(trades, n_resamples=50, seed=7)
        assert r1 == r2


class TestSummarizeBootstrap:
    def test_empty_results(self):
        stats = summarize_bootstrap([])
        assert stats == {"mean": 0.0, "p5": 0.0, "p95": 0.0, "pct_positive": 0.0}

    def test_all_positive_gives_100pct_positive(self):
        stats = summarize_bootstrap([10.0, 20.0, 30.0])
        assert stats["pct_positive"] == 100.0

    def test_mixed_results_pct_positive_between_0_and_100(self):
        stats = summarize_bootstrap([10.0, -10.0, 5.0, -5.0])
        assert 0.0 < stats["pct_positive"] < 100.0


class TestPnlExcludingTopNTrades:
    def test_excludes_top_n_largest_trades(self):
        trades = make_trades([1000, 500, 100, -50, -100])
        # 拿掉最大的3筆(1000,500,100)，剩下-50,-100
        assert pnl_excluding_top_n_trades(trades, n=3) == pytest.approx(-150)

    def test_empty_trades_returns_zero(self):
        assert pnl_excluding_top_n_trades([], n=3) == 0.0

    def test_n_larger_than_trade_count_returns_zero(self):
        trades = make_trades([100, 200])
        assert pnl_excluding_top_n_trades(trades, n=5) == 0.0


class TestBootstrapPValue:
    def test_empty_results_returns_one(self):
        assert bootstrap_p_value([]) == 1.0

    def test_all_positive_gives_p_value_zero(self):
        assert bootstrap_p_value([10.0, 20.0, 30.0]) == 0.0

    def test_all_non_positive_gives_p_value_one(self):
        assert bootstrap_p_value([-10.0, -20.0, 0.0]) == 1.0

    def test_matches_one_minus_pct_positive(self):
        results = [10.0, -10.0, 5.0, -5.0, 3.0]
        p = bootstrap_p_value(results)
        pct_pos = summarize_bootstrap(results)["pct_positive"]
        assert p == pytest.approx(1 - pct_pos / 100)
