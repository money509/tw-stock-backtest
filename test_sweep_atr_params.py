"""
test_sweep_atr_params.py
============================
針對 sweep_atr_params.py 的合成資料測試。不連網路、不重下載。驗證：
  1. build_atr_grid() 只產生 atr_target_mult >= atr_stop_mult 的組合，欄位正確
  2. run_sweep() 只用 IS 區間跑，回傳的 DataFrame 每個組合都有完整欄位
     （包含出場原因佔比 pct_target/pct_forced_close/pct_stop/pct_gap_stop）
  3. summarize_exit_reasons() 對空交易/正常交易都能算出合理比例，加總接近100%
  4. rank_results() 會濾掉交易筆數太少的組合，且依 profit_factor 由高到低排序
  5. validate_best_on_oos() 能正常在 OOS 跑，不會因為 dtype 轉換出錯（同類
     regression 邏輯跟 sweep_overnight_params.py 一樣重要，這裡也要保。）
"""

import unittest
import datetime
import pandas as pd
import numpy as np

import sweep_atr_params as atr_sweep
import compare_overnight as co


def make_ohlcv_df(n_days, base_price=100.0, start="2026-01-01", trend=0.3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    closes = base_price + np.cumsum(rng.normal(trend, 0.5, size=n_days))
    closes = np.maximum(closes, 1.0)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * 1.01
    lows = np.minimum(opens, closes) * 0.99
    volumes = rng.integers(500_000, 2_000_000, size=n_days)
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes,
    }, index=dates)
    df.index.name = "Date"
    return df


def make_chip_df(n_days, start="2026-01-01", seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    df = pd.DataFrame({
        "foreign_net": rng.integers(-100_000, 100_000, size=n_days),
        "trust_net": rng.integers(-50_000, 50_000, size=n_days),
        "dealer_net": rng.integers(-10_000, 10_000, size=n_days),
    }, index=dates)
    df["total_net"] = df["foreign_net"] + df["trust_net"] + df["dealer_net"]
    df.index.name = "date"
    return df


class FakeDayTradingLoader:
    def __init__(self, codes, n_days, start="2026-01-01", seed=2):
        self.codes = codes
        self.n_days = n_days
        self.start = start
        self.seed = seed

    def __call__(self, start_date, end_date, universe_codes=None, refresh=False):
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range(start=self.start, periods=self.n_days)
        rows = []
        for d in dates:
            for code in self.codes:
                if universe_codes is not None and code not in universe_codes:
                    continue
                rows.append({
                    "date": d.strftime("%Y%m%d"), "code": code, "name": code,
                    "day_trading_shares": int(rng.integers(0, 50_000)),
                })
        return pd.DataFrame(rows)


def make_pipeline_inputs(n_days=100, codes=("2330", "2317", "2454")):
    def fake_price_loader(whitelist, start, end, refresh=False):
        return {code: make_ohlcv_df(n_days, base_price=100 + i * 50, seed=i)
                for i, code in enumerate(codes)}

    def fake_chip_loader(start, end, universe_codes=None, refresh=False):
        data = {code: make_chip_df(n_days, seed=i) for i, code in enumerate(codes)}
        if universe_codes is not None:
            data = {c: df for c, df in data.items() if c in universe_codes}
        return data

    def fake_us_market_loader(start, end, refresh=False):
        dates = pd.bdate_range(start="2026-01-01", periods=n_days)
        return pd.DataFrame({
            "date": [d.strftime("%Y%m%d") for d in dates],
            "close": [4000.0] * n_days,
            "return_pct": [0.1] * n_days,
        })

    def fake_dividend_loader(whitelist, start, end, refresh=False):
        return {code: set() for code in whitelist}

    whitelist = {c: c for c in codes}
    return co.build_pipeline_inputs(
        whitelist, datetime.date(2026, 1, 1), datetime.date(2026, 6, 1),
        price_loader=fake_price_loader, chip_loader=fake_chip_loader,
        day_trading_loader_fn=FakeDayTradingLoader(list(codes), n_days),
        us_market_loader_fn=fake_us_market_loader,
        dividend_loader_fn=fake_dividend_loader,
    )


class TestBuildAtrGrid(unittest.TestCase):
    def test_grid_only_has_target_gte_stop_and_correct_fields(self):
        grid = atr_sweep.build_atr_grid()
        self.assertGreater(len(grid), 0)
        for combo in grid:
            self.assertIn("atr_stop_mult", combo)
            self.assertIn("atr_target_mult", combo)
            self.assertGreaterEqual(combo["atr_target_mult"], combo["atr_stop_mult"])

    def test_grid_excludes_target_less_than_stop_combo(self):
        grid = atr_sweep.build_atr_grid()
        # 0.5 stop / 1.0 target 之類的組合應該存在
        self.assertTrue(any(
            c["atr_stop_mult"] == 0.5 and c["atr_target_mult"] == 1.0 for c in grid
        ))
        # 但 1.5 stop / 1.0 target（風報比 < 1）不應該出現
        self.assertFalse(any(
            c["atr_stop_mult"] == 1.5 and c["atr_target_mult"] == 1.0 for c in grid
        ))


class TestSummarizeExitReasons(unittest.TestCase):
    def test_empty_trades_returns_zeros(self):
        result = atr_sweep.summarize_exit_reasons([])
        self.assertEqual(result["pct_target"], 0.0)
        self.assertEqual(result["pct_forced_close"], 0.0)
        self.assertEqual(result["pct_stop"], 0.0)
        self.assertEqual(result["pct_gap_stop"], 0.0)

    def test_percentages_sum_to_100(self):
        trades = [
            {"exit_reason": "target"}, {"exit_reason": "target"},
            {"exit_reason": "forced_close"}, {"exit_reason": "stop"},
            {"exit_reason": "gap_stop"},
        ]
        result = atr_sweep.summarize_exit_reasons(trades)
        total = (result["pct_target"] + result["pct_forced_close"]
                  + result["pct_stop"] + result["pct_gap_stop"])
        self.assertAlmostEqual(total, 100.0)
        self.assertAlmostEqual(result["pct_target"], 40.0)


class TestRunSweep(unittest.TestCase):
    def setUp(self):
        self.pipeline_inputs = make_pipeline_inputs()

    def test_run_sweep_returns_expected_columns(self):
        small_grid = [
            {"atr_stop_mult": 0.5, "atr_target_mult": 1.0},
            {"atr_stop_mult": 0.8, "atr_target_mult": 1.5},
        ]
        result_df, is_days, oos_days = atr_sweep.run_sweep(
            self.pipeline_inputs, param_grid=small_grid)

        self.assertEqual(len(result_df), 2)
        for col in ["atr_stop_mult", "atr_target_mult", "total_trades", "win_rate",
                    "profit_factor", "total_pnl", "avg_pnl",
                    "pct_target", "pct_forced_close", "pct_stop", "pct_gap_stop"]:
            self.assertIn(col, result_df.columns)

        full_days = self.pipeline_inputs["trading_days"]
        expected_is, expected_oos = co.split_is_oos(full_days)
        self.assertEqual(is_days, expected_is)
        self.assertEqual(oos_days, expected_oos)

    def test_different_atr_mults_change_exit_distribution(self):
        # 停損停利拉得很近 vs 拉得很遠，出場結構分佈應該不一樣
        # （不強求哪個方向，只驗證這個維度真的有在影響結果，不是死的）
        tight, _, _ = atr_sweep.run_sweep(
            self.pipeline_inputs, param_grid=[{"atr_stop_mult": 0.5, "atr_target_mult": 1.0}])
        wide, _, _ = atr_sweep.run_sweep(
            self.pipeline_inputs, param_grid=[{"atr_stop_mult": 1.5, "atr_target_mult": 3.0}])

        tight_row = tight.iloc[0]
        wide_row = wide.iloc[0]
        distributions_differ = not (
            abs(tight_row["pct_target"] - wide_row["pct_target"]) < 1e-9
            and abs(tight_row["pct_forced_close"] - wide_row["pct_forced_close"]) < 1e-9
            and abs(tight_row["pct_stop"] - wide_row["pct_stop"]) < 1e-9
        )
        self.assertTrue(distributions_differ)


    def test_run_sweep_passes_through_min_candidates_and_min_trust_ratio(self):
        """驗證min_candidates/min_trust_ratio真的有傳到run_overnight_backtest，
        不是被忽略的死參數——重掃ATR時要能固定住新鎖定的選股邏輯。"""
        import overnight_momentum_engine as ome
        captured = []
        real_run = ome.run_overnight_backtest

        def spy(**kwargs):
            captured.append((kwargs.get("min_candidates"), kwargs.get("min_trust_ratio")))
            return real_run(**kwargs)

        ome.run_overnight_backtest = spy
        try:
            atr_sweep.run_sweep(
                self.pipeline_inputs,
                param_grid=[{"atr_stop_mult": 0.8, "atr_target_mult": 1.5}],
                min_candidates=3, min_trust_ratio=0.02,
            )
        finally:
            ome.run_overnight_backtest = real_run

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], (3, 0.02))

    def test_run_sweep_passes_through_slippage_pct(self):
        """驗證slippage_pct真的有傳到run_overnight_backtest，掃ATR倍數時
        套用滑價才能避免掃到滑價/雜訊主導的假最佳解。"""
        import overnight_momentum_engine as ome
        captured = []
        real_run = ome.run_overnight_backtest

        def spy(**kwargs):
            captured.append(kwargs.get("slippage_pct"))
            return real_run(**kwargs)

        ome.run_overnight_backtest = spy
        try:
            atr_sweep.run_sweep(
                self.pipeline_inputs,
                param_grid=[{"atr_stop_mult": 0.8, "atr_target_mult": 1.5}],
                slippage_pct=0.1,
            )
        finally:
            ome.run_overnight_backtest = real_run

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], 0.1)

    def test_run_sweep_default_none_keeps_old_behavior(self):
        small_grid = [{"atr_stop_mult": 0.8, "atr_target_mult": 1.5}]
        default_df, _, _ = atr_sweep.run_sweep(self.pipeline_inputs, param_grid=small_grid)
        explicit_none_df, _, _ = atr_sweep.run_sweep(
            self.pipeline_inputs, param_grid=small_grid,
            min_candidates=None, min_trust_ratio=None,
        )
        self.assertAlmostEqual(
            default_df.iloc[0]["total_pnl"], explicit_none_df.iloc[0]["total_pnl"])


class TestRankResults(unittest.TestCase):
    def test_filters_low_trade_count_and_sorts_by_pf(self):
        df = pd.DataFrame([
            {"atr_stop_mult": 0.5, "atr_target_mult": 1.0, "total_trades": 5, "profit_factor": 5.0},
            {"atr_stop_mult": 0.8, "atr_target_mult": 1.5, "total_trades": 100, "profit_factor": 1.2},
            {"atr_stop_mult": 1.0, "atr_target_mult": 2.0, "total_trades": 200, "profit_factor": 2.5},
        ])
        ranked = atr_sweep.rank_results(df, min_trades=30)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked.iloc[0]["atr_stop_mult"], 1.0)
        self.assertEqual(ranked.iloc[1]["atr_stop_mult"], 0.8)

    def test_falls_back_to_all_when_none_reliable(self):
        df = pd.DataFrame([
            {"atr_stop_mult": 0.5, "atr_target_mult": 1.0, "total_trades": 2, "profit_factor": 5.0},
            {"atr_stop_mult": 0.8, "atr_target_mult": 1.5, "total_trades": 1, "profit_factor": 1.2},
        ])
        ranked = atr_sweep.rank_results(df, min_trades=30)
        self.assertEqual(len(ranked), 2)


class TestValidateBestOnOos(unittest.TestCase):
    def setUp(self):
        self.pipeline_inputs = make_pipeline_inputs()

    def test_validate_best_on_oos_runs_and_returns_summary(self):
        _, is_days, oos_days = atr_sweep.run_sweep(
            self.pipeline_inputs,
            param_grid=[{"atr_stop_mult": 0.8, "atr_target_mult": 1.5}],
        )
        best_params = {"atr_stop_mult": 0.8, "atr_target_mult": 1.5}
        summary = atr_sweep.validate_best_on_oos(self.pipeline_inputs, best_params, oos_days)
        for key in ["total_trades", "win_rate", "profit_factor", "total_pnl"]:
            self.assertIn(key, summary)

    def test_validate_best_on_oos_passes_through_min_candidates_and_min_trust_ratio(self):
        import overnight_momentum_engine as ome
        captured = []
        real_run = ome.run_overnight_backtest

        def spy(**kwargs):
            captured.append((kwargs.get("min_candidates"), kwargs.get("min_trust_ratio")))
            return real_run(**kwargs)

        ome.run_overnight_backtest = spy
        try:
            _, is_days, oos_days = atr_sweep.run_sweep(
                self.pipeline_inputs,
                param_grid=[{"atr_stop_mult": 0.8, "atr_target_mult": 1.5}],
            )
            best_params = {"atr_stop_mult": 0.8, "atr_target_mult": 1.5}
            atr_sweep.validate_best_on_oos(
                self.pipeline_inputs, best_params, oos_days,
                min_candidates=3, min_trust_ratio=0.02,
            )
        finally:
            ome.run_overnight_backtest = real_run

        self.assertIn((3, 0.02), captured)

    def test_validate_best_on_oos_handles_numpy_float_from_dataframe_row(self):
        """同 sweep_overnight_params.py 的 regression：從 DataFrame.iloc[0].to_dict()
        拿到的數值可能是 numpy 型別而非原生 float，這裡確認不會因此出錯。"""
        result_df, is_days, oos_days = atr_sweep.run_sweep(
            self.pipeline_inputs,
            param_grid=[{"atr_stop_mult": 0.8, "atr_target_mult": 1.5}],
        )
        ranked = atr_sweep.rank_results(result_df, min_trades=0)
        best = ranked.iloc[0].to_dict()
        summary = atr_sweep.validate_best_on_oos(self.pipeline_inputs, best, oos_days)
        self.assertIn("total_trades", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
