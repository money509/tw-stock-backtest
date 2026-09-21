"""
test_sweep_overnight_params.py
=================================
針對 sweep_overnight_params.py 的合成資料測試。不連網路、不重下載，
用跟 test_compare_overnight.py 一樣的合成資料建構手法，驗證：
  1. build_param_grid() 產生的組合數量、欄位正確
  2. run_sweep() 只用 IS 區間跑，且回傳的 DataFrame 每個組合都有完整欄位
  3. rank_results() 會濾掉交易筆數太少的組合，且依 profit_factor 由高到低排序
  4. validate_best_on_oos() 用指定參數在 OOS 跑，不會影響/汙染 IS 的排名結果
"""

import unittest
import datetime
import pandas as pd
import numpy as np

import sweep_overnight_params as sweep
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


class TestBuildParamGrid(unittest.TestCase):
    def test_grid_size_and_fields(self):
        grid = sweep.build_param_grid()
        self.assertEqual(len(grid), 3 * 3 * 3)  # top_n * weight組合 * gap門檻
        for combo in grid:
            self.assertIn("top_n", combo)
            self.assertIn("tech_weight", combo)
            self.assertIn("chip_weight", combo)
            self.assertIn("gap_stop_threshold", combo)
            self.assertAlmostEqual(combo["tech_weight"] + combo["chip_weight"], 1.0)


class TestRunSweep(unittest.TestCase):
    def setUp(self):
        self.pipeline_inputs = make_pipeline_inputs()

    def test_run_sweep_returns_expected_columns(self):
        small_grid = [
            {"top_n": 2, "tech_weight": 0.35, "chip_weight": 0.65, "gap_stop_threshold": -0.015},
            {"top_n": 3, "tech_weight": 0.5, "chip_weight": 0.5, "gap_stop_threshold": -0.01},
        ]
        result_df, is_days, oos_days = sweep.run_sweep(self.pipeline_inputs, param_grid=small_grid)

        self.assertEqual(len(result_df), 2)
        for col in ["top_n", "tech_weight", "chip_weight", "gap_stop_threshold",
                    "total_trades", "win_rate", "profit_factor", "total_pnl", "avg_pnl"]:
            self.assertIn(col, result_df.columns)

        # is_days 應該恰好是 trading_days 的前 70%
        full_days = self.pipeline_inputs["trading_days"]
        expected_is, expected_oos = co.split_is_oos(full_days)
        self.assertEqual(is_days, expected_is)
        self.assertEqual(oos_days, expected_oos)


class TestRankResults(unittest.TestCase):
    def test_filters_low_trade_count_and_sorts_by_pf(self):
        df = pd.DataFrame([
            {"top_n": 3, "total_trades": 5, "profit_factor": 5.0},   # 筆數太少，該被濾掉
            {"top_n": 5, "total_trades": 100, "profit_factor": 1.2},
            {"top_n": 8, "total_trades": 200, "profit_factor": 2.5},
        ])
        ranked = sweep.rank_results(df, min_trades=30)
        self.assertEqual(len(ranked), 2)  # 只剩兩組
        self.assertEqual(ranked.iloc[0]["top_n"], 8)  # PF最高的排第一
        self.assertEqual(ranked.iloc[1]["top_n"], 5)

    def test_falls_back_to_all_when_none_reliable(self):
        df = pd.DataFrame([
            {"top_n": 3, "total_trades": 2, "profit_factor": 5.0},
            {"top_n": 5, "total_trades": 1, "profit_factor": 1.2},
        ])
        ranked = sweep.rank_results(df, min_trades=30)
        self.assertEqual(len(ranked), 2)  # 全部都不可靠時，原樣列出，不會回傳空的


class TestValidateBestOnOos(unittest.TestCase):
    def setUp(self):
        self.pipeline_inputs = make_pipeline_inputs()

    def test_validate_best_on_oos_handles_float_top_n_from_dataframe_row(self):
        """回歸測試：ranked.iloc[0].to_dict() 會把 top_n 這種整數欄位連帶轉成 float
        （因為同一列混了 win_rate/profit_factor 等浮點欄位，pandas 統一成同一種 dtype）。
        這個路徑之前在真實跑 sweep 時炸過：
        TypeError: cannot do positional indexing on RangeIndex with these indexers [8.0]
        這裡直接模擬那個轉換過程，確保 validate_best_on_oos 能吃 float 型的 top_n。"""
        result_df, is_days, oos_days = sweep.run_sweep(
            self.pipeline_inputs,
            param_grid=[{"top_n": 2, "tech_weight": 0.5, "chip_weight": 0.5,
                         "gap_stop_threshold": -0.015}],
        )
        ranked = sweep.rank_results(result_df, min_trades=0)
        best = ranked.iloc[0].to_dict()
        self.assertIsInstance(best["top_n"], float)  # 先確認真的會被轉成 float，不是這個測試自己假設

        # 不應該丟出 TypeError
        summary = sweep.validate_best_on_oos(self.pipeline_inputs, best, oos_days)
        self.assertIn("total_trades", summary)

    def test_validate_best_on_oos_runs_and_returns_summary(self):
        _, is_days, oos_days = sweep.run_sweep(
            self.pipeline_inputs,
            param_grid=[{"top_n": 2, "tech_weight": 0.35, "chip_weight": 0.65,
                         "gap_stop_threshold": -0.015}],
        )
        best_params = {"top_n": 2, "tech_weight": 0.35, "chip_weight": 0.65,
                        "gap_stop_threshold": -0.015}
        summary = sweep.validate_best_on_oos(self.pipeline_inputs, best_params, oos_days)
        for key in ["total_trades", "win_rate", "profit_factor", "total_pnl"]:
            self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
