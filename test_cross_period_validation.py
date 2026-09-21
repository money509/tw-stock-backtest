"""
test_cross_period_validation.py
===================================
針對 cross_period_validation.py 的合成資料測試。不連網路、不重下載。驗證：
  1. run_cross_period_validation() 對每個候選組合都跑一次，用「整段」trading_days
     （不切IS/OOS），回傳的 DataFrame 欄位完整
  2. 不同候選組合（不同訊號權重/ATR參數）確實跑出不同結果，不是每組都一樣
     （代表參數真的有被套用，不是死的）
"""

import unittest
import datetime
import pandas as pd
import numpy as np

import cross_period_validation as cpv
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

    whitelist = {c: c for c in codes}
    return co.build_pipeline_inputs(
        whitelist, datetime.date(2026, 1, 1), datetime.date(2026, 6, 1),
        price_loader=fake_price_loader, chip_loader=fake_chip_loader,
        day_trading_loader_fn=FakeDayTradingLoader(list(codes), n_days),
        us_market_loader_fn=fake_us_market_loader,
    )


class TestRunCrossPeriodValidation(unittest.TestCase):
    def setUp(self):
        self.pipeline_inputs = make_pipeline_inputs()

    def test_runs_all_candidates_with_expected_columns(self):
        small_candidates = [
            {"label": "只用量比", "signal_weights": {"score_volume_ratio": 1.0},
             "atr_stop_mult": 0.8, "atr_target_mult": 1.2},
            {"label": "只用投信", "signal_weights": {"score_trust": 1.0},
             "atr_stop_mult": 0.5, "atr_target_mult": 3.0},
        ]
        result_df = cpv.run_cross_period_validation(
            self.pipeline_inputs, candidates=small_candidates, top_n=2)

        self.assertEqual(len(result_df), 2)
        for col in ["label", "total_trades", "win_rate", "profit_factor",
                    "total_pnl", "avg_pnl"]:
            self.assertIn(col, result_df.columns)
        self.assertEqual(list(result_df["label"]), ["只用量比", "只用投信"])

    def test_uses_full_trading_days_not_split(self):
        """驗證這支腳本用的是整段 trading_days，不像 sweep 那樣切 IS/OOS。"""
        captured = []
        real_run = __import__("overnight_momentum_engine").run_overnight_backtest

        import overnight_momentum_engine as ome

        def spy(**kwargs):
            captured.append(kwargs["trading_days"])
            return real_run(**kwargs)

        ome.run_overnight_backtest = spy
        try:
            small_candidates = [
                {"label": "只用量比", "signal_weights": {"score_volume_ratio": 1.0},
                 "atr_stop_mult": 0.8, "atr_target_mult": 1.2},
            ]
            cpv.run_cross_period_validation(
                self.pipeline_inputs, candidates=small_candidates, top_n=2)
        finally:
            ome.run_overnight_backtest = real_run

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], self.pipeline_inputs["trading_days"])

    def test_different_candidates_produce_different_results(self):
        small_candidates = [
            {"label": "只用量比", "signal_weights": {"score_volume_ratio": 1.0},
             "atr_stop_mult": 0.8, "atr_target_mult": 1.2},
            {"label": "只用投信", "signal_weights": {"score_trust": 1.0},
             "atr_stop_mult": 0.5, "atr_target_mult": 3.0},
        ]
        result_df = cpv.run_cross_period_validation(
            self.pipeline_inputs, candidates=small_candidates, top_n=2)
        row_a = result_df.iloc[0]
        row_b = result_df.iloc[1]
        self.assertFalse(
            row_a["total_pnl"] == row_b["total_pnl"] and
            row_a["total_trades"] == row_b["total_trades"]
        )


class TestCandidatesConstant(unittest.TestCase):
    def test_candidates_have_required_fields(self):
        self.assertGreater(len(cpv.CANDIDATES), 0)
        for cand in cpv.CANDIDATES:
            self.assertIn("label", cand)
            self.assertIn("signal_weights", cand)
            self.assertIn("atr_stop_mult", cand)
            self.assertIn("atr_target_mult", cand)
            self.assertGreater(len(cand["signal_weights"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
