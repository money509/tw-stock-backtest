"""
test_sweep_combined_best.py
==============================
針對 sweep_combined_best.py 的合成資料測試。不連網路、不重下載。驗證：
  1. run_combined_sweep() 對每一組「訊號權重 x ATR參數」都跑一次，回傳的
     DataFrame 筆數等於兩邊數量相乘，欄位完整，只用IS區間
  2. rank_results() 會濾掉交易筆數太少的組合，且依 profit_factor 由高到低排序
  3. validate_best_on_oos() 能正確用排名第一那組的訊號權重+ATR參數在OOS重跑，
     不會因為 DataFrame.iloc[0].to_dict() 的 dtype 轉換出錯
"""

import unittest
import datetime
import pandas as pd
import numpy as np

import sweep_combined_best as combined
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


class TestRunCombinedSweep(unittest.TestCase):
    def setUp(self):
        self.pipeline_inputs = make_pipeline_inputs()

    def test_returns_one_row_per_signal_x_atr_combo(self):
        small_signals = {
            "全部等權重": {"score_volume_ratio": 1.0, "score_trust": 1.0},
            "只用量比": {"score_volume_ratio": 1.0},
        }
        small_atr = {
            "組合A": {"atr_stop_mult": 0.8, "atr_target_mult": 1.2},
            "組合B": {"atr_stop_mult": 0.5, "atr_target_mult": 3.0},
        }
        result_df, is_days, oos_days = combined.run_combined_sweep(
            self.pipeline_inputs, top_n=2,
            signal_variants=small_signals, atr_variants=small_atr,
        )
        self.assertEqual(len(result_df), 4)  # 2 x 2
        for col in ["signal_variant", "atr_variant", "atr_stop_mult", "atr_target_mult",
                    "total_trades", "win_rate", "profit_factor", "total_pnl", "avg_pnl"]:
            self.assertIn(col, result_df.columns)

        full_days = self.pipeline_inputs["trading_days"]
        expected_is, expected_oos = co.split_is_oos(full_days)
        self.assertEqual(is_days, expected_is)
        self.assertEqual(oos_days, expected_oos)

    def test_different_signal_weights_change_results(self):
        """驗證不同訊號權重真的有被套用，不是每組都跑出一樣的結果。"""
        small_signals = {
            "只用量比": {"score_volume_ratio": 1.0},
            "只用投信": {"score_trust": 1.0},
        }
        small_atr = {"固定組合": {"atr_stop_mult": 0.8, "atr_target_mult": 1.2}}
        result_df, _, _ = combined.run_combined_sweep(
            self.pipeline_inputs, top_n=2,
            signal_variants=small_signals, atr_variants=small_atr,
        )
        row_a = result_df[result_df["signal_variant"] == "只用量比"].iloc[0]
        row_b = result_df[result_df["signal_variant"] == "只用投信"].iloc[0]
        # 不強求哪個比較好，只驗證兩者確實跑出不同結果（訊號權重真的生效）
        self.assertFalse(
            row_a["total_pnl"] == row_b["total_pnl"] and
            row_a["total_trades"] == row_b["total_trades"]
        )


class TestRankResults(unittest.TestCase):
    def test_filters_low_trade_count_and_sorts_by_pf(self):
        df = pd.DataFrame([
            {"signal_variant": "a", "atr_variant": "x", "total_trades": 5, "profit_factor": 5.0},
            {"signal_variant": "b", "atr_variant": "y", "total_trades": 100, "profit_factor": 1.2},
            {"signal_variant": "c", "atr_variant": "z", "total_trades": 200, "profit_factor": 2.5},
        ])
        ranked = combined.rank_results(df, min_trades=30)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked.iloc[0]["signal_variant"], "c")
        self.assertEqual(ranked.iloc[1]["signal_variant"], "b")

    def test_falls_back_to_all_when_none_reliable(self):
        df = pd.DataFrame([
            {"signal_variant": "a", "atr_variant": "x", "total_trades": 2, "profit_factor": 5.0},
            {"signal_variant": "b", "atr_variant": "y", "total_trades": 1, "profit_factor": 1.2},
        ])
        ranked = combined.rank_results(df, min_trades=30)
        self.assertEqual(len(ranked), 2)


class TestValidateBestOnOos(unittest.TestCase):
    def setUp(self):
        self.pipeline_inputs = make_pipeline_inputs()

    def test_runs_and_returns_summary(self):
        small_signals = {"只用量比": {"score_volume_ratio": 1.0}}
        small_atr = {"固定組合": {"atr_stop_mult": 0.8, "atr_target_mult": 1.2}}
        result_df, is_days, oos_days = combined.run_combined_sweep(
            self.pipeline_inputs, top_n=2,
            signal_variants=small_signals, atr_variants=small_atr,
        )
        best = result_df.iloc[0].to_dict()
        summary = combined.validate_best_on_oos(
            self.pipeline_inputs, best, oos_days, top_n=2,
            signal_weight_variants=small_signals,
        )
        for key in ["total_trades", "win_rate", "profit_factor", "total_pnl"]:
            self.assertIn(key, summary)

    def test_handles_float_atr_mult_from_dataframe_row(self):
        """回歸測試：跟 sweep_overnight_params.py 一樣，ranked.iloc[0].to_dict()
        可能把數值欄位轉成非預期的 dtype，確認 validate_best_on_oos 能正常處理。"""
        small_signals = {"只用量比": {"score_volume_ratio": 1.0}}
        small_atr = {"固定組合": {"atr_stop_mult": 0.8, "atr_target_mult": 1.2}}
        result_df, is_days, oos_days = combined.run_combined_sweep(
            self.pipeline_inputs, top_n=2,
            signal_variants=small_signals, atr_variants=small_atr,
        )
        ranked = combined.rank_results(result_df, min_trades=0)
        best = ranked.iloc[0].to_dict()
        # 不應該丟出 TypeError 或 KeyError
        summary = combined.validate_best_on_oos(
            self.pipeline_inputs, best, oos_days, top_n=2,
            signal_weight_variants=small_signals,
        )
        self.assertIn("total_trades", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
