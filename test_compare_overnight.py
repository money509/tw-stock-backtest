"""
test_compare_overnight.py
============================
針對 compare_overnight.py 的整合測試。用注入的假資料源(price_loader/chip_loader/
day_trading_loader_fn)取代真實的 data_loader/chip_data_loader/day_trading_loader，
驗證整條管線（股價轉換 -> 籌碼轉接 -> 當沖比例合併 -> IS/OOS切分 -> 回測 -> 彙總）
真的能串起來，不需要連網路、不需要真實資料。

驗證重點：
  1. convert_price_data_to_indicators() 正確把 data_loader 格式轉成引擎格式
  2. build_volume_df() 正確攤平成長格式
  3. build_pipeline_inputs() 端到端組出全部回測所需輸入，且各 DataFrame 欄位正確
  4. split_is_oos() 依比例正確切分，且 IS 在前、OOS 在後、沒有重疊
  5. run_is_oos_backtest() 全流程跑得通，回傳結構正確（即使沒有交易也不出錯）
"""

import unittest
import datetime
import pandas as pd
import numpy as np

import compare_overnight as co
import overnight_momentum_engine as ome


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


class TestConversionHelpers(unittest.TestCase):
    def test_convert_price_data_to_indicators_basic(self):
        price_data = {"2330": make_ohlcv_df(80)}
        indicators = co.convert_price_data_to_indicators(price_data)
        self.assertIn("2330", indicators)
        df = indicators["2330"]
        for col in ["date", "open", "high", "low", "close", "volume", "ma20", "atr14"]:
            self.assertIn(col, df.columns)
        # date 欄位要是 YYYYMMDD 字串格式（8碼數字）
        self.assertTrue(df["date"].iloc[0].isdigit())
        self.assertEqual(len(df["date"].iloc[0]), 8)

    def test_convert_price_data_skips_empty_frames(self):
        price_data = {"2330": make_ohlcv_df(80), "EMPTY": pd.DataFrame()}
        indicators = co.convert_price_data_to_indicators(price_data)
        self.assertIn("2330", indicators)
        self.assertNotIn("EMPTY", indicators)

    def test_build_volume_df_long_format(self):
        price_data = {"2330": make_ohlcv_df(10), "2317": make_ohlcv_df(10, base_price=50)}
        vol_df = co.build_volume_df(price_data)
        self.assertEqual(set(vol_df["code"].unique()), {"2330", "2317"})
        self.assertEqual(len(vol_df), 20)
        self.assertIn("volume", vol_df.columns)
        self.assertIn("date", vol_df.columns)


class TestSplitIsOos(unittest.TestCase):
    def test_split_ratio_and_no_overlap(self):
        days = [f"2026{str(m).zfill(2)}{str(d).zfill(2)}" for m in range(1, 3) for d in range(1, 21)]
        is_days, oos_days = co.split_is_oos(days, is_ratio=0.7)
        self.assertEqual(len(is_days) + len(oos_days), len(days))
        self.assertAlmostEqual(len(is_days) / len(days), 0.7, delta=0.05)
        self.assertEqual(set(is_days) & set(oos_days), set())
        self.assertEqual(is_days + oos_days, days)  # IS在前, OOS在後, 順序不變


class FakeDayTradingLoader:
    """模擬 day_trading_loader.load_day_trading_data 的介面。"""
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
                    "date": d.strftime("%Y%m%d"),
                    "code": code,
                    "name": code,
                    "day_trading_shares": int(rng.integers(0, 50_000)),
                })
        return pd.DataFrame(rows)


class TestFullPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.n_days = 100
        self.codes = ["2330", "2317", "2454"]

        def fake_price_loader(whitelist, start, end, refresh=False):
            return {code: make_ohlcv_df(self.n_days, base_price=100 + i * 50, seed=i)
                    for i, code in enumerate(self.codes)}

        def fake_chip_loader(start, end, universe_codes=None, refresh=False):
            data = {code: make_chip_df(self.n_days, seed=i) for i, code in enumerate(self.codes)}
            if universe_codes is not None:
                data = {c: df for c, df in data.items() if c in universe_codes}
            return data

        def fake_us_market_loader(start, end, refresh=False):
            dates = pd.bdate_range(start="2026-01-01", periods=self.n_days)
            return pd.DataFrame({
                "date": [d.strftime("%Y%m%d") for d in dates],
                "close": [4000.0] * self.n_days,
                "return_pct": [0.1] * self.n_days,  # 溫和正報酬，不會觸發濾網
            })

        self.fake_price_loader = fake_price_loader
        self.fake_chip_loader = fake_chip_loader
        self.fake_day_trading_loader = FakeDayTradingLoader(self.codes, self.n_days)
        self.fake_us_market_loader = fake_us_market_loader

        self.whitelist = {c: c for c in self.codes}

    def test_build_pipeline_inputs_end_to_end(self):
        inputs = co.build_pipeline_inputs(
            self.whitelist,
            datetime.date(2026, 1, 1), datetime.date(2026, 6, 1),
            price_loader=self.fake_price_loader,
            chip_loader=self.fake_chip_loader,
            day_trading_loader_fn=self.fake_day_trading_loader,
            us_market_loader_fn=self.fake_us_market_loader,
        )

        self.assertEqual(set(inputs["indicators_by_code"].keys()), set(self.codes))
        self.assertGreater(len(inputs["trading_days"]), 0)
        self.assertIn("return_pct", inputs["us_market_returns_df"].columns)
        self.assertIn("ratio", inputs["foreign_ratio_df"].columns)
        self.assertIn("ratio", inputs["trust_ratio_df"].columns)
        self.assertIn("day_trading_ratio", inputs["day_trading_ratio_df"].columns)
        self.assertEqual(inputs["universe_codes"], set(self.codes))

    def test_missing_reference_code_raises(self):
        with self.assertRaises(RuntimeError):
            co.build_pipeline_inputs(
                {"9999": "無參考股"},
                datetime.date(2026, 1, 1), datetime.date(2026, 6, 1),
                price_loader=lambda w, s, e, refresh=False: {},
                chip_loader=self.fake_chip_loader,
                day_trading_loader_fn=self.fake_day_trading_loader,
            )

    def test_run_is_oos_backtest_end_to_end_no_crash(self):
        inputs = co.build_pipeline_inputs(
            self.whitelist,
            datetime.date(2026, 1, 1), datetime.date(2026, 6, 1),
            price_loader=self.fake_price_loader,
            chip_loader=self.fake_chip_loader,
            day_trading_loader_fn=self.fake_day_trading_loader,
            us_market_loader_fn=self.fake_us_market_loader,
        )

        is_trades, oos_trades, is_summary, oos_summary = co.run_is_oos_backtest(inputs, top_n=2)

        for summary in (is_summary, oos_summary):
            self.assertIn("total_trades", summary)
            self.assertIn("win_rate", summary)
            self.assertIn("profit_factor", summary)
            self.assertIn("total_pnl", summary)

        # IS 交易日期都應該早於（或等於）OOS 最早的交易日期
        if is_trades and oos_trades:
            self.assertLessEqual(max(t["entry_date"] for t in is_trades),
                                  min(t["entry_date"] for t in oos_trades))

    def test_print_summary_does_not_crash_on_empty(self):
        empty_summary = ome.summarize_overnight([])
        # 不應該丟出例外
        co.print_summary("空交易測試", empty_summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
