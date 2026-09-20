"""
test_overnight_chip_adapter.py
=================================
針對 overnight_chip_adapter.py 的合成資料測試，驗證：
  1. 股數 -> 金額 -> 比重 的換算公式正確
  2. 日期格式轉換正確（Timestamp -> YYYYMMDD 字串），能跟 indicators_by_code 對上
  3. 成交金額 <=0 的日期會被跳過，不會產生除以零或負比重的錯誤結果
  4. 沒有對應股價資料的股票代碼會被安靜跳過，不會報錯
  5. 多檔股票同時處理時彼此不會互相污染
"""

import unittest
import pandas as pd

import overnight_chip_adapter as adapter


def make_indicators(dates, closes, volumes):
    return pd.DataFrame({
        "date": dates,
        "close": closes,
        "turnover_value": [c * v for c, v in zip(closes, volumes)],
    })


class TestBuildInstitutionalRatioDfs(unittest.TestCase):
    def test_basic_ratio_computation(self):
        dates_ts = pd.to_datetime(["2026-01-05", "2026-01-06"])
        chip_df = pd.DataFrame({
            "foreign_net": [1000, -500],
            "trust_net": [200, 300],
            "dealer_net": [0, 0],
            "total_net": [1200, -200],
        }, index=dates_ts)
        chip_df.index.name = "date"

        indicators = make_indicators(
            dates=["20260105", "20260106"],
            closes=[100.0, 105.0],
            volumes=[10000, 20000],
        )

        chip_data = {"2330": chip_df}
        indicators_by_code = {"2330": indicators}

        foreign_df, trust_df = adapter.build_institutional_ratio_dfs(chip_data, indicators_by_code)

        self.assertEqual(len(foreign_df), 2)
        self.assertEqual(len(trust_df), 2)

        row0 = foreign_df[foreign_df["date"] == "20260105"].iloc[0]
        expected_ratio0 = (1000 * 100.0) / (100.0 * 10000)
        self.assertAlmostEqual(row0["ratio"], expected_ratio0)
        self.assertEqual(row0["code"], "2330")

        row1 = foreign_df[foreign_df["date"] == "20260106"].iloc[0]
        expected_ratio1 = (-500 * 105.0) / (105.0 * 20000)
        self.assertAlmostEqual(row1["ratio"], expected_ratio1)
        self.assertLess(row1["ratio"], 0)  # 賣超應該是負比重

        trust_row0 = trust_df[trust_df["date"] == "20260105"].iloc[0]
        expected_trust0 = (200 * 100.0) / (100.0 * 10000)
        self.assertAlmostEqual(trust_row0["ratio"], expected_trust0)

    def test_zero_turnover_day_is_skipped(self):
        dates_ts = pd.to_datetime(["2026-01-05", "2026-01-06"])
        chip_df = pd.DataFrame({
            "foreign_net": [1000, 2000],
            "trust_net": [100, 200],
            "dealer_net": [0, 0],
            "total_net": [1100, 2200],
        }, index=dates_ts)
        chip_df.index.name = "date"

        # 第二天成交量為 0 -> turnover_value = 0，應該被排除
        indicators = make_indicators(
            dates=["20260105", "20260106"],
            closes=[100.0, 105.0],
            volumes=[10000, 0],
        )

        chip_data = {"2330": chip_df}
        indicators_by_code = {"2330": indicators}

        foreign_df, _ = adapter.build_institutional_ratio_dfs(chip_data, indicators_by_code)
        self.assertEqual(len(foreign_df), 1)
        self.assertEqual(foreign_df.iloc[0]["date"], "20260105")

    def test_missing_price_data_for_code_is_skipped_silently(self):
        dates_ts = pd.to_datetime(["2026-01-05"])
        chip_df = pd.DataFrame({
            "foreign_net": [1000], "trust_net": [100],
            "dealer_net": [0], "total_net": [1100],
        }, index=dates_ts)
        chip_df.index.name = "date"

        chip_data = {"9999": chip_df}  # 這個代碼沒有對應的股價資料
        indicators_by_code = {}  # 空的

        foreign_df, trust_df = adapter.build_institutional_ratio_dfs(chip_data, indicators_by_code)
        self.assertTrue(foreign_df.empty)
        self.assertTrue(trust_df.empty)

    def test_date_only_in_chip_not_in_price_is_dropped(self):
        # 籌碼資料涵蓋範圍比股價資料還長的情況：多出來的那天應該被丟掉，不報錯
        dates_ts = pd.to_datetime(["2026-01-05", "2026-01-07"])  # 01-07 股價沒有
        chip_df = pd.DataFrame({
            "foreign_net": [1000, 5000], "trust_net": [100, 500],
            "dealer_net": [0, 0], "total_net": [1100, 5500],
        }, index=dates_ts)
        chip_df.index.name = "date"

        indicators = make_indicators(dates=["20260105"], closes=[100.0], volumes=[10000])

        chip_data = {"2330": chip_df}
        indicators_by_code = {"2330": indicators}

        foreign_df, _ = adapter.build_institutional_ratio_dfs(chip_data, indicators_by_code)
        self.assertEqual(len(foreign_df), 1)
        self.assertEqual(foreign_df.iloc[0]["date"], "20260105")

    def test_multiple_codes_do_not_cross_contaminate(self):
        dates_ts = pd.to_datetime(["2026-01-05"])
        chip_a = pd.DataFrame({
            "foreign_net": [1000], "trust_net": [0], "dealer_net": [0], "total_net": [1000]
        }, index=dates_ts)
        chip_a.index.name = "date"
        chip_b = pd.DataFrame({
            "foreign_net": [-2000], "trust_net": [0], "dealer_net": [0], "total_net": [-2000]
        }, index=dates_ts)
        chip_b.index.name = "date"

        ind_a = make_indicators(["20260105"], [50.0], [1000])
        ind_b = make_indicators(["20260105"], [200.0], [500])

        chip_data = {"1101": chip_a, "2882": chip_b}
        indicators_by_code = {"1101": ind_a, "2882": ind_b}

        foreign_df, _ = adapter.build_institutional_ratio_dfs(chip_data, indicators_by_code)
        self.assertEqual(len(foreign_df), 2)

        row_a = foreign_df[foreign_df["code"] == "1101"].iloc[0]
        row_b = foreign_df[foreign_df["code"] == "2882"].iloc[0]
        self.assertGreater(row_a["ratio"], 0)
        self.assertLess(row_b["ratio"], 0)
        self.assertAlmostEqual(row_a["ratio"], (1000 * 50.0) / (50.0 * 1000))
        self.assertAlmostEqual(row_b["ratio"], (-2000 * 200.0) / (200.0 * 500))

    def test_empty_chip_data_returns_empty_frames_with_right_columns(self):
        foreign_df, trust_df = adapter.build_institutional_ratio_dfs({}, {})
        self.assertTrue(foreign_df.empty)
        self.assertTrue(trust_df.empty)
        self.assertListEqual(list(foreign_df.columns), ["date", "code", "ratio"])
        self.assertListEqual(list(trust_df.columns), ["date", "code", "ratio"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
