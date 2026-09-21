"""
test_futures_liquidity_checker.py
=====================================
針對 futures_liquidity_checker.py 的合成資料測試，不連網路。驗證：
  1. annotate_trades_with_liquidity() 正確標記已知/未知、偏低/正常流動性，
     且「沒有提供futures_volume_by_code」時誠實回報liquidity_known=False，
     不是假裝流動性正常
  2. summarize_liquidity_coverage() 正確算coverage_pct跟low_liquidity統計，
     空trades/全部未知時不crash，且不會把「涵蓋率低」誤報成「流動性沒問題」
"""
import unittest

import futures_liquidity_checker as flc


class TestAnnotateTradesWithLiquidity(unittest.TestCase):
    def test_no_volume_data_marks_everything_unknown(self):
        trades = [{"code": "2330", "pnl": 100.0}]
        annotated = flc.annotate_trades_with_liquidity(trades)
        self.assertFalse(annotated[0]["liquidity_known"])
        self.assertIsNone(annotated[0]["liquidity_flag"])
        self.assertIsNone(annotated[0]["futures_avg_daily_volume"])

    def test_known_code_above_threshold_flagged_false(self):
        trades = [{"code": "2330", "pnl": 100.0}]
        annotated = flc.annotate_trades_with_liquidity(
            trades, futures_volume_by_code={"2330": 500})
        self.assertTrue(annotated[0]["liquidity_known"])
        self.assertFalse(annotated[0]["liquidity_flag"])
        self.assertEqual(annotated[0]["futures_avg_daily_volume"], 500)

    def test_known_code_below_threshold_flagged_true(self):
        trades = [{"code": "9999", "pnl": 100.0}]
        annotated = flc.annotate_trades_with_liquidity(
            trades, futures_volume_by_code={"9999": 5})
        self.assertTrue(annotated[0]["liquidity_known"])
        self.assertTrue(annotated[0]["liquidity_flag"])

    def test_custom_threshold_overrides_default(self):
        trades = [{"code": "2330", "pnl": 100.0}]
        annotated = flc.annotate_trades_with_liquidity(
            trades, futures_volume_by_code={"2330": 80}, min_daily_volume=100)
        self.assertTrue(annotated[0]["liquidity_flag"])  # 80 < 100(自訂門檻)

        annotated_default = flc.annotate_trades_with_liquidity(
            trades, futures_volume_by_code={"2330": 80})
        self.assertFalse(annotated_default[0]["liquidity_flag"])  # 80 >= 50(預設門檻)

    def test_does_not_mutate_original_trades(self):
        trades = [{"code": "2330", "pnl": 100.0}]
        flc.annotate_trades_with_liquidity(trades, futures_volume_by_code={"2330": 500})
        self.assertNotIn("liquidity_known", trades[0])

    def test_mixed_known_and_unknown_codes(self):
        trades = [
            {"code": "2330", "pnl": 100.0},
            {"code": "9999", "pnl": -50.0},
        ]
        annotated = flc.annotate_trades_with_liquidity(
            trades, futures_volume_by_code={"2330": 500})
        self.assertTrue(annotated[0]["liquidity_known"])
        self.assertFalse(annotated[1]["liquidity_known"])
        self.assertIsNone(annotated[1]["liquidity_flag"])


class TestSummarizeLiquidityCoverage(unittest.TestCase):
    def test_empty_trades_does_not_crash(self):
        summary = flc.summarize_liquidity_coverage([])
        self.assertEqual(summary["total_trades"], 0)
        self.assertIsNone(summary["coverage_pct"])
        self.assertEqual(summary["low_liquidity_codes"], [])

    def test_no_volume_data_reports_zero_coverage_not_all_clear(self):
        trades = [{"code": "2330", "pnl": 100.0}, {"code": "2317", "pnl": 200.0}]
        summary = flc.summarize_liquidity_coverage(trades)
        self.assertEqual(summary["coverage_pct"], 0.0)
        self.assertIsNone(summary["low_liquidity_pct_of_known"])
        self.assertEqual(summary["low_liquidity_count"], 0)

    def test_partial_coverage_computes_correctly(self):
        trades = [
            {"code": "2330", "pnl": 100.0},
            {"code": "2317", "pnl": 200.0},
            {"code": "9999", "pnl": -50.0},
            {"code": "9999", "pnl": 30.0},
        ]
        summary = flc.summarize_liquidity_coverage(
            trades, futures_volume_by_code={"2330": 500, "9999": 5})
        # 4筆交易，2330+9999*2共3筆有資料(2317沒有) -> coverage 75%
        self.assertAlmostEqual(summary["coverage_pct"], 75.0)
        self.assertEqual(summary["known_count"], 3)
        # 已知的3筆裡，9999的2筆流動性偏低 -> 2/3
        self.assertAlmostEqual(summary["low_liquidity_pct_of_known"], 2 / 3 * 100)
        self.assertEqual(summary["low_liquidity_codes"], ["9999"])

    def test_full_coverage_all_liquid(self):
        trades = [{"code": "2330", "pnl": 100.0}, {"code": "2317", "pnl": 200.0}]
        summary = flc.summarize_liquidity_coverage(
            trades, futures_volume_by_code={"2330": 500, "2317": 600})
        self.assertAlmostEqual(summary["coverage_pct"], 100.0)
        self.assertEqual(summary["low_liquidity_count"], 0)
        self.assertAlmostEqual(summary["low_liquidity_pct_of_known"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
