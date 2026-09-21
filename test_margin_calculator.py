"""
test_margin_calculator.py
============================
針對 margin_calculator.py 的合成資料測試。不連網路。驗證：
  1. estimate_trade_margin() 算法正確(價格*契約股數*保證金比例*口數)
  2. annotate_trades_with_margin() 幫每筆交易正確加上margin_required，
     不在taifex_universe清單裡的代碼設為None，不是0
  3. summarize_capital_requirement() 正確抓出「單日保證金需求最高」的那一天，
     報酬率算法正確，max_concurrent_positions正確截斷
"""
import unittest

import margin_calculator as mc
import taifex_universe as tu


class TestEstimateTradeMargin(unittest.TestCase):
    def test_matches_taifex_universe_formula(self):
        # 2330: has_mini=True(100股/口), margin_ratio=0.135
        margin = mc.estimate_trade_margin("2330", 900.0, lots=1)
        expected = 900.0 * 100 * 0.135 * 1
        self.assertAlmostEqual(margin, expected)

    def test_multiple_lots_scales_linearly(self):
        margin_1lot = mc.estimate_trade_margin("2330", 900.0, lots=1)
        margin_3lots = mc.estimate_trade_margin("2330", 900.0, lots=3)
        self.assertAlmostEqual(margin_3lots, margin_1lot * 3)

    def test_unknown_code_raises(self):
        with self.assertRaises(ValueError):
            mc.estimate_trade_margin("999999", 100.0, lots=1)

    def test_default_margin_ratio_used_for_unlisted_ratio_code(self):
        # 找一個在STOCK_FUTURES_UNIVERSE裡、但不在MARGIN_RATIO_TABLE裡的代碼，
        # 應該要用DEFAULT_MARGIN_RATIO，而不是丟例外
        code_without_ratio = next(
            c for c in tu.STOCK_FUTURES_UNIVERSE if c not in tu.MARGIN_RATIO_TABLE
        )
        margin = mc.estimate_trade_margin(code_without_ratio, 100.0, lots=1)
        mult = 100 if tu.STOCK_FUTURES_UNIVERSE[code_without_ratio]["has_mini"] else 2000
        expected = 100.0 * mult * tu.DEFAULT_MARGIN_RATIO * 1
        self.assertAlmostEqual(margin, expected)


class TestAnnotateTradesWithMargin(unittest.TestCase):
    def test_known_code_gets_margin_value(self):
        trades = [{"code": "2330", "entry_date": "20260101", "entry_price": 900.0, "pnl": 1000.0}]
        annotated = mc.annotate_trades_with_margin(trades)
        self.assertIsNotNone(annotated[0]["margin_required"])
        self.assertGreater(annotated[0]["margin_required"], 0)

    def test_unknown_code_gets_none_not_zero(self):
        trades = [{"code": "999999", "entry_date": "20260101", "entry_price": 100.0, "pnl": 1000.0}]
        annotated = mc.annotate_trades_with_margin(trades)
        self.assertIsNone(annotated[0]["margin_required"])

    def test_does_not_mutate_original_trades(self):
        trades = [{"code": "2330", "entry_date": "20260101", "entry_price": 900.0, "pnl": 1000.0}]
        mc.annotate_trades_with_margin(trades)
        self.assertNotIn("margin_required", trades[0])


class TestSummarizeCapitalRequirement(unittest.TestCase):
    def test_empty_trades_returns_zeros(self):
        summary = mc.summarize_capital_requirement([])
        self.assertEqual(summary["peak_daily_margin"], 0.0)
        self.assertIsNone(summary["peak_day"])
        self.assertIsNone(summary["return_on_peak_capital_pct"])

    def test_finds_peak_day_across_multiple_days(self):
        trades = [
            # day1: 只有1筆
            {"code": "2330", "entry_date": "20260101", "entry_price": 900.0, "pnl": 1000.0},
            # day2: 2筆，保證金加總應該比day1高
            {"code": "2330", "entry_date": "20260102", "entry_price": 900.0, "pnl": 500.0},
            {"code": "2454", "entry_date": "20260102", "entry_price": 1000.0, "pnl": -300.0},
        ]
        summary = mc.summarize_capital_requirement(trades)
        self.assertEqual(summary["peak_day"], "20260102")

        margin_2330 = mc.estimate_trade_margin("2330", 900.0)
        margin_2454 = mc.estimate_trade_margin("2454", 1000.0)
        self.assertAlmostEqual(summary["peak_daily_margin"], margin_2330 + margin_2454)

    def test_return_on_peak_capital_uses_valid_trades_pnl(self):
        trades = [
            {"code": "2330", "entry_date": "20260101", "entry_price": 900.0, "pnl": 2000.0},
        ]
        summary = mc.summarize_capital_requirement(trades)
        margin = mc.estimate_trade_margin("2330", 900.0)
        expected_return_pct = 2000.0 / margin * 100.0
        self.assertAlmostEqual(summary["return_on_peak_capital_pct"], expected_return_pct)

    def test_unknown_codes_excluded_from_margin_but_listed(self):
        trades = [
            {"code": "2330", "entry_date": "20260101", "entry_price": 900.0, "pnl": 1000.0},
            {"code": "999999", "entry_date": "20260101", "entry_price": 50.0, "pnl": 200.0},
        ]
        summary = mc.summarize_capital_requirement(trades)
        self.assertIn("999999", summary["skipped_codes_not_in_universe"])
        self.assertEqual(summary["total_trades_with_margin"], 1)
        # 保證金/損益只算已知代碼那一筆，不含999999
        self.assertAlmostEqual(summary["total_pnl_of_counted_trades"], 1000.0)

    def test_max_concurrent_positions_truncates_to_highest_margin(self):
        trades = [
            {"code": "2330", "entry_date": "20260101", "entry_price": 100.0, "pnl": 100.0},   # 保證金較小
            {"code": "2454", "entry_date": "20260101", "entry_price": 2000.0, "pnl": 200.0},  # 保證金較大
        ]
        summary_no_limit = mc.summarize_capital_requirement(trades)
        summary_limited = mc.summarize_capital_requirement(trades, max_concurrent_positions=1)

        margin_2330 = mc.estimate_trade_margin("2330", 100.0)
        margin_2454 = mc.estimate_trade_margin("2454", 2000.0)
        self.assertAlmostEqual(summary_no_limit["peak_daily_margin"], margin_2330 + margin_2454)
        # 限制成只算1筆，應該保留保證金較高的那筆(2454)
        self.assertAlmostEqual(summary_limited["peak_daily_margin"], margin_2454)
        self.assertAlmostEqual(summary_limited["total_pnl_of_counted_trades"], 200.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
