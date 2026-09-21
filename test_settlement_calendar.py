"""
test_settlement_calendar.py
===============================
針對 settlement_calendar.py 的測試，不連網路。驗證：
  1. third_wednesday() 算出來的日期，星期幾一定是星期三，而且是當月第3個
  2. is_settlement_day() / is_near_settlement() 邊界判斷正確
  3. annotate_trades_with_settlement_proximity() / summarize... 正確分組統計
"""
import datetime
import unittest

import settlement_calendar as sc


class TestThirdWednesday(unittest.TestCase):
    def test_is_actually_a_wednesday(self):
        for year, month in [(2026, 1), (2026, 6), (2027, 12), (2024, 2)]:
            d = sc.third_wednesday(year, month)
            self.assertEqual(d.weekday(), 2)

    def test_is_the_third_one_in_month(self):
        # 手動核對已知案例：2026年1月1日是星期四，第一個星期三是1/7，
        # 第二個是1/14，第三個是1/21
        d = sc.third_wednesday(2026, 1)
        self.assertEqual(d, datetime.date(2026, 1, 21))

    def test_another_known_case(self):
        # 2026年9月：9/1是星期二，第一個星期三是9/2，第二個9/9，第三個9/16
        d = sc.third_wednesday(2026, 9)
        self.assertEqual(d, datetime.date(2026, 9, 16))


class TestIsSettlementDay(unittest.TestCase):
    def test_settlement_day_returns_true(self):
        self.assertTrue(sc.is_settlement_day("20260121"))
        self.assertTrue(sc.is_settlement_day(datetime.date(2026, 1, 21)))

    def test_non_settlement_day_returns_false(self):
        self.assertFalse(sc.is_settlement_day("20260122"))
        self.assertFalse(sc.is_settlement_day("20260114"))  # 第二個星期三，不是第三個


class TestIsNearSettlement(unittest.TestCase):
    def test_exact_settlement_day_is_near(self):
        self.assertTrue(sc.is_near_settlement("20260121", days_before=1, days_after=0))

    def test_one_day_before_is_near_with_default_window(self):
        self.assertTrue(sc.is_near_settlement("20260120", days_before=1, days_after=0))

    def test_two_days_before_is_not_near_with_default_window(self):
        self.assertFalse(sc.is_near_settlement("20260119", days_before=1, days_after=0))

    def test_day_after_not_near_by_default(self):
        self.assertFalse(sc.is_near_settlement("20260122", days_before=1, days_after=0))

    def test_day_after_is_near_when_window_includes_it(self):
        self.assertTrue(sc.is_near_settlement("20260122", days_before=1, days_after=1))


class TestSettlementDatesInRange(unittest.TestCase):
    def test_returns_one_per_month(self):
        dates = sc.settlement_dates_in_range(2026, 1, 2026, 3)
        self.assertEqual(len(dates), 3)
        self.assertTrue(all(d.weekday() == 2 for d in dates))

    def test_handles_year_rollover(self):
        dates = sc.settlement_dates_in_range(2026, 11, 2027, 1)
        self.assertEqual(len(dates), 3)
        self.assertEqual(dates[0].month, 11)
        self.assertEqual(dates[1].month, 12)
        self.assertEqual(dates[2].year, 2027)
        self.assertEqual(dates[2].month, 1)


class TestAnnotateAndSummarize(unittest.TestCase):
    def test_annotate_marks_near_settlement_correctly(self):
        trades = [
            {"entry_date": "20260121", "pnl": 100.0},  # 結算日當天
            {"entry_date": "20260105", "pnl": -50.0},  # 遠離結算日
        ]
        annotated = sc.annotate_trades_with_settlement_proximity(trades)
        self.assertTrue(annotated[0]["near_settlement"])
        self.assertFalse(annotated[1]["near_settlement"])

    def test_annotate_does_not_mutate_original(self):
        trades = [{"entry_date": "20260121", "pnl": 100.0}]
        sc.annotate_trades_with_settlement_proximity(trades)
        self.assertNotIn("near_settlement", trades[0])

    def test_summarize_groups_correctly(self):
        trades = [
            {"entry_date": "20260121", "pnl": 100.0},
            {"entry_date": "20260120", "pnl": 200.0},
            {"entry_date": "20260105", "pnl": -50.0},
            {"entry_date": "20260106", "pnl": 30.0},
        ]
        summary = sc.summarize_settlement_proximity_impact(trades)
        self.assertEqual(summary["near_settlement"]["count"], 2)
        self.assertAlmostEqual(summary["near_settlement"]["total_pnl"], 300.0)
        self.assertEqual(summary["not_near_settlement"]["count"], 2)
        self.assertAlmostEqual(summary["not_near_settlement"]["total_pnl"], -20.0)

    def test_summarize_empty_group_does_not_crash(self):
        trades = [{"entry_date": "20260105", "pnl": -50.0}]
        summary = sc.summarize_settlement_proximity_impact(trades)
        self.assertEqual(summary["near_settlement"]["count"], 0)
        self.assertEqual(summary["near_settlement"]["win_rate"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
