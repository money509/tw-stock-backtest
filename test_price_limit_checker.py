"""
test_price_limit_checker.py
===============================
針對 price_limit_checker.py 的測試，純數學計算，不連網路。
"""
import unittest

import price_limit_checker as plc


class TestPriceLimitBand(unittest.TestCase):
    def test_default_ten_percent_band(self):
        lower, upper = plc.price_limit_band(100.0)
        self.assertAlmostEqual(lower, 90.0)
        self.assertAlmostEqual(upper, 110.0)

    def test_custom_limit_pct(self):
        lower, upper = plc.price_limit_band(100.0, limit_pct=7.0)
        self.assertAlmostEqual(lower, 93.0)
        self.assertAlmostEqual(upper, 107.0)

    def test_non_positive_prev_close_raises(self):
        with self.assertRaises(ValueError):
            plc.price_limit_band(0.0)
        with self.assertRaises(ValueError):
            plc.price_limit_band(-5.0)
        with self.assertRaises(ValueError):
            plc.price_limit_band(None)


class TestIsWithinLimitBand(unittest.TestCase):
    def test_price_inside_band(self):
        self.assertTrue(plc.is_within_limit_band(105.0, 100.0))

    def test_price_at_exact_boundary_is_within(self):
        self.assertTrue(plc.is_within_limit_band(110.0, 100.0))
        self.assertTrue(plc.is_within_limit_band(90.0, 100.0))

    def test_price_outside_band(self):
        self.assertFalse(plc.is_within_limit_band(111.0, 100.0))
        self.assertFalse(plc.is_within_limit_band(89.0, 100.0))


class TestClampToLimitBand(unittest.TestCase):
    def test_price_inside_band_unchanged(self):
        self.assertAlmostEqual(plc.clamp_to_limit_band(105.0, 100.0), 105.0)

    def test_price_above_band_clamped_to_upper(self):
        self.assertAlmostEqual(plc.clamp_to_limit_band(150.0, 100.0), 110.0)

    def test_price_below_band_clamped_to_lower(self):
        self.assertAlmostEqual(plc.clamp_to_limit_band(50.0, 100.0), 90.0)


class TestCheckTradeExposure(unittest.TestCase):
    def test_all_prices_within_band(self):
        result = plc.check_trade_exposure(
            entry_price=102.0, prev_close=100.0, stop_price=99.0, target_price=108.0)
        self.assertTrue(result["entry_within_band"])
        self.assertTrue(result["stop_within_band"])
        self.assertTrue(result["target_within_band"])
        self.assertAlmostEqual(result["limit_down"], 90.0)
        self.assertAlmostEqual(result["limit_up"], 110.0)

    def test_target_exceeds_limit_up(self):
        # 進場價102，3倍ATR若ATR夠大，停利價可能算出超過漲停110
        result = plc.check_trade_exposure(
            entry_price=102.0, prev_close=100.0, stop_price=99.0, target_price=115.0)
        self.assertTrue(result["entry_within_band"])
        self.assertTrue(result["stop_within_band"])
        self.assertFalse(result["target_within_band"])

    def test_none_price_returns_none_not_false(self):
        result = plc.check_trade_exposure(
            entry_price=102.0, prev_close=100.0, stop_price=None, target_price=None)
        self.assertIsNone(result["stop_within_band"])
        self.assertIsNone(result["target_within_band"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
