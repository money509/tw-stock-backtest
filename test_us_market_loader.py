"""
test_us_market_loader.py
===========================
針對 us_market_loader.py 的測試。用 monkeypatch 取代 yf.download，不連網路。
驗證：
  1. 回傳的 DataFrame 欄位正確，return_pct 計算正確（今天收盤/昨天收盤-1）
  2. 第一筆日期沒有前一天可比較，return_pct 應該是 NaN，不是報錯或0
  3. 快取生效：第二次呼叫不會再呼叫 yf.download
  4. 下載失敗（yf.download回傳空）時，回傳空 DataFrame 但欄位仍然正確，不會讓呼叫端崩潰
"""

import unittest
import os
import shutil
import pandas as pd

import us_market_loader as uml


class FakeYfDownloadOK:
    def __init__(self, closes, start_date="2026-01-01"):
        self.closes = closes
        self.start_date = start_date
        self.call_count = 0

    def __call__(self, tickers, start, end, interval, progress, auto_adjust):
        self.call_count += 1
        dates = pd.bdate_range(start=self.start_date, periods=len(self.closes))
        df = pd.DataFrame({"Close": self.closes}, index=dates)
        df.index.name = "Date"
        return df


class TestLoadUsMarketReturns(unittest.TestCase):
    def setUp(self):
        self.test_cache_dir = "test_us_market_cache"
        uml.CACHE_DIR = self.test_cache_dir
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def tearDown(self):
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_return_pct_computed_correctly(self):
        fake = FakeYfDownloadOK([4000.0, 4040.0, 3999.6])
        orig = uml.yf.download
        uml.yf.download = fake
        try:
            df = uml.load_us_market_returns("2026-01-01", "2026-01-10")
        finally:
            uml.yf.download = orig

        self.assertEqual(len(df), 3)
        self.assertTrue(pd.isna(df["return_pct"].iloc[0]))
        self.assertAlmostEqual(df["return_pct"].iloc[1], 1.0)   # 4040/4000-1=1%
        self.assertAlmostEqual(df["return_pct"].iloc[2], -1.0)  # 3999.6/4040-1=-1%

    def test_date_format_is_yyyymmdd_string(self):
        fake = FakeYfDownloadOK([4000.0, 4010.0])
        orig = uml.yf.download
        uml.yf.download = fake
        try:
            df = uml.load_us_market_returns("2026-01-01", "2026-01-10")
        finally:
            uml.yf.download = orig
        self.assertTrue(df["date"].iloc[0].isdigit())
        self.assertEqual(len(df["date"].iloc[0]), 8)

    def test_cache_hit_skips_second_download(self):
        fake = FakeYfDownloadOK([4000.0, 4010.0])
        orig = uml.yf.download
        uml.yf.download = fake
        try:
            uml.load_us_market_returns("2026-01-01", "2026-01-10")
            self.assertEqual(fake.call_count, 1)
            uml.load_us_market_returns("2026-01-01", "2026-01-10")
            self.assertEqual(fake.call_count, 1, "第二次呼叫應該直接讀快取，不重新下載")
        finally:
            uml.yf.download = orig

    def test_refresh_forces_redownload(self):
        fake = FakeYfDownloadOK([4000.0, 4010.0])
        orig = uml.yf.download
        uml.yf.download = fake
        try:
            uml.load_us_market_returns("2026-01-01", "2026-01-10")
            uml.load_us_market_returns("2026-01-01", "2026-01-10", refresh=True)
            self.assertEqual(fake.call_count, 2)
        finally:
            uml.yf.download = orig

    def test_download_failure_returns_empty_df_with_correct_columns(self):
        def fake_fail(*args, **kwargs):
            return pd.DataFrame()

        orig = uml.yf.download
        uml.yf.download = fake_fail
        try:
            df = uml.load_us_market_returns("2026-01-01", "2026-01-10")
        finally:
            uml.yf.download = orig

        self.assertTrue(df.empty)
        self.assertListEqual(list(df.columns), ["date", "close", "return_pct"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
