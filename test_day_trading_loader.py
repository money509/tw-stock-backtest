"""
test_day_trading_loader.py
============================
針對 day_trading_loader.py 的合成/模擬資料測試，沿用本專案對 chip_data_loader.py
已經證實有效的測試模式：不打真實 API，用 monkeypatch 取代 _fetch_one_day，
驗證：
  1. FetchFailed 會被重試恰好 MAX_RETRIES+1 次，且從不寫入快取
  2. 已存在的快取檔案，rerun 時會被直接讀取（不會再呼叫下載函式）
  3. universe_codes 過濾能正確剔除不在名單內的代碼
  4. 「確定無資料」（回傳 []）的日期會被快取為空結果，且不算失敗
  5. compute_day_trading_ratio 正確計算比例、且 volume=0 或缺值時給 NaN
"""

import os
import shutil
import unittest
import datetime
import pandas as pd

import day_trading_loader as dtl


class TestRetryLogic(unittest.TestCase):
    def setUp(self):
        self.test_cache_dir = "test_day_trading_cache_retry"
        dtl.CACHE_DIR = self.test_cache_dir
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
        self.call_count = 0

    def tearDown(self):
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_fetch_failed_retried_exact_times_and_not_cached(self):
        """FetchFailed 應該被重試 MAX_RETRIES 次（總共 MAX_RETRIES+1 次嘗試），
        全部失敗後不應該寫入快取檔案。"""

        def always_fail(date_str):
            self.call_count += 1
            raise dtl.FetchFailed(f"模擬失敗 {date_str}")

        orig_fetch = dtl._fetch_one_day
        orig_sleep_delay = dtl.REQUEST_DELAY_SEC
        orig_backoff = dtl.RETRY_BACKOFF_BASE_SEC
        dtl._fetch_one_day = always_fail
        dtl.REQUEST_DELAY_SEC = 0  # 加速測試
        dtl.RETRY_BACKOFF_BASE_SEC = 0

        try:
            with self.assertRaises(dtl.FetchFailed):
                dtl._fetch_one_day_with_retry("20260101")
            self.assertEqual(
                self.call_count, dtl.MAX_RETRIES + 1,
                f"應該嘗試 {dtl.MAX_RETRIES + 1} 次（1 次原始 + {dtl.MAX_RETRIES} 次重試）"
            )

            # 透過 load_day_trading_data 確認：完全失敗的日期不會產生快取檔
            df = dtl.load_day_trading_data(
                datetime.date(2026, 1, 1), datetime.date(2026, 1, 1)
            )
            self.assertTrue(df.empty)
            self.assertFalse(
                os.path.exists(dtl._cache_path("20260101")),
                "全部重試失敗的日期不應該被寫入快取"
            )
        finally:
            dtl._fetch_one_day = orig_fetch
            dtl.REQUEST_DELAY_SEC = orig_sleep_delay
            dtl.RETRY_BACKOFF_BASE_SEC = orig_backoff

    def test_succeeds_after_transient_failures(self):
        """前兩次失敗、第三次成功時，應該回傳正確資料且不再繼續重試。"""
        attempts = {"n": 0}

        def fail_twice_then_succeed(date_str):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise dtl.FetchFailed("暫時性失敗")
            return [{"date": date_str, "code": "2330", "name": "台積電",
                     "day_trading_shares": 12345}]

        orig_fetch = dtl._fetch_one_day
        orig_backoff = dtl.RETRY_BACKOFF_BASE_SEC
        dtl._fetch_one_day = fail_twice_then_succeed
        dtl.RETRY_BACKOFF_BASE_SEC = 0

        try:
            result = dtl._fetch_one_day_with_retry("20260102")
            self.assertEqual(attempts["n"], 3)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["code"], "2330")
        finally:
            dtl._fetch_one_day = orig_fetch
            dtl.RETRY_BACKOFF_BASE_SEC = orig_backoff


class TestCaching(unittest.TestCase):
    def setUp(self):
        self.test_cache_dir = "test_day_trading_cache_cache"
        dtl.CACHE_DIR = self.test_cache_dir
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
        self.call_count = 0

    def tearDown(self):
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_cached_day_not_refetched(self):
        """第一次下載後應該產生快取檔；第二次呼叫（同一天）不應該再呼叫下載函式。"""

        def fake_fetch(date_str):
            self.call_count += 1
            return [{"date": date_str, "code": "2317", "name": "鴻海",
                     "day_trading_shares": 999}]

        orig_fetch = dtl._fetch_one_day
        orig_delay = dtl.REQUEST_DELAY_SEC
        dtl._fetch_one_day = fake_fetch
        dtl.REQUEST_DELAY_SEC = 0

        try:
            d = datetime.date(2026, 1, 5)  # 週一，會被視為交易日
            df1 = dtl.load_day_trading_data(d, d)
            self.assertEqual(self.call_count, 1)
            self.assertTrue(os.path.exists(dtl._cache_path("20260105")))
            self.assertEqual(len(df1), 1)

            # 第二次呼叫：不應該再打「下載」函式
            df2 = dtl.load_day_trading_data(d, d)
            self.assertEqual(self.call_count, 1, "已快取的日期不應該重新下載")
            self.assertEqual(len(df2), 1)
        finally:
            dtl._fetch_one_day = orig_fetch
            dtl.REQUEST_DELAY_SEC = orig_delay

    def test_refresh_forces_redownload(self):
        """refresh=True 應該強制重新下載，即使已有快取。"""

        def fake_fetch(date_str):
            self.call_count += 1
            return [{"date": date_str, "code": "2317", "name": "鴻海",
                     "day_trading_shares": self.call_count * 100}]

        orig_fetch = dtl._fetch_one_day
        orig_delay = dtl.REQUEST_DELAY_SEC
        dtl._fetch_one_day = fake_fetch
        dtl.REQUEST_DELAY_SEC = 0

        try:
            d = datetime.date(2026, 1, 6)  # 週二
            dtl.load_day_trading_data(d, d)
            self.assertEqual(self.call_count, 1)
            dtl.load_day_trading_data(d, d, refresh=True)
            self.assertEqual(self.call_count, 2, "refresh=True 應該強制重新下載")
        finally:
            dtl._fetch_one_day = orig_fetch
            dtl.REQUEST_DELAY_SEC = orig_delay

    def test_no_data_day_is_cached_as_empty_and_not_retried(self):
        """回傳 [] 代表「確定無資料」，應該被快取（空），且不視為失敗、不觸發重試。"""
        calls = {"n": 0}

        def fake_fetch(date_str):
            calls["n"] += 1
            return []  # 確定無資料

        orig_fetch = dtl._fetch_one_day
        orig_delay = dtl.REQUEST_DELAY_SEC
        dtl._fetch_one_day = fake_fetch
        dtl.REQUEST_DELAY_SEC = 0

        try:
            d = datetime.date(2026, 1, 7)  # 週三
            df = dtl.load_day_trading_data(d, d)
            self.assertTrue(df.empty)
            self.assertEqual(calls["n"], 1, "確定無資料不應該重試")
            self.assertTrue(os.path.exists(dtl._cache_path("20260107")))

            # 再次呼叫：應該直接讀快取，不再呼叫下載函式
            dtl.load_day_trading_data(d, d)
            self.assertEqual(calls["n"], 1)
        finally:
            dtl._fetch_one_day = orig_fetch
            dtl.REQUEST_DELAY_SEC = orig_delay


class TestUniverseFiltering(unittest.TestCase):
    def setUp(self):
        self.test_cache_dir = "test_day_trading_cache_universe"
        dtl.CACHE_DIR = self.test_cache_dir
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def tearDown(self):
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_universe_codes_filters_out_unlisted_codes(self):
        def fake_fetch(date_str):
            return [
                {"date": date_str, "code": "2330", "name": "台積電", "day_trading_shares": 100},
                {"date": date_str, "code": "2317", "name": "鴻海", "day_trading_shares": 200},
                {"date": date_str, "code": "0050", "name": "元大台灣50(ETF)", "day_trading_shares": 300},
                {"date": date_str, "code": "091912", "name": "某權證", "day_trading_shares": 400},
            ]

        orig_fetch = dtl._fetch_one_day
        orig_delay = dtl.REQUEST_DELAY_SEC
        dtl._fetch_one_day = fake_fetch
        dtl.REQUEST_DELAY_SEC = 0

        try:
            d = datetime.date(2026, 1, 8)  # 週四
            universe = {"2330", "2317"}
            df = dtl.load_day_trading_data(d, d, universe_codes=universe)
            self.assertEqual(set(df["code"]), {"2330", "2317"})
            self.assertEqual(len(df), 2)
        finally:
            dtl._fetch_one_day = orig_fetch
            dtl.REQUEST_DELAY_SEC = orig_delay

    def test_no_universe_codes_keeps_everything(self):
        def fake_fetch(date_str):
            return [
                {"date": date_str, "code": "2330", "name": "台積電", "day_trading_shares": 100},
                {"date": date_str, "code": "091912", "name": "某權證", "day_trading_shares": 400},
            ]

        orig_fetch = dtl._fetch_one_day
        orig_delay = dtl.REQUEST_DELAY_SEC
        dtl._fetch_one_day = fake_fetch
        dtl.REQUEST_DELAY_SEC = 0

        try:
            d = datetime.date(2026, 1, 9)  # 週五
            df = dtl.load_day_trading_data(d, d, universe_codes=None)
            self.assertEqual(len(df), 2)
        finally:
            dtl._fetch_one_day = orig_fetch
            dtl.REQUEST_DELAY_SEC = orig_delay


class TestRatioComputation(unittest.TestCase):
    def test_ratio_basic(self):
        day_trading_df = pd.DataFrame([
            {"date": "20260101", "code": "2330", "name": "台積電", "day_trading_shares": 1000},
            {"date": "20260101", "code": "2317", "name": "鴻海", "day_trading_shares": 500},
        ])
        volume_df = pd.DataFrame([
            {"date": "20260101", "code": "2330", "volume": 10000},
            {"date": "20260101", "code": "2317", "volume": 0},  # volume 0 -> NaN ratio
            # 2317 有 volume=0，2330 正常；並多一筆無關代碼確保 merge 不受影響
            {"date": "20260101", "code": "9999", "volume": 5000},
        ])

        merged = dtl.compute_day_trading_ratio(day_trading_df, volume_df)
        row_2330 = merged[merged["code"] == "2330"].iloc[0]
        row_2317 = merged[merged["code"] == "2317"].iloc[0]

        self.assertAlmostEqual(row_2330["day_trading_ratio"], 0.1)
        self.assertTrue(pd.isna(row_2317["day_trading_ratio"]))

    def test_ratio_missing_volume_gives_nan(self):
        day_trading_df = pd.DataFrame([
            {"date": "20260101", "code": "1101", "name": "台泥", "day_trading_shares": 300},
        ])
        volume_df = pd.DataFrame([
            {"date": "20260101", "code": "9999", "volume": 12345},  # 沒有 1101 的成交量資料
        ])
        merged = dtl.compute_day_trading_ratio(day_trading_df, volume_df)
        self.assertTrue(pd.isna(merged.iloc[0]["day_trading_ratio"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
