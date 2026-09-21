"""
test_dividend_data_loader.py
================================
針對 dividend_data_loader.py 的合成資料測試。不連真實網路——用假的
yf.Ticker(sym).actions 取代真實的 yfinance API，驗證：
  1. load_dividend_events() 正確從 actions 的 Dividends/Stock Splits 欄位
     萃取出除權息事件日期，轉成 YYYYMMDD 字串
  2. 只保留 [start, end] 區間內的事件，區間外的日期不算
  3. 本機 CSV 快取：第二次呼叫不會再打 yfinance
  4. 單一標的例外(下載失敗)不會讓整個流程崩潰，該檔回傳空集合，其餘標的正常
  5. _symbol_for()：舊格式(字串entry)用 data_loader.OTC_STOCKS 判斷，
     新格式(dict帶otc欄位)直接讀取判斷
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import pandas as pd

import dividend_data_loader as ddl


def make_fake_actions(events):
    """events: list[(date_str YYYY-MM-DD, dividends_value, splits_value)]"""
    if not events:
        return pd.DataFrame(columns=["Dividends", "Stock Splits"])
    idx = pd.DatetimeIndex([e[0] for e in events])
    return pd.DataFrame({
        "Dividends": [e[1] for e in events],
        "Stock Splits": [e[2] for e in events],
    }, index=idx)


class FakeTicker:
    def __init__(self, actions_by_symbol, fail_symbols=None):
        self.actions_by_symbol = actions_by_symbol
        self.fail_symbols = fail_symbols or set()

    def __call__(self, sym):
        if sym in self.fail_symbols:
            raise RuntimeError(f"模擬 {sym} 下載失敗")
        return _FakeTickerInstance(self.actions_by_symbol.get(sym, pd.DataFrame()))


class _FakeTickerInstance:
    def __init__(self, actions_df):
        self.actions = actions_df


class TestLoadDividendEvents(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patcher_cache_dir = mock.patch.object(ddl, "CACHE_DIR", self.tmpdir)
        self.patcher_cache_dir.start()

    def tearDown(self):
        self.patcher_cache_dir.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_extracts_dividend_and_split_dates_within_range(self):
        actions_by_symbol = {
            "2330.TW": make_fake_actions([
                ("2026-03-15", 2.5, 0),   # 現金股利，區間內
                ("2026-07-01", 0, 2.0),   # 股票分割，區間內
                ("2020-01-01", 1.0, 0),   # 區間外，應該被排除
            ]),
        }
        fake_ticker = FakeTicker(actions_by_symbol)
        with mock.patch.object(ddl.yf, "Ticker", fake_ticker):
            result = ddl.load_dividend_events(
                {"2330": "台積電"}, "2026-01-01", "2026-12-31", refresh=True,
            )
        self.assertEqual(result["2330"], {"20260315", "20260701"})

    def test_zero_dividend_rows_not_counted_as_events(self):
        actions_by_symbol = {
            "2330.TW": make_fake_actions([
                ("2026-03-15", 0, 0),  # 兩欄都是0，不是真正的事件
            ]),
        }
        fake_ticker = FakeTicker(actions_by_symbol)
        with mock.patch.object(ddl.yf, "Ticker", fake_ticker):
            result = ddl.load_dividend_events(
                {"2330": "台積電"}, "2026-01-01", "2026-12-31", refresh=True,
            )
        self.assertEqual(result["2330"], set())

    def test_multiple_codes_independent(self):
        actions_by_symbol = {
            "2330.TW": make_fake_actions([("2026-03-15", 2.5, 0)]),
            "2317.TW": make_fake_actions([("2026-06-01", 1.0, 0)]),
        }
        fake_ticker = FakeTicker(actions_by_symbol)
        with mock.patch.object(ddl.yf, "Ticker", fake_ticker):
            result = ddl.load_dividend_events(
                {"2330": "台積電", "2317": "鴻海"}, "2026-01-01", "2026-12-31", refresh=True,
            )
        self.assertEqual(result["2330"], {"20260315"})
        self.assertEqual(result["2317"], {"20260601"})

    def test_empty_actions_returns_empty_set(self):
        fake_ticker = FakeTicker({"2330.TW": pd.DataFrame()})
        with mock.patch.object(ddl.yf, "Ticker", fake_ticker):
            result = ddl.load_dividend_events(
                {"2330": "台積電"}, "2026-01-01", "2026-12-31", refresh=True,
            )
        self.assertEqual(result["2330"], set())

    def test_failed_symbol_returns_empty_set_and_does_not_crash(self):
        actions_by_symbol = {"2317.TW": make_fake_actions([("2026-06-01", 1.0, 0)])}
        fake_ticker = FakeTicker(actions_by_symbol, fail_symbols={"2330.TW"})
        with mock.patch.object(ddl.yf, "Ticker", fake_ticker), \
             mock.patch.object(ddl.time, "sleep", return_value=None):
            result = ddl.load_dividend_events(
                {"2330": "台積電", "2317": "鴻海"}, "2026-01-01", "2026-12-31", refresh=True,
            )
        self.assertEqual(result["2330"], set())  # 失敗的那檔回傳空集合，不中斷
        self.assertEqual(result["2317"], {"20260601"})  # 其他檔正常

    def test_cache_hit_skips_network_call(self):
        actions_by_symbol = {"2330.TW": make_fake_actions([("2026-03-15", 2.5, 0)])}
        fake_ticker = FakeTicker(actions_by_symbol)

        with mock.patch.object(ddl.yf, "Ticker", fake_ticker) as ticker_mock:
            ddl.load_dividend_events({"2330": "台積電"}, "2026-01-01", "2026-12-31", refresh=True)

        # 第二次呼叫（refresh=False），應該直接讀快取，完全不呼叫 yf.Ticker
        with mock.patch.object(ddl.yf, "Ticker", side_effect=AssertionError("不該打網路")):
            result = ddl.load_dividend_events(
                {"2330": "台積電"}, "2026-01-01", "2026-12-31", refresh=False,
            )
        self.assertEqual(result["2330"], {"20260315"})

    def test_refresh_true_forces_redownload(self):
        actions_by_symbol_v1 = {"2330.TW": make_fake_actions([("2026-03-15", 2.5, 0)])}
        actions_by_symbol_v2 = {"2330.TW": make_fake_actions([("2026-08-01", 1.0, 0)])}

        with mock.patch.object(ddl.yf, "Ticker", FakeTicker(actions_by_symbol_v1)):
            ddl.load_dividend_events({"2330": "台積電"}, "2026-01-01", "2026-12-31", refresh=True)

        with mock.patch.object(ddl.yf, "Ticker", FakeTicker(actions_by_symbol_v2)):
            result = ddl.load_dividend_events(
                {"2330": "台積電"}, "2026-01-01", "2026-12-31", refresh=True,
            )
        self.assertEqual(result["2330"], {"20260801"})


class TestSymbolFor(unittest.TestCase):
    def test_old_string_format_uses_otc_stocks_set(self):
        # 3324 在 data_loader.OTC_STOCKS 裡，應該判定為上櫃 (.TWO)
        self.assertEqual(ddl._symbol_for("3324", "雙鴻"), "3324.TWO")
        self.assertEqual(ddl._symbol_for("2330", "台積電"), "2330.TW")

    def test_new_dict_format_reads_otc_field_directly(self):
        self.assertEqual(ddl._symbol_for("9999", {"otc": True}), "9999.TWO")
        self.assertEqual(ddl._symbol_for("9999", {"otc": False}), "9999.TW")


if __name__ == "__main__":
    unittest.main(verbosity=2)
