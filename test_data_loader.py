"""
test_data_loader.py
=======================
針對 data_loader.py 的合成資料測試。不連網路——yf.download 用 mock 取代，
CACHE_DIR 導向暫存目錄，測試結束自動清除。

驗證重點：
  1. _symbol_for() 正確判斷 .TW / .TWO 後綴，同時支援舊格式(entry=股票名稱字串，
     靠OTC_STOCKS這份59檔清單判斷)跟新格式(entry=dict，含"otc"欄位，
     來自taifex_universe.py的249檔全市場清單，直接讀取判斷)——這是
     --universe full 這個新選項能不能正確運作的關鍵。
  2. load_price_data() 有快取就不重新下載；沒快取才會呼叫yf.download，
     批次下載結果正確拆解、存快取、回傳。
  3. build_master_calendar() 正確用參考股票的日期序列，參考股票缺資料時
     正確退回聯集。
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import data_loader as dl


class TestSymbolFor(unittest.TestCase):
    def test_old_format_string_entry_uses_otc_stocks_set(self):
        # "3324"(雙鴻) 在 OTC_STOCKS 裡，entry 是舊格式的股票名稱字串
        self.assertEqual(dl._symbol_for("3324", "雙鴻"), "3324.TWO")
        self.assertEqual(dl._symbol_for("2330", "台積電"), "2330.TW")

    def test_old_format_no_entry_defaults_to_otc_stocks_lookup(self):
        self.assertEqual(dl._symbol_for("3324"), "3324.TWO")
        self.assertEqual(dl._symbol_for("2330"), "2330.TW")

    def test_new_format_dict_entry_with_otc_true(self):
        self.assertEqual(dl._symbol_for("1234", {"otc": True, "has_mini": False}), "1234.TWO")

    def test_new_format_dict_entry_with_otc_false(self):
        self.assertEqual(dl._symbol_for("1234", {"otc": False, "has_mini": True}), "1234.TW")

    def test_new_format_dict_entry_ignores_otc_stocks_set(self):
        # code不在OTC_STOCKS裡，但dict entry說是otc=True，應該以dict為準
        self.assertNotIn("9999", dl.OTC_STOCKS)
        self.assertEqual(dl._symbol_for("9999", {"otc": True, "has_mini": False}), "9999.TWO")

    def test_taifex_universe_entries_all_resolve_without_error(self):
        import taifex_universe as tu
        for code, entry in list(tu.STOCK_FUTURES_UNIVERSE.items())[:20]:
            sym = dl._symbol_for(code, entry)
            self.assertTrue(sym.endswith(".TW") or sym.endswith(".TWO"))
            self.assertTrue(sym.startswith(code))


def _make_yf_multi_index_df(symbols, n_days=5, base_price=100.0):
    """模擬 yf.download(group_by='ticker') 回傳的DataFrame。
    load_price_data() 對「只有1檔」跟「多檔」的處理方式不同(len(batch_syms)==1時
    直接假設raw本身就是那一檔的flat欄位，不是MultiIndex)，這裡照樣模擬那個差異，
    確保fixture跟production code對同一份yf.download()回傳格式的假設一致。"""
    dates = pd.bdate_range(start="2026-01-01", periods=n_days)

    def _fields(offset):
        d = {}
        for field in ["Open", "High", "Low", "Close"]:
            d[field] = base_price + offset * 10 + np.arange(n_days) * 0.5
        d["Volume"] = np.full(n_days, 1_000_000 + offset * 1000)
        d["Adj Close"] = d["Close"]
        return d

    if len(symbols) == 1:
        return pd.DataFrame(_fields(0), index=dates)

    cols = pd.MultiIndex.from_product(
        [symbols, ["Open", "High", "Low", "Close", "Volume", "Adj Close"]])
    data = {}
    for i, sym in enumerate(symbols):
        fields = _fields(i)
        for field, values in fields.items():
            data[(sym, field)] = values
    return pd.DataFrame(data, index=dates, columns=cols)


class TestLoadPriceData(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._orig_cache_dir = dl.CACHE_DIR
        dl.CACHE_DIR = self.tmp_dir

    def tearDown(self):
        dl.CACHE_DIR = self._orig_cache_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_downloads_and_caches_new_data(self):
        whitelist = {"2330": "台積電", "2317": "鴻海"}
        fake_df = _make_yf_multi_index_df(["2330.TW", "2317.TW"])

        with patch.object(dl.yf, "download", return_value=fake_df) as mock_download:
            price_data = dl.load_price_data(whitelist, "2026-01-01", "2026-01-10")

        self.assertEqual(mock_download.call_count, 1)
        self.assertEqual(set(price_data.keys()), {"2330", "2317"})
        for code, df in price_data.items():
            self.assertIn("Close", df.columns)
            self.assertIn("Volume", df.columns)
            cache_path = os.path.join(self.tmp_dir, f"{code}_2026-01-01_2026-01-10.csv")
            self.assertTrue(os.path.exists(cache_path))

    def test_uses_cache_and_skips_download(self):
        whitelist = {"2330": "台積電"}
        fake_df = _make_yf_multi_index_df(["2330.TW"])

        with patch.object(dl.yf, "download", return_value=fake_df) as mock_download:
            dl.load_price_data(whitelist, "2026-01-01", "2026-01-10")
            # 第二次呼叫，快取應該已經存在，不該再呼叫yf.download
            dl.load_price_data(whitelist, "2026-01-01", "2026-01-10")

        self.assertEqual(mock_download.call_count, 1)

    def test_refresh_true_forces_redownload(self):
        whitelist = {"2330": "台積電"}
        fake_df = _make_yf_multi_index_df(["2330.TW"])

        with patch.object(dl.yf, "download", return_value=fake_df) as mock_download:
            dl.load_price_data(whitelist, "2026-01-01", "2026-01-10")
            dl.load_price_data(whitelist, "2026-01-01", "2026-01-10", refresh=True)

        self.assertEqual(mock_download.call_count, 2)

    def test_new_format_dict_whitelist_resolves_correct_symbols(self):
        # 驗證--universe full用的dict格式whitelist，能正確組出.TW/.TWO symbol
        # 傳給yf.download。
        whitelist = {"1234": {"otc": True, "has_mini": False}}
        fake_df = _make_yf_multi_index_df(["1234.TWO"])

        with patch.object(dl.yf, "download", return_value=fake_df) as mock_download:
            price_data = dl.load_price_data(whitelist, "2026-01-01", "2026-01-10")

        called_symbols = mock_download.call_args.kwargs["tickers"]
        self.assertEqual(called_symbols, ["1234.TWO"])
        self.assertIn("1234", price_data)

    def test_empty_or_none_download_result_skips_batch_without_crash(self):
        whitelist = {"2330": "台積電"}
        with patch.object(dl.yf, "download", return_value=pd.DataFrame()):
            price_data = dl.load_price_data(whitelist, "2026-01-01", "2026-01-10")
        self.assertEqual(price_data, {})

    def test_download_exception_retries_then_gives_up_gracefully(self):
        whitelist = {"2330": "台積電"}
        with patch.object(dl.yf, "download", side_effect=RuntimeError("boom")), \
             patch.object(dl.time, "sleep"):  # 不要真的睡，加速測試
            price_data = dl.load_price_data(whitelist, "2026-01-01", "2026-01-10")
        self.assertEqual(price_data, {})


class TestBuildMasterCalendar(unittest.TestCase):
    def test_uses_reference_code_when_present(self):
        idx_2330 = pd.date_range("2026-01-01", periods=5)
        idx_other = pd.date_range("2026-01-01", periods=3)
        price_data = {
            "2330": pd.DataFrame({"Close": range(5)}, index=idx_2330),
            "2317": pd.DataFrame({"Close": range(3)}, index=idx_other),
        }
        calendar = dl.build_master_calendar(price_data, reference_code="2330")
        self.assertTrue(calendar.equals(idx_2330))

    def test_falls_back_to_union_when_reference_missing(self):
        idx_a = pd.date_range("2026-01-01", periods=3)
        idx_b = pd.date_range("2026-01-03", periods=3)
        price_data = {
            "2317": pd.DataFrame({"Close": range(3)}, index=idx_a),
            "2454": pd.DataFrame({"Close": range(3)}, index=idx_b),
        }
        calendar = dl.build_master_calendar(price_data, reference_code="2330")
        expected = pd.DatetimeIndex(sorted(set(idx_a) | set(idx_b)))
        self.assertTrue(calendar.equals(expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
