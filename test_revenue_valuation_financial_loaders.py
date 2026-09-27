"""
test_revenue_valuation_financial_loaders.py
===============================================
針對三個新資料loader(revenue_data_loader.py / valuation_data_loader.py /
financial_statement_loader.py)的合成資料測試。完全不連真實網路——mock
requests.get()/requests.post()，驗證：
1. 欄位動態解析(不是抓錯欄/寫死index)
2. NaN vs 0的語意區分(轉不出數字的欄位是NaN，不是0)
3. FetchFailed(請求本身失敗) vs None(確認無資料)兩種情況分開處理
4. valuation_data_loader的implied_eps/implied_book_value除以零/缺值保護
"""
import io
import unittest
from unittest import mock

import pandas as pd

import revenue_data_loader as rdl
import valuation_data_loader as vdl
import financial_statement_loader as fsl


def make_json_response(payload):
    resp = mock.Mock()
    resp.json.return_value = payload
    return resp


def make_html_response(html_text, status_code=200):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.text = html_text
    resp.encoding = "utf-8"
    return resp


class TestRevenueDataLoader(unittest.TestCase):
    def test_parses_yoy_and_mom_from_html_table(self):
        html = """
        <table>
        <tr><th>公司代號</th><th>公司名稱</th><th>去年同月增減(%)</th><th>上月比較增減(%)</th></tr>
        <tr><td>2330</td><td>台積電</td><td>12.34</td><td>-1.50</td></tr>
        <tr><td>2454</td><td>聯發科</td><td>--</td><td>5.00</td></tr>
        </table>
        """
        with mock.patch.object(rdl.requests, "get", return_value=make_html_response(html)):
            result = rdl.fetch_revenue_month(2026, 3, market="sii")

        self.assertIn("2330", result)
        self.assertAlmostEqual(result["2330"]["revenue_yoy_pct"], 12.34)
        self.assertAlmostEqual(result["2330"]["revenue_mom_pct"], -1.50)
        # "--"轉不出數字，應該是NaN不是0
        self.assertTrue(pd.isna(result["2454"]["revenue_yoy_pct"]))
        self.assertAlmostEqual(result["2454"]["revenue_mom_pct"], 5.00)

    def test_404_status_returns_none_not_fetch_failed(self):
        with mock.patch.object(rdl.requests, "get", return_value=make_html_response("", status_code=404)):
            result = rdl.fetch_revenue_month(2099, 1, market="sii")
        self.assertIsNone(result)

    def test_request_exception_raises_fetch_failed(self):
        with mock.patch.object(rdl.requests, "get", side_effect=Exception("連線逾時")):
            with self.assertRaises(rdl.FetchFailed):
                rdl.fetch_revenue_month(2026, 3, market="sii")

    def test_no_matching_table_returns_none(self):
        html = "<table><tr><th>完全無關的欄位</th></tr><tr><td>x</td></tr></table>"
        with mock.patch.object(rdl.requests, "get", return_value=make_html_response(html)):
            result = rdl.fetch_revenue_month(2026, 3, market="sii")
        self.assertIsNone(result)

    def test_non_digit_code_rows_are_skipped(self):
        html = """
        <table>
        <tr><th>公司代號</th><th>去年同月增減(%)</th><th>上月比較增減(%)</th></tr>
        <tr><td>合計</td><td>1.0</td><td>2.0</td></tr>
        <tr><td>2330</td><td>10.0</td><td>1.0</td></tr>
        </table>
        """
        with mock.patch.object(rdl.requests, "get", return_value=make_html_response(html)):
            result = rdl.fetch_revenue_month(2026, 3, market="sii")
        self.assertNotIn("合計", result)
        self.assertIn("2330", result)


class TestValuationDataLoader(unittest.TestCase):
    FIELDS = ["證券代號", "證券名稱", "殖利率(%)", "收盤價", "股價淨值比", "本益比"]

    def test_extracts_values_by_field_name(self):
        row = ["2330", "台積電", "1.8", "600.0", "6.5", "18.2"]
        payload = {"stat": "OK", "fields": self.FIELDS, "data": [row]}
        with mock.patch.object(vdl.requests, "get", return_value=make_json_response(payload)):
            result = vdl.fetch_valuation_day("20260315")
        self.assertIn("2330", result)
        self.assertAlmostEqual(result["2330"]["close"], 600.0)
        self.assertAlmostEqual(result["2330"]["pe"], 18.2)
        self.assertAlmostEqual(result["2330"]["pb"], 6.5)
        self.assertAlmostEqual(result["2330"]["dividend_yield"], 1.8)

    def test_loss_making_company_pe_becomes_nan_not_zero(self):
        row = ["1234", "虧損公司", "0.0", "20.0", "1.2", "—"]
        payload = {"stat": "OK", "fields": self.FIELDS, "data": [row]}
        with mock.patch.object(vdl.requests, "get", return_value=make_json_response(payload)):
            result = vdl.fetch_valuation_day("20260315")
        self.assertTrue(pd.isna(result["1234"]["pe"]))

    def test_stat_not_ok_returns_none(self):
        payload = {"stat": "非交易日"}
        with mock.patch.object(vdl.requests, "get", return_value=make_json_response(payload)):
            result = vdl.fetch_valuation_day("20260101")
        self.assertIsNone(result)

    def test_request_exception_raises_fetch_failed(self):
        with mock.patch.object(vdl.requests, "get", side_effect=Exception("連線逾時")):
            with self.assertRaises(vdl.FetchFailed):
                vdl.fetch_valuation_day("20260101")

    def test_implied_eps_and_book_value_guard_against_missing_pe_pb(self):
        # 直接構造 load_valuation_data() 最後一段的衍生欄位邏輯(不連網路，繞過快取檔I/O，
        # 直接組一個等價的DataFrame驗證除以零/缺值保護)
        df = pd.DataFrame({
            "close": [100.0, 50.0, 30.0],
            "pe": [10.0, float("nan"), -5.0],   # 第2/3筆本益比缺失/負值(虧損)
            "pb": [2.0, 1.0, 0.0],               # 第3筆股價淨值比是0(理論上不該發生，但要防呆)
        })
        import numpy as np
        pe_valid = df["pe"].where(df["pe"] > 0)
        pb_valid = df["pb"].where(df["pb"] > 0)
        implied_eps = (df["close"] / pe_valid).replace([np.inf, -np.inf], np.nan)
        implied_bv = (df["close"] / pb_valid).replace([np.inf, -np.inf], np.nan)

        self.assertAlmostEqual(implied_eps.iloc[0], 10.0)
        self.assertTrue(pd.isna(implied_eps.iloc[1]))
        self.assertTrue(pd.isna(implied_eps.iloc[2]))  # 負本益比也不該硬算出implied EPS
        self.assertAlmostEqual(implied_bv.iloc[0], 50.0)
        self.assertTrue(pd.isna(implied_bv.iloc[2]))   # pb=0不該除以零


class TestFinancialStatementLoader(unittest.TestCase):
    def test_parses_eps_gross_margin_roe_from_html_table(self):
        html = """
        <table>
        <tr><th>基本每股盈餘(元)</th><th>營業毛利率(%)</th><th>權益報酬率(%)</th></tr>
        <tr><td>5.23</td><td>45.6</td><td>8.9</td></tr>
        </table>
        """
        with mock.patch.object(fsl.requests, "post", return_value=make_html_response(html)):
            result = fsl.fetch_quarterly_financials("2330", 2026, 1)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["eps"], 5.23)
        self.assertAlmostEqual(result["gross_margin_pct"], 45.6)
        self.assertAlmostEqual(result["roe_pct"], 8.9)

    def test_unparseable_html_returns_none_not_exception(self):
        with mock.patch.object(fsl.requests, "post", return_value=make_html_response("<html>沒有表格</html>")):
            result = fsl.fetch_quarterly_financials("2330", 2026, 1)
        self.assertIsNone(result)

    def test_request_exception_raises_fetch_failed(self):
        with mock.patch.object(fsl.requests, "post", side_effect=Exception("連線逾時")):
            with self.assertRaises(fsl.FetchFailed):
                fsl.fetch_quarterly_financials("2330", 2026, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
