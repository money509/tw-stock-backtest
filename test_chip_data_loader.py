"""
test_chip_data_loader.py
============================
針對 chip_data_loader.py 的合成資料測試。不連真實網路——mock requests.get()。

背景（這次要修的bug）：實際在GitHub Actions對2015-09-15~2019-09-14這段舊資料
跑cross_period_validation時，fetch_t86_day() 用寫死的欄位index (COL_TOTAL_NET=18)
去讀某一天的row，直接丟出 IndexError: list index out of range，整個流程中斷。
原因：證交所T86報表的欄位隨時間變動過(自營商買賣超後來被拆成「自行買賣」/「避險」
兩組明細欄位)，2015~2019年的舊格式欄位數量/位置跟現在(2023年後)不一樣，寫死index
沒辦法同時兼容新舊格式。

修正後的 fetch_t86_day() 改成動態從當天payload附帶的"fields"欄位名稱陣列去
「查名字找位置」，這裡驗證：
  1. 新格式(19欄，含自行買賣/避險明細)能正確抓到跟以前寫死index一樣的值
  2. 模擬的舊格式(欄位較少、自營商沒有拆自行買賣/避險明細)也能正確抓到值，
     不會 IndexError
  3. 真的找不到某個欄位時，安全地把那個值當0處理，不會讓整支程式崩潔
  4. stat!=OK的日子正常回傳None(不是失敗，不用重試)
  5. request本身失敗(逾時/連線錯誤)會丟出FetchFailed，由重試邏輯處理
"""

import unittest
from unittest import mock

import chip_data_loader as cdl


# 目前(2023年後)的T86真實欄位順序，19欄
NEW_FORMAT_FIELDS = [
    "證券代號", "證券名稱",
    "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)",
    "外陸資買賣超股數(不含外資自營商)",
    "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
    "投信買進股數", "投信賣出股數", "投信買賣超股數",
    "自營商買賣超股數",
    "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)",
    "自營商買進股數(避險)", "自營商賣出股數(避險)", "自營商買賣超股數(避險)",
    "三大法人買賣超股數合計",
]

# 模擬的舊格式(假設2015~2019年沒有自行買賣/避險明細，只有16欄)——
# 這是造成原始bug的那種「欄位比預期少」的情況
OLD_FORMAT_FIELDS = [
    "證券代號", "證券名稱",
    "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)",
    "外陸資買賣超股數(不含外資自營商)",
    "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
    "投信買進股數", "投信賣出股數", "投信買賣超股數",
    "自營商買進股數", "自營商賣出股數", "自營商買賣超股數",
    "三大法人買賣超股數合計",
]


def make_response(payload):
    resp = mock.Mock()
    resp.json.return_value = payload
    return resp


class TestFetchT86DayNewFormat(unittest.TestCase):
    def test_new_format_extracts_correct_values(self):
        row = ["0"] * len(NEW_FORMAT_FIELDS)
        row[0] = "2330"
        row[1] = "台積電"
        row[4] = "500"    # 外陸資買賣超股數(不含外資自營商)
        row[10] = "20"    # 投信買賣超股數
        row[11] = "300"   # 自營商買賣超股數
        row[18] = "9999"  # 三大法人買賣超股數合計
        payload = {"stat": "OK", "fields": NEW_FORMAT_FIELDS, "data": [row]}
        with mock.patch.object(cdl.requests, "get", return_value=make_response(payload)):
            result = cdl.fetch_t86_day("20260315")

        self.assertIn("2330", result)
        self.assertEqual(result["2330"]["foreign_net"], 500)
        self.assertEqual(result["2330"]["trust_net"], 20)
        self.assertEqual(result["2330"]["dealer_net"], 300)
        self.assertEqual(result["2330"]["total_net"], 9999)


class TestFetchT86DayOldFormat(unittest.TestCase):
    def test_old_format_with_fewer_columns_does_not_crash(self):
        # 14欄 (index 0~13)，用舊的寫死index(COL_TOTAL_NET=18)去讀一定會IndexError，
        # 這支測試就是要驗證新的動態欄位解析法不會有這個問題
        row = ["0"] * len(OLD_FORMAT_FIELDS)
        row[0] = "2330"
        row[1] = "台積電"
        row[4] = "777"    # 外陸資買賣超股數(不含外資自營商)
        row[10] = "30"    # 投信買賣超股數
        row[13] = "888"   # 自營商買賣超股數(舊格式沒有拆自行買賣/避險，直接是合計)
        row[14] = "5555"  # 三大法人買賣超股數合計(舊格式的最後一欄，位置跟新格式的18不同)
        payload = {"stat": "OK", "fields": OLD_FORMAT_FIELDS, "data": [row]}
        with mock.patch.object(cdl.requests, "get", return_value=make_response(payload)):
            result = cdl.fetch_t86_day("20160315")  # 這次bug實際發生的舊資料區間

        self.assertIn("2330", result)
        self.assertEqual(result["2330"]["foreign_net"], 777)
        self.assertEqual(result["2330"]["trust_net"], 30)
        self.assertEqual(result["2330"]["dealer_net"], 888)
        self.assertEqual(result["2330"]["total_net"], 5555)


class TestResolveColumnIndices(unittest.TestCase):
    def test_new_format_indices(self):
        col = cdl._resolve_column_indices(NEW_FORMAT_FIELDS)
        self.assertEqual(col["foreign_net"], 4)
        self.assertEqual(col["trust_net"], 10)
        self.assertEqual(col["dealer_net"], 11)
        self.assertEqual(col["total_net"], 18)

    def test_missing_total_field_falls_back_to_last_column(self):
        fields = ["證券代號", "證券名稱", "投信買賣超股數", "隨便一個沒人認得的欄位"]
        col = cdl._resolve_column_indices(fields)
        self.assertEqual(col["total_net"], 3)  # 找不到合計欄名稱時，退回最後一欄

    def test_completely_unknown_fields_returns_none_for_missing(self):
        fields = ["證券代號", "證券名稱", "某個全新未知欄位"]
        col = cdl._resolve_column_indices(fields)
        self.assertIsNone(col["foreign_net"])
        self.assertIsNone(col["trust_net"])
        self.assertIsNone(col["dealer_net"])

    def test_unresolvable_field_defaults_to_zero_not_crash(self):
        fields = ["證券代號", "證券名稱", "完全陌生欄位A", "完全陌生欄位B"]
        row = ["2330", "台積電", "111", "222"]
        payload = {"stat": "OK", "fields": fields, "data": [row]}
        with mock.patch.object(cdl.requests, "get", return_value=make_response(payload)):
            result = cdl.fetch_t86_day("20160101")
        # 查不到的欄位安全地當0，不會拋例外，也不會整天資料整個消失
        self.assertIn("2330", result)
        self.assertEqual(result["2330"]["foreign_net"], 0)
        self.assertEqual(result["2330"]["trust_net"], 0)
        self.assertEqual(result["2330"]["dealer_net"], 0)


class TestFetchT86DayStatAndFailure(unittest.TestCase):
    def test_stat_not_ok_returns_none(self):
        payload = {"stat": "非交易日"}
        with mock.patch.object(cdl.requests, "get", return_value=make_response(payload)):
            result = cdl.fetch_t86_day("20260101")
        self.assertIsNone(result)

    def test_request_exception_raises_fetch_failed(self):
        with mock.patch.object(cdl.requests, "get", side_effect=Exception("連線逾時")):
            with self.assertRaises(cdl.FetchFailed):
                cdl.fetch_t86_day("20260101")

    def test_non_digit_code_rows_are_skipped(self):
        row_warrant = ["03001P", "某權證"] + ["0"] * (len(NEW_FORMAT_FIELDS) - 2)
        row_stock = ["2330", "台積電"] + ["100"] * (len(NEW_FORMAT_FIELDS) - 2)
        payload = {"stat": "OK", "fields": NEW_FORMAT_FIELDS, "data": [row_warrant, row_stock]}
        with mock.patch.object(cdl.requests, "get", return_value=make_response(payload)):
            result = cdl.fetch_t86_day("20260315")
        self.assertNotIn("03001P", result)
        self.assertIn("2330", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
