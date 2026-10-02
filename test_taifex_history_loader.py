"""
test_taifex_history_loader.py / 附帶test_validate_tx_history_accuracy.py的比對數學測試
=============================================================================
taifex_history_loader.py的單元測試：全部用合成資料(小型本地檔案)，mock
gdown.download，完全不打真實網路(不連Google Drive)。涵蓋：
1. check_file_signature()：MZ/ELF/shebang/HTML開頭都要被擋下來，不會被拿去解析。
2. parse_history_file()：合成的「逐筆成交(1個價格欄)」跟「OHLC(4個價格欄)」兩種
   欄位配置都要能正確解析；格式辨識不出來(欄位猜不到/價格欄數量既不是1也不是4)
   要回傳明確reason，不能崩潰。
3. load_tx_history_bars()：其中一個period失敗，不能連累其他period。
4. validate_tx_history_accuracy.py的比對數學(平均/中位數/最大誤差、容忍範圍內
   天數比例)用已知答案的合成日線資料驗證算出來的數字是對的。
"""
import datetime
import io
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import pytest

import taifex_history_loader as thl
import validate_tx_history_accuracy as vtha


def _make_tick_csv_bytes(n=30, start=None):
    start = start or datetime.datetime(2021, 1, 5, 8, 45, 0)
    rows = []
    for i in range(n):
        t = start + datetime.timedelta(minutes=i)
        rows.append([t.strftime("%Y%m%d"), t.strftime("%H%M%S"), 18000 + i, 10])
    df = pd.DataFrame(rows, columns=["date", "time", "price", "vol"])
    return df.to_csv(index=False).encode("utf-8")


def _make_ohlc_csv_bytes(n=30, start=None):
    start = start or datetime.datetime(2021, 1, 5, 8, 45, 0)
    rows = []
    for i in range(n):
        t = start + datetime.timedelta(minutes=i)
        o = 18000 + i
        rows.append([t.strftime("%Y%m%d"), t.strftime("%H%M%S"), o, o + 5, o - 5, o + 2, 123])
    df = pd.DataFrame(rows, columns=["date", "time", "open", "high", "low", "close", "vol"])
    return df.to_csv(index=False).encode("utf-8")


def _write_tmp(content_bytes):
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "wb") as f:
        f.write(content_bytes)
    return path


class TestCheckFileSignature:
    def test_pe_header_rejected(self):
        path = _write_tmp(b"MZ\x90\x00" + b"\x00" * 100)
        is_safe, reason = thl.check_file_signature(path)
        assert is_safe is False
        assert reason == "unsafe_file_signature"

    def test_elf_header_rejected(self):
        path = _write_tmp(b"\x7fELF" + b"\x00" * 100)
        is_safe, reason = thl.check_file_signature(path)
        assert is_safe is False
        assert reason == "unsafe_file_signature"

    def test_shebang_rejected(self):
        path = _write_tmp(b"#!/bin/sh\necho hi\n" * 5)
        is_safe, reason = thl.check_file_signature(path)
        assert is_safe is False
        assert reason == "unsafe_file_signature"

    def test_html_rejected(self):
        path = _write_tmp(b"<html><body>Sign in to Google Drive</body></html>" * 3)
        is_safe, reason = thl.check_file_signature(path)
        assert is_safe is False
        assert reason == "unsafe_file_signature"

    def test_plain_csv_accepted(self):
        path = _write_tmp(_make_tick_csv_bytes(5))
        is_safe, reason = thl.check_file_signature(path)
        assert is_safe is True
        assert reason == "ok"

    def test_empty_file_rejected(self):
        path = _write_tmp(b"")
        is_safe, reason = thl.check_file_signature(path)
        assert is_safe is False
        assert reason == "empty_file"


class TestParseHistoryFile:
    def test_unsafe_signature_never_parsed(self):
        path = _write_tmp(b"MZ\x90\x00" + b"\x00" * 100)
        df, reason = thl.parse_history_file(path)
        assert df is None
        assert reason == "unsafe_file_signature"

    def test_tick_level_single_price_column_parses(self):
        path = _write_tmp(_make_tick_csv_bytes(40))
        df, reason = thl.parse_history_file(path)
        assert reason == "ok"
        assert df.attrs["source_level"] == "tick"
        assert list(df.columns) == ["ProductCode", "ContractMonth", "Timestamp", "Price", "Volume"]
        assert len(df) == 40
        assert df["Price"].iloc[0] == 18000.0
        assert df["Timestamp"].iloc[0] == pd.Timestamp("2021-01-05 08:45:00")

    def test_bar_level_four_price_columns_parses_as_ohlc(self):
        path = _write_tmp(_make_ohlc_csv_bytes(40))
        df, reason = thl.parse_history_file(path)
        assert reason == "ok"
        assert df.attrs["source_level"] == "bar"
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 40
        assert (df["High"] >= df["Open"]).all()
        assert (df["High"] >= df["Close"]).all()
        assert (df["Low"] <= df["Open"]).all()
        assert (df["Low"] <= df["Close"]).all()

    def test_malformed_text_reports_no_date_col(self):
        path = _write_tmp(b"not a csv at all just words words words\nmore words here too\n")
        df, reason = thl.parse_history_file(path)
        assert df is None
        assert reason == "no_date_col"

    def test_empty_file_reports_empty_file(self):
        path = _write_tmp(b"")
        df, reason = thl.parse_history_file(path)
        assert df is None
        assert reason == "empty_file"

    def test_ambiguous_price_column_count_reports_unrecognized_format(self):
        # 日期/時間欄位正常，但價格欄位數量是2個(不是1個tick、也不是4個OHLC)，
        # 應該誠實回報辨識不出來，而不是硬猜。
        start = datetime.datetime(2021, 1, 5, 8, 45, 0)
        rows = []
        for i in range(40):
            t = start + datetime.timedelta(minutes=i)
            rows.append([t.strftime("%Y%m%d"), t.strftime("%H%M%S"), 18000 + i, 18005 + i])
        df_in = pd.DataFrame(rows, columns=["date", "time", "price_a", "price_b"])
        path = _write_tmp(df_in.to_csv(index=False).encode("utf-8"))
        df, reason = thl.parse_history_file(path)
        assert df is None
        assert reason == "unrecognized_format"

    def test_zip_wrapped_csv_parses(self):
        import zipfile
        csv_bytes = _make_tick_csv_bytes(40)
        fd, zip_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", csv_bytes)
        df, reason = thl.parse_history_file(zip_path)
        assert reason == "ok"
        assert df.attrs["source_level"] == "tick"
        assert len(df) == 40

    def test_missing_volume_column_defaults_to_zero_not_crash(self):
        start = datetime.datetime(2021, 1, 5, 8, 45, 0)
        rows = []
        for i in range(40):
            t = start + datetime.timedelta(minutes=i)
            rows.append([t.strftime("%Y%m%d"), t.strftime("%H%M%S"), 18000 + i])
        df_in = pd.DataFrame(rows, columns=["date", "time", "price"])
        path = _write_tmp(df_in.to_csv(index=False).encode("utf-8"))
        df, reason = thl.parse_history_file(path)
        assert reason == "ok"
        assert (df["Volume"] == 0).all()


class TestDownloadPeriod:
    def test_unknown_period_key(self):
        local_path, reason = thl.download_period("not_a_real_period")
        assert local_path is None
        assert reason == "unknown_period"

    def test_uses_cache_when_present_and_safe(self, tmp_path):
        cache_dir = str(tmp_path)
        cache_path = os.path.join(cache_dir, "2011_2020.raw")
        with open(cache_path, "wb") as f:
            f.write(_make_tick_csv_bytes(5))
        with mock.patch("taifex_history_loader._gdown_download") as mock_dl:
            local_path, reason = thl.download_period("2011_2020", cache_dir=cache_dir, refresh=False)
        assert reason == "cached"
        assert local_path == cache_path
        mock_dl.assert_not_called()

    def test_download_error_returns_reason_and_no_cache_file(self, tmp_path):
        cache_dir = str(tmp_path)
        with mock.patch("taifex_history_loader._gdown_download", return_value=None):
            with mock.patch("taifex_history_loader.RETRY_BACKOFF_SECONDS", 0):
                local_path, reason = thl.download_period("2011_2020", cache_dir=cache_dir, refresh=True)
        assert local_path is None
        assert reason == "download_error"
        assert not os.path.exists(os.path.join(cache_dir, "2011_2020.raw"))

    def test_successful_download_is_cached(self, tmp_path):
        cache_dir = str(tmp_path)

        def fake_gdown(file_id, output_path):
            with open(output_path, "wb") as f:
                f.write(_make_tick_csv_bytes(5))
            return output_path

        with mock.patch("taifex_history_loader._gdown_download", side_effect=fake_gdown):
            local_path, reason = thl.download_period("2011_2020", cache_dir=cache_dir, refresh=True)
        assert reason == "ok"
        assert os.path.exists(local_path)

        # 第二次呼叫(refresh=False)應該直接用快取，不會再呼叫gdown
        with mock.patch("taifex_history_loader._gdown_download") as mock_dl2:
            local_path2, reason2 = thl.download_period("2011_2020", cache_dir=cache_dir, refresh=False)
        assert reason2 == "cached"
        mock_dl2.assert_not_called()

    def test_unsafe_downloaded_content_not_cached(self, tmp_path):
        cache_dir = str(tmp_path)

        def fake_gdown_html(file_id, output_path):
            with open(output_path, "wb") as f:
                f.write(b"<html><body>Sign in required</body></html>" * 3)
            return output_path

        with mock.patch("taifex_history_loader._gdown_download", side_effect=fake_gdown_html):
            local_path, reason = thl.download_period("2011_2020", cache_dir=cache_dir, refresh=True)
        assert local_path is None
        assert reason == "unsafe_file_signature"
        assert not os.path.exists(os.path.join(cache_dir, "2011_2020.raw"))


class TestLoadTxHistoryBars:
    def test_one_bad_period_does_not_stop_others(self, tmp_path):
        cache_dir = str(tmp_path)
        good_path = os.path.join(cache_dir, "2011_2020.raw")
        os.makedirs(cache_dir, exist_ok=True)
        with open(good_path, "wb") as f:
            f.write(_make_ohlc_csv_bytes(60))

        def fake_download(period_key, cache_dir=None, refresh=False):
            if period_key == "2011_2020":
                return good_path, "ok"
            return None, "download_error"

        with mock.patch("taifex_history_loader.download_period", side_effect=fake_download):
            bars, diagnostics = thl.load_tx_history_bars(
                ["2011_2020", "2021_2023"], bar_minutes=5, cache_dir=cache_dir)

        assert bars is not None
        assert not bars.empty
        assert diagnostics["n_periods_ok"] == 1
        assert diagnostics["n_periods_failed"] == 1
        assert diagnostics["periods"]["2011_2020"]["reason"] == "ok"
        assert diagnostics["periods"]["2021_2023"]["reason"] == "download_error"

    def test_all_periods_failing_returns_none_with_diagnostics(self, tmp_path):
        with mock.patch("taifex_history_loader.download_period", return_value=(None, "download_error")):
            bars, diagnostics = thl.load_tx_history_bars(["2011_2020"], cache_dir=str(tmp_path))
        assert bars is None
        assert diagnostics["n_periods_ok"] == 0
        assert diagnostics["n_periods_failed"] == 1

    def test_concatenates_and_sorts_multiple_periods(self, tmp_path):
        cache_dir = str(tmp_path)
        os.makedirs(cache_dir, exist_ok=True)
        path_a = os.path.join(cache_dir, "a.raw")
        path_b = os.path.join(cache_dir, "b.raw")
        with open(path_a, "wb") as f:
            f.write(_make_ohlc_csv_bytes(10, start=datetime.datetime(2020, 1, 1, 8, 45)))
        with open(path_b, "wb") as f:
            f.write(_make_ohlc_csv_bytes(10, start=datetime.datetime(2021, 1, 1, 8, 45)))

        def fake_download(period_key, cache_dir=None, refresh=False):
            return (path_a if period_key == "p1" else path_b), "ok"

        with mock.patch("taifex_history_loader.download_period", side_effect=fake_download):
            bars, diagnostics = thl.load_tx_history_bars(["p1", "p2"], bar_minutes=1, cache_dir=cache_dir)

        assert bars is not None
        assert bars.index.is_monotonic_increasing
        assert diagnostics["n_periods_ok"] == 2


class TestValidateTxHistoryAccuracyComparisonMath:
    def test_compare_daily_ohlc_known_differences(self):
        dates = [datetime.date(2021, 1, 4), datetime.date(2021, 1, 5), datetime.date(2021, 1, 6)]
        community = pd.DataFrame({
            "Open": [18000, 18100, 18200],
            "High": [18050, 18150, 18250],
            "Low": [17950, 18050, 18150],
            "Close": [18010, 18110, 18210],
        }, index=pd.Index(dates, name="Date"))
        # 官方：收盤分別差 +2 / +20(大誤差) / +2
        official = pd.DataFrame({
            "Open": [18000, 18100, 18200],
            "High": [18050, 18150, 18250],
            "Low": [17950, 18050, 18150],
            "Close": [18012, 18130, 18212],
        }, index=pd.Index(dates, name="Date"))

        comparison, stats = vtha.compare_daily_ohlc(community, official, tolerance_pct=0.05)

        assert stats["n_dates_compared"] == 3
        assert stats["mean_abs_error_close"] == pytest.approx((2 + 20 + 2) / 3)
        assert stats["median_abs_error_close"] == pytest.approx(2.0)
        assert stats["max_abs_error_close"] == pytest.approx(20.0)
        # 0.05%容忍範圍(約9點@18000)下，只有誤差2點的兩天落在範圍內，誤差20點那天不會
        assert stats["n_within_tolerance"] == 2
        assert stats["pct_within_tolerance"] == pytest.approx(200 / 3)

    def test_compare_daily_ohlc_no_overlap_returns_empty_stats(self):
        community = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1]},
            index=pd.Index([datetime.date(2021, 1, 4)], name="Date"))
        official = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1]},
            index=pd.Index([datetime.date(2021, 1, 5)], name="Date"))
        comparison, stats = vtha.compare_daily_ohlc(community, official)
        assert comparison.empty
        assert stats["n_dates_compared"] == 0
        assert stats["mean_abs_error_close"] is None

    def test_select_sample_dates_spreads_across_range_not_just_head(self):
        dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
        sample = vtha.select_sample_dates(dates, n=10)
        assert len(sample) == 10
        assert sample[0] == dates[0]
        assert sample[-1] == dates[-1]
        # 不是只取前10天
        assert sample != dates[:10]

    def test_select_sample_dates_returns_all_when_fewer_than_n(self):
        dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(5)]
        sample = vtha.select_sample_dates(dates, n=30)
        assert sample == dates

    def test_aggregate_minute_bars_to_daily(self):
        start = datetime.datetime(2021, 1, 5, 8, 45, 0)
        rows = []
        for i in range(10):
            t = start + datetime.timedelta(minutes=i)
            rows.append({"Timestamp": t, "Open": 18000 + i, "High": 18010 + i,
                         "Low": 17990 + i, "Close": 18005 + i, "Volume": 1})
        bars = pd.DataFrame(rows).set_index("Timestamp")
        daily = vtha.aggregate_minute_bars_to_daily(bars)
        assert len(daily) == 1
        row = daily.iloc[0]
        assert row["Open"] == 18000
        assert row["Close"] == 18014
        assert row["High"] == 18019
        assert row["Low"] == 17990


class TestFetchOfficialDailyOhlcMocked:
    def test_taifex_endpoint_success_short_circuits_fallback(self):
        # futDailyMarketReport一次查一天，回傳HTML表格(不是CSV)，這裡用
        # pd.DataFrame.to_html()組一個跟真實回應同樣形狀的HTML表格。
        html_table = pd.DataFrame({
            "契約": ["TXF"], "到期月份(週別)": ["202101"],
            "開盤價": [18000], "最高價": [18050], "最低價": [17950], "收盤價": [18010],
        }).to_html(index=False)
        resp = mock.Mock(status_code=200, content=html_table.encode("utf-8"))
        with mock.patch("validate_tx_history_accuracy.requests.post", return_value=resp):
            with mock.patch("validate_tx_history_accuracy.requests.get") as mock_get:
                df, reason, diag = vtha.fetch_official_daily_ohlc(
                    [datetime.date(2021, 1, 4), datetime.date(2021, 1, 5)])
        assert reason == "ok"
        assert len(df) == 2  # 兩個查詢日期各自用fallback_date當Date，不會互相覆蓋
        assert diag["n_dates_ok"] == 2
        mock_get.assert_not_called()  # TAIFEX兩天都成功，不需要用到data.gov.tw保底

    def test_both_sources_fail_returns_none_with_reason(self):
        # 回傳的HTML解析不出任何像樣的表格(沒有table元素)，模擬TAIFEX端點
        # 回應格式跟預期不符的情況。
        bad_resp = mock.Mock(status_code=200, content=b"<html><body>no table here</body></html>")
        get_fail_resp = mock.Mock(status_code=500, content=b"", headers={})
        with mock.patch("validate_tx_history_accuracy.requests.post", return_value=bad_resp):
            with mock.patch("validate_tx_history_accuracy.requests.get", return_value=get_fail_resp):
                with mock.patch("validate_tx_history_accuracy.MAX_RETRIES", 0):
                    df, reason, diag = vtha.fetch_official_daily_ohlc(
                        [datetime.date(2021, 1, 4), datetime.date(2021, 1, 5)])
        assert df is None
        assert reason is not None

    def test_both_sources_fail_diag_keeps_both_individual_reasons(self):
        # 這一輪新增：diag裡要同時留著TAIFEX端點跟data.gov.tw備援各自的失敗原因，
        # 不是只留最後一個(上一次真實跑就是只看到最後一個原因，看不出TAIFEX
        # 端點本身是怎麼死的，這是直接的動機)。
        bad_resp = mock.Mock(status_code=200, content=b"<html><body>no table here</body></html>")
        get_fail_resp = mock.Mock(status_code=500, content=b"", headers={})
        with mock.patch("validate_tx_history_accuracy.requests.post", return_value=bad_resp):
            with mock.patch("validate_tx_history_accuracy.requests.get", return_value=get_fail_resp):
                with mock.patch("validate_tx_history_accuracy.MAX_RETRIES", 0):
                    df, reason, diag = vtha.fetch_official_daily_ohlc(
                        [datetime.date(2021, 1, 4), datetime.date(2021, 1, 5)])
        assert diag["taifex_reason"] == "taifex_daily_no_table_parsed"
        assert diag["n_dates_ok"] == 0
        assert diag["n_dates_failed"] == 2
        assert diag["data_gov_reason"] == "data_gov_api_http_500"

    def test_some_dates_ok_some_fail_returns_partial_results(self):
        # 某幾天查得到、某幾天查不到，不該整組放棄——回傳查得到的那幾天就好，
        # 跟taifex_history_loader.py「單一period失敗不連累其他period」同樣精神。
        good_html = pd.DataFrame({
            "契約": ["TXF"], "開盤價": [18000], "最高價": [18050],
            "最低價": [17950], "收盤價": [18010],
        }).to_html(index=False)

        call_dates = []

        def fake_post(url, data=None, headers=None, timeout=None):
            call_dates.append(data["queryDate"])
            if data["queryDate"] == "2021/01/04":
                return mock.Mock(status_code=200, content=good_html.encode("utf-8"))
            return mock.Mock(status_code=200, content=b"<html><body>no table</body></html>")

        with mock.patch("validate_tx_history_accuracy.requests.post", side_effect=fake_post):
            with mock.patch("validate_tx_history_accuracy.MAX_RETRIES", 0):
                df, reason, diag = vtha.fetch_official_daily_ohlc(
                    [datetime.date(2021, 1, 4), datetime.date(2021, 1, 5)])
        assert reason == "ok"
        assert diag["n_dates_ok"] == 1
        assert diag["n_dates_failed"] == 1
        assert "2021-01-05" in diag["date_failures"]


class TestRawContentPreview:
    def test_plain_text_preview_is_safe_repr(self):
        path = _write_tmp(b"not a csv at all just words words words\n")
        preview = thl._raw_content_preview(path)
        assert "not a csv at all" in preview

    def test_zip_preview_lists_members_and_first_member_head(self):
        import zipfile
        csv_bytes = _make_tick_csv_bytes(5)
        fd, zip_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("weird_name.dat", csv_bytes)
        preview = thl._raw_content_preview(zip_path)
        assert "weird_name.dat" in preview
        assert "date" in preview  # csv header應該出現在預覽內容裡

    def test_unreadable_path_does_not_raise(self):
        preview = thl._raw_content_preview("/not/a/real/path/at/all")
        assert "失敗" in preview

    def test_unrecognized_format_diagnostics_include_raw_preview(self, tmp_path):
        cache_dir = str(tmp_path)
        os.makedirs(cache_dir, exist_ok=True)
        bad_path = os.path.join(cache_dir, "bad.raw")
        with open(bad_path, "wb") as f:
            f.write(b"totally not tabular data, just prose\nmore prose\n")

        def fake_download(period_key, cache_dir=None, refresh=False):
            return bad_path, "ok"

        with mock.patch("taifex_history_loader.download_period", side_effect=fake_download):
            bars, diagnostics = thl.load_tx_history_bars(["2011_2020"], cache_dir=cache_dir)

        assert bars is None
        period_diag = diagnostics["periods"]["2011_2020"]
        assert "raw_preview" in period_diag
        assert "prose" in period_diag["raw_preview"]


def _make_7z_bytes(csv_bytes, member_name="data.csv"):
    """用py7zr實際打包一個7z檔案(不是mock，因為第一次真實跑才發現
    crazyindicator.pixnet.net的Google Drive檔案實際是7z格式，這裡用真正的
    py7zr往返測試，確保_extract_first_member()真的能解開7z、不是只是語法上
    看起來對。"""
    import py7zr
    with tempfile.TemporaryDirectory() as src_dir:
        src_path = os.path.join(src_dir, member_name)
        with open(src_path, "wb") as f:
            f.write(csv_bytes)
        fd, archive_path = tempfile.mkstemp(suffix=".7z")
        os.close(fd)
        os.remove(archive_path)  # py7zr要自己建立檔案
        with py7zr.SevenZipFile(archive_path, mode="w") as szf:
            szf.write(src_path, arcname=member_name)
    with open(archive_path, "rb") as f:
        return f.read()


class TestSevenZipSupport:
    def test_is_archive_detects_7z_magic(self):
        assert thl._is_archive(b"7z\xbc\xaf\x27\x1c\x00\x04" + b"\x00" * 20) is True
        assert thl._is_archive(b"PK\x03\x04" + b"\x00" * 20) is True
        assert thl._is_archive(b"not an archive at all") is False

    def test_extract_first_member_from_real_7z(self):
        csv_bytes = _make_tick_csv_bytes(10)
        archive_bytes = _make_7z_bytes(csv_bytes, member_name="data.csv")
        extracted, names = thl._extract_first_member(archive_bytes)
        assert names == ["data.csv"]
        assert extracted == csv_bytes

    def test_parse_history_file_handles_7z_wrapped_csv(self):
        csv_bytes = _make_ohlc_csv_bytes(40)
        archive_bytes = _make_7z_bytes(csv_bytes, member_name="history.csv")
        path = _write_tmp(archive_bytes)
        df, reason = thl.parse_history_file(path)
        assert reason == "ok"
        assert df.attrs["source_level"] == "bar"
        assert len(df) == 40

    def test_raw_content_preview_lists_7z_member_names(self):
        csv_bytes = _make_tick_csv_bytes(5)
        archive_bytes = _make_7z_bytes(csv_bytes, member_name="weird_7z_member.csv")
        path = _write_tmp(archive_bytes)
        preview = thl._raw_content_preview(path)
        assert "weird_7z_member.csv" in preview
        assert "date" in preview  # csv header內容應該出現在預覽裡

    def test_extract_first_member_missing_py7zr_returns_none_gracefully(self):
        csv_bytes = _make_tick_csv_bytes(5)
        archive_bytes = _make_7z_bytes(csv_bytes)
        with mock.patch.dict("sys.modules", {"py7zr": None}):
            extracted, names = thl._extract_first_member(archive_bytes)
        assert extracted is None
        assert names == []


class TestColonSeparatedTimeFormat:
    """第一次真實跑crazyindicator.pixnet.net的7z檔案才發現的真實格式：
    Date欄"2011/01/03"(帶斜線，_col_matches_date()本來就有處理)，
    Time欄"08:46:00"(帶冒號，原本沒處理，回報no_time_col)。這裡用跟
    raw_preview診斷裡看到的完全一樣的欄位格式做回歸測試。"""

    def test_colon_separated_time_is_recognized(self):
        series = pd.Series(["08:46:00", "08:47:00", "13:44:59", "00:00:01"])
        assert bool(thl._col_matches_time(series)) is True

    def test_plain_digit_time_still_recognized(self):
        series = pd.Series(["084600", "084700", "134459"])
        assert bool(thl._col_matches_time(series)) is True

    def test_real_world_crazyindicator_csv_format_parses_end_to_end(self):
        # 跟validation_report.txt附的raw_preview診斷裡實際看到的欄位/格式
        # 完全一致：Date,Time,Open,High,Low,Close,Volume，Date帶斜線、
        # Time帶冒號、沒有商品代號/到期月份欄(整份資料預設都是台指期)。
        csv_text = (
            "Date,Time,Open,High,Low,Close,Volume\r\n"
            "2011/01/03,08:46:00,9000,9008,8995,9006,1340\r\n"
            "2011/01/03,08:47:00,9004,9006,9002,9003,336\r\n"
            "2011/01/03,08:48:00,9003,9009,9003,9009,514\r\n"
            "2011/01/03,08:49:00,9009,9010,9005,9008,465\r\n"
            "2011/01/03,08:50:00,9008,9015,9008,9015,672\r\n"
        )
        archive_bytes = _make_7z_bytes(csv_text.encode("utf-8"),
                                        member_name="TXF20110101_20201231(CrazyIndicator.pixnet.net).csv")
        path = _write_tmp(archive_bytes)
        df, reason = thl.parse_history_file(path)
        assert reason == "ok"
        assert df.attrs["source_level"] == "bar"
        assert len(df) == 5
        assert df.index[0] == pd.Timestamp("2011-01-03 08:46:00")
        assert df["Open"].iloc[0] == 9000
        assert df["Close"].iloc[0] == 9006


class TestOfficialDailyFetchContentPreview:
    def test_parse_failure_captures_content_preview(self):
        # 這一輪新增：HTTP本身成功但解析失敗時，應該留下實際回應內容的預覽，
        # 不是只留一個"taifex_daily_no_table_parsed"字串——上一次真實跑30天
        # 全部是這個reason，完全看不出TAIFEX真正回了什麼內容。
        bad_resp = mock.Mock(status_code=200,
                              content=b"<html><body>weird unexpected content</body></html>")
        get_fail_resp = mock.Mock(status_code=500, content=b"", headers={})
        with mock.patch("validate_tx_history_accuracy.requests.post", return_value=bad_resp):
            with mock.patch("validate_tx_history_accuracy.requests.get", return_value=get_fail_resp):
                with mock.patch("validate_tx_history_accuracy.MAX_RETRIES", 0):
                    df, reason, diag = vtha.fetch_official_daily_ohlc(
                        [datetime.date(2021, 1, 4)])
        assert df is None
        assert diag["sample_content_preview"] is not None
        assert "weird unexpected content" in diag["sample_content_preview"]

    def test_connection_failure_has_no_content_preview(self):
        # HTTP層級失敗(逾時/連線錯誤)時沒有回應內容可以預覽，不該假造一個。
        with mock.patch("validate_tx_history_accuracy.requests.post",
                         side_effect=Exception("逾時")):
            with mock.patch("validate_tx_history_accuracy.requests.get",
                             side_effect=Exception("逾時")):
                with mock.patch("validate_tx_history_accuracy.MAX_RETRIES", 0):
                    df, reason, diag = vtha.fetch_official_daily_ohlc(
                        [datetime.date(2021, 1, 4)])
        assert df is None
        assert diag["sample_content_preview"] is None

    def test_build_response_diagnostic_finds_table_tag_and_shows_its_vicinity(self):
        # 上一次真實跑只看到<head>裡的內容，完全看不出<body>有沒有表格——
        # 這裡確認診斷字串會明確指出有沒有<table>，並且摘出它附近的內容，
        # 不是永遠只看開頭500 bytes(那樣table在後面就永遠看不到)。
        content = (b"<html><head>" + b"x" * 1000 + b"</head><body><table class='table_f'>"
                   b"<tr><td>real data here</td></tr></table></body></html>")
        diagnostic = vtha._build_response_diagnostic(content)
        assert "True" in diagnostic.split("\n")[0]  # 第一行講has_table_tag
        assert "real data here" in diagnostic

    def test_build_response_diagnostic_reports_no_table_tag(self):
        content = b"<html><body>just a form page, no results</body></html>"
        diagnostic = vtha._build_response_diagnostic(content)
        assert "False" in diagnostic.split("\n")[0]
        assert "just a form page" in diagnostic

    def test_build_response_diagnostic_flags_viewstate(self):
        content = b'<html><body><input name="__VIEWSTATE" value="abc"></body></html>'
        diagnostic = vtha._build_response_diagnostic(content)
        assert "ASP.NET WebForms" in diagnostic
        assert diagnostic.split("\n")[0].count("True") == 1  # 有__VIEWSTATE，沒有<table>
