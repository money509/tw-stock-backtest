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
        csv_bytes = (
            "交易日期,開盤價,最高價,最低價,收盤價\n"
            "20210104,18000,18050,17950,18010\n"
            "20210105,18010,18060,17960,18020\n"
        ).encode("utf-8")
        resp = mock.Mock(status_code=200, content=csv_bytes, headers={"Content-Type": "text/csv"})
        with mock.patch("validate_tx_history_accuracy.requests.post", return_value=resp):
            with mock.patch("validate_tx_history_accuracy.requests.get") as mock_get:
                df, reason = vtha.fetch_official_daily_ohlc(
                    datetime.date(2021, 1, 4), datetime.date(2021, 1, 5))
        assert reason == "ok"
        assert len(df) == 2
        mock_get.assert_not_called()

    def test_both_sources_fail_returns_none_with_reason(self):
        html_resp = mock.Mock(status_code=200, content=b"<html>not data</html>",
                               headers={"Content-Type": "text/html"})
        get_fail_resp = mock.Mock(status_code=500, content=b"", headers={})
        with mock.patch("validate_tx_history_accuracy.requests.post", return_value=html_resp):
            with mock.patch("validate_tx_history_accuracy.requests.get", return_value=get_fail_resp):
                with mock.patch("validate_tx_history_accuracy.MAX_RETRIES", 0):
                    df, reason = vtha.fetch_official_daily_ohlc(
                        datetime.date(2021, 1, 4), datetime.date(2021, 1, 5))
        assert df is None
        assert reason is not None
