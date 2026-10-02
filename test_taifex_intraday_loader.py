"""taifex_intraday_loader.py 的單元測試：全部用合成tick資料/合成CSV bytes，不打網路。"""
import io

import pandas as pd
import pytest

from taifex_intraday_loader import (
    parse_tick_csv, select_front_month, aggregate_ticks_to_hourly, aggregate_ticks_to_bars,
    DAY_SESSION_START, DAY_SESSION_END,
)


def _make_csv_bytes(rows):
    """rows: list of dict with keys 商品代號/到期月份(週別)/成交日期/成交時間/成交價格/成交數量(B or S)"""
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


class TestParseTickCsv:
    def test_happy_path_parses_expected_columns(self):
        rows = [
            {"商品代號": "TXF", "到期月份(週別)": "202601", "成交日期": "20260115",
             "成交時間": "084501", "成交價格": "18000", "成交數量(B or S)": "3"},
            {"商品代號": "TXF", "到期月份(週別)": "202601", "成交日期": "20260115",
             "成交時間": "084515", "成交價格": "18005", "成交數量(B or S)": "1"},
        ]
        content = _make_csv_bytes(rows)
        df, reason = parse_tick_csv(content)
        assert reason == "ok"
        assert list(df.columns) == ["ProductCode", "ContractMonth", "Price", "Volume", "Timestamp"]
        assert len(df) == 2
        assert df["Price"].iloc[0] == 18000.0
        assert df["Timestamp"].iloc[0] == pd.Timestamp("2026-01-15 08:45:01")

    def test_missing_expected_column_reports_specific_reason(self):
        # 沒有價格欄
        rows = [{"商品代號": "TXF", "到期月份(週別)": "202601", "成交日期": "20260115",
                  "成交時間": "084501", "成交數量(B or S)": "3"}]
        content = _make_csv_bytes(rows)
        df, reason = parse_tick_csv(content)
        assert df is None
        assert reason.startswith("missing_columns:")
        assert "price" in reason

    def test_garbage_content_returns_not_parsed(self):
        content = b"\x00\x01\x02not a csv at all"
        df, reason = parse_tick_csv(content)
        assert df is None
        assert reason in ("no_table_parsed", "empty_after_filter") or reason.startswith("missing_columns")

    def test_unparseable_price_rows_are_dropped_not_crashed(self):
        rows = [
            {"商品代號": "TXF", "到期月份(週別)": "202601", "成交日期": "20260115",
             "成交時間": "084501", "成交價格": "N/A", "成交數量(B or S)": "3"},
            {"商品代號": "TXF", "到期月份(週別)": "202601", "成交日期": "20260115",
             "成交時間": "084502", "成交價格": "18000", "成交數量(B or S)": "2"},
        ]
        content = _make_csv_bytes(rows)
        df, reason = parse_tick_csv(content)
        assert reason == "ok"
        assert len(df) == 1  # 第一列價格解析不出來被丟棄，不會讓整個解析失敗


class TestSelectFrontMonth:
    def test_picks_contract_month_with_most_ticks(self):
        idx = pd.date_range("2026-01-15 08:45", periods=10, freq="min")
        df = pd.DataFrame({
            "ProductCode": ["TXF"] * 10,
            "ContractMonth": ["202601"] * 7 + ["202602"] * 3,
            "Price": [18000.0] * 10,
            "Volume": [1.0] * 10,
            "Timestamp": idx,
        })
        filtered, front_month = select_front_month(df)
        assert front_month == "202601"
        assert len(filtered) == 7
        assert (filtered["ContractMonth"] == "202601").all()

    def test_empty_input_returns_none_front_month(self):
        empty = pd.DataFrame(columns=["ProductCode", "ContractMonth", "Price", "Volume", "Timestamp"])
        filtered, front_month = select_front_month(empty)
        assert front_month is None
        assert filtered.empty


class TestAggregateTicksToHourly:
    def _tick(self, ts, price, vol=1.0):
        return {"Timestamp": pd.Timestamp(ts), "Price": price, "Volume": vol}

    def test_ohlcv_within_single_hourly_bar(self):
        ticks = pd.DataFrame([
            self._tick("2026-01-15 08:45:00", 18000.0, 2),
            self._tick("2026-01-15 09:10:00", 18050.0, 1),
            self._tick("2026-01-15 09:20:00", 17950.0, 3),
            self._tick("2026-01-15 09:40:00", 18010.0, 1),
        ])
        hourly, n_dropped = aggregate_ticks_to_hourly(ticks)
        assert n_dropped == 0
        assert len(hourly) == 1
        bar = hourly.iloc[0]
        assert bar["Open"] == 18000.0
        assert bar["High"] == 18050.0
        assert bar["Low"] == 17950.0
        assert bar["Close"] == 18010.0
        assert bar["Volume"] == 7.0
        assert hourly.index[0] == pd.Timestamp("2026-01-15 08:45:00")

    def test_splits_into_multiple_hourly_bars_across_session(self):
        ticks = pd.DataFrame([
            self._tick("2026-01-15 08:50:00", 18000.0),
            self._tick("2026-01-15 09:50:00", 18010.0),  # 第二根K棒(09:45-10:45)
            self._tick("2026-01-15 13:40:00", 18020.0),  # 最後一根K棒(12:45-13:45)
        ])
        hourly, n_dropped = aggregate_ticks_to_hourly(ticks)
        assert n_dropped == 0
        assert len(hourly) == 3
        expected_starts = [
            pd.Timestamp("2026-01-15 08:45:00"),
            pd.Timestamp("2026-01-15 09:45:00"),
            pd.Timestamp("2026-01-15 12:45:00"),
        ]
        assert list(hourly.index) == expected_starts

    def test_night_session_ticks_are_dropped_and_counted(self):
        ticks = pd.DataFrame([
            self._tick("2026-01-15 08:50:00", 18000.0),
            self._tick("2026-01-15 20:00:00", 18100.0),  # 夜盤，應被過濾
            self._tick("2026-01-16 03:00:00", 18200.0),  # 夜盤，應被過濾
        ])
        hourly, n_dropped = aggregate_ticks_to_hourly(ticks)
        assert n_dropped == 2
        assert len(hourly) == 1

    def test_empty_ticks_returns_empty_bars(self):
        empty = pd.DataFrame(columns=["Timestamp", "Price", "Volume"])
        hourly, n_dropped = aggregate_ticks_to_hourly(empty)
        assert hourly.empty
        assert n_dropped == 0

    def test_session_boundaries_match_module_constants(self):
        # 確保測試本身用的時段假設跟模組實際常數一致，未來調整DAY_SESSION_*常數
        # 時這個測試會提醒要一併檢查。
        assert DAY_SESSION_START == "08:45"
        assert DAY_SESSION_END == "13:45"


class TestAggregateTicksToBars:
    """aggregate_ticks_to_bars()的通用版本(bar_minutes參數化)測試，這一輪新增
    (見taifex_intraday_loader.py模組docstring/tx_intraday_engine.py)。"""

    def _full_day_session_ticks(self, seed=0):
        """構造一整個完整日盤(08:45~13:45，共5小時=300分鐘)、每分鐘一筆的合成ticks，
        用來驗證「已知交易日長度」下算出來的K棒根數是否正確
        (5小時 -> 20根15分鐘K棒 vs 5根1小時K棒)。"""
        idx = pd.date_range("2026-01-15 08:45", "2026-01-15 13:44", freq="min")
        rng = pd.Series(range(len(idx)))
        return pd.DataFrame({
            "Timestamp": idx,
            "Price": 18000.0 + (rng % 10).to_numpy(),
            "Volume": 1.0,
        })

    def test_bar_minutes_60_matches_aggregate_ticks_to_hourly(self):
        ticks = self._full_day_session_ticks()
        bars_generic, n_dropped_generic = aggregate_ticks_to_bars(ticks, bar_minutes=60)
        bars_hourly, n_dropped_hourly = aggregate_ticks_to_hourly(ticks)
        pd.testing.assert_frame_equal(bars_generic, bars_hourly)
        assert n_dropped_generic == n_dropped_hourly

    def test_full_day_session_produces_20_bars_at_15_minutes(self):
        ticks = self._full_day_session_ticks()
        bars, n_dropped = aggregate_ticks_to_bars(ticks, bar_minutes=15)
        assert n_dropped == 0
        # (13:45-08:45) = 5小時 = 300分鐘 / 15分鐘 = 20根
        assert len(bars) == 20
        assert bars.index[0] == pd.Timestamp("2026-01-15 08:45:00")
        assert bars.index[-1] == pd.Timestamp("2026-01-15 13:30:00")

    def test_full_day_session_produces_5_bars_at_60_minutes(self):
        ticks = self._full_day_session_ticks()
        bars, n_dropped = aggregate_ticks_to_bars(ticks, bar_minutes=60)
        assert n_dropped == 0
        # (13:45-08:45) = 5小時 / 1小時 = 5根
        assert len(bars) == 5
        assert bars.index[0] == pd.Timestamp("2026-01-15 08:45:00")
        assert bars.index[-1] == pd.Timestamp("2026-01-15 12:45:00")

    def test_15_minute_bars_have_more_bars_than_60_minute_bars(self):
        ticks = self._full_day_session_ticks()
        bars_15, _ = aggregate_ticks_to_bars(ticks, bar_minutes=15)
        bars_60, _ = aggregate_ticks_to_bars(ticks, bar_minutes=60)
        assert len(bars_15) > len(bars_60)
        assert len(bars_15) == 4 * len(bars_60)

    def test_empty_ticks_returns_empty_bars_for_any_bar_minutes(self):
        empty = pd.DataFrame(columns=["Timestamp", "Price", "Volume"])
        bars, n_dropped = aggregate_ticks_to_bars(empty, bar_minutes=15)
        assert bars.empty
        assert n_dropped == 0

    def test_night_session_ticks_dropped_regardless_of_bar_minutes(self):
        ticks = pd.DataFrame([
            {"Timestamp": pd.Timestamp("2026-01-15 08:50:00"), "Price": 18000.0, "Volume": 1.0},
            {"Timestamp": pd.Timestamp("2026-01-15 20:00:00"), "Price": 18100.0, "Volume": 1.0},  # 夜盤
        ])
        bars, n_dropped = aggregate_ticks_to_bars(ticks, bar_minutes=15)
        assert n_dropped == 1
        assert len(bars) == 1


class TestLoadTxBarsGeneralization:
    """load_tx_bars()/load_tx_hourly_bars()/load_tx_15min_bars()的參數轉發測試，
    不打網路：直接monkeypatch掉下載/解析鏈路最底層的_fetch_raw_with_retry()，
    確認bar_minutes有正確傳到aggregate_ticks_to_bars()。"""

    def test_load_tx_hourly_bars_is_backward_compatible_wrapper(self, monkeypatch):
        import taifex_intraday_loader as mod

        captured = {}

        def fake_load_tx_bars(commodity_id="TXF", refresh=False, bar_minutes=60):
            captured["bar_minutes"] = bar_minutes
            captured["commodity_id"] = commodity_id
            return "sentinel_bars", {"reason": "ok"}

        monkeypatch.setattr(mod, "load_tx_bars", fake_load_tx_bars)
        result, diag = mod.load_tx_hourly_bars()
        assert captured["bar_minutes"] == 60
        assert result == "sentinel_bars"

    def test_load_tx_15min_bars_forwards_bar_minutes_15(self, monkeypatch):
        import taifex_intraday_loader as mod

        captured = {}

        def fake_load_tx_bars(commodity_id="TXF", refresh=False, bar_minutes=60):
            captured["bar_minutes"] = bar_minutes
            return "sentinel_bars", {"reason": "ok"}

        monkeypatch.setattr(mod, "load_tx_bars", fake_load_tx_bars)
        result, diag = mod.load_tx_15min_bars()
        assert captured["bar_minutes"] == 15
        assert result == "sentinel_bars"
