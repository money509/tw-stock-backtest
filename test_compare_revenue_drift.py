"""
test_compare_revenue_drift.py
================================
compare_revenue_drift.py的單元測試。用合成的多檔股價+月營收資料，確認兩個階段的
比較函式各自回傳正確形狀的結果，以及main()整個CLI流程在monkeypatch掉data_loader.
load_price_data跟revenue_data_loader.load_revenue_data(完全不碰網路)之後可以
端到端跑完、產生summary.txt。
"""
import sys

import numpy as np
import pandas as pd
import pytest

import compare_revenue_drift as crd
from revenue_drift_engine import (
    precompute_revenue_drift_price_indicators, precompute_all_revenue_surprise,
    build_revenue_drift_events, build_entry_date_index, HOLD_DAYS_OPTIONS,
)


def _make_price_df(closes, start="2022-01-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="B")
    closes = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
        "Volume": pd.Series(2_000_000.0, index=idx),
    }, index=idx)


def _make_revenue_df(yoy_values, start="2021-08-10", freq="MS"):
    n = len(yoy_values)
    idx = pd.date_range(start, periods=n, freq=freq) + pd.Timedelta(days=9)
    return pd.DataFrame({
        "revenue_yoy_pct": yoy_values, "revenue_mom_pct": [0.0] * n,
    }, index=idx)


def _build_synthetic_universe(n=300, seed=3, n_revenue_months=22):
    np.random.seed(seed)
    # main()固定用"2330"當參考交易日曆的標的(INDEX_PROXY_CODE)，合成資料裡一定要有它，
    # 否則main()會在抓不到2330資料時提早丟RuntimeError。
    real_codes = ["2330", "1101", "1102", "1210", "1216", "1301"]
    universe, price_data, revenue_data = {}, {}, {}
    for i, code in enumerate(real_codes):
        base = 80 + i * 10
        noise = np.random.normal(0, 1.2, n)
        closes = np.maximum(base + np.cumsum(noise * 0.25), 1.0)
        price_data[code] = _make_price_df(list(closes))
        universe[code] = {}
        yoy = list(np.random.normal(10, 5, n_revenue_months))
        revenue_data[code] = _make_revenue_df(yoy)
    index_code = real_codes[0]
    price_indicators_by_code = precompute_revenue_drift_price_indicators(price_data, universe)
    surprise_by_code = precompute_all_revenue_surprise(revenue_data)
    events_by_date = build_revenue_drift_events(surprise_by_code)
    master_calendar = price_data[index_code].index
    events_by_entry_date = build_entry_date_index(events_by_date, master_calendar)
    return (universe, price_data, revenue_data, price_indicators_by_code,
            events_by_date, events_by_entry_date, master_calendar)


class TestSplitIsOos:
    def test_splits_by_ratio(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="B")
        is_cal, oos_cal = crd.split_is_oos(idx, is_ratio=0.7)
        assert len(is_cal) == 70
        assert len(oos_cal) == 30
        assert is_cal[-1] < oos_cal[0]


class TestRunHoldDaysComparison:
    def test_returns_one_row_per_hold_days_option_with_expected_columns(self):
        (universe, price_data, revenue_data, price_indicators_by_code,
         events_by_date, events_by_entry_date, master_calendar) = _build_synthetic_universe()
        is_calendar, _ = crd.split_is_oos(master_calendar)
        execution_kwargs = dict(lots=2, top_n=2, atr_stop_mult=2.0, max_concurrent_positions=4)
        df = crd.run_hold_days_comparison(
            price_data, price_indicators_by_code, events_by_date, events_by_entry_date,
            is_calendar, 1_000_000, execution_kwargs,
        )
        expected_cols = {"hold_days", "trade_count", "profit_factor", "win_rate",
                          "total_pnl_ntd", "avg_hold_days"}
        assert expected_cols.issubset(set(df.columns))
        assert sorted(df["hold_days"]) == sorted(HOLD_DAYS_OPTIONS)

    def test_select_winning_hold_days_falls_back_to_five_when_no_rows(self):
        empty_df = pd.DataFrame(columns=["hold_days", "trade_count", "profit_factor",
                                          "win_rate", "total_pnl_ntd", "avg_hold_days"])
        assert crd.select_winning_hold_days(empty_df) == HOLD_DAYS_OPTIONS[0]

    def test_select_winning_hold_days_picks_highest_pnl_among_reliable(self):
        df = pd.DataFrame([
            {"hold_days": 5, "trade_count": 25, "profit_factor": 1.5, "win_rate": 60.0,
             "total_pnl_ntd": 5000.0, "avg_hold_days": 5.0},
            {"hold_days": 10, "trade_count": 25, "profit_factor": 0.8, "win_rate": 40.0,
             "total_pnl_ntd": -1000.0, "avg_hold_days": 10.0},
        ])
        assert crd.select_winning_hold_days(df) == 5


class TestEvaluateFinalCombo:
    def test_returns_is_oos_and_bootstrap_sections(self):
        (universe, price_data, revenue_data, price_indicators_by_code,
         events_by_date, events_by_entry_date, master_calendar) = _build_synthetic_universe()
        is_calendar, oos_calendar = crd.split_is_oos(master_calendar)
        execution_kwargs = dict(lots=2, top_n=2, atr_stop_mult=2.0, max_concurrent_positions=4)
        result = crd.evaluate_final_combo(
            "測試組合", price_data, price_indicators_by_code, events_by_date, events_by_entry_date,
            is_calendar, oos_calendar, 1_000_000, hold_days=5, execution_kwargs=execution_kwargs,
        )
        assert set(result.keys()) >= {"label", "IS", "OOS", "bootstrap", "oos_trades"}
        for split_name in ("IS", "OOS"):
            assert "trade_count" in result[split_name]
            assert "profit_factor" in result[split_name]
        b = result["bootstrap"]
        assert set(b.keys()) >= {"mean", "p5", "p95", "pct_positive", "p_value", "pnl_excluding_top3_ntd"}


class TestMainCliEndToEnd:
    def test_main_runs_end_to_end_without_network_and_writes_summary(self, tmp_path, monkeypatch):
        """用monkeypatch掉data_loader.load_price_data跟revenue_data_loader.
        load_revenue_data確保完全不碰網路，把RESULTS_DIR導到tmp_path，端到端跑一次
        main()，確認summary.txt跟各階段的CSV都有產生出來。"""
        (universe, price_data, revenue_data, _price_indicators_by_code,
         _events_by_date, _events_by_entry_date, _master_calendar) = _build_synthetic_universe(n=250, seed=5)

        def _fake_load_price_data(whitelist, start, end, refresh=False):
            return {code: price_data[code] for code in whitelist if code in price_data}

        def _fake_load_revenue_data(start, end, universe_codes=None, refresh=False, markets=("sii", "otc")):
            if universe_codes is None:
                return dict(revenue_data)
            return {code: df for code, df in revenue_data.items() if code in universe_codes}

        monkeypatch.setattr(crd, "load_price_data", _fake_load_price_data)
        monkeypatch.setattr(crd, "load_revenue_data", _fake_load_revenue_data)
        monkeypatch.setattr(crd, "RESULTS_DIR", str(tmp_path))

        argv = [
            "compare_revenue_drift.py",
            "--start", "2021-08-01", "--end", "2023-06-01",
            "--starting-capital", "1000000", "--max-stocks", "5", "--top-n", "2",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        crd.main()

        summary_path = tmp_path / "summary.txt"
        assert summary_path.exists()
        content = summary_path.read_text(encoding="utf-8")
        assert "月營收意外漂移" in content
        assert (tmp_path / "hold_days_comparison.csv").exists()
        assert (tmp_path / "trades_OOS_final_combo.csv").exists()

    def test_main_never_calls_real_network_loaders(self, monkeypatch, tmp_path):
        import data_loader
        import revenue_data_loader

        def _boom_price(*args, **kwargs):
            raise AssertionError("main()不應該呼叫真正的data_loader.load_price_data(應該被monkeypatch掉)")

        def _boom_revenue(*args, **kwargs):
            raise AssertionError("main()不應該呼叫真正的revenue_data_loader.load_revenue_data(應該被monkeypatch掉)")

        (universe, price_data, revenue_data, _price_indicators_by_code,
         _events_by_date, _events_by_entry_date, _master_calendar) = _build_synthetic_universe(n=250, seed=5)

        def _fake_load_price_data(whitelist, start, end, refresh=False):
            return {code: price_data[code] for code in whitelist if code in price_data}

        def _fake_load_revenue_data(start, end, universe_codes=None, refresh=False, markets=("sii", "otc")):
            if universe_codes is None:
                return dict(revenue_data)
            return {code: df for code, df in revenue_data.items() if code in universe_codes}

        monkeypatch.setattr(crd, "load_price_data", _fake_load_price_data)
        monkeypatch.setattr(crd, "load_revenue_data", _fake_load_revenue_data)
        monkeypatch.setattr(data_loader, "load_price_data", _boom_price)
        monkeypatch.setattr(revenue_data_loader, "load_revenue_data", _boom_revenue)
        monkeypatch.setattr(crd, "RESULTS_DIR", str(tmp_path))

        argv = [
            "compare_revenue_drift.py",
            "--start", "2021-08-01", "--end", "2023-06-01",
            "--starting-capital", "1000000", "--max-stocks", "5", "--top-n", "2",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        crd.main()  # 不應該拋例外，因為main()用的是已經被monkeypatch成假資料的crd.load_price_data/load_revenue_data
