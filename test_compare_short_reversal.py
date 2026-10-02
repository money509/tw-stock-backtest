"""
test_compare_short_reversal.py
================================
compare_short_reversal.py的單元測試。用合成的多檔股價資料，確認三個階段的比較函式
各自回傳正確形狀的結果，以及main()整個CLI流程在monkeypatch掉data_loader.load_price_data
(不碰網路)之後可以端到端跑完、產生summary.txt。
"""
import sys

import numpy as np
import pandas as pd
import pytest

import compare_short_reversal as csr
from short_reversal_engine import precompute_all_reversal_indicators, LOOKBACK_WINDOW_OPTIONS, HOLD_DAYS_OPTIONS


def _make_price_df(closes, start="2022-01-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="B")
    closes = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
        "Volume": pd.Series(2_000_000.0, index=idx),
    }, index=idx)


def _build_synthetic_universe(n=300, seed=3):
    np.random.seed(seed)
    # main()固定用"2330"當參考交易日曆的標的(INDEX_PROXY_CODE)，合成資料裡一定要有它，
    # 否則main()會在抓不到2330資料時提早丟RuntimeError。
    real_codes = ["2330", "1101", "1102", "1210", "1216", "1301"]
    universe, price_data = {}, {}
    for i, code in enumerate(real_codes):
        base = 80 + i * 10
        noise = np.random.normal(0, 1.2, n)
        closes = np.maximum(base + np.cumsum(noise * 0.25), 1.0)
        price_data[code] = _make_price_df(list(closes))
        universe[code] = {}
    index_code = real_codes[0]
    indicators_by_code = precompute_all_reversal_indicators(price_data, universe)
    return universe, price_data, indicators_by_code, price_data[index_code].index


class TestSplitIsOos:
    def test_splits_by_ratio(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="B")
        is_cal, oos_cal = csr.split_is_oos(idx, is_ratio=0.7)
        assert len(is_cal) == 70
        assert len(oos_cal) == 30
        assert is_cal[-1] < oos_cal[0]


class TestRunLookbackWindowComparison:
    def test_returns_one_row_per_window_with_expected_columns(self):
        universe, price_data, indicators_by_code, master_calendar = _build_synthetic_universe()
        is_calendar, _ = csr.split_is_oos(master_calendar)
        execution_kwargs = dict(lots=2, top_n=2, atr_stop_mult=2.0, max_concurrent_positions=4)
        df = csr.run_lookback_window_comparison(
            price_data, indicators_by_code, is_calendar, 1_000_000, execution_kwargs,
        )
        expected_cols = {"lookback_window", "trade_count", "profit_factor", "win_rate", "total_pnl_ntd"}
        assert expected_cols.issubset(set(df.columns))
        assert sorted(df["lookback_window"]) == sorted(LOOKBACK_WINDOW_OPTIONS)

    def test_select_winning_window_falls_back_to_five_when_no_rows(self):
        empty_df = pd.DataFrame(columns=["lookback_window", "trade_count", "profit_factor",
                                          "win_rate", "total_pnl_ntd"])
        assert csr.select_winning_lookback_window(empty_df) == 5

    def test_select_winning_window_picks_highest_pnl_among_reliable(self):
        df = pd.DataFrame([
            {"lookback_window": 3, "trade_count": 25, "profit_factor": 1.2, "win_rate": 55.0, "total_pnl_ntd": 1000.0},
            {"lookback_window": 5, "trade_count": 25, "profit_factor": 1.5, "win_rate": 60.0, "total_pnl_ntd": 5000.0},
            {"lookback_window": 10, "trade_count": 25, "profit_factor": 0.8, "win_rate": 40.0, "total_pnl_ntd": -1000.0},
        ])
        assert csr.select_winning_lookback_window(df) == 5


class TestRunHoldDaysComparison:
    def test_returns_one_row_per_hold_days_option(self):
        universe, price_data, indicators_by_code, master_calendar = _build_synthetic_universe()
        is_calendar, _ = csr.split_is_oos(master_calendar)
        execution_kwargs = dict(lots=2, top_n=2, atr_stop_mult=2.0, max_concurrent_positions=4)
        df = csr.run_hold_days_comparison(
            price_data, indicators_by_code, is_calendar, 1_000_000, lookback_window=5,
            execution_kwargs=execution_kwargs,
        )
        expected_cols = {"hold_days", "trade_count", "profit_factor", "win_rate", "total_pnl_ntd", "avg_hold_days"}
        assert expected_cols.issubset(set(df.columns))
        assert sorted(df["hold_days"]) == sorted(HOLD_DAYS_OPTIONS)

    def test_select_winning_hold_days_falls_back_to_five_when_no_rows(self):
        empty_df = pd.DataFrame(columns=["hold_days", "trade_count", "profit_factor",
                                          "win_rate", "total_pnl_ntd", "avg_hold_days"])
        assert csr.select_winning_hold_days(empty_df) == 5


class TestEvaluateFinalCombo:
    def test_returns_is_oos_and_bootstrap_sections(self):
        universe, price_data, indicators_by_code, master_calendar = _build_synthetic_universe()
        is_calendar, oos_calendar = csr.split_is_oos(master_calendar)
        execution_kwargs = dict(lots=2, top_n=2, atr_stop_mult=2.0, max_concurrent_positions=4)
        result = csr.evaluate_final_combo(
            "測試組合", price_data, indicators_by_code, is_calendar, oos_calendar,
            1_000_000, lookback_window=5, hold_days=5, execution_kwargs=execution_kwargs,
        )
        assert set(result.keys()) >= {"label", "IS", "OOS", "bootstrap", "oos_trades"}
        for split_name in ("IS", "OOS"):
            assert "trade_count" in result[split_name]
            assert "profit_factor" in result[split_name]
        b = result["bootstrap"]
        assert set(b.keys()) >= {"mean", "p5", "p95", "pct_positive", "p_value", "pnl_excluding_top3_ntd"}


class TestMainCliEndToEnd:
    def test_main_runs_end_to_end_without_network_and_writes_summary(self, tmp_path, monkeypatch):
        """用monkeypatch掉data_loader.load_price_data確保完全不碰網路，把RESULTS_DIR
        導到tmp_path，端到端跑一次main()，確認summary.txt跟各階段的CSV都有產生出來。"""
        universe, price_data, _, master_calendar = _build_synthetic_universe(n=250, seed=5)

        def _fake_load_price_data(whitelist, start, end, refresh=False):
            return {code: price_data[code] for code in whitelist if code in price_data}

        monkeypatch.setattr(csr, "load_price_data", _fake_load_price_data)
        monkeypatch.setattr(csr, "RESULTS_DIR", str(tmp_path))

        argv = [
            "compare_short_reversal.py",
            "--start", "2022-01-01", "--end", "2023-06-01",
            "--starting-capital", "1000000", "--max-stocks", "5", "--top-n", "2",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        csr.main()

        summary_path = tmp_path / "summary.txt"
        assert summary_path.exists()
        content = summary_path.read_text(encoding="utf-8")
        assert "短期反轉" in content
        assert (tmp_path / "lookback_window_comparison.csv").exists()
        assert (tmp_path / "hold_days_comparison.csv").exists()
        assert (tmp_path / "trades_OOS_final_combo.csv").exists()

    def test_main_never_calls_real_network_loader(self, monkeypatch, tmp_path):
        import data_loader

        def _boom(*args, **kwargs):
            raise AssertionError("main()不應該呼叫真正的data_loader.load_price_data(應該被monkeypatch掉)")

        # 確認：如果忘記monkeypatch csr.load_price_data(只patch底層data_loader模組)，
        # 這裡應該要真的被呼叫到而丟出例外，驗證main()的下載路徑只有這一個入口。
        universe, price_data, _, master_calendar = _build_synthetic_universe(n=250, seed=5)

        def _fake_load_price_data(whitelist, start, end, refresh=False):
            return {code: price_data[code] for code in whitelist if code in price_data}

        monkeypatch.setattr(csr, "load_price_data", _fake_load_price_data)
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(csr, "RESULTS_DIR", str(tmp_path))

        argv = [
            "compare_short_reversal.py",
            "--start", "2022-01-01", "--end", "2023-06-01",
            "--starting-capital", "1000000", "--max-stocks", "5", "--top-n", "2",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        csr.main()  # 不應該拋例外，因為main()用的是已經被monkeypatch成假資料的csr.load_price_data
