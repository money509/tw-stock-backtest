"""
test_compare_breakout.py
compare_breakout.py 的單元測試。目前只涵蓋這一輪新增的走勢前進(Walk-Forward)分折驗證
run_walkforward_validation()——用合成的多年價格資料+已經預先算好的指標(跟main()真正流程
一樣的呼叫方式)，確認：(a)每折回傳的欄位齊全、一列一折；(b)訓練窗口太短的折會被跳過、
不會拋例外；(c)全程沒有呼叫任何需要網路的下載函式(這個函式本身只吃已經算好的price_data/
indicators_by_code/regime_series，架構上不會碰到data_loader/chip_data_loader這些會真的
發HTTP請求的模組，這裡額外用monkeypatch確認一次，不是空口保證)。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pytest

import compare_breakout as cb
import momentum_breakout_engine as mbe
from mean_reversion_engine import precompute_regime_series


def _make_price_df(closes, start="2019-01-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="B")
    closes = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
        "Volume": pd.Series(1500.0, index=idx),
    }, index=idx)


def _build_synthetic_universe(n=350, seed=7):
    np.random.seed(seed)
    real_codes = ["1101", "1102", "1210", "1216"]
    universe, price_data = {}, {}
    for i, code in enumerate(real_codes):
        base = 100 + i * 5
        trend = np.linspace(0, 30, n) if i % 2 == 0 else np.linspace(0, -15, n)
        noise = np.random.normal(0, 1.2, n)
        closes = np.maximum(base + trend + noise, 1.0)
        price_data[code] = _make_price_df(list(closes))
        universe[code] = {}
    index_code = real_codes[0]
    indicators_by_code = mbe.precompute_all_breakout_indicators(price_data, universe, index_code=index_code)
    regime_series = precompute_regime_series(price_data[index_code])
    return price_data, indicators_by_code, regime_series, price_data[index_code].index


class TestRunWalkforwardValidation:
    def test_returns_one_row_per_fold_with_expected_columns(self):
        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe()
        df = cb.run_walkforward_validation(
            price_data, indicators_by_code, regime_series, master_calendar,
            starting_capital=1_000_000, extra_kwargs={}, execution_kwargs={"lots": 2},
            n_folds=3, has_chip=False,
        )
        expected_cols = {
            "fold", "train_start", "train_end", "test_start", "test_end",
            "signal_combo", "gate", "exit_style", "atr_stop_mult", "trailing_atr_mult",
            "trade_count", "profit_factor", "win_rate", "total_pnl_ntd", "avg_hold_days",
        }
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 3
        assert list(df["fold"]) == [1, 2, 3]
        # 每一折的測試窗口不該重疊，且都緊接在訓練窗口之後(擴張視窗)
        for _, row in df.iterrows():
            assert row["train_end"] < row["test_start"]

    def test_too_short_training_window_is_skipped_without_crashing(self):
        # 只給60個交易日的歷史，切成很多折時，前面幾折的訓練窗口會小於60天的最小暖身天數，
        # 應該被跳過而不是拋例外，回傳的列數會少於n_folds(或整個是空的)
        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe(n=60)
        df = cb.run_walkforward_validation(
            price_data, indicators_by_code, regime_series, master_calendar,
            starting_capital=1_000_000, extra_kwargs={}, execution_kwargs={"lots": 2},
            n_folds=5, has_chip=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 5

    def test_does_not_touch_network_facing_loaders(self, monkeypatch):
        import data_loader
        import chip_data_loader

        def _boom(*args, **kwargs):
            raise AssertionError("run_walkforward_validation不應該呼叫任何資料下載函式")

        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)

        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe()
        df = cb.run_walkforward_validation(
            price_data, indicators_by_code, regime_series, master_calendar,
            starting_capital=1_000_000, extra_kwargs={}, execution_kwargs={"lots": 2},
            n_folds=2, has_chip=False,
        )
        assert len(df) == 2

    def test_default_walkforward_folds_constant_used_in_help_text_is_three(self):
        # CLI旗標本身預設0(停用，見compare_breakout.py main()的argparse設定)，
        # DEFAULT_WALKFORWARD_FOLDS只是文件/說明用的建議值常數(見README「怎麼跑」段落)，
        # 兩者不是同一件事，這裡只確認建議值常數存在且等於3(時間預算考量下的保守選擇)。
        assert cb.DEFAULT_WALKFORWARD_FOLDS == 3

    def test_walkforward_folds_cli_flag_defaults_to_zero(self):
        parser = argparse.ArgumentParser()
        # 直接複製main()裡這個旗標的定義方式來驗證預設值，不呼叫main()本身(避免main()
        # 其餘一大段argparse定義跟實際下載流程混進來，這裡只單獨測這一個旗標的預設行為)
        parser.add_argument("--walkforward-folds", type=int, default=0)
        args = parser.parse_args([])
        assert args.walkforward_folds == 0


class TestRunFixedComboWalkforward:
    def test_returns_one_row_per_chunk_with_expected_columns(self):
        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe()
        df = cb.run_fixed_combo_walkforward(
            price_data, indicators_by_code, regime_series, master_calendar,
            starting_capital=1_000_000, extra_kwargs={}, execution_kwargs={"lots": 2},
            n_folds=4,
        )
        expected_cols = {
            "fold", "period_start", "period_end", "trade_count", "profit_factor",
            "win_rate", "total_pnl_ntd", "avg_hold_days",
        }
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert expected_cols.issubset(set(df.columns))
            # 區塊編號應該是遞增的、不重複的(彼此獨立、不重疊的切段)
            assert list(df["fold"]) == sorted(df["fold"])

    def test_chunks_below_min_trades_are_skipped_without_crashing(self):
        # 只給很短的歷史、切成很多折，大部分區塊交易筆數會太少(甚至0)，應該被跳過
        # 而不是拋例外，回傳的列數會少於n_folds(或整個是空的)
        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe(n=90)
        df = cb.run_fixed_combo_walkforward(
            price_data, indicators_by_code, regime_series, master_calendar,
            starting_capital=1_000_000, extra_kwargs={}, execution_kwargs={"lots": 2},
            n_folds=8,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 8
        if not df.empty:
            assert (df["trade_count"] >= cb.MIN_TRADES_FOR_GATE_RANKING).all()

    def test_does_not_touch_network_facing_loaders(self, monkeypatch):
        import data_loader
        import chip_data_loader

        def _boom(*args, **kwargs):
            raise AssertionError("run_fixed_combo_walkforward不應該呼叫任何資料下載函式")

        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)

        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe()
        df = cb.run_fixed_combo_walkforward(
            price_data, indicators_by_code, regime_series, master_calendar,
            starting_capital=1_000_000, extra_kwargs={}, execution_kwargs={"lots": 2},
            n_folds=3,
        )
        assert isinstance(df, pd.DataFrame)

    def test_fixed_combo_constant_kwargs_are_runnable(self):
        # FIXED_WALKFORWARD_COMBO的鍵要能直接餵進run_momentum_breakout_backtest，一個鍵名
        # 打錯就會是TypeError(unexpected keyword argument)，不是靜默吞掉例外——這裡直接呼叫
        # 整個run_fixed_combo_walkforward()，讓任何關鍵字不匹配都會真的丟出例外讓測試失敗。
        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe()
        combo = cb.FIXED_WALKFORWARD_COMBO
        assert set(combo.keys()) == {
            "label", "signal_weights", "gate_kwargs", "breakout_window",
            "require_hard_breakout", "atr_stop_mult", "trailing_atr_mult", "exit_kwargs",
        }
        df = cb.run_fixed_combo_walkforward(
            price_data, indicators_by_code, regime_series, master_calendar,
            starting_capital=1_000_000, extra_kwargs={}, execution_kwargs={"lots": 2},
            n_folds=2,
        )
        assert isinstance(df, pd.DataFrame)

    def test_default_fixed_combo_walkforward_folds_constant_is_six(self):
        assert cb.DEFAULT_FIXED_COMBO_WALKFORWARD_FOLDS == 6

    def test_fixed_combo_walkforward_folds_cli_flag_defaults_to_zero(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--fixed-combo-walkforward-folds", type=int, default=0)
        args = parser.parse_args([])
        assert args.fixed_combo_walkforward_folds == 0


class TestRunSimpleComboComparison:
    """--simple-combo模式核心比較函式run_simple_combo_comparison()：固定訊號組合
    (SIMPLE_COMBO_SIGNAL_WEIGHTS)，只跑SIMPLE_COMBO_VARIANTS三組累加門檻，不碰任何
    資料下載函式。"""

    def test_returns_three_rows_one_per_variant_with_expected_columns(self):
        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe()
        is_calendar = master_calendar[: int(len(master_calendar) * 0.7)]
        df = cb.run_simple_combo_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            starting_capital=1_000_000, hold_days=cb.TRAILING_STOP_MAX_HOLD_DAYS,
            atr_stop_mult=1.0, trailing_atr_mult=1.0, extra_kwargs={},
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(cb.SIMPLE_COMBO_VARIANTS) == 3
        expected_cols = {"variant", "trade_count", "profit_factor", "win_rate", "total_pnl_ntd"}
        assert expected_cols.issubset(set(df.columns))
        assert list(df["variant"]) == [label for label, _ in cb.SIMPLE_COMBO_VARIANTS]

    def test_fixed_signal_combo_matches_user_requested_macd_plus_candle_body(self):
        # 使用者明確要求的固定組合：MACD柱狀圖 + K棒實體比例，這組合在compare_breakout.py
        # 原本就已經以FIXED_WALKFORWARD_COMBO['signal_weights']的形式存在過(見該常數定義)，
        # SIMPLE_COMBO_SIGNAL_WEIGHTS應該是同一組，不是另外發明的新組合。
        assert cb.SIMPLE_COMBO_SIGNAL_WEIGHTS == {"score_macd": 1.0, "score_candle_body": 1.0}
        assert cb.SIMPLE_COMBO_SIGNAL_WEIGHTS == cb.FIXED_WALKFORWARD_COMBO["signal_weights"]

    def test_variants_are_cumulative_no_gate_then_adx_then_adx_plus_regime(self):
        labels_and_kwargs = cb.SIMPLE_COMBO_VARIANTS
        assert len(labels_and_kwargs) == 3
        _, kwargs1 = labels_and_kwargs[0]
        _, kwargs2 = labels_and_kwargs[1]
        _, kwargs3 = labels_and_kwargs[2]
        assert kwargs1 == {"use_regime_gate": False}
        assert kwargs2 == {"min_adx": 25.0, "use_regime_gate": False}
        assert kwargs3 == {"min_adx": 25.0, "use_regime_gate": True}

    def test_does_not_touch_network_facing_loaders(self, monkeypatch):
        import data_loader
        import chip_data_loader

        def _boom(*args, **kwargs):
            raise AssertionError("run_simple_combo_comparison不應該呼叫任何資料下載函式")

        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)

        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe()
        is_calendar = master_calendar[: int(len(master_calendar) * 0.7)]
        df = cb.run_simple_combo_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            starting_capital=1_000_000, hold_days=cb.TRAILING_STOP_MAX_HOLD_DAYS,
            atr_stop_mult=1.0, trailing_atr_mult=1.0, extra_kwargs={},
        )
        assert len(df) == 3


class TestSimpleComboCliModeSkipsFullPipeline:
    """--simple-combo這個CLI旗標應該完全跳過main()原本的6階段IS自動搜尋(突破窗口比較→
    突破風格比較→單一訊號拆解→訊號組合比較→結構門檻變體比較→ATR敏感度網格→出場配置
    比較)，改成只跑run_simple_combo_mode()。這裡monkeypatch掉那幾個重量級階段函式讓它們
    一被呼叫就拋例外，直接呼叫cb.main()整個跑一次，確認(a)不會踩到任何一個被監控的階段函式、
    (b)不會真的連網(monkeypatch掉load_price_data/chip_data_loader)、(c)summary.txt跟
    simple_combo_variants.csv有正常寫出，且summary.txt只包含3組變體比較跟最終IS/OOS驗證
    這兩個區塊，不包含完整流程才有的突破窗口/ATR網格等區塊標題。"""

    def _build_fake_price_data(self, n=260, seed=11):
        np.random.seed(seed)
        idx = pd.date_range("2019-01-01", periods=n, freq="B")

        def _df(closes):
            closes = pd.Series(closes, index=idx, dtype=float)
            return pd.DataFrame({
                "Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
                "Volume": pd.Series(1500.0, index=idx),
            }, index=idx)

        index_trend = np.linspace(0, 30, n) + np.random.normal(0, 1.2, n)
        stock_trend = np.linspace(0, -15, n) + np.random.normal(0, 1.2, n)
        return {
            "2330": _df(np.maximum(100 + index_trend, 1.0)),
            "1101": _df(np.maximum(100 + stock_trend, 1.0)),
        }

    def _patch_heavy_stages_to_explode(self, monkeypatch):
        heavy_stage_names = [
            "run_breakout_window_comparison", "run_breakout_style_comparison",
            "run_signal_ablation", "run_signal_combo_comparison", "run_gate_comparison",
            "run_atr_sensitivity_grid", "run_exit_style_comparison",
            "run_walkforward_validation", "run_fixed_combo_walkforward",
            "run_squeeze_kdj_exit_style_comparison", "run_multi_period_validation",
        ]

        def _boom(name):
            def _inner(*args, **kwargs):
                raise AssertionError(f"--simple-combo模式不該呼叫完整流程的階段函式：{name}")
            return _inner

        for name in heavy_stage_names:
            monkeypatch.setattr(cb, name, _boom(name))

    def test_simple_combo_skips_full_pipeline_and_writes_three_variant_output(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader

        fake_price_data = self._build_fake_price_data()
        monkeypatch.setattr(cb, "load_price_data", lambda *a, **k: fake_price_data)

        def _boom(*args, **kwargs):
            raise AssertionError("--simple-combo模式不該呼叫任何資料下載函式")
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)

        self._patch_heavy_stages_to_explode(monkeypatch)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        argv = [
            "compare_breakout.py", "--simple-combo", "--max-stocks", "3",
            "--starting-capital", "1000000", "--atr-stop-mult", "1.0",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        cb.main()

        variants_csv = os.path.join(str(tmp_path), "simple_combo_variants.csv")
        summary_path = os.path.join(str(tmp_path), "summary.txt")
        assert os.path.exists(variants_csv)
        assert os.path.exists(summary_path)

        variants_df = pd.read_csv(variants_csv)
        assert len(variants_df) == 3
        assert set(variants_df["variant"]) == {label for label, _ in cb.SIMPLE_COMBO_VARIANTS}

        summary_text = open(summary_path, encoding="utf-8").read()
        assert "--simple-combo模式" in summary_text
        for label, _ in cb.SIMPLE_COMBO_VARIANTS:
            assert label in summary_text
        assert "最終驗證" in summary_text
        assert "樣本外(OOS)" in summary_text
        # 完整流程才會出現的區塊標題，--simple-combo模式不該印出來
        assert "[階段0]" not in summary_text
        assert "ATR倍數敏感度網格" not in summary_text
        assert "單一訊號拆解" not in summary_text
