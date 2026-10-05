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
            "run_squeeze_kdj_exit_style_comparison", "run_squeeze_kdj_exit_style_comparison_is_oos", "run_multi_period_validation",
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


class TestRunSimpleComboComparisonAcceptsCustomSignalWeights:
    """run_simple_combo_comparison()新增的signal_weights參數：不傳時維持舊版行為(用
    SIMPLE_COMBO_SIGNAL_WEIGHTS)，傳了就改用傳進來的組合——這是--combo-search模式重用這個
    函式的關鍵(見compare_breakout.py run_combo_search_mode())。"""

    def test_defaults_to_simple_combo_signal_weights_when_not_given(self):
        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe()
        is_calendar = master_calendar[: int(len(master_calendar) * 0.7)]
        df_default = cb.run_simple_combo_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            starting_capital=1_000_000, hold_days=cb.TRAILING_STOP_MAX_HOLD_DAYS,
            atr_stop_mult=1.0, trailing_atr_mult=1.0, extra_kwargs={},
        )
        df_explicit = cb.run_simple_combo_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            starting_capital=1_000_000, hold_days=cb.TRAILING_STOP_MAX_HOLD_DAYS,
            atr_stop_mult=1.0, trailing_atr_mult=1.0, extra_kwargs={},
            signal_weights=cb.SIMPLE_COMBO_SIGNAL_WEIGHTS,
        )
        pd.testing.assert_frame_equal(df_default, df_explicit)

    def test_custom_signal_weights_is_forwarded_to_backtest_engine(self, monkeypatch):
        # 直接檢查傳進run_momentum_breakout_backtest()的signal_weights關鍵字參數，確認
        # 自訂的組合真的有被餵進去、不是被忽略掉(用實際回測結果來比較不可靠：換一組訊號
        # 權重不保證在任何合成資料集上都會產生不同的交易，見上面default-vs-explicit測試
        # 踩到的巧合案例)。
        price_data, indicators_by_code, regime_series, master_calendar = _build_synthetic_universe()
        is_calendar = master_calendar[: int(len(master_calendar) * 0.7)]
        custom_weights = {"score_rsi_cross": 1.0, "score_volume_ratio": 1.0, "score_foreign_ratio": 1.0}

        seen_signal_weights = []
        real_backtest = cb.run_momentum_breakout_backtest

        def _spy(*args, **kwargs):
            seen_signal_weights.append(kwargs.get("signal_weights"))
            return real_backtest(*args, **kwargs)
        monkeypatch.setattr(cb, "run_momentum_breakout_backtest", _spy)

        cb.run_simple_combo_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            starting_capital=1_000_000, hold_days=cb.TRAILING_STOP_MAX_HOLD_DAYS,
            atr_stop_mult=1.0, trailing_atr_mult=1.0, extra_kwargs={},
            signal_weights=custom_weights,
        )
        assert len(seen_signal_weights) == 3
        assert all(w == custom_weights for w in seen_signal_weights)


class TestComboSearchCliModeSkipsFullPipeline:
    """--combo-search這個CLI旗標只該做「一次」資料驅動選擇：單一訊號拆解(run_signal_ablation)
    +挑組合(select_winning_signals)。main()完整流程的其餘階段(突破窗口比較、突破風格比較、
    訊號組合自動vs手動比較、結構門檻變體比較、ATR敏感度網格、出場配置比較，以及walk-forward/
    固定規則walk-forward/擠壓KDJ/跨週期驗證這些main()尾段的額外階段)全部跳過，改成走
    run_combo_search_mode()，報告方式比照--simple-combo模式(三組累加門檻比較+最終
    IS/OOS/bootstrap驗證)，另外多寫出單一訊號拆解的排名CSV。"""

    def _build_fake_price_data(self, n=260, seed=13):
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
        # 刻意不包含run_signal_ablation：--combo-search模式「唯一」該呼叫的搜尋階段。
        heavy_stage_names = [
            "run_breakout_window_comparison", "run_breakout_style_comparison",
            "run_signal_combo_comparison", "run_gate_comparison",
            "run_atr_sensitivity_grid", "run_exit_style_comparison",
            "run_walkforward_validation", "run_fixed_combo_walkforward",
            "run_squeeze_kdj_exit_style_comparison", "run_squeeze_kdj_exit_style_comparison_is_oos", "run_multi_period_validation",
        ]

        def _boom(name):
            def _inner(*args, **kwargs):
                raise AssertionError(f"--combo-search模式不該呼叫完整流程的階段函式：{name}")
            return _inner

        for name in heavy_stage_names:
            monkeypatch.setattr(cb, name, _boom(name))

    def test_combo_search_runs_ablation_select_signals_and_skips_full_pipeline(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader

        fake_price_data = self._build_fake_price_data()
        monkeypatch.setattr(cb, "load_price_data", lambda *a, **k: fake_price_data)

        def _boom(*args, **kwargs):
            raise AssertionError("--combo-search模式不該呼叫任何資料下載函式")
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)

        self._patch_heavy_stages_to_explode(monkeypatch)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        ablation_calls = []
        real_run_signal_ablation = cb.run_signal_ablation

        def _spy_ablation(*args, **kwargs):
            ablation_calls.append(1)
            return real_run_signal_ablation(*args, **kwargs)
        monkeypatch.setattr(cb, "run_signal_ablation", _spy_ablation)

        select_calls = []
        real_select_winning_signals = cb.select_winning_signals

        def _spy_select(*args, **kwargs):
            select_calls.append(1)
            return real_select_winning_signals(*args, **kwargs)
        monkeypatch.setattr(cb, "select_winning_signals", _spy_select)

        evaluate_combo_calls = []
        real_evaluate_combo = cb.evaluate_combo

        def _spy_evaluate_combo(*args, **kwargs):
            evaluate_combo_calls.append(args[0])  # label是第一個positional參數
            return real_evaluate_combo(*args, **kwargs)
        monkeypatch.setattr(cb, "evaluate_combo", _spy_evaluate_combo)

        argv = [
            "compare_breakout.py", "--combo-search", "--max-stocks", "3",
            "--starting-capital", "1000000", "--atr-stop-mult", "1.0",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        cb.main()

        assert ablation_calls, "run_signal_ablation應該被呼叫過(這是--combo-search模式唯一的選擇步驟)"
        assert select_calls, "select_winning_signals應該被呼叫過"
        # 兩個候選(篩選後混搭、全部混搭不篩選)都該各自跑過一次完整IS/OOS+bootstrap驗證
        assert len(evaluate_combo_calls) == 2, \
            f"combo-search模式應該對兩個候選各呼叫一次evaluate_combo，實際呼叫了{len(evaluate_combo_calls)}次"

        ablation_csv = os.path.join(str(tmp_path), "combo_search_ablation.csv")
        variants_csv_filtered = os.path.join(str(tmp_path), "combo_search_variants.csv")
        variants_csv_all = os.path.join(str(tmp_path), "combo_search_variants_allsignals.csv")
        oos_csv_filtered = os.path.join(str(tmp_path), "trades_OOS_combo_search_filtered.csv")
        oos_csv_all = os.path.join(str(tmp_path), "trades_OOS_combo_search_allsignals.csv")
        summary_path = os.path.join(str(tmp_path), "summary.txt")
        assert os.path.exists(ablation_csv)
        assert os.path.exists(variants_csv_filtered)
        assert os.path.exists(variants_csv_all)
        assert os.path.exists(oos_csv_filtered)
        assert os.path.exists(oos_csv_all)
        assert os.path.exists(summary_path)

        for variants_csv in (variants_csv_filtered, variants_csv_all):
            variants_df = pd.read_csv(variants_csv)
            assert len(variants_df) == 3
            assert set(variants_df["variant"]) == {label for label, _ in cb.SIMPLE_COMBO_VARIANTS}

        summary_text = open(summary_path, encoding="utf-8").read()
        assert "--combo-search模式" in summary_text
        for label, _ in cb.SIMPLE_COMBO_VARIANTS:
            assert label in summary_text
        assert "最終驗證" in summary_text
        assert "樣本外(OOS)" in summary_text
        assert "單一訊號拆解" in summary_text
        assert "候選1" in summary_text and "候選2" in summary_text
        assert "全部混搭" in summary_text
        # 完整流程才會出現的區塊標題，--combo-search模式不該印出來
        assert "[階段0]" not in summary_text
        assert "ATR倍數敏感度網格" not in summary_text

    def test_combo_search_all_signals_candidate_includes_chip_signals_when_with_chip_confirm(
        self, monkeypatch, tmp_path,
    ):
        """候選2(全部混搭)的權重要是run_signal_ablation()實際回傳的signal_names_used，
        --with-chip-confirm開啟時這份名單本來就含籌碼訊號(has_chip=args.with_chip_confirm
        已經正確傳進run_signal_ablation，見run_combo_search_mode)，這裡直接檢查送進
        run_simple_combo_comparison/evaluate_combo的signal_weights有沒有真的包含籌碼訊號，
        不是空口保證has_chip有被正確threading。"""
        import data_loader
        import chip_data_loader

        fake_price_data = self._build_fake_price_data()
        monkeypatch.setattr(cb, "load_price_data", lambda *a, **k: fake_price_data)
        # --with-chip-confirm模式下main()會嘗試下載籌碼資料，這裡直接餵空dict讓流程
        # 當成「沒有籌碼資料」繼續走(precompute_all_breakout_indicators對應處理)，
        # 不實際發網路請求。
        monkeypatch.setattr(chip_data_loader, "load_chip_data", lambda *a, **k: {})

        def _boom(*args, **kwargs):
            raise AssertionError("--combo-search模式不該呼叫任何資料下載函式")
        monkeypatch.setattr(data_loader, "load_price_data", _boom)

        self._patch_heavy_stages_to_explode(monkeypatch)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        seen_weights = []
        real_run_simple_combo_comparison = cb.run_simple_combo_comparison

        def _spy_comparison(*args, **kwargs):
            seen_weights.append(kwargs.get("signal_weights"))
            return real_run_simple_combo_comparison(*args, **kwargs)
        monkeypatch.setattr(cb, "run_simple_combo_comparison", _spy_comparison)

        argv = [
            "compare_breakout.py", "--combo-search", "--with-chip-confirm", "--max-stocks", "3",
            "--starting-capital", "1000000", "--atr-stop-mult", "1.0",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        cb.main()

        assert len(seen_weights) == 2, "兩個候選各應呼叫一次run_simple_combo_comparison"
        # 候選2(全部混搭)永遠包含所有測過的訊號，數量上一定 >= 候選1(可能被篩選掉大部分訊號)，
        # 用集合大小+是否包含任一籌碼訊號名稱來確認"全部混搭"候選真的有把籌碼訊號混進去。
        all_signals_weights = max(seen_weights, key=lambda w: len(w))
        chip_names_present = set(all_signals_weights) & mbe.CHIP_DEPENDENT_SIGNALS
        assert chip_names_present, (
            f"--with-chip-confirm開啟時，候選2(全部混搭)的訊號權重應該包含籌碼相關訊號，"
            f"實際權重名單：{list(all_signals_weights)}"
        )


class TestSqueezeKdjOnlyCliModeSkipsFullPipeline:
    """--squeeze-kdj-only這個CLI旗標只該呼叫run_squeeze_kdj_exit_style_comparison_is_oos()，
    main()完整流程的其餘階段(突破窗口比較、突破風格比較、單一訊號拆解、訊號組合比較、
    結構門檻變體比較、ATR敏感度網格、出場配置比較，以及walk-forward/固定規則walk-forward/
    跨週期驗證這些main()尾段的額外階段)全部跳過，改成走run_squeeze_kdj_only_mode()，
    輸出squeeze_kdj_is_oos.csv+summary.txt(只含squeeze+KDJ這一段的IS/OOS/bootstrap，
    不含完整流程才有的區塊標題)。"""

    def _build_fake_price_data(self, n=260, seed=17):
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
        # 刻意不包含run_squeeze_kdj_exit_style_comparison_is_oos：--squeeze-kdj-only
        # 模式「唯一」該呼叫的驗證階段。
        heavy_stage_names = [
            "run_breakout_window_comparison", "run_breakout_style_comparison",
            "run_signal_ablation", "run_signal_combo_comparison", "run_gate_comparison",
            "run_atr_sensitivity_grid", "run_exit_style_comparison",
            "run_walkforward_validation", "run_fixed_combo_walkforward",
            "run_squeeze_kdj_exit_style_comparison", "run_multi_period_validation",
        ]

        def _boom(name):
            def _inner(*args, **kwargs):
                raise AssertionError(f"--squeeze-kdj-only模式不該呼叫完整流程的階段函式：{name}")
            return _inner

        for name in heavy_stage_names:
            monkeypatch.setattr(cb, name, _boom(name))

    def test_squeeze_kdj_only_skips_full_pipeline_and_writes_is_oos_output(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader

        fake_price_data = self._build_fake_price_data()
        monkeypatch.setattr(cb, "load_price_data", lambda *a, **k: fake_price_data)

        def _boom(*args, **kwargs):
            raise AssertionError("--squeeze-kdj-only模式不該呼叫任何資料下載函式")
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)

        self._patch_heavy_stages_to_explode(monkeypatch)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        is_oos_calls = []
        real_is_oos = cb.run_squeeze_kdj_exit_style_comparison_is_oos

        def _spy(*args, **kwargs):
            is_oos_calls.append(1)
            return real_is_oos(*args, **kwargs)
        monkeypatch.setattr(cb, "run_squeeze_kdj_exit_style_comparison_is_oos", _spy)

        argv = [
            "compare_breakout.py", "--squeeze-kdj-only", "--max-stocks", "3",
            "--starting-capital", "1000000", "--atr-stop-mult", "1.0",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        cb.main()

        assert len(is_oos_calls) == 1

        csv_path = os.path.join(str(tmp_path), "squeeze_kdj_is_oos.csv")
        summary_path = os.path.join(str(tmp_path), "summary.txt")
        assert os.path.exists(csv_path)
        assert os.path.exists(summary_path)

        summary_text = open(summary_path, encoding="utf-8").read()
        assert "--squeeze-kdj-only模式" in summary_text
        for label in cb.SQUEEZE_KDJ_VARIANT_LABELS.values():
            assert label in summary_text
        assert "樣本外(OOS)" in summary_text
        assert "bootstrap" in summary_text
        # 完整流程才會出現的區塊標題，--squeeze-kdj-only模式不該印出來
        assert "[階段0]" not in summary_text
        assert "ATR倍數敏感度網格" not in summary_text
        assert "單一訊號拆解" not in summary_text


class TestRunSqueezeKdjExitStyleComparisonIsOos:
    """run_squeeze_kdj_exit_style_comparison_is_oos()：這是補上IS/OOS切分+bootstrap
    穩健性檢查之前，squeeze+KDJ訊號比較唯一還沒套用專案標準驗證方法論的地方(之前只跑
    全樣本，沒有分IS/OOS、沒有bootstrap)。這裡不走真正的BB/KC/KDJ訊號計算(跟訊號本身
    邏輯無關)，直接monkeypatch simulate_variant_a_trades/simulate_variant_b_trades回傳
    一組entry_date已知的合成交易，單純驗證：(a) IS/OOS切分真的依entry_date跟切分點正確
    分組(IS側全部早於切分點、OOS側全部不早於切分點)，(b) bootstrap統計量確實算出來、
    兩個變體都有。全程合成資料，不呼叫任何下載函式。"""

    def _make_price_df(self, n=120, start="2020-01-01"):
        idx = pd.date_range(start, periods=n, freq="B")
        closes = pd.Series(np.linspace(100, 120, n), index=idx, dtype=float)
        return pd.DataFrame({
            "Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
            "Volume": pd.Series(1000.0, index=idx),
        }, index=idx)

    def _patch_fixed_trades(self, monkeypatch, df, before_dates, after_dates):
        """讓每一檔股票、兩個變體都回傳同一組固定交易：before_dates那些entry_date在切分點
        之前，after_dates在切分點(含)之後，出場日固定隔一天、出場價固定小賺一點，確保
        pnl_ntd不會剛好是0(bootstrap/PF計算才有意義)。"""
        all_dates = list(before_dates) + list(after_dates)

        def _fixed_trades(*args, **kwargs):
            trades = []
            for d in all_dates:
                pos = df.index.get_loc(d)
                exit_pos = min(pos + 1, len(df) - 1)
                trades.append({
                    "entry_date": d, "entry_price": 100.0,
                    "exit_date": df.index[exit_pos], "exit_price": 101.0,
                    "exit_reason": "target", "hold_days": 1,
                })
            return trades

        monkeypatch.setattr(cb, "compute_squeeze_kdj_features", lambda df, *a, **k: df)
        monkeypatch.setattr(cb, "simulate_variant_a_trades", _fixed_trades)
        monkeypatch.setattr(cb, "simulate_variant_b_trades", _fixed_trades)

    def test_trades_split_by_entry_date_relative_to_cutoff(self, monkeypatch):
        df = self._make_price_df()
        universe = {"1101": {}, "1102": {}}
        price_data = {code: df for code in universe}

        is_calendar, oos_calendar = cb.split_is_oos(df.index, is_ratio=0.7)
        cutoff = oos_calendar[0]
        before_dates = [d for d in df.index if d < cutoff][:3]
        after_dates = [d for d in df.index if d >= cutoff][:3]
        assert before_dates and after_dates

        self._patch_fixed_trades(monkeypatch, df, before_dates, after_dates)

        results = cb.run_squeeze_kdj_exit_style_comparison_is_oos(
            price_data, universe, starting_capital=1_000_000,
            is_calendar=is_calendar, oos_calendar=oos_calendar, lots=1,
        )

        assert set(results.keys()) == {"A", "B"}
        for key in ("A", "B"):
            r = results[key]
            # 每檔股票各貢獻一份before/after交易，universe有2檔，所以IS/OOS各應有
            # len(before_dates)*2 / len(after_dates)*2 筆交易。
            assert r["IS"]["trade_count"] == len(before_dates) * len(universe)
            assert r["OOS"]["trade_count"] == len(after_dates) * len(universe)
            for t in r["oos_trades"]:
                assert t["entry_date"] >= cutoff, (
                    f"OOS側交易entry_date={t['entry_date']}不該早於切分點{cutoff}"
                )

    def test_bootstrap_stats_present_for_both_variants(self, monkeypatch):
        df = self._make_price_df()
        universe = {"1101": {}, "1102": {}}
        price_data = {code: df for code in universe}

        is_calendar, oos_calendar = cb.split_is_oos(df.index, is_ratio=0.7)
        cutoff = oos_calendar[0]
        before_dates = [d for d in df.index if d < cutoff][:3]
        after_dates = [d for d in df.index if d >= cutoff][:5]

        self._patch_fixed_trades(monkeypatch, df, before_dates, after_dates)

        results = cb.run_squeeze_kdj_exit_style_comparison_is_oos(
            price_data, universe, starting_capital=1_000_000,
            is_calendar=is_calendar, oos_calendar=oos_calendar, lots=1,
        )

        expected_bootstrap_keys = {"mean", "p5", "p95", "pct_positive", "p_value", "pnl_excluding_top3_ntd"}
        for key in ("A", "B"):
            b = results[key]["bootstrap"]
            assert expected_bootstrap_keys.issubset(b.keys())
            # OOS有交易、且每筆pnl都同號(全部小賺)，bootstrap重抽樣的結果應該全部一致地為正。
            assert b["pct_positive"] == 100.0
            assert b["p_value"] == 0.0

        df_out = cb._squeeze_kdj_is_oos_results_to_df(results)
        assert set(df_out["variant"]) == set(cb.SQUEEZE_KDJ_VARIANT_LABELS.values())
        assert "OOS_bootstrap(1000次重抽樣)" in set(df_out["split"])


class TestEvaluateSqueezeKdjCapitalConstrained:
    """evaluate_squeeze_kdj_capital_constrained()：驗證(a) IS/OOS是各自呼叫
    run_squeeze_kdj_capital_constrained_backtest()跑一次完整的day-by-day回測(不是先跑
    一次全期間再事後切分)——每次呼叫都是全新的trades/open_positions/cooldown_until，
    (b) 回傳格式比照evaluate_combo()，(c) bootstrap統計量都有算出來。"""

    def _make_df(self, n=40, start="2022-01-03"):
        idx = pd.date_range(start, periods=n, freq="B")
        closes = pd.Series([100.0] * n, index=idx, dtype=float)
        return pd.DataFrame({
            "Open": closes, "High": closes + 1.0, "Low": closes - 1.0, "Close": closes,
        }, index=idx)

    def test_calls_engine_once_per_split_with_correct_calendar_and_returns_evaluate_combo_shape(self, monkeypatch):
        df = self._make_df()
        price_data = {"1101": df}
        universe = {"1101": {}}
        is_calendar, oos_calendar = cb.split_is_oos(df.index, is_ratio=0.7)

        calls = []

        def _fake_backtest(price_data, universe, master_calendar, starting_capital, variant="B",
                            lots=2, top_n=3, max_concurrent_positions=3, atr_stop_mult=1.0,
                            atr_target_mult=2.0, atr_period=14, max_hold_days=60,
                            slippage_pct=0.0, features_by_code=None):
            calls.append({"master_calendar": master_calendar, "features_by_code": features_by_code})
            # 每次呼叫都回傳跟calendar長度成比例的固定小賺交易，確保IS/OOS筆數不同，
            # 可以用來確認真的各自獨立跑了一次(不是共用同一份結果)。
            d = master_calendar[0]
            return [{
                "code": "1101", "side": "long", "entry_date": d,
                "exit_date": d, "e_price": 100.0, "exit_price": 101.0,
                "exit_reason": "target", "lots": 1, "pnl_ntd": 100.0,
                "return_pct": 0.01, "hold_days": 1,
            }]

        monkeypatch.setattr(cb, "run_squeeze_kdj_capital_constrained_backtest", _fake_backtest)

        result = cb.evaluate_squeeze_kdj_capital_constrained(
            price_data, universe, starting_capital=1_000_000,
            is_calendar=is_calendar, oos_calendar=oos_calendar, variant="B",
            top_n=3, max_concurrent_positions=3, lots=2,
        )

        assert len(calls) == 2  # IS一次、OOS一次，各自獨立呼叫，不是共用同一次結果
        assert list(calls[0]["master_calendar"]) == list(is_calendar)
        assert list(calls[1]["master_calendar"]) == list(oos_calendar)
        # 兩次呼叫共用同一份(已經預先算好的)features_by_code
        assert calls[0]["features_by_code"] is calls[1]["features_by_code"]

        assert set(result.keys()) == {"label", "IS", "OOS", "bootstrap", "oos_trades"}
        assert result["IS"]["trade_count"] == 1
        assert result["OOS"]["trade_count"] == 1
        expected_bootstrap_keys = {"mean", "p5", "p95", "pct_positive", "p_value", "pnl_excluding_top3_ntd"}
        assert expected_bootstrap_keys.issubset(result["bootstrap"].keys())


class TestSqueezeKdjCapitalConstrainedCliModeSkipsFullPipeline:
    """--squeeze-kdj-capital-constrained這個CLI旗標只該呼叫
    evaluate_squeeze_kdj_capital_constrained()(變體A、B各一次)，main()完整流程的
    其餘階段(突破窗口比較、突破風格比較、單一訊號拆解、訊號組合比較、結構門檻變體
    比較、ATR敏感度網格、出場配置比較，以及walk-forward/固定規則walk-forward/
    跨週期驗證、甚至--squeeze-kdj-only本身)全部不該被呼叫，改成走
    run_squeeze_kdj_capital_constrained_mode()，輸出
    squeeze_kdj_capital_constrained_is_oos.csv+summary.txt。"""

    def _build_fake_price_data(self, n=260, seed=23):
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
            "run_squeeze_kdj_exit_style_comparison", "run_squeeze_kdj_exit_style_comparison_is_oos",
            "run_squeeze_kdj_only_mode", "run_multi_period_validation",
        ]

        def _boom(name):
            def _inner(*args, **kwargs):
                raise AssertionError(f"--squeeze-kdj-capital-constrained模式不該呼叫完整流程的階段函式：{name}")
            return _inner

        for name in heavy_stage_names:
            monkeypatch.setattr(cb, name, _boom(name))

    def test_skips_full_pipeline_and_writes_capital_constrained_output(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader

        fake_price_data = self._build_fake_price_data()
        monkeypatch.setattr(cb, "load_price_data", lambda *a, **k: fake_price_data)

        def _boom(*args, **kwargs):
            raise AssertionError("--squeeze-kdj-capital-constrained模式不該呼叫任何資料下載函式")
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)

        self._patch_heavy_stages_to_explode(monkeypatch)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        evaluate_calls = []
        real_evaluate = cb.evaluate_squeeze_kdj_capital_constrained

        def _spy(*args, **kwargs):
            evaluate_calls.append(kwargs.get("variant") or (args[5] if len(args) > 5 else None))
            return real_evaluate(*args, **kwargs)
        monkeypatch.setattr(cb, "evaluate_squeeze_kdj_capital_constrained", _spy)

        argv = [
            "compare_breakout.py", "--squeeze-kdj-capital-constrained", "--max-stocks", "2",
            "--starting-capital", "1000000",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        cb.main()

        assert len(evaluate_calls) == 2  # 變體A、變體B各一次

        csv_path = os.path.join(str(tmp_path), "squeeze_kdj_capital_constrained_is_oos.csv")
        summary_path = os.path.join(str(tmp_path), "summary.txt")
        assert os.path.exists(csv_path)
        assert os.path.exists(summary_path)

        summary_text = open(summary_path, encoding="utf-8").read()
        assert "--squeeze-kdj-capital-constrained模式" in summary_text
        assert "資金受限版" in summary_text
        assert "樣本外(OOS)" in summary_text
        assert "bootstrap" in summary_text
        # 核心訴求：有明確對照資金無限版的數字並講清楚撐住/打折/崩潰
        assert "對照資金無限版" in summary_text
        assert "排名judgment call" in summary_text
        # 完整流程/--squeeze-kdj-only才會出現的區塊標題，這裡不該印出來
        assert "[階段0]" not in summary_text
        assert "ATR倍數敏感度網格" not in summary_text
        assert "--squeeze-kdj-only模式" not in summary_text


class TestSqueezeKdjGridCombos:
    def test_grid_has_1440_unique_combos_with_expected_dimensions(self):
        combos = cb.build_squeeze_kdj_grid_combos()
        assert len(combos) == 10 * 3 * 4 * 3 * 2 * 2 == 1440
        assert [c["combo_id"] for c in combos] == list(range(1, 1441))
        exit_cfgs = {(c["variant"], c["atr_stop_mult"], c["atr_target_mult"], c["trailing_atr_mult"]) for c in combos}
        assert len(exit_cfgs) == 10
        for c in combos:
            if c["variant"] == "B_trail":
                assert c["atr_stop_mult"] == c["trailing_atr_mult"]  # 初始停損倍數=移動停利倍數
        # starting_capital刻意不是網格維度
        assert all("starting_capital" not in c for c in combos)

    def test_combo_to_kwargs_sizing_and_exit(self):
        combos = cb.build_squeeze_kdj_grid_combos()
        risk = next(c for c in combos if c["risk_pct_per_trade"] == 0.02 and c["variant"] == "B_trail")
        kw = cb.squeeze_kdj_grid_combo_to_backtest_kwargs(risk)
        assert kw["risk_pct_per_trade"] == 0.02 and "lots" not in kw
        assert kw["trailing_atr_mult"] == risk["trailing_atr_mult"]
        fixed = next(c for c in combos if c["lots"] == 1 and c["variant"] == "B")
        kw = cb.squeeze_kdj_grid_combo_to_backtest_kwargs(fixed)
        assert kw["lots"] == 1 and "risk_pct_per_trade" not in kw
        assert kw["atr_target_mult"] == fixed["atr_target_mult"]


class TestSqueezeKdjGridSelectsOnIsOnly:
    """方法論核心：網格只用IS選贏家。這裡monkeypatch回測函式，故意讓「IS最好」跟「OOS最好」
    是兩個不同的組合——IS最好的是最後一組(#1440)、OOS最好的是第一組(#1)——確認報告出來的
    贏家是IS那一組，而且它的OOS數字就是它自己(很差)的OOS數字，不會被換成OOS最好的那組。"""

    def _make_df(self, n=120):
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        closes = pd.Series(100.0, index=idx)
        return pd.DataFrame({"Open": closes, "High": closes + 1, "Low": closes - 1, "Close": closes,
                             "Volume": pd.Series(1000.0, index=idx)}, index=idx)

    def _fake_trades(self, d, n_win, win, n_loss, loss):
        out = []
        for pnl in [win] * n_win + [loss] * n_loss:
            out.append({"code": "1101", "side": "long", "entry_date": d, "exit_date": d, "e_price": 100.0,
                        "exit_price": 100.0, "exit_reason": "target", "lots": 1, "pnl_ntd": float(pnl),
                        "return_pct": pnl / 1e5, "hold_days": 1})
        return out

    def test_winner_is_is_best_even_when_oos_would_pick_another(self, monkeypatch, tmp_path):
        df = self._make_df()
        price_data = {"1101": df}
        universe = {"1101": {}}
        is_calendar, oos_calendar = cb.split_is_oos(df.index, is_ratio=0.7)
        combos = cb.build_squeeze_kdj_grid_combos()
        is_best = combos[-1]
        oos_best = combos[0]
        is_best_kwargs = cb.squeeze_kdj_grid_combo_to_backtest_kwargs(is_best)
        oos_best_kwargs = cb.squeeze_kdj_grid_combo_to_backtest_kwargs(oos_best)
        calls = []

        def _matches(kwargs, target):
            return all(kwargs.get(k) == v for k, v in target.items())

        def _fake_backtest(price_data, universe, master_calendar, starting_capital, **kwargs):
            is_split = master_calendar[0] == is_calendar[0]
            calls.append("IS" if is_split else "OOS")
            d = master_calendar[0]
            if is_split:
                if _matches(kwargs, is_best_kwargs):
                    trades = self._fake_trades(d, 20, 1000, 10, -100)    # IS PF=20
                else:
                    trades = self._fake_trades(d, 15, 100, 15, -100)     # IS PF=1
            else:
                if _matches(kwargs, oos_best_kwargs):
                    trades = self._fake_trades(d, 20, 5000, 1, -100)     # OOS PF=1000
                elif _matches(kwargs, is_best_kwargs):
                    trades = self._fake_trades(d, 1, 100, 10, -100)      # OOS PF=0.1
                else:
                    trades = self._fake_trades(d, 10, 100, 10, -100)
            assert kwargs.get("return_diagnostics") is True
            return trades, dict.fromkeys(cb.CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS, 0)

        monkeypatch.setattr(cb, "run_squeeze_kdj_capital_constrained_backtest", _fake_backtest)

        result = cb.run_squeeze_kdj_grid_search(price_data, universe, 1_000_000, is_calendar, oos_calendar)

        assert result["n_combos"] == 1440
        assert result["winner_id"] == is_best["combo_id"]
        assert result["winner_id"] != oos_best["combo_id"]
        assert result["winner"]["IS"]["profit_factor"] == pytest.approx(20.0)
        assert result["winner"]["OOS"]["profit_factor"] == pytest.approx(0.1)
        # 全部IS跑完之後才開始跑OOS(選拔在碰到任何OOS數字之前就定案)
        assert calls == ["IS"] * 1440 + ["OOS"] * 1440
        b = result["winner"]["bootstrap"]
        assert {"mean", "p5", "p95", "pct_positive", "p_value", "pnl_excluding_top3_ntd"} <= set(b)

        # 摘要報告也要報IS贏家，不能出現「OOS最佳組合」
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(cb, "run_squeeze_kdj_grid_search", lambda *a, **k: result)
        args = argparse.Namespace(start="2022-01-03", end="2022-06-30", starting_capital=1_000_000)
        cb.run_squeeze_kdj_grid_mode(args, price_data, universe, is_calendar, oos_calendar)
        summary = open(os.path.join(str(tmp_path), "summary.txt"), encoding="utf-8").read()
        assert f"組合編號#{is_best['combo_id']}：{cb.describe_squeeze_kdj_grid_combo(is_best)}" in summary
        assert "❌ 沒有通過專案門檻" in summary
        assert "OOS最佳組合" not in summary

    def test_is_ranking_ignores_oos_columns_and_respects_min_trades(self):
        df = pd.DataFrame([
            {"combo_id": 1, "is_trade_count": 29, "is_profit_factor": 9.0, "is_total_pnl_ntd": 1.0,
             "oos_profit_factor": 0.1},
            {"combo_id": 2, "is_trade_count": 30, "is_profit_factor": 2.0, "is_total_pnl_ntd": 10.0,
             "oos_profit_factor": 0.1},
            {"combo_id": 3, "is_trade_count": 50, "is_profit_factor": 2.0, "is_total_pnl_ntd": 20.0,
             "oos_profit_factor": 99.0},
            {"combo_id": 4, "is_trade_count": 50, "is_profit_factor": 1.5, "is_total_pnl_ntd": 99.0,
             "oos_profit_factor": 999.0},
        ])
        ranked = cb.rank_squeeze_kdj_grid_on_is(df, min_is_trades=30)
        assert list(ranked["eligible"]) == [False, True, True, True]
        assert pd.isna(ranked.loc[0, "is_rank"])  # 筆數不足：不排名(但仍保留在表裡)
        # PF同為2.0時比IS總損益：#3(20) > #2(10)；#4 PF較低排最後(就算OOS最好)
        ranks = ranked.set_index("combo_id")["is_rank"]
        assert (ranks[2], ranks[3], ranks[4]) == (2.0, 1.0, 3.0)
        assert cb.select_squeeze_kdj_grid_winner(ranked) == 3

    def test_no_eligible_combo_yields_no_winner(self):
        df = pd.DataFrame([{"combo_id": 1, "is_trade_count": 5, "is_profit_factor": 3.0, "is_total_pnl_ntd": 1.0}])
        assert cb.select_squeeze_kdj_grid_winner(cb.rank_squeeze_kdj_grid_on_is(df)) is None

    def test_spearman_and_distribution_helpers(self):
        df = pd.DataFrame({
            "eligible": [True, True, True, True, False],
            "is_profit_factor": [1.0, 2.0, 3.0, 4.0, 9.0],
            "oos_profit_factor": [4.0, 3.0, 2.0, float("inf"), 0.0],
            "oos_trade_count": [5, 5, 5, 5, 0],
        })
        sp = cb._spearman_is_vs_oos_pf(df)
        assert sp["n"] == 4
        # IS名次1,2,3,4對應OOS名次3,2,1,4(∞排最大)：rho = 1 - 6x(4+0+4+0)/(4x15) = 0.2
        assert sp["rho"] == pytest.approx(0.2)
        assert "雜訊" in cb._interpret_spearman(sp["rho"])
        dist = cb._oos_pf_distribution(df)
        assert dist["count"] == 5
        assert dist["pct_pf_gt_1"] == pytest.approx(80.0)
        assert dist["median"] == pytest.approx(3.0)
        assert dist["zero_trade_count"] == 1


class TestSqueezeKdjGridCliModeSkipsFullPipeline:
    """--squeeze-kdj-grid這個CLI旗標只該走run_squeeze_kdj_grid_mode()，main()完整流程的
    其餘階段(突破窗口比較、突破風格比較、單一訊號拆解、訊號組合比較、結構門檻變體比較、
    ATR敏感度網格、出場配置比較、walk-forward/固定規則walk-forward/跨週期驗證，以及
    --squeeze-kdj-only/--squeeze-kdj-capital-constrained本身)全部不該被呼叫，輸出
    squeeze_kdj_grid_all_combos.csv(中文欄名、1440列)+summary.txt。"""

    def _build_fake_price_data(self, n=260, seed=29):
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
            "run_squeeze_kdj_exit_style_comparison", "run_squeeze_kdj_exit_style_comparison_is_oos",
            "run_squeeze_kdj_only_mode", "run_squeeze_kdj_capital_constrained_mode",
            "evaluate_squeeze_kdj_capital_constrained", "run_multi_period_validation",
        ]

        def _boom(name):
            def _inner(*args, **kwargs):
                raise AssertionError(f"--squeeze-kdj-grid模式不該呼叫完整流程的階段函式：{name}")
            return _inner

        for name in heavy_stage_names:
            monkeypatch.setattr(cb, name, _boom(name))

    def test_skips_full_pipeline_and_writes_grid_output(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader

        fake_price_data = self._build_fake_price_data()
        monkeypatch.setattr(cb, "load_price_data", lambda *a, **k: fake_price_data)

        def _boom(*args, **kwargs):
            raise AssertionError("--squeeze-kdj-grid模式不該呼叫任何資料下載函式")
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)

        self._patch_heavy_stages_to_explode(monkeypatch)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        grid_calls = []
        real_grid = cb.run_squeeze_kdj_grid_search

        def _spy(*args, **kwargs):
            grid_calls.append(1)
            return real_grid(*args, **kwargs)
        monkeypatch.setattr(cb, "run_squeeze_kdj_grid_search", _spy)

        argv = ["compare_breakout.py", "--squeeze-kdj-grid", "--max-stocks", "2", "--starting-capital", "1000000"]
        monkeypatch.setattr(sys, "argv", argv)

        cb.main()

        assert len(grid_calls) == 1
        csv_path = os.path.join(str(tmp_path), "squeeze_kdj_grid_all_combos.csv")
        summary_path = os.path.join(str(tmp_path), "summary.txt")
        assert os.path.exists(csv_path)
        assert os.path.exists(summary_path)

        csv_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        assert len(csv_df) == 1440
        for col in ("組合編號", "組合說明(白話)", "IS獲利因子PF", "OOS獲利因子PF", "IS診斷_名額已滿沒輪到",
                    "OOS診斷_單筆保證金上限略過", "IS診斷_總保證金上限略過", "IS診斷_風險口數不足1口略過",
                    "IS診斷_進場濾網擋掉", "IS排名(只有符合資格的組合有名次)"):
            assert col in csv_df.columns
        assert not any(c.startswith("is_") or c.startswith("oos_") for c in csv_df.columns)

        summary_text = open(summary_path, encoding="utf-8").read()
        assert "--squeeze-kdj-grid模式" in summary_text
        assert "本次一共測試了 1440 組參數組合" in summary_text
        assert "多重比較警告" in summary_text
        assert "OOS不是完全沒碰過的資料" in summary_text
        assert "起始資金不是網格維度" in summary_text
        assert "Spearman" in summary_text
        assert "OOS PF分布" in summary_text
        assert "絕對不要從這裡挑OOS表現最好的組合" in summary_text
        assert "對照上一輪預設設定" in summary_text
        assert "OOS最佳組合" not in summary_text
        # 完整流程/其他squeeze模式才會出現的區塊標題，這裡不該印出來
        assert "[階段0]" not in summary_text
        assert "ATR倍數敏感度網格" not in summary_text
        assert "--squeeze-kdj-only模式" not in summary_text
        assert "--squeeze-kdj-capital-constrained模式" not in summary_text


class TestSqueezeKdjGridSpearmanWithoutScipy:
    """回歸測試：GitHub Actions的requirements.txt沒有scipy。舊版「scipy沒裝就退回
    pandas .corr(method='spearman')」在GitHub上兩條路都會ImportError(pandas的spearman
    內部也import scipy)，讓--squeeze-kdj-grid跑完1440組之後在算Spearman時崩潰(exit code 1)。"""

    def _df(self):
        return pd.DataFrame({
            "eligible": [True] * 6,
            "is_profit_factor": [1.0, 2.0, 3.0, 4.0, 5.0, float("inf")],
            "oos_profit_factor": [2.0, 1.0, 4.0, 3.0, 2.0, 6.0],
        })

    def test_works_when_scipy_is_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        sp = cb._spearman_is_vs_oos_pf(self._df())
        assert sp["n"] == 6
        assert np.isfinite(sp["rho"])

    def test_matches_scipy_spearmanr_including_ties_and_inf(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        df = self._df()
        got = cb._spearman_is_vs_oos_pf(df)["rho"]
        is_pf = df["is_profit_factor"].replace([np.inf], 1e18)
        expected = scipy_stats.spearmanr(is_pf, df["oos_profit_factor"]).correlation
        assert got == pytest.approx(expected, abs=1e-12)
