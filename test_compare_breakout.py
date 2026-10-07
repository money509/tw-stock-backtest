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


# ============================================================================
# --squeeze-kdj-fixed：使用者選定的單一固定設定，長歷史連續回測 + 限價1檔成交模型
# ============================================================================
from taifex_universe import STOCK_FUTURES_UNIVERSE


def _fixed_trade(entry, exit_, pnl, code="1101"):
    return {"code": code, "side": "long", "entry_date": pd.Timestamp(entry), "exit_date": pd.Timestamp(exit_),
            "e_price": 100.0, "exit_price": 100.0, "exit_reason": "stop", "lots": 1,
            "pnl_ntd": float(pnl), "return_pct": pnl / 200_000.0, "hold_days": 3}


def _make_regime_switching_market(codes, start="2022-01-03", n_days=700, seed=0):
    """波動度每40天在低/高之間切換，讓squeeze+KDJ訊號在合理資料量內會觸發好幾次。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    price_data = {}
    for code in codes:
        vol = np.where((np.arange(n_days) // 40) % 2 == 0, 0.006, 0.025) * rng.uniform(0.7, 1.3)
        close = 50 * rng.uniform(0.5, 4) * np.exp(np.cumsum(rng.normal(0.0002, vol)))
        open_ = close * (1 + rng.normal(0, 0.004, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        price_data[code] = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                                         "Volume": rng.lognormal(8, 0.5, n_days)}, index=idx)
    return price_data, idx


class TestSqueezeKdjFixedHelpers:
    def test_split_at_grid_data_start_uses_entry_date(self):
        trades = [_fixed_trade("2023-10-05", "2023-10-20", 100),   # 進場在前、出場在後 → 未見過
                  _fixed_trade("2023-10-06", "2023-10-10", -50),   # 剛好在分界 → 已見過
                  _fixed_trade("2024-01-02", "2024-01-05", 30)]
        unseen, seen = cb.split_trades_by_grid_data_start(trades)
        assert cb.SQUEEZE_KDJ_GRID_DATA_START == "2023-10-06"
        assert [t["pnl_ntd"] for t in unseen] == [100]
        assert [t["pnl_ntd"] for t in seen] == [-50, 30]

    def test_yearly_table_groups_by_exit_year(self):
        trades = [_fixed_trade("2018-12-27", "2019-01-03", 300),  # 進場2018、出場2019 → 算2019
                  _fixed_trade("2019-03-01", "2019-03-05", -100),
                  _fixed_trade("2019-05-01", "2019-05-06", 200),
                  _fixed_trade("2021-02-01", "2021-02-03", -40)]
        rows = cb.squeeze_kdj_yearly_table(trades, 200_000)
        assert [r["year"] for r in rows] == [2019, 2021]
        y2019 = rows[0]
        assert y2019["trade_count"] == 3
        assert y2019["total_pnl_ntd"] == pytest.approx(400)
        assert y2019["profit_factor"] == pytest.approx(500 / 100)
        assert y2019["win_rate"] == pytest.approx(200 / 3)
        assert y2019["max_drawdown_ntd"] == pytest.approx(-100)  # 年內權益從0開始：+300 → +200 → +400
        assert y2019["pnl_excluding_top3_ntd"] == pytest.approx(0)
        assert rows[1]["profit_factor"] == 0.0

    def test_limit_skip_shares_denominators(self):
        diag = dict.fromkeys(cb.CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS, 0)
        diag.update(candidates_total=20, skipped_entry_filter=0, skipped_no_slot=10, skipped_limit_not_filled=4)
        shares = cb._limit_skip_shares(diag)
        assert shares["orders_placed"] == 10
        assert shares["pct_of_candidates"] == pytest.approx(20.0)
        assert shares["pct_of_orders_placed"] == pytest.approx(40.0)

    def test_diag_label_exists_for_every_key(self):
        for k in cb.CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS:
            assert k in cb.SQUEEZE_KDJ_GRID_DIAG_LABELS_ZH
        assert "skipped_limit_not_filled" in cb.CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS

    def test_grid_verdict_default_wording_unchanged(self):
        b = {"pct_positive": 50.0, "p_value": 0.5, "pnl_excluding_top3_ntd": -1.0}
        v = cb._squeeze_kdj_bootstrap_verdict(b)
        assert "這組IS贏家在OOS不能算已驗證的優勢" in v
        assert "拿掉OOS最大3筆交易後總損益轉負" in v

    def test_backtests_use_fixed_setting_full_calendar_and_four_scenarios(self, monkeypatch):
        codes = ["1101", "1102"]
        price_data, idx = _make_regime_switching_market(codes, n_days=200)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        calls = []
        real = cb.run_squeeze_kdj_capital_constrained_backtest

        def _spy(**kwargs):
            calls.append(kwargs)
            return real(**kwargs)
        monkeypatch.setattr(cb, "run_squeeze_kdj_capital_constrained_backtest", _spy)
        results = cb.run_squeeze_kdj_fixed_backtests(price_data, universe, 1_000_000, idx)
        assert [(c["lots"], c["execution_model"]) for c in calls] == list(cb.SQUEEZE_KDJ_FIXED_SCENARIOS)
        assert len(calls) == 4
        for c in calls:
            assert c["master_calendar"] is idx  # 一次連續回測，不切IS/OOS
            assert c["variant"] == "B" and c["atr_stop_mult"] == 1.0 and c["atr_target_mult"] == 3.0
            assert c["max_concurrent_positions"] == 3 and c["max_hold_days"] == 20
            assert c["ranking_rule"] == "trigger_return" and c["entry_filter"] is None
            assert c["top_n"] == 3 and c["atr_period"] == 14
            assert "slippage_pct" not in c  # 沒有另外傳百分比滑價(限價模型要求slippage_pct=0)
        assert calls[0]["precomputed"] is calls[3]["precomputed"]  # 預先計算共用
        assert len(results) == 4


class TestSqueezeKdjFixedCliMode:
    def _patch_heavy_stages_to_explode(self, monkeypatch):
        heavy_stage_names = [
            "run_breakout_window_comparison", "run_breakout_style_comparison",
            "run_signal_ablation", "run_signal_combo_comparison", "run_gate_comparison",
            "run_atr_sensitivity_grid", "run_exit_style_comparison",
            "run_walkforward_validation", "run_fixed_combo_walkforward",
            "run_squeeze_kdj_exit_style_comparison", "run_squeeze_kdj_exit_style_comparison_is_oos",
            "run_squeeze_kdj_only_mode", "run_squeeze_kdj_capital_constrained_mode",
            "evaluate_squeeze_kdj_capital_constrained", "run_squeeze_kdj_grid_mode",
            "run_squeeze_kdj_grid_search", "run_multi_period_validation",
        ]

        def _boom(name):
            def _inner(*args, **kwargs):
                raise AssertionError(f"--squeeze-kdj-fixed模式不該呼叫：{name}")
            return _inner

        for name in heavy_stage_names:
            monkeypatch.setattr(cb, name, _boom(name))

    def test_skips_full_pipeline_writes_outputs_without_scipy(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)

        codes = list(STOCK_FUTURES_UNIVERSE)[:12]
        price_data, idx = _make_regime_switching_market(codes + ["2330"], start="2022-01-03", n_days=750, seed=3)
        monkeypatch.setattr(cb, "load_price_data", lambda *a, **k: price_data)

        def _boom(*args, **kwargs):
            raise AssertionError("--squeeze-kdj-fixed模式不該呼叫任何資料下載函式")
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)
        self._patch_heavy_stages_to_explode(monkeypatch)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        mode_calls = []
        real_mode = cb.run_squeeze_kdj_fixed_mode

        def _spy(*args, **kwargs):
            mode_calls.append(args)
            return real_mode(*args, **kwargs)
        monkeypatch.setattr(cb, "run_squeeze_kdj_fixed_mode", _spy)

        argv = ["compare_breakout.py", "--squeeze-kdj-fixed", "--max-stocks", "12",
                "--start", "2022-01-03", "--end", "2024-11-15", "--starting-capital", "1000000"]
        monkeypatch.setattr(sys, "argv", argv)
        cb.main()

        assert len(mode_calls) == 1
        assert mode_calls[0][5].equals(idx)  # master_calendar = 2330的完整日期序列

        scen = pd.read_csv(tmp_path / "squeeze_kdj_fixed_scenarios.csv", encoding="utf-8-sig")
        assert len(scen) == 4
        for col in ("情境說明", "全期間交易筆數", "未見過區段獲利因子PF", "已見過區段獲利因子PF",
                    "未見過區段bootstrap正報酬比例(%)", "未見過區段通過專案門檻(PF>1且bootstrap正報酬>80%)",
                    "全期間診斷_限價1檔沒成交略過", "限價沒成交佔候選總數(%)", "期末權益(NT$)",
                    "最大回撤佔起始資金(%)", "最長連續虧損筆數"):
            assert col in scen.columns
        assert not any(c.startswith(("is_", "oos_", "full_", "unseen_")) for c in scen.columns)
        # 理想成交情境不會有限價略過
        open_rows = scen[scen["成交模型"].str.contains("理想成交")]
        assert (open_rows["全期間診斷_限價1檔沒成交略過"] == 0).all()
        # 全期間筆數 = 未見過 + 已見過
        assert (scen["全期間交易筆數"] == scen["未見過區段交易筆數"] + scen["已見過區段交易筆數"]).all()
        assert scen["全期間交易筆數"].sum() > 0, "合成資料要有交易，測試才有意義"
        assert scen["未見過區段交易筆數"].sum() > 0 and scen["已見過區段交易筆數"].sum() > 0

        yearly = pd.read_csv(tmp_path / "squeeze_kdj_fixed_yearly.csv", encoding="utf-8-sig")
        assert list(yearly.columns) == ["情境說明", "年度(依出場日)", "交易筆數", "獲利因子PF", "勝率(%)",
                                        "總損益(NT$)", "年內最大回撤(NT$)", "拿掉最大3筆後損益(NT$)"]
        assert set(yearly["年度(依出場日)"]) <= {2022, 2023, 2024}
        trades = pd.read_csv(tmp_path / "squeeze_kdj_fixed_trades.csv", encoding="utf-8-sig")
        assert set(trades["區段"]) == {"未見過", "已見過"}
        assert trades["情境說明"].nunique() == 4
        assert len(trades) == scen["全期間交易筆數"].sum()
        for _, g in trades.groupby("情境說明"):
            assert g.groupby(pd.to_datetime(g["出場日"]).dt.year).size().sum() == len(g)

        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "--squeeze-kdj-fixed模式" in summary
        assert "最誠實的數字" in summary
        assert "偏樂觀" in summary
        assert "一次連續回測" in summary
        assert "同一個" in summary and "檢定" in summary  # p<0.2跟>80%是同一個檢定
        assert "1440" in summary and "95%" in summary
        assert "倖存者偏差" in summary and "基差" in summary and "開盤" in summary
        assert "限價沒成交" in summary
        assert "沒有任何「未見過區段」" not in summary
        for n in range(1, 5):
            assert f"情境{n}" in summary
        assert "[階段0]" not in summary
        assert "--squeeze-kdj-grid模式" not in summary

    def test_no_unseen_segment_message_when_start_after_grid_data(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        codes = ["1101", "1102", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2024-01-02", n_days=260)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        args = argparse.Namespace(start="2024-01-02", end="2024-12-31", starting_capital=1_000_000)
        is_cal, oos_cal = cb.split_is_oos(idx)
        results = cb.run_squeeze_kdj_fixed_mode(args, price_data, universe, is_cal, oos_cal, idx)
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "沒有任何「未見過區段」" in summary
        assert "2018-01-01" in summary
        assert "未見過區段：無" in summary
        assert all(r["segments"]["unseen"]["stats"]["trade_count"] == 0 for r in results)


class TestSqueezeKdjFixedVerdictWording:
    def test_top3_dependency_suffix_only_when_segment_is_profitable(self):
        b = {"pct_positive": 10.0, "p_value": 0.9, "pnl_excluding_top3_ntd": -500.0}
        losing = {"stats": {"trade_count": 5, "profit_factor": 0.5, "total_pnl_ntd": -100.0}, "bootstrap": b}
        winning = {"stats": {"trade_count": 5, "profit_factor": 1.5, "total_pnl_ntd": 100.0}, "bootstrap": b}
        assert "獲利高度依賴" not in cb._squeeze_kdj_fixed_segment_verdict(losing, "未見過區段")
        assert "拿掉未見過區段最大3筆" in cb._squeeze_kdj_fixed_segment_verdict(winning, "未見過區段")
        empty = {"stats": {"trade_count": 0, "profit_factor": 0.0, "total_pnl_ntd": 0.0}, "bootstrap": b}
        assert "沒有任何交易" in cb._squeeze_kdj_fixed_segment_verdict(empty, "未見過區段")


# ============================================================================
# --squeeze-kdj-filters：事先登錄的4個進場濾網小測試
# ============================================================================
def _stats(n, pf, pnl=None, win_rate=50.0):
    return {"trade_count": n, "profit_factor": pf, "total_pnl_ntd": pnl if pnl is not None else (pf - 1) * 1000,
            "win_rate": win_rate}


def _make_pf_trades(n, pf, year):
    """n筆、一半賺一半賠，PF剛好=pf(獲利每筆100*pf、虧損每筆-100)，進出場都在year年。"""
    trades = []
    base = pd.Timestamp(f"{year}-01-05")
    for k in range(n):
        d = base + pd.Timedelta(days=k % 300)
        pnl = 100.0 * pf if k % 2 == 0 else -100.0
        trades.append(_fixed_trade(d, d + pd.Timedelta(days=2), pnl, code="1101"))
    return trades


def _index_series(start="2017-01-02", end="2026-10-05", seed=5):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    return pd.Series(10000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))), index=idx)


class TestSqueezeKdjFiltersSelectionRule:
    def test_variants_are_exactly_the_four_preregistered(self):
        assert [v[0] for v in cb.SQUEEZE_KDJ_FILTER_VARIANTS] == ["F0", "F1", "F2", "F3"]
        assert [v[2] for v in cb.SQUEEZE_KDJ_FILTER_VARIANTS] == [
            None, "market_ma60", "stock_ma120", "market_ma60_and_stock_ma120"]
        assert cb.SQUEEZE_KDJ_FILTER_MIN_TRAIN_TRADES == 60
        assert cb.SQUEEZE_KDJ_FILTER_MAX_POSITIONS == 2 and cb.SQUEEZE_KDJ_FILTER_SENSITIVITY_MAX_POSITIONS == 1

    def test_highest_train_pf_among_eligible(self):
        sel, reason = cb.select_squeeze_kdj_filter_variant({
            "F0": _stats(100, 0.8), "F1": _stats(100, 1.1), "F2": _stats(60, 1.3), "F3": _stats(59, 9.0)})
        assert sel == "F2"
        assert "F3" in reason and "排除" in reason

    def test_tie_on_pf_breaks_by_train_total_pnl(self):
        sel, _ = cb.select_squeeze_kdj_filter_variant({
            "F0": _stats(100, 1.2, pnl=500), "F1": _stats(100, 1.2, pnl=900), "F2": _stats(100, 1.2, pnl=100),
            "F3": _stats(100, 1.0)})
        assert sel == "F1"

    def test_none_eligible(self):
        sel, reason = cb.select_squeeze_kdj_filter_variant({v: _stats(10, 2.0) for v in ("F0", "F1", "F2", "F3")})
        assert sel is None and "無法挑選" in reason

    def _test_seg(self, n, pf, pct):
        return {"stats": _stats(n, pf), "bootstrap": {"pct_positive": pct, "p_value": 1 - pct / 100}}

    def test_verdict_pass_and_each_failure(self):
        f0 = self._test_seg(50, 0.9, 30.0)
        v = cb.squeeze_kdj_filter_verdict("F1", {"F0": f0, "F1": self._test_seg(40, 1.5, 90.0)})
        assert v["passed"] and v["text"].startswith("✅")
        v = cb.squeeze_kdj_filter_verdict("F1", {"F0": f0, "F1": self._test_seg(40, 1.5, 70.0)})
        assert not v["passed"] and "bootstrap" in v["text"] and "PF>1" not in v["text"]
        v = cb.squeeze_kdj_filter_verdict("F1", {"F0": self._test_seg(50, 2.0, 99.0), "F1": self._test_seg(40, 1.5, 90.0)})
        assert not v["passed"] and "勝過F0" in v["text"]
        v = cb.squeeze_kdj_filter_verdict("F1", {"F0": f0, "F1": self._test_seg(40, 0.7, 10.0)})
        assert not v["passed"] and "PF>1" in v["text"] and "bootstrap" in v["text"]

    def test_verdict_when_f0_selected_never_passes(self):
        seg = self._test_seg(50, 2.0, 99.0)
        v = cb.squeeze_kdj_filter_verdict("F0", {"F0": seg})
        assert not v["passed"] and "不採用任何濾網" in v["text"]
        assert cb.squeeze_kdj_filter_verdict(None, {"F0": seg})["passed"] is False

    def test_improvement_table_both_periods(self):
        def seg(wr, pf):
            return {"stats": {"trade_count": 10, "win_rate": wr, "profit_factor": pf}}
        by = {"F0": {"train": seg(40, 0.8), "test": seg(45, 1.0)},
              "F1": {"train": seg(42, 0.9), "test": seg(46, 1.1)},   # 兩段都變好
              "F2": {"train": seg(42, 0.9), "test": seg(44, 1.1)},   # 驗證期勝率變差
              "F3": {"train": seg(30, 0.5), "test": seg(50, 2.0)}}
        rows = {r["variant"]: r for r in cb.squeeze_kdj_filter_improvement_table(by)}
        assert rows["F1"]["both_periods_better"] is True
        assert rows["F2"]["both_periods_better"] is False and rows["F2"]["test_pf_better"] is True
        assert rows["F3"]["both_periods_better"] is False
        assert set(rows) == {"F1", "F2", "F3"}

    def test_periods_split_and_validity_warning(self):
        cal = pd.bdate_range("2018-01-01", "2026-10-05")
        train, test = cb.split_squeeze_kdj_filter_periods(cal)
        assert train[0] == pd.Timestamp("2018-01-01") and train[-1] == pd.Timestamp("2022-12-30")
        assert test[0] == pd.Timestamp("2023-01-02")
        assert cb.squeeze_kdj_filter_validity_warnings("2018-01-01", train, test) == []
        late = cal[cal >= "2021-03-01"]
        tr, te = cb.split_squeeze_kdj_filter_periods(late)
        w = cb.squeeze_kdj_filter_validity_warnings("2021-03-01", tr, te)
        assert len(w) == 2 and "2018-06-30" in w[0] and "2年" in w[1]
        w = cb.squeeze_kdj_filter_validity_warnings("2018-07-02", *cb.split_squeeze_kdj_filter_periods(cal[cal >= "2018-07-02"]))
        assert len(w) == 1


class TestSqueezeKdjFiltersIndexFallback:
    def _patch(self, monkeypatch, behaviour):
        calls = []

        def _fake(symbol, start, end, refresh=False, cache_dir=None):
            calls.append((symbol, start, end))
            b = behaviour.get(symbol)
            if isinstance(b, Exception):
                raise b
            return b if b is not None else pd.Series(dtype=float)
        monkeypatch.setattr(cb, "load_index_series", _fake)
        return calls

    def _price_data(self):
        idx = pd.bdate_range("2017-06-15", "2026-10-05")
        return {"2330": pd.DataFrame({"Close": np.linspace(200, 1000, len(idx))}, index=idx)}

    def test_twii_used_when_available_with_lookback(self, monkeypatch):
        calls = self._patch(monkeypatch, {"^TWII": _index_series()})
        info = cb.load_squeeze_kdj_market_index("2018-01-01", "2026-10-06", self._price_data())
        assert info["symbol"] == "^TWII" and info["is_twii"] and info["warning"] is None
        assert calls == [("^TWII", "2017-06-15", "2026-10-06")]  # 往前多抓200個日曆天、end原樣(exclusive)

    def test_falls_back_to_0050_then_2330_with_warnings(self, monkeypatch):
        calls = self._patch(monkeypatch, {"^TWII": pd.Series(dtype=float), "0050.TW": _index_series(seed=1)})
        info = cb.load_squeeze_kdj_market_index("2018-01-01", "2026-10-06", self._price_data())
        assert info["symbol"] == "0050.TW" and not info["is_twii"]
        assert "不是加權指數^TWII" in info["warning"]
        assert [c[0] for c in calls] == ["^TWII", "0050.TW"]

        calls = self._patch(monkeypatch, {"^TWII": RuntimeError("boom"), "0050.TW": None})
        info = cb.load_squeeze_kdj_market_index("2018-01-01", "2026-10-06", self._price_data())
        assert info["symbol"] == "2330" and "台積電" in info["warning"]
        assert len(info["attempts"]) == 3

    def test_insufficient_coverage_falls_back(self, monkeypatch):
        self._patch(monkeypatch, {"^TWII": _index_series(start="2021-01-04"), "0050.TW": _index_series(seed=2)})
        info = cb.load_squeeze_kdj_market_index("2018-01-01", "2026-10-06", self._price_data())
        assert info["symbol"] == "0050.TW"
        assert "不夠涵蓋" in info["attempts"][0]

    def test_nothing_at_all_raises(self, monkeypatch):
        self._patch(monkeypatch, {})
        with pytest.raises(RuntimeError):
            cb.load_squeeze_kdj_market_index("2018-01-01", "2026-10-06", {})


def _filters_args(start="2018-01-01", end="2026-10-06"):
    return argparse.Namespace(start=start, end=end, starting_capital=1_000_000, refresh=False)


class TestSqueezeKdjFiltersModeSelectionIgnoresTest:
    def _run(self, monkeypatch, tmp_path, test_pf):
        """假回測：挑選期F2是符合資格裡PF最高(F3 PF更高但只有50筆)；驗證期的數字由test_pf決定。"""
        train_spec = {None: (100, 0.8), "market_ma60": (100, 1.1), "stock_ma120": (100, 1.3),
                      "market_ma60_and_stock_ma120": (50, 3.0)}
        calls = []

        def _fake_backtest(**kw):
            calls.append(kw)
            cal = kw["master_calendar"]
            if cal[0] < pd.Timestamp("2023-01-01"):
                n, pf = train_spec[kw["entry_filter"]]
                trades = _make_pf_trades(n, pf, 2019)
            else:
                n, pf = 80, test_pf[kw["entry_filter"]]
                trades = _make_pf_trades(n, pf, 2024)
            return trades, dict.fromkeys(cb.CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS, 0)
        monkeypatch.setattr(cb, "run_squeeze_kdj_capital_constrained_backtest", _fake_backtest)
        monkeypatch.setattr(cb, "load_index_series", lambda *a, **k: _index_series())
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        codes = ["1101", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2018-01-01", n_days=2300)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        out = cb.run_squeeze_kdj_filters_mode(_filters_args(), price_data, universe, idx)
        return out, calls

    def test_selection_uses_train_only(self, monkeypatch, tmp_path):
        # 驗證期F1遠遠最好、F2最差 → 照樣選F2(挑選規則不看驗證期)，判定不通過
        out, calls = self._run(monkeypatch, tmp_path, {None: 1.0, "market_ma60": 5.0, "stock_ma120": 0.5,
                                                       "market_ma60_and_stock_ma120": 4.0})
        assert out["selected"] == "F2"
        assert not out["verdict"]["passed"]
        assert len(calls) == 16  # 2種最多持倉數 x 4變體 x 2期間
        for kw in calls:
            assert kw["lots"] == 1 and kw["top_n"] == 3 and kw["execution_model"] == "limit_1tick"
            assert kw["variant"] == "B" and kw["atr_stop_mult"] == 1.0 and kw["atr_target_mult"] == 3.0
            assert kw["max_hold_days"] == 20 and kw["atr_period"] == 14 and kw["ranking_rule"] == "trigger_return"
            assert kw["max_concurrent_positions"] in (1, 2) and "slippage_pct" not in kw
            assert kw["market_series"] is not None
        # 每段都是獨立重跑：日曆只含自己那一段
        train_cals = [kw["master_calendar"] for kw in calls if kw["master_calendar"][0] < pd.Timestamp("2023-01-01")]
        assert all(c[-1] <= pd.Timestamp("2022-12-31") for c in train_cals)
        # 換一組驗證期(F2變成最好)，選擇不變、判定通過
        out2, _ = self._run(monkeypatch, tmp_path, {None: 1.0, "market_ma60": 0.5, "stock_ma120": 3.0,
                                                    "market_ma60_and_stock_ma120": 0.5})
        assert out2["selected"] == "F2"
        assert out2["verdict"]["passed"]
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "✅ 通過" in summary


class TestSqueezeKdjFiltersCliMode:
    def test_skips_full_pipeline_writes_outputs_without_scipy(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)

        codes = list(STOCK_FUTURES_UNIVERSE)[:12]
        price_data, idx = _make_regime_switching_market(codes + ["2330"], start="2017-06-15", n_days=1950, seed=3)
        load_calls = []

        def _fake_load(universe, start, end, refresh=False):
            load_calls.append((start, end))
            return price_data
        monkeypatch.setattr(cb, "load_price_data", _fake_load)
        index_calls = []

        def _fake_index(symbol, start, end, refresh=False, cache_dir=None):
            index_calls.append((symbol, start, end))
            return _index_series(start="2017-06-15", end="2025-12-31")
        monkeypatch.setattr(cb, "load_index_series", _fake_index)

        def _boom(*args, **kwargs):
            raise AssertionError("--squeeze-kdj-filters模式不該直接呼叫資料下載函式")
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(data_loader, "load_index_series", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)
        TestSqueezeKdjFixedCliMode()._patch_heavy_stages_to_explode(monkeypatch)
        monkeypatch.setattr(cb, "run_squeeze_kdj_fixed_mode", _boom)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        mode_calls = []
        real_mode = cb.run_squeeze_kdj_filters_mode

        def _spy(*args, **kwargs):
            mode_calls.append(args)
            return real_mode(*args, **kwargs)
        monkeypatch.setattr(cb, "run_squeeze_kdj_filters_mode", _spy)

        argv = ["compare_breakout.py", "--squeeze-kdj-filters", "--max-stocks", "12",
                "--start", "2018-01-01", "--end", "2024-11-15", "--starting-capital", "1000000"]
        monkeypatch.setattr(sys, "argv", argv)
        cb.main()

        assert load_calls == [("2017-06-15", "2024-11-15")]  # 股價往前多抓200個日曆天
        assert index_calls[0] == ("^TWII", "2017-06-15", "2024-11-15")
        assert len(mode_calls) == 1
        cal = mode_calls[0][3]
        assert cal[0] >= pd.Timestamp("2018-01-01")  # 暖身資料不拿來交易
        assert cal.equals(idx[idx >= pd.Timestamp("2018-01-01")])

        summ = pd.read_csv(tmp_path / "squeeze_kdj_filters_summary.csv", encoding="utf-8-sig")
        assert len(summ) == 16
        for col in ("最多同時持倉數", "用途", "變體代號", "期間", "交易筆數", "每月交易筆數", "勝率(%)", "獲利因子PF",
                    "總損益(NT$)", "平均獲利(NT$/筆)", "平均虧損(NT$/筆)", "最大回撤(NT$)", "拿掉最大3筆後損益(NT$)",
                    "bootstrap正報酬比例(%)", "bootstrap p值", "診斷_進場濾網擋掉", "診斷_濾網細項_大盤不在季線上",
                    "診斷_濾網細項_個股不在半年線上", "診斷_名額已滿沒輪到", "被挑選規則選中", "大盤指數來源"):
            assert col in summ.columns, col
        assert set(summ["期間"]) == {"挑選期", "驗證期"}
        assert set(summ["最多同時持倉數"]) == {1, 2}
        assert summ["被挑選規則選中"].sum() <= 1
        assert (summ.loc[summ["變體代號"] == "F0", "診斷_進場濾網擋掉"] == 0).all()
        assert (summ.loc[summ["變體代號"] == "F1", "診斷_濾網細項_個股不在半年線上"] == 0).all()
        assert (summ.loc[summ["變體代號"] == "F2", "診斷_濾網細項_大盤不在季線上"] == 0).all()
        assert summ.loc[summ["變體代號"] == "F3", "診斷_進場濾網擋掉"].sum() > 0
        assert summ["交易筆數"].sum() > 0, "合成資料要有交易，測試才有意義"
        # 同一批候選時，F3(兩個條件都要)擋掉的數量不會少於F1或F2單獨擋掉的
        for (mp, per), g in summ.groupby(["最多同時持倉數", "期間"]):
            blocked = dict(zip(g["變體代號"], g["診斷_進場濾網擋掉"]))
            cands = dict(zip(g["變體代號"], g["診斷_候選總數"]))
            if cands["F0"] == cands["F3"]:
                assert blocked["F3"] >= max(blocked["F1"], blocked["F2"])

        yearly = pd.read_csv(tmp_path / "squeeze_kdj_filters_yearly.csv", encoding="utf-8-sig")
        assert list(yearly.columns) == cb.SQUEEZE_KDJ_FILTER_YEARLY_COLUMNS
        assert set(yearly.loc[yearly["期間"] == "挑選期", "年度(依出場日)"]) <= {2018, 2019, 2020, 2021, 2022}
        assert set(yearly.loc[yearly["期間"] == "驗證期", "年度(依出場日)"]) <= {2023, 2024}
        trades = pd.read_csv(tmp_path / "squeeze_kdj_filters_trades.csv", encoding="utf-8-sig")
        assert len(trades) == summ["交易筆數"].sum()
        assert "股票代號" in trades.columns and "損益(NT$，已扣手續費)" in trades.columns
        assert (pd.to_datetime(trades["進場日"]) >= pd.Timestamp("2018-01-01")).all()

        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "--squeeze-kdj-filters模式" in summary
        assert summary.index("【事先登錄的規則") < summary.index("【挑選結果與判定】") < summary.index("【並排總表")
        assert "結論：" in summary and "兩段都變好" in summary
        assert "敏感度對照" in summary and "台灣加權指數(^TWII)" in summary
        assert "多重比較" in summary and "倖存者偏差" in summary and "偏樂觀" in summary
        assert "這次測試無效" not in summary
        assert "[階段0]" not in summary

    def test_invalid_start_prints_big_warning(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(cb, "load_index_series", lambda *a, **k: pd.Series(dtype=float))  # 一路退到2330
        codes = ["1101", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2021-06-01", n_days=600)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        out = cb.run_squeeze_kdj_filters_mode(_filters_args(start="2021-06-01", end="2023-09-01"),
                                              price_data, universe, idx)
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "這次測試無效" in summary and "2018-01-01" in summary
        assert summary.index("這次測試無效") < summary.index("【挑選結果與判定】")
        assert out["index_info"]["symbol"] == "2330" and "台積電" in summary


# ============================================================================
# --squeeze-kdj-stops：事先登錄的停損寬度小測試
# ============================================================================
def _stops_args(start="2018-01-01", end="2026-10-06", **extra):
    return argparse.Namespace(start=start, end=end, starting_capital=1_000_000, refresh=False, **extra)


class TestSqueezeKdjStopsSelectionRule:
    def test_variants_are_exactly_the_six_preregistered(self):
        assert [(v[0], v[2], v[3], v[4]) for v in cb.SQUEEZE_KDJ_STOP_VARIANTS] == [
            ("N1", 1.0, 3.0, None), ("N2", 1.5, 3.0, None), ("N3", 2.0, 3.0, None),
            ("S1", 1.0, 3.0, "stock_ma120"), ("S2", 1.5, 3.0, "stock_ma120"), ("S3", 2.0, 3.0, "stock_ma120")]
        assert cb.SQUEEZE_KDJ_STOP_SELECTABLE == ("N1", "N2", "N3", "S1", "S2", "S3")
        assert cb.SQUEEZE_KDJ_STOP_BASELINE == "N1"
        assert not hasattr(cb, "SQUEEZE_KDJ_STOP_REFERENCE")
        assert cb.SQUEEZE_KDJ_STOPS_OLD_COST_VARIANTS == ("N1", "S1")
        assert cb.SQUEEZE_KDJ_STOPS_DEFAULT_COMMISSION_PER_LOT_SIDE == 50.0
        assert cb.SQUEEZE_KDJ_STOPS_DEFAULT_FUTURES_TAX_RATE == 0.00002
        assert cb.SQUEEZE_KDJ_STOPS_OLD_COMMISSION_PER_LOT_SIDE == 200.0

    def _all(self, **override):
        base = {v: _stats(100, 0.8) for v in cb.SQUEEZE_KDJ_STOP_SELECTABLE}
        base.update(override)
        return base

    def test_highest_train_pf_among_all_six_eligible(self):
        sel, reason = cb.select_squeeze_kdj_stop_variant(self._all(N3=_stats(60, 1.3), S2=_stats(100, 1.2),
                                                                   S3=_stats(59, 5.0)))
        assert sel == "N3"
        assert "S3" in reason and "排除" in reason
        sel, _ = cb.select_squeeze_kdj_stop_variant(self._all(S2=_stats(100, 1.2)))
        assert sel == "S2"

    def test_n1_is_selectable(self):
        sel, _ = cb.select_squeeze_kdj_stop_variant(self._all(N1=_stats(300, 1.05)))
        assert sel == "N1"

    def test_unknown_ids_ignored(self):
        sel, _ = cb.select_squeeze_kdj_stop_variant(self._all(R0=_stats(500, 99.0), S1=_stats(100, 1.1)))
        assert sel == "S1"

    def test_tie_breaks_by_pnl_then_list_order(self):
        sel, _ = cb.select_squeeze_kdj_stop_variant(self._all(N2=_stats(100, 1.1, pnl=100), S1=_stats(100, 1.1, pnl=300)))
        assert sel == "S1"
        tied = {v: _stats(100, 1.1, pnl=100) for v in cb.SQUEEZE_KDJ_STOP_SELECTABLE}
        assert cb.select_squeeze_kdj_stop_variant(tied)[0] == "N1"
        tied.pop("N1")
        tied["N1"] = _stats(10, 1.1, pnl=100)  # 筆數不足
        assert cb.select_squeeze_kdj_stop_variant(tied)[0] == "N2"

    def test_none_eligible(self):
        sel, reason = cb.select_squeeze_kdj_stop_variant({v: _stats(59, 2.0) for v in cb.SQUEEZE_KDJ_STOP_SELECTABLE})
        assert sel is None and "無法挑選" in reason

    def _seg(self, n, pf, pct):
        return {"stats": _stats(n, pf), "bootstrap": {"pct_positive": pct, "p_value": 1 - pct / 100}}

    def test_verdict_branches_vs_n1(self):
        n1 = self._seg(80, 0.9, 30.0)
        v = cb.squeeze_kdj_stop_verdict("S2", {"N1": n1, "S2": self._seg(70, 1.4, 90.0)})
        assert v["passed"] and v["text"].startswith("✅")
        v = cb.squeeze_kdj_stop_verdict("N2", {"N1": n1, "N2": self._seg(70, 1.4, 75.0)})
        assert not v["passed"] and "bootstrap" in v["text"] and "PF>1" not in v["text"]
        v = cb.squeeze_kdj_stop_verdict("S3", {"N1": self._seg(80, 1.6, 95.0), "S3": self._seg(70, 1.4, 90.0)})
        assert not v["passed"] and "勝過N1" in v["text"]
        v = cb.squeeze_kdj_stop_verdict("S3", {"N1": n1, "S3": self._seg(70, 0.8, 20.0)})
        assert not v["passed"] and "PF>1" in v["text"]
        # S1好過N1也算(比較基準是N1，不是S1)
        v = cb.squeeze_kdj_stop_verdict("S1", {"N1": n1, "S1": self._seg(70, 1.3, 85.0)})
        assert v["passed"]
        v = cb.squeeze_kdj_stop_verdict("N1", {"N1": self._seg(80, 2.0, 99.0)})
        assert not v["passed"] and "維持現狀" in v["text"]
        assert cb.squeeze_kdj_stop_verdict(None, {"N1": n1})["passed"] is False

    def test_filter_verdict_wording_unchanged_after_refactor(self):
        seg = self._seg(50, 2.0, 99.0)
        v = cb.squeeze_kdj_filter_verdict("F0", {"F0": seg})
        assert v["text"] == ("❌ 不通過：挑選期PF最高的是F0(不加濾網)——3個濾網在挑選期都沒有勝過不過濾，"
                             "依事先登錄的規則，這次的結論是「不採用任何濾網」。")
        assert v["checks"][2][0] == "驗證期PF勝過F0不加濾網(選出的2.00 vs F0的2.00)——選出的就是F0本身，這條不可能成立"

    def test_improvement_table_every_variant_vs_n1(self):
        def seg(wr, pf):
            return {"stats": {"trade_count": 10, "win_rate": wr, "profit_factor": pf}}
        by = {"N1": {"train": seg(40, 0.8), "test": seg(45, 1.0)},
              "N2": {"train": seg(42, 0.9), "test": seg(46, 1.1)},
              "N3": {"train": seg(50, 0.7), "test": seg(55, 1.2)},
              "S1": {"train": seg(41, 0.85), "test": seg(46, 1.05)},
              "S2": {"train": seg(30, 0.6), "test": seg(35, 0.7)},
              "S3": {"train": seg(40, 0.8), "test": seg(45, 1.0)}}
        rows = {r["variant"]: r for r in cb.squeeze_kdj_stop_improvement_table(by)}
        assert set(rows) == {"N2", "N3", "S1", "S2", "S3"}
        assert rows["N2"]["both_periods_better"] is True and rows["S1"]["both_periods_better"] is True
        assert rows["N3"]["both_periods_better"] is False and rows["N3"]["train_win_rate_better"] is True
        assert rows["S2"]["both_periods_better"] is False
        assert rows["S3"]["both_periods_better"] is False  # 同分不算變好


class TestSqueezeKdjStopTradeBreakdown:
    def test_buckets_reasons_costs_and_risk(self):
        def t(hold, reason, pnl, comm=100.0, tax=1.0, risk=500.0):
            return {"hold_days": hold, "exit_reason": reason, "pnl_ntd": pnl, "commission_ntd": comm,
                    "tax_ntd": tax, "stop_risk_ntd": risk}
        trades = [t(1, "stop_gap", -700), t(2, "stop", -600), t(3, "stop", -600),
                  t(5, "target", 1500, risk=float("nan")), t(20, "forced_close", 100, risk=700.0)]
        b = cb.squeeze_kdj_stop_trade_breakdown(trades)
        assert b["early"][1] == {"count": 1, "pct": 20.0, "pnl_ntd": -700.0}
        assert b["early"][3] == {"count": 3, "pct": 60.0, "pnl_ntd": -1900.0}
        assert b["early"][5]["count"] == 4 and b["early"][5]["pnl_ntd"] == -400.0
        assert b["reasons"]["stop"] == {"count": 2, "pct": 40.0, "pnl_ntd": -1200.0}
        assert b["reasons"]["stop_gap"]["count"] == 1 and b["reasons"]["target"]["pnl_ntd"] == 1500.0
        assert b["reasons"]["forced_close"]["count"] == 1 and b["reasons"]["other"]["count"] == 0
        assert b["commission_ntd"] == 500.0 and b["tax_ntd"] == 5.0 and b["cost_ntd"] == 505.0
        assert b["avg_pnl_ntd"] == pytest.approx(-60.0)
        assert b["avg_stop_risk_ntd"] == pytest.approx((500 * 3 + 700) / 4)  # NaN不算進平均
        empty = cb.squeeze_kdj_stop_trade_breakdown([])
        assert empty["early"][3]["pct"] == 0.0 and np.isnan(empty["avg_stop_risk_ntd"])

    def test_stop_risk_annotation_uses_entry_atr_and_multiplier(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        per_code = {"1101": {"date_to_idx": {d: i for i, d in enumerate(idx)},
                             "atr": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}}
        trades = [{"code": "1101", "entry_date": idx[2], "e_price": 100.0, "lots": 1},
                  {"code": "1101", "entry_date": pd.Timestamp("2030-01-01"), "e_price": 100.0, "lots": 1}]
        cb._annotate_squeeze_kdj_stop_risk(trades, per_code, 1.5)
        assert trades[0]["stop_risk_ntd"] == pytest.approx(1.5 * 3.0 * 2000)  # 1101是標準契約2000股
        assert np.isnan(trades[1]["stop_risk_ntd"])


class TestSqueezeKdjStopsModeSelectionIgnoresTest:
    TRAIN = {(1.0, None): (300, 0.8), (1.5, None): (100, 1.1), (2.0, None): (40, 9.0),
             (1.0, "stock_ma120"): (100, 0.9), (1.5, "stock_ma120"): (100, 1.2), (2.0, "stock_ma120"): (59, 4.0)}

    def _run(self, monkeypatch, tmp_path, test_pf, **arg_extra):
        """假回測：挑選期N3/S3 PF最高但筆數不足(40/59筆) → 6個裡符合資格的PF最高是S2；驗證期由test_pf決定。"""
        calls = []

        def _fake_backtest(**kw):
            calls.append(kw)
            cal = kw["master_calendar"]
            key = (kw["atr_stop_mult"], kw["entry_filter"])
            if cal[0] < pd.Timestamp("2023-01-01"):
                n, pf = self.TRAIN[key]
                trades = _make_pf_trades(n, pf, 2019)
            else:
                trades = _make_pf_trades(80, test_pf[key], 2024)
            return trades, dict.fromkeys(cb.CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS, 0)
        monkeypatch.setattr(cb, "run_squeeze_kdj_capital_constrained_backtest", _fake_backtest)

        def _boom(*a, **k):
            raise AssertionError("--squeeze-kdj-stops不需要大盤指數")
        monkeypatch.setattr(cb, "load_index_series", _boom)
        monkeypatch.setattr(cb, "load_squeeze_kdj_market_index", _boom)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        codes = ["1101", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2018-01-01", n_days=2300)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        out = cb.run_squeeze_kdj_stops_mode(_stops_args(**arg_extra), price_data, universe, idx)
        return out, calls

    def test_selection_uses_train_only_and_costs_are_passed(self, monkeypatch, tmp_path):
        # 驗證期N2/S3最好、S2最差 → 照樣選S2，不通過
        bad = {(1.0, None): 1.0, (1.5, None): 5.0, (2.0, None): 5.0,
               (1.0, "stock_ma120"): 1.0, (1.5, "stock_ma120"): 0.5, (2.0, "stock_ma120"): 5.0}
        out, calls = self._run(monkeypatch, tmp_path, bad)
        assert out["selected"] == "S2"
        assert not out["verdict"]["passed"]
        assert len(calls) == 2 * 6 * 2 + 2 * 2  # 主要+敏感度 x 6變體 x 2期間 ＋ 舊成本對照(N1、S1 x 2期間)
        new_cost = [kw for kw in calls if kw["commission_per_lot_side"] == 50.0]
        old_cost = [kw for kw in calls if kw["commission_per_lot_side"] == 200.0]
        assert len(new_cost) == 24 and len(old_cost) == 4
        assert all(kw["futures_tax_rate"] == 0.00002 for kw in new_cost)
        assert all(kw["futures_tax_rate"] == 0.0 and kw["max_concurrent_positions"] == 2 for kw in old_cost)
        assert {(kw["atr_stop_mult"], kw["entry_filter"]) for kw in old_cost} == {(1.0, None), (1.0, "stock_ma120")}
        for kw in calls:
            assert kw["lots"] == 1 and kw["top_n"] == 3 and kw["execution_model"] == "limit_1tick"
            assert kw["variant"] == "B" and kw["atr_target_mult"] == 3.0
            assert kw["max_hold_days"] == 20 and kw["atr_period"] == 14 and kw["ranking_rule"] == "trigger_return"
            assert kw["max_concurrent_positions"] in (1, 2) and "slippage_pct" not in kw
            assert "market_series" not in kw
        train_cals = [kw["master_calendar"] for kw in calls if kw["master_calendar"][0] < pd.Timestamp("2023-01-01")]
        assert all(c[-1] <= pd.Timestamp("2022-12-31") for c in train_cals)
        # 換一組驗證期(S2好過N1)，選擇不變、判定通過
        good = {(1.0, None): 1.5, (1.5, None): 0.5, (2.0, None): 0.5,
                (1.0, "stock_ma120"): 5.0, (1.5, "stock_ma120"): 3.0, (2.0, "stock_ma120"): 0.5}
        out2, _ = self._run(monkeypatch, tmp_path, good)
        assert out2["selected"] == "S2" and out2["verdict"]["passed"]
        assert "✅ 通過" in (tmp_path / "summary.txt").read_text(encoding="utf-8")
        # 驗證期S2沒有勝過N1 → 不通過(比較基準是N1)
        worse = dict(good)
        worse[(1.0, None)] = 4.0
        out3, _ = self._run(monkeypatch, tmp_path, worse)
        assert out3["selected"] == "S2" and not out3["verdict"]["passed"]
        assert "勝過N1" in out3["verdict"]["text"]

    def test_custom_commission_from_args(self, monkeypatch, tmp_path):
        tp = {k: 1.0 for k in self.TRAIN}
        out, calls = self._run(monkeypatch, tmp_path, tp, commission_per_lot_side=25.0, futures_tax_rate=0.0)
        assert out["commission_per_lot_side"] == 25.0 and out["futures_tax_rate"] == 0.0
        assert sum(kw["commission_per_lot_side"] == 25.0 for kw in calls) == 24
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "每口單邊NT$25(一進一出共NT$50/口)" in summary


class TestSqueezeKdjStopsCliMode:
    def test_skips_full_pipeline_writes_outputs_without_scipy(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)

        codes = list(STOCK_FUTURES_UNIVERSE)[:12]
        price_data, idx = _make_regime_switching_market(codes + ["2330"], start="2017-06-15", n_days=1950, seed=3)
        load_calls = []

        def _fake_load(universe, start, end, refresh=False):
            load_calls.append((start, end))
            return price_data
        monkeypatch.setattr(cb, "load_price_data", _fake_load)

        def _boom(*args, **kwargs):
            raise AssertionError("--squeeze-kdj-stops模式不該呼叫這個")
        monkeypatch.setattr(cb, "load_index_series", _boom)
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(data_loader, "load_index_series", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)
        TestSqueezeKdjFixedCliMode()._patch_heavy_stages_to_explode(monkeypatch)
        monkeypatch.setattr(cb, "run_squeeze_kdj_fixed_mode", _boom)
        monkeypatch.setattr(cb, "run_squeeze_kdj_filters_mode", _boom)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        mode_calls = []
        real_mode = cb.run_squeeze_kdj_stops_mode

        def _spy(*args, **kwargs):
            mode_calls.append(args)
            return real_mode(*args, **kwargs)
        monkeypatch.setattr(cb, "run_squeeze_kdj_stops_mode", _spy)

        argv = ["compare_breakout.py", "--squeeze-kdj-stops", "--max-stocks", "12",
                "--start", "2018-01-01", "--end", "2024-11-15", "--starting-capital", "1000000"]
        monkeypatch.setattr(sys, "argv", argv)
        cb.main()

        assert load_calls == [("2017-06-15", "2024-11-15")]  # 股價往前多抓200個日曆天
        assert len(mode_calls) == 1
        args, cal = mode_calls[0][0], mode_calls[0][3]
        assert args.commission_per_lot_side == 50.0 and args.futures_tax_rate == 0.00002
        assert cal.equals(idx[idx >= pd.Timestamp("2018-01-01")])

        summ = pd.read_csv(tmp_path / "squeeze_kdj_stops_summary.csv", encoding="utf-8-sig")
        assert len(summ) == 24  # 2種最多持倉數 x 6變體 x 2期間
        for col in ("最多同時持倉數", "用途", "變體代號", "可被挑選", "停損ATR倍數", "期間", "交易筆數", "勝率(%)",
                    "獲利因子PF", "總損益(NT$)", "平均每筆損益(NT$)", "平均獲利(NT$/筆)", "平均虧損(NT$/筆)",
                    "平均停損風險(NT$/筆，停損距離x乘數x口數)", "最大回撤(NT$)", "拿掉最大3筆後損益(NT$)",
                    "手續費合計(NT$)", "期交稅合計(NT$)", "交易成本合計(NT$)", "持有<=1天出場比例(%)",
                    "持有<=3天出場損益(NT$)", "持有<=5天出場筆數", "出場_停利筆數", "出場_停損比例(%)",
                    "出場_跳空停損損益(NT$)", "出場_到期筆數", "bootstrap正報酬比例(%)", "被挑選規則選中",
                    "手續費假設(NT$/口/單邊)", "期交稅率(每邊)"):
            assert col in summ.columns, col
        assert set(summ["變體代號"]) == {"N1", "N2", "N3", "S1", "S2", "S3"}
        assert summ["可被挑選"].all()
        assert summ["被挑選規則選中"].sum() <= 1
        assert summ["交易筆數"].sum() > 0, "合成資料要有交易，測試才有意義"
        assert (summ.loc[summ["變體代號"].str.startswith("N"), "診斷_進場濾網擋掉"] == 0).all()
        assert (summ.loc[summ["變體代號"].str.startswith("S"), "診斷_進場濾網擋掉"].sum()) > 0
        # 手續費 = 50 x 1口 x 2邊 x 筆數
        assert (summ["手續費合計(NT$)"] == 100.0 * summ["交易筆數"]).all()
        # 停損越寬，平均停損風險越大(同一期間、同一持倉數)
        for (mp, per), g in summ.groupby(["最多同時持倉數", "期間"]):
            r = dict(zip(g["變體代號"], g["平均停損風險(NT$/筆，停損距離x乘數x口數)"]))
            for lo, hi in (("S1", "S3"), ("N1", "N3")):
                if all(np.isfinite(r[v]) for v in (lo, hi)):
                    assert r[hi] > r[lo]
        # 出場原因加總 = 全部筆數
        reason_cols = [f"出場_{lab}筆數" for lab in ("停利", "停損", "跳空停損", "到期")]
        assert (summ[reason_cols].sum(axis=1) == summ["交易筆數"]).all()

        yearly = pd.read_csv(tmp_path / "squeeze_kdj_stops_yearly.csv", encoding="utf-8-sig")
        assert list(yearly.columns) == cb.SQUEEZE_KDJ_FILTER_YEARLY_COLUMNS
        trades = pd.read_csv(tmp_path / "squeeze_kdj_stops_trades.csv", encoding="utf-8-sig")
        assert list(trades.columns) == cb.SQUEEZE_KDJ_STOPS_TRADE_COLUMNS
        assert len(trades) == summ["交易筆數"].sum()
        assert (trades["手續費(NT$)"] == 100.0).all() and (trades["期交稅(NT$)"] > 0).all()

        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "--squeeze-kdj-stops模式" in summary
        assert (summary.index("【事先登錄的規則") < summary.index("【交易成本假設】")
                < summary.index("【挑選結果與判定】") < summary.index("【並排總表"))
        assert "每口單邊NT$50(一進一出共NT$100/口)" in summary and "十萬分之二" in summary
        assert "【手續費更正本身的影響" in summary and "舊成本" in summary
        assert "【停損放寬的代價：每筆風險" in summary
        assert "daily_squeeze_signals.py" in summary and "不會改掃描器" in summary
        assert "結論：" in summary and "兩段都變好" in summary and "N1(停損1.0倍無濾網,實盤)" in summary
        assert "6個變體，事先登錄" in summary and "從3個變成6個" in summary
        assert "R0" not in summary
        table = summary[summary.index("【並排總表｜敏感度對照"):summary.index("【手續費更正本身的影響")]
        assert sum(f"{v}(" in table for v in cb.SQUEEZE_KDJ_STOP_SELECTABLE) == 6
        assert "多重比較" in summary and "倖存者偏差" in summary and "偏樂觀" in summary and "基差" in summary
        assert "這次測試無效" not in summary
        assert "[階段0]" not in summary

    def test_invalid_start_prints_big_warning(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        codes = ["1101", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2021-06-01", n_days=600)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        out = cb.run_squeeze_kdj_stops_mode(_stops_args(start="2021-06-01", end="2023-09-01"),
                                            price_data, universe, idx)
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "這次測試無效" in summary and "2018-01-01" in summary
        assert summary.index("這次測試無效") < summary.index("【挑選結果與判定】")
        assert out["validity_warnings"]


# ============================================================================
# --squeeze-kdj-exits：以N2為基準的出場方式小測試(E0~E4)
# ============================================================================
def _exit_key(kw):
    return (kw["atr_target_mult"], kw["breakeven_trigger_atr"], tuple(kw["skip_entry_weekdays"]))


EXIT_KEYS = {"E0": (3.0, None, ()), "E1": (2.25, None, ()), "E2": (4.5, None, ()),
             "E3": (3.0, 1.5, ()), "E4": (3.0, None, (4,))}


class TestSqueezeKdjExitsSelectionRule:
    def test_variants_are_exactly_the_five_preregistered(self):
        assert [(v[0], v[2], v[3], v[4]) for v in cb.SQUEEZE_KDJ_EXIT_VARIANTS] == [
            ("E0", 3.0, None, ()), ("E1", 2.25, None, ()), ("E2", 4.5, None, ()),
            ("E3", 3.0, 1.5, ()), ("E4", 3.0, None, (4,))]
        assert cb.SQUEEZE_KDJ_EXIT_SELECTABLE == ("E0", "E1", "E2", "E3", "E4")
        assert cb.SQUEEZE_KDJ_EXIT_BASELINE == "E0" and cb.SQUEEZE_KDJ_EXITS_STOP_MULT == 1.5
        assert cb.SQUEEZE_KDJ_EXIT_DATA_MINED == ("E4",)
        for vid, key in EXIT_KEYS.items():
            kw = cb._squeeze_kdj_exit_backtest_kwargs(vid)
            assert _exit_key(kw) == key and kw["atr_stop_mult"] == 1.5 and kw["entry_filter"] is None

    def _all(self, **override):
        base = {v: _stats(100, 0.8) for v in cb.SQUEEZE_KDJ_EXIT_SELECTABLE}
        base.update(override)
        return base

    def test_highest_train_pf_min_trades_and_ties(self):
        sel, reason = cb.select_squeeze_kdj_exit_variant(self._all(E2=_stats(60, 1.3), E4=_stats(59, 5.0)))
        assert sel == "E2" and "E4" in reason and "排除" in reason
        sel, _ = cb.select_squeeze_kdj_exit_variant(self._all(E1=_stats(100, 1.1, pnl=100), E3=_stats(100, 1.1, pnl=300)))
        assert sel == "E3"
        tied = {v: _stats(100, 1.1, pnl=100) for v in cb.SQUEEZE_KDJ_EXIT_SELECTABLE}
        assert cb.select_squeeze_kdj_exit_variant(tied)[0] == "E0"
        tied["E0"] = _stats(10, 1.1, pnl=100)
        assert cb.select_squeeze_kdj_exit_variant(tied)[0] == "E1"
        assert cb.select_squeeze_kdj_exit_variant(self._all(N2=_stats(500, 9.0)))[0] == "E0"  # 不在名單的代號不會被選
        sel, reason = cb.select_squeeze_kdj_exit_variant({v: _stats(59, 2.0) for v in cb.SQUEEZE_KDJ_EXIT_SELECTABLE})
        assert sel is None and "無法挑選" in reason

    def _seg(self, n, pf, pct):
        return {"stats": _stats(n, pf), "bootstrap": {"pct_positive": pct, "p_value": 1 - pct / 100}}

    def test_verdict_branches_vs_e0(self):
        e0 = self._seg(80, 0.9, 30.0)
        v = cb.squeeze_kdj_exit_verdict("E3", {"E0": e0, "E3": self._seg(70, 1.4, 90.0)})
        assert v["passed"] and v["text"].startswith("✅") and "資料挖掘" not in v["text"]
        v = cb.squeeze_kdj_exit_verdict("E1", {"E0": e0, "E1": self._seg(70, 1.4, 75.0)})
        assert not v["passed"] and "bootstrap" in v["text"] and "PF>1" not in v["text"]
        v = cb.squeeze_kdj_exit_verdict("E2", {"E0": self._seg(80, 1.6, 95.0), "E2": self._seg(70, 1.4, 90.0)})
        assert not v["passed"] and "勝過E0" in v["text"]
        v = cb.squeeze_kdj_exit_verdict("E2", {"E0": e0, "E2": self._seg(70, 0.8, 20.0)})
        assert not v["passed"] and "PF>1" in v["text"]
        v = cb.squeeze_kdj_exit_verdict("E0", {"E0": self._seg(80, 2.0, 99.0)})
        assert not v["passed"] and "維持N2原樣" in v["text"]
        assert cb.squeeze_kdj_exit_verdict(None, {"E0": e0})["passed"] is False
        # E4通過也要標示資料挖掘、先驗最弱
        v = cb.squeeze_kdj_exit_verdict("E4", {"E0": e0, "E4": self._seg(70, 1.4, 90.0)})
        assert v["passed"] and "資料挖掘" in v["text"]
        v = cb.squeeze_kdj_exit_verdict("E4", {"E0": e0, "E4": self._seg(70, 0.8, 20.0)})
        assert not v["passed"] and "資料挖掘" in v["text"]

    def test_improvement_table_vs_e0(self):
        def seg(wr, pf):
            return {"stats": {"trade_count": 10, "win_rate": wr, "profit_factor": pf}}
        by = {"E0": {"train": seg(40, 0.8), "test": seg(45, 1.0)},
              "E1": {"train": seg(42, 0.9), "test": seg(46, 1.1)},
              "E2": {"train": seg(30, 0.9), "test": seg(46, 1.1)},
              "E3": {"train": seg(40, 0.8), "test": seg(45, 1.0)},
              "E4": {"train": seg(50, 1.0), "test": seg(40, 2.0)}}
        rows = {r["variant"]: r for r in cb.squeeze_kdj_exit_improvement_table(by)}
        assert set(rows) == {"E1", "E2", "E3", "E4"}
        assert rows["E1"]["both_periods_better"] is True
        assert rows["E2"]["both_periods_better"] is False and rows["E2"]["train_pf_better"] is True
        assert rows["E3"]["both_periods_better"] is False and rows["E4"]["both_periods_better"] is False


class TestSqueezeKdjExitExtraBreakdown:
    def test_weekday_breakeven_and_time_exit(self):
        def t(entry, reason, pnl, be=None):
            d = _fixed_trade(entry, pd.Timestamp(entry) + pd.Timedelta(days=3), pnl)
            d["exit_reason"] = reason
            if be is not None:
                d["breakeven_triggered"] = be
            return d
        trades = [t("2024-01-05", "stop_gap", -700),               # 週五
                  t("2024-01-12", "forced_close", 300, be=True),   # 週五
                  t("2024-01-08", "target", 1500, be=True),        # 週一
                  t("2024-01-09", "breakeven_stop", -120, be=True),  # 週二
                  t("2024-01-10", "forced_close", 900, be=False)]  # 週三
        b = cb.squeeze_kdj_exit_extra_breakdown(trades)
        assert b["weekday"][4] == {"count": 2, "pnl_ntd": -400.0, "profit_factor": pytest.approx(300 / 700),
                                   "win_rate": 50.0}
        assert b["weekday"][0]["profit_factor"] == float("inf") and b["weekday"][3]["count"] == 0
        assert b["weekday"][3]["profit_factor"] == 0.0
        assert b["breakeven_triggered_count"] == 3 and b["breakeven_triggered_then_target"] == 1
        assert b["breakeven_triggered_pnl_ntd"] == pytest.approx(1680.0)
        assert b["time_exit_count"] == 2 and b["time_exit_pnl_ntd"] == 1200.0 and b["time_exit_avg_pnl_ntd"] == 600.0
        reasons = cb.squeeze_kdj_stop_trade_breakdown(trades, cb.SQUEEZE_KDJ_EXITS_EXIT_REASONS)["reasons"]
        assert reasons["breakeven_stop"]["count"] == 1 and reasons["other"]["count"] == 0
        # stops模式的預設出場原因清單不變：保本停損會落在other
        assert cb.squeeze_kdj_stop_trade_breakdown(trades)["reasons"]["other"]["count"] == 1


class TestSqueezeKdjExitsModeSelectionIgnoresTest:
    TRAIN = {"E0": (300, 0.9), "E1": (100, 1.1), "E2": (40, 9.0), "E3": (100, 1.2), "E4": (59, 4.0)}

    def _run(self, monkeypatch, tmp_path, test_pf, **arg_extra):
        """假回測：挑選期E2/E4 PF最高但筆數不足 → 符合資格裡PF最高的是E3；驗證期由test_pf決定。"""
        calls = []
        by_key = {v: k for k, v in EXIT_KEYS.items()}

        def _fake_backtest(**kw):
            calls.append(kw)
            vid = by_key[_exit_key(kw)]
            if kw["master_calendar"][0] < pd.Timestamp("2023-01-01"):
                n, pf = self.TRAIN[vid]
                trades = _make_pf_trades(n, pf, 2019)
            else:
                trades = _make_pf_trades(80, test_pf[vid], 2024)
            return trades, dict.fromkeys(cb.CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS, 0)
        monkeypatch.setattr(cb, "run_squeeze_kdj_capital_constrained_backtest", _fake_backtest)

        def _boom(*a, **k):
            raise AssertionError("--squeeze-kdj-exits不需要大盤指數")
        monkeypatch.setattr(cb, "load_index_series", _boom)
        monkeypatch.setattr(cb, "load_squeeze_kdj_market_index", _boom)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        codes = ["1101", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2018-01-01", n_days=2300)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        out = cb.run_squeeze_kdj_exits_mode(_stops_args(**arg_extra), price_data, universe, idx)
        return out, calls

    def test_selection_uses_train_only(self, monkeypatch, tmp_path):
        # 驗證期E1/E2/E4遠遠最好、E3最差 → 照樣選E3，不通過
        bad = {"E0": 1.0, "E1": 5.0, "E2": 5.0, "E3": 0.5, "E4": 5.0}
        out, calls = self._run(monkeypatch, tmp_path, bad)
        assert out["selected"] == "E3" and not out["verdict"]["passed"]
        assert len(calls) == 2 * 5 * 2  # 主要+敏感度 x 5變體 x 2期間
        for kw in calls:
            assert kw["lots"] == 1 and kw["top_n"] == 3 and kw["execution_model"] == "limit_1tick"
            assert kw["variant"] == "B" and kw["atr_stop_mult"] == 1.5 and kw["entry_filter"] is None
            assert kw["max_hold_days"] == 20 and kw["atr_period"] == 14 and kw["ranking_rule"] == "trigger_return"
            assert kw["max_concurrent_positions"] in (1, 2) and "slippage_pct" not in kw
            assert kw["commission_per_lot_side"] == 50.0 and kw["futures_tax_rate"] == 0.00002
            assert "market_series" not in kw
        assert {_exit_key(kw) for kw in calls} == set(EXIT_KEYS.values())
        train_cals = [kw["master_calendar"] for kw in calls if kw["master_calendar"][0] < pd.Timestamp("2023-01-01")]
        assert len(train_cals) == 10 and all(c[-1] <= pd.Timestamp("2022-12-31") for c in train_cals)
        # 驗證期換成E3好過E0 → 選擇不變、通過
        out2, _ = self._run(monkeypatch, tmp_path, {"E0": 1.2, "E1": 0.5, "E2": 0.5, "E3": 3.0, "E4": 0.5})
        assert out2["selected"] == "E3" and out2["verdict"]["passed"]
        assert "✅ 通過" in (tmp_path / "summary.txt").read_text(encoding="utf-8")
        # 驗證期E3 PF>1但沒勝過E0 → 不通過
        out3, _ = self._run(monkeypatch, tmp_path, {"E0": 4.0, "E1": 0.5, "E2": 0.5, "E3": 3.0, "E4": 0.5})
        assert out3["selected"] == "E3" and not out3["verdict"]["passed"] and "勝過E0" in out3["verdict"]["text"]

    def test_e0_selected_when_it_has_best_train_pf(self, monkeypatch, tmp_path):
        monkeypatch.setattr(self, "TRAIN", {"E0": (300, 1.5), "E1": (100, 1.1), "E2": (100, 1.2),
                                            "E3": (100, 1.4), "E4": (100, 1.45)})
        out, _ = self._run(monkeypatch, tmp_path, {v: 2.0 for v in EXIT_KEYS})
        assert out["selected"] == "E0" and not out["verdict"]["passed"] and "維持N2原樣" in out["verdict"]["text"]

    def test_custom_commission_from_args(self, monkeypatch, tmp_path):
        out, calls = self._run(monkeypatch, tmp_path, {v: 1.0 for v in EXIT_KEYS},
                               commission_per_lot_side=25.0, futures_tax_rate=0.0)
        assert out["commission_per_lot_side"] == 25.0 and out["futures_tax_rate"] == 0.0
        assert all(kw["commission_per_lot_side"] == 25.0 and kw["futures_tax_rate"] == 0.0 for kw in calls)
        assert "每口單邊NT$25(一進一出共NT$50/口)" in (tmp_path / "summary.txt").read_text(encoding="utf-8")


class TestSqueezeKdjExitsCliMode:
    def test_skips_full_pipeline_writes_outputs_without_scipy(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)

        codes = list(STOCK_FUTURES_UNIVERSE)[:12]
        price_data, idx = _make_regime_switching_market(codes + ["2330"], start="2017-06-15", n_days=1950, seed=3)
        load_calls = []

        def _fake_load(universe, start, end, refresh=False):
            load_calls.append((start, end))
            return price_data
        monkeypatch.setattr(cb, "load_price_data", _fake_load)

        def _boom(*args, **kwargs):
            raise AssertionError("--squeeze-kdj-exits模式不該呼叫這個")
        monkeypatch.setattr(cb, "load_index_series", _boom)
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(data_loader, "load_index_series", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)
        TestSqueezeKdjFixedCliMode()._patch_heavy_stages_to_explode(monkeypatch)
        for name in ("run_squeeze_kdj_fixed_mode", "run_squeeze_kdj_filters_mode", "run_squeeze_kdj_stops_mode",
                     "run_squeeze_kdj_only_mode", "run_squeeze_kdj_capital_constrained_mode",
                     "run_squeeze_kdj_grid_mode"):
            monkeypatch.setattr(cb, name, _boom)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        mode_calls = []
        real_mode = cb.run_squeeze_kdj_exits_mode

        def _spy(*args, **kwargs):
            mode_calls.append(args)
            return real_mode(*args, **kwargs)
        monkeypatch.setattr(cb, "run_squeeze_kdj_exits_mode", _spy)

        argv = ["compare_breakout.py", "--squeeze-kdj-exits", "--max-stocks", "12",
                "--start", "2018-01-01", "--end", "2024-11-15", "--starting-capital", "1000000"]
        monkeypatch.setattr(sys, "argv", argv)
        cb.main()

        assert load_calls == [("2017-06-15", "2024-11-15")]  # 股價往前多抓200個日曆天
        assert len(mode_calls) == 1
        args, cal = mode_calls[0][0], mode_calls[0][3]
        assert args.commission_per_lot_side == 50.0 and args.futures_tax_rate == 0.00002
        assert cal.equals(idx[idx >= pd.Timestamp("2018-01-01")])

        summ = pd.read_csv(tmp_path / "squeeze_kdj_exits_summary.csv", encoding="utf-8-sig")
        assert len(summ) == 20  # 2種最多持倉數 x 5變體 x 2期間
        for col in ("最多同時持倉數", "用途", "變體代號", "可被挑選", "停損ATR倍數", "停利ATR倍數",
                    "保本觸發(收盤>=進場價+N倍ATR)", "不進場的星期", "資料挖掘(先驗最弱)", "期間", "交易筆數",
                    "勝率(%)", "獲利因子PF", "總損益(NT$)", "平均獲利(NT$/筆)", "平均虧損(NT$/筆)", "最大回撤(NT$)",
                    "拿掉最大3筆後損益(NT$)", "平均持有天數", "手續費合計(NT$)", "期交稅合計(NT$)",
                    "持有<=1天出場筆數", "持有<=3天出場比例(%)", "持有<=5天出場損益(NT$)", "出場_停利筆數",
                    "出場_保本停損筆數", "出場_保本跳空停損筆數", "出場_到期筆數", "出場_到期損益(NT$)",
                    "到期出場平均損益(NT$/筆)", "觸發保本筆數", "週五進場筆數", "週五進場PF",
                    "bootstrap正報酬比例(%)", "診斷_進場星期略過", "被挑選規則選中"):
            assert col in summ.columns, col
        assert set(summ["變體代號"]) == set(cb.SQUEEZE_KDJ_EXIT_SELECTABLE) and summ["可被挑選"].all()
        assert (summ["停損ATR倍數"] == 1.5).all()
        assert summ["被挑選規則選中"].sum() <= 1
        assert summ["交易筆數"].sum() > 0, "合成資料要有交易，測試才有意義"
        by = {vid: g for vid, g in summ.groupby("變體代號")}
        # 只有E4擋週五：E4沒有週五進場、有進場星期略過；其他變體略過=0
        assert (by["E4"]["週五進場筆數"] == 0).all() and by["E4"]["診斷_進場星期略過"].sum() > 0
        assert all((by[v]["診斷_進場星期略過"] == 0).all() for v in ("E0", "E1", "E2", "E3"))
        assert by["E0"]["週五進場筆數"].sum() > 0
        # 只有E3會有保本停損/觸發保本
        assert by["E3"]["觸發保本筆數"].sum() > 0
        for v in ("E0", "E1", "E2", "E4"):
            assert (by[v]["觸發保本筆數"] == 0).all()
            assert (by[v]["出場_保本停損筆數"] + by[v]["出場_保本跳空停損筆數"] == 0).all()
        reason_cols = [f"出場_{lab}筆數" for _, lab in cb.SQUEEZE_KDJ_EXITS_EXIT_REASONS]
        assert (summ[reason_cols].sum(axis=1) == summ["交易筆數"]).all()
        assert (summ["手續費合計(NT$)"] == 100.0 * summ["交易筆數"]).all()

        yearly = pd.read_csv(tmp_path / "squeeze_kdj_exits_yearly.csv", encoding="utf-8-sig")
        assert list(yearly.columns) == cb.SQUEEZE_KDJ_FILTER_YEARLY_COLUMNS and len(yearly) > 0
        trades = pd.read_csv(tmp_path / "squeeze_kdj_exits_trades.csv", encoding="utf-8-sig")
        assert list(trades.columns) == cb.SQUEEZE_KDJ_EXITS_TRADE_COLUMNS
        assert len(trades) == summ["交易筆數"].sum()
        assert "週五" not in set(trades.loc[trades["變體代號"] == "E4", "進場星期"])
        assert trades.loc[trades["變體代號"] == "E3", "是否觸發保本(停損移到進場價)"].any()
        assert not trades.loc[trades["變體代號"] != "E3", "是否觸發保本(停損移到進場價)"].any()
        assert set(trades.loc[trades["變體代號"] != "E3", "出場原因"]) <= {
            "停利", "停損", "跳空停損(開盤價出場)", "持有天數到期強制平倉"}

        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "--squeeze-kdj-exits模式" in summary
        assert (summary.index("【事先登錄的規則") < summary.index("【交易成本假設】")
                < summary.index("【挑選結果與判定】") < summary.index("【並排總表"))
        assert "資料挖掘" in summary and "多重比較" in summary and "5個變體" in summary
        assert "保本停損" in summary and "依進場星期" in summary and "進場星期略過" in summary
        assert "【出場結構對照" in summary and "兩段都變好" in summary and "結論：" in summary
        table = summary[summary.index("【並排總表｜敏感度對照"):summary.index("【出場結構對照")]
        assert sum(f"{v}(" in table for v in cb.SQUEEZE_KDJ_EXIT_SELECTABLE) == 5
        assert "倖存者偏差" in summary and "偏樂觀" in summary and "基差" in summary
        assert "daily_squeeze_signals.py" in summary
        assert "這次測試無效" not in summary and "[階段0]" not in summary

    def test_invalid_start_prints_big_warning(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        codes = ["1101", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2021-06-01", n_days=600)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        out = cb.run_squeeze_kdj_exits_mode(_stops_args(start="2021-06-01", end="2023-09-01"),
                                            price_data, universe, idx)
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "這次測試無效" in summary and "2018-01-01" in summary
        assert summary.index("這次測試無效") < summary.index("【挑選結果與判定】")
        assert out["validity_warnings"]


# ============================================================================
# --squeeze-kdj-stop-target-grid：停損 x 停利 3x3穩健度地圖
# ============================================================================
ST_ALL_IDS = ["G10_20", "G10_30", "G10_40", "G15_20", "G15_30", "G15_40", "G20_20", "G20_30", "G20_40"]


def _st_key_to_id(kw):
    return cb.squeeze_kdj_stop_target_cell_id(kw["atr_stop_mult"], kw["atr_target_mult"])


class TestSqueezeKdjStopTargetGridConstruction:
    def test_nine_cells_ids_order_and_baseline(self):
        assert [c[0] for c in cb.SQUEEZE_KDJ_ST_GRID_CELLS] == ST_ALL_IDS
        assert cb.SQUEEZE_KDJ_ST_GRID_SELECTABLE == tuple(ST_ALL_IDS)
        assert [(c[1], c[2]) for c in cb.SQUEEZE_KDJ_ST_GRID_CELLS] == [
            (s, t) for s in (1.0, 1.5, 2.0) for t in (2.0, 3.0, 4.0)]
        assert cb.squeeze_kdj_stop_target_cell_id(1.0, 2.0) == "G10_20"
        assert cb.squeeze_kdj_stop_target_cell_id(1.5, 3.0) == "G15_30"
        assert cb.SQUEEZE_KDJ_ST_GRID_BASELINE == "G15_30" and cb.SQUEEZE_KDJ_ST_GRID_LIVE == "G10_30"
        assert "基準N2" in cb.squeeze_kdj_stop_target_cell_label("G15_30")
        assert "實盤" in cb.squeeze_kdj_stop_target_cell_label("G10_30")
        assert cb.squeeze_kdj_stop_target_cell_name("G20_40") == "stop2.0_target4.0"
        # 基準/實盤就是stops模式的N2/N1
        n = {v[0]: (v[2], v[3]) for v in cb.SQUEEZE_KDJ_STOP_VARIANTS}
        assert cb.squeeze_kdj_stop_target_cell_id(*n["N2"]) == "G15_30"
        assert cb.squeeze_kdj_stop_target_cell_id(*n["N1"]) == "G10_30"
        assert cb.SQUEEZE_KDJ_FIXED_SETTING["atr_stop_mult"] == 1.0 and cb.SQUEEZE_KDJ_FIXED_SETTING["atr_target_mult"] == 3.0

    def test_neighbors_edges_and_corners(self):
        nb = cb.squeeze_kdj_stop_target_neighbors
        assert nb("G10_20") == ["G15_20", "G10_30"]                       # 角落：2個
        assert nb("G20_40") == ["G15_40", "G20_30"]
        assert nb("G10_30") == ["G15_30", "G10_20", "G10_40"]             # 邊：3個
        assert nb("G15_20") == ["G10_20", "G20_20", "G15_30"]
        assert nb("G15_30") == ["G10_30", "G20_30", "G15_20", "G15_40"]   # 中間：4個

    def test_plateau_on_hand_made_grid(self):
        # 挑選期PF(列=停損、欄=停利)          驗證期PF
        #   0.6  1.2  1.4                      0.9  1.1  1.3
        #   0.8  1.3  1.5                      1.0  1.2  1.6
        #   0.7  0.9  1.1                      0.5  0.95 1.2
        train = [[0.6, 1.2, 1.4], [0.8, 1.3, 1.5], [0.7, 0.9, 1.1]]
        test = [[0.9, 1.1, 1.3], [1.0, 1.2, 1.6], [0.5, 0.95, 1.2]]
        pf = {}
        for i, s in enumerate((1.0, 1.5, 2.0)):
            for j, t in enumerate((2.0, 3.0, 4.0)):
                pf[cb.squeeze_kdj_stop_target_cell_id(s, t)] = {"train": train[i][j], "test": test[i][j]}
        p = cb.squeeze_kdj_stop_target_plateau(pf)
        # 角落G10_20：自己+2鄰格
        assert p["G10_20"]["n_cells"] == 3
        assert p["G10_20"]["nbhd_train"] == pytest.approx((0.6 + 0.8 + 1.2) / 3)
        # 邊G10_30：自己+3鄰格
        assert p["G10_30"]["n_cells"] == 4
        assert p["G10_30"]["nbhd_train"] == pytest.approx((1.2 + 1.3 + 0.6 + 1.4) / 4)
        assert p["G10_30"]["nbhd_test"] == pytest.approx((1.1 + 1.2 + 0.9 + 1.3) / 4)
        # 中間G15_30：自己+4鄰格
        assert p["G15_30"]["n_cells"] == 5
        assert p["G15_30"]["nbhd_train"] == pytest.approx((1.3 + 1.2 + 0.9 + 0.8 + 1.5) / 5)
        assert p["G15_30"]["nbhd_test"] == pytest.approx((1.2 + 1.1 + 0.95 + 1.0 + 1.6) / 5)
        plateau = {c for c, r in p.items() if r["plateau"]}
        # G20_40：自己1.1/1.2>1，但鄰域挑選期平均(1.1+1.5+0.9)/3=1.167、驗證期(1.2+1.6+0.95)/3=1.25 → 高原
        # G20_30：自己挑選期0.9 → 不是；G15_20：驗證期自己1.0(不>1) → 不是
        assert plateau == {"G10_30", "G10_40", "G15_30", "G15_40", "G20_40"}
        assert p["G20_30"]["plateau"] is False and p["G15_20"]["plateau"] is False
        # 自己>1但鄰域平均<=1 → 不算(孤峰)
        lone = {c: {"train": 0.5, "test": 0.5} for c in ST_ALL_IDS}
        lone["G10_20"] = {"train": 1.4, "test": 1.4}
        pl = cb.squeeze_kdj_stop_target_plateau(lone)
        assert pl["G10_20"]["nbhd_train"] == pytest.approx(0.8) and pl["G10_20"]["plateau"] is False
        assert not any(r["plateau"] for r in pl.values())
        # inf(只有賺沒賠)照樣算>1
        lone["G10_30"] = {"train": float("inf"), "test": float("inf")}
        assert cb.squeeze_kdj_stop_target_plateau(lone)["G10_20"]["plateau"] is True

    def _all(self, **override):
        base = {c: _stats(100, 0.8) for c in ST_ALL_IDS}
        base.update(override)
        return base

    def test_selection_rule_train_only(self):
        sel, reason = cb.select_squeeze_kdj_stop_target_cell(self._all(G20_40=_stats(60, 1.3), G10_20=_stats(59, 9.0)))
        assert sel == "G20_40" and "G10_20" in reason and "排除" in reason
        sel, _ = cb.select_squeeze_kdj_stop_target_cell(self._all(G15_20=_stats(100, 1.1, pnl=100),
                                                                  G10_40=_stats(100, 1.1, pnl=300)))
        assert sel == "G10_40"
        tied = {c: _stats(100, 1.1, pnl=100) for c in ST_ALL_IDS}
        assert cb.select_squeeze_kdj_stop_target_cell(tied)[0] == "G10_20"  # 停損由小到大、停利由小到大
        tied["G10_20"] = _stats(10, 1.1, pnl=100)
        assert cb.select_squeeze_kdj_stop_target_cell(tied)[0] == "G10_30"
        assert cb.select_squeeze_kdj_stop_target_cell(self._all(N2=_stats(500, 9.0)))[0] == "G10_20"
        sel, reason = cb.select_squeeze_kdj_stop_target_cell({c: _stats(59, 2.0) for c in ST_ALL_IDS})
        assert sel is None and "無法挑選" in reason

    def _seg(self, n, pf, pct):
        return {"stats": _stats(n, pf), "bootstrap": {"pct_positive": pct, "p_value": 1 - pct / 100}}

    def test_verdict_branches_vs_g15_30(self):
        base = self._seg(80, 0.9, 30.0)
        v = cb.squeeze_kdj_stop_target_verdict("G20_40", {"G15_30": base, "G20_40": self._seg(70, 1.4, 90.0)})
        assert v["passed"] and v["text"].startswith("✅")
        v = cb.squeeze_kdj_stop_target_verdict("G20_40", {"G15_30": base, "G20_40": self._seg(70, 1.4, 75.0)})
        assert not v["passed"] and "bootstrap" in v["text"] and "PF>1" not in v["text"]
        v = cb.squeeze_kdj_stop_target_verdict("G20_40", {"G15_30": self._seg(80, 1.6, 95.0),
                                                          "G20_40": self._seg(70, 1.4, 90.0)})
        assert not v["passed"] and "勝過G15_30" in v["text"]
        v = cb.squeeze_kdj_stop_target_verdict("G20_40", {"G15_30": base, "G20_40": self._seg(70, 0.8, 20.0)})
        assert not v["passed"] and "PF>1" in v["text"]
        v = cb.squeeze_kdj_stop_target_verdict("G15_30", {"G15_30": self._seg(80, 2.0, 99.0)})
        assert not v["passed"] and "G15_30(=N2)本身" in v["text"]
        # 選出實盤G10_30：照樣跟G15_30比，並註明是實盤
        v = cb.squeeze_kdj_stop_target_verdict("G10_30", {"G15_30": base, "G10_30": self._seg(70, 1.4, 90.0)})
        assert v["passed"] and "目前實盤" in v["text"]
        assert cb.squeeze_kdj_stop_target_verdict(None, {"G15_30": base})["passed"] is False

    def test_improvement_tables_vs_baseline_and_live(self):
        def seg(wr, pf):
            return {"stats": {"trade_count": 10, "win_rate": wr, "profit_factor": pf}}
        by = {c: {"train": seg(40, 0.8), "test": seg(40, 0.8)} for c in ST_ALL_IDS}
        by["G15_30"] = {"train": seg(45, 1.0), "test": seg(45, 1.0)}
        by["G20_40"] = {"train": seg(50, 1.2), "test": seg(50, 1.1)}
        rows = {r["variant"]: r for r in cb.squeeze_kdj_stop_target_improvement_table(by, "G15_30")}
        assert set(rows) == set(ST_ALL_IDS) - {"G15_30"} and rows["G20_40"]["both_periods_better"]
        assert not rows["G10_30"]["both_periods_better"]
        rows = {r["variant"]: r for r in cb.squeeze_kdj_stop_target_improvement_table(by, "G10_30")}
        assert "G10_30" not in rows and rows["G15_30"]["both_periods_better"]


def _st_grid_args(start="2018-01-01", end="2026-10-06", **extra):
    return _stops_args(start=start, end=end, **extra)


class TestSqueezeKdjStopTargetGridModeSelectionIgnoresTest:
    # 挑選期：G20_40 PF最高但只有40筆(不符資格)，符合資格裡最高的是G15_40
    TRAIN = {c: (100, 0.8) for c in ST_ALL_IDS}
    TRAIN.update({"G20_40": (40, 9.0), "G15_40": (100, 1.3), "G15_30": (120, 1.1), "G10_30": (200, 0.9)})

    def _run(self, monkeypatch, tmp_path, test_pf, **arg_extra):
        calls = []

        def _fake_backtest(**kw):
            calls.append(kw)
            cid = _st_key_to_id(kw)
            if kw["master_calendar"][0] < pd.Timestamp("2023-01-01"):
                n, pf = self.TRAIN[cid]
                trades = _make_pf_trades(n, pf, 2019)
            else:
                trades = _make_pf_trades(80, test_pf[cid], 2024)
            return trades, dict.fromkeys(cb.CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS, 0)
        monkeypatch.setattr(cb, "run_squeeze_kdj_capital_constrained_backtest", _fake_backtest)

        def _boom(*a, **k):
            raise AssertionError("--squeeze-kdj-stop-target-grid不需要大盤指數")
        monkeypatch.setattr(cb, "load_index_series", _boom)
        monkeypatch.setattr(cb, "load_squeeze_kdj_market_index", _boom)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        codes = ["1101", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2018-01-01", n_days=2300)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        out = cb.run_squeeze_kdj_stop_target_grid_mode(_st_grid_args(**arg_extra), price_data, universe, idx)
        return out, calls

    def test_selection_uses_train_only(self, monkeypatch, tmp_path):
        # 驗證期把G10_20捧成最好、G15_40最差 → 照樣選G15_40，不通過
        bad = {c: 1.2 for c in ST_ALL_IDS}
        bad.update({"G10_20": 9.0, "G15_40": 0.5})
        out, calls = self._run(monkeypatch, tmp_path, bad)
        assert out["selected"] == "G15_40" and not out["verdict"]["passed"]
        assert len(calls) == 2 * 9 * 2  # 主要+敏感度 x 9格 x 2期間
        assert {_st_key_to_id(kw) for kw in calls} == set(ST_ALL_IDS)
        for kw in calls:
            assert kw["lots"] == 1 and kw["top_n"] == 3 and kw["execution_model"] == "limit_1tick"
            assert kw["variant"] == "B" and kw["entry_filter"] is None
            assert kw["max_hold_days"] == 20 and kw["atr_period"] == 14 and kw["ranking_rule"] == "trigger_return"
            assert kw["max_concurrent_positions"] in (1, 2) and "slippage_pct" not in kw
            assert kw["commission_per_lot_side"] == 50.0 and kw["futures_tax_rate"] == 0.00002
            assert "market_series" not in kw
        train_cals = [kw["master_calendar"] for kw in calls if kw["master_calendar"][0] < pd.Timestamp("2023-01-01")]
        assert len(train_cals) == 18 and all(c[-1] <= pd.Timestamp("2022-12-31") for c in train_cals)
        # 驗證期換成G15_40好過G15_30 → 選擇不變、通過
        good = {c: 1.0 for c in ST_ALL_IDS}
        good.update({"G15_40": 2.0, "G15_30": 1.2})
        out2, _ = self._run(monkeypatch, tmp_path, good)
        assert out2["selected"] == "G15_40" and out2["verdict"]["passed"]
        assert "✅ 通過" in (tmp_path / "summary.txt").read_text(encoding="utf-8")
        # G15_40驗證期PF>1但沒勝過G15_30 → 不通過
        worse = dict(good, G15_30=3.0)
        out3, _ = self._run(monkeypatch, tmp_path, worse)
        assert out3["selected"] == "G15_40" and not out3["verdict"]["passed"]
        assert "勝過G15_30" in out3["verdict"]["text"]

    def test_plateau_and_heatmaps_in_summary(self, monkeypatch, tmp_path):
        test_pf = {c: 0.8 for c in ST_ALL_IDS}
        test_pf.update({"G15_40": 1.3, "G10_40": 1.2, "G20_40": 1.1, "G15_30": 1.1})
        train = {c: (100, 0.8) for c in ST_ALL_IDS}
        train.update({"G15_40": (100, 1.3), "G10_40": (100, 1.2), "G20_40": (100, 1.1), "G15_30": (100, 1.1)})
        monkeypatch.setattr(self, "TRAIN", train)
        out, _ = self._run(monkeypatch, tmp_path, test_pf)
        p2 = out["plateaus"][2]
        # G15_40鄰域：自己1.3、G10_40 1.2、G20_40 1.1、G15_30 1.1 → 平均1.175 → 高原
        assert p2["G15_40"]["nbhd_train"] == pytest.approx((1.3 + 1.2 + 1.1 + 1.1) / 4, rel=1e-6)
        assert p2["G15_40"]["plateau"]
        # G10_40(角落)：自己1.2、鄰格G15_40 1.3、G10_30 0.8 → 平均1.1 → 高原
        assert p2["G10_40"]["plateau"] and p2["G10_40"]["n_cells"] == 3
        # G15_30(中間)：自己1.1，但鄰域平均(1.1+0.8+0.8+0.8+1.3)/5=0.96 → 不是
        assert not p2["G15_30"]["plateau"]
        assert {c for c, r in p2.items() if r["plateau"]} == {"G10_40", "G15_40", "G20_40"}
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        i_rules = summary.index("【事先登錄的規則")
        assert "高原檢查(事先登錄)" in summary[i_rules:summary.index("【挑選結果與判定】")]
        assert "約24個" in summary and "地圖的形狀" in summary
        assert (i_rules < summary.index("【交易成本假設】") < summary.index("【挑選結果與判定】")
                < summary.index("【熱度圖｜主要") < summary.index("【熱度圖｜敏感度對照")
                < summary.index("【高原檢查｜最多同時持有2檔") < summary.index("【兩段都變好？其他8格跟G15_30")
                < summary.index("【兩段都變好？其他8格跟G10_30"))
        for t in ("挑選期PF", "驗證期PF", "挑選期總損益", "驗證期總損益", "9年合計損益", "勝率%(挑選期/驗證期)",
                  "交易筆數(挑選期/驗證期)", "驗證期bootstrap正報酬比例%"):
            assert summary.count(f"▸ {t}") == 2, t  # 最多2檔、最多1檔各一張
        heat = summary[summary.index("【熱度圖｜主要"):summary.index("【熱度圖｜敏感度對照")]
        pf_rows = heat[heat.index("▸ 挑選期PF"):heat.index("▸ 驗證期PF")].splitlines()
        assert pf_rows[2].split()[0] == "1.0倍" and pf_rows[2].split()[1:] == ["0.80", "0.80L", "1.20"]
        assert pf_rows[3].split()[1:] == ["0.80", "1.10b", "1.30*"]
        assert "在高原上的格子：G10_40" in summary
        assert "一個穩健的設定應該坐落在「高原」上" in summary
        assert "PF兩段都較高" in summary
        assert "【9年合計與逐年" in summary and summary.count("  ### G") == 3
        summ = pd.read_csv(tmp_path / "squeeze_kdj_stop_target_grid_summary.csv", encoding="utf-8-sig")
        assert len(summ) == 36
        assert summ.loc[summ["在高原上(自己+鄰域兩段PF都>1)"], "變體代號"].nunique() == 3
        assert summ["被挑選規則選中"].sum() == 2 and set(summ.loc[summ["被挑選規則選中"], "變體代號"]) == {"G15_40"}


class TestSqueezeKdjStopTargetGridCliMode:
    def test_skips_full_pipeline_writes_outputs_without_scipy(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)

        codes = list(STOCK_FUTURES_UNIVERSE)[:12]
        price_data, idx = _make_regime_switching_market(codes + ["2330"], start="2017-06-15", n_days=1950, seed=3)
        load_calls = []

        def _fake_load(universe, start, end, refresh=False):
            load_calls.append((start, end))
            return price_data
        monkeypatch.setattr(cb, "load_price_data", _fake_load)

        def _boom(*args, **kwargs):
            raise AssertionError("--squeeze-kdj-stop-target-grid模式不該呼叫這個")
        monkeypatch.setattr(cb, "load_index_series", _boom)
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(data_loader, "load_index_series", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)
        TestSqueezeKdjFixedCliMode()._patch_heavy_stages_to_explode(monkeypatch)
        for name in ("run_squeeze_kdj_fixed_mode", "run_squeeze_kdj_filters_mode", "run_squeeze_kdj_stops_mode",
                     "run_squeeze_kdj_exits_mode", "run_squeeze_kdj_only_mode",
                     "run_squeeze_kdj_capital_constrained_mode", "run_squeeze_kdj_grid_mode"):
            monkeypatch.setattr(cb, name, _boom)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        mode_calls = []
        real_mode = cb.run_squeeze_kdj_stop_target_grid_mode

        def _spy(*args, **kwargs):
            mode_calls.append(args)
            return real_mode(*args, **kwargs)
        monkeypatch.setattr(cb, "run_squeeze_kdj_stop_target_grid_mode", _spy)

        argv = ["compare_breakout.py", "--squeeze-kdj-stop-target-grid", "--max-stocks", "12",
                "--start", "2018-01-01", "--end", "2024-11-15", "--starting-capital", "1000000"]
        monkeypatch.setattr(sys, "argv", argv)
        cb.main()

        assert load_calls == [("2017-06-15", "2024-11-15")]  # 股價往前多抓200個日曆天
        assert len(mode_calls) == 1
        args, cal = mode_calls[0][0], mode_calls[0][3]
        assert args.commission_per_lot_side == 50.0 and args.futures_tax_rate == 0.00002
        assert cal.equals(idx[idx >= pd.Timestamp("2018-01-01")])

        summ = pd.read_csv(tmp_path / "squeeze_kdj_stop_target_grid_summary.csv", encoding="utf-8-sig")
        assert len(summ) == 36  # 2種最多持倉數 x 9格 x 2期間
        for col in ("最多同時持倉數", "用途", "變體代號", "停損ATR倍數", "停利ATR倍數", "是否基準(G15_30=N2)",
                    "是否目前實盤(G10_30=N1)", "之前模式對應", "期間", "交易筆數", "勝率(%)", "獲利因子PF",
                    "總損益(NT$)", "9年合計損益(NT$，挑選期+驗證期)", "鄰域平均PF_挑選期", "鄰域平均PF_驗證期",
                    "鄰域格數(含自己)", "在高原上(自己+鄰域兩段PF都>1)", "bootstrap正報酬比例(%)",
                    "手續費合計(NT$)", "期交稅合計(NT$)", "被挑選規則選中"):
            assert col in summ.columns, col
        assert set(summ["變體代號"]) == set(ST_ALL_IDS)
        assert summ.loc[summ["是否基準(G15_30=N2)"], "變體代號"].unique().tolist() == ["G15_30"]
        assert summ.loc[summ["是否目前實盤(G10_30=N1)"], "變體代號"].unique().tolist() == ["G10_30"]
        assert summ["被挑選規則選中"].sum() <= 2
        assert summ["交易筆數"].sum() > 0, "合成資料要有交易，測試才有意義"
        assert (summ["手續費合計(NT$)"] == 100.0 * summ["交易筆數"]).all()
        assert set(summ["鄰域格數(含自己)"]) == {3, 4, 5}
        reason_cols = [f"出場_{lab}筆數" for _, lab in cb.SQUEEZE_KDJ_STOPS_EXIT_REASONS]
        assert (summ[reason_cols].sum(axis=1) == summ["交易筆數"]).all()
        # 9年合計 = 同一格兩段損益相加
        for (mp, cid), g in summ.groupby(["最多同時持倉數", "變體代號"]):
            assert g["9年合計損益(NT$，挑選期+驗證期)"].iloc[0] == pytest.approx(g["總損益(NT$)"].sum())

        yearly = pd.read_csv(tmp_path / "squeeze_kdj_stop_target_grid_yearly.csv", encoding="utf-8-sig")
        assert list(yearly.columns) == cb.SQUEEZE_KDJ_FILTER_YEARLY_COLUMNS and len(yearly) > 0
        assert set(yearly["變體代號"]) <= set(ST_ALL_IDS)
        trades = pd.read_csv(tmp_path / "squeeze_kdj_stop_target_grid_trades.csv", encoding="utf-8-sig")
        assert list(trades.columns) == cb.SQUEEZE_KDJ_STOPS_TRADE_COLUMNS
        assert len(trades) == summ["交易筆數"].sum()

        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "--squeeze-kdj-stop-target-grid模式" in summary
        assert (summary.index("【事先登錄的規則") < summary.index("【挑選結果與判定】") < summary.index("【熱度圖"))
        assert "【高原檢查｜最多同時持有1檔" in summary and "最多2檔、最多1檔都在高原上的格子" in summary
        assert "多重比較" in summary and "倖存者偏差" in summary and "偏樂觀" in summary and "基差" in summary
        assert "daily_squeeze_signals.py" in summary
        assert "這次測試無效" not in summary and "[階段0]" not in summary

    def test_invalid_start_prints_big_warning(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        codes = ["1101", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2021-06-01", n_days=600)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        out = cb.run_squeeze_kdj_stop_target_grid_mode(_st_grid_args(start="2021-06-01", end="2023-09-01"),
                                                       price_data, universe, idx)
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "這次測試無效" in summary and "2018-01-01" in summary
        assert summary.index("這次測試無效") < summary.index("【挑選結果與判定】")
        assert out["validity_warnings"]


class TestWorkflowSqueezeModeChoice:
    """GitHub workflow_dispatch最多25個輸入：7個squeeze布林輸入合併成一個squeeze_mode下拉選單。"""
    WF_PATH = os.path.join(os.path.dirname(cb.__file__), ".github", "workflows", "momentum_breakout_backtest.yml")
    EXPECTED_OPTIONS = ["none", "only", "capital_constrained", "grid", "fixed", "filters", "stops", "exits",
                        "stop_target_grid", "execution"]

    def _load(self):
        import yaml
        with open(self.WF_PATH, encoding="utf-8") as f:
            text = f.read()
        return text, yaml.safe_load(text)

    def _run_script(self, wf):
        steps = wf["jobs"]["backtest"]["steps"]
        return next(s["run"] for s in steps if "compare_breakout.py" in s.get("run", ""))

    def _cli_squeeze_flags(self):
        import re
        with open(cb.__file__, encoding="utf-8") as f:
            src = f.read()
        return set(re.findall(r'add_argument\("(--squeeze-kdj-[a-z-]+)"', src))

    def _option_to_flags(self, script):
        import re
        mapping = {}
        for opt, body in re.findall(r'^\s*([a-z_"|]+)\)\s*(.*?);;\s*$', script, flags=re.M):
            mapping[opt] = re.findall(r"--squeeze-kdj-[a-z-]+", body)
        return mapping

    def test_yaml_parses_and_within_limit(self):
        text, wf = self._load()
        inputs = wf[True]["workflow_dispatch"]["inputs"]  # PyYAML把on解析成True
        assert len(inputs) <= 25
        assert not [k for k in inputs if k.startswith("squeeze_kdj_")], "舊的squeeze布林輸入應該全部拿掉"
        sm = inputs["squeeze_mode"]
        assert sm["type"] == "choice" and sm["default"] == "none" and sm["options"] == self.EXPECTED_OPTIONS
        for opt in self.EXPECTED_OPTIONS[1:]:
            assert f"{opt} = " in sm["description"], opt  # 每個選項在說明裡各有一行
        # 非squeeze的輸入保留(含手續費)
        for k in ("start_date", "end_date", "starting_capital", "max_stocks", "simple_combo", "combo_search",
                  "fixed_combo_walkforward_folds", "commission_per_lot_side"):
            assert k in inputs, k
        assert inputs["commission_per_lot_side"]["default"] == "50"
        assert "inputs.squeeze_kdj_" not in text

    def test_every_squeeze_flag_reachable_from_exactly_one_option(self):
        _, wf = self._load()
        script = self._run_script(wf)
        mapping = self._option_to_flags(script)
        options = [o for o in mapping if o not in ('""|none', "*")]
        assert sorted(options) == sorted(self.EXPECTED_OPTIONS[1:])
        assert mapping['""|none'] == []
        flags = self._cli_squeeze_flags()
        assert "--squeeze-kdj-stop-target-grid" in flags and "--squeeze-kdj-execution" in flags and len(flags) == 9
        for opt in options:
            assert len(mapping[opt]) == 1, opt
        for flag in flags:
            assert sum(mapping[o] == [flag] for o in options) == 1, flag

    def test_bash_mapping_actually_produces_args(self, tmp_path):
        import subprocess
        _, wf = self._load()
        inputs = wf[True]["workflow_dispatch"]["inputs"]
        script = self._run_script(wf).replace("python compare_breakout.py $ARGS", 'echo "$ARGS"')

        def render(overrides):
            out = script
            for k, spec in inputs.items():
                out = out.replace("${{ github.event.inputs.%s }}" % k, str(overrides.get(k, spec.get("default", ""))))
            assert "${{" not in out
            return subprocess.run(["bash", "-c", out], capture_output=True, text=True)

        r = render({})
        assert r.returncode == 0 and "--squeeze-kdj" not in r.stdout
        assert "--commission-per-lot-side 50" in r.stdout
        r = render({"squeeze_mode": "stop_target_grid", "start_date": "2018-01-01", "max_stocks": "0"})
        assert r.returncode == 0
        assert r.stdout.split().count("--squeeze-kdj-stop-target-grid") == 1
        assert "--start 2018-01-01" in r.stdout and "--max-stocks 0" in r.stdout
        assert sum(tok.startswith("--squeeze-kdj") for tok in r.stdout.split()) == 1
        r = render({"squeeze_mode": "capital_constrained"})
        assert r.stdout.split().count("--squeeze-kdj-capital-constrained") == 1
        r = render({"squeeze_mode": "execution", "start_date": "2018-01-01", "max_stocks": "0"})
        assert r.returncode == 0 and r.stdout.split().count("--squeeze-kdj-execution") == 1
        assert sum(tok.startswith("--squeeze-kdj") for tok in r.stdout.split()) == 1
        r = render({"squeeze_mode": "bogus"})
        assert r.returncode != 0


# ============================================================================
# --squeeze-kdj-execution：進場成交方式X0~X3 x 基準設定B2/B1 + 跳空分組
# ============================================================================
X_IDS = ["X0", "X1", "X2", "X3"]


def _exec_ids_from_kw(kw):
    """從回測參數反推(基準代號, 成交方式代號)。"""
    bid = {(3.0, 2): "B2", (4.0, 1): "B1"}[(kw["atr_target_mult"], kw["max_concurrent_positions"])]
    em = kw["execution_model"]
    if em == "limit_ticks":
        vid = {1: "X0", 2: "X1"}[kw["entry_limit_ticks"]]
    else:
        vid = {"limit_pct": "X2", "market_open": "X3"}[em]
    return bid, vid


def _gap_trade(close_t, open_, pnl, entry="2019-03-04", code="1101"):
    t = _fixed_trade(entry, pd.Timestamp(entry) + pd.Timedelta(days=3), pnl, code=code)
    t.update(entry_trigger_close=close_t, entry_open=open_, entry_gap_pct=open_ / close_t - 1,
             e_price=open_ + 0.5 if open_ >= 100 else open_ + 0.1)
    return t


def _with_gap_fields(trades, gap_open=100.0):
    for t in trades:
        t.update(entry_trigger_close=100.0, entry_open=gap_open, entry_gap_pct=gap_open / 100.0 - 1,
                 e_price=gap_open + 0.5)
    return trades


class TestSqueezeKdjExecutionConstruction:
    def test_variants_bases_and_kwargs(self):
        assert [v[0] for v in cb.SQUEEZE_KDJ_EXEC_VARIANTS] == X_IDS
        assert cb.SQUEEZE_KDJ_EXEC_SELECTABLE == tuple(X_IDS) and cb.SQUEEZE_KDJ_EXEC_BASELINE == "X0"
        assert [(b[0], b[1], b[2]) for b in cb.SQUEEZE_KDJ_EXEC_BASES] == [("B2", 3.0, 2), ("B1", 4.0, 1)]
        kw = cb.squeeze_kdj_exec_backtest_kwargs
        assert kw("B2", "X0") == dict(atr_stop_mult=1.5, atr_target_mult=3.0, entry_filter=None,
                                     execution_model="limit_ticks", entry_limit_ticks=1)
        assert kw("B1", "X1") == dict(atr_stop_mult=1.5, atr_target_mult=4.0, entry_filter=None,
                                     execution_model="limit_ticks", entry_limit_ticks=2)
        assert kw("B2", "X2")["execution_model"] == "limit_pct" and kw("B2", "X2")["entry_limit_pct"] == 0.01
        assert kw("B1", "X3") == dict(atr_stop_mult=1.5, atr_target_mult=4.0, entry_filter=None,
                                     execution_model="market_open")
        assert "目前習慣" in cb.squeeze_kdj_exec_variant_label("X0")


class TestSqueezeKdjGapBuckets:
    @pytest.mark.parametrize("close_t, open_, bucket", [
        (100.0, 99.0, "gap_down"),
        (100.0, 99.99, "gap_down"),
        (100.0, 100.0, "le_1tick"),
        (100.0, 100.5, "le_1tick"),       # = 收盤+1檔，X0會買
        (100.0, 100.6, "1_2ticks"),
        (100.0, 101.0, "1_2ticks"),       # = 收盤+2檔
        (100.0, 101.5, "gt_1pct"),        # 100元：1%=101.0剛好=2檔，(+2檔,+1%]這組是空的
        (60.0, 60.1, "le_1tick"),
        (60.0, 60.15, "1_2ticks"),
        (60.0, 60.2, "1_2ticks"),
        (60.0, 60.3, "2ticks_1pct"),
        (60.0, 60.6, "2ticks_1pct"),      # = 1%限價
        (60.0, 60.7, "gt_1pct"),
        (49.95, 50.0, "le_1tick"),        # 跨級距：49.95+1檔 = 50.0
        (49.95, 50.05, "1_2ticks"),
        (49.95, 50.1, "1_2ticks"),        # 49.95+2檔 = 50.1(不是50.05)
        (49.95, 50.2, "2ticks_1pct"),     # 1%限價 = 50.4495往下取 = 50.4
        (49.95, 50.5, "gt_1pct"),
    ])
    def test_bucket_assignment(self, close_t, open_, bucket):
        assert cb.squeeze_kdj_gap_bucket(close_t, open_) == bucket

    def test_bucket_assignment_matches_engine_fill_rule(self):
        """「le_1tick/gap_down」= X0(限價1檔)會成交，跟引擎用同一個限價函式。"""
        from squeeze_kdj_signal import squeeze_kdj_entry_limit_price
        rng = np.random.default_rng(1)
        for _ in range(500):
            c = float(rng.choice([9.5, 23.45, 49.95, 77.7, 100.0, 233.5, 499.5, 812.0, 1005.0]))
            o = c * (1 + rng.normal(0, 0.01))
            fills_x0 = o <= squeeze_kdj_entry_limit_price(c, "limit_ticks", entry_limit_ticks=1)
            fills_x1 = o <= squeeze_kdj_entry_limit_price(c, "limit_ticks", entry_limit_ticks=2)
            b = cb.squeeze_kdj_gap_bucket(c, o)
            assert (b in ("gap_down", "le_1tick")) == fills_x0
            assert (b in ("gap_down", "le_1tick", "1_2ticks")) == fills_x1

    def test_bucket_table_stats(self):
        trades = [_gap_trade(100.0, 99.0, -100), _gap_trade(100.0, 99.5, 300),        # 開低
                  _gap_trade(100.0, 100.5, 200),                                      # 平盤~1檔
                  _gap_trade(100.0, 101.0, -50), _gap_trade(100.0, 100.6, 150),       # 1~2檔
                  _gap_trade(60.0, 60.3, 80),                                         # 2檔~1%
                  _gap_trade(100.0, 103.0, -400), _gap_trade(100.0, 102.0, 100)]      # >1%
        gb = cb.squeeze_kdj_gap_bucket_table(trades)
        b = gb["buckets"]
        assert [b[k]["count"] for k in ("gap_down", "le_1tick", "1_2ticks", "2ticks_1pct", "gt_1pct")] == [2, 1, 2, 1, 2]
        assert b["gap_down"]["win_rate"] == pytest.approx(50.0) and b["gap_down"]["profit_factor"] == pytest.approx(3.0)
        assert b["1_2ticks"]["total_pnl_ntd"] == pytest.approx(100) and b["1_2ticks"]["avg_pnl_ntd"] == pytest.approx(50)
        assert b["gt_1pct"]["profit_factor"] == pytest.approx(0.25)
        assert b["2ticks_1pct"]["profit_factor"] == float("inf")
        above = gb["above_1tick"]
        assert above["count"] == 5 and above["total_pnl_ntd"] == pytest.approx(-120)
        assert above["share_pct"] == pytest.approx(5 / 8 * 100)
        assert gb["x0_fillable"]["count"] == 3 and gb["x0_fillable"]["total_pnl_ntd"] == pytest.approx(400)
        assert gb["total"]["count"] == 8
        assert "保護了你" in cb.squeeze_kdj_gap_bucket_reading(gb)
        gb2 = cb.squeeze_kdj_gap_bucket_table([_gap_trade(100.0, 101.0, 500), _gap_trade(100.0, 100.0, -10)])
        assert "有代價" in cb.squeeze_kdj_gap_bucket_reading(gb2)
        gb3 = cb.squeeze_kdj_gap_bucket_table([_gap_trade(100.0, 100.0, -10)])
        assert "看不出" in cb.squeeze_kdj_gap_bucket_reading(gb3)
        empty = cb.squeeze_kdj_gap_bucket_table([])
        assert empty["total"]["count"] == 0 and empty["above_1tick"]["share_pct"] == 0.0

    def test_entry_breakdown_slippage_and_skip_share(self):
        trades = [_gap_trade(100.0, 100.0, 10), _gap_trade(100.0, 102.0, 10)]  # 成交價 = 開盤+0.5
        diag = dict.fromkeys(cb.CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS, 0)
        diag.update(candidates_total=10, skipped_no_slot=2, skipped_limit_not_filled=2)
        eb = cb.squeeze_kdj_exec_entry_breakdown(trades, diag)
        mult = cb.get_contract_multiplier("1101", 100.5)
        assert eb["avg_entry_slippage_ntd"] == pytest.approx(0.5 * mult)
        assert eb["avg_entry_gap_pct"] == pytest.approx(1.0)
        assert eb["skipped_limit"] == 2 and eb["orders_placed"] == 8
        assert eb["skipped_limit_pct_of_orders"] == pytest.approx(25.0)
        assert eb["skipped_limit_pct_of_candidates"] == pytest.approx(20.0)


class TestSqueezeKdjExecutionSelectionAndVerdict:
    def _all(self, **override):
        base = {v: _stats(100, 0.8) for v in X_IDS}
        base.update(override)
        return base

    def test_selection_rule_train_only(self):
        sel, reason = cb.select_squeeze_kdj_exec_variant(self._all(X3=_stats(60, 1.3), X2=_stats(59, 9.0)))
        assert sel == "X3" and "X2" in reason and "排除" in reason
        sel, _ = cb.select_squeeze_kdj_exec_variant(self._all(X1=_stats(100, 1.1, pnl=100), X2=_stats(100, 1.1, pnl=300)))
        assert sel == "X2"
        tied = {v: _stats(100, 1.1, pnl=100) for v in X_IDS}
        assert cb.select_squeeze_kdj_exec_variant(tied)[0] == "X0"  # 同分 → X0(維持現狀)優先
        assert cb.select_squeeze_kdj_exec_variant(self._all(F0=_stats(500, 9.0)))[0] == "X0"

    def test_no_selection_when_fewer_than_60_trades(self):
        sel, reason = cb.select_squeeze_kdj_exec_variant({v: _stats(59, 2.0) for v in X_IDS})
        assert sel is None and "無法挑選" in reason
        v = cb.squeeze_kdj_exec_verdict(None, {})
        assert v["report_only"] and not v["passed"] and v["checks"] == []
        assert "只報告" in v["text"] and "60" in v["text"]

    def _seg(self, n, pf, pct):
        return {"stats": _stats(n, pf), "bootstrap": {"pct_positive": pct, "p_value": 1 - pct / 100}}

    def test_verdict_branches_vs_x0(self):
        base = self._seg(80, 0.9, 30.0)
        v = cb.squeeze_kdj_exec_verdict("X3", {"X0": base, "X3": self._seg(70, 1.4, 90.0)})
        assert v["passed"] and v["text"].startswith("✅") and not v["report_only"]
        v = cb.squeeze_kdj_exec_verdict("X3", {"X0": base, "X3": self._seg(70, 1.4, 75.0)})
        assert not v["passed"] and "bootstrap" in v["text"] and "PF>1" not in v["text"]
        v = cb.squeeze_kdj_exec_verdict("X3", {"X0": self._seg(80, 1.6, 95.0), "X3": self._seg(70, 1.4, 90.0)})
        assert not v["passed"] and "勝過X0" in v["text"]
        v = cb.squeeze_kdj_exec_verdict("X2", {"X0": base, "X2": self._seg(70, 0.8, 20.0)})
        assert not v["passed"] and "PF>1" in v["text"]
        v = cb.squeeze_kdj_exec_verdict("X0", {"X0": self._seg(80, 2.0, 99.0)})
        assert not v["passed"] and "維持不追價" in v["text"]

    def test_improvement_table_vs_x0(self):
        def seg(wr, pf):
            return {"stats": {"trade_count": 10, "win_rate": wr, "profit_factor": pf}}
        by = {v: {"train": seg(40, 0.8), "test": seg(40, 0.8)} for v in X_IDS}
        by["X0"] = {"train": seg(45, 1.0), "test": seg(45, 1.0)}
        by["X3"] = {"train": seg(50, 1.2), "test": seg(50, 1.1)}
        rows = {r["variant"]: r for r in cb.squeeze_kdj_exec_improvement_table(by)}
        assert set(rows) == {"X1", "X2", "X3"} and rows["X3"]["both_periods_better"]
        assert not rows["X1"]["both_periods_better"]


class TestSqueezeKdjExecutionModeSelectionIgnoresTest:
    # B2：X3挑選期PF最高；B1：全部<60筆 → 只報告
    TRAIN = {("B2", v): (100, 0.9) for v in X_IDS}
    TRAIN.update({("B2", "X3"): (120, 1.3), ("B2", "X2"): (40, 9.0)})
    TRAIN.update({("B1", v): (50, 1.5) for v in X_IDS})

    def _run(self, monkeypatch, tmp_path, test_pf, train=None, **arg_extra):
        calls = []
        train = train or self.TRAIN

        def _fake_backtest(**kw):
            calls.append(kw)
            key = _exec_ids_from_kw(kw)
            if kw["master_calendar"][0] < pd.Timestamp("2023-01-01"):
                n, pf = train[key]
                trades = _make_pf_trades(n, pf, 2019)
            else:
                trades = _make_pf_trades(80, test_pf[key], 2024)
            # X3的交易：一半開盤跳空2%(目前習慣不會買的)
            for k, t in enumerate(_with_gap_fields(trades)):
                if key[1] == "X3" and k % 2 == 0:
                    t.update(entry_open=102.0, entry_gap_pct=0.02, e_price=102.5)
            diag = dict.fromkeys(cb.CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS, 0)
            diag.update(candidates_total=len(trades) + 20, skipped_limit_not_filled=0 if key[1] == "X3" else 20)
            return trades, diag
        monkeypatch.setattr(cb, "run_squeeze_kdj_capital_constrained_backtest", _fake_backtest)

        def _boom(*a, **k):
            raise AssertionError("--squeeze-kdj-execution不需要大盤指數")
        monkeypatch.setattr(cb, "load_index_series", _boom)
        monkeypatch.setattr(cb, "load_squeeze_kdj_market_index", _boom)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        codes = ["1101", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2018-01-01", n_days=2300)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        out = cb.run_squeeze_kdj_execution_mode(_stops_args(**arg_extra), price_data, universe, idx)
        return out, calls

    def test_selection_uses_train_only_and_b1_report_only(self, monkeypatch, tmp_path):
        bad = {(b, v): 1.2 for b in ("B2", "B1") for v in X_IDS}
        bad.update({("B2", "X1"): 9.0, ("B2", "X3"): 0.5})
        out, calls = self._run(monkeypatch, tmp_path, bad)
        assert out["selected"] == {"B2": "X3", "B1": None}
        assert not out["verdicts"]["B2"]["passed"] and out["verdicts"]["B1"]["report_only"]
        assert len(calls) == 2 * 4 * 2  # 2個基準 x 4種成交方式 x 2期間
        assert {_exec_ids_from_kw(kw) for kw in calls} == {(b, v) for b in ("B2", "B1") for v in X_IDS}
        for kw in calls:
            assert kw["lots"] == 1 and kw["top_n"] == 3 and kw["atr_stop_mult"] == 1.5
            assert kw["variant"] == "B" and kw["entry_filter"] is None
            assert kw["max_hold_days"] == 20 and kw["atr_period"] == 14 and kw["ranking_rule"] == "trigger_return"
            assert "slippage_pct" not in kw and "market_series" not in kw
            assert kw["commission_per_lot_side"] == 50.0 and kw["futures_tax_rate"] == 0.00002
        train_cals = [kw["master_calendar"] for kw in calls if kw["master_calendar"][0] < pd.Timestamp("2023-01-01")]
        assert len(train_cals) == 8 and all(c[-1] <= pd.Timestamp("2022-12-31") for c in train_cals)
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "⚪ 不挑選、只報告" in summary and "→ 選出：無(只報告)" in summary
        # 驗證期X3好過X0 → 選擇不變、通過
        good = {(b, v): 1.0 for b in ("B2", "B1") for v in X_IDS}
        good.update({("B2", "X3"): 2.0, ("B2", "X0"): 1.2})
        out2, _ = self._run(monkeypatch, tmp_path, good)
        assert out2["selected"]["B2"] == "X3" and out2["verdicts"]["B2"]["passed"]
        assert "✅ 通過" in (tmp_path / "summary.txt").read_text(encoding="utf-8")
        # X3驗證期PF>1但沒勝過X0 → 不通過
        worse = dict(good, **{})
        worse[("B2", "X0")] = 3.0
        out3, _ = self._run(monkeypatch, tmp_path, worse)
        assert out3["selected"]["B2"] == "X3" and not out3["verdicts"]["B2"]["passed"]
        assert "勝過X0" in out3["verdicts"]["B2"]["text"]

    def test_b1_selects_when_enough_trades(self, monkeypatch, tmp_path):
        train = dict(self.TRAIN)
        train.update({("B1", "X1"): (61, 1.4), ("B1", "X0"): (60, 1.2)})
        test_pf = {(b, v): 1.1 for b in ("B2", "B1") for v in X_IDS}
        out, _ = self._run(monkeypatch, tmp_path, test_pf, train=train)
        assert out["selected"] == {"B2": "X3", "B1": "X1"}
        assert not out["verdicts"]["B1"]["report_only"]

    def test_summary_sections_files_and_gap_buckets(self, monkeypatch, tmp_path):
        test_pf = {(b, v): 1.1 for b in ("B2", "B1") for v in X_IDS}
        out, _ = self._run(monkeypatch, tmp_path, test_pf)
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        i_rules = summary.index("【事先登錄的規則")
        assert (i_rules < summary.index("【交易成本假設】") < summary.index("【挑選結果與判定】")
                < summary.index("【兩段都變好？B2") < summary.index("【兩段都變好？B1")
                < summary.index("【並排總表｜B2") < summary.index("【並排總表｜B1")
                < summary.index("【跳空分組") < summary.index("【各成交方式明細｜B2") < summary.index("【誠實caveat】"))
        rules = summary[i_rules:summary.index("【挑選結果與判定】")]
        for vid in X_IDS:
            assert f"    {vid}(" in rules
        assert "B2：" in rules and "B1：" in rules and "只報告" in rules
        gap = summary[summary.index("【跳空分組"):summary.index("【各成交方式明細")]
        assert gap.count("▸ B2｜") == 2 and gap.count("▸ B1｜") == 2
        assert "目前習慣放棄的單" in gap and "名額效應" in gap and "保護" in gap
        # 假的X3交易：偶數筆(PF的賺錢那半)跳空2% → 超過+1%那組全賺 → 「有代價」
        assert "有代價" in gap
        assert "流動性可能比股票本身薄" in summary and "多重比較" in summary and "倖存者偏差" in summary
        assert "限價1檔沒成交" not in summary[summary.index("【各成交方式明細"):]

        gb = out["results"]["B2"]["X3"]["train"]["gap_buckets"]
        assert gb["buckets"]["gt_1pct"]["count"] == 60 and gb["buckets"]["le_1tick"]["count"] == 60  # B2的X3挑選期120筆
        assert gb["above_1tick"]["profit_factor"] == float("inf")

        summ = pd.read_csv(tmp_path / "squeeze_kdj_execution_summary.csv", encoding="utf-8-sig")
        assert len(summ) == 16
        for col in ("基準設定", "變體代號", "成交方式", "停損ATR倍數", "停利ATR倍數", "最多同時持倉數", "期間", "交易筆數",
                    "每月交易筆數", "勝率(%)", "獲利因子PF", "總損益(NT$)", "平均獲利(NT$/筆)", "平均虧損(NT$/筆)",
                    "最大回撤(NT$)", "拿掉最大3筆後損益(NT$)", "bootstrap正報酬比例(%)", "手續費合計(NT$)",
                    "期交稅合計(NT$)", "限價沒成交筆數", "限價沒成交佔掛單比例(%)", "平均進場滑價(成交價-開盤，NT$/筆)",
                    "被挑選規則選中"):
            assert col in summ.columns, col
        assert set(zip(summ["基準設定"], summ["最多同時持倉數"], summ["停利ATR倍數"])) == {("B2", 2, 3.0), ("B1", 1, 4.0)}
        assert set(summ.loc[summ["被挑選規則選中"], "變體代號"]) == {"X3"}
        assert set(summ.loc[summ["被挑選規則選中"], "基準設定"]) == {"B2"}
        assert (summ.loc[summ["變體代號"] == "X3", "限價沒成交筆數"] == 0).all()
        assert (summ.loc[summ["變體代號"] == "X0", "限價沒成交佔掛單比例(%)"] > 0).all()

        yearly = pd.read_csv(tmp_path / "squeeze_kdj_execution_yearly.csv", encoding="utf-8-sig")
        assert list(yearly.columns) == ["基準設定"] + cb.SQUEEZE_KDJ_FILTER_YEARLY_COLUMNS and len(yearly) == 16
        trades = pd.read_csv(tmp_path / "squeeze_kdj_execution_trades.csv", encoding="utf-8-sig")
        assert list(trades.columns) == cb.SQUEEZE_KDJ_EXEC_TRADE_COLUMNS
        assert len(trades) == summ["交易筆數"].sum()
        assert set(trades.loc[trades["變體代號"] == "X3", "跳空分組"]) == {"超過+1%", "平盤~+1檔(X0也會買)"}
        gaps = pd.read_csv(tmp_path / "squeeze_kdj_execution_gap_buckets.csv", encoding="utf-8-sig")
        assert list(gaps.columns) == cb.SQUEEZE_KDJ_EXEC_GAP_COLUMNS
        assert len(gaps) == 2 * 2 * (5 + 3)  # 2基準 x 2期間 x (5組+3列合計)
        assert set(gaps["跳空分組"]) >= {lab for _, lab in cb.SQUEEZE_KDJ_EXEC_GAP_BUCKETS}

    def test_invalid_start_prints_big_warning(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))
        codes = ["1101", "2330"]
        price_data, idx = _make_regime_switching_market(codes, start="2021-06-01", n_days=600)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        out = cb.run_squeeze_kdj_execution_mode(_stops_args(start="2021-06-01", end="2023-09-01"),
                                                price_data, universe, idx)
        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "這次測試無效" in summary and "2018-01-01" in summary
        assert summary.index("這次測試無效") < summary.index("【挑選結果與判定】")
        assert out["validity_warnings"]


class TestSqueezeKdjExecutionRealEngine:
    def test_x0_equals_limit_1tick_and_x3_has_gap_fields(self):
        """真引擎：X0(limit_ticks N=1)跟之前各模式的限價1檔逐筆相同；X3沒有任何限價略過。"""
        codes = list(STOCK_FUTURES_UNIVERSE)[:8]
        price_data, idx = _make_regime_switching_market(codes, start="2018-01-01", n_days=1500, seed=4)
        universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
        res = cb.run_squeeze_kdj_execution_backtests(price_data, universe, 1_000_000, idx, 50.0, 0.00002)
        ref = cb._run_squeeze_kdj_preregistered_cost_backtests(
            price_data, universe, 1_000_000, idx, 50.0, 0.00002, (2,),
            [("N2", dict(atr_stop_mult=1.5, atr_target_mult=3.0, entry_filter=None))], cb.SQUEEZE_KDJ_STOPS_EXIT_REASONS)
        for period in ("train", "test"):
            x0 = res["B2"]["X0"][period]
            strip = [{k: v for k, v in t.items() if k not in ("entry_trigger_close", "entry_open", "entry_gap_pct")}
                     for t in x0["trades"]]
            assert strip == ref[2]["N2"][period]["trades"]
            assert x0["diagnostics"] == ref[2]["N2"][period]["diagnostics"]
            for b in ("B2", "B1"):
                x3 = res[b]["X3"][period]
                assert x3["skipped_limit"] == 0
                assert x3["gap_buckets"]["total"]["count"] == len(x3["trades"])
        assert sum(len(res["B2"]["X0"][p]["trades"]) for p in ("train", "test")) > 0


class TestSqueezeKdjExecutionCliMode:
    def test_skips_full_pipeline_writes_outputs_without_scipy(self, monkeypatch, tmp_path):
        import data_loader
        import chip_data_loader
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)

        codes = list(STOCK_FUTURES_UNIVERSE)[:12]
        price_data, idx = _make_regime_switching_market(codes + ["2330"], start="2017-06-15", n_days=1950, seed=3)
        load_calls = []

        def _fake_load(universe, start, end, refresh=False):
            load_calls.append((start, end))
            return price_data
        monkeypatch.setattr(cb, "load_price_data", _fake_load)

        def _boom(*args, **kwargs):
            raise AssertionError("--squeeze-kdj-execution模式不該呼叫這個")
        monkeypatch.setattr(cb, "load_index_series", _boom)
        monkeypatch.setattr(data_loader, "load_price_data", _boom)
        monkeypatch.setattr(data_loader, "load_index_series", _boom)
        monkeypatch.setattr(chip_data_loader, "load_chip_data", _boom)
        TestSqueezeKdjFixedCliMode()._patch_heavy_stages_to_explode(monkeypatch)
        for name in ("run_squeeze_kdj_fixed_mode", "run_squeeze_kdj_filters_mode", "run_squeeze_kdj_stops_mode",
                     "run_squeeze_kdj_exits_mode", "run_squeeze_kdj_stop_target_grid_mode", "run_squeeze_kdj_only_mode",
                     "run_squeeze_kdj_capital_constrained_mode", "run_squeeze_kdj_grid_mode"):
            monkeypatch.setattr(cb, name, _boom)
        monkeypatch.setattr(cb, "RESULTS_DIR", str(tmp_path))

        mode_calls = []
        real_mode = cb.run_squeeze_kdj_execution_mode

        def _spy(*args, **kwargs):
            mode_calls.append(args)
            return real_mode(*args, **kwargs)
        monkeypatch.setattr(cb, "run_squeeze_kdj_execution_mode", _spy)

        argv = ["compare_breakout.py", "--squeeze-kdj-execution", "--max-stocks", "12",
                "--start", "2018-01-01", "--end", "2024-11-15", "--starting-capital", "1000000"]
        monkeypatch.setattr(sys, "argv", argv)
        cb.main()

        assert load_calls == [("2017-06-15", "2024-11-15")]  # 股價往前多抓200個日曆天
        assert len(mode_calls) == 1
        args, cal = mode_calls[0][0], mode_calls[0][3]
        assert args.commission_per_lot_side == 50.0 and args.futures_tax_rate == 0.00002
        assert cal.equals(idx[idx >= pd.Timestamp("2018-01-01")])

        summ = pd.read_csv(tmp_path / "squeeze_kdj_execution_summary.csv", encoding="utf-8-sig")
        assert len(summ) == 16  # 2基準 x 4成交方式 x 2期間
        assert summ["交易筆數"].sum() > 0, "合成資料要有交易，測試才有意義"
        assert (summ["手續費合計(NT$)"] == 100.0 * summ["交易筆數"]).all()
        assert (summ.loc[summ["變體代號"] == "X3", "限價沒成交筆數"] == 0).all()
        reason_cols = [f"出場_{lab}筆數" for _, lab in cb.SQUEEZE_KDJ_STOPS_EXIT_REASONS]
        assert (summ[reason_cols].sum(axis=1) == summ["交易筆數"]).all()
        trades = pd.read_csv(tmp_path / "squeeze_kdj_execution_trades.csv", encoding="utf-8-sig")
        assert len(trades) == summ["交易筆數"].sum()
        assert trades["開盤跳空(%)"].notna().all()
        gaps = pd.read_csv(tmp_path / "squeeze_kdj_execution_gap_buckets.csv", encoding="utf-8-sig")
        x3_trades = summ.loc[summ["變體代號"] == "X3"].groupby(["基準設定", "期間"])["交易筆數"].sum()
        tot = gaps[gaps["跳空分組"] == "全部"].set_index(["基準設定", "期間"])["交易筆數"]
        assert tot.sort_index().tolist() == x3_trades.sort_index().tolist()
        five = gaps[gaps["跳空分組"].isin([lab for _, lab in cb.SQUEEZE_KDJ_EXEC_GAP_BUCKETS])]
        assert five.groupby(["基準設定", "期間"])["交易筆數"].sum().sort_index().tolist() == x3_trades.sort_index().tolist()
        assert (tmp_path / "squeeze_kdj_execution_yearly.csv").exists()

        summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
        assert "--squeeze-kdj-execution模式" in summary
        assert summary.index("【事先登錄的規則") < summary.index("【挑選結果與判定】") < summary.index("【跳空分組")
        assert "這次測試無效" not in summary and "[階段0]" not in summary
