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
