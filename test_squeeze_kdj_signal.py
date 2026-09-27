"""squeeze_kdj_signal.py 的單元測試：全部用合成OHLC資料，不打網路。"""
import numpy as np
import pandas as pd
import pytest

from squeeze_kdj_signal import (
    compute_keltner_channel, compute_squeeze_flag, compute_kdj_k,
    compute_squeeze_kdj_features, compute_entry_state_machine,
    simulate_variant_a_trades, simulate_variant_b_trades,
)
from mean_reversion_engine import compute_bollinger


def _make_df(opens, highs, lows, closes):
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


class TestSqueezeFlag:
    def test_narrow_close_range_with_wide_bars_is_squeeze(self):
        # Close幾乎不動(std很小 -> BB很窄)，但每天K棒本身的高低點範圍不小(每天真實區間
        # 約1.0 -> ATR約1.0 -> KC用1.5倍ATR，通道相對寬) -> BB應該收在KC裡面，判定擠壓。
        n = 30
        closes = [100.0 + (0.05 if i % 2 == 0 else -0.05) for i in range(n)]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        df = _make_df(closes, highs, lows, closes)
        _, bb_upper, bb_lower = compute_bollinger(df["Close"])
        _, kc_upper, kc_lower = compute_keltner_channel(df)
        squeeze = compute_squeeze_flag(bb_upper, bb_lower, kc_upper, kc_lower)
        assert bool(squeeze.iloc[-1])

    def test_high_volatility_not_squeeze(self):
        # 每天大幅震盪(High/Low遠離Close)，ATR很大 -> KC很寬；Close本身只小幅擺動 ->
        # BB相對窄，BB應該收在KC裡面，這個案例其實還是squeeze == True，
        # 所以反過來構造一個「Close本身劇烈跳動、真實區間反而窄」的案例來製造非擠壓。
        n = 30
        rng = np.random.RandomState(0)
        closes = 100 + np.cumsum(rng.choice([-5, 5], size=n))  # 每天不是+5就是-5，劇烈震盪
        highs = closes + 0.1
        lows = closes - 0.1
        opens = closes
        df = _make_df(opens, highs, lows, closes)
        _, bb_upper, bb_lower = compute_bollinger(df["Close"])
        _, kc_upper, kc_lower = compute_keltner_channel(df)
        squeeze = compute_squeeze_flag(bb_upper, bb_lower, kc_upper, kc_lower)
        # 真實區間(High-Low)很窄 -> ATR很小 -> KC很窄；但Close本身劇烈跳動 -> BB很寬 ->
        # BB應該比KC寬，不是擠壓。
        assert not bool(squeeze.iloc[-1])

    def test_nan_warmup_is_not_squeeze(self):
        closes = [100.0] * 5
        df = _make_df(closes, closes, closes, closes)
        _, bb_upper, bb_lower = compute_bollinger(df["Close"])
        _, kc_upper, kc_lower = compute_keltner_channel(df)
        squeeze = compute_squeeze_flag(bb_upper, bb_lower, kc_upper, kc_lower)
        assert not squeeze.any()


class TestKDJ:
    def test_hand_computed_k_values(self):
        # 手算3筆(period=25, min_periods=1的關係，前幾筆用當下累積的rolling window)：
        # Row0: H10 L10 C10 -> rng=0 -> rsv填50 -> K0 = 2/3*50 + 1/3*50 = 50
        # Row1: H12 L8 C12 -> window[10,12]=low_min8,high_max12,rng4 -> rsv=(12-8)/4*100=100
        #       K1 = 2/3*50 + 1/3*100 = 66.66666...
        # Row2: H12 L8 C8 -> window still low_min8,high_max12,rng4 -> rsv=(8-8)/4*100=0
        #       K2 = 2/3*K1 + 1/3*0 = 44.44444...
        opens = [10, 10, 8]
        highs = [10, 12, 12]
        lows = [10, 8, 8]
        closes = [10, 12, 8]
        df = _make_df(opens, highs, lows, closes)
        k = compute_kdj_k(df, period=25)
        assert k.iloc[0] == pytest.approx(50.0)
        assert k.iloc[1] == pytest.approx(66.666667, abs=1e-4)
        assert k.iloc[2] == pytest.approx(44.444444, abs=1e-4)

    def test_k_bounded_0_100(self):
        rng = np.random.RandomState(1)
        n = 60
        closes = 100 + np.cumsum(rng.normal(0, 2, n))
        highs = closes + np.abs(rng.normal(1, 0.5, n))
        lows = closes - np.abs(rng.normal(1, 0.5, n))
        df = _make_df(closes, highs, lows, closes)
        k = compute_kdj_k(df, period=25)
        assert (k >= 0).all() and (k <= 100).all()


class TestEntryStateMachine:
    def test_fires_exactly_on_key_candle_not_before_or_after(self):
        idx = pd.date_range("2024-01-01", periods=8, freq="B")
        close = pd.Series([100, 99, 98, 97, 96, 97, 98, 99], index=idx, dtype=float)
        low = close - 0.5
        bb_lower = pd.Series([99.5] * 8, index=idx, dtype=float)  # close跌破bb_lower從idx2開始
        # K持續下降到<20，第5根(idx5)開始回升到>20
        k = pd.Series([50, 30, 15, 10, 8, 25, 40, 45], index=idx, dtype=float)
        squeeze_recent = pd.Series([True] * 8, index=idx)

        entry_flag, prior_low = compute_entry_state_machine(
            close=close, low=low, bb_lower=bb_lower, k=k, squeeze_recent=squeeze_recent,
            oversold_k=20.0, max_armed_bars=15,
        )
        # 武裝發生在idx2(close=98<99.5 且 k=15<20)。
        # idx3: close97<98(收黑，不是收紅) -> 不觸發，即使idx3也close<bb_lower/k<20都不影響已armed。
        # idx4: close96<97(收黑) -> 不觸發。
        # idx5: close97>96(收紅) 且 k=25>20 -> 觸發！這是第一個滿足「收紅+K>20」的武裝後K棒。
        assert entry_flag.tolist() == [False, False, False, False, False, True, False, False]
        assert prior_low[5] == pytest.approx(low.iloc[4])
        # 觸發後武裝重置，idx6/idx7即使收紅也不會再度觸發(除非重新武裝)
        assert not entry_flag[6] and not entry_flag[7]

    def test_no_arming_without_squeeze_recent(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        close = pd.Series([100, 98, 97, 98, 99], index=idx, dtype=float)
        low = close - 0.5
        bb_lower = pd.Series([99.0] * 5, index=idx, dtype=float)
        k = pd.Series([50, 15, 10, 25, 30], index=idx, dtype=float)
        squeeze_recent = pd.Series([False] * 5, index=idx)  # 完全沒有擠壓過

        entry_flag, prior_low = compute_entry_state_machine(
            close=close, low=low, bb_lower=bb_lower, k=k, squeeze_recent=squeeze_recent,
        )
        assert not entry_flag.any()

    def test_stale_arming_expires_after_max_armed_bars(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        close = pd.Series([100, 95, 96, 96, 96, 96, 96, 96, 95, 96], index=idx, dtype=float)
        low = close - 0.5
        bb_lower = pd.Series([99.0] * 10, index=idx, dtype=float)
        # idx1武裝(close95<99, k=10<20)。之後K一直維持<20(不會再滿足"k>20"觸發條件)，
        # 直到idx8又跌回K=10(仍<20，不影響已armed狀態，也不會重新武裝，因為已經armed)，
        # idx9才k回升到25，但此時已經超過max_armed_bars=3，武裝早該過期，不該觸發。
        k = pd.Series([50, 10, 10, 10, 10, 10, 10, 10, 10, 25], index=idx, dtype=float)
        squeeze_recent = pd.Series([True] * 10, index=idx)

        entry_flag, prior_low = compute_entry_state_machine(
            close=close, low=low, bb_lower=bb_lower, k=k, squeeze_recent=squeeze_recent,
            oversold_k=20.0, max_armed_bars=3,
        )
        assert not entry_flag.any()


class TestSqueezeKdjFeatures:
    def test_returns_expected_columns_and_no_crash_on_short_series(self):
        closes = [100.0 + i * 0.1 for i in range(10)]
        df = _make_df(closes, closes, closes, closes)
        features = compute_squeeze_kdj_features(df)
        for col in ("Squeeze", "SqueezeRecent", "K", "EntryFlag", "PriorLow"):
            assert col in features.columns
        assert len(features) == len(df)


class TestVariantAExit:
    def test_stop_hits_prior_candle_low(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="B")
        df = _make_df(
            opens=[100, 100, 100, 100, 100, 100],
            highs=[101, 101, 101, 101, 101, 101],
            lows=[99, 99, 90, 99, 99, 99],   # entry_idx=2(第3根)的前一根(idx1)最低價=99
            closes=[100, 100, 100, 100, 100, 100],
        )
        n = len(df)
        entry_flag = np.zeros(n, dtype=bool)
        entry_flag[1] = True  # 觸發在idx1，隔天(idx2)開盤進場，停損=idx1前一根(idx0)的最低價
        prior_low = np.full(n, np.nan)
        prior_low[1] = df["Low"].iloc[0]  # =99
        k = pd.Series([50] * n, index=df.index)  # K全程不到80，不會觸發停利
        features = pd.DataFrame({"EntryFlag": entry_flag, "PriorLow": prior_low, "K": k}, index=df.index)

        trades = simulate_variant_a_trades(df, features)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "stop"
        assert t["exit_price"] == pytest.approx(99.0)
        # idx2進場(open=100)，idx2的low=90 <= stop(99)，同一天就停損出場
        assert t["hold_days"] == 1

    def test_take_profit_fires_on_k_cross_back_below_80_after_reaching_80(self):
        idx = pd.date_range("2024-01-01", periods=7, freq="B")
        df = _make_df(
            opens=[100] * 7,
            highs=[101] * 7,
            lows=[95] * 7,     # 停損價設得很低，不會被碰到
            closes=[100, 100, 101, 102, 103, 102, 101],
        )
        n = len(df)
        entry_flag = np.zeros(n, dtype=bool)
        entry_flag[1] = True
        prior_low = np.full(n, np.nan)
        prior_low[1] = 50.0  # 遠低於任何K棒的Low，確保不會被停損打到
        # 進場在idx2；K：idx2=70(<80) idx3=85(>=80，武裝停利) idx4=90(仍>=80)
        # idx5=75(<80，且已經武裝過 -> 觸發停利，用idx5收盤價出場)
        k = pd.Series([50, 60, 70, 85, 90, 75, 60], index=df.index, dtype=float)
        features = pd.DataFrame({"EntryFlag": entry_flag, "PriorLow": prior_low, "K": k}, index=df.index)

        trades = simulate_variant_a_trades(df, features)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "target"
        assert t["exit_price"] == pytest.approx(df["Close"].iloc[5])
        assert t["exit_date"] == df.index[5]

    def test_forced_close_when_no_exit_before_data_ends(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        df = _make_df(
            opens=[100] * 4, highs=[101] * 4, lows=[95] * 4, closes=[100, 100, 101, 102],
        )
        n = len(df)
        entry_flag = np.zeros(n, dtype=bool)
        entry_flag[0] = True
        prior_low = np.full(n, np.nan)
        prior_low[0] = 50.0
        k = pd.Series([50] * n, index=df.index, dtype=float)
        features = pd.DataFrame({"EntryFlag": entry_flag, "PriorLow": prior_low, "K": k}, index=df.index)

        trades = simulate_variant_a_trades(df, features)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "forced_close"
        assert trades[0]["exit_price"] == pytest.approx(df["Close"].iloc[-1])


class TestVariantBExit:
    def test_atr_stop_hit(self):
        idx = pd.date_range("2024-01-01", periods=25, freq="B")
        n = len(idx)
        # 用平穩價格讓ATR是個穩定小數，方便算出精確的停損價
        closes = [100.0] * (n - 3) + [100.0, 100.0, 100.0]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        opens = closes
        df = _make_df(opens, highs, lows, closes)
        # 進場後(idx=n-2)low大幅跌破，觸發停損
        df.loc[df.index[-2], "Low"] = 50.0
        entry_flag = np.zeros(n, dtype=bool)
        entry_flag[n - 3] = True  # 隔天(n-2)開盤進場
        features = pd.DataFrame({"EntryFlag": entry_flag}, index=df.index)

        trades = simulate_variant_b_trades(df, features, atr_period=20, atr_stop_mult=1.0,
                                            atr_target_mult=2.0)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "stop"

    def test_atr_target_hit(self):
        idx = pd.date_range("2024-01-01", periods=25, freq="B")
        n = len(idx)
        closes = [100.0] * n
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        opens = closes
        df = _make_df(opens, highs, lows, closes)
        df.loc[df.index[-1], "High"] = 200.0  # 最後一天大幅衝高，觸發停利
        entry_flag = np.zeros(n, dtype=bool)
        entry_flag[n - 3] = True
        features = pd.DataFrame({"EntryFlag": entry_flag}, index=df.index)

        trades = simulate_variant_b_trades(df, features, atr_period=20, atr_stop_mult=1.0,
                                            atr_target_mult=2.0)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "target"

    def test_max_hold_days_forces_exit(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        n = len(idx)
        closes = [100.0] * n
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        opens = closes
        df = _make_df(opens, highs, lows, closes)
        entry_flag = np.zeros(n, dtype=bool)
        entry_flag[20] = True
        features = pd.DataFrame({"EntryFlag": entry_flag}, index=df.index)

        trades = simulate_variant_b_trades(df, features, atr_period=20, atr_stop_mult=1.0,
                                            atr_target_mult=None, max_hold_days=3)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "hold_days_reached"
        assert trades[0]["hold_days"] == 3
