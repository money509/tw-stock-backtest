"""squeeze_kdj_signal.py 的單元測試：全部用合成OHLC資料，不打網路。"""
import numpy as np
import pandas as pd
import pytest

from squeeze_kdj_signal import (
    compute_keltner_channel, compute_squeeze_flag, compute_kdj_k,
    compute_squeeze_kdj_features, compute_entry_state_machine,
    simulate_variant_a_trades, simulate_variant_b_trades,
    precompute_squeeze_kdj_features_by_code, run_squeeze_kdj_capital_constrained_backtest,
    _process_squeeze_kdj_variant_a_day,
)
from mean_reversion_engine import compute_bollinger, STOP_LOSS_COOLDOWN_DAYS


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


# ============================================================================
# run_squeeze_kdj_capital_constrained_backtest() 及相關輔助函式的測試。
#
# 共用做法：直接手造「已經算好」的features DataFrame(EntryFlag/PriorLow/K欄位)，
# 透過features_by_code參數直接餵給回測函式，跳過真正的BB/KC/KDJ計算鏈路——
# 這些鏈路本身(擠壓判定/KDJ的K值)已經在上面的TestSqueezeFlag/TestKDJ/
# TestEntryStateMachine測過，這裡只測「資金受限版的day-by-day迴圈邏輯本身對不對」，
# 兩件事不要混在一起測，才看得出問題出在哪一層。
# ============================================================================

def _make_flat_df(price, n=20, start="2022-01-03"):
    idx = pd.date_range(start, periods=n, freq="B")
    closes = pd.Series([price] * n, index=idx, dtype=float)
    return pd.DataFrame({
        "Open": closes, "High": closes + 1.0, "Low": closes - 1.0, "Close": closes,
    }, index=idx)


def _make_fixed_features(df, entry_idx, prior_low, k_series=None):
    """造一份只在entry_idx這天EntryFlag=True的features，K預設全程50(不會觸發變體A停利)。"""
    n = len(df)
    entry_flag = np.zeros(n, dtype=bool)
    entry_flag[entry_idx] = True
    prior_low_arr = np.full(n, np.nan)
    prior_low_arr[entry_idx] = prior_low
    if k_series is None:
        k_series = pd.Series([50.0] * n, index=df.index)
    return pd.DataFrame({
        "EntryFlag": entry_flag, "PriorLow": prior_low_arr, "K": k_series,
    }, index=df.index)


class TestPrecomputeSqueezeKdjFeaturesByCode:
    def test_skips_stocks_shorter_than_60_rows_and_keeps_the_rest(self):
        price_data = {
            "1101": _make_flat_df(20.0, n=59),   # 太短，該被跳過
            "1102": _make_flat_df(15.0, n=60),   # 剛好60，該保留
        }
        universe = {"1101": {}, "1102": {}}
        result = precompute_squeeze_kdj_features_by_code(price_data, universe)
        assert set(result.keys()) == {"1102"}
        for col in ("Squeeze", "SqueezeRecent", "K", "EntryFlag", "PriorLow"):
            assert col in result["1102"].columns


class TestCapitalConstrainedInvalidVariant:
    def test_invalid_variant_raises_value_error(self):
        price_data = {"1101": _make_flat_df(20.0)}
        universe = {"1101": {}}
        with pytest.raises(ValueError):
            run_squeeze_kdj_capital_constrained_backtest(
                price_data, universe, price_data["1101"].index, starting_capital=1_000_000,
                variant="C",
            )


class TestCapitalConstrainedEntryTiming:
    def test_entry_executes_at_open_of_day_after_entry_flag_not_same_day_or_two_days_later(self):
        """EntryFlag在idx5觸發，進場該發生在idx6開盤，不是idx5或idx7——
        這是確認features_by_code的.shift(1)對齊沒有錯位(見
        run_squeeze_kdj_capital_constrained_backtest() docstring的「進場時機」說明)。"""
        df = _make_flat_df(20.0, n=20)
        features = _make_fixed_features(df, entry_idx=5, prior_low=1.0)  # 停損價很低，不會被打到
        price_data = {"1101": df}
        universe = {"1101": {}}

        trades = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, df.index, starting_capital=1_000_000, variant="A",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features},
            max_hold_days=3,
        )
        assert len(trades) == 1
        assert trades[0]["entry_date"] == df.index[6]


class TestCapitalConstrainedVariantAExit:
    def test_target_fires_after_k_reaches_80_then_drops_below(self):
        idx = pd.date_range("2022-01-03", periods=10, freq="B")
        closes = pd.Series([100.0] * 10, index=idx)
        df = pd.DataFrame({
            "Open": closes, "High": closes + 1.0, "Low": closes - 20.0,  # 停損價設得極低，不會被打到
            "Close": closes,
        }, index=idx)
        # 進場於idx2(EntryFlag在idx1)；K：idx2=70 idx3=85(武裝) idx4=90 idx5=75(武裝後跌破80 -> 觸發停利)
        k = pd.Series([50, 50, 70, 85, 90, 75, 60, 60, 60, 60], index=idx, dtype=float)
        features = _make_fixed_features(df, entry_idx=1, prior_low=-1000.0, k_series=k)
        price_data = {"1101": df}
        universe = {"1101": {}}

        trades = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, df.index, starting_capital=1_000_000, variant="A",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features},
            max_hold_days=30,
        )
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "target"
        assert trades[0]["exit_date"] == idx[5]
        assert trades[0]["exit_price"] == pytest.approx(100.0)

    def test_stop_hits_prior_low(self):
        idx = pd.date_range("2022-01-03", periods=10, freq="B")
        closes = pd.Series([100.0] * 10, index=idx)
        lows = pd.Series([99.0] * 10, index=idx)
        lows.iloc[3] = 50.0  # idx3大跌破停損
        df = pd.DataFrame({"Open": closes, "High": closes + 1.0, "Low": lows, "Close": closes}, index=idx)
        features = _make_fixed_features(df, entry_idx=1, prior_low=95.0)
        price_data = {"1101": df}
        universe = {"1101": {}}

        trades = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, df.index, starting_capital=1_000_000, variant="A",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features},
            max_hold_days=30,
        )
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "stop"
        assert trades[0]["exit_price"] == pytest.approx(95.0)


class TestProcessSqueezeKdjVariantADayDirectly:
    def test_stop_gap_uses_open_price_and_sets_cooldown(self):
        idx = pd.date_range("2022-01-03", periods=3, freq="B")
        row = pd.Series({"Open": 90.0, "High": 91.0, "Low": 89.0, "Close": 90.0})
        position = {
            "code": "1101", "side": "long", "entry_date": idx[0], "e_price": 100.0,
            "target_price": None, "stop_price": 95.0, "lots": 1, "hold_days": 1,
            "margin_used": 0.0, "target_armed": False,
        }
        trades = []
        cooldown_until = {}
        result = _process_squeeze_kdj_variant_a_day(
            position, row, k_today=50.0, date=idx[1], trades=trades,
            max_hold_days=30, cooldown_until=cooldown_until,
        )
        assert result is None
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "stop_gap"
        assert trades[0]["exit_price"] == pytest.approx(90.0)
        assert cooldown_until["1101"] == idx[1] + pd.Timedelta(days=STOP_LOSS_COOLDOWN_DAYS * 2)


class TestCapitalConstrainedVariantBAtrFramework:
    def test_stop_and_target_reuse_process_mr_day_shape(self):
        idx = pd.date_range("2022-01-03", periods=25, freq="B")
        n = len(idx)
        closes = pd.Series([100.0] * n, index=idx)
        df = pd.DataFrame({
            "Open": closes, "High": closes + 1.0, "Low": closes - 1.0, "Close": closes,
        }, index=idx)
        df.loc[df.index[-1], "High"] = 300.0  # 最後一天大幅衝高，確保能在資料結束前觸發停利
        entry_idx = n - 5
        features = _make_fixed_features(df, entry_idx=entry_idx, prior_low=np.nan)
        price_data = {"1101": df}
        universe = {"1101": {}}

        trades = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, df.index, starting_capital=1_000_000, variant="B",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features},
            atr_stop_mult=1.0, atr_target_mult=2.0, atr_period=14, max_hold_days=30,
        )
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "target"
        # trade dict形狀要跟mean_reversion_engine._close_mr_trade()輸出相容
        for key in ("code", "side", "entry_date", "exit_date", "e_price", "exit_price",
                    "exit_reason", "lots", "pnl_ntd", "return_pct", "hold_days"):
            assert key in t


class TestCapitalConstrainedMarginCap:
    def test_candidate_skipped_when_margin_exceeds_single_trade_cap(self):
        # 2330是台積電(mini合約100股，保證金比例13.5%)，故意用很小的starting_capital
        # 讓單筆保證金超過35%上限，確認候選被跳過、不會硬擠進場。
        idx = pd.date_range("2022-01-03", periods=10, freq="B")
        price = 600.0
        closes = pd.Series([price] * 10, index=idx)
        df = pd.DataFrame({
            "Open": closes, "High": closes + 1.0, "Low": closes - 1.0, "Close": closes,
        }, index=idx)
        features = _make_fixed_features(df, entry_idx=1, prior_low=price - 100.0)
        price_data = {"2330": df}
        universe = {"2330": {}}

        trades = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, df.index, starting_capital=1_000.0,  # 小到任何保證金都會超過35%
            variant="A", max_concurrent_positions=1, top_n=1,
            features_by_code={"2330": features}, max_hold_days=5,
        )
        assert trades == []


class TestCapitalConstrainedMaxConcurrentPositionsAndRanking:
    def test_slots_fill_up_to_max_concurrent_positions(self):
        idx = pd.date_range("2022-01-03", periods=10, freq="B")
        price_data = {}
        features_by_code = {}
        universe = {}
        for i, code in enumerate(["1101", "1102", "1210"]):
            df = _make_flat_df(20.0 + i, n=10, start="2022-01-03")
            df.index = idx
            price_data[code] = df
            features_by_code[code] = _make_fixed_features(df, entry_idx=1, prior_low=1.0)
            universe[code] = {}

        trades = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, idx, starting_capital=10_000_000, variant="A",
            max_concurrent_positions=2, top_n=3, features_by_code=features_by_code,
            max_hold_days=3,
        )
        # 3檔股票同一天都觸發，名額只有2個，只能有2筆交易成交
        assert len(trades) == 2

    def test_ranking_prefers_bigger_trigger_day_return_when_slots_limited(self):
        # 兩檔股票同一天(idx=t=1)觸發，但觸發K棒當天(t=1)的漲幅不同：code A漲5%、
        # code B漲1%。只有1個名額時，應該優先選A(漲幅較大者)。
        idx = pd.date_range("2022-01-03", periods=10, freq="B")

        def _df_with_trigger_return(closes_t0, closes_t1):
            closes = [closes_t0, closes_t1] + [closes_t1] * 8
            s = pd.Series(closes, index=idx, dtype=float)
            return pd.DataFrame({"Open": s, "High": s + 1.0, "Low": s - 1.0, "Close": s}, index=idx)

        df_big = _df_with_trigger_return(100.0, 105.0)   # t=1漲5%
        df_small = _df_with_trigger_return(100.0, 101.0)  # t=1漲1%
        price_data = {"1101": df_big, "1102": df_small}
        universe = {"1101": {}, "1102": {}}
        features_by_code = {
            "1101": _make_fixed_features(df_big, entry_idx=1, prior_low=1.0),
            "1102": _make_fixed_features(df_small, entry_idx=1, prior_low=1.0),
        }

        trades = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, idx, starting_capital=10_000_000, variant="A",
            max_concurrent_positions=1, top_n=1, features_by_code=features_by_code,
            max_hold_days=3,
        )
        assert len(trades) == 1
        assert trades[0]["code"] == "1101"


class TestCapitalConstrainedCooldownAfterStop:
    def test_stopped_out_code_excluded_during_cooldown_window(self):
        idx = pd.date_range("2022-01-03", periods=20, freq="B")
        closes = pd.Series([100.0] * 20, index=idx)
        lows = pd.Series([99.0] * 20, index=idx)
        lows.iloc[3] = 50.0  # idx3觸發停損出場(進場於idx2)
        df = pd.DataFrame({"Open": closes, "High": closes + 1.0, "Low": lows, "Close": closes}, index=idx)

        entry_flag = np.zeros(20, dtype=bool)
        entry_flag[1] = True   # 第一次武裝觸發 -> idx2進場 -> idx3停損
        entry_flag[5] = True   # 冷卻期內的第二次觸發，idx6理論上該進場但被冷卻排除
        prior_low = np.full(20, np.nan)
        prior_low[1] = 95.0
        prior_low[5] = 95.0
        k = pd.Series([50.0] * 20, index=idx)
        features = pd.DataFrame({"EntryFlag": entry_flag, "PriorLow": prior_low, "K": k}, index=idx)

        price_data = {"1101": df}
        universe = {"1101": {}}
        trades = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, idx, starting_capital=1_000_000, variant="A",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features},
            max_hold_days=30,
        )
        # 只有第一筆(idx2進場、idx3停損)會成交，第二次觸發(idx6該進場)落在冷卻期內被排除
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "stop"
