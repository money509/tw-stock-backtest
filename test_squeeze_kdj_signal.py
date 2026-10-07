"""squeeze_kdj_signal.py 的單元測試：全部用合成OHLC資料，不打網路。"""
import numpy as np
import pandas as pd
import pytest

from squeeze_kdj_signal import (
    compute_keltner_channel, compute_squeeze_flag, compute_kdj_k,
    compute_squeeze_kdj_features, compute_entry_state_machine,
    simulate_variant_a_trades, simulate_variant_b_trades,
    precompute_squeeze_kdj_features_by_code, run_squeeze_kdj_capital_constrained_backtest,
    _process_squeeze_kdj_variant_a_day, precompute_squeeze_kdj_backtest_arrays,
    CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS, MAX_HOLD_DAYS_CAPITAL_CONSTRAINED_DEFAULT,
)
import mean_reversion_engine
from mean_reversion_engine import (
    compute_bollinger, STOP_LOSS_COOLDOWN_DAYS, compute_atr_correct, DEFAULT_MARGIN_CAP_RATIO,
    _process_mr_day,
)
from taifex_universe import STOCK_FUTURES_UNIVERSE, estimate_margin, get_contract_multiplier


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


class TestCapitalConstrainedNaNPriceRows:
    """回歸測試：GitHub Actions --squeeze-kdj-grid在真實資料上exit code 1。
    data_loader只用dropna(how="all")清資料，yfinance偶爾回傳「有成交量但價格是NaN」的列，
    這種列以前會被拿去當成交價/強制平倉價，損益變NaN後風險預算口數int(NaN)直接崩潰。
    現在價格不完整的列一律當成「這天沒有這檔的資料」。"""

    def test_no_entry_when_entry_day_open_is_nan(self):
        df = _make_flat_df(20.0, n=20)
        df.iloc[6, df.columns.get_loc("Open")] = np.nan  # EntryFlag在idx5 → 原本該在idx6開盤進場
        features = _make_fixed_features(df, entry_idx=5, prior_low=1.0)
        trades = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, starting_capital=1_000_000, variant="A",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features}, max_hold_days=3,
        )
        assert trades == []

    def test_open_position_skips_nan_day_instead_of_closing_at_nan_price(self):
        df = _make_flat_df(20.0, n=20)
        df.iloc[8, :] = np.nan  # 原本max_hold_days=3會在idx8用收盤價強制平倉
        features = _make_fixed_features(df, entry_idx=5, prior_low=1.0)
        trades = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, starting_capital=1_000_000, variant="A",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features}, max_hold_days=3,
        )
        assert len(trades) == 1
        assert trades[0]["exit_date"] == df.index[9]
        assert np.isfinite(trades[0]["exit_price"]) and np.isfinite(trades[0]["pnl_ntd"])

    def test_risk_sizing_does_not_crash_with_nan_rows(self):
        df = _make_flat_df(20.0, n=30)
        df.iloc[8, :] = np.nan
        df.iloc[16, df.columns.get_loc("Open")] = np.nan
        n = len(df)
        entry_flag = np.zeros(n, dtype=bool)
        entry_flag[[5, 15, 20]] = True
        features = pd.DataFrame({
            "EntryFlag": entry_flag, "PriorLow": np.where(entry_flag, 15.0, np.nan),
            "K": [50.0] * n,
        }, index=df.index)
        trades = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, starting_capital=1_000_000, variant="A",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features},
            max_hold_days=3, risk_pct_per_trade=0.02,
        )
        assert all(np.isfinite(t["pnl_ntd"]) for t in trades)


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


# ============================================================================
# 預先計算(precompute)重構的「行為不變」快照測試
#
# 下面的_legacy_capital_constrained_backtest()是重構之前(commit db84650)
# run_squeeze_kdj_capital_constrained_backtest()的凍結複製品(逐字複製，只拿掉docstring、
# 改函式名)，當作「重構前行為」的快照。為什麼用凍結的舊版實作、而不是存一份交易明細的
# JSON：交易明細要靠BB/KC/KDJ計算鏈路產生，pandas版本不同時rolling標準差之類的計算可能有
# 極微小的浮點差異，存成JSON的快照會在不同環境間變得脆弱；用凍結的舊版實作在同一份輸入
# 上即時比對，比的純粹是「逐日迴圈重構前後」這件事，不受環境影響。
# 重構後的版本在預設參數(不開任何新功能)下，必須產生跟舊版「逐筆、逐欄位完全相同」的交易。
# ============================================================================

def _legacy_capital_constrained_backtest(price_data: dict, universe: dict,
                                                   master_calendar: pd.DatetimeIndex,
                                                   starting_capital: float, variant: str = "B",
                                                   lots: int = 2, top_n: int = 3,
                                                   max_concurrent_positions: int = 3,
                                                   atr_stop_mult: float = 1.0, atr_target_mult: float = 2.0,
                                                   atr_period: int = 14,
                                                   max_hold_days: int = MAX_HOLD_DAYS_CAPITAL_CONSTRAINED_DEFAULT,
                                                   slippage_pct: float = 0.0,
                                                   features_by_code: dict = None) -> list:
    """凍結的重構前舊版實作(逐字複製自commit db84650，只拿掉docstring、改函式名)，見上方說明。"""
    if variant not in ("A", "B"):
        raise ValueError(f"variant必須是'A'或'B'，收到{variant!r}")

    if features_by_code is None:
        features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)

    # 逐碼預先算好「查表版」的進場訊號(已經shift(1)對齊成「今天能不能進場」)，
    # 避免在day-by-day迴圈裡對每個日期重複做t/t+1的日期運算。
    entry_by_code = {}
    atr_by_code = {}
    for code, features in features_by_code.items():
        df = price_data.get(code)
        if df is None:
            continue
        close = df["Close"]
        trigger_return = (close / close.shift(1) - 1)  # t日(觸發K棒)本身的漲幅，排名用
        entry_by_code[code] = pd.DataFrame({
            "EntryToday": features["EntryFlag"].shift(1).fillna(False).astype(bool),
            "StopPriceA": features["PriorLow"].shift(1),
            "TriggerStrength": trigger_return.shift(1),
        }, index=df.index)
        if variant == "B":
            atr_by_code[code] = compute_atr_correct(df, period=atr_period).shift(1)

    effective_total_margin_cap_ratio = None
    if max_concurrent_positions > 1:
        effective_total_margin_cap_ratio = min(DEFAULT_MARGIN_CAP_RATIO * max_concurrent_positions, 0.9)

    trades = []
    cooldown_until = {}
    open_positions = []

    for date in master_calendar:
        # 1) 先處理既有部位的出場判定
        still_open = []
        for position in open_positions:
            df = price_data.get(position["code"])
            if df is None or date not in df.index:
                still_open.append(position)
                continue
            if date != position["entry_date"]:
                position["hold_days"] += 1
            row = df.loc[date]
            if variant == "B":
                updated = _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until,
                                           slippage_pct=slippage_pct)
            else:
                k_series = features_by_code[position["code"]]["K"]
                k_today = k_series.loc[date] if date in k_series.index else np.nan
                updated = _process_squeeze_kdj_variant_a_day(position, row, k_today, date, trades,
                                                               max_hold_days, cooldown_until,
                                                               slippage_pct=slippage_pct)
            if updated is not None:
                still_open.append(updated)
        open_positions = still_open

        # 2) 收集今天觸發進場的候選，依「觸發K棒當天漲幅」排名，依序補進空出來的名額
        held_codes = {p["code"] for p in open_positions}
        excluded_codes = {c for c, until in cooldown_until.items() if date < until} | held_codes
        slots_available = max_concurrent_positions - len(open_positions)
        if slots_available <= 0:
            continue

        candidates = []
        for code, sig_df in entry_by_code.items():
            if code in excluded_codes:
                continue
            if date not in sig_df.index:
                continue
            sig_row = sig_df.loc[date]
            if not bool(sig_row["EntryToday"]):
                continue
            if pd.isna(sig_row["TriggerStrength"]):
                continue
            df = price_data.get(code)
            if df is None or date not in df.index:
                continue
            candidates.append({
                "code": code,
                "stop_price_a": sig_row["StopPriceA"],
                "trigger_strength": float(sig_row["TriggerStrength"]),
            })

        candidates.sort(key=lambda c: c["trigger_strength"], reverse=True)
        candidates = candidates[: max(top_n, slots_available)]

        used_margin = sum(p["margin_used"] for p in open_positions)
        while slots_available > 0 and candidates:
            cand = candidates.pop(0)
            code = cand["code"]
            df = price_data[code]
            open_p = df.loc[date, "Open"]
            e_price = open_p * (1 + slippage_pct)

            if variant == "B":
                atr_at_signal = atr_by_code[code].loc[date] if date in atr_by_code[code].index else np.nan
                if pd.isna(atr_at_signal) or atr_at_signal <= 0:
                    continue
                stop_price = e_price - atr_stop_mult * atr_at_signal
                target_price = e_price + atr_target_mult * atr_at_signal
            else:
                stop_price = cand["stop_price_a"]
                if pd.isna(stop_price):
                    continue
                target_price = None  # 變體A的停利靠target_armed狀態判定，不是固定價位

            margin_needed = estimate_margin(code, open_p, lots)
            if margin_needed > starting_capital * DEFAULT_MARGIN_CAP_RATIO:
                continue
            if effective_total_margin_cap_ratio is not None and \
                    used_margin + margin_needed > starting_capital * effective_total_margin_cap_ratio:
                continue

            position = {
                "code": code, "side": "long", "entry_date": date,
                "e_price": e_price, "target_price": target_price, "stop_price": stop_price,
                "lots": lots, "hold_days": 1, "margin_used": margin_needed,
            }
            if variant == "A":
                position["target_armed"] = False

            used_margin += margin_needed
            slots_available -= 1

            row = df.loc[date]
            if variant == "B":
                updated = _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until,
                                           slippage_pct=slippage_pct)
            else:
                k_series = features_by_code[code]["K"]
                k_today = k_series.loc[date] if date in k_series.index else np.nan
                updated = _process_squeeze_kdj_variant_a_day(position, row, k_today, date, trades,
                                                               max_hold_days, cooldown_until,
                                                               slippage_pct=slippage_pct)
            if updated is not None:
                open_positions.append(updated)
            else:
                used_margin -= margin_needed

    return trades


def _make_synthetic_market(n_stocks=20, n_days=400, seed=0):
    """合成多檔股票的OHLCV：波動度每40天在「低波動(容易擠壓)」跟「高波動(容易跌破下軌+
    K<20再反彈)」之間切換，讓squeeze+KDJ訊號在合理的資料量內就會觸發好幾次。
    股票代號取自STOCK_FUTURES_UNIVERSE(合約乘數/保證金查表需要真實代號)。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    codes = list(STOCK_FUTURES_UNIVERSE.keys())[:n_stocks]
    price_data = {}
    for code in codes:
        vol = np.where((np.arange(n_days) // 40) % 2 == 0, 0.006, 0.025) * rng.uniform(0.7, 1.3)
        close = 50 * rng.uniform(0.5, 4) * np.exp(np.cumsum(rng.normal(0.0002, vol)))
        open_ = close * (1 + rng.normal(0, 0.004, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        volume = rng.lognormal(8, 0.5, n_days)
        price_data[code] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
    universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
    return price_data, universe, idx


_SNAPSHOT_SCENARIOS = [
    # (seed, variant, max_concurrent_positions, lots, starting_capital, slippage_pct, top_n, max_hold_days, calendar)
    (0, "A", 3, 2, 200_000, 0.0, 3, MAX_HOLD_DAYS_CAPITAL_CONSTRAINED_DEFAULT, "full"),
    (0, "B", 3, 2, 200_000, 0.0, 3, MAX_HOLD_DAYS_CAPITAL_CONSTRAINED_DEFAULT, "full"),
    (1, "A", 1, 1, 1_000_000, 0.002, 1, 10, "is"),
    (1, "B", 1, 2, 1_000_000, 0.002, 3, 10, "oos"),
    (2, "A", 5, 2, 1_000_000, 0.0, 3, 20, "is"),
    (2, "B", 5, 1, 200_000, 0.001, 3, 60, "oos"),
    (3, "B", 3, 2, 500_000, 0.0, 2, 15, "full"),
    (3, "A", 2, 2, 500_000, 0.0, 3, 5, "full"),
]


class TestPrecomputeRefactorMatchesLegacySnapshot:
    @pytest.mark.parametrize("scenario", _SNAPSHOT_SCENARIOS)
    def test_default_params_produce_identical_trades_to_pre_refactor_version(self, scenario):
        seed, variant, mcp, lots, capital, slip, top_n, hold, cal = scenario
        price_data, universe, idx = _make_synthetic_market(seed=seed)
        calendar = {"full": idx, "is": idx[:280], "oos": idx[280:]}[cal]
        features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)
        kwargs = dict(variant=variant, lots=lots, top_n=top_n, max_concurrent_positions=mcp,
                      max_hold_days=hold, slippage_pct=slip, features_by_code=features_by_code)

        legacy = _legacy_capital_constrained_backtest(price_data, universe, calendar, capital, **kwargs)
        new = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, calendar, capital, **kwargs)

        assert len(legacy) > 0, "快照情境本身要有交易，不然比對是空的、沒有意義"
        assert new == legacy  # dict逐欄位相等(含浮點數值完全相同)，順序也要一樣

    def test_builds_precompute_internally_when_nothing_passed(self):
        """features_by_code/precomputed都不給時，函式自己算，結果跟舊版一樣。"""
        price_data, universe, idx = _make_synthetic_market(n_stocks=10, seed=4)
        legacy = _legacy_capital_constrained_backtest(price_data, universe, idx, 500_000, variant="A")
        new = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 500_000, variant="A")
        assert new == legacy

    def test_shared_precomputed_gives_same_result_as_internal_build(self):
        price_data, universe, idx = _make_synthetic_market(n_stocks=10, seed=5)
        features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)
        pre = precompute_squeeze_kdj_backtest_arrays(price_data, features_by_code, atr_period=14)
        a = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 500_000, variant="B",
                                                          features_by_code=features_by_code)
        b = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 500_000, variant="B",
                                                          precomputed=pre)
        assert a == b


class TestCapitalConstrainedArgumentValidation:
    def test_precomputed_with_different_atr_period_raises(self):
        price_data = {"1101": _make_flat_df(20.0, n=30)}
        features = _make_fixed_features(price_data["1101"], entry_idx=5, prior_low=1.0)
        pre = precompute_squeeze_kdj_backtest_arrays(price_data, {"1101": features}, atr_period=20)
        with pytest.raises(ValueError):
            run_squeeze_kdj_capital_constrained_backtest(
                price_data, {"1101": {}}, price_data["1101"].index, 1_000_000, atr_period=14, precomputed=pre)

    @pytest.mark.parametrize("bad_kwargs", [{"ranking_rule": "rsi"}, {"entry_filter": "above_ma20"}])
    def test_invalid_ranking_rule_or_entry_filter_raises(self, bad_kwargs):
        price_data = {"1101": _make_flat_df(20.0)}
        with pytest.raises(ValueError):
            run_squeeze_kdj_capital_constrained_backtest(
                price_data, {"1101": {}}, price_data["1101"].index, 1_000_000, **bad_kwargs)


def _make_trailing_df():
    """30天：前22天(idx0~21)完全持平在100(H=101/L=99，ATR(14)剛好=2.0)，idx20觸發→idx21進場(開盤100)；
    idx22收104、idx23收108(一路往上，移動停利跟著上移)；idx24開107、盤中殺到90。"""
    n = 30
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l = np.full(n, 99.0)
    c = np.full(n, 100.0)
    o[22], h[22], l[22], c[22] = 102.0, 105.0, 101.0, 104.0
    o[23], h[23], l[23], c[23] = 106.0, 109.0, 105.0, 108.0
    o[24], h[24], l[24], c[24] = 107.0, 108.0, 90.0, 95.0
    o[25:], h[25:], l[25:], c[25:] = 95.0, 96.0, 94.0, 95.0
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)


class TestCapitalConstrainedBTrailUsesTrailingMachinery:
    def test_b_trail_ratchets_stop_via_update_trailing_stop_and_ignores_fixed_target(self, monkeypatch):
        """B_trail：初始停損=100-1.5x2=97，不設固定停利；之後交給_process_mr_day()既有的移動停利
        (update_trailing_stop)：idx22收104→停損101、idx23收108→停損105，idx24盤中殺到90 → 在105停損出場。
        同一份資料如果是變體B(停利2.0xATR=104)，idx22盤中高105就會先碰到固定停利——用來對照
        B_trail確實沒有固定停利目標。另外spy mean_reversion_engine.update_trailing_stop，
        確認真的是重用既有的移動停利函式，不是另外寫一套。"""
        df = _make_trailing_df()
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        price_data = {"1101": df}

        calls = []
        real_update = mean_reversion_engine.update_trailing_stop

        def _spy(position, row):
            calls.append(position["code"])
            return real_update(position, row)
        monkeypatch.setattr(mean_reversion_engine, "update_trailing_stop", _spy)

        trades = run_squeeze_kdj_capital_constrained_backtest(
            price_data, {"1101": {}}, df.index, 1_000_000, variant="B_trail",
            atr_stop_mult=1.5, trailing_atr_mult=1.5, atr_target_mult=2.0,
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features}, max_hold_days=30,
        )
        assert len(trades) == 1
        t = trades[0]
        assert t["entry_date"] == df.index[21]
        assert t["exit_date"] == df.index[24]
        assert t["exit_reason"] == "stop"
        assert t["exit_price"] == pytest.approx(105.0)  # 108 - 1.5 x 2.0，不是初始停損97
        assert len(calls) >= 3  # 進場日、idx22、idx23都有呼叫移動停利

        trades_b = run_squeeze_kdj_capital_constrained_backtest(
            price_data, {"1101": {}}, df.index, 1_000_000, variant="B",
            atr_stop_mult=1.5, atr_target_mult=2.0,
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features}, max_hold_days=30,
        )
        assert trades_b[0]["exit_reason"] == "target"
        assert trades_b[0]["exit_price"] == pytest.approx(104.0)

    def test_b_trail_position_fields_match_try_enter_breakout(self, monkeypatch):
        """部位欄位逐字照momentum_breakout_engine.try_enter_breakout()設定。攔截進場當天第一次
        _process_mr_day呼叫，檢查position dict。"""
        import squeeze_kdj_signal as skd
        df = _make_trailing_df()
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        seen = []

        def _spy(position, row, date, trades, max_hold_days, cooldown_until, slippage_pct=0.0):
            seen.append(dict(position))
            return _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until,
                                   slippage_pct=slippage_pct)
        monkeypatch.setattr(skd, "_process_mr_day", _spy)

        run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, 1_000_000, variant="B_trail",
            atr_stop_mult=2.0, max_concurrent_positions=1, top_n=1,
            features_by_code={"1101": features}, max_hold_days=30,
        )
        first = seen[0]
        assert first["target_price"] is None
        assert first["stop_price"] == pytest.approx(100.0 - 2.0 * 2.0)
        assert first["trailing_stop"] is True
        assert first["trailing_atr_mult"] == 2.0  # trailing_atr_mult沒給時=atr_stop_mult
        assert first["atr_entry"] == pytest.approx(2.0)
        assert first["trailing_anchor"] == pytest.approx(100.0)
        assert first["trailing_activation_days"] == 0
        assert first["trailing_activation_profit_atr"] == 0.0


class TestCapitalConstrainedRiskSizing:
    def _df_and_features(self):
        idx = pd.date_range("2022-01-03", periods=25, freq="B")
        closes = pd.Series([100.0] * 25, index=idx)
        df = pd.DataFrame({"Open": closes, "High": closes + 1.0, "Low": closes - 1.0, "Close": closes}, index=idx)
        features = _make_fixed_features(df, entry_idx=20, prior_low=95.0)  # 變體A停損距離 = 100-95 = 5
        return df, features

    def test_variant_a_lots_from_risk_budget_over_stop_distance(self):
        df, features = self._df_and_features()
        risk_pct = 0.01
        trades = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, 1_000_000, variant="A", risk_pct_per_trade=risk_pct,
            lots=99, max_concurrent_positions=1, top_n=1, features_by_code={"1101": features}, max_hold_days=2,
        )
        mult = get_contract_multiplier("1101", 100.0)
        expected_lots = int(1_000_000 * risk_pct // (5.0 * mult))
        assert expected_lots >= 1
        assert len(trades) == 1
        assert trades[0]["lots"] == expected_lots  # lots=99被忽略

    def test_variant_b_stop_distance_is_atr_multiple(self):
        df, features = self._df_and_features()  # 持平資料，ATR(14)=2.0
        trades = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, 1_000_000, variant="B", atr_stop_mult=1.5,
            risk_pct_per_trade=0.02, max_concurrent_positions=1, top_n=1,
            features_by_code={"1101": features}, max_hold_days=2,
        )
        mult = get_contract_multiplier("1101", 100.0)
        assert trades[0]["lots"] == int(1_000_000 * 0.02 // (1.5 * 2.0 * mult))

    def test_candidate_skipped_when_risk_budget_buys_less_than_one_lot(self):
        df, features = self._df_and_features()
        trades, diag = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, 1_000_000, variant="A", risk_pct_per_trade=0.0001,
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features}, max_hold_days=2,
            return_diagnostics=True,
        )
        assert trades == []
        assert diag["skipped_risk_lots_lt1"] == 1

    def test_equity_includes_realized_pnl(self):
        """帳戶權益 = starting_capital + 已實現損益：第一筆停損虧錢之後，第二筆的口數要用
        「虧完之後」的權益反推，不是永遠用starting_capital。用2330(mini合約100股)讓口數夠大，
        虧損金額足以改變第二筆的口數。"""
        n = 40
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        o = np.full(n, 100.0)
        h = np.full(n, 101.0)
        l = np.full(n, 99.5)
        c = np.full(n, 100.0)
        l[3] = 50.0  # idx2進場、idx3停損在95
        df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)
        entry_flag = np.zeros(n, dtype=bool)
        entry_flag[[1, 20]] = True
        prior_low = np.full(n, np.nan)
        prior_low[[1, 20]] = 95.0
        features = pd.DataFrame({"EntryFlag": entry_flag, "PriorLow": prior_low,
                                 "K": pd.Series(50.0, index=idx)}, index=idx)
        capital, risk_pct = 2_000_000, 0.01
        trades = run_squeeze_kdj_capital_constrained_backtest(
            {"2330": df}, {"2330": {}}, idx, capital, variant="A", risk_pct_per_trade=risk_pct,
            max_concurrent_positions=1, top_n=1, features_by_code={"2330": features}, max_hold_days=3,
        )
        assert len(trades) == 2
        mult = get_contract_multiplier("2330", 100.0)
        assert trades[0]["lots"] == int(capital * risk_pct // (5.0 * mult))
        equity_after = capital + trades[0]["pnl_ntd"]
        assert trades[1]["lots"] == int(equity_after * risk_pct // (5.0 * mult))
        assert trades[1]["lots"] < trades[0]["lots"]


class TestCapitalConstrainedVolumeRatioRanking:
    def test_volume_ratio_rule_reorders_candidates_vs_trigger_return(self):
        """兩檔同一天(t=21)觸發、只有1個名額：1101觸發當天漲5%但量比普通(1倍)，1102只漲1%但
        爆量(5倍)。trigger_return排名選1101；volume_ratio排名選1102。"""
        n = 30
        idx = pd.date_range("2022-01-03", periods=n, freq="B")

        def _df(ret_on_trigger, vol_mult_on_trigger):
            closes = np.full(n, 100.0)
            closes[21:] = 100.0 * (1 + ret_on_trigger)
            vol = np.full(n, 1000.0)
            vol[21] = 1000.0 * vol_mult_on_trigger
            s = pd.Series(closes, index=idx)
            return pd.DataFrame({"Open": s, "High": s + 1.0, "Low": s - 1.0, "Close": s,
                                 "Volume": pd.Series(vol, index=idx)}, index=idx)

        price_data = {"1101": _df(0.05, 1.0), "1102": _df(0.01, 5.0)}
        features_by_code = {c: _make_fixed_features(df, entry_idx=21, prior_low=1.0) for c, df in price_data.items()}
        universe = {"1101": {}, "1102": {}}

        common = dict(variant="A", max_concurrent_positions=1, top_n=1,
                      features_by_code=features_by_code, max_hold_days=2)
        by_return = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 10_000_000, **common)
        by_volume = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 10_000_000,
                                                                  ranking_rule="volume_ratio", **common)
        assert [t["code"] for t in by_return] == ["1101"]
        assert [t["code"] for t in by_volume] == ["1102"]

    def test_volume_ratio_is_shift_aligned_and_uses_trigger_day_volume(self):
        """precompute裡volume_ratio第d列 = 前一個交易日(觸發K棒)的成交量/含當天的20日均量。"""
        n = 30
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        vol = np.full(n, 1000.0)
        vol[21] = 3000.0
        s = pd.Series(100.0, index=idx)
        df = pd.DataFrame({"Open": s, "High": s + 1, "Low": s - 1, "Close": s, "Volume": vol}, index=idx)
        features = _make_fixed_features(df, entry_idx=21, prior_low=1.0)
        pre = precompute_squeeze_kdj_backtest_arrays({"1101": df}, {"1101": features}, atr_period=14)
        vr = pre["per_code"]["1101"]["volume_ratio"]
        expected = 3000.0 / ((19 * 1000.0 + 3000.0) / 20)
        assert vr[22] == pytest.approx(expected)   # 進場日(t+1)查到的是觸發日t的量比
        assert vr[21] == pytest.approx(1.0)        # 觸發日當天查到的是前一天的量比(=1)，不偷看
        assert pre["events_by_date"][idx[22]] == [("1101", 22)]


class TestCapitalConstrainedAboveMa60Filter:
    def _trend_df(self, slope):
        n = 80
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        closes = pd.Series(100.0 + slope * np.arange(n), index=idx)
        return pd.DataFrame({"Open": closes, "High": closes + 1.0, "Low": closes - 1.0, "Close": closes}, index=idx)

    def test_filter_blocks_entry_when_trigger_close_below_ma60(self):
        df = self._trend_df(-0.5)  # 一路下跌：收盤價永遠在60日均線之下
        features = _make_fixed_features(df, entry_idx=70, prior_low=1.0)
        common = dict(variant="A", max_concurrent_positions=1, top_n=1,
                      features_by_code={"1101": features}, max_hold_days=3)
        no_filter = run_squeeze_kdj_capital_constrained_backtest({"1101": df}, {"1101": {}}, df.index, 1_000_000, **common)
        filtered, diag = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, 1_000_000, entry_filter="above_ma60",
            return_diagnostics=True, **common)
        assert len(no_filter) == 1
        assert filtered == []
        assert diag["skipped_entry_filter"] == 1

    def test_filter_allows_entry_when_trigger_close_above_ma60(self):
        df = self._trend_df(+0.5)
        features = _make_fixed_features(df, entry_idx=70, prior_low=1.0)
        trades = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, 1_000_000, variant="A", entry_filter="above_ma60",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features}, max_hold_days=3)
        assert len(trades) == 1

    def test_filter_blocks_during_ma60_warmup(self):
        """均線暖身期(前59天)均線是NaN，濾網開啟時一律擋掉(保守：資料不足時不假設站上季線)。"""
        df = self._trend_df(+0.5)
        features = _make_fixed_features(df, entry_idx=30, prior_low=1.0)
        trades = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, 1_000_000, variant="A", entry_filter="above_ma60",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": features}, max_hold_days=3)
        assert trades == []


class TestCapitalConstrainedDiagnostics:
    def test_default_return_type_is_still_a_list(self):
        df = _make_flat_df(20.0, n=20)
        features = _make_fixed_features(df, entry_idx=5, prior_low=1.0)
        result = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df}, {"1101": {}}, df.index, 1_000_000, variant="A", features_by_code={"1101": features})
        assert isinstance(result, list)

    def test_no_slot_counter(self):
        idx = pd.date_range("2022-01-03", periods=10, freq="B")
        price_data, features_by_code = {}, {}
        for i, code in enumerate(["1101", "1102", "1210"]):
            df = _make_flat_df(20.0 + i, n=10)
            df.index = idx
            price_data[code] = df
            features_by_code[code] = _make_fixed_features(df, entry_idx=1, prior_low=1.0)
        trades, diag = run_squeeze_kdj_capital_constrained_backtest(
            price_data, {c: {} for c in price_data}, idx, 10_000_000, variant="A",
            max_concurrent_positions=2, top_n=3, features_by_code=features_by_code, max_hold_days=3,
            return_diagnostics=True,
        )
        assert set(diag.keys()) == set(CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS)
        assert len(trades) == 2
        assert diag["candidates_total"] == 3
        assert diag["skipped_no_slot"] == 1
        assert diag["skipped_single_margin_cap"] == 0
        assert diag["skipped_total_margin_cap"] == 0

    def test_no_slot_counter_counts_triggers_on_days_with_all_slots_full(self):
        """名額已經全滿的日子，當天的觸發一樣要記在「名額已滿」底下(不是默默消失)。"""
        idx = pd.date_range("2022-01-03", periods=10, freq="B")
        df1 = _make_flat_df(20.0, n=10); df1.index = idx
        df2 = _make_flat_df(21.0, n=10); df2.index = idx
        f1 = _make_fixed_features(df1, entry_idx=1, prior_low=1.0)   # idx2進場，抱到max_hold_days
        f2 = _make_fixed_features(df2, entry_idx=3, prior_low=1.0)   # idx4想進場時名額已滿
        trades, diag = run_squeeze_kdj_capital_constrained_backtest(
            {"1101": df1, "1102": df2}, {"1101": {}, "1102": {}}, idx, 10_000_000, variant="A",
            max_concurrent_positions=1, top_n=1, features_by_code={"1101": f1, "1102": f2}, max_hold_days=5,
            return_diagnostics=True,
        )
        assert [t["code"] for t in trades] == ["1101"]
        assert diag["skipped_no_slot"] == 1

    def test_single_margin_cap_counter(self):
        idx = pd.date_range("2022-01-03", periods=10, freq="B")
        closes = pd.Series([600.0] * 10, index=idx)
        df = pd.DataFrame({"Open": closes, "High": closes + 1, "Low": closes - 1, "Close": closes}, index=idx)
        features = _make_fixed_features(df, entry_idx=1, prior_low=500.0)
        trades, diag = run_squeeze_kdj_capital_constrained_backtest(
            {"2330": df}, {"2330": {}}, idx, 1_000.0, variant="A", max_concurrent_positions=1, top_n=1,
            features_by_code={"2330": features}, max_hold_days=5, return_diagnostics=True,
        )
        assert trades == []
        assert diag["skipped_single_margin_cap"] == 1

    def test_total_margin_cap_counter(self):
        """3檔同一天觸發、max_concurrent_positions=3(整體上限=min(35%x3, 90%)=90%)。把三檔的價格
        調到每檔保證金剛好一樣、各佔起始資金34%(單筆35%上限過得了)，前兩檔進場後已用68%，
        第三檔再加34%=102% > 90% → 被整體保證金上限擋掉。"""
        idx = pd.date_range("2022-01-03", periods=10, freq="B")
        codes = ["1101", "1102", "2330"]
        target_margin = 100_000.0
        price_data, features_by_code = {}, {}
        for code in codes:
            margin_per_price = estimate_margin(code, 1.0, 1)  # 保證金對價格是線性的
            price = target_margin / margin_per_price
            closes = pd.Series([price] * 10, index=idx)
            df = pd.DataFrame({"Open": closes, "High": closes * 1.001, "Low": closes * 0.999,
                               "Close": closes}, index=idx)
            price_data[code] = df
            features_by_code[code] = _make_fixed_features(df, entry_idx=1, prior_low=price * 0.5)
        capital = target_margin / 0.34
        trades, diag = run_squeeze_kdj_capital_constrained_backtest(
            price_data, {c: {} for c in codes}, idx, capital, variant="A", lots=1,
            max_concurrent_positions=3, top_n=3, features_by_code=features_by_code, max_hold_days=5,
            return_diagnostics=True,
        )
        assert len(trades) == 2
        assert diag["skipped_total_margin_cap"] == 1
        assert diag["skipped_single_margin_cap"] == 0


# ============================================================================
# 成交模型 execution_model="limit_1tick"(--squeeze-kdj-fixed新增)：
# 限價 = 觸發K棒收盤 + 1檔，開盤 > 限價不追；成交 = min(開盤+1檔, 限價)；
# 市價型出場(停損/跳空停損/到期強制平倉/移動停利停損/變體A K值停利)多付1檔；變體B固定停利不加滑價。
# ============================================================================
from squeeze_kdj_signal import taiwan_tick_size, VALID_EXECUTION_MODELS
import squeeze_kdj_signal as skd
from mean_reversion_engine import COMMISSION_PER_LOT_PER_LEG


class TestTaiwanTickSize:
    @pytest.mark.parametrize("price, tick", [
        (0.5, 0.01), (9.99, 0.01), (10.0, 0.05), (49.95, 0.05), (50.0, 0.1), (99.9, 0.1),
        (100.0, 0.5), (499.5, 0.5), (500.0, 1.0), (999.0, 1.0), (1000.0, 5.0), (5000.0, 5.0),
    ])
    def test_boundaries(self, price, tick):
        assert taiwan_tick_size(price) == tick

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_price_raises(self, bad):
        with pytest.raises(ValueError):
            taiwan_tick_size(bad)


def _limit_df(n=30, price=100.0):
    """持平100(H=101/L=99 → ATR14=2.0)。EntryFlag放在idx20 → 觸發K棒收盤100、限價100.5，idx21進場。"""
    return _make_flat_df(price, n=n)


def _run_limit(df, features, code="1101", **kwargs):
    params = dict(variant="B", atr_stop_mult=1.0, atr_target_mult=3.0, max_concurrent_positions=1, top_n=1,
                  max_hold_days=30, execution_model="limit_1tick", return_diagnostics=True, lots=2)
    params.update(kwargs)
    return run_squeeze_kdj_capital_constrained_backtest(
        {code: df}, {code: {}}, df.index, 1_000_000, features_by_code={code: features}, **params)


def _expected_pnl(code, e_price, exit_price, lots):
    mult = get_contract_multiplier(code, e_price)
    return (exit_price - e_price) * mult * lots - COMMISSION_PER_LOT_PER_LEG * lots * 2


class TestLimit1TickEntry:
    def test_precompute_trigger_close_is_previous_row_close(self):
        df = _limit_df()
        df.iloc[20, df.columns.get_loc("Close")] = 101.0
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        pre = precompute_squeeze_kdj_backtest_arrays({"1101": df}, {"1101": features}, atr_period=14)
        tc = pre["per_code"]["1101"]["trigger_close"]
        assert tc[21] == 101.0 and tc[20] == 100.0 and np.isnan(tc[0])

    def test_skip_when_open_above_close_plus_one_tick(self):
        df = _limit_df()
        df.iloc[21, df.columns.get_loc("Open")] = 100.6  # 限價100.5，開盤高出 → 不追
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        trades, diag = _run_limit(df, features, max_hold_days=3)
        assert trades == []
        assert diag["skipped_limit_not_filled"] == 1
        assert diag["candidates_total"] == 1
        # 同一份資料用理想成交模型會進場(確認略過真的是限價規則造成的)
        trades_open, diag_open = _run_limit(df, features, execution_model="open", max_hold_days=3)
        assert len(trades_open) == 1 and diag_open["skipped_limit_not_filled"] == 0

    @pytest.mark.parametrize("open_price, expected_fill", [
        (100.0, 100.5),   # min(100+0.5, 100.5)
        (100.2, 100.5),   # min(100.7, 100.5) → 被限價封頂
        (100.5, 100.5),   # 剛好等於限價 → 成交在限價
        (99.95, 100.05),  # 99.95的tick=0.1 → 100.05 < 限價
        (95.0, 95.1),     # 跳空開低：開盤+1檔
    ])
    def test_fill_price_and_b_stop_target_from_fill(self, monkeypatch, open_price, expected_fill):
        df = _limit_df()
        df.iloc[21, df.columns.get_loc("Open")] = open_price
        df.iloc[21, df.columns.get_loc("Low")] = min(99.0, open_price)
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        seen = []

        def _spy(position, row, date, trades, max_hold_days, cooldown_until, slippage_pct=0.0):
            seen.append(dict(position))
            return _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until,
                                   slippage_pct=slippage_pct)
        monkeypatch.setattr(skd, "_process_mr_day", _spy)
        trades, diag = _run_limit(df, features, max_hold_days=3)
        first = seen[0]
        assert first["e_price"] == pytest.approx(expected_fill)
        assert first["stop_price"] == pytest.approx(expected_fill - 1.0 * 2.0)
        assert first["target_price"] == pytest.approx(expected_fill + 3.0 * 2.0)
        assert diag["skipped_limit_not_filled"] == 0
        assert trades[0]["e_price"] == pytest.approx(expected_fill)

    def test_slippage_pct_with_limit_model_raises(self):
        df = _limit_df()
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        with pytest.raises(ValueError):
            _run_limit(df, features, slippage_pct=0.001)

    def test_invalid_execution_model_raises(self):
        df = _limit_df()
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        assert VALID_EXECUTION_MODELS == ("open", "limit_1tick")
        with pytest.raises(ValueError):
            _run_limit(df, features, execution_model="vwap")


class TestLimit1TickExits:
    """進場都是idx21開盤100 → 成交100.5，停損98.5、停利106.5(變體B，ATR=2)。"""

    def _df_features(self):
        df = _limit_df()
        return df, _make_fixed_features(df, entry_idx=20, prior_low=np.nan)

    def test_stop_exit_one_tick_worse_and_pnl_recomputed(self):
        df, features = self._df_features()
        df.iloc[23, df.columns.get_loc("Low")] = 90.0
        trades, _ = _run_limit(df, features)
        assert len(trades) == 1  # 重算時有把原本那筆拿掉，不會重複
        t = trades[0]
        assert t["exit_reason"] == "stop"
        assert t["exit_price"] == pytest.approx(98.5 - 0.1)
        assert t["pnl_ntd"] == pytest.approx(_expected_pnl("1101", 100.5, 98.4, 2))
        # 手續費不變：損益 + 手續費 = 純價差
        mult = get_contract_multiplier("1101", 100.5)
        assert t["pnl_ntd"] + COMMISSION_PER_LOT_PER_LEG * 2 * 2 == pytest.approx((98.4 - 100.5) * mult * 2)
        assert t["return_pct"] == pytest.approx(t["pnl_ntd"] / (100.5 * mult * 2))

    def test_stop_gap_one_tick_below_open(self):
        df, features = self._df_features()
        df.iloc[23, df.columns.get_loc("Open")] = 97.0
        df.iloc[23, df.columns.get_loc("Low")] = 96.0
        trades, _ = _run_limit(df, features)
        assert trades[0]["exit_reason"] == "stop_gap"
        assert trades[0]["exit_price"] == pytest.approx(97.0 - 0.1)

    def test_fixed_target_exit_has_no_slippage(self):
        df, features = self._df_features()
        df.iloc[23, df.columns.get_loc("High")] = 110.0
        trades, _ = _run_limit(df, features)
        assert trades[0]["exit_reason"] == "target"
        assert trades[0]["exit_price"] == pytest.approx(106.5)
        assert trades[0]["pnl_ntd"] == pytest.approx(_expected_pnl("1101", 100.5, 106.5, 2))

    def test_forced_close_one_tick_worse(self):
        df, features = self._df_features()
        trades, _ = _run_limit(df, features, max_hold_days=3)
        t = trades[0]
        assert t["exit_reason"] == "forced_close"
        assert t["exit_date"] == df.index[23]
        assert t["exit_price"] == pytest.approx(100.0 - 0.5)  # 收盤100的tick是0.5

    def test_variant_a_k_exit_one_tick_worse(self):
        idx = pd.date_range("2022-01-03", periods=10, freq="B")
        closes = pd.Series([100.0] * 10, index=idx)
        df = pd.DataFrame({"Open": closes, "High": closes + 1.0, "Low": closes - 20.0, "Close": closes}, index=idx)
        k = pd.Series([50, 50, 70, 85, 90, 75, 60, 60, 60, 60], index=idx, dtype=float)
        features = _make_fixed_features(df, entry_idx=1, prior_low=-1000.0, k_series=k)
        trades, _ = _run_limit(df, features, variant="A")
        t = trades[0]
        assert t["e_price"] == pytest.approx(100.5)
        assert t["exit_reason"] == "target"
        assert t["exit_date"] == idx[5]
        assert t["exit_price"] == pytest.approx(99.5)

    def test_variant_a_stop_one_tick_worse(self):
        df, _ = self._df_features()
        df.iloc[23, df.columns.get_loc("Low")] = 90.0
        features = _make_fixed_features(df, entry_idx=20, prior_low=95.0)
        trades, _ = _run_limit(df, features, variant="A")
        assert trades[0]["exit_reason"] == "stop"
        assert trades[0]["exit_price"] == pytest.approx(95.0 - 0.1)

    def test_trailing_stop_exit_one_tick_worse(self):
        """B_trail 1.5倍：成交100.5 → 初始停損97.5；idx22收104→停損101、idx23收108→停損105；
        idx24盤中殺到90 → 105停損，再多付1檔(105的tick=0.5) → 104.5。"""
        df = _make_trailing_df()
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        trades, _ = _run_limit(df, features, variant="B_trail", atr_stop_mult=1.5, trailing_atr_mult=1.5)
        t = trades[0]
        assert t["e_price"] == pytest.approx(100.5)
        assert t["exit_date"] == df.index[24]
        assert t["exit_reason"] == "stop"
        assert t["exit_price"] == pytest.approx(104.5)

    def test_cooldown_still_applied_after_stop(self):
        """idx23停損 → 冷卻到idx23+10個日曆天；冷卻期內的觸發(idx26進場)被排除，
        冷卻期過後的觸發(idx31進場)正常進場。"""
        df = _limit_df(n=40)
        df.iloc[23, df.columns.get_loc("Low")] = 90.0
        entry_flag = np.zeros(40, dtype=bool)
        entry_flag[[20, 25, 30]] = True
        features = pd.DataFrame({"EntryFlag": entry_flag, "PriorLow": np.nan,
                                 "K": pd.Series(50.0, index=df.index)}, index=df.index)
        cooldown_end = df.index[23] + pd.Timedelta(days=STOP_LOSS_COOLDOWN_DAYS * 2)
        assert df.index[26] < cooldown_end <= df.index[31]
        trades, diag = _run_limit(df, features, max_hold_days=3)
        assert [t["entry_date"] for t in trades] == [df.index[21], df.index[31]]
        assert trades[0]["exit_reason"] == "stop"
        assert diag["candidates_total"] == 2  # 冷卻期內那次在候選之前就被排除

    def test_cooldown_set_exactly_once_per_stop(self, monkeypatch):
        """冷卻期只在原本的出場函式裡設定一次，重算出場價時不會再碰cooldown_until。"""
        df, features = self._df_features()
        df.iloc[23, df.columns.get_loc("Low")] = 90.0
        writes = []

        class _CountingDict(dict):
            def __setitem__(self, key, value):
                writes.append((key, value))
                super().__setitem__(key, value)

        real = skd._process_mr_day
        holder = {}

        def _spy(position, row, date, trades, max_hold_days, cooldown_until, slippage_pct=0.0):
            # 第一次呼叫時把引擎內部的cooldown_until換成會計數的dict(同一個物件持續沿用)
            proxy = holder.setdefault("proxy", _CountingDict(cooldown_until))
            result = real(position, row, date, trades, max_hold_days, proxy, slippage_pct=slippage_pct)
            cooldown_until.update(proxy)
            return result
        monkeypatch.setattr(skd, "_process_mr_day", _spy)
        trades, _ = _run_limit(df, features)
        assert len(trades) == 1
        assert writes == [("1101", df.index[23] + pd.Timedelta(days=STOP_LOSS_COOLDOWN_DAYS * 2))]


class TestExecutionModelOpenMatchesLegacy:
    @pytest.mark.parametrize("scenario", _SNAPSHOT_SCENARIOS[:4])
    def test_explicit_open_model_identical_to_pre_refactor_version(self, scenario):
        seed, variant, mcp, lots, capital, slip, top_n, hold, cal = scenario
        price_data, universe, idx = _make_synthetic_market(seed=seed)
        calendar = {"full": idx, "is": idx[:280], "oos": idx[280:]}[cal]
        features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)
        kwargs = dict(variant=variant, lots=lots, top_n=top_n, max_concurrent_positions=mcp,
                      max_hold_days=hold, slippage_pct=slip, features_by_code=features_by_code)
        legacy = _legacy_capital_constrained_backtest(price_data, universe, calendar, capital, **kwargs)
        new, diag = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, calendar, capital, execution_model="open", return_diagnostics=True, **kwargs)
        assert len(legacy) > 0
        assert new == legacy
        assert diag["skipped_limit_not_filled"] == 0

    def test_limit_model_differs_and_is_never_better_per_trade_on_synthetic_market(self):
        """同一份合成市場：限價模型每一筆跟理想模型「同一天進場的同一檔」相比，進場價只會更高、
        市價出場價只會更低(1檔)，所以這類配對交易的損益不會比較好。"""
        price_data, universe, idx = _make_synthetic_market(seed=0)
        kw = dict(variant="B", lots=1, top_n=3, max_concurrent_positions=3, max_hold_days=20,
                  atr_target_mult=3.0, return_diagnostics=True)
        ideal, _ = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000, **kw)
        lim, diag = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000,
                                                                  execution_model="limit_1tick", **kw)
        assert lim != ideal
        ideal_by_key = {(t["code"], t["entry_date"]): t for t in ideal}
        for t in lim:
            o = ideal_by_key.get((t["code"], t["entry_date"]))
            if o is None:
                continue
            assert t["e_price"] >= o["e_price"] - 1e-9
        assert set(diag) == set(CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS)


# ============================================================================
# --squeeze-kdj-filters：大盤季線(market_ma60)/個股半年線(stock_ma120)/兩者(market_ma60_and_stock_ma120)濾網
# ============================================================================
from squeeze_kdj_signal import (
    squeeze_kdj_entry_filter_flags, squeeze_kdj_entry_filter_allows, _market_ok_by_code,
)


def _flat_with_trigger_close(n, t, close_t, price=100.0, start="2020-01-02"):
    """全程收盤=price(開=price、高低±1)，只有t那天收盤=close_t。"""
    idx = pd.date_range(start, periods=n, freq="B")
    closes = pd.Series(price, index=idx, dtype=float)
    closes.iloc[t] = close_t
    opens = pd.Series(price, index=idx, dtype=float)
    return pd.DataFrame({"Open": opens, "High": np.maximum(opens, closes) + 1.0,
                         "Low": np.minimum(opens, closes) - 1.0, "Close": closes}, index=idx)


def _run_filter(df, features, entry_filter, market_series=None, **kw):
    return run_squeeze_kdj_capital_constrained_backtest(
        {"1101": df}, {"1101": {}}, df.index, 1_000_000, variant="A", max_concurrent_positions=1, top_n=1,
        features_by_code={"1101": features}, max_hold_days=3, entry_filter=entry_filter,
        market_series=market_series, return_diagnostics=True, **kw)


class TestMarketMa60Filter:
    N, T = 100, 80  # 觸發K棒t=80 → t+1=81開盤進場

    def _market(self, idx, value_t, value_t1, drop_t=False, drop_values_at=None):
        m = pd.Series(100.0, index=idx)
        m.iloc[self.T] = value_t
        m.iloc[self.T + 1] = value_t1
        if drop_t:
            m = m.drop(idx[self.T])
        return m

    def test_blocks_when_index_below_ma60_on_trigger_day_even_if_next_day_soars(self):
        df = _flat_with_trigger_close(self.N, self.T, 100.0)
        features = _make_fixed_features(df, entry_idx=self.T, prior_low=1.0)
        market = self._market(df.index, value_t=90.0, value_t1=500.0)
        trades, diag = _run_filter(df, features, "market_ma60", market)
        assert trades == []
        assert diag["skipped_entry_filter"] == 1 and diag["skipped_filter_market"] == 1
        assert diag["skipped_filter_stock_trend"] == 0

    def test_allows_when_index_above_ma60_on_trigger_day_even_if_next_day_crashes(self):
        df = _flat_with_trigger_close(self.N, self.T, 100.0)
        features = _make_fixed_features(df, entry_idx=self.T, prior_low=1.0)
        market = self._market(df.index, value_t=110.0, value_t1=1.0)
        trades, diag = _run_filter(df, features, "market_ma60", market)
        assert len(trades) == 1 and trades[0]["entry_date"] == df.index[self.T + 1]
        assert diag["skipped_entry_filter"] == 0

    def test_index_values_after_trigger_day_never_change_decision(self):
        df = _flat_with_trigger_close(self.N, self.T, 100.0)
        features = _make_fixed_features(df, entry_idx=self.T, prior_low=1.0)
        base = self._market(df.index, value_t=110.0, value_t1=110.0)
        for later in (1.0, 1e6):
            m = base.copy()
            m.iloc[self.T + 1:] = later
            trades, _ = _run_filter(df, features, "market_ma60", m)
            assert len(trades) == 1

    def test_index_holiday_on_trigger_day_uses_last_bar_before(self):
        df = _flat_with_trigger_close(self.N, self.T, 100.0)
        features = _make_fixed_features(df, entry_idx=self.T, prior_low=1.0)
        # 指數在t沒有K棒；t-1在季線上、t+1崩跌 → 用t-1 → 放行
        m = pd.Series(100.0, index=df.index)
        m.iloc[self.T - 1] = 110.0
        m.iloc[self.T + 1] = 1.0
        m = m.drop(df.index[self.T])
        trades, _ = _run_filter(df, features, "market_ma60", m)
        assert len(trades) == 1
        # 反過來：t-1在季線下、t沒有K棒、t+1暴漲 → 用t-1 → 擋掉
        m2 = pd.Series(100.0, index=df.index)
        m2.iloc[self.T - 1] = 90.0
        m2.iloc[self.T + 1] = 500.0
        m2 = m2.drop(df.index[self.T])
        trades2, diag2 = _run_filter(df, features, "market_ma60", m2)
        assert trades2 == [] and diag2["skipped_filter_market"] == 1

    def test_index_ma60_warmup_blocks(self):
        df = _flat_with_trigger_close(self.N, self.T, 100.0)
        features = _make_fixed_features(df, entry_idx=self.T, prior_low=1.0)
        m = pd.Series(np.linspace(50, 200, 40), index=df.index[self.T - 39:self.T + 1])  # 只有40根，季線算不出來
        trades, _ = _run_filter(df, features, "market_ma60", m)
        assert trades == []

    def test_nan_index_rows_are_dropped_not_poisoning_the_ma(self):
        df = _flat_with_trigger_close(self.N, self.T, 100.0)
        features = _make_fixed_features(df, entry_idx=self.T, prior_low=1.0)
        m = self._market(df.index, value_t=110.0, value_t1=110.0)
        m.iloc[self.T - 5] = np.nan  # 不丟掉的話，含這列的60日均線全部是NaN → 會被誤擋
        trades, _ = _run_filter(df, features, "market_ma60", m)
        assert len(trades) == 1

    def test_requires_market_series(self):
        df = _flat_with_trigger_close(self.N, self.T, 100.0)
        features = _make_fixed_features(df, entry_idx=self.T, prior_low=1.0)
        for f in ("market_ma60", "market_ma60_and_stock_ma120"):
            with pytest.raises(ValueError):
                _run_filter(df, features, f, None)
        # 個股濾網不需要大盤
        _run_filter(df, features, "stock_ma120", None)


class TestStockMa120Filter:
    N = 200

    def test_blocks_below_and_allows_above_own_ma120(self):
        t = 150
        for close_t, expect in ((110.0, 1), (90.0, 0)):
            df = _flat_with_trigger_close(self.N, t, close_t)
            features = _make_fixed_features(df, entry_idx=t, prior_low=1.0)
            trades, diag = _run_filter(df, features, "stock_ma120")
            assert len(trades) == expect
            assert diag["skipped_filter_stock_trend"] == 1 - expect
            assert diag["skipped_filter_market"] == 0

    def test_prices_after_trigger_day_never_change_decision(self):
        t = 150
        for close_t, expect in ((110.0, 1), (90.0, 0)):
            for later in (1.0, 1000.0):
                df = _flat_with_trigger_close(self.N, t, close_t)
                df.iloc[t + 1:, df.columns.get_loc("Close")] = later
                df["High"] = df[["Open", "Close"]].max(axis=1) + 1.0
                df["Low"] = df[["Open", "Close"]].min(axis=1) - 0.5
                features = _make_fixed_features(df, entry_idx=t, prior_low=0.01)
                trades, _ = _run_filter(df, features, "stock_ma120")
                assert len(trades) == expect

    def test_ma120_warmup_blocks_then_first_full_window_allows(self):
        df = _flat_with_trigger_close(self.N, 118, 110.0)  # t=118：只有119根，120日均線算不出來
        features = _make_fixed_features(df, entry_idx=118, prior_low=1.0)
        trades, _ = _run_filter(df, features, "stock_ma120")
        assert trades == []
        df = _flat_with_trigger_close(self.N, 119, 110.0)  # t=119：剛好120根
        features = _make_fixed_features(df, entry_idx=119, prior_low=1.0)
        trades, _ = _run_filter(df, features, "stock_ma120")
        assert len(trades) == 1


class TestBothFilter:
    def test_requires_both_conditions(self):
        t = 150
        cases = [  # (個股t收盤, 大盤t值, 預期筆數, 大盤細項, 個股細項)
            (110.0, 110.0, 1, 0, 0),
            (110.0, 90.0, 0, 1, 0),
            (90.0, 110.0, 0, 0, 1),
            (90.0, 90.0, 0, 1, 1),  # 兩個都不成立：細項各記1次，合計只記1次
        ]
        for close_t, mkt_t, expect, n_mkt, n_stock in cases:
            df = _flat_with_trigger_close(200, t, close_t)
            features = _make_fixed_features(df, entry_idx=t, prior_low=1.0)
            m = pd.Series(100.0, index=df.index)
            m.iloc[t] = mkt_t
            trades, diag = _run_filter(df, features, "market_ma60_and_stock_ma120", m)
            assert len(trades) == expect
            assert diag["skipped_entry_filter"] == 1 - expect
            assert diag["skipped_filter_market"] == n_mkt and diag["skipped_filter_stock_trend"] == n_stock


class TestNoFilterUnaffectedByMarketSeries:
    @pytest.mark.parametrize("scenario", _SNAPSHOT_SCENARIOS[:4])
    def test_filter_none_with_market_series_identical_to_legacy(self, scenario):
        seed, variant, mcp, lots, capital, slip, top_n, hold, cal = scenario
        price_data, universe, idx = _make_synthetic_market(seed=seed)
        calendar = {"full": idx, "is": idx[:280], "oos": idx[280:]}[cal]
        features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)
        kwargs = dict(variant=variant, lots=lots, top_n=top_n, max_concurrent_positions=mcp,
                      max_hold_days=hold, slippage_pct=slip, features_by_code=features_by_code)
        legacy = _legacy_capital_constrained_backtest(price_data, universe, calendar, capital, **kwargs)
        market = pd.Series(np.random.default_rng(9).lognormal(0, 0.1, len(idx)).cumprod() * 10000, index=idx)
        new, diag = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, calendar, capital, entry_filter=None, market_series=market,
            return_diagnostics=True, **kwargs)
        assert len(legacy) > 0
        assert new == legacy
        assert diag["skipped_entry_filter"] == diag["skipped_filter_market"] == diag["skipped_filter_stock_trend"] == 0


class TestPureFilterFunctionMatchesBacktestArrays:
    def test_scanner_function_agrees_with_backtest_alignment(self):
        """每日掃描要用的純函式(squeeze_kdj_entry_filter_flags)跟回測裡向量化、shift(1)對齊的版本，
        對每一個「觸發K棒t → 進場列t+1」的判斷必須完全一致(含指數缺日、個股NaN列)。"""
        price_data, universe, idx = _make_synthetic_market(n_stocks=4, n_days=300, seed=11)
        code0 = list(price_data)[0]
        price_data[code0].iloc[150, price_data[code0].columns.get_loc("Close")] = np.nan  # 個股NaN列
        rng = np.random.default_rng(1)
        market = pd.Series(10000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
        market = market.drop(idx[rng.choice(np.arange(70, 290), size=15, replace=False)])  # 指數缺日
        market.iloc[100] = np.nan
        features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)
        pre = precompute_squeeze_kdj_backtest_arrays(price_data, features_by_code)
        market_ok = _market_ok_by_code(pre["per_code"], market)
        checked = 0
        for code, arrs in pre["per_code"].items():
            dates = arrs["dates"]
            for i in list(range(1, len(dates), 7)) + [151, 152]:
                t = dates[i - 1]
                flags = squeeze_kdj_entry_filter_flags(price_data[code]["Close"], market, t)
                assert flags["market_ok"] == bool(market_ok[code][i]), (code, t)
                assert flags["stock_trend_ok"] == bool(arrs["above_ma120"][i]), (code, t)
                checked += 1
        assert checked > 100
        # 兩種結果都有出現，比對才有意義
        assert any(market_ok[code0]) and not all(market_ok[code0])
        assert any(pre["per_code"][code0]["above_ma120"]) and not all(pre["per_code"][code0]["above_ma120"][130:])

    def test_pure_function_ignores_data_after_trigger_date(self):
        df = _flat_with_trigger_close(200, 150, 110.0)
        market = pd.Series(100.0, index=df.index)
        market.iloc[150] = 110.0
        t = df.index[150]
        base = squeeze_kdj_entry_filter_flags(df["Close"], market, t)
        df2, m2 = df.copy(), market.copy()
        df2.iloc[151:, df2.columns.get_loc("Close")] = 1.0
        m2.iloc[151:] = 1.0
        assert squeeze_kdj_entry_filter_flags(df2["Close"], m2, t) == base == {"market_ok": True, "stock_trend_ok": True}

    def test_allows_wrapper(self):
        df = _flat_with_trigger_close(200, 150, 110.0)
        market = pd.Series(100.0, index=df.index)
        market.iloc[150] = 90.0
        t = df.index[150]
        assert squeeze_kdj_entry_filter_allows(None, df["Close"], None, t) is True
        assert squeeze_kdj_entry_filter_allows("stock_ma120", df["Close"], None, t) is True
        assert squeeze_kdj_entry_filter_allows("market_ma60", df["Close"], market, t) is False
        assert squeeze_kdj_entry_filter_allows("market_ma60", df["Close"], None, t) is False
        assert squeeze_kdj_entry_filter_allows("market_ma60_and_stock_ma120", df["Close"], market, t) is False
        with pytest.raises(ValueError):
            squeeze_kdj_entry_filter_allows("above_ma60", df["Close"], market, t)


# ============================================================================
# 自訂交易成本(commission_per_lot_side/futures_tax_rate，--squeeze-kdj-stops新增)
# ============================================================================
from squeeze_kdj_signal import apply_squeeze_kdj_trade_costs

MINI_CODE = "1477"      # has_mini=True → 合約乘數100
STANDARD_CODE = "1101"  # has_mini=False → 合約乘數2000


class TestCustomTradeCostsDefaultsUnchanged:
    @pytest.mark.parametrize("scenario", _SNAPSHOT_SCENARIOS[:4])
    def test_explicit_defaults_identical_to_legacy_and_no_new_fields(self, scenario):
        seed, variant, mcp, lots, capital, slip, top_n, hold, cal = scenario
        price_data, universe, idx = _make_synthetic_market(seed=seed)
        calendar = {"full": idx, "is": idx[:280], "oos": idx[280:]}[cal]
        features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)
        kwargs = dict(variant=variant, lots=lots, top_n=top_n, max_concurrent_positions=mcp,
                      max_hold_days=hold, slippage_pct=slip, features_by_code=features_by_code)
        legacy = _legacy_capital_constrained_backtest(price_data, universe, calendar, capital, **kwargs)
        new = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, calendar, capital, commission_per_lot_side=None, futures_tax_rate=0.0, **kwargs)
        assert len(legacy) > 0 and new == legacy
        assert all("commission_ntd" not in t and "tax_ntd" not in t for t in new)

    def test_old_cost_given_explicitly_matches_default_pnl_and_adds_fields(self):
        price_data, universe, idx = _make_synthetic_market(seed=0)
        kw = dict(variant="B", lots=2, top_n=3, max_concurrent_positions=3, execution_model="limit_1tick")
        base = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000, **kw)
        explicit = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, idx, 1_000_000, commission_per_lot_side=200, futures_tax_rate=0.0, **kw)
        assert len(base) > 0 and len(explicit) == len(base)
        for a, b in zip(base, explicit):
            assert b["pnl_ntd"] == pytest.approx(a["pnl_ntd"], abs=1e-9)
            assert b["return_pct"] == pytest.approx(a["return_pct"], abs=1e-12)
            assert b["commission_ntd"] == 200 * 2 * 2 and b["tax_ntd"] == 0.0
            assert {k: v for k, v in b.items() if k not in ("commission_ntd", "tax_ntd", "pnl_ntd", "return_pct")} == \
                {k: v for k, v in a.items() if k not in ("pnl_ntd", "return_pct")}

    def test_costs_never_change_which_trades_happen(self):
        price_data, universe, idx = _make_synthetic_market(seed=1)
        kw = dict(variant="B", lots=1, top_n=3, max_concurrent_positions=2, execution_model="limit_1tick")
        a = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000, **kw)
        b = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000,
                                                         commission_per_lot_side=50, futures_tax_rate=0.00002, **kw)
        key = lambda t: (t["code"], t["entry_date"], t["exit_date"], t["e_price"], t["exit_price"], t["exit_reason"])
        assert [key(t) for t in a] == [key(t) for t in b] and len(a) > 0
        for x, y in zip(a, b):
            # 舊：-200x2；新：-50x2 - 稅 → 差 = 300 - 稅
            assert y["pnl_ntd"] - x["pnl_ntd"] == pytest.approx(300 - y["tax_ntd"])

    @pytest.mark.parametrize("bad", [dict(commission_per_lot_side=-1), dict(futures_tax_rate=-0.1),
                                     dict(commission_per_lot_side=float("nan"))])
    def test_invalid_cost_params_raise(self, bad):
        df = _limit_df()
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        with pytest.raises(ValueError):
            _run_limit(df, features, **bad)


class TestCustomTradeCostsHandComputed:
    """進場都是idx21開盤100 → 限價模型成交100.5，停損98.5、停利106.5(變體B，ATR=2)；2口。
    手續費50/口/單邊 → 50x2口x2邊=200；期交稅 = 0.00002 x (進場價+出場價) x 乘數 x 2口。"""
    COSTS = dict(commission_per_lot_side=50, futures_tax_rate=0.00002)

    def _df_features(self):
        df = _limit_df()
        return df, _make_fixed_features(df, entry_idx=20, prior_low=np.nan)

    def test_mini_stop_exit_via_limit_reclose_path(self):
        df, features = self._df_features()
        df.iloc[23, df.columns.get_loc("Low")] = 90.0
        trades, _ = _run_limit(df, features, code=MINI_CODE, **self.COSTS)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "stop" and t["exit_price"] == pytest.approx(98.4)
        # 價差 (98.4-100.5)x100x2 = -420；手續費200；稅 0.00002x(100.5+98.4)x100x2 = 0.7956
        assert t["commission_ntd"] == pytest.approx(200.0)
        assert t["tax_ntd"] == pytest.approx(0.7956)
        assert t["pnl_ntd"] == pytest.approx(-620.7956)
        assert t["return_pct"] == pytest.approx(-620.7956 / (100.5 * 100 * 2))

    def test_mini_target_exit(self):
        df, features = self._df_features()
        df.iloc[23, df.columns.get_loc("High")] = 110.0
        trades, _ = _run_limit(df, features, code=MINI_CODE, **self.COSTS)
        t = trades[0]
        assert t["exit_reason"] == "target" and t["exit_price"] == pytest.approx(106.5)
        # 價差 6x100x2 = 1200；稅 0.00002x207x200 = 0.828
        assert t["pnl_ntd"] == pytest.approx(1200 - 200 - 0.828)
        assert t["return_pct"] == pytest.approx((1200 - 200 - 0.828) / 20100)

    def test_standard_forced_close(self):
        df, features = self._df_features()
        trades, _ = _run_limit(df, features, code=STANDARD_CODE, max_hold_days=3, **self.COSTS)
        t = trades[0]
        assert t["exit_reason"] == "forced_close" and t["exit_price"] == pytest.approx(99.5)
        # 價差 (99.5-100.5)x2000x2 = -4000；稅 0.00002x200x2000x2 = 16
        assert t["tax_ntd"] == pytest.approx(16.0)
        assert t["pnl_ntd"] == pytest.approx(-4000 - 200 - 16)
        assert t["return_pct"] == pytest.approx(-4216 / (100.5 * 2000 * 2))

    def test_standard_stop_gap_via_reclose(self):
        df, features = self._df_features()
        df.iloc[23, df.columns.get_loc("Open")] = 97.0
        df.iloc[23, df.columns.get_loc("Low")] = 96.0
        trades, _ = _run_limit(df, features, code=STANDARD_CODE, **self.COSTS)
        t = trades[0]
        assert t["exit_reason"] == "stop_gap" and t["exit_price"] == pytest.approx(96.9)
        price = (96.9 - 100.5) * 2000 * 2
        tax = 0.00002 * (100.5 + 96.9) * 2000 * 2
        assert t["pnl_ntd"] == pytest.approx(price - 200 - tax)

    def test_open_model_stop_without_reclose_standard(self):
        df, features = self._df_features()
        df.iloc[23, df.columns.get_loc("Low")] = 90.0
        trades, _ = _run_limit(df, features, code=STANDARD_CODE, execution_model="open", lots=1,
                               commission_per_lot_side=75, futures_tax_rate=0.0001)
        t = trades[0]
        assert t["e_price"] == pytest.approx(100.0) and t["exit_reason"] == "stop"
        assert t["exit_price"] == pytest.approx(98.0)
        price = (98.0 - 100.0) * 2000
        tax = 0.0001 * (100.0 + 98.0) * 2000
        assert t["commission_ntd"] == pytest.approx(150.0)
        assert t["pnl_ntd"] == pytest.approx(price - 150 - tax)

    def test_tax_only_keeps_old_commission(self):
        df, features = self._df_features()
        df.iloc[23, df.columns.get_loc("High")] = 110.0
        trades, _ = _run_limit(df, features, code=MINI_CODE, futures_tax_rate=0.00002)
        t = trades[0]
        assert t["commission_ntd"] == COMMISSION_PER_LOT_PER_LEG * 2 * 2
        assert t["pnl_ntd"] == pytest.approx(1200 - 800 - 0.828)

    def test_short_side_formula(self):
        t = {"code": MINI_CODE, "side": "short", "lots": 1, "e_price": 100.0, "exit_price": 90.0,
             "pnl_ntd": None, "return_pct": None}
        apply_squeeze_kdj_trade_costs(t, 50, 0.00002)
        assert t["pnl_ntd"] == pytest.approx(10 * 100 - 100 - 0.00002 * 190 * 100)

    def test_adjusted_pnl_visible_inside_loop(self, monkeypatch):
        """出場當下就換成新成本：之後每一天的逐日處理看到的trades(=risk_pct_per_trade算權益用的那份)
        已經是新成本的損益，不是事後才改。"""
        df = _limit_df(n=40)
        df.iloc[23, df.columns.get_loc("High")] = 110.0   # 第1筆在idx23停利
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        features.iloc[30, features.columns.get_loc("EntryFlag")] = True  # 第2筆idx31進場
        seen = []

        def _spy(position, row, date, trades, max_hold_days, cooldown_until, slippage_pct=0.0):
            seen.append([t["pnl_ntd"] for t in trades])
            return _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until,
                                   slippage_pct=slippage_pct)
        monkeypatch.setattr(skd, "_process_mr_day", _spy)
        trades, _ = _run_limit(df, features, code=MINI_CODE, max_hold_days=3, **self.COSTS)
        assert len(trades) == 2
        assert seen[-1] == [pytest.approx(1200 - 200 - 0.828)]


# ============================================================================
# --squeeze-kdj-exits：保本停損(breakeven_trigger_atr) / 進場星期濾網(skip_entry_weekdays)
# ============================================================================
from squeeze_kdj_signal import BREAKEVEN_STOP_REASON, BREAKEVEN_STOP_GAP_REASON


def _set(df, i, **cols):
    for c, v in cols.items():
        df.iloc[i, df.columns.get_loc(c)] = v


class TestBreakevenStop:
    """持平100(ATR=2)，idx21開盤100 → 限價1檔成交100.5；停損1.5倍 → 97.5、停利3.0倍 → 106.5；
    保本觸發 = 100.5 + 1.5x2 = 103.5(收盤要 >= 103.5)。迷你契約(乘數100)、1口、手續費50/口/單邊、期交稅0.002%/邊。"""
    COSTS = dict(commission_per_lot_side=50, futures_tax_rate=0.00002)
    KW = dict(atr_stop_mult=1.5, atr_target_mult=3.0, lots=1, max_hold_days=30)

    def _df_features(self, n=40):
        df = _limit_df(n=n)
        return df, _make_fixed_features(df, entry_idx=20, prior_low=np.nan)

    def _spy_stops(self, monkeypatch):
        seen = []

        def _spy(position, row, date, trades, max_hold_days, cooldown_until, slippage_pct=0.0):
            seen.append((date, position["stop_price"]))
            return _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until,
                                   slippage_pct=slippage_pct)
        monkeypatch.setattr(skd, "_process_mr_day", _spy)
        return seen

    def test_stop_moves_to_entry_only_from_day_after_trigger_and_hand_computed_pnl(self, monkeypatch):
        df, features = self._df_features()
        _set(df, 22, Close=104.0, High=105.0)          # idx22收盤104 >= 103.5 → 觸發；當天Low=99已低於100.5但停損還是97.5
        _set(df, 23, Open=103.0, High=104.0, Low=100.0, Close=101.0)  # idx23盤中碰到100.5 → 保本停損
        seen = self._spy_stops(monkeypatch)
        trades, _ = _run_limit(df, features, code=MINI_CODE, breakeven_trigger_atr=1.5, **self.KW, **self.COSTS)
        assert len(trades) == 1
        t = trades[0]
        assert [s for _, s in seen] == [pytest.approx(97.5), pytest.approx(97.5), pytest.approx(100.5)]
        assert t["exit_date"] == df.index[23] and t["exit_reason"] == BREAKEVEN_STOP_REASON
        assert t["e_price"] == pytest.approx(100.5)
        assert t["exit_price"] == pytest.approx(100.0)  # 保本停損是市價型出場：100.5 - 1檔(0.5)
        # 價差 (100.0-100.5)x100x1 = -50；手續費50x1x2 = 100；稅 0.00002x(100.5+100.0)x100 = 0.401
        assert t["commission_ntd"] == pytest.approx(100.0) and t["tax_ntd"] == pytest.approx(0.401)
        assert t["pnl_ntd"] == pytest.approx(-150.401)
        assert t["breakeven_triggered"] is True
        # 同一份資料不保本：idx23的Low=100沒碰到97.5，不會出場
        no_be, _ = _run_limit(df, features, code=MINI_CODE, **{**self.KW, "max_hold_days": 10}, **self.COSTS)
        assert len(no_be) == 1 and no_be[0]["exit_reason"] == "forced_close" and no_be[0]["exit_date"] == df.index[30]
        assert "breakeven_triggered" not in no_be[0]

    def test_breakeven_gap_exit(self):
        df, features = self._df_features()
        _set(df, 22, Close=104.0, High=105.0)
        _set(df, 23, Open=100.0, High=101.0, Low=99.0, Close=100.0)  # 開盤就低於100.5
        trades, _ = _run_limit(df, features, code=MINI_CODE, breakeven_trigger_atr=1.5, **self.KW, **self.COSTS)
        t = trades[0]
        assert t["exit_reason"] == BREAKEVEN_STOP_GAP_REASON
        assert t["exit_price"] == pytest.approx(99.5)  # 開盤100 - 1檔(0.5)
        assert t["pnl_ntd"] == pytest.approx((99.5 - 100.5) * 100 - 100 - 0.00002 * (100.5 + 99.5) * 100)

    def test_trigger_on_entry_day_close(self, monkeypatch):
        df, features = self._df_features()
        _set(df, 21, Close=103.5, High=104.0)  # 進場當天收盤剛好等於觸發價(>=)
        _set(df, 22, Open=101.0)                 # 隔天開盤在100.5之上、盤中Low=99碰到
        seen = self._spy_stops(monkeypatch)
        trades, _ = _run_limit(df, features, code=MINI_CODE, breakeven_trigger_atr=1.5, **self.KW)
        # idx22 Low=99 <= 100.5 → 隔天就保本停損
        assert [s for _, s in seen] == [pytest.approx(97.5), pytest.approx(100.5)]
        assert trades[0]["exit_reason"] == BREAKEVEN_STOP_REASON and trades[0]["exit_date"] == df.index[22]

    def test_never_moves_down_after_trigger(self, monkeypatch):
        df, features = self._df_features()
        _set(df, 22, Close=104.0, High=105.0)
        for i in range(23, 27):  # 之後收盤都回到觸發價以下，但盤中都在100.5之上
            _set(df, i, Open=102.0, High=103.0, Low=101.0, Close=102.0)
        seen = self._spy_stops(monkeypatch)
        trades, _ = _run_limit(df, features, code=MINI_CODE, breakeven_trigger_atr=1.5,
                               **{**self.KW, "max_hold_days": 6})
        stops = [s for _, s in seen]
        assert stops[:2] == [pytest.approx(97.5), pytest.approx(97.5)]
        assert all(s == pytest.approx(100.5) for s in stops[2:]) and len(stops) == 6
        assert trades[0]["exit_reason"] == "forced_close" and trades[0]["breakeven_triggered"] is True

    def test_not_triggered_when_close_stays_below_threshold(self, monkeypatch):
        df, features = self._df_features()
        _set(df, 22, Close=103.45, High=106.0)  # 收盤差一點點(盤中最高超過觸發價也不算)
        _set(df, 23, Open=102.0, High=103.0, Low=100.0, Close=101.0)
        seen = self._spy_stops(monkeypatch)
        trades, _ = _run_limit(df, features, code=MINI_CODE, breakeven_trigger_atr=1.5,
                               **{**self.KW, "max_hold_days": 4})
        assert all(s == pytest.approx(97.5) for _, s in seen)
        t = trades[0]
        assert t["exit_reason"] == "forced_close" and t["breakeven_triggered"] is False
        ref, _ = _run_limit(df, features, code=MINI_CODE, **{**self.KW, "max_hold_days": 4})
        assert {k: v for k, v in t.items() if k != "breakeven_triggered"} == ref[0]

    def test_existing_higher_trailing_stop_is_kept_and_labelled_plain_stop(self, monkeypatch):
        df, features = self._df_features()
        _set(df, 22, Close=110.0, High=110.5)   # B_trail：停損 = 110 - 1.0x2 = 108 > 進場價
        _set(df, 23, Open=109.0, High=109.5, Low=107.0, Close=108.0)
        seen = self._spy_stops(monkeypatch)
        trades, _ = _run_limit(df, features, code=MINI_CODE, variant="B_trail", trailing_atr_mult=1.0,
                               breakeven_trigger_atr=1.5, **self.KW)
        assert seen[-1][1] == pytest.approx(108.0)  # 沒有被保本拉回100.5
        assert trades[0]["exit_reason"] == "stop" and trades[0]["breakeven_triggered"] is True

    def test_cooldown_after_breakeven_stop(self):
        df, features = self._df_features()
        _set(df, 22, Close=104.0, High=105.0)
        _set(df, 23, Open=103.0, High=104.0, Low=100.0, Close=101.0)
        features.iloc[24, features.columns.get_loc("EntryFlag")] = True  # idx25想再進場 → 冷卻期內
        trades, _ = _run_limit(df, features, code=MINI_CODE, breakeven_trigger_atr=1.5, **self.KW)
        assert len(trades) == 1

    @pytest.mark.parametrize("scenario", _SNAPSHOT_SCENARIOS[:4])
    def test_defaults_identical_to_legacy(self, scenario):
        seed, variant, mcp, lots, capital, slip, top_n, hold, cal = scenario
        price_data, universe, idx = _make_synthetic_market(seed=seed)
        calendar = {"full": idx, "is": idx[:280], "oos": idx[280:]}[cal]
        features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)
        kwargs = dict(variant=variant, lots=lots, top_n=top_n, max_concurrent_positions=mcp,
                      max_hold_days=hold, slippage_pct=slip, features_by_code=features_by_code)
        legacy = _legacy_capital_constrained_backtest(price_data, universe, calendar, capital, **kwargs)
        new, diag = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, calendar, capital, breakeven_trigger_atr=None, skip_entry_weekdays=(),
            return_diagnostics=True, **kwargs)
        assert len(legacy) > 0 and new == legacy
        assert all("breakeven_triggered" not in t for t in new) and diag["skipped_weekday"] == 0

    def test_synthetic_market_breakeven_changes_only_some_exits(self):
        price_data, universe, idx = _make_synthetic_market(seed=0)
        kw = dict(variant="B", lots=1, top_n=3, max_concurrent_positions=2, max_hold_days=20,
                  atr_stop_mult=1.5, atr_target_mult=3.0, execution_model="limit_1tick")
        be = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000,
                                                          breakeven_trigger_atr=1.5, **kw)
        assert any(t["exit_reason"] in (BREAKEVEN_STOP_REASON, BREAKEVEN_STOP_GAP_REASON) for t in be)
        for t in be:
            if t["exit_reason"] in (BREAKEVEN_STOP_REASON, BREAKEVEN_STOP_GAP_REASON):
                assert t["breakeven_triggered"] and t["exit_price"] < t["e_price"]

    @pytest.mark.parametrize("bad", [dict(breakeven_trigger_atr=1.5, variant="A"),
                                     dict(breakeven_trigger_atr=-1.0), dict(breakeven_trigger_atr=float("nan")),
                                     dict(skip_entry_weekdays=(7,)), dict(skip_entry_weekdays=("Fri",)),
                                     dict(skip_entry_weekdays=(True,))])
    def test_invalid_params_raise(self, bad):
        df, features = self._df_features()
        with pytest.raises(ValueError):
            _run_limit(df, features, **bad)


class TestSkipEntryWeekdays:
    """idx24 = 2022-02-04(週五)、idx25 = 2022-02-07(週一)。A檔在idx23觸發(週五進場)，B檔在idx24觸發(週一進場)。"""

    def _market(self):
        df_a, df_b = _limit_df(n=45), _limit_df(n=45)
        feats = {"1101": _make_fixed_features(df_a, entry_idx=23, prior_low=np.nan),
                 "1102": _make_fixed_features(df_b, entry_idx=24, prior_low=np.nan)}
        assert df_a.index[24].weekday() == 4 and df_a.index[25].weekday() == 0
        return {"1101": df_a, "1102": df_b}, feats

    def _run(self, **kw):
        price_data, feats = self._market()
        idx = price_data["1101"].index
        return run_squeeze_kdj_capital_constrained_backtest(
            price_data, {c: {} for c in price_data}, idx, 1_000_000, variant="B", atr_stop_mult=1.5,
            atr_target_mult=3.0, max_concurrent_positions=1, top_n=1, max_hold_days=10, lots=1,
            execution_model="limit_1tick", features_by_code=feats, return_diagnostics=True, **kw)

    def test_friday_entry_skipped_and_slot_goes_to_next_candidate(self):
        base, diag0 = self._run()
        assert [(t["code"], t["entry_date"].weekday()) for t in base] == [("1101", 4)]
        assert diag0["skipped_no_slot"] == 1 and diag0["skipped_weekday"] == 0  # 週一那檔沒名額
        skip, diag = self._run(skip_entry_weekdays=(4,))
        assert [(t["code"], t["entry_date"].weekday()) for t in skip] == [("1102", 0)]
        assert diag["skipped_weekday"] == 1 and diag["skipped_no_slot"] == 0
        assert diag["candidates_total"] == diag0["candidates_total"] == 2

    def test_other_days_unaffected(self):
        base, _ = self._run()
        for wds in [(0,), (1, 2, 3)]:
            # 週五那筆照樣進場；週一的候選本來就因為沒名額進不去 → 交易完全相同
            got, diag = self._run(skip_entry_weekdays=wds)
            assert got == base
        # 星期濾網排在名額檢查之前：週一的候選改記在skipped_weekday(每個候選只記一個原因)
        _, diag = self._run(skip_entry_weekdays=(0,))
        assert diag["skipped_weekday"] == 1 and diag["skipped_no_slot"] == 0

    def test_synthetic_market_no_friday_entries(self):
        price_data, universe, idx = _make_synthetic_market(seed=1)
        kw = dict(variant="B", lots=1, top_n=3, max_concurrent_positions=2, max_hold_days=20,
                  atr_stop_mult=1.5, atr_target_mult=3.0, execution_model="limit_1tick", return_diagnostics=True)
        base, d0 = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000, **kw)
        got, d = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000,
                                                              skip_entry_weekdays=(4,), **kw)
        assert any(t["entry_date"].weekday() == 4 for t in base)
        assert all(t["entry_date"].weekday() != 4 for t in got) and d["skipped_weekday"] > 0
        assert d["candidates_total"] >= d["skipped_weekday"]


# ============================================================================
# --squeeze-kdj-execution新增的成交模型：limit_ticks(收盤+N檔)/limit_pct(收盤x(1+p)往下取檔)/market_open(開盤+1檔)
# ============================================================================
from squeeze_kdj_signal import (
    taiwan_add_ticks, taiwan_round_down_to_tick, squeeze_kdj_entry_limit_price,
    EXTENDED_EXECUTION_MODELS, ALL_EXECUTION_MODELS,
)

_GAP_KEYS = ("entry_trigger_close", "entry_open", "entry_gap_pct")


class TestTickWalkingAndRounding:
    @pytest.mark.parametrize("price, n, expected", [
        (49.95, 1, 50.0),
        (49.95, 2, 50.1),     # 49.95 →(0.05) 50.0 →(0.1) 50.1，不是49.95+2x0.05=50.05
        (49.9, 3, 50.1),      # 49.9 → 49.95 → 50.0 → 50.1
        (99.9, 2, 100.5),     # 99.9 →(0.1) 100.0 →(0.5) 100.5
        (9.99, 2, 10.05),     # 9.99 →(0.01) 10.0 →(0.05) 10.05
        (499.5, 3, 502.0),    # 499.5 → 500 → 501 → 502
        (995.0, 6, 1005.0),   # 995..999 → 1000 → 1005
        (100.0, 2, 101.0),
    ])
    def test_add_ticks_walks_across_boundaries(self, price, n, expected):
        assert taiwan_add_ticks(price, n) == pytest.approx(expected, abs=1e-9)

    def test_add_one_tick_is_bit_identical_to_limit_1tick_formula(self):
        for p in (49.95, 49.97, 99.9, 100.0, 123.45, 999.0, 10.0, 9.999, 37.123456):
            assert taiwan_add_ticks(p, 1) == p + taiwan_tick_size(p)

    @pytest.mark.parametrize("bad", [0, -1, 1.0, True, None])
    def test_add_ticks_rejects_bad_n(self, bad):
        with pytest.raises(ValueError):
            taiwan_add_ticks(100.0, bad)

    @pytest.mark.parametrize("price, expected", [
        (100.0 * 1.01, 101.0), (50.03, 50.0), (49.97, 49.95), (101.4, 101.0), (101.303, 101.0),
        (1012.0, 1010.0), (9.999, 9.99), (60.66, 60.6), (10.1, 10.1), (500.9, 500.0), (49.95, 49.95),
    ])
    def test_round_down_to_tick(self, price, expected):
        assert taiwan_round_down_to_tick(price) == pytest.approx(expected, abs=1e-9)

    def test_entry_limit_price_per_model(self):
        assert squeeze_kdj_entry_limit_price(100.0, "limit_1tick") == 100.5
        assert squeeze_kdj_entry_limit_price(49.95, "limit_ticks", entry_limit_ticks=2) == pytest.approx(50.1)
        assert squeeze_kdj_entry_limit_price(100.3, "limit_pct", entry_limit_pct=0.01) == pytest.approx(101.0)
        assert squeeze_kdj_entry_limit_price(60.0, "limit_pct", entry_limit_pct=0.01) == pytest.approx(60.6)
        assert squeeze_kdj_entry_limit_price(100.0, "market_open") is None
        assert squeeze_kdj_entry_limit_price(100.0, "open") is None

    def test_model_lists_backwards_compatible(self):
        assert VALID_EXECUTION_MODELS == ("open", "limit_1tick")
        assert EXTENDED_EXECUTION_MODELS == ("limit_ticks", "limit_pct", "market_open")
        assert ALL_EXECUTION_MODELS == VALID_EXECUTION_MODELS + EXTENDED_EXECUTION_MODELS


def _run_exec(df, open_price, close_t=None, **kwargs):
    """觸發K棒idx20(收盤close_t)、idx21開盤open_price進場；回傳(trades, diag, 第一次_process_mr_day看到的部位)。"""
    df = df.copy()
    if close_t is not None:
        df.iloc[20, df.columns.get_loc("Close")] = close_t
    df.iloc[21, df.columns.get_loc("Open")] = open_price
    df.iloc[21, df.columns.get_loc("High")] = max(df.iloc[21]["High"], open_price)
    df.iloc[21, df.columns.get_loc("Low")] = min(df.iloc[21]["Low"], open_price)
    features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
    params = dict(max_hold_days=3)
    params.update(kwargs)
    return _run_limit(df, features, **params)


class TestLimitTicksEntry:
    @pytest.mark.parametrize("open_price, expected_fill", [
        (49.95, 50.0),   # 開盤+1檔
        (50.0, 50.1),    # 50的tick是0.1 → 50.1 = 限價
        (50.05, 50.1),   # min(50.15, 50.1) → 被限價封頂
        (50.1, 50.1),    # 剛好等於限價
        (48.0, 48.05),   # 開低
    ])
    def test_two_ticks_across_boundary_fill(self, open_price, expected_fill):
        df = _make_flat_df(49.95, n=30)
        trades, diag = _run_exec(df, open_price, execution_model="limit_ticks", entry_limit_ticks=2)
        assert diag["skipped_limit_not_filled"] == 0
        assert trades[0]["e_price"] == pytest.approx(expected_fill)
        assert trades[0]["entry_trigger_close"] == 49.95 and trades[0]["entry_open"] == open_price
        assert trades[0]["entry_gap_pct"] == pytest.approx(open_price / 49.95 - 1)

    def test_two_ticks_skip_above_limit_but_one_tick_skips_earlier(self):
        df = _make_flat_df(49.95, n=30)
        trades, diag = _run_exec(df, 50.2, execution_model="limit_ticks", entry_limit_ticks=2)
        assert trades == [] and diag["skipped_limit_not_filled"] == 1
        # 50.05：2檔(限價50.1)會買、1檔(限價50.0)不買
        t2, d2 = _run_exec(df, 50.05, execution_model="limit_ticks", entry_limit_ticks=2)
        t1, d1 = _run_exec(df, 50.05, execution_model="limit_ticks", entry_limit_ticks=1)
        tl, dl = _run_exec(df, 50.05, execution_model="limit_1tick")
        assert len(t2) == 1 and d2["skipped_limit_not_filled"] == 0
        assert t1 == [] and d1["skipped_limit_not_filled"] == 1 == dl["skipped_limit_not_filled"] and tl == []

    def test_stop_targets_use_fill_and_market_exit_one_tick_worse(self):
        df = _limit_df()
        df.iloc[23, df.columns.get_loc("Low")] = 90.0
        trades, _ = _run_exec(df, 100.0, execution_model="limit_ticks", entry_limit_ticks=2, max_hold_days=30)
        t = trades[0]
        assert t["e_price"] == pytest.approx(100.5) and t["exit_reason"] == "stop"
        assert t["exit_price"] == pytest.approx(100.5 - 2.0 - 0.1)  # 停損98.5，市價多付1檔(98.5的tick=0.1)


class TestLimitPctEntry:
    def test_pct_limit_rounds_down_and_skips(self):
        df = _limit_df()
        # 收盤100.3 → 100.3x1.01 = 101.303 → 往下取到101.0；開盤101.2(跳空<1%)仍然超過限價 → 不買
        trades, diag = _run_exec(df, 101.2, close_t=100.3, execution_model="limit_pct", entry_limit_pct=0.01)
        assert trades == [] and diag["skipped_limit_not_filled"] == 1
        trades, diag = _run_exec(df, 101.0, close_t=100.3, execution_model="limit_pct", entry_limit_pct=0.01)
        assert trades[0]["e_price"] == pytest.approx(101.0)  # min(101.5, 101.0)
        assert diag["skipped_limit_not_filled"] == 0

    def test_pct_fill_below_limit(self):
        df = _make_flat_df(60.0, n=30)
        trades, _ = _run_exec(df, 60.3, execution_model="limit_pct", entry_limit_pct=0.01)
        assert trades[0]["e_price"] == pytest.approx(60.4)  # 限價60.6，開盤+1檔=60.4
        trades, diag = _run_exec(df, 60.7, execution_model="limit_pct", entry_limit_pct=0.01)
        assert trades == [] and diag["skipped_limit_not_filled"] == 1


class TestMarketOpenEntry:
    @pytest.mark.parametrize("open_price, expected_fill", [(105.0, 105.5), (100.0, 100.5), (95.0, 95.1), (49.0, 49.05)])
    def test_always_fills_at_open_plus_one_tick(self, open_price, expected_fill):
        df = _limit_df()
        trades, diag = _run_exec(df, open_price, execution_model="market_open")
        assert diag["skipped_limit_not_filled"] == 0 and len(trades) == 1
        t = trades[0]
        assert t["e_price"] == pytest.approx(expected_fill)
        assert t["entry_gap_pct"] == pytest.approx(open_price / 100.0 - 1)
        if open_price == 100.0:  # 其他開盤價會在進場當天/隔天碰到停損或停利，只檢查平盤那個的到期出場
            assert t["exit_reason"] == "forced_close" and t["exit_price"] == pytest.approx(100.0 - 0.5)

    def test_gap_fields_only_on_new_models(self):
        df = _limit_df()
        for model in ("open", "limit_1tick"):
            trades, _ = _run_exec(df, 100.0, execution_model=model)
            assert not any(k in trades[0] for k in _GAP_KEYS), model
        for model, kw in (("limit_ticks", dict(entry_limit_ticks=1)), ("limit_pct", dict(entry_limit_pct=0.01)),
                          ("market_open", {})):
            trades, _ = _run_exec(df, 100.0, execution_model=model, **kw)
            assert all(k in trades[0] for k in _GAP_KEYS), model
            assert trades[0]["entry_gap_pct"] == 0.0

    def test_custom_costs_applied_on_new_models(self):
        df = _limit_df()
        trades, _ = _run_exec(df, 100.0, execution_model="market_open", commission_per_lot_side=50.0,
                              futures_tax_rate=0.00002)
        t = trades[0]
        mult = get_contract_multiplier("1101", 100.5)
        expected = (99.5 - 100.5) * mult * 2 - 50.0 * 2 * 2 - 0.00002 * (100.5 + 99.5) * mult * 2
        assert t["pnl_ntd"] == pytest.approx(expected) and t["commission_ntd"] == 200.0


class TestExtendedExecutionArgumentValidation:
    @pytest.mark.parametrize("kw", [
        dict(execution_model="limit_ticks"),
        dict(execution_model="limit_ticks", entry_limit_ticks=0),
        dict(execution_model="limit_ticks", entry_limit_ticks=1.5),
        dict(execution_model="limit_1tick", entry_limit_ticks=1),
        dict(execution_model="market_open", entry_limit_ticks=2),
        dict(execution_model="limit_pct"),
        dict(execution_model="limit_pct", entry_limit_pct=-0.01),
        dict(execution_model="limit_pct", entry_limit_pct=np.nan),
        dict(execution_model="limit_ticks", entry_limit_ticks=2, entry_limit_pct=0.01),
        dict(execution_model="open", entry_limit_pct=0.01),
        dict(execution_model="market_open", slippage_pct=0.001),
        dict(execution_model="limit_pct", entry_limit_pct=0.01, slippage_pct=0.001),
    ])
    def test_bad_combinations_raise(self, kw):
        df = _limit_df()
        features = _make_fixed_features(df, entry_idx=20, prior_low=np.nan)
        with pytest.raises(ValueError):
            _run_limit(df, features, **kw)


def _strip_gap_keys(trades):
    return [{k: v for k, v in t.items() if k not in _GAP_KEYS} for t in trades]


class TestLimitTicksN1MatchesLimit1Tick:
    """凍結比對：limit_ticks N=1 跟 limit_1tick 在合成市場上逐筆相同(拿掉三個新欄位後)，診斷計數器也相同。"""

    @pytest.mark.parametrize("seed, variant, mcp, top_n, extra", [
        (0, "B", 3, 3, {}),
        (1, "B", 2, 3, dict(atr_stop_mult=1.5, atr_target_mult=3.0, commission_per_lot_side=50.0,
                            futures_tax_rate=0.00002)),
        (2, "A", 2, 2, {}),
        (3, "B_trail", 3, 3, dict(atr_stop_mult=1.5)),
        (0, "B", 1, 3, dict(atr_stop_mult=1.5, atr_target_mult=4.0, breakeven_trigger_atr=1.5)),
        (1, "B", 2, 3, dict(risk_pct_per_trade=0.01, skip_entry_weekdays=(4,))),
    ])
    def test_identical_trades_and_diagnostics(self, seed, variant, mcp, top_n, extra):
        price_data, universe, idx = _make_synthetic_market(seed=seed)
        feats = precompute_squeeze_kdj_features_by_code(price_data, universe)
        kw = dict(variant=variant, lots=1, top_n=top_n, max_concurrent_positions=mcp, max_hold_days=20,
                  features_by_code=feats, return_diagnostics=True, **extra)
        old, d_old = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000,
                                                                  execution_model="limit_1tick", **kw)
        new, d_new = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000,
                                                                  execution_model="limit_ticks", entry_limit_ticks=1,
                                                                  **kw)
        assert len(old) > 0
        assert _strip_gap_keys(new) == old
        assert d_new == d_old
        assert all(all(k in t for k in _GAP_KEYS) for t in new)

    def test_skip_counts_and_market_open_never_skips(self):
        price_data, universe, idx = _make_synthetic_market(seed=0)
        feats = precompute_squeeze_kdj_features_by_code(price_data, universe)
        kw = dict(variant="B", lots=1, top_n=3, max_concurrent_positions=3, max_hold_days=20,
                  features_by_code=feats, return_diagnostics=True)
        _, d1 = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000,
                                                             execution_model="limit_1tick", **kw)
        t3, d3 = run_squeeze_kdj_capital_constrained_backtest(price_data, universe, idx, 1_000_000,
                                                              execution_model="market_open", **kw)
        assert d1["skipped_limit_not_filled"] > 0 and d3["skipped_limit_not_filled"] == 0
        for t in t3:
            assert t["e_price"] == pytest.approx(t["entry_open"] + taiwan_tick_size(t["entry_open"]))
            assert t["entry_gap_pct"] == pytest.approx(t["entry_open"] / t["entry_trigger_close"] - 1)
        # 跳空 > 1檔的成交在X3裡一定存在(合成市場開盤有雜訊)，而且正是limit_1tick不會買的那種
        assert any(t["entry_open"] > t["entry_trigger_close"] + taiwan_tick_size(t["entry_trigger_close"]) for t in t3)
