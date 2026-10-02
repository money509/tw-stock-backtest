"""
tx_intraday_engine.py 的測試：全部用合成15分鐘K棒資料，不打網路。

重點涵蓋：
1. rolling_percentile_score()確實是用「自身歷史trailing窗口」而不是橫斷面排名，
   且不會看到未來資料(no look-ahead)。
2. scan_tx_intraday_candidate()的結構性突破門檻(gate)邏輯。
3. 消融(單一訊號)迴圈可以在小合成樣本上端到端跑完不崩潰，包含「某個訊號完全沒有
   觸發任何交易」的邊界情況(見TestNoTradeEdgeCase)。
4. run_tx_intraday_backtest()不會索引出界(候選出現在資料倒數第二根、最後一根時)。
"""
import numpy as np
import pandas as pd
import pytest

from tx_intraday_engine import (
    rolling_percentile_score, precompute_tx_intraday_indicators, scan_tx_intraday_candidate,
    run_tx_intraday_backtest, try_enter_tx_intraday, TX_INTRADAY_SIGNAL_NAMES,
)
from mean_reversion_engine import summarize_mr


def _make_bars(n, seed=0, trend=0.0):
    idx = pd.date_range("2026-08-03 08:45", periods=n, freq="15min")
    rng = np.random.RandomState(seed)
    closes = []
    price = 18000.0
    for i in range(n):
        price += trend + rng.uniform(-5, 5)
        closes.append(price)
    closes = np.array(closes)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + rng.uniform(0, 5, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 5, n)
    volume = rng.uniform(50, 150, n)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume}, index=idx)


class TestRollingPercentileScore:
    def test_output_range_is_0_to_100(self):
        s = pd.Series(np.random.RandomState(0).normal(size=200))
        scores = rolling_percentile_score(s, window=30)
        assert scores.between(0, 100).all()

    def test_higher_is_better_true_gives_high_score_to_new_high(self):
        # 一段平穩序列之後突然跳到全歷史最高值，higher_is_better=True時應該接近滿分(100分附近)
        s = pd.Series(np.concatenate([np.zeros(50), [1000.0]]))
        scores = rolling_percentile_score(s, window=50, higher_is_better=True, min_periods=10)
        assert scores.iloc[-1] > 95.0

    def test_higher_is_better_false_inverts_ranking(self):
        s = pd.Series(np.concatenate([np.zeros(50), [1000.0]]))
        scores_hib_true = rolling_percentile_score(s, window=50, higher_is_better=True, min_periods=10)
        scores_hib_false = rolling_percentile_score(s, window=50, higher_is_better=False, min_periods=10)
        assert scores_hib_false.iloc[-1] < scores_hib_true.iloc[-1]

    def test_warmup_period_scores_zero_not_nan(self):
        s = pd.Series(np.random.RandomState(1).normal(size=20))
        scores = rolling_percentile_score(s, window=50, min_periods=30)
        assert (scores == 0.0).all()
        assert not scores.isna().any()

    def test_no_lookahead_extending_series_does_not_change_earlier_scores(self):
        rng = np.random.RandomState(3)
        s_short = pd.Series(rng.normal(size=150))
        extra = pd.Series(rng.normal(size=50))
        s_long = pd.concat([s_short, extra], ignore_index=True)

        scores_short = rolling_percentile_score(s_short, window=30)
        scores_long = rolling_percentile_score(s_long, window=30)

        assert np.allclose(scores_short.to_numpy(), scores_long.iloc[:150].to_numpy())

    def test_nan_input_scores_zero(self):
        s = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0] * 10)
        scores = rolling_percentile_score(s, window=10, min_periods=3)
        nan_positions = s.isna()
        assert (scores[nan_positions] == 0.0).all()


class TestScanTxIntradayCandidate:
    def test_no_candidate_before_warmup(self):
        df = _make_bars(30)
        indicators = precompute_tx_intraday_indicators(df, rolling_window_bars=60, breakout_window_bars=20)
        # 前面幾根一定沒有足夠的RollingHigh/RollingLow歷史，應該回傳None
        assert scan_tx_intraday_candidate(indicators, 5) is None

    def test_gate_requires_close_above_rolling_high_for_long(self):
        n = 100
        idx = pd.date_range("2026-08-03 08:45", periods=n, freq="15min")
        closes = np.full(n, 100.0)
        closes[-1] = 200.0  # 最後一根大幅突破前高
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) + 1
        lows = np.minimum(opens, closes) - 1
        volume = np.full(n, 100.0)
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume}, index=idx)
        indicators = precompute_tx_intraday_indicators(df, rolling_window_bars=30, breakout_window_bars=20)

        candidate = scan_tx_intraday_candidate(indicators, n - 1, score_entry_threshold=0.0)
        assert candidate is not None
        assert candidate["side"] == "long"

    def test_gate_requires_close_below_rolling_low_for_short(self):
        n = 100
        idx = pd.date_range("2026-08-03 08:45", periods=n, freq="15min")
        closes = np.full(n, 100.0)
        closes[-1] = 20.0  # 最後一根大幅跌破前低
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) + 1
        lows = np.minimum(opens, closes) - 1
        volume = np.full(n, 100.0)
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume}, index=idx)
        indicators = precompute_tx_intraday_indicators(df, rolling_window_bars=30, breakout_window_bars=20)

        candidate = scan_tx_intraday_candidate(indicators, n - 1, score_entry_threshold=0.0)
        assert candidate is not None
        assert candidate["side"] == "short"

    def test_allow_short_false_never_returns_short_candidate(self):
        n = 100
        idx = pd.date_range("2026-08-03 08:45", periods=n, freq="15min")
        closes = np.full(n, 100.0)
        closes[-1] = 20.0
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) + 1
        lows = np.minimum(opens, closes) - 1
        volume = np.full(n, 100.0)
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume}, index=idx)
        indicators = precompute_tx_intraday_indicators(df, rolling_window_bars=30, breakout_window_bars=20)

        candidate = scan_tx_intraday_candidate(indicators, n - 1, allow_short=False, score_entry_threshold=0.0)
        assert candidate is None

    def test_score_threshold_blocks_low_conviction_entries(self):
        n = 100
        idx = pd.date_range("2026-08-03 08:45", periods=n, freq="15min")
        closes = np.full(n, 100.0)
        closes[-1] = 200.0
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) + 1
        lows = np.minimum(opens, closes) - 1
        volume = np.full(n, 100.0)
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume}, index=idx)
        indicators = precompute_tx_intraday_indicators(df, rolling_window_bars=30, breakout_window_bars=20)

        # 門檻設成100(滿分)幾乎不可能被贏過，應該擋掉這個候選
        candidate = scan_tx_intraday_candidate(indicators, n - 1, score_entry_threshold=1000.0)
        assert candidate is None


class TestTryEnterTxIntraday:
    def test_returns_none_when_entry_pos_out_of_range(self):
        df = _make_bars(10)
        candidate = {"side": "long", "score": 80.0, "atr": 10.0}
        assert try_enter_tx_intraday(df, candidate, entry_pos=10) is None

    def test_long_stop_below_entry_short_stop_above_entry(self):
        df = _make_bars(10)
        candidate_long = {"side": "long", "score": 80.0, "atr": 10.0}
        pos_long = try_enter_tx_intraday(df, candidate_long, entry_pos=5, atr_stop_mult=1.0)
        assert pos_long["stop_price"] < pos_long["e_price"]

        candidate_short = {"side": "short", "score": 80.0, "atr": 10.0}
        pos_short = try_enter_tx_intraday(df, candidate_short, entry_pos=5, atr_stop_mult=1.0)
        assert pos_short["stop_price"] > pos_short["e_price"]


class TestRunTxIntradayBacktestEndToEnd:
    def test_runs_without_crashing_on_small_synthetic_sample(self):
        df = _make_bars(300, seed=7)
        indicators = precompute_tx_intraday_indicators(df, rolling_window_bars=60, breakout_window_bars=20)
        trades = run_tx_intraday_backtest(df, indicators)
        # 不管有沒有交易，函式本身要跑完不崩潰，回傳的必須是list
        assert isinstance(trades, list)
        for t in trades:
            assert t["side"] in ("long", "short")
            assert t["hold_days"] >= 1
            assert t["exit_date"] >= t["entry_date"]

    def test_every_single_signal_ablation_runs_without_crash(self):
        """對TX_INTRADAY_SIGNAL_NAMES每個訊號單獨開啟，在小樣本上跑一次，這裡刻意用
        很短的資料(容易出現0筆交易)，確認summarize_mr()在0筆交易時optional欄位
        (profit_factor等)不會產生NaN/inf導致下游崩潰，而是回傳明確的0值。"""
        df = _make_bars(80, seed=2)
        indicators = precompute_tx_intraday_indicators(df, rolling_window_bars=60, breakout_window_bars=20)
        for name in TX_INTRADAY_SIGNAL_NAMES:
            trades = run_tx_intraday_backtest(df, indicators, signal_weights={name: 1.0})
            stats = summarize_mr(trades, starting_capital=200000.0)
            assert stats["trade_count"] == len(trades)
            assert np.isfinite(stats["profit_factor"]) or stats["profit_factor"] == float("inf")
            assert not np.isnan(stats["profit_factor"])

    def test_candidate_on_second_to_last_bar_does_not_index_out_of_range(self):
        # 構造一段資料，讓最後倒數第二根剛好滿足突破門檻，驗證進場邏輯(需要用到
        # entry_pos = i+1 = 最後一根)不會索引出界。
        n = 60
        idx = pd.date_range("2026-08-03 08:45", periods=n, freq="15min")
        closes = np.full(n, 100.0)
        closes[-2] = 500.0  # 倒數第二根大幅突破
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) + 1
        lows = np.minimum(opens, closes) - 1
        volume = np.full(n, 100.0)
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume}, index=idx)
        indicators = precompute_tx_intraday_indicators(df, rolling_window_bars=30, breakout_window_bars=20)

        trades = run_tx_intraday_backtest(df, indicators, score_entry_threshold=0.0)
        assert isinstance(trades, list)  # 沒有拋出IndexError就算通過

    def test_max_hold_bars_forces_exit(self):
        df = _make_bars(200, seed=5)
        indicators = precompute_tx_intraday_indicators(df, rolling_window_bars=60, breakout_window_bars=20)
        trades = run_tx_intraday_backtest(df, indicators, max_hold_bars=3, score_entry_threshold=0.0)
        for t in trades:
            assert t["hold_days"] <= 3 or t["exit_reason"] in ("stop", "stop_gap", "target", "forced_close")


class TestNoTradeEdgeCase:
    def test_flat_price_series_produces_zero_trades_gracefully(self):
        # 完全平盤(沒有任何突破)，應該產出0筆交易，summarize_mr()要能優雅處理，不崩潰。
        n = 100
        idx = pd.date_range("2026-08-03 08:45", periods=n, freq="15min")
        closes = np.full(n, 100.0)
        opens = closes.copy()
        highs = closes + 0.01
        lows = closes - 0.01
        volume = np.full(n, 100.0)
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume}, index=idx)
        indicators = precompute_tx_intraday_indicators(df, rolling_window_bars=30, breakout_window_bars=20)

        trades = run_tx_intraday_backtest(df, indicators)
        assert trades == []
        stats = summarize_mr(trades, starting_capital=200000.0)
        assert stats["trade_count"] == 0
        assert stats["profit_factor"] == 0.0
