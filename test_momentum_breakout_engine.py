"""
test_momentum_breakout_engine.py
右側順勢突破引擎的單元測試。專注在這個模組「新增」的邏輯：
結構性突破門檻(含4個可選的額外門檻)、10個可搭配訊號的評分/權重、進場跳空風控、
指標計算(均線/黃金交叉/價量同步創高/跳空/K棒實體/籌碼比重)、以及跟均值回歸引擎
共用出場機制之後的整合式煙霧測試。
"""
import numpy as np
import pandas as pd
import pytest

import momentum_breakout_engine as mbe
from mean_reversion_engine import precompute_regime_series


def make_price_df(closes, highs=None, lows=None, opens=None, volumes=None, start="2024-01-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="B")
    closes = pd.Series(closes, index=idx, dtype=float)
    highs = pd.Series(highs, index=idx, dtype=float) if highs is not None else closes * 1.01
    lows = pd.Series(lows, index=idx, dtype=float) if lows is not None else closes * 0.99
    opens = pd.Series(opens, index=idx, dtype=float) if opens is not None else closes
    volumes = pd.Series(volumes, index=idx, dtype=float) if volumes is not None else pd.Series(1000.0, index=idx)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes}, index=idx)


class TestComputeMacd:
    def test_uptrend_gives_positive_histogram_eventually(self):
        close = pd.Series(np.linspace(10, 40, 80))
        _, _, hist = mbe.compute_macd(close)
        assert hist.iloc[-1] > 0

    def test_downtrend_gives_negative_histogram_eventually(self):
        close = pd.Series(np.linspace(40, 10, 80))
        _, _, hist = mbe.compute_macd(close)
        assert hist.iloc[-1] < 0


class TestComputeInstitutionalRatio:
    def test_ratio_positive_when_buying(self):
        df = make_price_df([100.0] * 10, volumes=[1000.0] * 10)
        chip_df = pd.DataFrame({"foreign_net": [500.0] * 10}, index=df.index)
        ratio = mbe.compute_institutional_ratio(chip_df, df, "foreign_net")
        # 買超500股 x 收盤100 = 50,000元；成交金額 = 100*1000=100,000元 -> ratio=0.5
        assert ratio.iloc[-1] == pytest.approx(0.5)

    def test_ratio_nan_when_turnover_zero(self):
        df = make_price_df([100.0] * 5, volumes=[0.0] * 5)
        chip_df = pd.DataFrame({"foreign_net": [100.0] * 5}, index=df.index)
        ratio = mbe.compute_institutional_ratio(chip_df, df, "foreign_net")
        assert ratio.isna().all()


class TestPrecomputeBreakoutIndicators:
    def test_rolling_high_excludes_current_day(self):
        closes = [100.0] * 25 + [200.0]
        df = make_price_df(closes)
        index_close = pd.Series(100.0, index=df.index)
        ind = mbe.precompute_breakout_indicators(df, index_close, breakout_lookback=20)
        last_rolling_high = ind["RollingHigh"].iloc[-1]
        assert last_rolling_high == pytest.approx(100.0)
        assert ind["Close"].iloc[-1] > last_rolling_high

    def test_golden_cross_flag_detects_recent_ma5_cross_above_ma20(self):
        # 前面走平(MA5<=MA20附近)，中段急拉讓MA5穿越MA20向上，之後維持高檔
        closes = [100.0] * 40 + [100.0, 102.0, 106.0, 112.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0]
        df = make_price_df(closes)
        index_close = pd.Series(100.0, index=df.index)
        ind = mbe.precompute_breakout_indicators(df, index_close, golden_cross_lookback=3)
        assert ind["MA5"].iloc[-1] > ind["MA20"].iloc[-1]
        # 穿越事件剛發生後幾天內(窗口內)，旗標該是1
        cross_pos = ind["GoldenCrossFlag"].to_numpy().nonzero()[0]
        assert len(cross_pos) > 0
        first_cross = cross_pos[0]
        # 事件發生很久之後(窗口外)，旗標該回到0，不會一路維持1到最後
        assert ind["GoldenCrossFlag"].iloc[-1] == 0.0
        assert ind["GoldenCrossFlag"].iloc[first_cross] == 1.0

    def test_price_volume_new_high_long_requires_both(self):
        closes = [100.0] * 25 + [200.0]  # 價創高
        volumes = [1000.0] * 25 + [1000.0]  # 量沒有創高
        df = make_price_df(closes, volumes=volumes)
        index_close = pd.Series(100.0, index=df.index)
        ind = mbe.precompute_breakout_indicators(df, index_close, breakout_lookback=20)
        assert ind["PriceVolNewHighLong"].iloc[-1] == 0.0  # 只有價創高，量沒創高，不算數

        volumes2 = [1000.0] * 25 + [5000.0]  # 這次量也創高
        df2 = make_price_df(closes, volumes=volumes2)
        ind2 = mbe.precompute_breakout_indicators(df2, index_close, breakout_lookback=20)
        assert ind2["PriceVolNewHighLong"].iloc[-1] == 1.0

    def test_gap_up_flag_detects_recent_gap(self):
        closes = [100.0] * 10
        opens = [100.0] * 9 + [103.0]  # 最後一天跳空開高3%
        df = make_price_df(closes, opens=opens)
        index_close = pd.Series(100.0, index=df.index)
        ind = mbe.precompute_breakout_indicators(df, index_close, gap_lookback=5, gap_threshold=0.02)
        assert ind["GapUpFlag"].iloc[-1] == 1.0
        assert ind["GapDownFlag"].iloc[-1] == 0.0

    def test_candle_body_score_bullish_vs_bearish(self):
        # 一根長紅K：開低走高，實體占比大
        df = make_price_df([100.0], opens=[95.0], highs=[101.0], lows=[94.0], volumes=[1000.0])
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        closes = [100.0] * 9 + [100.0]
        opens = [100.0] * 9 + [95.0]
        highs = [101.0] * 9 + [101.0]
        lows = [99.0] * 9 + [94.0]
        df = make_price_df(closes, opens=opens, highs=highs, lows=lows)
        index_close = pd.Series(100.0, index=df.index)
        ind = mbe.precompute_breakout_indicators(df, index_close, candle_lookback=1)
        assert ind["BullishCandleScore"].iloc[-1] > 0
        assert ind["BearishCandleScore"].iloc[-1] == 0

    def test_chip_ratio_columns_present_when_chip_df_given(self):
        df = make_price_df([100.0] * 10)
        index_close = pd.Series(100.0, index=df.index)
        chip_df = pd.DataFrame({"foreign_net": [100.0] * 10, "trust_net": [-50.0] * 10}, index=df.index)
        ind = mbe.precompute_breakout_indicators(df, index_close, chip_df=chip_df)
        assert ind["ForeignRatio"].notna().any()
        assert ind["TrustRatio"].notna().any()

    def test_chip_ratio_columns_nan_when_no_chip_df(self):
        df = make_price_df([100.0] * 10)
        index_close = pd.Series(100.0, index=df.index)
        ind = mbe.precompute_breakout_indicators(df, index_close, chip_df=None)
        assert ind["ForeignRatio"].isna().all()
        assert ind["TrustRatio"].isna().all()


class TestScanMomentumBreakoutCandidates:
    DEFAULT_COLS = {
        "Close": 100.0, "MA5": 96.0, "MA20": 95.0, "MA60": 90.0, "ATR": 2.0,
        "VolumeRatio": 1.0, "RollingHigh": 105.0, "RollingLow": 95.0,
        "RSI": 50.0, "MACDHist": 0.0, "RelStrength": 0.0,
        "GoldenCrossFlag": 0.0, "DeathCrossFlag": 0.0,
        "PriceVolNewHighLong": 0.0, "PriceVolNewHighShort": 0.0,
        "GapUpFlag": 0.0, "GapDownFlag": 0.0,
        "BullishCandleScore": 0.0, "BearishCandleScore": 0.0,
        "ForeignRatio": np.nan, "TrustRatio": np.nan,
    }

    def _build_indicators(self, code_specs, as_of_pos=65):
        n = as_of_pos + 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        indicators_by_code = {}
        for code, spec in code_specs.items():
            df = pd.DataFrame({col: val for col, val in self.DEFAULT_COLS.items()}, index=idx)
            for col, val in spec.items():
                df.loc[idx[as_of_pos - 1], col] = val
            indicators_by_code[code] = df
        return indicators_by_code, idx[as_of_pos]

    def test_long_requires_above_ma20_and_above_rolling_high(self):
        specs = {
            "0001": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0},
            "0002": {"Close": 102.0, "MA20": 100.0, "RollingHigh": 105.0},
            "0003": {"Close": 98.0, "MA20": 100.0, "RollingHigh": 105.0},
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(),
            allow_short=False, top_n=5,
        )
        codes = {c["code"] for c in result}
        assert codes == {"0001"}

    def test_short_requires_below_ma20_and_below_rolling_low(self):
        specs = {
            "0001": {"Close": 90.0, "MA20": 100.0, "RollingLow": 95.0},
            "0002": {"Close": 98.0, "MA20": 100.0, "RollingLow": 95.0},
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(),
            allow_short=True, top_n=5,
        )
        codes = {c["code"] for c in result if c["side"] == "short"}
        assert codes == {"0001"}

    def test_bear_regime_disables_long_signals(self):
        specs = {"0001": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0}}
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="bear", excluded_codes=set(), allow_short=False,
        )
        assert result == []

    def test_require_above_ma60_filters_out_below_season_line(self):
        specs = {
            "0001": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0, "MA60": 120.0},  # 站上均線突破，但在季線下
            "0002": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0, "MA60": 90.0},   # 也在季線上
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(), allow_short=False,
            require_above_ma60=True, top_n=5,
        )
        codes = {c["code"] for c in result}
        assert codes == {"0002"}

    def test_require_ma_bullish_alignment_needs_full_stacking(self):
        specs = {
            "0001": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0, "MA5": 105.0, "MA60": 90.0},  # 5>20>60
            "0002": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0, "MA5": 95.0, "MA60": 90.0},   # 5<20
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(), allow_short=False,
            require_ma_bullish_alignment=True, top_n=5,
        )
        codes = {c["code"] for c in result}
        assert codes == {"0001"}

    def test_require_dual_institutional_buy_needs_both_positive(self):
        specs = {
            "0001": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0, "ForeignRatio": 0.1, "TrustRatio": 0.05},
            "0002": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0, "ForeignRatio": 0.1, "TrustRatio": -0.05},
            "0003": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0},  # 沒有籌碼資料(NaN)，保守排除
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(), allow_short=False,
            require_dual_institutional_buy=True, top_n=5,
        )
        codes = {c["code"] for c in result}
        assert codes == {"0001"}

    def test_min_volume_ratio_filters_low_volume_candidates(self):
        specs = {
            "0001": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0, "VolumeRatio": 2.0},
            "0002": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0, "VolumeRatio": 1.1},
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(), allow_short=False,
            min_volume_ratio=1.5, top_n=5,
        )
        codes = {c["code"] for c in result}
        assert codes == {"0001"}

    def test_single_signal_weight_drives_ranking(self):
        specs = {
            "0001": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0,
                     "VolumeRatio": 3.0, "RelStrength": 0.01},
            "0002": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0,
                     "VolumeRatio": 1.1, "RelStrength": 0.10},
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)

        by_volume = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(), allow_short=False,
            signal_weights={"score_volume_ratio": 1.0}, top_n=5,
        )
        assert by_volume[0]["code"] == "0001"

        by_rel_strength = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(), allow_short=False,
            signal_weights={"score_rel_strength": 1.0}, top_n=5,
        )
        assert by_rel_strength[0]["code"] == "0002"

    def test_foreign_ratio_signal_used_when_weighted(self):
        specs = {
            "0001": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0, "ForeignRatio": 0.2},
            "0002": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0, "ForeignRatio": -0.1},
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(), allow_short=False,
            signal_weights={"score_foreign_ratio": 1.0}, top_n=5,
        )
        assert result[0]["code"] == "0001"

    def test_top_n_limits_result_count_per_side(self):
        specs = {f"code{i}": {"Close": 110.0, "MA20": 100.0, "RollingHigh": 105.0} for i in range(5)}
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(), allow_short=False, top_n=2,
        )
        assert len(result) == 2


class TestTryEnterBreakout:
    def test_long_blocked_by_unfavorable_gap_down(self):
        df = make_price_df([100.0, 99.4], opens=[100.0, 99.4])
        price_data = {"1101": df}
        candidates = [{"code": "1101", "side": "long", "c_prev": 100.0, "atr": 2.0}]
        pos = mbe.try_enter_breakout(price_data, candidates, df.index[1], starting_capital=1_000_000)
        assert pos is None

    def test_short_blocked_by_unfavorable_gap_up(self):
        df = make_price_df([100.0, 100.6], opens=[100.0, 100.6])
        price_data = {"1101": df}
        candidates = [{"code": "1101", "side": "short", "c_prev": 100.0, "atr": 2.0}]
        pos = mbe.try_enter_breakout(price_data, candidates, df.index[1], starting_capital=1_000_000)
        assert pos is None

    def test_long_enters_and_sets_atr_based_stop_target(self):
        df = make_price_df([100.0, 100.1], opens=[100.0, 100.1])
        price_data = {"1101": df}
        candidates = [{"code": "1101", "side": "long", "c_prev": 100.0, "atr": 2.0}]
        pos = mbe.try_enter_breakout(
            price_data, candidates, df.index[1], starting_capital=1_000_000,
            atr_stop_mult=0.5, atr_target_mult=2.0,
        )
        assert pos is not None
        assert pos["stop_price"] == pytest.approx(pos["e_price"] - 0.5 * 2.0)
        assert pos["target_price"] == pytest.approx(pos["e_price"] + 2.0 * 2.0)

    def test_short_enters_and_sets_atr_based_stop_target(self):
        df = make_price_df([100.0, 99.9], opens=[100.0, 99.9])
        price_data = {"1101": df}
        candidates = [{"code": "1101", "side": "short", "c_prev": 100.0, "atr": 2.0}]
        pos = mbe.try_enter_breakout(
            price_data, candidates, df.index[1], starting_capital=1_000_000,
            atr_stop_mult=0.5, atr_target_mult=2.0,
        )
        assert pos is not None
        assert pos["stop_price"] == pytest.approx(pos["e_price"] + 0.5 * 2.0)
        assert pos["target_price"] == pytest.approx(pos["e_price"] - 2.0 * 2.0)

    def test_margin_cap_skips_to_next_candidate(self):
        df1 = make_price_df([500.0, 500.0], opens=[500.0, 500.0])
        df2 = make_price_df([50.0, 50.0], opens=[50.0, 50.0])
        price_data = {"1101": df1, "1102": df2}
        candidates = [
            {"code": "1101", "side": "long", "c_prev": 500.0, "atr": 2.0},
            {"code": "1102", "side": "long", "c_prev": 50.0, "atr": 1.0},
        ]
        pos = mbe.try_enter_breakout(price_data, candidates, df1.index[1], starting_capital=200_000)
        assert pos is not None
        assert pos["code"] == "1102"


class TestRunMomentumBreakoutBacktestSmoke:
    def test_runs_without_error_and_produces_summarizable_trades(self):
        np.random.seed(0)
        n = 140
        real_codes = ["1101", "1102", "1210", "1216", "1301", "1303"]
        universe = {}
        price_data = {}
        for i, code in enumerate(real_codes):
            base = 100 + i * 5
            trend = np.linspace(0, 20, n) if i % 2 == 0 else np.linspace(0, -10, n)
            noise = np.random.normal(0, 1, n)
            closes = base + trend + noise
            closes = np.maximum(closes, 1.0)
            df = make_price_df(list(closes), start="2024-01-01")
            price_data[code] = df
            universe[code] = {}
        index_code = real_codes[0]

        indicators_by_code = mbe.precompute_all_breakout_indicators(price_data, universe, index_code=index_code)
        regime_series = precompute_regime_series(price_data[index_code])

        trades = mbe.run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=price_data[index_code].index, max_hold_days=5, starting_capital=1_000_000,
            allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0, top_n=2,
        )
        assert isinstance(trades, list)
        from mean_reversion_engine import summarize_mr
        stats = summarize_mr(trades, 1_000_000)
        assert stats["trade_count"] == len(trades)
        for t in trades:
            assert t["exit_reason"] in ("stop", "target", "forced_close")
            assert t["side"] in ("long", "short")

    def test_runs_with_extra_gates_enabled_without_error(self):
        np.random.seed(1)
        n = 140
        real_codes = ["1101", "1102", "1210", "1216", "1301", "1303"]
        universe = {}
        price_data = {}
        for i, code in enumerate(real_codes):
            base = 100 + i * 5
            trend = np.linspace(0, 20, n) if i % 2 == 0 else np.linspace(0, -10, n)
            noise = np.random.normal(0, 1, n)
            closes = np.maximum(base + trend + noise, 1.0)
            df = make_price_df(list(closes), start="2024-01-01")
            price_data[code] = df
            universe[code] = {}
        index_code = real_codes[0]

        indicators_by_code = mbe.precompute_all_breakout_indicators(price_data, universe, index_code=index_code)
        regime_series = precompute_regime_series(price_data[index_code])

        trades = mbe.run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=price_data[index_code].index, max_hold_days=15, starting_capital=1_000_000,
            allow_short=True, lots=2, top_n=2,
            require_above_ma60=True, require_ma_bullish_alignment=True, min_volume_ratio=1.2,
        )
        assert isinstance(trades, list)
