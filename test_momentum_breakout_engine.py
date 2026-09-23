"""
test_momentum_breakout_engine.py
右側順勢突破引擎的單元測試。專注在這個模組「新增」的邏輯：
結構性突破門檻、訊號評分/權重、進場跳空風控、MACD/指標計算、以及跟均值回歸引擎
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

    def test_flat_series_histogram_near_zero(self):
        close = pd.Series([100.0] * 60)
        _, _, hist = mbe.compute_macd(close)
        assert abs(hist.iloc[-1]) < 1e-6


class TestPrecomputeBreakoutIndicators:
    def test_rolling_high_excludes_current_day(self):
        # 前面平盤在100，最後一天暴衝到200；rolling_high(當天)不該把這天自己算進去，
        # 否則突破判斷會變成恆真(股價永遠等於自己的新高)。
        closes = [100.0] * 25 + [200.0]
        df = make_price_df(closes)
        index_close = pd.Series(100.0, index=df.index)
        ind = mbe.precompute_breakout_indicators(df, index_close, breakout_lookback=20)
        last_rolling_high = ind["RollingHigh"].iloc[-1]
        assert last_rolling_high == pytest.approx(100.0)
        assert ind["Close"].iloc[-1] > last_rolling_high  # 這天應該被判定為突破前高

    def test_volume_ratio_computed(self):
        closes = list(np.linspace(100, 110, 30))
        volumes = [1000.0] * 29 + [3000.0]
        df = make_price_df(closes, volumes=volumes)
        index_close = pd.Series(100.0, index=df.index)
        ind = mbe.precompute_breakout_indicators(df, index_close)
        # rolling(20)在最後一天涵蓋前19天(1000)+當天自己(3000)：均量=(19*1000+3000)/20=1100
        expected_ratio = 3000.0 / ((19 * 1000.0 + 3000.0) / 20)
        assert ind["VolumeRatio"].iloc[-1] == pytest.approx(expected_ratio, rel=1e-6)

    def test_rel_strength_positive_when_stock_outperforms(self):
        closes = list(np.linspace(100, 130, 30))  # 股票大漲
        df = make_price_df(closes)
        index_close = pd.Series(np.linspace(100, 102, 30), index=df.index)  # 大盤小漲
        ind = mbe.precompute_breakout_indicators(df, index_close, rel_strength_window=5)
        assert ind["RelStrength"].iloc[-1] > 0


class TestScanMomentumBreakoutCandidates:
    def _build_indicators(self, code_specs, as_of_pos=65):
        """code_specs: {code: dict(close, ma20, atr, rolling_high, rolling_low, volume_ratio,
                                    rel_strength, rsi, macd_hist)}，建出長度足夠(>60列)的指標DataFrame，
        最後一列(索引as_of_pos-1)放真正要測試的值，其餘列填無害的預設值。"""
        n = as_of_pos + 5
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        indicators_by_code = {}
        for code, spec in code_specs.items():
            df = pd.DataFrame({
                "Close": 100.0, "MA20": 95.0, "ATR": 2.0,
                "VolumeRatio": 1.0, "RollingHigh": 105.0, "RollingLow": 95.0,
                "RSI": 50.0, "MACDHist": 0.0,
            }, index=idx)
            for key, val in spec.items():
                col = {"close": "Close", "ma20": "MA20", "atr": "ATR", "rolling_high": "RollingHigh",
                       "rolling_low": "RollingLow", "volume_ratio": "VolumeRatio",
                       "rel_strength": "RelStrength", "rsi": "RSI", "macd_hist": "MACDHist"}[key]
                if col == "RelStrength" and col not in df.columns:
                    df[col] = 0.0
                df.loc[idx[as_of_pos - 1], col] = val
            if "RelStrength" not in df.columns:
                df["RelStrength"] = 0.0
            indicators_by_code[code] = df
        return indicators_by_code, idx[as_of_pos]

    def test_long_requires_above_ma20_and_above_rolling_high(self):
        specs = {
            "0001": {"close": 110.0, "ma20": 100.0, "rolling_high": 105.0},  # 真突破：過門檻
            "0002": {"close": 102.0, "ma20": 100.0, "rolling_high": 105.0},  # 站上均線但沒破前高
            "0003": {"close": 98.0, "ma20": 100.0, "rolling_high": 105.0},   # 連均線都沒站上
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
            "0001": {"close": 90.0, "ma20": 100.0, "rolling_low": 95.0},   # 真跌破：過門檻
            "0002": {"close": 98.0, "ma20": 100.0, "rolling_low": 95.0},   # 跌破均線但沒破前低
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(),
            allow_short=True, top_n=5,
        )
        codes = {c["code"] for c in result if c["side"] == "short"}
        assert codes == {"0001"}

    def test_bear_regime_disables_long_signals(self):
        specs = {"0001": {"close": 110.0, "ma20": 100.0, "rolling_high": 105.0}}
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="bear", excluded_codes=set(), allow_short=False,
        )
        assert result == []

    def test_bull_regime_disables_short_signals(self):
        specs = {"0001": {"close": 90.0, "ma20": 100.0, "rolling_low": 95.0}}
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="bull", excluded_codes=set(), allow_short=True,
        )
        assert result == []

    def test_excluded_codes_skipped(self):
        specs = {"0001": {"close": 110.0, "ma20": 100.0, "rolling_high": 105.0}}
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes={"0001"}, allow_short=False,
        )
        assert result == []

    def test_single_signal_weight_drives_ranking(self):
        # 0001量比較高但相對大盤強弱較低，0002相反；只開量比權重時0001該排前面，
        # 只開相對大盤強弱權重時0002該排前面。
        specs = {
            "0001": {"close": 110.0, "ma20": 100.0, "rolling_high": 105.0,
                     "volume_ratio": 3.0, "rel_strength": 0.01},
            "0002": {"close": 110.0, "ma20": 100.0, "rolling_high": 105.0,
                     "volume_ratio": 1.1, "rel_strength": 0.10},
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

    def test_chip_confirm_signal_used_when_provided(self):
        specs = {
            "0001": {"close": 110.0, "ma20": 100.0, "rolling_high": 105.0},
            "0002": {"close": 110.0, "ma20": 100.0, "rolling_high": 105.0},
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)
        idx = indicators_by_code["0001"].index
        chip_streak_by_code = {
            "0001": pd.Series(5.0, index=idx),   # 連續買超5天
            "0002": pd.Series(-3.0, index=idx),  # 連續賣超3天
        }
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(), allow_short=False,
            signal_weights={"score_chip_confirm": 1.0}, chip_streak_by_code=chip_streak_by_code, top_n=5,
        )
        assert result[0]["code"] == "0001"

    def test_top_n_limits_result_count_per_side(self):
        specs = {
            f"code{i}": {"close": 110.0, "ma20": 100.0, "rolling_high": 105.0}
            for i in range(5)
        }
        indicators_by_code, as_of_date = self._build_indicators(specs)
        result = mbe.scan_momentum_breakout_candidates(
            indicators_by_code, as_of_date, regime="neutral", excluded_codes=set(), allow_short=False, top_n=2,
        )
        assert len(result) == 2


class TestTryEnterBreakout:
    def test_long_blocked_by_unfavorable_gap_down(self):
        df = make_price_df([100.0, 99.4], opens=[100.0, 99.4])  # 跳空跌破-0.5%
        # 讓entry_date那天開盤=99.4, c_prev=100 -> gap = -0.6%，應該被擋掉
        price_data = {"0001": df}
        candidates = [{"code": "0001", "side": "long", "c_prev": 100.0, "atr": 2.0}]
        pos = mbe.try_enter_breakout(price_data, candidates, df.index[1], starting_capital=1_000_000)
        assert pos is None

    def test_short_blocked_by_unfavorable_gap_up(self):
        df = make_price_df([100.0, 100.6], opens=[100.0, 100.6])
        price_data = {"0001": df}
        candidates = [{"code": "0001", "side": "short", "c_prev": 100.0, "atr": 2.0}]
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
        # 1101/1102都沒有小型契約，統一用標準契約(2000股)，保證金比例查表都是20%：
        # 進場價500 -> 保證金=500*2000*2*0.2=400,000；進場價50 -> 保證金=50*2000*2*0.2=40,000。
        # 起始資金20萬、上限35% = 70,000：第一檔超過上限該被跳過，第二檔沒超過該進場。
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
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
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
        # 用其中一檔股票當大盤代理指標
        index_code = real_codes[0]

        indicators_by_code = mbe.precompute_all_breakout_indicators(price_data, universe, index_code=index_code)
        regime_series = precompute_regime_series(price_data[index_code])

        trades = mbe.run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=price_data[index_code].index, max_hold_days=5, starting_capital=1_000_000,
            allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0, top_n=2,
        )
        assert isinstance(trades, list)
        # 不強求一定有交易(合成資料本來就可能剛好不觸發突破門檻)，但如果有交易，
        # 每筆都該有出場理由跟pnl欄位，且能被summarize_mr()正常彙總不出錯。
        from mean_reversion_engine import summarize_mr
        stats = summarize_mr(trades, 1_000_000)
        assert stats["trade_count"] == len(trades)
        for t in trades:
            assert t["exit_reason"] in ("stop", "target", "forced_close")
            assert t["side"] in ("long", "short")
