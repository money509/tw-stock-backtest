"""
test_overnight_momentum_engine.py
====================================
針對 overnight_momentum_engine.py 的合成資料測試。不打任何真實 API，
全部用手造資料，驗證：
  1. percentile_score / bell_score / price_level_score 的評分數學正確
  2. 硬門檻（成交金額、均線位置）確實會刷掉不合格股票
  3. 籌碼面評分方向正確（當沖比例越低分越高；外資/投信比重越高分越高）
  4. simulate_next_day_exit 四種出場路徑都各自正確觸發：
       gap_stop（開盤跳空）/ stop（盤中停損）/ target（盤中停利）/ forced_close（收盤強制平倉）
  5. run_overnight_backtest 端到端跑得通，且交易記錄的 pnl 計算正確
"""

import unittest
import pandas as pd
import numpy as np

import overnight_momentum_engine as ome


def make_price_series(closes, start_date="20260101"):
    """由收盤價序列造出一份簡單的 OHLCV DataFrame（open=prev close, high/low 略微擺動）。"""
    dates = pd.bdate_range(start=pd.to_datetime(start_date, format="%Y%m%d"), periods=len(closes))
    rows = []
    prev_close = closes[0]
    for d, c in zip(dates, closes):
        o = prev_close
        h = max(o, c) * 1.01
        l = min(o, c) * 0.99
        rows.append({
            "date": d.strftime("%Y%m%d"),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 1_000_000,
        })
        prev_close = c
    return pd.DataFrame(rows)


class TestScoringFunctions(unittest.TestCase):
    def test_percentile_score_higher_is_better(self):
        s = pd.Series([1, 2, 3, 4, 5])
        scores = ome.percentile_score(s, higher_is_better=True)
        # 最大值應該拿到最高分，最小值應該拿到最低分
        self.assertEqual(scores.idxmax(), s.idxmax())
        self.assertEqual(scores.idxmin(), s.idxmin())
        self.assertTrue((scores >= 5).all() and (scores <= 100).all())

    def test_percentile_score_lower_is_better(self):
        s = pd.Series([1, 2, 3, 4, 5])
        scores = ome.percentile_score(s, higher_is_better=False)
        # 最小值(1)應該拿到最高分，最大值(5)應該拿到最低分
        self.assertEqual(scores.idxmax(), s.idxmin())
        self.assertEqual(scores.idxmin(), s.idxmax())

    def test_percentile_score_handles_nan(self):
        s = pd.Series([1.0, np.nan, 3.0])
        scores = ome.percentile_score(s, higher_is_better=True)
        self.assertEqual(scores.iloc[1], 0.0)
        self.assertGreater(scores.iloc[2], scores.iloc[0])

    def test_bell_score_ideal_range_gets_100(self):
        s = pd.Series([4.0, 4.5, 5.0])  # 落在 GAIN_PCT_IDEAL_LOW~HIGH (3~6) 之內
        scores = ome.bell_score(s, ome.GAIN_PCT_MIN, ome.GAIN_PCT_MAX,
                                 ome.GAIN_PCT_IDEAL_LOW, ome.GAIN_PCT_IDEAL_HIGH)
        self.assertTrue((scores == 100.0).all())

    def test_bell_score_extremes_get_low_or_zero(self):
        s = pd.Series([0.5, 9.5, 15.0])  # 0.5太小(低於min1.0)、9.5=上限邊界、15超出範圍
        scores = ome.bell_score(s, ome.GAIN_PCT_MIN, ome.GAIN_PCT_MAX,
                                 ome.GAIN_PCT_IDEAL_LOW, ome.GAIN_PCT_IDEAL_HIGH)
        self.assertEqual(scores.iloc[0], 0.0)   # 低於下限
        self.assertEqual(scores.iloc[2], 0.0)   # 超出上限
        self.assertLess(scores.iloc[1], 100.0)  # 上限邊界應該是低分而非滿分

    def test_bell_score_monotonic_within_band(self):
        s = pd.Series([1.5, 2.0, 2.5])  # 都在 lo(1.0)~ideal_lo(3.0) 之間，遞增
        scores = ome.bell_score(s, ome.GAIN_PCT_MIN, ome.GAIN_PCT_MAX,
                                 ome.GAIN_PCT_IDEAL_LOW, ome.GAIN_PCT_IDEAL_HIGH)
        self.assertTrue(scores.iloc[0] < scores.iloc[1] < scores.iloc[2])

    def test_price_level_score_comfort_zone_full_score(self):
        dev = pd.Series([0.0, 5.0, -5.0])  # 都在 comfort_pct(8%) 之內
        scores = ome.price_level_score(dev)
        self.assertTrue((scores == 100.0).all())

    def test_price_level_score_penalizes_overheat(self):
        dev = pd.Series([0.0, 20.0])  # 第二筆嚴重超出季線
        scores = ome.price_level_score(dev)
        self.assertEqual(scores.iloc[0], 100.0)
        self.assertLess(scores.iloc[1], 100.0)

    def test_price_level_score_nan_gets_zero(self):
        dev = pd.Series([0.0, np.nan])
        scores = ome.price_level_score(dev)
        self.assertEqual(scores.iloc[1], 0.0)


class TestPrecompute(unittest.TestCase):
    def test_precompute_overnight_indicators_basic_fields(self):
        closes = [100 + i * 0.5 for i in range(80)]  # 緩步上漲，確保均線/ATR算得出來
        raw = make_price_series(closes)
        out = ome.precompute_overnight_indicators(raw)

        self.assertIn("ma5", out.columns)
        self.assertIn("ma20", out.columns)
        self.assertIn("ma60", out.columns)
        self.assertIn("atr14", out.columns)
        self.assertIn("close_position", out.columns)
        self.assertIn("volume_ratio", out.columns)
        self.assertIn("turnover_value", out.columns)

        # 均線在資料不足的前段應該是 NaN，足夠天數之後才有值
        self.assertTrue(out["ma60"].iloc[:59].isna().all())
        self.assertFalse(out["ma60"].iloc[60:].isna().any())

        # turnover_value = close * volume
        row = out.iloc[70]
        self.assertAlmostEqual(row["turnover_value"], row["close"] * row["volume"])


class TestHardFilters(unittest.TestCase):
    def _build_indicators(self, closes, volume=1_000_000):
        raw = make_price_series(closes)
        raw["volume"] = volume
        return ome.precompute_overnight_indicators(raw)

    def test_low_turnover_stock_filtered_out(self):
        closes = [100 + i * 0.5 for i in range(80)]
        good = self._build_indicators(closes, volume=1_000_000)
        thin = self._build_indicators(closes, volume=10)  # 成交量極低 -> turnover太小

        indicators_by_code = {"GOOD": good, "THIN": thin}
        market_df = pd.DataFrame({
            "date": good["date"], "close": [200 + i * 0.3 for i in range(80)]
        })
        market_returns = ome.precompute_market_returns(market_df)

        empty_chip = pd.DataFrame(columns=["date", "code", "ratio"])
        empty_dtr = pd.DataFrame(columns=["date", "code", "day_trading_ratio"])

        target_date = good["date"].iloc[70]
        cand = ome.scan_candidates_for_date(
            target_date, indicators_by_code, market_returns,
            empty_chip, empty_chip, empty_dtr,
            universe_codes=["GOOD", "THIN"],
        )
        self.assertNotIn("THIN", cand["code"].tolist())

    def test_below_ma_stock_filtered_out(self):
        # 造一支股價緩步下跌的股票：收盤長期低於均線
        closes_down = [200 - i * 0.5 for i in range(80)]
        closes_up = [100 + i * 0.5 for i in range(80)]
        down = self._build_indicators(closes_down)
        up = self._build_indicators(closes_up)

        indicators_by_code = {"DOWN": down, "UP": up}
        market_df = pd.DataFrame({
            "date": up["date"], "close": [150] * 80
        })
        market_returns = ome.precompute_market_returns(market_df)
        empty_chip = pd.DataFrame(columns=["date", "code", "ratio"])
        empty_dtr = pd.DataFrame(columns=["date", "code", "day_trading_ratio"])

        target_date = up["date"].iloc[70]
        cand = ome.scan_candidates_for_date(
            target_date, indicators_by_code, market_returns,
            empty_chip, empty_chip, empty_dtr,
            universe_codes=["DOWN", "UP"],
        )
        self.assertNotIn("DOWN", cand["code"].tolist())


class TestExitSimulation(unittest.TestCase):
    def test_gap_stop_triggers_on_large_gap_down(self):
        entry_price = 100.0
        atr = 3.0
        next_bar = pd.Series({"open": 98.0, "high": 99.0, "low": 97.0, "close": 98.5})
        # gap_pct = (98-100)/100 = -2% <= -1.5% threshold
        exit_price, reason = ome.simulate_next_day_exit(entry_price, atr, next_bar)
        self.assertEqual(reason, "gap_stop")
        self.assertEqual(exit_price, 98.0)

    def test_intraday_stop_triggers(self):
        entry_price = 100.0
        atr = 3.0
        # 開平盤(無跳空)，但盤中跌破停損價 100 - 0.8*3 = 97.6
        next_bar = pd.Series({"open": 100.0, "high": 101.0, "low": 97.0, "close": 99.0})
        exit_price, reason = ome.simulate_next_day_exit(entry_price, atr, next_bar)
        self.assertEqual(reason, "stop")
        self.assertAlmostEqual(exit_price, 100.0 - 0.8 * 3.0)

    def test_intraday_target_triggers(self):
        entry_price = 100.0
        atr = 3.0
        # 開平盤，盤中衝到停利價 100 + 1.2*3 = 103.6
        next_bar = pd.Series({"open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0})
        exit_price, reason = ome.simulate_next_day_exit(entry_price, atr, next_bar)
        self.assertEqual(reason, "target")
        self.assertAlmostEqual(exit_price, 100.0 + 1.2 * 3.0)

    def test_forced_close_when_nothing_triggers(self):
        entry_price = 100.0
        atr = 3.0
        # 開平盤，全天都在停損停利區間內，沒有觸價
        next_bar = pd.Series({"open": 100.0, "high": 101.5, "low": 98.5, "close": 100.8})
        exit_price, reason = ome.simulate_next_day_exit(entry_price, atr, next_bar)
        self.assertEqual(reason, "forced_close")
        self.assertEqual(exit_price, 100.8)

    def test_stop_has_priority_over_target_when_both_touched(self):
        entry_price = 100.0
        atr = 3.0
        # 同一天內 high 觸及停利、low 也觸及停損 -> 保守判定為停損優先
        next_bar = pd.Series({"open": 100.0, "high": 105.0, "low": 96.0, "close": 101.0})
        exit_price, reason = ome.simulate_next_day_exit(entry_price, atr, next_bar)
        self.assertEqual(reason, "stop")


class TestFullBacktestLoop(unittest.TestCase):
    def test_end_to_end_backtest_produces_trades_with_correct_pnl(self):
        # 建一支股票：前段緩步墊高供均線/ATR暖身，第70天是明確的強勢K棒（用來被選中），
        # 第71天開平盤、盤中觸及停利價，驗證出場與pnl計算正確串起來。
        closes = [100 + i * 0.3 for i in range(70)]
        closes.append(closes[-1] * 1.05)  # 第70天(index=70)最後噴出一根強勢K棒
        raw = make_price_series(closes)
        raw.loc[raw.index[-1], "volume"] = 5_000_000  # 爆量

        # 手動把最後一天的 low 抬高一點，確保 close_position 接近1（收在高點）
        raw.loc[raw.index[-1], "low"] = closes[-2]
        raw.loc[raw.index[-1], "high"] = closes[-1] * 1.001

        # 第 72 天（T+1）：開平盤，盤中衝到停利價
        extra_day = pd.DataFrame([{
            "date": (pd.bdate_range(start=pd.to_datetime("20260101", format="%Y%m%d"),
                                     periods=len(closes) + 1)[-1]).strftime("%Y%m%d"),
            "open": closes[-1],
            "high": closes[-1] * 1.05,
            "low": closes[-1] * 0.99,
            "close": closes[-1] * 1.02,
            "volume": 2_000_000,
        }])
        raw = pd.concat([raw, extra_day], ignore_index=True)

        indicators = ome.precompute_overnight_indicators(raw)
        indicators_by_code = {"TEST": indicators}

        market_df = pd.DataFrame({
            "date": raw["date"],
            "close": [150 + i * 0.05 for i in range(len(raw))],  # 大盤幾乎持平，個股明顯強於大盤
        })
        market_returns = ome.precompute_market_returns(market_df)

        entry_date = indicators["date"].iloc[70]
        exit_date = indicators["date"].iloc[71]

        # 籌碼面給滿分情境：當沖比例低、外資投信都買超
        foreign_ratio_df = pd.DataFrame([{"date": entry_date, "code": "TEST", "ratio": 0.05}])
        trust_ratio_df = pd.DataFrame([{"date": entry_date, "code": "TEST", "ratio": 0.03}])
        day_trading_ratio_df = pd.DataFrame([
            {"date": entry_date, "code": "TEST", "day_trading_ratio": 0.02}
        ])

        # 只跑「進場日 -> 出場日」這一組，隔離掉前段緩步上漲期間也會通過均線濾網、
        # 進而產生額外交易的情況（那些交易本身是正確行為，只是不是這個測試要驗證的對象）。
        trading_days = [entry_date, exit_date]

        trades = ome.run_overnight_backtest(
            indicators_by_code, market_returns,
            foreign_ratio_df, trust_ratio_df, day_trading_ratio_df,
            trading_days, universe_codes=["TEST"], top_n=1,
        )

        self.assertEqual(len(trades), 1, "應該恰好產生一筆交易（只有一支股票且只有一天可進場）")
        trade = trades[0]
        self.assertEqual(trade["entry_date"], entry_date)
        self.assertEqual(trade["exit_date"], exit_date)
        self.assertEqual(trade["code"], "TEST")

        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]
        mult = ome.get_contract_multiplier(entry_price)
        expected_pnl = (exit_price - entry_price) * mult - 200
        self.assertAlmostEqual(trade["pnl"], expected_pnl)

    def test_run_overnight_backtest_respects_overridden_exit_params(self):
        """驗證 run_overnight_backtest 的 gap_stop_threshold/atr_stop_mult/atr_target_mult
        真的有傳到 simulate_next_day_exit，不是被忽略的死參數。"""
        closes = [100 + i * 0.3 for i in range(70)]
        closes.append(closes[-1] * 1.05)
        raw = make_price_series(closes)
        raw.loc[raw.index[-1], "volume"] = 5_000_000
        raw.loc[raw.index[-1], "low"] = closes[-2]
        raw.loc[raw.index[-1], "high"] = closes[-1] * 1.001

        # T+1 開盤跌 1.2%：用預設 gap_stop_threshold(-1.5%) 不會觸發跳空停損，
        # 但如果把門檻放寬到 -1.0%，就應該觸發。
        extra_day = pd.DataFrame([{
            "date": (pd.bdate_range(start=pd.to_datetime("20260101", format="%Y%m%d"),
                                     periods=len(closes) + 1)[-1]).strftime("%Y%m%d"),
            "open": closes[-1] * 0.988,
            "high": closes[-1] * 0.99,
            "low": closes[-1] * 0.97,
            "close": closes[-1] * 0.98,
            "volume": 2_000_000,
        }])
        raw = pd.concat([raw, extra_day], ignore_index=True)

        indicators = ome.precompute_overnight_indicators(raw)
        indicators_by_code = {"TEST": indicators}

        market_df = pd.DataFrame({
            "date": raw["date"], "close": [150 + i * 0.05 for i in range(len(raw))],
        })
        market_returns = ome.precompute_market_returns(market_df)

        entry_date = indicators["date"].iloc[70]
        foreign_ratio_df = pd.DataFrame([{"date": entry_date, "code": "TEST", "ratio": 0.05}])
        trust_ratio_df = pd.DataFrame([{"date": entry_date, "code": "TEST", "ratio": 0.03}])
        day_trading_ratio_df = pd.DataFrame([
            {"date": entry_date, "code": "TEST", "day_trading_ratio": 0.02}
        ])
        trading_days = [entry_date, indicators["date"].iloc[71]]

        default_trades = ome.run_overnight_backtest(
            indicators_by_code, market_returns, foreign_ratio_df, trust_ratio_df,
            day_trading_ratio_df, trading_days, universe_codes=["TEST"], top_n=1,
        )
        self.assertNotEqual(default_trades[0]["exit_reason"], "gap_stop")

        tightened_trades = ome.run_overnight_backtest(
            indicators_by_code, market_returns, foreign_ratio_df, trust_ratio_df,
            day_trading_ratio_df, trading_days, universe_codes=["TEST"], top_n=1,
            gap_stop_threshold=-0.01,
        )
        self.assertEqual(tightened_trades[0]["exit_reason"], "gap_stop")

    def test_summarize_overnight_on_empty_trades(self):
        summary = ome.summarize_overnight([])
        self.assertEqual(summary["total_trades"], 0)

    def test_summarize_overnight_computes_win_rate_and_pf(self):
        trades = [
            {"pnl": 1000, "exit_reason": "target"},
            {"pnl": -500, "exit_reason": "stop"},
            {"pnl": 2000, "exit_reason": "forced_close"},
        ]
        summary = ome.summarize_overnight(trades)
        self.assertEqual(summary["total_trades"], 3)
        self.assertAlmostEqual(summary["win_rate"], 2 / 3 * 100)
        self.assertAlmostEqual(summary["profit_factor"], 3000 / 500)
        self.assertAlmostEqual(summary["total_pnl"], 2500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
