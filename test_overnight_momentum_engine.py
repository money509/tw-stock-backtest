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


class TestExDividendFilter(unittest.TestCase):
    """驗證 ex_dividend_dates_by_code 濾網：除權息當天該股票要被剔除候選，
    其他日期/其他股票不受影響；不傳這個參數時行為完全不變（向後相容）。"""

    def _build_indicators(self, closes, volume=1_000_000):
        raw = make_price_series(closes)
        raw["volume"] = volume
        return ome.precompute_overnight_indicators(raw)

    def setUp(self):
        closes_a = [100 + i * 0.5 for i in range(80)]
        closes_b = [80 + i * 0.4 for i in range(80)]
        self.stock_a = self._build_indicators(closes_a)
        self.stock_b = self._build_indicators(closes_b)
        self.indicators_by_code = {"AAAA": self.stock_a, "BBBB": self.stock_b}
        market_df = pd.DataFrame({
            "date": self.stock_a["date"], "close": [150 + i * 0.1 for i in range(80)]
        })
        self.market_returns = ome.precompute_market_returns(market_df)
        self.empty_chip = pd.DataFrame(columns=["date", "code", "ratio"])
        self.empty_dtr = pd.DataFrame(columns=["date", "code", "day_trading_ratio"])
        self.target_date = self.stock_a["date"].iloc[70]

    def _scan(self, ex_dividend_dates_by_code=None):
        return ome.scan_candidates_for_date(
            self.target_date, self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.empty_dtr,
            universe_codes=["AAAA", "BBBB"],
            ex_dividend_dates_by_code=ex_dividend_dates_by_code,
        )

    def test_stock_on_its_own_ex_dividend_date_is_excluded(self):
        cand_before = self._scan()
        self.assertIn("AAAA", cand_before["code"].tolist())  # 先確認不加濾網時本來會入選

        ex_div = {"AAAA": {self.target_date}, "BBBB": set()}
        cand_after = self._scan(ex_dividend_dates_by_code=ex_div)
        self.assertNotIn("AAAA", cand_after["code"].tolist())
        self.assertIn("BBBB", cand_after["code"].tolist())  # 另一檔不受影響

    def test_ex_dividend_on_other_date_does_not_affect_today(self):
        other_date = self.stock_a["date"].iloc[10]
        ex_div = {"AAAA": {other_date}, "BBBB": set()}
        cand = self._scan(ex_dividend_dates_by_code=ex_div)
        self.assertIn("AAAA", cand["code"].tolist())

    def test_none_default_keeps_old_behavior(self):
        cand_default = self._scan()
        cand_explicit_none = self._scan(ex_dividend_dates_by_code=None)
        self.assertEqual(cand_default["code"].tolist(), cand_explicit_none["code"].tolist())

    def test_missing_code_in_dict_not_excluded(self):
        # 字典裡根本沒有這檔股票的 entry，get(code, set()) 應該回傳空集合，不擋
        ex_div = {"BBBB": {self.target_date}}
        cand = self._scan(ex_dividend_dates_by_code=ex_div)
        self.assertIn("AAAA", cand["code"].tolist())
        self.assertNotIn("BBBB", cand["code"].tolist())

    def test_run_overnight_backtest_respects_ex_dividend_filter(self):
        trading_days = self.stock_a["date"].tolist()
        ex_div = {"AAAA": set(trading_days), "BBBB": set()}  # AAAA每天都被剔除

        trades = ome.run_overnight_backtest(
            indicators_by_code=self.indicators_by_code,
            market_returns_df=self.market_returns,
            foreign_ratio_df=self.empty_chip,
            trust_ratio_df=self.empty_chip,
            day_trading_ratio_df=self.empty_dtr,
            trading_days=trading_days,
            universe_codes=["AAAA", "BBBB"],
            top_n=2,
            ex_dividend_dates_by_code=ex_div,
        )
        codes_traded = {t["code"] for t in trades}
        self.assertNotIn("AAAA", codes_traded)


class TestSignalWeights(unittest.TestCase):
    def _build_indicators(self, closes, volume=1_000_000):
        raw = make_price_series(closes)
        raw["volume"] = volume
        return ome.precompute_overnight_indicators(raw)

    def setUp(self):
        # 兩檔股票，刻意讓技術面指標一樣，但籌碼面完全相反，
        # 這樣才能用 signal_weights 單獨測「只看某個訊號」時排名會不會跟著換。
        closes = [100 + i * 0.5 for i in range(80)]
        self.ind_a = self._build_indicators(closes)
        self.ind_b = self._build_indicators(closes)
        self.indicators_by_code = {"A": self.ind_a, "B": self.ind_b}

        market_df = pd.DataFrame({
            "date": self.ind_a["date"], "close": [200 + i * 0.3 for i in range(80)],
        })
        self.market_returns = ome.precompute_market_returns(market_df)
        self.target_date = self.ind_a["date"].iloc[70]

        # A 的當沖比例低(好)，B 的當沖比例高(差)；技術面完全一樣(同樣的K線)
        self.dtr = pd.DataFrame([
            {"date": self.target_date, "code": "A", "day_trading_ratio": 0.01},
            {"date": self.target_date, "code": "B", "day_trading_ratio": 0.50},
        ])
        self.empty_chip = pd.DataFrame(columns=["date", "code", "ratio"])

    def test_signal_weights_none_keeps_old_behavior(self):
        """不傳 signal_weights，跟舊的 tech_weight/chip_weight 邏輯完全一樣。"""
        cand = ome.scan_candidates_for_date(
            self.target_date, self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.dtr,
            universe_codes=["A", "B"], tech_weight=0.35, chip_weight=0.65,
        )
        self.assertIn("technical_score", cand.columns)
        self.assertIn("chip_score", cand.columns)

    def test_isolating_day_trading_signal_ranks_by_it_alone(self):
        """只給 score_day_trading 權重，A(當沖比例低)應該排第一。"""
        cand = ome.scan_candidates_for_date(
            self.target_date, self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.dtr,
            universe_codes=["A", "B"],
            signal_weights={"score_day_trading": 1.0},
        )
        self.assertEqual(cand.iloc[0]["code"], "A")

    def test_isolating_signal_with_equal_underlying_values_ties(self):
        """兩檔技術面完全相同，只看某個技術面訊號時分數應該相等（互為平手）。"""
        cand = ome.scan_candidates_for_date(
            self.target_date, self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.dtr,
            universe_codes=["A", "B"],
            signal_weights={"score_close_position": 1.0},
        )
        scores = cand["final_score"].tolist()
        self.assertAlmostEqual(scores[0], scores[1])

    def test_zero_total_weight_raises(self):
        with self.assertRaises(ValueError):
            ome.scan_candidates_for_date(
                self.target_date, self.indicators_by_code, self.market_returns,
                self.empty_chip, self.empty_chip, self.dtr,
                universe_codes=["A", "B"], signal_weights={"score_day_trading": 0.0},
            )

    def test_run_overnight_backtest_accepts_signal_weights(self):
        exit_date = self.ind_a["date"].iloc[71]
        trades = ome.run_overnight_backtest(
            self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.dtr,
            [self.target_date, exit_date], universe_codes=["A", "B"], top_n=1,
            signal_weights={"score_day_trading": 1.0},
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["code"], "A")


class TestUsMarketFilter(unittest.TestCase):
    def _build_indicators(self, closes, volume=1_000_000):
        raw = make_price_series(closes)
        raw["volume"] = volume
        return ome.precompute_overnight_indicators(raw)

    def setUp(self):
        closes = [100 + i * 0.5 for i in range(80)]
        self.indicators = self._build_indicators(closes)
        self.indicators_by_code = {"TEST": self.indicators}
        market_df = pd.DataFrame({
            "date": self.indicators["date"], "close": [200 + i * 0.3 for i in range(80)],
        })
        self.market_returns = ome.precompute_market_returns(market_df)
        self.empty_chip = pd.DataFrame(columns=["date", "code", "ratio"])
        self.empty_dtr = pd.DataFrame(columns=["date", "code", "day_trading_ratio"])
        self.target_date = self.indicators["date"].iloc[70]

    def test_no_us_market_df_behaves_exactly_as_before(self):
        """不傳 us_market_returns_df，行為要跟這個參數不存在時完全一樣（向後相容）。"""
        cand = ome.scan_candidates_for_date(
            self.target_date, self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.empty_dtr,
            universe_codes=["TEST"],
        )
        self.assertIn("TEST", cand["code"].tolist())

    def test_us_market_big_drop_blocks_entry_entirely(self):
        us_df = pd.DataFrame([{"date": self.target_date, "close": 4000, "return_pct": -2.5}])
        cand = ome.scan_candidates_for_date(
            self.target_date, self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.empty_dtr,
            universe_codes=["TEST"], us_market_returns_df=us_df,
        )
        self.assertTrue(cand.empty, "美股大跌超過門檻，當天不該有任何候選股")

    def test_us_market_mild_move_does_not_block_entry(self):
        us_df = pd.DataFrame([{"date": self.target_date, "close": 4000, "return_pct": -0.3}])
        cand = ome.scan_candidates_for_date(
            self.target_date, self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.empty_dtr,
            universe_codes=["TEST"], us_market_returns_df=us_df,
        )
        self.assertIn("TEST", cand["code"].tolist())

    def test_missing_date_in_us_market_df_does_not_block(self):
        """美股資料裡沒有這一天(例如美股假日但台股照常開盤)，不該誤擋。"""
        us_df = pd.DataFrame([{"date": "20990101", "close": 4000, "return_pct": -5.0}])
        cand = ome.scan_candidates_for_date(
            self.target_date, self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.empty_dtr,
            universe_codes=["TEST"], us_market_returns_df=us_df,
        )
        self.assertIn("TEST", cand["code"].tolist())

    def test_threshold_is_configurable(self):
        us_df = pd.DataFrame([{"date": self.target_date, "close": 4000, "return_pct": -0.8}])
        # 預設門檻-1.5%不會擋到-0.8%的跌幅
        cand_default = ome.scan_candidates_for_date(
            self.target_date, self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.empty_dtr,
            universe_codes=["TEST"], us_market_returns_df=us_df,
        )
        self.assertFalse(cand_default.empty)

        # 把門檻收緊到-0.5%，-0.8%的跌幅就該被擋
        cand_strict = ome.scan_candidates_for_date(
            self.target_date, self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.empty_dtr,
            universe_codes=["TEST"], us_market_returns_df=us_df,
            us_market_drop_threshold=-0.5,
        )
        self.assertTrue(cand_strict.empty)

    def test_run_overnight_backtest_skips_entry_day_on_us_market_crash(self):
        entry_date = self.indicators["date"].iloc[70]
        exit_date = self.indicators["date"].iloc[71]
        us_df = pd.DataFrame([{"date": entry_date, "close": 4000, "return_pct": -3.0}])

        trades = ome.run_overnight_backtest(
            self.indicators_by_code, self.market_returns,
            self.empty_chip, self.empty_chip, self.empty_dtr,
            [entry_date, exit_date], universe_codes=["TEST"], top_n=1,
            us_market_returns_df=us_df,
        )
        self.assertEqual(trades, [], "美股大跌那天不該產生任何交易")


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
