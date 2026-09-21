"""
test_sweep_signal_ablation.py
=================================
針對 sweep_signal_ablation.py 的合成資料測試。不連網路、不重下載，
用跟 test_sweep_overnight_params.py 一樣的合成資料建構手法，驗證：
  1. run_ablation() 對 SIGNAL_LABELS 裡每個訊號都跑一次，回傳的 DataFrame
     包含每個訊號 + 一個基準(等權重)列，欄位完整
  2. run_ablation() 只用 IS 區間跑（不會偷看 OOS）
  3. rank_results() 會濾掉交易筆數太少的組合，且依 profit_factor 由高到低排序
  4. 單獨開啟不同訊號，選出的候選股會不同（訊號真的有被隔離，不是每次都選一樣的）
"""

import unittest
import datetime
import pandas as pd
import numpy as np

import sweep_signal_ablation as ablation
import compare_overnight as co
import overnight_momentum_engine as ome


def make_ohlcv_df(n_days, base_price=100.0, start="2026-01-01", trend=0.3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    closes = base_price + np.cumsum(rng.normal(trend, 0.5, size=n_days))
    closes = np.maximum(closes, 1.0)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * 1.01
    lows = np.minimum(opens, closes) * 0.99
    volumes = rng.integers(500_000, 2_000_000, size=n_days)
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes,
    }, index=dates)
    df.index.name = "Date"
    return df


def make_chip_df(n_days, start="2026-01-01", seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    df = pd.DataFrame({
        "foreign_net": rng.integers(-100_000, 100_000, size=n_days),
        "trust_net": rng.integers(-50_000, 50_000, size=n_days),
        "dealer_net": rng.integers(-10_000, 10_000, size=n_days),
    }, index=dates)
    df["total_net"] = df["foreign_net"] + df["trust_net"] + df["dealer_net"]
    df.index.name = "date"
    return df


class FakeDayTradingLoader:
    def __init__(self, codes, n_days, start="2026-01-01", seed=2):
        self.codes = codes
        self.n_days = n_days
        self.start = start
        self.seed = seed

    def __call__(self, start_date, end_date, universe_codes=None, refresh=False):
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range(start=self.start, periods=self.n_days)
        rows = []
        for d in dates:
            for code in self.codes:
                if universe_codes is not None and code not in universe_codes:
                    continue
                rows.append({
                    "date": d.strftime("%Y%m%d"), "code": code, "name": code,
                    "day_trading_shares": int(rng.integers(0, 50_000)),
                })
        return pd.DataFrame(rows)


def make_pipeline_inputs(n_days=100, codes=("2330", "2317", "2454")):
    def fake_price_loader(whitelist, start, end, refresh=False):
        return {code: make_ohlcv_df(n_days, base_price=100 + i * 50, seed=i)
                for i, code in enumerate(codes)}

    def fake_chip_loader(start, end, universe_codes=None, refresh=False):
        data = {code: make_chip_df(n_days, seed=i) for i, code in enumerate(codes)}
        if universe_codes is not None:
            data = {c: df for c, df in data.items() if c in universe_codes}
        return data

    def fake_us_market_loader(start, end, refresh=False):
        dates = pd.bdate_range(start="2026-01-01", periods=n_days)
        return pd.DataFrame({
            "date": [d.strftime("%Y%m%d") for d in dates],
            "close": [4000.0] * n_days,
            "return_pct": [0.1] * n_days,
        })

    whitelist = {c: c for c in codes}
    return co.build_pipeline_inputs(
        whitelist, datetime.date(2026, 1, 1), datetime.date(2026, 6, 1),
        price_loader=fake_price_loader, chip_loader=fake_chip_loader,
        day_trading_loader_fn=FakeDayTradingLoader(list(codes), n_days),
        us_market_loader_fn=fake_us_market_loader,
    )


class TestRunAblation(unittest.TestCase):
    def setUp(self):
        self.pipeline_inputs = make_pipeline_inputs()

    def test_returns_one_row_per_signal_plus_baseline(self):
        result_df = ablation.run_ablation(self.pipeline_inputs, top_n=2)

        # 8個訊號 + 1個基準列
        self.assertEqual(len(result_df), len(ablation.SIGNAL_LABELS) + 1)

        expected_signals = set(ablation.SIGNAL_LABELS.keys()) | {"__baseline_equal_weight__"}
        self.assertEqual(set(result_df["signal"]), expected_signals)

        for col in ["signal", "label", "total_trades", "win_rate",
                    "profit_factor", "total_pnl", "avg_pnl"]:
            self.assertIn(col, result_df.columns)

    def test_only_uses_is_days_not_oos(self):
        full_days = self.pipeline_inputs["trading_days"]
        expected_is, expected_oos = co.split_is_oos(full_days)

        # 用一個很短的 IS 天數，藉由跑出來的最大交易筆數間接確認範圍沒有包含 OOS。
        # 更直接的作法：monkeypatch run_overnight_backtest 記錄實際傳入的 trading_days。
        captured = []
        real_run = ome.run_overnight_backtest

        def spy_run_overnight_backtest(**kwargs):
            captured.append(kwargs["trading_days"])
            return real_run(**kwargs)

        ome.run_overnight_backtest = spy_run_overnight_backtest
        try:
            ablation.run_ablation(self.pipeline_inputs, top_n=2)
        finally:
            ome.run_overnight_backtest = real_run

        for days_used in captured:
            self.assertEqual(days_used, expected_is)
            self.assertTrue(set(days_used).isdisjoint(set(expected_oos)))


class TestRankResults(unittest.TestCase):
    def test_filters_low_trade_count_and_sorts_by_pf(self):
        df = pd.DataFrame([
            {"signal": "a", "label": "A", "total_trades": 5, "profit_factor": 5.0},
            {"signal": "b", "label": "B", "total_trades": 100, "profit_factor": 1.2},
            {"signal": "c", "label": "C", "total_trades": 200, "profit_factor": 2.5},
        ])
        ranked = ablation.rank_results(df, min_trades=30)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked.iloc[0]["signal"], "c")
        self.assertEqual(ranked.iloc[1]["signal"], "b")

    def test_falls_back_to_all_when_none_reliable(self):
        df = pd.DataFrame([
            {"signal": "a", "label": "A", "total_trades": 2, "profit_factor": 5.0},
            {"signal": "b", "label": "B", "total_trades": 1, "profit_factor": 1.2},
        ])
        ranked = ablation.rank_results(df, min_trades=30)
        self.assertEqual(len(ranked), 2)


class TestSignalIsolationChangesSelection(unittest.TestCase):
    """驗證單一訊號拆解真的有隔離效果：兩檔股票在不同訊號上表現相反時，
    單獨開啟不同訊號應該選出不同的股票，而不是每次都選一樣的（代表 signal_weights 沒生效）。"""

    def test_opposite_signals_flip_top_pick(self):
        n_days = 60
        dates = pd.bdate_range(start="2026-01-01", periods=n_days)

        # code A: 技術面強（大漲、量大），但外資賣超
        # code B: 技術面弱，但外資大買超
        def build_price_data():
            price_data = {}
            for code, gain in (("AAAA", 6.0), ("BBBB", 0.5)):
                base = 100.0
                closes = [base]
                for _ in range(n_days - 1):
                    closes.append(closes[-1] * (1 + gain / 100 / 5))
                closes = np.array(closes)
                opens = np.concatenate([[closes[0]], closes[:-1]])
                highs = np.maximum(opens, closes) * 1.01
                lows = np.minimum(opens, closes) * 0.99
                volumes = np.full(n_days, 1_000_000)
                df = pd.DataFrame({
                    "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes,
                }, index=dates)
                df.index.name = "Date"
                price_data[code] = df
            return price_data

        foreign_sign_by_code = {"AAAA": -1, "BBBB": 1}
        indicators_by_code = co.convert_price_data_to_indicators(build_price_data())

        market_returns_df = ome.precompute_market_returns(pd.DataFrame({
            "date": [d.strftime("%Y%m%d") for d in dates],
            "close": [4000.0 + i for i in range(n_days)],
        }))
        us_market_returns_df = pd.DataFrame({
            "date": [d.strftime("%Y%m%d") for d in dates],
            "close": [4000.0] * n_days,
            "return_pct": [0.1] * n_days,
        })

        foreign_rows, trust_rows, day_trading_rows = [], [], []
        for d in dates:
            for code, sign in foreign_sign_by_code.items():
                foreign_rows.append({"date": d.strftime("%Y%m%d"), "code": code, "ratio": sign * 5.0})
                trust_rows.append({"date": d.strftime("%Y%m%d"), "code": code, "ratio": 0.0})
                day_trading_rows.append({"date": d.strftime("%Y%m%d"), "code": code, "day_trading_ratio": 5.0})
        foreign_ratio_df = pd.DataFrame(foreign_rows)
        trust_ratio_df = pd.DataFrame(trust_rows)
        day_trading_ratio_df = pd.DataFrame(day_trading_rows)

        test_date = dates[30].strftime("%Y%m%d")

        picks_tech = ome.scan_candidates_for_date(
            test_date, indicators_by_code, market_returns_df,
            foreign_ratio_df, trust_ratio_df, day_trading_ratio_df,
            top_n=2, us_market_returns_df=us_market_returns_df,
            signal_weights={"score_gain_pct": 1.0},
        )
        picks_chip = ome.scan_candidates_for_date(
            test_date, indicators_by_code, market_returns_df,
            foreign_ratio_df, trust_ratio_df, day_trading_ratio_df,
            top_n=2, us_market_returns_df=us_market_returns_df,
            signal_weights={"score_foreign": 1.0},
        )

        self.assertFalse(picks_tech.empty)
        self.assertFalse(picks_chip.empty)
        self.assertEqual(picks_tech.iloc[0]["code"], "AAAA")  # 技術面訊號應該選漲多的 AAAA
        self.assertEqual(picks_chip.iloc[0]["code"], "BBBB")  # 籌碼(外資)訊號應該選外資買超的 BBBB


if __name__ == "__main__":
    unittest.main(verbosity=2)
