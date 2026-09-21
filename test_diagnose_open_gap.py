"""
test_diagnose_open_gap.py
============================
針對 diagnose_open_gap.py 的合成資料測試。不連網路、不重下載。驗證：
  1. run_backtest_with_open_diagnostics() 正確查出每筆交易「隔天開盤價」，
     並正確判斷有沒有開紅、正確算出假設在開盤出場的損益
  2. summarize_by_open_gap() 依 opened_up 分組，統計數字正確
  3. exit_reason_breakdown_for_group() 只看指定那一組的出場原因分佈，加總為100%
  4. 對完整合成 pipeline 跑一次不會出錯（整合測試，只在 IS 內跑）
"""

import unittest
import datetime
import pandas as pd
import numpy as np

import diagnose_open_gap as diag
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


class TestRunBacktestWithOpenDiagnostics(unittest.TestCase):
    def test_enriches_trades_with_correct_open_classification_and_pnl(self):
        indicators_by_code = {
            "AAAA": pd.DataFrame({
                "date": ["20260101", "20260102", "20260103"],
                "open": [100.0, 105.0, 95.0],
            }),
        }
        # 一筆交易 20260101 進場，20260102 出場（隔天開盤105 > 進場價100 => 開紅）
        # 另一筆 20260102 進場，20260103 出場（隔天開盤95 <= 進場價100 => 沒開紅）
        fake_trades = [
            {"code": "AAAA", "entry_date": "20260101", "exit_date": "20260102",
             "entry_price": 100.0, "exit_price": 102.0, "exit_reason": "forced_close",
             "final_score": 50.0, "technical_score": 50.0, "chip_score": 50.0, "pnl": 1800.0},
            {"code": "AAAA", "entry_date": "20260102", "exit_date": "20260103",
             "entry_price": 100.0, "exit_price": 96.0, "exit_reason": "stop",
             "final_score": 50.0, "technical_score": 50.0, "chip_score": 50.0, "pnl": -4200.0},
        ]

        real_run = ome.run_overnight_backtest
        ome.run_overnight_backtest = lambda **kwargs: fake_trades
        try:
            enriched = diag.run_backtest_with_open_diagnostics(
                indicators_by_code=indicators_by_code,
                market_returns_df=pd.DataFrame(), foreign_ratio_df=pd.DataFrame(),
                trust_ratio_df=pd.DataFrame(), day_trading_ratio_df=pd.DataFrame(),
                trading_days=["20260101", "20260102", "20260103"],
            )
        finally:
            ome.run_overnight_backtest = real_run

        self.assertEqual(len(enriched), 2)

        row0 = enriched.iloc[0]
        self.assertEqual(row0["next_open"], 105.0)
        self.assertTrue(row0["opened_up"])
        mult = ome.get_contract_multiplier(100.0)
        expected_hyp_pnl_0 = (105.0 - 100.0) * mult - 200
        self.assertAlmostEqual(row0["hypothetical_open_exit_pnl"], expected_hyp_pnl_0)

        row1 = enriched.iloc[1]
        self.assertEqual(row1["next_open"], 95.0)
        self.assertFalse(row1["opened_up"])
        expected_hyp_pnl_1 = (95.0 - 100.0) * mult - 200
        self.assertAlmostEqual(row1["hypothetical_open_exit_pnl"], expected_hyp_pnl_1)

    def test_trade_with_no_matching_next_row_is_dropped(self):
        indicators_by_code = {
            "AAAA": pd.DataFrame({"date": ["20260101"], "open": [100.0]}),
        }
        fake_trades = [
            {"code": "AAAA", "entry_date": "20260101", "exit_date": "20260199",  # 找不到的日期
             "entry_price": 100.0, "exit_price": 102.0, "exit_reason": "forced_close",
             "final_score": 50.0, "technical_score": 50.0, "chip_score": 50.0, "pnl": 1800.0},
        ]
        real_run = ome.run_overnight_backtest
        ome.run_overnight_backtest = lambda **kwargs: fake_trades
        try:
            enriched = diag.run_backtest_with_open_diagnostics(
                indicators_by_code=indicators_by_code,
                market_returns_df=pd.DataFrame(), foreign_ratio_df=pd.DataFrame(),
                trust_ratio_df=pd.DataFrame(), day_trading_ratio_df=pd.DataFrame(),
                trading_days=["20260101"],
            )
        finally:
            ome.run_overnight_backtest = real_run
        self.assertTrue(enriched.empty)


class TestSummarizeByOpenGap(unittest.TestCase):
    def test_groups_and_averages_correctly(self):
        enriched = pd.DataFrame([
            {"opened_up": True, "pnl": 1000.0, "hypothetical_open_exit_pnl": 500.0},
            {"opened_up": True, "pnl": 2000.0, "hypothetical_open_exit_pnl": 700.0},
            {"opened_up": False, "pnl": -1000.0, "hypothetical_open_exit_pnl": -200.0},
            {"opened_up": False, "pnl": -3000.0, "hypothetical_open_exit_pnl": -400.0},
        ])
        summary = diag.summarize_by_open_gap(enriched)
        self.assertEqual(len(summary), 2)

        up_row = summary[summary["opened_up"] == True].iloc[0]
        self.assertEqual(up_row["trade_count"], 2)
        self.assertAlmostEqual(up_row["avg_actual_pnl"], 1500.0)
        self.assertAlmostEqual(up_row["avg_hypothetical_open_exit_pnl"], 600.0)

        down_row = summary[summary["opened_up"] == False].iloc[0]
        self.assertEqual(down_row["trade_count"], 2)
        self.assertAlmostEqual(down_row["avg_actual_pnl"], -2000.0)
        self.assertAlmostEqual(down_row["avg_hypothetical_open_exit_pnl"], -300.0)

    def test_empty_input_returns_empty_with_columns(self):
        summary = diag.summarize_by_open_gap(pd.DataFrame())
        self.assertTrue(summary.empty)
        for col in ["opened_up", "trade_count", "avg_actual_pnl",
                    "avg_hypothetical_open_exit_pnl", "total_actual_pnl"]:
            self.assertIn(col, summary.columns)


class TestExitReasonBreakdown(unittest.TestCase):
    def test_breakdown_sums_to_100_for_specified_group(self):
        enriched = pd.DataFrame([
            {"opened_up": False, "exit_reason": "forced_close"},
            {"opened_up": False, "exit_reason": "forced_close"},
            {"opened_up": False, "exit_reason": "stop"},
            {"opened_up": True, "exit_reason": "target"},  # 不該被算進去
        ])
        breakdown = diag.exit_reason_breakdown_for_group(enriched, opened_up_value=False)
        self.assertAlmostEqual(breakdown.sum(), 100.0)
        self.assertAlmostEqual(breakdown["forced_close"], 66.7, places=1)

    def test_empty_group_returns_empty_series(self):
        enriched = pd.DataFrame([
            {"opened_up": True, "exit_reason": "target"},
        ])
        breakdown = diag.exit_reason_breakdown_for_group(enriched, opened_up_value=False)
        self.assertTrue(breakdown.empty)


class TestIntegrationWithSyntheticPipeline(unittest.TestCase):
    def test_full_run_does_not_crash_and_only_uses_is_days(self):
        pipeline_inputs = make_pipeline_inputs()
        is_days, oos_days = co.split_is_oos(pipeline_inputs["trading_days"])

        enriched = diag.run_backtest_with_open_diagnostics(
            indicators_by_code=pipeline_inputs["indicators_by_code"],
            market_returns_df=pipeline_inputs["market_returns_df"],
            foreign_ratio_df=pipeline_inputs["foreign_ratio_df"],
            trust_ratio_df=pipeline_inputs["trust_ratio_df"],
            day_trading_ratio_df=pipeline_inputs["day_trading_ratio_df"],
            universe_codes=pipeline_inputs["universe_codes"],
            us_market_returns_df=pipeline_inputs.get("us_market_returns_df"),
            trading_days=is_days,
        )

        self.assertFalse(enriched.empty)
        for col in ["next_open", "opened_up", "hypothetical_open_exit_pnl"]:
            self.assertIn(col, enriched.columns)

        # 所有交易的 exit_date 都應該落在 is_days 範圍內，不會偷跑到 OOS
        self.assertTrue(set(enriched["exit_date"]).issubset(set(is_days)))

        summary = diag.summarize_by_open_gap(enriched)
        self.assertGreater(len(summary), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
