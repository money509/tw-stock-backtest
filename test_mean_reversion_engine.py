"""
test_mean_reversion_engine.py
==============================
這個模組原本完全沒有單元測試(README裡誠實列過這個已知風險)。這次因為要修改
共用的出場機制(check_exit/_process_mr_day)、新增移動停利、以及擴充summarize_mr()
的統計量，這些都是均值回歸跟右側突破兩套引擎共用的核心邏輯，所以先補上針對
「這次改動/新增的部分」的測試，不是補齊整個模組的完整覆蓋率。
"""
import numpy as np
import pandas as pd
import pytest

from mean_reversion_engine import (
    check_exit, update_trailing_stop, summarize_mr, _max_consecutive_losses,
)


def make_row(open_=100.0, high=101.0, low=99.0, close=100.5):
    return pd.Series({"Open": open_, "High": high, "Low": low, "Close": close})


class TestCheckExit:
    def test_long_normal_stop_uses_theoretical_stop_price(self):
        position = {"side": "long", "stop_price": 98.0, "target_price": 105.0}
        row = make_row(open_=100.0, high=101.0, low=97.5, close=98.0)
        event, price = check_exit(row, position)
        assert event == "stop"
        assert price == 98.0

    def test_long_gap_through_stop_uses_open_price_not_theoretical_stop(self):
        # 開盤就已經跳空跌破停損價95，理論停損價98根本沒有機會成交
        position = {"side": "long", "stop_price": 98.0, "target_price": 105.0}
        row = make_row(open_=95.0, high=96.0, low=94.0, close=95.5)
        event, price = check_exit(row, position)
        assert event == "stop_gap"
        assert price == 95.0

    def test_short_gap_through_stop_uses_open_price(self):
        position = {"side": "short", "stop_price": 105.0, "target_price": 95.0}
        row = make_row(open_=108.0, high=109.0, low=107.0, close=108.0)
        event, price = check_exit(row, position)
        assert event == "stop_gap"
        assert price == 108.0

    def test_long_target_hit(self):
        position = {"side": "long", "stop_price": 98.0, "target_price": 105.0}
        row = make_row(open_=100.0, high=106.0, low=99.5, close=105.5)
        event, price = check_exit(row, position)
        assert event == "target"
        assert price == 105.0

    def test_none_target_price_never_triggers_target_long(self):
        # 移動停利模式：target_price=None，不管漲多高都不該被判定成target
        position = {"side": "long", "stop_price": 98.0, "target_price": None}
        row = make_row(open_=100.0, high=999.0, low=99.5, close=500.0)
        event, price = check_exit(row, position)
        assert event is None

    def test_no_event_when_price_stays_between_stop_and_target(self):
        position = {"side": "long", "stop_price": 98.0, "target_price": 105.0}
        row = make_row(open_=100.0, high=101.0, low=99.5, close=100.5)
        event, price = check_exit(row, position)
        assert event is None
        assert price is None


class TestUpdateTrailingStop:
    def test_long_stop_moves_up_when_price_rises(self):
        position = {
            "side": "long", "e_price": 100.0, "stop_price": 98.0,
            "atr_entry": 2.0, "trailing_atr_mult": 1.0,
        }
        row = make_row(close=110.0)
        updated = update_trailing_stop(position, row)
        assert updated["stop_price"] == pytest.approx(110.0 - 1.0 * 2.0)
        assert updated["trailing_anchor"] == pytest.approx(110.0)

    def test_long_stop_never_moves_down_on_pullback(self):
        position = {
            "side": "long", "e_price": 100.0, "stop_price": 108.0,
            "trailing_anchor": 110.0, "atr_entry": 2.0, "trailing_atr_mult": 1.0,
        }
        row = make_row(close=105.0)  # 從110拉回到105
        updated = update_trailing_stop(position, row)
        assert updated["stop_price"] == 108.0  # 停損不該跟著拉回，維持原本較高的停損
        assert updated["trailing_anchor"] == 110.0  # 錨點也維持最高點，不會被拉回覆蓋

    def test_short_stop_moves_down_when_price_falls(self):
        position = {
            "side": "short", "e_price": 100.0, "stop_price": 102.0,
            "atr_entry": 2.0, "trailing_atr_mult": 1.0,
        }
        row = make_row(close=90.0)
        updated = update_trailing_stop(position, row)
        assert updated["stop_price"] == pytest.approx(90.0 + 1.0 * 2.0)

    def test_short_stop_never_moves_up_on_bounce(self):
        position = {
            "side": "short", "e_price": 100.0, "stop_price": 92.0,
            "trailing_anchor": 90.0, "atr_entry": 2.0, "trailing_atr_mult": 1.0,
        }
        row = make_row(close=95.0)  # 從90反彈到95
        updated = update_trailing_stop(position, row)
        assert updated["stop_price"] == 92.0


class TestMaxConsecutiveLosses:
    def test_counts_longest_losing_streak(self):
        pnl_list = [100, -50, -30, -20, 200, -10, -10, -10, -10, 50]
        assert _max_consecutive_losses(pnl_list) == 4

    def test_no_losses_returns_zero(self):
        assert _max_consecutive_losses([100, 200, 300]) == 0

    def test_all_losses(self):
        assert _max_consecutive_losses([-1, -2, -3]) == 3


class TestSummarizeMrNewStats:
    def _make_trades(self, pnl_list, starting_capital=100.0):
        trades = []
        for i, pnl in enumerate(pnl_list):
            trades.append({
                "code": "1101", "side": "long",
                "entry_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "exit_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=i),
                "e_price": 100.0, "exit_price": 100.0 + pnl, "exit_reason": "target",
                "lots": 1, "pnl_ntd": pnl, "return_pct": pnl / 1000.0, "hold_days": 1,
            })
        return trades

    def test_profit_factor_basic(self):
        trades = self._make_trades([100, 100, -50])
        stats = summarize_mr(trades, 1_000_000)
        assert stats["profit_factor"] == pytest.approx(200 / 50)

    def test_profit_factor_infinite_when_no_losses(self):
        trades = self._make_trades([100, 200])
        stats = summarize_mr(trades, 1_000_000)
        assert stats["profit_factor"] == float("inf")

    def test_profit_factor_zero_when_no_wins(self):
        trades = self._make_trades([-100, -200])
        stats = summarize_mr(trades, 1_000_000)
        assert stats["profit_factor"] == 0.0

    def test_max_consecutive_losses_in_summary(self):
        trades = self._make_trades([100, -10, -10, -10, 50])
        stats = summarize_mr(trades, 1_000_000)
        assert stats["max_consecutive_losses"] == 3

    def test_top_trade_pct_of_total_pnl(self):
        trades = self._make_trades([80, 10, 10])  # 總損益100，最大一筆佔80%
        stats = summarize_mr(trades, 1_000_000)
        assert stats["top_trade_pct_of_total_pnl"] == pytest.approx(80.0)

    def test_pnl_excluding_top3_drops_below_zero_when_dependent_on_few_trades(self):
        trades = self._make_trades([1000, 1000, 1000, -100, -100])
        stats = summarize_mr(trades, 1_000_000)
        assert stats["pnl_excluding_top3_ntd"] == pytest.approx(-200)

    def test_empty_trades_returns_zeroed_new_fields(self):
        stats = summarize_mr([], 1_000_000)
        assert stats["profit_factor"] == 0.0
        assert stats["max_consecutive_losses"] == 0
        assert stats["sharpe_like"] == 0.0
        assert stats["calmar_like"] == 0.0


class TestCheckExitSlippage:
    def test_long_stop_with_slippage_fills_worse_than_stop_price(self):
        position = {"side": "long", "stop_price": 98.0, "target_price": 105.0}
        row = make_row(open_=100.0, high=101.0, low=97.5, close=98.0)
        event, price = check_exit(row, position, slippage_pct=0.01)
        assert event == "stop"
        assert price == pytest.approx(98.0 * 0.99)

    def test_long_stop_gap_with_slippage_fills_worse_than_open(self):
        position = {"side": "long", "stop_price": 98.0, "target_price": 105.0}
        row = make_row(open_=95.0, high=96.0, low=94.0, close=95.5)
        event, price = check_exit(row, position, slippage_pct=0.01)
        assert event == "stop_gap"
        assert price == pytest.approx(95.0 * 0.99)

    def test_short_stop_with_slippage_fills_worse_than_stop_price(self):
        position = {"side": "short", "stop_price": 102.0, "target_price": 95.0}
        row = make_row(open_=100.0, high=103.0, low=99.0, close=102.0)
        event, price = check_exit(row, position, slippage_pct=0.01)
        assert event == "stop"
        assert price == pytest.approx(102.0 * 1.01)

    def test_target_exit_is_never_affected_by_slippage(self):
        position = {"side": "long", "stop_price": 98.0, "target_price": 105.0}
        row = make_row(open_=100.0, high=106.0, low=99.5, close=105.5)
        event, price = check_exit(row, position, slippage_pct=0.01)
        assert event == "target"
        assert price == 105.0  # target是限價出場，不套滑價

    def test_zero_slippage_matches_old_behavior_exactly(self):
        position = {"side": "long", "stop_price": 98.0, "target_price": 105.0}
        row = make_row(open_=100.0, high=101.0, low=97.5, close=98.0)
        event, price = check_exit(row, position)  # slippage_pct 預設 0.0
        assert price == 98.0
