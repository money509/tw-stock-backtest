"""
test_revenue_drift_engine.py
===============================
revenue_drift_engine.py的單元測試。全程用合成的小DataFrame，不碰網路(沒有任何
revenue_data_loader/data_loader/yfinance/requests呼叫)。涵蓋：營收意外(surprise)
計算正確性(過去6個月平均、不偷看未來)、冷啟動(歷史不足6個月)排除、事件日偵測(只有
真的有公告的那天才產生候選，不是每天都有)、橫斷面percentile排名的多空分組正確性、
進場嚴格晚於公告可見日(不是同一天)、固定持有天數出場是否正確觸發、放空的損益正負號
是否正確。
"""
import numpy as np
import pandas as pd
import pytest

from revenue_drift_engine import (
    precompute_revenue_drift_price_indicators, precompute_revenue_surprise,
    precompute_all_revenue_surprise, build_revenue_drift_events, build_entry_date_index,
    scan_revenue_drift_candidates, run_revenue_drift_backtest, TRAILING_MONTHS,
)
from mean_reversion_engine import summarize_mr


def _make_price_df(closes, start="2024-01-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="B")
    closes = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
        "Volume": pd.Series(2_000_000.0, index=idx),  # close*volume遠大於MIN_TURNOVER_VALUE(2000萬)
    }, index=idx)


def _make_revenue_df(yoy_values, start="2024-01-10", freq="MS"):
    """造一檔股票的月營收序列：index是「公告可見日」(revenue_data_loader.py的次月10日
    近似值寫法)，每個月一筆，revenue_mom_pct這裡不重要，填0即可。"""
    n = len(yoy_values)
    idx = pd.date_range(start, periods=n, freq=freq) + pd.Timedelta(days=9)  # 每月10日
    return pd.DataFrame({
        "revenue_yoy_pct": yoy_values, "revenue_mom_pct": [0.0] * n,
    }, index=idx)


class TestPrecomputeRevenueSurprise:
    def test_requires_at_least_six_prior_months(self):
        """前6筆(含)歷史不足trailing_months=6筆「之前」的紀錄，surprise必須是NaN
        (冷啟動安全機制)；第7筆(index 6)開始，前面剛好累積滿6筆，才第一次算得出來。"""
        yoy = [10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 20.0, 5.0]
        revenue_df = _make_revenue_df(yoy)
        surprise_df = precompute_revenue_surprise(revenue_df, trailing_months=6)
        assert surprise_df["surprise"].iloc[:6].isna().all()
        assert not pd.isna(surprise_df["surprise"].iloc[6])
        assert not pd.isna(surprise_df["surprise"].iloc[7])

    def test_surprise_value_matches_deviation_from_trailing_average(self):
        """第7筆(index 6)的surprise應該等於第7筆yoy減去前6筆(index 0~5)的平均，
        不含當月本身。"""
        yoy = [10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 20.0]
        revenue_df = _make_revenue_df(yoy)
        surprise_df = precompute_revenue_surprise(revenue_df, trailing_months=6)
        expected_avg = sum(yoy[:6]) / 6
        expected_surprise = yoy[6] - expected_avg
        assert surprise_df["surprise"].iloc[6] == pytest.approx(expected_surprise)
        assert surprise_df["trailing_avg_yoy_pct"].iloc[6] == pytest.approx(expected_avg)

    def test_surprise_does_not_use_future_data(self):
        """驗證第t列的surprise只用t以前(嚴格早於t)的資料：把t之後的月份資料改掉，
        不該影響t列本身算出來的surprise(沒有偷看未來)——題目特別要求的驗證點。"""
        yoy_a = [10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 20.0, 5.0, 7.0]
        yoy_b = [10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 20.0, 999.0, -999.0]  # 只改最後兩個月(未來)
        surprise_a = precompute_revenue_surprise(_make_revenue_df(yoy_a), trailing_months=6)
        surprise_b = precompute_revenue_surprise(_make_revenue_df(yoy_b), trailing_months=6)
        # 第8、9筆(index 6、7)之前的surprise都不該因為未來月份被改而變動
        assert surprise_a["surprise"].iloc[:7].equals(surprise_b["surprise"].iloc[:7])

    def test_default_trailing_months_constant_is_six(self):
        assert TRAILING_MONTHS == 6


class TestBuildRevenueDriftEvents:
    def test_only_companies_with_announcement_on_exact_date_appear(self):
        """同一天的事件表裡，只有「那天真的有公告」的公司才會出現，不是每檔股票每天
        都出現(事件驅動，不是每天都有候選的連續排名)。"""
        yoy_a = [10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 20.0]  # 7個月，第7個月才有surprise
        yoy_b = [1.0, 2.0, 3.0]  # 只有3個月，不足6個月歷史，永遠不會進events表
        surprise_by_code = precompute_all_revenue_surprise(
            {"1101": _make_revenue_df(yoy_a), "1102": _make_revenue_df(yoy_b)}
        )
        events_by_date = build_revenue_drift_events(surprise_by_code)
        # 1102歷史不足，不該在任何一天的事件表裡出現
        all_codes_in_events = {ev["code"] for evs in events_by_date.values() for ev in evs}
        assert "1102" not in all_codes_in_events
        assert "1101" in all_codes_in_events
        # 1101只有1個月(第7個月)有效，事件表應該只有1天、1筆事件
        assert len(events_by_date) == 1
        only_date = next(iter(events_by_date))
        assert len(events_by_date[only_date]) == 1

    def test_companies_sharing_announcement_date_are_grouped_together(self):
        """兩檔股票同一個(年,月)算出來的公告可見日是同一個Timestamp，事件表裡應該
        被歸在同一天，驗證「台灣月營收公告叢集在同一天」這件事能正確反映在事件表結構。"""
        yoy = [10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 20.0]
        surprise_by_code = precompute_all_revenue_surprise(
            {"1101": _make_revenue_df(yoy), "1102": _make_revenue_df([v + 1 for v in yoy])}
        )
        events_by_date = build_revenue_drift_events(surprise_by_code)
        assert len(events_by_date) == 1
        only_date = next(iter(events_by_date))
        codes = {ev["code"] for ev in events_by_date[only_date]}
        assert codes == {"1101", "1102"}


class TestBuildEntryDateIndex:
    def test_entry_date_is_strictly_after_announcement_date(self):
        """進場日一定要嚴格晚於公告可見日，絕不能同一天——題目特別要求的驗證點。"""
        master_calendar = pd.date_range("2024-01-01", periods=60, freq="B")
        ann_date = master_calendar[20]  # 剛好是交易日的公告可見日
        events_by_date = {ann_date: [{"code": "1101", "surprise": 5.0}]}
        events_by_entry_date = build_entry_date_index(events_by_date, master_calendar)
        assert len(events_by_entry_date) == 1
        entry_date = next(iter(events_by_entry_date))
        assert entry_date > ann_date
        assert entry_date == master_calendar[21]  # 下一個交易日，不是同一天

    def test_announcement_date_not_a_trading_day_still_gets_next_trading_day(self):
        """公告可見日剛好是週末(非交易日)，entry_date應該是它之後的第一個交易日，
        不應該因為當天不是交易日就整個找不到進場日。"""
        master_calendar = pd.date_range("2024-01-01", periods=30, freq="B")
        # 選一個確定不在master_calendar裡的週末日期
        ann_date = pd.Timestamp("2024-01-06")  # 週六
        assert ann_date not in master_calendar
        events_by_date = {ann_date: [{"code": "1101", "surprise": 5.0}]}
        events_by_entry_date = build_entry_date_index(events_by_date, master_calendar)
        entry_date = next(iter(events_by_entry_date))
        assert entry_date > ann_date

    def test_announcement_after_calendar_end_is_dropped(self):
        """公告可見日晚於master_calendar涵蓋範圍，找不到對應的進場日，這個事件應該
        被捨棄，不該丟例外或產生不存在的日期。"""
        master_calendar = pd.date_range("2024-01-01", periods=10, freq="B")
        ann_date = master_calendar[-1] + pd.Timedelta(days=30)
        events_by_date = {ann_date: [{"code": "1101", "surprise": 5.0}]}
        events_by_entry_date = build_entry_date_index(events_by_date, master_calendar)
        assert events_by_entry_date == {}


class TestScanRevenueDriftCandidates:
    def _build_scenario(self, surprises_by_code, warmup=25):
        """每檔股票價格走勢在暖身期(warmup天)都是平的，之後固定不變，單純用來讓
        流動性/ATR門檻通過；真正要測的是surprise排名，不是價格走勢本身。"""
        price_data, price_indicators = {}, {}
        for code in surprises_by_code:
            closes = [100.0] * (warmup + 5)
            price_data[code] = _make_price_df(closes)
        from revenue_drift_engine import precompute_revenue_drift_price_indicators
        universe = {code: {} for code in surprises_by_code}
        price_indicators = precompute_revenue_drift_price_indicators(price_data, universe)

        ann_date = price_data[next(iter(surprises_by_code))].index[20]
        entry_date = price_data[next(iter(surprises_by_code))].index[26]
        events_by_date = {
            ann_date: [{"code": code, "surprise": s} for code, s in surprises_by_code.items()]
        }
        return events_by_date, ann_date, price_indicators, entry_date

    def test_long_candidates_are_highest_surprise(self):
        surprises = {"1101": -10.0, "1102": -2.0, "1210": 5.0, "1216": 10.0}
        events_by_date, ann_date, price_indicators, entry_date = self._build_scenario(surprises)
        candidates = scan_revenue_drift_candidates(
            events_by_date, ann_date, price_indicators, entry_date, excluded_codes=set(),
            top_n=2, allow_short=False,
        )
        long_codes = {c["code"] for c in candidates if c["side"] == "long"}
        assert long_codes == {"1210", "1216"}  # 意外最高(超預期)的兩檔

    def test_short_candidates_are_lowest_surprise(self):
        surprises = {"1101": -10.0, "1102": -2.0, "1210": 5.0, "1216": 10.0}
        events_by_date, ann_date, price_indicators, entry_date = self._build_scenario(surprises)
        candidates = scan_revenue_drift_candidates(
            events_by_date, ann_date, price_indicators, entry_date, excluded_codes=set(),
            top_n=2, allow_short=True,
        )
        short_codes = {c["code"] for c in candidates if c["side"] == "short"}
        assert short_codes == {"1101", "1102"}  # 意外最低(miss最嚴重)的兩檔

    def test_allow_short_false_produces_no_short_candidates(self):
        surprises = {"1101": -10.0, "1102": 10.0}
        events_by_date, ann_date, price_indicators, entry_date = self._build_scenario(surprises)
        candidates = scan_revenue_drift_candidates(
            events_by_date, ann_date, price_indicators, entry_date, excluded_codes=set(),
            top_n=2, allow_short=False,
        )
        assert all(c["side"] == "long" for c in candidates)

    def test_excluded_codes_are_skipped(self):
        surprises = {"1101": -10.0, "1102": -2.0}
        events_by_date, ann_date, price_indicators, entry_date = self._build_scenario(surprises)
        candidates = scan_revenue_drift_candidates(
            events_by_date, ann_date, price_indicators, entry_date, excluded_codes={"1101"},
            top_n=2, allow_short=False,
        )
        assert all(c["code"] != "1101" for c in candidates)

    def test_no_events_on_date_returns_empty_list(self):
        """事件日偵測：如果查詢的announcement_date根本不在events_by_date裡
        (沒有任何公司在那天公告)，應該回傳空名單，不是意外報錯或回傳其他天的候選。"""
        surprises = {"1101": -10.0}
        events_by_date, ann_date, price_indicators, entry_date = self._build_scenario(surprises)
        other_date = ann_date + pd.Timedelta(days=100)
        candidates = scan_revenue_drift_candidates(
            events_by_date, other_date, price_indicators, entry_date, excluded_codes=set(),
        )
        assert candidates == []

    def test_wide_top_n_does_not_assign_same_code_to_both_sides(self):
        """top_n=max(top_n, slots_available)在呼叫端可能把top_n撐大到超過當天候選
        總數(事件日公司數很少時很常見，不是邊緣情況)——撐大後，若未去重，多/空兩份
        head(top_n)名單會整批重疊，同一支股票會同時出現在兩份名單裡，最後真正的
        成交方向會變成由候選順序(多方先)決定，不是訊號強弱，這是個真正的正確性問題
        (見revenue_drift_engine.scan_revenue_drift_candidates()的去重複指派說明)。
        這裡故意把top_n(5)設得遠大於候選總數(3)，重現「正常情況下也會撐大」的
        組態，驗證修正後每支股票只會落在其中一份名單。"""
        surprises = {"1101": -10.0, "1102": 0.0, "1210": 10.0}
        events_by_date, ann_date, price_indicators, entry_date = self._build_scenario(surprises)
        candidates = scan_revenue_drift_candidates(
            events_by_date, ann_date, price_indicators, entry_date, excluded_codes=set(),
            top_n=5, allow_short=True,
        )
        codes_by_side = {}
        for c in candidates:
            codes_by_side.setdefault(c["code"], set()).add(c["side"])
        assert all(len(sides) == 1 for sides in codes_by_side.values()), (
            f"某支股票同時出現在多方跟空方名單：{codes_by_side}"
        )
        # 意外最高的1210應該只在多方、意外最低的1101應該只在空方
        assert codes_by_side["1210"] == {"long"}
        assert codes_by_side["1101"] == {"short"}

    def test_liquidity_gate_filters_low_turnover_stock(self):
        surprises = {"1101": -10.0, "1102": -9.0}
        price_data = {}
        closes = [100.0] * 30
        price_data["1101"] = _make_price_df(closes)  # 預設Volume=2,000,000，成交金額夠高
        low_liq_df = _make_price_df(closes)
        low_liq_df["Volume"] = 100.0  # 成交金額遠低於MIN_TURNOVER_VALUE
        price_data["1102"] = low_liq_df
        universe = {"1101": {}, "1102": {}}
        price_indicators = precompute_revenue_drift_price_indicators(price_data, universe)
        ann_date = price_data["1101"].index[20]
        entry_date = price_data["1101"].index[26]
        events_by_date = {
            ann_date: [{"code": "1101", "surprise": -10.0}, {"code": "1102", "surprise": -9.0}]
        }
        candidates = scan_revenue_drift_candidates(
            events_by_date, ann_date, price_indicators, entry_date, excluded_codes=set(),
            top_n=5, allow_short=False,
        )
        codes = {c["code"] for c in candidates}
        assert "1102" not in codes
        assert "1101" in codes


class TestRunRevenueDriftBacktestExitTiming:
    def _flat_then_drop_universe(self, ann_offset=25, tail_days=20):
        """造兩檔股票：一檔之後急跌(多方候選，意外最高)，一檔之後急漲(空方候選，
        意外最低)——跟short_reversal_engine的測試精神一致，確保候選一定被選到，
        之後專心驗證出場時機/損益正負號。公告可見日設在暖身期結束那天，entry_date
        自動是它的下一個交易日。"""
        warmup = ann_offset + 5
        long_closes = [100.0] * warmup + list(np.linspace(100.0, 80.0, 6))[1:] + [80.0] * tail_days
        short_closes = [100.0] * warmup + list(np.linspace(100.0, 125.0, 6))[1:] + [125.0] * tail_days
        price_data = {
            "1101": _make_price_df(long_closes),
            "1102": _make_price_df(short_closes),
        }
        universe = {"1101": {}, "1102": {}}
        price_indicators = precompute_revenue_drift_price_indicators(price_data, universe)
        master_calendar = price_data["1101"].index
        ann_date = master_calendar[ann_offset]
        events_by_date = {
            ann_date: [{"code": "1101", "surprise": 10.0}, {"code": "1102", "surprise": -10.0}]
        }
        events_by_entry_date = build_entry_date_index(events_by_date, master_calendar)
        return price_data, price_indicators, events_by_date, events_by_entry_date, master_calendar

    def test_entry_happens_on_day_after_announcement_not_same_day(self):
        price_data, price_indicators, events_by_date, events_by_entry_date, master_calendar = \
            self._flat_then_drop_universe()
        ann_date = next(iter(events_by_date))
        trades = run_revenue_drift_backtest(
            price_data, price_indicators, events_by_date, events_by_entry_date, master_calendar,
            max_hold_days=5, starting_capital=1_000_000, allow_short=True, lots=2,
            atr_stop_mult=100.0, top_n=2, max_concurrent_positions=4,
        )
        assert len(trades) > 0
        for t in trades:
            assert t["entry_date"] > ann_date

    def test_fixed_hold_days_forces_exit_on_the_right_day(self):
        price_data, price_indicators, events_by_date, events_by_entry_date, master_calendar = \
            self._flat_then_drop_universe()
        max_hold_days = 3
        trades = run_revenue_drift_backtest(
            price_data, price_indicators, events_by_date, events_by_entry_date, master_calendar,
            max_hold_days=max_hold_days, starting_capital=1_000_000, allow_short=True, lots=2,
            atr_stop_mult=100.0,  # ATR停損設得極寬，確保不會提早被停損出場，只測固定天數出場
            top_n=2, max_concurrent_positions=4,
        )
        assert len(trades) > 0
        for t in trades:
            assert t["hold_days"] <= max_hold_days
        assert any(t["exit_reason"] == "forced_close" for t in trades)

    def _build_two_code_scenario(self, target_closes, target_surprise, filler_surprise, warmup=26):
        """造兩檔股票的事件：1101(真正要測的那檔)用target_closes/target_surprise，
        1102是「陪榜」的對照股(filler_surprise跟target相反號、價格全程持平，不會
        觸發任何停損/跳空風控)。

        這個版本取代了舊版「只放1101一檔」的寫法：scan_revenue_drift_candidates()
        2026-10修正後，同一支股票不會再同時被歸類成多方又空方候選(見revenue_drift_
        engine.scan_revenue_drift_candidates()的去重複指派說明)，單一股票的
        surprise在只有它自己一筆的集合裡會是退化情況(跟自己比，歸類沒有意義)。
        改成放進一支surprise方向明確相反的陪榜股，1101的多空分類才有真正的排名
        依據可比，不用再靠「用跳空風控擋掉不想測的那一側」這種間接手法控制成交
        方向——直接讓1101依訊號強弱被分類到想測的那一側。top_n=1確保只取各一名，
        1102的平盤價格不會自己觸發任何停損/強制出場，單純當分母用。"""
        warmup_filler = max(warmup, 26)
        filler_closes = [100.0] * (warmup_filler + 10)
        price_data = {
            "1101": _make_price_df(target_closes),
            "1102": _make_price_df(filler_closes[:len(target_closes)]),
        }
        universe = {"1101": {}, "1102": {}}
        price_indicators = precompute_revenue_drift_price_indicators(price_data, universe)
        master_calendar = price_data["1101"].index
        ann_date = master_calendar[25]
        events_by_date = {
            ann_date: [
                {"code": "1101", "surprise": target_surprise},
                {"code": "1102", "surprise": filler_surprise},
            ]
        }
        events_by_entry_date = build_entry_date_index(events_by_date, master_calendar)
        return price_data, price_indicators, events_by_date, events_by_entry_date, master_calendar

    def test_short_side_profits_when_price_falls(self):
        """放空的股票(意外最低/miss最嚴重，被選為空方候選)之後持續下跌，空單應該賺錢，
        驗證放空的損益正負號沒有寫反——題目特別提醒容易出錯的地方，跟test_short_
        reversal_engine.py的對應測試同樣精神。

        1101意外=-10.0(miss)、陪榜股1102意外=+10.0(beat)，兩者相對排名明確，1101
        會被歸類成空方候選(見scan_revenue_drift_candidates()的去重複指派邏輯：
        score_long < score_short時歸空方)，不需要再靠跳空風控間接控制成交方向。
        entry_date(index 26)之後股價持續下跌，驗證空單損益正負號正確。"""
        warmup = 25
        closes = [100.0] * warmup + list(np.linspace(100.0, 70.0, 10))
        price_data, price_indicators, events_by_date, events_by_entry_date, master_calendar = \
            self._build_two_code_scenario(closes, target_surprise=-10.0, filler_surprise=10.0, warmup=warmup)

        trades = run_revenue_drift_backtest(
            price_data, price_indicators, events_by_date, events_by_entry_date, master_calendar,
            max_hold_days=5, starting_capital=1_000_000, allow_short=True, lots=2,
            atr_stop_mult=100.0, top_n=1, max_concurrent_positions=2,
        )
        short_trades = [t for t in trades if t["code"] == "1101" and t["side"] == "short"]
        assert len(short_trades) > 0
        for t in short_trades:
            assert t["exit_price"] < t["e_price"]
            assert t["pnl_ntd"] > 0

    def test_long_side_loses_when_price_keeps_falling(self):
        """多方候選(意外最高，被選中)如果之後繼續跌，多單應該賠錢，同樣驗證做多方向
        的損益正負號正確(跟放空互為對照)。

        1101意外=+10.0(beat)、陪榜股1102意外=-10.0(miss)，1101會被歸類成多方候選。
        進場之後股價反轉下跌(105一路跌到70)，讓這筆多單最終賠錢，驗證多方的損益
        正負號。"""
        warmup = 26
        closes = [100.0] * warmup + [105.0] + list(np.linspace(105.0, 70.0, 6))[1:]
        price_data, price_indicators, events_by_date, events_by_entry_date, master_calendar = \
            self._build_two_code_scenario(closes, target_surprise=10.0, filler_surprise=-10.0, warmup=warmup)

        trades = run_revenue_drift_backtest(
            price_data, price_indicators, events_by_date, events_by_entry_date, master_calendar,
            max_hold_days=5, starting_capital=1_000_000, allow_short=True, lots=2,
            atr_stop_mult=100.0, top_n=1, max_concurrent_positions=2,
        )
        long_trades = [t for t in trades if t["code"] == "1101" and t["side"] == "long"]
        assert len(long_trades) > 0
        for t in long_trades:
            assert t["exit_price"] < t["e_price"]
            assert t["pnl_ntd"] < 0

    def test_no_trades_on_days_without_announcement_events(self):
        """事件驅動：沒有公告事件的交易日不該產生任何新部位，用一個完全沒有events的
        calendar確認回測結果是空的(不是每天都掃全市場找候選)。"""
        price_data, price_indicators, _events_by_date, _events_by_entry_date, master_calendar = \
            self._flat_then_drop_universe()
        trades = run_revenue_drift_backtest(
            price_data, price_indicators, {}, {}, master_calendar,
            max_hold_days=5, starting_capital=1_000_000, allow_short=True, lots=2,
            atr_stop_mult=100.0, top_n=2, max_concurrent_positions=4,
        )
        assert trades == []


class TestRunRevenueDriftBacktestSmoke:
    def test_runs_without_error_and_produces_summarizable_trades(self):
        np.random.seed(11)
        n = 260
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        price_data, universe = {}, {}
        for i, code in enumerate(["1101", "1102", "1210", "1216", "1301"]):
            base = 80 + i * 10
            noise = np.random.normal(0, 1.5, n)
            closes = np.maximum(base + np.cumsum(noise * 0.3), 1.0)
            df = pd.DataFrame({
                "Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
                "Volume": pd.Series(2_000_000.0, index=idx),
            }, index=idx)
            price_data[code] = df
            universe[code] = {}

        # 每檔股票每個月都造一筆(隨機)營收意外資料，橫跨master_calendar
        revenue_data = {}
        np.random.seed(5)
        for code in universe:
            yoy = list(np.random.normal(10, 5, 15))
            revenue_data[code] = _make_revenue_df(yoy, start="2022-08-10")

        price_indicators = precompute_revenue_drift_price_indicators(price_data, universe)
        surprise_by_code = precompute_all_revenue_surprise(revenue_data)
        events_by_date = build_revenue_drift_events(surprise_by_code)
        events_by_entry_date = build_entry_date_index(events_by_date, idx)

        trades = run_revenue_drift_backtest(
            price_data, price_indicators, events_by_date, events_by_entry_date, idx,
            max_hold_days=5, starting_capital=1_000_000, allow_short=True, lots=2,
            top_n=2, max_concurrent_positions=4,
        )
        stats = summarize_mr(trades, 1_000_000)
        assert "trade_count" in stats
        assert "profit_factor" in stats
