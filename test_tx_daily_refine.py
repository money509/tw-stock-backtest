"""
test_tx_daily_refine.py —— 精煉模式(tx_daily_refine + compare_tx_daily --refine)離線測試：
變體數312、標準MACD參數 = 網格MACD訊號、只做多不會有空單、逐日市值回撤(手算小例子 + >=平倉回撤)、
walk-forward不偷看未來/平手規則/逐年串接、成本加壓PF <= 原PF、診斷彙總、判定、執行時間、
scipy被擋+不連網的端對端，以及8變體/網格模式的輸出除了網格新增的兩欄之外完全沒變。
"""
import hashlib
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pandas as pd
import pytest

import compare_tx_daily as ctd
import tx_daily_data as tdd
import tx_daily_engine as eng
import tx_daily_grid as grid
import tx_daily_refine as rf
from momentum_breakout_engine import compute_macd
from test_tx_daily_data import make_index, make_futures_daily_truth, daily_to_minutes, make_yf_frame
from test_tx_daily_engine import mk
from test_tx_daily_grid import manual_cross

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
class TestVariants:
    def test_count_and_order(self):
        vs = rf.refine_variant_list()
        assert len(vs) == 312 == 4 * 3 * 2 * 12 + 1 * 2 * 12
        assert len({v["name"] for v in vs}) == 312
        assert sum(1 for v in vs if v["macd"] is not None) == 288
        assert sum(1 for v in vs if v["macd"] is None) == 24
        assert all(v["rule"] == "MA_5_20&DONCHIAN20" for v in vs if v["macd"] is None)
        assert [r for r, _ in rf.REFINE_RULES] == ["MACD", "MA_5_20&MACD", "MACD&DONCHIAN20", "MACD&RSI50",
                                                  "MA_5_20&DONCHIAN20"]
        assert vs[0]["name"] == rf.fixed_baseline_name() == "MACD[標準12/26/9]｜多空｜停損1.0/停利3.0"
        assert vs[11]["name"] == "MACD[標準12/26/9]｜多空｜停損2.0/移動3.0"
        assert vs[12]["name"] == "MACD[標準12/26/9]｜只做多｜停損1.0/停利3.0"
        assert vs[24]["name"].startswith("MACD[快8/17/9]｜多空")
        assert vs[-1]["name"] == "MA_5_20&DONCHIAN20｜只做多｜停損2.0/移動3.0"
        assert not any(v.get("gate") for v in vs)  # 無動能門檻

    def test_exit_configs(self):
        cfg = rf.refine_exit_configs()
        assert len(cfg) == 12
        assert [(c["stop_atr"], c["exit_kind"], c["exit_mult"]) for c in cfg[:4]] == [
            (1.0, "target", 3.0), (1.0, "target", 4.0), (1.0, "target", 5.0), (1.0, "trail", 3.0)]
        assert {c["stop_atr"] for c in cfg} == {1.0, 1.5, 2.0}
        # 網格預設沒變
        assert [(c["exit_kind"], c["exit_mult"]) for c in eng.grid_exit_configs()[:4]] == [
            ("target", 2.0), ("target", 3.0), ("target", 4.0), ("trail", 3.0)]

    def test_pair_names_match_grid(self):
        grid_rules = {r["rule"]: r["components"] for r in eng.grid_rule_list()}
        for rule, comps in rf.REFINE_RULES:
            assert grid_rules[rule] == comps


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ind_pair():
    df = make_index("2001-01-01", "2012-12-31", seed=9)
    return df, eng.compute_grid_indicators(df), rf.compute_refine_indicators(df)


class TestMacdParams:
    def test_standard_equals_grid(self, ind_pair):
        df, g, r = ind_pair
        assert list(r["sig_MACD@標準"]) == list(g["sig_MACD"])
        for rule in ("MACD", "MA_5_20&MACD", "MACD&DONCHIAN20", "MACD&RSI50"):
            assert list(r[f"sig_{rule}@標準"]) == list(g[f"sig_{rule}"]), rule
        # 預設參數的macd_cross_signal = 手算(compute_macd + 穿越 + 前35根暖身)
        m, s, _ = compute_macd(df["Close"], 12, 26, 9)
        exp = manual_cross(m, s)
        exp[:35] = [0] * 35
        assert eng.macd_cross_signal(df["Close"])[0].tolist() == exp

    def test_fast_slow_params(self, ind_pair):
        df, _, r = ind_pair
        for pk, (f, sl, sg) in rf.MACD_PARAMS:
            m, s, _ = compute_macd(df["Close"], f, sl, sg)
            exp = manual_cross(m, s)
            exp[:sl + sg] = [0] * (sl + sg)
            assert list(r[f"sig_MACD@{pk}"]) == exp, pk
            assert (r[f"sig_MACD@{pk}"] == 1).any() and (r[f"sig_MACD@{pk}"] == -1).any()
        assert list(r["sig_MACD@快"]) != list(r["sig_MACD@標準"])
        assert list(r["sig_MACD@慢"]) != list(r["sig_MACD@標準"])
        # 混搭用網格的compose_pair
        assert list(r["sig_MACD&RSI50@慢"]) == eng.compose_pair(r["sig_MACD@慢"], r["sig_RSI50"]).tolist()
        assert list(r["sig_MA_5_20&MACD@快"]) == eng.compose_pair(r["sig_MA_5_20"], r["sig_MACD@快"]).tolist()

    def test_every_variant_column_exists(self, ind_pair):
        _, _, r = ind_pair
        for v in rf.refine_variant_list():
            assert v["sig_col"] in r.columns

    def test_no_lookahead(self):
        df = make_index("2008-01-01", "2012-12-31", seed=13)
        full = rf.compute_refine_indicators(df)
        cols = sorted({v["sig_col"] for v in rf.refine_variant_list()})
        for cut in (300, 777):
            part = rf.compute_refine_indicators(df.iloc[:cut])
            for c in cols:
                assert list(part[c]) == list(full[c].iloc[:cut]), (cut, c)


# ---------------------------------------------------------------------------
def _arrs(df, sig, atr=10.0, col="sig_X"):
    ind = pd.DataFrame(index=df.index)
    ind["ATR"] = atr
    full = np.zeros(len(df), int)
    for i, v in sig.items():
        full[i] = v
    ind[col] = full
    ind["gate_long"] = True
    ind["gate_short"] = True
    return eng.prepare_grid_arrays(df, ind)


class TestDirection:
    def test_long_only_ignores_short_entries_but_uses_them_for_exit(self):
        df = mk([100.0] * 20)
        arrs = _arrs(df, {2: 1, 6: -1, 12: -1})
        v_long = {"sig_col": "sig_X", "direction": "long"}
        v_both = {"sig_col": "sig_X", "direction": "both"}
        es, rs = rf.refine_entry_signals(arrs, v_long)
        assert es.tolist().count(-1) == 0 and rs.tolist()[6] == -1
        tr = eng.run_tx_daily_grid_backtest(arrs, es, rs, 1.0, "target", 4.0, commission_per_side=0, tax_rate=0)
        assert [t["side"] for t in tr] == ["long"]
        assert tr[0]["exit_reason"] == "反向訊號" and tr[0]["exit_date"] == df.index[7]
        es, rs = rf.refine_entry_signals(arrs, v_both)
        tr = eng.run_tx_daily_grid_backtest(arrs, es, rs, 1.0, "target", 4.0, commission_per_side=0, tax_rate=0)
        assert [t["side"] for t in tr] == ["long", "short"]

    def test_long_only_no_short_trades_on_synthetic(self, refine_res):
        res, vs = refine_res
        n_long_only = 0
        for v in vs:
            for p in rf.PERIOD_KEYS:
                tr = res["trades"][(v["name"], p)]
                if v["direction"] == "long":
                    assert all(t["side"] == "long" for t in tr), v["name"]
                    assert res["stats"][(v["name"], p)]["short_count"] == 0
                    n_long_only += len(tr)
        assert n_long_only > 0
        # 多空版本確實有空單(不然這個測試沒有意義)
        assert any(res["stats"][(v["name"], "test")]["short_count"] > 0 for v in vs if v["direction"] == "both")


# ---------------------------------------------------------------------------
def _trade(entry_date, exit_date, pnl, side="long", entry_price=100.0, period="train"):
    return {"entry_date": pd.Timestamp(entry_date), "exit_date": pd.Timestamp(exit_date), "pnl_ntd": float(pnl),
            "side": side, "entry_price": entry_price, "period": period, "hold_days": 1}


class TestMtm:
    def test_hand_computed_long(self):
        df = mk([100.0, 100.0, 110.0, 90.0, 105.0, 100.0])
        arrs = _arrs(df, {})
        d = df.index
        t = _trade(d[1], d[4], 200.0, entry_price=100.0)
        eq = rf.mtm_equity_curve([t], arrs)
        # 第1天(進場日)收100 → 0；第2天110 → +500；第3天90 → −500；第4天出場 → 已實現200；第5天200
        assert eq.tolist() == [0.0, 0.0, 500.0, -500.0, 200.0, 200.0]
        assert rf.max_drawdown(eq) == -1000.0
        closed = grid.quick_stats([dict(t, hold_days=3)])["max_drawdown_ntd"]
        assert closed == 0.0 and abs(rf.max_drawdown(eq)) >= abs(closed)

    def test_hand_computed_short_and_two_trades(self):
        df = mk([100.0, 100.0, 96.0, 104.0, 100.0, 100.0, 108.0, 100.0])
        arrs = _arrs(df, {})
        d = df.index
        t1 = _trade(d[1], d[3], -300.0, side="short", entry_price=100.0)
        t2 = _trade(d[5], d[7], -150.0, side="long", entry_price=100.0)
        eq = rf.mtm_equity_curve([t1, t2], arrs)
        # 空單：第1天0、第2天(100−96)×50=+200；第3天出場已實現−300；第4天−300；
        # 多單：第5天進場收100 → −300；第6天108 → −300+400=100；第7天出場 → −450
        assert eq.tolist() == [0.0, 0.0, 200.0, -300.0, -300.0, -300.0, 100.0, -450.0]
        assert rf.max_drawdown(eq) == -650.0  # 高點200 → −450
        closed = grid.quick_stats([dict(t1), dict(t2)])["max_drawdown_ntd"]
        assert closed == -450.0 and abs(rf.max_drawdown(eq)) >= abs(closed)

    def test_same_day_entry_exit_and_bounds(self):
        df = mk([100.0, 90.0, 100.0])
        arrs = _arrs(df, {})
        d = df.index
        eq = rf.mtm_equity_curve([_trade(d[1], d[1], -500.0)], arrs)
        assert eq.tolist() == [0.0, -500.0, -500.0]
        with pytest.raises(ValueError):
            rf.mtm_equity_curve([_trade(d[0], d[2], 1.0)], arrs, start=d[1])
        assert rf.max_drawdown([]) == 0.0

    def test_segments_carry(self):
        df = mk([100.0, 110.0, 120.0])
        arrs = _arrs(df, {})
        d = df.index
        eq = rf.mtm_equity_segments([([_trade(d[0], d[1], 400.0)], arrs, d[0], d[1]),
                                     ([_trade(d[2], d[2], -100.0)], arrs, d[2], d[2])])
        assert eq.tolist() == [0.0, 400.0, 300.0]

    def test_mtm_ge_closed_on_real_backtests(self, refine_res):
        res, vs = refine_res
        for v in vs:
            for p in rf.PERIOD_KEYS:
                s = res["stats"][(v["name"], p)]
                assert s["mtm_mdd_ntd"] <= s["max_drawdown_ntd"] + 1e-6, (v["name"], p)
                # 曲線終點 = 總損益
                fr = res["frames"][p]
                start, end = grid._period_bounds(p)
                eq = rf.mtm_equity_curve(res["trades"][(v["name"], p)], fr["arrs"], start, end)
                assert eq.iloc[-1] == pytest.approx(s["total_pnl_ntd"])

    def test_buy_hold_mtm(self, refine_res):
        res, _ = refine_res
        for p in rf.PERIOD_KEYS:
            b = res["bh"][p]
            assert len(b["trades"]) == 1
            assert b["mtm_mdd_ntd"] <= min(0.0, b["total_pnl_ntd"]) + 1e-6
            assert b["ratio"] == pytest.approx(rf.pnl_to_dd_ratio(b["total_pnl_ntd"], b["mtm_mdd_ntd"]))
        assert rf.pnl_to_dd_ratio(100.0, -50.0) == 2.0
        assert rf.pnl_to_dd_ratio(100.0, 0.0) == float("inf") and rf.pnl_to_dd_ratio(-1.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def refine_res():
    """合成指數資料(備援模式：三段都是指數)跑全部312變體一次，給多個測試共用。"""
    idx = make_index("1998-01-05", "2026-09-30", seed=17)
    ds = {"source": "index", "main": tdd._index_frame(idx), "index": tdd._index_frame(idx)}
    vs = rf.refine_variant_list()
    t0 = time.perf_counter()
    res = rf.run_refine(ds, vs, n_boot=200)
    res["wall_sec"] = time.perf_counter() - t0
    return res, vs


class TestMetrics:
    def test_cost_stress_pf_not_better(self, refine_res):
        """滑價加大 → 進場價變差，而停損/停利價是從成交價算的，會跟著移動 → 少數交易的路徑會變
        (例如原本碰到停損的現在剛好沒碰到)，所以PF不是數學上保證單調。要求：>=99%的(變體, 期間)
        成本加壓PF <= 原PF，例外的差距也很小；中位數一定變差。"""
        res, vs = refine_res
        worse, viol, n = 0, [], 0
        for v in vs:
            for p in rf.PERIOD_KEYS:
                s = res["stats"][(v["name"], p)]
                n += 1
                if s["stress_pf"] > s["profit_factor"] + 1e-9:
                    viol.append(s["stress_pf"] - s["profit_factor"])
                worse += s["stress_pf"] < s["profit_factor"]
        assert len(viol) <= 0.01 * n, len(viol)
        assert all(d < 0.02 for d in viol), viol
        assert worse >= 0.95 * n
        base = np.median([res["stats"][(v["name"], "test")]["profit_factor"] for v in vs])
        stress = np.median([res["stats"][(v["name"], "test")]["stress_pf"] for v in vs])
        assert stress < base

    def test_cost_stress_hand_case(self):
        closes = [100.0] * 12
        highs = [101.0] * 12
        highs[6] = 135.0
        df = mk(closes, highs=highs)
        arrs = _arrs(df, {2: 1, 8: -1})
        es, rs = rf.refine_entry_signals(arrs, {"sig_col": "sig_X", "direction": "both"})
        a = eng.run_tx_daily_grid_backtest(arrs, es, rs, 1.0, "target", 3.0, commission_per_side=0, tax_rate=0)
        b = eng.run_tx_daily_grid_backtest(arrs, es, rs, 1.0, "target", 3.0, commission_per_side=0, tax_rate=0,
                                           slippage_points=rf.STRESS_SLIPPAGE_POINTS)
        # 多：進場101/停利131 vs 進場103/停利133(限價不扣滑價) → 點數都是30；空：期末平倉多扣2+2點
        assert a[0]["points"] == b[0]["points"] == 30.0
        assert b[1]["points"] == a[1]["points"] - 4.0
        assert rf.profit_factor([t["pnl_ntd"] for t in b]) <= rf.profit_factor([t["pnl_ntd"] for t in a])

    def test_hold_and_time_in_market(self, refine_res):
        res, vs = refine_res
        for v in vs[:24]:
            for p in rf.PERIOD_KEYS:
                s = res["stats"][(v["name"], p)]
                tr = res["trades"][(v["name"], p)]
                fr = res["frames"][p]
                n_days = grid.period_n_days(fr["arrs"]["dates"], *grid._period_bounds(p))
                assert s["avg_hold_days"] == pytest.approx(np.mean([t["hold_days"] for t in tr]))
                assert s["time_in_market_pct"] == pytest.approx(sum(t["hold_days"] for t in tr) / n_days * 100)
                assert 0 < s["time_in_market_pct"] <= 100

    def test_bootstrap_uses_existing_function_and_seed(self, refine_res):
        from robustness_analysis import bootstrap_resample_pnl, summarize_bootstrap
        res, vs = refine_res
        tr = res["trades"][(vs[0]["name"], "test")]
        exp = summarize_bootstrap(bootstrap_resample_pnl(tr, n_resamples=200, seed=42))["pct_positive"]
        assert res["stats"][(vs[0]["name"], "test")]["bootstrap_pct_positive"] == exp

    def test_positive_year_share(self):
        tr = [_trade("2010-03-01", "2010-03-05", 100), _trade("2010-12-28", "2011-01-10", -300),
              _trade("2011-05-01", "2011-05-03", 50), _trade("2012-05-01", "2012-05-03", -1)]
        # 依進場日：2010 = 100−300 = −200；2011 = 50；2012 = −1；2013 沒交易
        assert rf.positive_year_share(tr, [2010, 2011, 2012, 2013]) == (1, 4, 25.0)
        assert rf.yearly_pnl_by_entry(tr) == {2010: -200.0, 2011: 50.0, 2012: -1.0}

    def test_period_years_and_pos_year_pct_in_stats(self, refine_res):
        res, vs = refine_res
        s = res["stats"][(vs[0]["name"], "train")]
        assert s["n_years"] == 12 and 0 <= s["pos_years"] <= 12
        assert res["stats"][(vs[0]["name"], "test")]["n_years"] == 11


# ---------------------------------------------------------------------------
def _fake_wf_variants(n=3):
    return rf.refine_variant_list()[:n]


def _trades_series(years, per_year, pnl_fn, start_month=2):
    out = []
    for y in years:
        for k in range(per_year):
            e = pd.Timestamp(year=y, month=start_month, day=1) + pd.Timedelta(days=10 * k)
            out.append(_trade(e, e + pd.Timedelta(days=3), pnl_fn(y, k)))
    return out


class TestWalkForward:
    def test_no_future_info(self):
        vs = _fake_wf_variants(2)
        a, b = vs[0]["name"], vs[1]["name"]
        # A：2001~2012每年8筆，PF 1.5(每4筆：+300,+300,−200,−200 → 600/400)
        ta = _trades_series(range(2001, 2013), 8, lambda y, k: [300, 300, -200, -200][k % 4])
        # B：2009年以前很爛；2010起每筆都大賺。另外2009/12/20進場、2010/1/15才出場的一筆超大獲利
        tb = _trades_series(range(2001, 2010), 8, lambda y, k: [100, -300][k % 2])
        tb.append(_trade("2009-12-20", "2010-01-15", 1e7))
        tb += _trades_series(range(2010, 2013), 8, lambda y, k: 5000.0)
        tb.sort(key=lambda t: t["entry_date"])
        rows = rf.walk_forward(vs, {a: ta, b: tb}, [2005, 2010, 2011])
        by = {r["year"]: r for r in rows}
        assert by[2005]["variant"]["name"] == a
        # 2010：B在2010才變好，而且跨年那筆在2010/1/1還沒出場 → 不能被算進去 → 仍選A
        assert by[2010]["variant"]["name"] == a
        assert by[2010]["hist_trades"] == 72  # A：2001~2009 × 8
        assert by[2010]["hist_pf"] == pytest.approx(1.5)
        assert all(pd.Timestamp(t["entry_date"]).year == 2010 for t in by[2010]["trades"])
        assert len(by[2010]["trades"]) == 8
        # 2011：B在2010年的8筆 + 跨年那筆都已出場 → B的歷史PF飆高 → 選B
        assert by[2011]["variant"]["name"] == b

    def test_entry_date_rule_would_have_leaked(self):
        """對照：如果用「進場日 < Y」，跨年那筆會讓B在2010被選中(=偷看未來)；我們用出場日就不會。"""
        vs = _fake_wf_variants(2)
        a, b = vs[0]["name"], vs[1]["name"]
        ta = _trades_series(range(2001, 2010), 8, lambda y, k: [300, -200][k % 2])
        tb = _trades_series(range(2001, 2010), 8, lambda y, k: [100, -300][k % 2])
        tb.append(_trade("2009-12-20", "2010-01-15", 1e7))
        rows = rf.walk_forward(vs, {a: ta, b: tb}, [2010])
        assert rows[0]["variant"]["name"] == a
        hist_b_by_entry = [t["pnl_ntd"] for t in tb if t["entry_date"] < pd.Timestamp("2010-01-01")]
        assert rf.profit_factor(hist_b_by_entry) > 1.5  # 用進場日的話B會贏

    def test_min_history_and_none(self):
        vs = _fake_wf_variants(2)
        a, b = vs[0]["name"], vs[1]["name"]
        ta = _trades_series([2001], 29, lambda y, k: 100.0 if k % 2 else -50.0)
        tb = _trades_series([2001], 30, lambda y, k: 10.0 if k % 2 else -50.0)
        rows = rf.walk_forward(vs, {a: ta, b: tb}, [2002])
        assert rows[0]["variant"]["name"] == b and rows[0]["n_candidates"] == 1  # A只有29筆
        rows = rf.walk_forward(vs, {a: ta, b: tb[:29]}, [2002])
        assert rows[0]["variant"] is None and rows[0]["trades"] == []

    def test_tie_breaking(self):
        vs = _fake_wf_variants(3)
        n0, n1, n2 = (v["name"] for v in vs)
        base = _trades_series([2001], 30, lambda y, k: 200.0 if k % 2 else -100.0)   # PF 2
        bigger = _trades_series([2001], 30, lambda y, k: 400.0 if k % 2 else -200.0)  # PF 2、損益較大
        rows = rf.walk_forward(vs, {n0: base, n1: bigger, n2: bigger}, [2002])
        assert rows[0]["variant"]["name"] == n1  # 同PF → 損益大 → 再同 → 登記順序(n1在n2前)
        rows = rf.walk_forward(vs, {n0: base, n1: base, n2: base}, [2002])
        assert rows[0]["variant"]["name"] == n0
        better_pf = _trades_series([2001], 30, lambda y, k: 50.0 if k % 2 else -10.0)  # PF 5、損益小
        rows = rf.walk_forward(vs, {n0: base, n1: bigger, n2: better_pf}, [2002])
        assert rows[0]["variant"]["name"] == n2

    def test_change_count_and_distribution(self):
        vs = rf.refine_variant_list()
        rows = [{"variant": vs[0]}, {"variant": vs[0]}, {"variant": vs[13]}, {"variant": None},
                {"variant": vs[-1]}]
        assert rf.wf_change_count(rows) == 3
        d = rf.wf_distribution(rows)
        assert d["方向"] == {"多空": 2, "只做多": 2}
        assert d["MACD參數"] == {"標準": 3, "(無MACD)": 1}
        assert d["出場"]["停損1.0/停利3.0"] == 2

    def test_on_synthetic_run(self, refine_res):
        res, vs = refine_res
        years = rf.wf_years(res["frames"])
        assert years[0] == 2006 and years[-1] == 2026
        by_v = rf.combined_trades(vs, res["trades"])
        rows = rf.walk_forward(vs, by_v, years)
        assert [r["year"] for r in rows] == years
        for r in rows:
            assert r["variant"] is not None and r["hist_trades"] >= 30
            # 選擇只看到出場日 < Y的交易：手動重算歷史PF
            h = [t["pnl_ntd"] for t in by_v[r["variant"]["name"]]
                 if pd.Timestamp(t["exit_date"]) < pd.Timestamp(year=r["year"], month=1, day=1)]
            assert len(h) == r["hist_trades"] and rf.profit_factor(h) == pytest.approx(r["hist_pf"])
            assert all(pd.Timestamp(t["entry_date"]).year == r["year"] for t in r["trades"])
            # 而且沒有其他合格變體的歷史PF更高
            for v in vs:
                hv = [t["pnl_ntd"] for t in by_v[v["name"]]
                      if pd.Timestamp(t["exit_date"]) < pd.Timestamp(year=r["year"], month=1, day=1)]
                if len(hv) >= 30:
                    assert rf.profit_factor(hv) <= r["hist_pf"] + 1e-12
        # 每段交易都有正確的period標記；2024~的交易來自近期段
        for lst in list(by_v.values())[:10]:
            for t in lst:
                y = pd.Timestamp(t["entry_date"]).year
                assert t["period"] == ("train" if y <= 2012 else "test" if y <= 2023 else "recent")

    def test_series_stats_and_buy_hold(self, refine_res):
        res, vs = refine_res
        years = rf.wf_years(res["frames"])
        by_v = rf.combined_trades(vs, res["trades"])
        fixed = [t for t in by_v[rf.fixed_baseline_name()] if pd.Timestamp(t["entry_date"]).year >= 2006]
        s = rf.series_stats(fixed, res["frames"], years, n_boot=50)
        assert s["trade_count"] == len(fixed) and s["n_years"] == len(years)
        assert s["mtm_mdd_ntd"] <= s["max_drawdown_ntd"] + 1e-6
        bh = rf.buy_hold_wf(res["frames"], years)
        assert bh["n_lots_segments"] == 2
        assert sum(bh["yearly"].values()) == pytest.approx(bh["total_pnl_ntd"])
        assert min(bh["yearly"]) == 2006

    def test_verdict(self):
        wf = {"profit_factor": 1.3, "bootstrap_pct_positive": 85.0, "pnl_excl_top3_ntd": 1.0, "ratio": 2.0,
              "pos_years": 12, "n_years": 21, "pos_year_pct": 12 / 21 * 100}
        bh = {"ratio": 1.5}
        ok, checks = rf.wf_verdict(wf, bh)
        assert ok and len(checks) == 5
        for k, bad in (("profit_factor", 1.0), ("bootstrap_pct_positive", 80.0), ("pnl_excl_top3_ntd", 0.0),
                       ("ratio", 1.5), ("pos_year_pct", 54.9)):
            assert not rf.wf_verdict(dict(wf, **{k: bad}), bh)[0], k
        assert rf.wf_verdict(dict(wf, pos_year_pct=55.0), bh)[0]


# ---------------------------------------------------------------------------
def _fake_stats(vs, fn):
    st = {}
    for v in vs:
        for p in rf.PERIOD_KEYS:
            pf, n, mdd = fn(v, p)
            st[(v["name"], p)] = {"profit_factor": pf, "trade_count": n, "total_pnl_ntd": 0.0, "mtm_mdd_ntd": mdd}
    return st


class TestDiagnostics:
    def test_axis_rows(self):
        vs = rf.refine_variant_list()

        def fn(v, p):
            good = v["direction"] == "long"
            return (1.5 if good else 0.8), (50 if v["macd"] != "慢" else 10), -1000.0
        rows = rf.axis_rows(vs, _fake_stats(vs, fn))
        assert len(rows) == 3 + 2 + 12 + 5
        by = {(r["維度"], r["設定"]): r for r in rows}
        assert by[("方向", "只做多")]["兩段PF都>1比例(%)"] == 100.0
        assert by[("方向", "多空")]["兩段PF都>1比例(%)"] == 0.0
        assert by[("MACD參數", "慢")]["合格數(訓練>=40筆)"] == 0 and by[("MACD參數", "慢")]["變體數"] == 96
        assert by[("MACD參數", "標準")]["測試PF中位數"] == pytest.approx((0.8 + 1.5) / 2)
        assert by[("進場規則", "MA_5_20&DONCHIAN20")]["變體數"] == 24
        assert by[("出場", "停損1.0/停利3.0")]["變體數"] == 26

    def test_head_to_head(self):
        vs = rf.refine_variant_list()

        def fn(v, p):
            pf = 1.2 if v["direction"] == "long" else 1.0
            if v["macd"] == "快":
                pf -= 0.5
            mdd = -500.0 if v["direction"] == "long" else -1000.0
            return pf, 50, mdd
        h = rf.head_to_head(vs, _fake_stats(vs, fn))
        assert h["只做多 vs 多空"] == {"n": 156, "test_pf_higher": 100.0, "recent_pf_higher": 100.0,
                                   "test_mdd_smaller": 100.0}
        assert h["MACD快 vs 標準"]["n"] == 96 and h["MACD快 vs 標準"]["test_pf_higher"] == 0.0
        assert h["MACD慢 vs 標準"]["test_pf_higher"] == 0.0 and h["MACD慢 vs 標準"]["test_mdd_smaller"] == 0.0


# ---------------------------------------------------------------------------
class TestRuntime:
    def test_runtime(self, refine_res):
        res, vs = refine_res
        print(f"\n[runtime] 精煉312變體×3期間×2(含成本加壓) = {res['timing']['n_runs']}次回測 + bootstrap(200次)："
              f"{res['wall_sec']:.1f}秒")
        assert res["timing"]["n_runs"] == 312 * 3 * 2
        assert len(res["stats"]) == 312 * 3
        assert res["wall_sec"] < 300


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth_market():
    index_df = make_index()
    fut, _ = make_futures_daily_truth(index_df)
    return index_df, daily_to_minutes(fut)


def _patch_sources(monkeypatch, index_df, bars):
    monkeypatch.setattr(tdd, "_yf_download", lambda symbol, start, end: make_yf_frame(index_df))
    monkeypatch.setattr(tdd, "_default_load_minute",
                        lambda refresh, cache_dir: (bars, {"periods": {k: {"reason": "ok", "n_rows": 1}
                                                                       for k in tdd.FUTURES_PERIOD_KEYS}}))


class TestRefineEndToEnd:
    def test_futures_mode(self, synth_market, tmp_path, monkeypatch, capsys):
        index_df, bars = synth_market
        _patch_sources(monkeypatch, index_df, bars)
        out = tmp_path / "res"
        rc = ctd.main(["--refine", "--results-dir", str(out), "--n-bootstrap", "100",
                       "--index-cache-dir", str(tmp_path / "ic")])
        assert rc == 0
        for f in ("summary_refine.txt", "tx_refine_all.csv", "tx_refine_walkforward.csv", "tx_refine_wf_trades.csv",
                  "tx_refine_axes.csv", "data_parse_report.txt", "data_validation.json"):
            assert (out / f).exists(), f
        assert not (out / "summary.txt").exists() and not (out / "summary_grid.txt").exists()
        s = (out / "summary_refine.txt").read_text(encoding="utf-8")
        assert s.splitlines()[0].startswith("價格來源：台指期")
        order = ["== 資料來源", "== 預先登記的規則", "== ⚠️ 多重比較 / 資料已經看過", "== Walk-forward判定",
                 "== 維度檢視", "== 配對比較", "== 各期間概況", "執行時間：", "== 注意事項"]
        pos = [s.index(k) for k in order]
        assert pos == sorted(pos)
        for key in ("312", "判定：", "固定基準", "買進持有", "換了", "被選方向分布",
                    "forward paper trading", "只做多 vs 多空", "MACD快 vs 標準", "MACD慢 vs 標準", "逐日市值",
                    "成本加壓", "出場日 < Y年1/1"):
            assert key in s, key
        assert s.count("[✓]") + s.count("[✗]") == 5
        printed = capsys.readouterr().out
        assert printed.index("預先登記的規則") < printed.index("Walk-forward判定")

        allv = pd.read_csv(out / "tx_refine_all.csv", encoding="utf-8-sig")
        assert len(allv) == 312 and list(allv["變體"]) == [v["name"] for v in rf.refine_variant_list()]
        for lab in ("訓練", "測試", "近期"):
            for c in ("交易數", "PF", "勝率(%)", "損益(NT$)", "平倉最大回撤(NT$)", "逐日市值最大回撤(NT$)",
                      "拿掉前3筆損益(NT$)", "bootstrap正報酬(%)", "平均持有天數", "在場時間比例(%)",
                      "多單損益(NT$)", "空單損益(NT$)", "成本加壓PF(滑價3點)", "獲利年度比例(%)"):
                assert f"{lab}{c}" in allv.columns, lab + c
            assert (allv[f"{lab}逐日市值最大回撤(NT$)"] <= allv[f"{lab}平倉最大回撤(NT$)"] + 1).all()
        assert (allv[allv["方向"] == "只做多"]["測試空單損益(NT$)"] == 0).all()

        wf = pd.read_csv(out / "tx_refine_walkforward.csv", encoding="utf-8-sig")
        assert list(wf["年度"]) == list(range(2006, 2027))
        assert wf["被選變體"].isin(allv["變體"]).all()
        assert allv["walk-forward被選中年數"].sum() == len(wf)
        tr = pd.read_csv(out / "tx_refine_wf_trades.csv", encoding="utf-8-sig")
        tr["年"] = pd.to_datetime(tr["進場日"]).dt.year
        chosen = dict(zip(wf["年度"], wf["被選變體"]))
        assert len(tr) == wf["當年交易數"].sum()
        assert all(chosen[y] == v for y, v in zip(tr["年"], tr["變體"]))
        assert set(tr["期間"]) <= {"train", "test", "recent"}
        assert (tr[tr["年"] >= 2024]["期間"] == "recent").all()
        ax = pd.read_csv(out / "tx_refine_axes.csv", encoding="utf-8-sig")
        assert len(ax) == 22 and list(ax["維度"].unique()) == ["MACD參數", "方向", "出場", "進場規則"]

    def test_flag_conflicts_and_no_data(self, synth_market, tmp_path, monkeypatch):
        assert ctd.main(["--refine", "--grid", "--results-dir", str(tmp_path)]) == 2
        assert ctd.main(["--refine", "--results-dir", str(tmp_path), "--variants", "DONCHIAN"]) == 2
        monkeypatch.setattr(tdd, "_yf_download", lambda *a: pd.DataFrame())
        monkeypatch.setattr(tdd.time, "sleep", lambda s: None)
        out = tmp_path / "res2"
        assert ctd.main(["--refine", "--results-dir", str(out), "--index-cache-dir", str(tmp_path / "ic2")]) == 1
        assert "回測沒有執行" in (out / "summary_refine.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
def _digest_csv(path, drop=()):
    """跟pandas版本無關的內容指紋：欄名 + 每列(數字一律格式化成小數4位)。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[[c for c in df.columns if c not in drop]]
    parts = ["|".join(df.columns)]
    for row in df.itertuples(index=False):
        parts.append("|".join(
            ("nan" if pd.isna(v) else f"{float(v):.4f}") if isinstance(v, (int, float, np.integer, np.floating))
            else str(v) for v in row))
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


# 在8f52b46(加--refine之前的HEAD)用同一份合成資料、--n-bootstrap 20 跑出來的指紋
HEAD_DIGESTS = {
    "basic/tx_daily_summary.csv": "9c664f61e083607d2b1ffaaf467d6080",
    "basic/tx_daily_trades.csv": "b4f865cb53210658386071dafc4792e1",
    "basic/tx_daily_yearly.csv": "d3c584f6dc2f34af26663c0a70e70dab",
    "grid/tx_grid_all.csv": "d61eeac36f9c4f29af62872b0d3fc3ef",
    "grid/tx_grid_top20.csv": "374ef4734db95b17aa0e03b8a976bf5c",
    "grid/tx_grid_family.csv": "1abd19baa28d9275b5b9c9e11644f270",
    "grid/tx_grid_exits.csv": "1f2659688a06a4ffe0e6ac62b84fc9d9",
    "grid/tx_grid_selected_trades.csv": "734a683e4167a9efcbddb0d3de948a01",
}
HEAD_SUMMARY_DIGESTS = {"basic/summary.txt": "55dc3ff768aea2ff609d74cc988468ca",
                        "grid/summary_grid.txt": "7cdbec781dbb088545c031f4c406f591"}  # 網格：去掉含「秒」的行
NEW_GRID_COLS = [f"{lab}{c}" for lab in ("訓練", "測試", "近期") for c in ("平均持有天數", "在場時間比例(%)")]


class TestBasicAndGridUnchanged:
    def test_outputs_identical_except_new_grid_columns(self, synth_market, tmp_path, monkeypatch):
        index_df, bars = synth_market
        _patch_sources(monkeypatch, index_df, bars)
        assert ctd.main(["--results-dir", str(tmp_path / "basic"), "--n-bootstrap", "20",
                         "--index-cache-dir", str(tmp_path / "ic")]) == 0
        assert ctd.main(["--grid", "--results-dir", str(tmp_path / "grid"), "--n-bootstrap", "20",
                         "--index-cache-dir", str(tmp_path / "ic")]) == 0
        for rel, exp in HEAD_DIGESTS.items():
            assert _digest_csv(tmp_path / rel, drop=NEW_GRID_COLS) == exp, rel
        for rel, exp in HEAD_SUMMARY_DIGESTS.items():
            txt = (tmp_path / rel).read_text(encoding="utf-8")
            if rel.startswith("grid"):
                txt = "\n".join(l for l in txt.splitlines() if "秒" not in l)
            assert hashlib.md5(txt.encode()).hexdigest() == exp, rel
        # 新欄位只在網格的all/top20，而且緊接在每段「拿掉前3筆損益」後面
        allv = pd.read_csv(tmp_path / "grid" / "tx_grid_all.csv", encoding="utf-8-sig")
        cols = list(allv.columns)
        for lab in ("訓練", "測試", "近期"):
            k = cols.index(f"{lab}拿掉前3筆損益(NT$)")
            assert cols[k + 1:k + 3] == [f"{lab}平均持有天數", f"{lab}在場時間比例(%)"]
            assert ((allv[f"{lab}在場時間比例(%)"] >= 0) & (allv[f"{lab}在場時間比例(%)"] <= 100)).all()
        basic_cols = pd.read_csv(tmp_path / "basic" / "tx_daily_summary.csv", encoding="utf-8-sig").columns
        assert not any(c in basic_cols for c in NEW_GRID_COLS)


# ---------------------------------------------------------------------------
class TestWithoutScipyAndNetwork:
    def test_refine_e2e_scipy_blocked_no_network(self, tmp_path):
        code = textwrap.dedent(f"""
            import sys
            class _Block:
                def find_spec(self, name, path=None, target=None):
                    if name == "scipy" or name.startswith("scipy."):
                        raise ImportError("scipy blocked for test")
                    return None
            sys.meta_path.insert(0, _Block())
            sys.path.insert(0, {REPO_DIR!r})
            import socket
            def _no_net(*a, **k):
                raise RuntimeError("network blocked in test")
            socket.socket.connect = _no_net
            import tx_daily_data as tdd
            import compare_tx_daily as ctd
            from test_tx_daily_data import make_index, make_futures_daily_truth, daily_to_minutes, make_yf_frame
            idx = make_index()
            fut, _ = make_futures_daily_truth(idx)
            bars = daily_to_minutes(fut)
            tdd._yf_download = lambda symbol, start, end: make_yf_frame(idx)
            tdd._default_load_minute = lambda refresh, cache_dir: (bars, {{"periods": {{}}}})
            rc = ctd.main(["--refine", "--results-dir", {str(tmp_path / 'res')!r}, "--n-bootstrap", "50",
                           "--index-cache-dir", {str(tmp_path / 'ic')!r}])
            assert rc == 0
            assert "scipy" not in sys.modules or sys.modules["scipy"] is None
            print("REFINE_E2E_OK")
        """)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_DIR)
        assert proc.returncode == 0, proc.stderr[-3000:]
        assert "REFINE_E2E_OK" in proc.stdout
        assert (tmp_path / "res" / "summary_refine.txt").exists()


class TestWorkflowMode:
    def test_refine_option(self):
        txt = open(os.path.join(REPO_DIR, ".github", "workflows", "tx_daily_backtest.yml"), encoding="utf-8").read()
        assert "- basic" in txt and "- grid" in txt and "- refine" in txt and "default: basic" in txt
        assert '"${{ github.event.inputs.mode }}" = "refine"' in txt and "--refine" in txt
        assert ("basic(8變體)/grid(1872變體網格)/refine(MACD等5種候選 × MACD參數 × 只做多/多空 × 12出場 = 312變體，"
                "walk-forward驗證)") in txt
        assert "test_tx_daily_refine.py" in txt
