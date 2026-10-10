"""
leader_breakout.py 的測試——全部合成資料，不連網路。
"""
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import leader_breakout as lb
import winners_study as ws

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW = os.path.join(HERE, ".github", "workflows", "leader_breakout.yml")


# ---------------------------------------------------------------------------
# 合成資料
# ---------------------------------------------------------------------------
def _ohlc(close, dates, volume, rng=None):
    close = np.asarray(close, dtype=float)
    prev = np.r_[close[0], close[:-1]]
    if rng is None:
        op = prev.copy()
        spread = np.full(len(close), 0.01)
    else:
        op = prev * (1 + rng.normal(0, 0.003, len(close)))
        spread = np.abs(rng.normal(0.01, 0.004, len(close)))
    hi = np.maximum(op, close) * (1 + spread)
    lo = np.minimum(op, close) * (1 - spread)
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Adj Close": close,
                         "Volume": np.full(len(close), float(volume))}, index=pd.DatetimeIndex(dates))


def build_synthetic(n=36, seed=7, start="2016-06-01", end="2025-06-30"):
    """n檔：一半電子股；每檔在不同時間有一段強勢上漲(會創新高)；^TWII 在2020年初大跌>15%後反彈。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    inds = ["半導體業", "電子零組件業", "航運業", "食品工業", "光電業", "資訊服務業"]
    # 大盤
    mret = rng.normal(0.0003, 0.009, T)
    crash0 = dates.searchsorted(pd.Timestamp("2020-02-20"))
    mret[crash0:crash0 + 20] = -0.012
    mret[crash0 + 20:crash0 + 60] = 0.008
    taiex = pd.Series(10000 * np.exp(np.cumsum(mret)), index=dates, name="Close")
    rows, prices = [], {}
    for i in range(n):
        code = f"{2300 + i}"
        rows.append({"code": code, "name": f"測試{i}", "market": "上市" if i % 3 else "上櫃",
                     "industry": inds[i % len(inds)]})
        beta = 0.5 + 1.5 * (i % 5) / 4
        drift = np.full(T, 0.0001)
        s0 = int(T * (0.25 + 0.6 * ((i * 7) % n) / n))
        drift[s0:s0 + 150] = 0.004
        rets = beta * mret + drift + rng.normal(0, 0.012, T)
        close = (30 + 3 * i) * np.exp(np.cumsum(rets))
        prices[code] = _ohlc(close, dates, 3_000_000, rng)
    uni = pd.DataFrame(rows)
    rev = []
    for i in range(n):
        for y in range(2016, 2026):
            for m in range(1, 13):
                yoy = rng.normal(40 if i % 2 == 0 else 0, 10)
                rev.append({"code": f"{2300 + i}", "name": "", "industry": "", "revenue": 1000.0,
                            "revenue_ly": 1000.0, "mom_pct": 0.0, "yoy_pct": yoy, "cum_yoy_pct": yoy,
                            "year": y, "month": m, "market": "上市"})
    ref = _ohlc(50 * np.exp(np.cumsum(mret * 0.95 + 0.0001)), dates, 1e7)
    return uni, prices, pd.DataFrame(rev), taiex, ref


@pytest.fixture(scope="module")
def synth():
    return build_synthetic()


@pytest.fixture(scope="module")
def backtest(synth):
    uni, prices, rev, taiex, ref = synth
    return lb.run_backtest(uni, prices, rev, taiex, ref, "2018-01-01", "2025-06-30", 200_000)


def make_P(close, open_=None, high=None, low=None, atr=1.0, ma60=None, rs=None, turnover=1e8, raw_close=None):
    """手工小型P(T×N)，給投資組合模擬的單元測試用。"""
    close = np.atleast_2d(np.asarray(close, dtype=float))
    if close.shape[0] == 1:
        close = close.T
    T, N = close.shape

    def arr(v, default):
        if v is None:
            return default.copy()
        a = np.asarray(v, dtype=float)
        if a.ndim == 0:
            return np.full((T, N), float(a))
        a = np.atleast_2d(a)
        return a.T if a.shape[0] == 1 and N == 1 else a
    o = arr(open_, close)
    A = {"close": close, "open": o, "high": arr(high, np.maximum(o, close)), "low": arr(low, np.minimum(o, close)),
         "atr14": arr(atr, close), "ma60": arr(ma60, np.full((T, N), np.nan)), "rs": arr(rs, np.full((T, N), 0.5)),
         "turnover20": arr(turnover, close), "raw_close": arr(raw_close, close)}
    A["valid"] = np.isfinite(A["close"]) & np.isfinite(A["open"])
    A["close_ff"] = pd.DataFrame(A["close"]).ffill().to_numpy()
    return {"A": A}


def sig_at(T, N, cells):
    s = np.zeros((T, N), dtype=bool)
    for t, j in cells:
        s[t, j] = True
    return s


# ---------------------------------------------------------------------------
# 變體
# ---------------------------------------------------------------------------
class TestVariants:
    def test_count_and_names(self):
        v = lb.build_variants()
        assert len(v) == 38
        names = [x["name"] for x in v]
        assert len(set(names)) == 38
        assert "E_D｜X3｜K1｜大盤濾網" in names and "E_A｜X2｜K3｜無濾網" in names
        assert "E_CRASH｜X2｜K1" in names and "E_D｜X3｜K3｜無濾網｜成交金額排序" in names
        assert sum(x["entry"] == "E_CRASH" for x in v) == 4
        assert all(x["filter"] is None for x in v if x["entry"] == "E_CRASH")
        assert sum(x["rank"] == "成交金額" and x["entry"] == "E_D" for x in v) == 2

    def test_market_filter_mask(self):
        sigs = {"E_A": np.ones((3, 2), bool), "E_CRASH": np.ones((3, 2), bool)}
        P = {"filter_ok": np.array([True, False, True])}
        v = {"entry": "E_A", "filter": True}
        assert lb.variant_signal(sigs, P, v)[:, 0].tolist() == [True, False, True]
        assert lb.variant_signal(sigs, P, {"entry": "E_A", "filter": False}).all()
        assert lb.variant_signal(sigs, P, {"entry": "E_CRASH", "filter": None}).all()   # E_CRASH不套濾網


# ---------------------------------------------------------------------------
# 特徵：point-in-time
# ---------------------------------------------------------------------------
class TestFeatures:
    def test_no_lookahead_when_future_appended(self, synth):
        uni, prices, rev, taiex, _ = synth
        cut = pd.Timestamp("2019-06-28")
        p_cut = {c: df[df.index <= cut] for c, df in prices.items()}
        rev_cut = rev[(rev["year"] * 12 + rev["month"]) <= (2019 * 12 + 5)]
        full = lb.build_panels(uni, prices, rev, taiex)
        part = lb.build_panels(uni, p_cut, rev_cut, taiex[taiex.index <= cut])
        n = len(part["cal"])
        assert list(full["cal"][:n]) == list(part["cal"])
        for k in ["close", "atr14", "ma60", "turnover20", "hh252", "ll252", "ret126", "rs", "yoy3", "beta", "beta_q"]:
            np.testing.assert_allclose(full["A"][k][:n], part["A"][k], rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=k)
        for k in ["elig", "entry_a"]:
            assert (full["A"][k][:n] == part["A"][k]).all(), k
        assert (full["filter_ok"][:n] == part["filter_ok"]).all()
        assert (full["crash_day"][:n] == part["crash_day"]).all()
        s_full, s_part = lb.entry_signals(full), lb.entry_signals(part)
        for k in s_part:
            assert (s_full[k][:n] == s_part[k]).all(), k

    def test_beta_matches_winners_extra(self, synth):
        import winners_extra as wx
        uni, prices, rev, taiex, _ = synth
        P = lb.build_panels(uni.head(3), prices, None, taiex)
        t = 900
        d = P["cal"][t]
        code = P["codes"][1]
        ind = ws.compute_indicators(ws.prepare_price_frame(prices[code]))
        ref = wx.tech_extra_at(ind, d, taiex)["beta"]
        assert np.isfinite(ref) and abs(P["A"]["beta"][t, 1] - ref) < 1e-8

    def test_rs_and_beta_quintile_cross_section(self, synth):
        uni, prices, rev, taiex, _ = synth
        P = lb.build_panels(uni, prices, rev, taiex)
        A = P["A"]
        t = 800
        e = A["elig"][t]
        ret = A["ret126"][t, e]
        rs = A["rs"][t, e]
        best = np.nanargmax(ret)
        assert rs[best] == 1.0 and np.all(np.isnan(A["rs"][t, ~e]))
        q = A["beta_q"][t, e]
        assert set(np.unique(q[np.isfinite(q)])) <= {1, 2, 3, 4, 5}
        b = A["beta"][t, e]
        assert q[np.nanargmax(b)] == 5 and q[np.nanargmin(b)] == 1
        assert P["elec"].sum() == sum(i in lb.ELEC_INDUSTRIES for i in uni["industry"])

    def test_eligibility_turnover_threshold(self):
        dates = pd.bdate_range("2016-06-01", periods=300)
        uni = pd.DataFrame({"code": ["1111", "2222"], "name": ["甲", "乙"], "market": ["上市", "上市"],
                            "industry": ["半導體業", "航運業"]})
        prices = {"1111": _ohlc(np.full(300, 100.0), dates, 400_000),   # 4,000萬
                  "2222": _ohlc(np.full(300, 100.0), dates, 200_000)}   # 2,000萬
        P = lb.build_panels(uni, prices, None, pd.Series(dtype=float))
        assert P["A"]["elig"][50:, 0].all() and not P["A"]["elig"][:, 1].any()
        assert not P["A"]["elig"][:18, 0].any()          # 20日均額還沒有值
        assert not P["twii_ok"] and not P["filter_ok"].any() and not P["crash_day"].any()


class TestRevenuePIT:
    def test_boundary_11th(self):
        dates = pd.bdate_range("2016-06-01", "2018-03-30")
        uni = pd.DataFrame({"code": ["1111"], "name": ["甲"], "market": ["上市"], "industry": ["半導體業"]})
        prices = {"1111": _ohlc(np.linspace(50, 60, len(dates)), dates, 1e6)}
        rev = []
        for y, m in ws.months_between(pd.Timestamp("2016-06-01"), pd.Timestamp("2018-02-01")):
            yoy = 100.0 if (y, m) == (2017, 12) else 0.0
            rev.append({"code": "1111", "yoy_pct": yoy, "revenue": 1.0, "year": y, "month": m})
        P = lb.build_panels(uni, prices, pd.DataFrame(rev), pd.Series(dtype=float))
        s = pd.Series(P["A"]["yoy3"][:, 0], index=P["cal"])
        # 2017-12 營收 2018-01-11 才看得到：1/10 的3月均YoY還是0，1/11 起 = (0+0+100)/3
        assert s.loc["2018-01-10"] == pytest.approx(0.0)
        assert s.loc["2018-01-11"] == pytest.approx(100 / 3)
        assert s.loc["2018-02-12"] == pytest.approx(100 / 3)       # 2018-01營收(0)加入，仍含12月
        assert s.loc["2018-03-12"] == pytest.approx(100 / 3)       # 2018-02 沒資料 → 停在最新已知月份
        assert s.loc["2018-01-02"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 進場規則
# ---------------------------------------------------------------------------
class TestEntryRules:
    def test_e_a_first_new_high_after_20_days(self):
        dates = pd.bdate_range("2016-06-01", periods=330)
        c = np.full(330, 100.0)
        c[280] = 110.0                  # 第一次新高
        c[281:] = 105.0
        c[290] = 112.0                  # 10天內再創新高 → 不是E_A
        c[291:] = 105.0
        c[320] = 120.0                  # 前20天沒有新高 → E_A
        uni = pd.DataFrame({"code": ["1111"], "name": ["甲"], "market": ["上市"], "industry": ["半導體業"]})
        P = lb.build_panels(uni, {"1111": _ohlc(c, dates, 1e6)}, None, pd.Series(dtype=float))
        ea = np.flatnonzero(lb.entry_signals(P)["E_A"][:, 0])
        assert ea.tolist() == [280, 320]
        assert np.isfinite(P["A"]["hh252_prev"][280, 0]) and P["A"]["ll252"][300, 0] == 100.0

    def _P(self):
        T, N = 3, 4
        A = {"entry_a": np.ones((T, N), bool), "elig": np.ones((T, N), bool),
             "yoy3": np.array([[30, 10, 30, np.nan]] * T, float),
             "rs": np.array([[0.9, 0.9, 0.5, 0.85]] * T, float),
             "beta_q": np.array([[5, 5, 4, 5]] * T, float)}
        A["elig"][:, 3] = [True, True, False]
        return {"A": A, "elec": np.array([True, False, True, True]), "crash_day": np.array([False, True, True])}

    def test_b_c_d(self):
        s = lb.entry_signals(self._P())
        assert s["E_A"][0].tolist() == [True, True, True, True]
        assert s["E_B"][0].tolist() == [True, False, True, False]       # 營收>20%
        assert s["E_C"][0].tolist() == [True, True, False, True]        # RS≥0.8
        assert s["E_D"][0].tolist() == [True, False, False, False]
        assert not s["E_A"][2, 3]                                       # 不合格的那天沒有訊號

    def test_crash_candidates(self):
        s = lb.entry_signals(self._P())["E_CRASH"]
        assert not s[0].any()                                           # 不是觸發日
        assert s[1].tolist() == [True, False, False, True]              # 電子+beta第5分位+合格
        assert s[2].tolist() == [True, False, False, False]             # 第4檔當天不合格


class TestCrashState:
    def test_fire_once_then_rearm_by_new_high(self):
        dd = np.array([0, -0.05, -0.16, -0.20, -0.18, -0.10, -0.05, 0.0, -0.1, -0.16, -0.12])
        cross = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1], bool)
        out = lb.crash_signal_days(dd, cross, lookback=60, rearm_days=120)
        assert np.flatnonzero(out).tolist() == [4, 10]   # 第5、6天再穿越不算；第7天新高重新武裝 → 第9天新深跌、第10天觸發

    def test_no_cross_before_deep_and_lookback(self):
        dd = np.array([-0.16, -0.1, -0.1, -0.1, -0.1])
        cross = np.array([0, 0, 0, 1, 0], bool)
        assert np.flatnonzero(lb.crash_signal_days(dd, cross, lookback=3)).tolist() == []   # 深跌距今≥3天
        assert np.flatnonzero(lb.crash_signal_days(dd, cross, lookback=4)).tolist() == [3]
        dd2 = np.array([-0.1, -0.12, -0.14, -0.1])
        assert not lb.crash_signal_days(dd2, np.ones(4, bool)).any()                         # 沒有到15%

    def test_rearm_after_days_in_persistent_bear(self):
        n = 20
        dd = np.full(n, -0.2)
        cross = np.zeros(n, bool)
        cross[[2, 5, 9, 12]] = True
        out = lb.crash_signal_days(dd, cross, lookback=60, rearm_days=8)
        # 第2天觸發；第10天(過了8天)重新武裝，之後(第11天起)的深跌才算 → 第12天觸發
        assert np.flatnonzero(out).tolist() == [2, 12]

    def test_deep_before_rearm_does_not_count(self):
        dd = np.array([-0.2, -0.2, 0.0, -0.05, -0.05])
        cross = np.array([0, 1, 0, 1, 1], bool)
        assert np.flatnonzero(lb.crash_signal_days(dd, cross)).tolist() == [1]

    def test_market_features(self):
        dates = pd.bdate_range("2018-01-01", periods=400)
        c = np.r_[np.linspace(100, 120, 300), np.linspace(120, 95, 30), np.linspace(95, 105, 70)]
        mf = lb.market_features(pd.Series(c, index=dates))
        assert mf["crash"].sum() == 1
        t = int(np.flatnonzero(mf["crash"].to_numpy())[0])
        assert c[t] > mf["ma20"].iloc[t] and c[t - 1] <= mf["ma20"].iloc[t - 1]
        assert mf["dd"].iloc[329] < -0.15
        assert bool(mf["filter_ok"].iloc[299]) and not bool(mf["filter_ok"].iloc[330])
        assert lb.market_features(pd.Series(dtype=float)).empty


# ---------------------------------------------------------------------------
# 出場、停損成交
# ---------------------------------------------------------------------------
def _run(P, sig, exit_kind="X2", k=1, capital=1e6, i0=1, i1=None, rank="RS"):
    i1 = P["A"]["close"].shape[0] - 1 if i1 is None else i1
    return lb.simulate_portfolio(P, sig, i0, i1, exit_kind, k, capital, rank=rank)


class TestExits:
    def test_initial_stop_intraday(self):
        c = [100, 100, 100, 100, 100]
        low = [99, 99, 99, 95, 99]
        P = make_P(c, low=low, atr=2.0)
        r = _run(P, sig_at(5, 1, [(1, 0)]))
        tr = r["trades"][0]
        assert tr["entry_idx"] == 2 and tr["entry_price"] == 100
        assert tr["exit_idx"] == 3 and tr["exit_price"] == pytest.approx(96.0) and tr["exit_reason"] == "初始停損"

    def test_initial_stop_gap(self):
        c = [100, 100, 100, 90, 90]
        o = [100, 100, 100, 92, 90]
        P = make_P(c, open_=o, atr=2.0)
        tr = _run(P, sig_at(5, 1, [(1, 0)]))["trades"][0]
        assert tr["exit_idx"] == 3 and tr["exit_price"] == 92 and tr["exit_reason"] == "初始停損(跳空)"

    def test_x2_trailing_and_gap(self):
        c = [100, 100, 100, 150, 140, 125, 110]
        o = [100, 100, 100, 140, 145, 135, 124]
        lo = [99, 99, 99, 139, 135, 121, 105]
        P = make_P(c, open_=o, low=lo, atr=2.0)
        tr = _run(P, sig_at(7, 1, [(1, 0)]), "X2")["trades"][0]
        # 最高收盤150 → 停損120，第6天最低105 碰到 → 120 成交
        assert tr["exit_idx"] == 6 and tr["exit_price"] == pytest.approx(120.0) and tr["exit_reason"] == "移動停損"
        o2 = list(o)
        o2[6] = 115
        lo2 = list(lo)
        lo2[6] = 110
        tr2 = _run(make_P(c, open_=o2, low=lo2, atr=2.0), sig_at(7, 1, [(1, 0)]), "X2")["trades"][0]
        assert tr2["exit_price"] == 115 and tr2["exit_reason"] == "移動停損(跳空)"

    def test_x3_close_below_ma60_next_open(self):
        c = [100, 100, 100, 105, 95, 97]
        o = [100, 100, 100, 101, 104, 96]
        ma = [90, 90, 90, 98, 98, 98]
        P = make_P(c, open_=o, atr=10.0, ma60=ma)
        tr = _run(P, sig_at(6, 1, [(1, 0)]), "X3")["trades"][0]
        assert tr["exit_idx"] == 5 and tr["exit_price"] == 96 and tr["exit_reason"] == "跌破MA60"

    def test_x3_has_no_trailing(self):
        c = [100, 100, 100, 200, 150, 150]
        P = make_P(c, atr=2.0, ma60=[50] * 6)
        r = _run(P, sig_at(6, 1, [(1, 0)]), "X3")
        assert r["trades"] == [] and len(r["open"]) == 1
        op = r["open"][0]
        assert op["open_at_end"] and op["exit_price"] == 150

    def test_open_position_marked_at_last_close(self):
        c = [100, 100, 100, 110, 120]
        P = make_P(c, atr=2.0)
        r = _run(P, sig_at(5, 1, [(1, 0)]), capital=100_000)
        op = r["open"][0]
        shares = op["shares"]
        assert op["pnl"] == pytest.approx(shares * 120 - (shares * 100 + op["buy_fee"]))
        assert r["equity"][-1] == pytest.approx(r["cash"] + shares * 120)


# ---------------------------------------------------------------------------
# 部位大小、成本
# ---------------------------------------------------------------------------
class TestSizing:
    def test_risk_based(self):
        # 權益20萬×1.5%=3000，每股風險2×ATR=10 → 300股
        assert lb.position_shares(200_000, 200_000, 100.0, 5.0, 1) == 300

    def test_equity_over_k_cap(self):
        # 風險算出 3000/0.2 = 15000股；權益/K = 66,666 → 666股
        assert lb.position_shares(200_000, 200_000, 100.0, 0.1, 3) == 666

    def test_cash_cap_including_fee(self):
        n = lb.position_shares(200_000, 10_000, 100.0, 0.1, 1)
        assert n == 99            # 100股要 10,000+20(最低手續費) > 10,000
        assert n * 100 + lb.buy_fee(n * 100) <= 10_000

    def test_odd_lot_and_zero(self):
        assert lb.position_shares(200_000, 200_000, 2000.0, 100.0, 1) == 15    # 零股
        assert lb.position_shares(200_000, 200_000, 5000.0, 2000.0, 1) == 0    # 連1股都不到 → 不買
        assert lb.position_shares(200_000, 15, 10.0, 0.1, 1) == 0              # 現金不夠付最低手續費

    def test_fees(self):
        assert lb.buy_fee(1000) == 20 and lb.buy_fee(100_000) == pytest.approx(142.5)
        assert lb.sell_cost(1000) == pytest.approx(20 + 3)
        assert lb.sell_cost(100_000) == pytest.approx(142.5 + 300)

    def test_trade_pnl_with_costs(self):
        c = [100, 100, 100, 100, 100, 100]
        o = [100, 100, 100, 100, 100, 110]
        P = make_P(c, open_=o, atr=5.0, ma60=[np.nan] * 4 + [101, 101])
        r = _run(P, sig_at(6, 1, [(1, 0)]), "X3", capital=200_000)
        tr = r["trades"][0]
        assert tr["shares"] == 300 and tr["buy_fee"] == pytest.approx(42.75)
        assert tr["exit_price"] == 110
        exp = 300 * 110 - (300 * 110 * 0.001425 + 300 * 110 * 0.003) - (30000 + 42.75)
        assert tr["pnl"] == pytest.approx(exp)
        assert r["cash"] == pytest.approx(200_000 + exp)

    def test_sizing_uses_prior_close_equity(self):
        # 第一筆部位賺錢後，第二筆用「訊號日收盤權益」算風險
        T = 8
        c = np.array([[100, 50]] * T, float)
        c[3:, 0] = 200
        P = make_P(c, atr=np.array([[5.0, 1.0]] * T))
        sig = sig_at(T, 2, [(1, 0), (4, 1)])
        r = _run(P, sig, "X2", k=3, capital=200_000)
        eq_sig = r["equity"][4 - 1]       # i0=1 → 第4天在 index 3
        second = [x for x in r["open"] if x["code_idx"] == 1][0]
        assert second["shares"] == min(int(eq_sig * 0.015 / 2.0), int(eq_sig / 3 / 50))


# ---------------------------------------------------------------------------
# 排序、空位、漲停
# ---------------------------------------------------------------------------
class TestRankingSlots:
    def _P3(self):
        T = 6
        c = np.full((T, 3), 100.0)
        rs = np.array([[0.85, 0.95, 0.95]] * T)
        to = np.array([[9e8, 1e8, 2e8]] * T)
        return make_P(c, atr=1.0, rs=rs, turnover=to)

    def test_rank_by_rs_then_turnover(self):
        order = lb.rank_order(np.array([0, 1, 2]), np.array([0.85, 0.95, 0.95]), np.array([9e8, 1e8, 2e8]))
        assert order.tolist() == [2, 1, 0]
        order2 = lb.rank_order(np.array([0, 1]), np.array([np.nan, 0.1]), np.array([1, 1]))
        assert order2.tolist() == [1, 0]

    def test_slot_limit_k1(self):
        P = self._P3()
        r = _run(P, sig_at(6, 3, [(1, 0), (1, 1), (1, 2)]), "X2", k=1)
        held = r["open"]
        assert len(held) == 1 and held[0]["code_idx"] == 2
        assert r["counts"]["slot_skipped"] == 2

    def test_slot_limit_k3_and_turnover_rank(self):
        P = self._P3()
        r = _run(P, sig_at(6, 3, [(1, 0), (1, 1), (1, 2)]), "X2", k=3)
        assert sorted(x["code_idx"] for x in r["open"]) == [0, 1, 2]
        r2 = _run(P, sig_at(6, 3, [(1, 0), (1, 1), (1, 2)]), "X2", k=1, rank="成交金額")
        assert r2["open"][0]["code_idx"] == 0

    def test_no_duplicate_and_slots_stay_full(self):
        P = self._P3()
        r = _run(P, sig_at(6, 3, [(1, 2), (2, 2), (3, 0)]), "X2", k=1)
        assert len(r["open"]) == 1 and r["open"][0]["code_idx"] == 2 and r["counts"]["slot_skipped"] == 1
        assert r["npos"].max() == 1

    def test_limit_up_skip_takes_next(self):
        T = 5
        c = np.full((T, 2), 100.0)
        o = c.copy()
        h = c.copy()
        lo = c.copy()
        o[2, 0] = h[2, 0] = lo[2, 0] = c[2, 0] = 110.0       # 第0檔進場日一價漲停
        P = make_P(c, open_=o, high=h, low=lo, atr=1.0, rs=np.array([[0.99, 0.9]] * T))
        r = _run(P, sig_at(T, 2, [(1, 0), (1, 1)]), "X2", k=1)
        assert r["counts"]["limit_up_skipped"] == 1
        assert r["open"][0]["code_idx"] == 1
        assert lb.is_limit_up_open(P["A"], 2, 0) and not lb.is_limit_up_open(P["A"], 2, 1)

    def test_limit_up_needs_locked_bar(self):
        T = 3
        c = np.full((T, 1), 100.0)
        o = c.copy()
        o[2, 0] = 110
        h = np.maximum(o, c) + 1
        P = make_P(c, open_=o, high=h, atr=1.0)
        assert not lb.is_limit_up_open(P["A"], 2, 0)      # 開漲停但盤中有打開 → 買得到

    def test_limit_up_uses_raw_prices(self):
        # 還原價在除息日會讓「開盤/前收」看起來不一樣；用原始價判斷
        T = 3
        c = np.full((T, 1), 100.0)
        o = c.copy()
        o[2] = 104.0
        P = make_P(c, open_=o, high=o, low=o, atr=1.0, raw_close=np.array([[100.0], [95.0], [100.0]]))
        # 原始開盤 = 104×100/100 = 104 ≥ 95×1.095=104.03? 否
        assert not lb.is_limit_up_open(P["A"], 2, 0)
        P["A"]["raw_close"][1, 0] = 94.0      # 94×1.095=102.93 ≤ 104 → 漲停(用還原前收100就不會判成漲停)
        assert lb.is_limit_up_open(P["A"], 2, 0)


# ---------------------------------------------------------------------------
# 權益、回撤、CAGR、PASS
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_mtm_equity_and_drawdown(self):
        c = [100, 100, 100, 120, 90, 130]
        P = make_P(c, atr=20.0)
        r = _run(P, sig_at(6, 1, [(1, 0)]), "X3", capital=200_000)
        op = r["open"][0]
        n = op["shares"]
        cash = 200_000 - n * 100 - op["buy_fee"]
        exp = [200_000, cash + n * 100, cash + n * 120, cash + n * 90, cash + n * 130]
        np.testing.assert_allclose(r["equity"], exp)
        eq = pd.Series([200_000] + exp, index=pd.bdate_range("2020-01-01", periods=6))
        assert lb.max_drawdown(eq) == pytest.approx((cash + n * 90) / (cash + n * 120) - 1)
        assert r["npos"].tolist() == [0, 1, 1, 1, 1]

    def test_missing_price_day_uses_last_close(self):
        c = np.array([100, 100, 100, np.nan, 110], float)
        P = make_P(c, atr=1.0)
        r = _run(P, sig_at(5, 1, [(1, 0)]), "X3", capital=100_000)
        n = r["open"][0]["shares"]
        assert r["equity"][2] == pytest.approx(r["equity"][1])          # 停牌日用前一天收盤估值

    def test_cagr(self):
        assert lb.cagr(100, 121, "2020-01-01", "2022-01-01") == pytest.approx(0.1, abs=1e-3)
        assert lb.cagr(100, 100, "2020-01-01", "2021-01-01") == 0
        assert np.isnan(lb.cagr(0, 100, "2020-01-01", "2021-01-01"))

    def test_yearly_and_window_slice(self):
        idx = pd.DatetimeIndex(["2017-12-29", "2018-06-01", "2018-12-31", "2019-06-03", "2019-12-31", "2020-03-02"])
        eq = pd.Series([100, 110, 120, 90, 108, 120], index=idx)
        yr = lb.yearly_returns(eq)
        assert yr[2018] == pytest.approx(0.2) and yr[2019] == pytest.approx(-0.1) and yr[2020] == pytest.approx(120 / 108 - 1)
        w = lb.window_slice(eq, "2019-01-01", "2019-12-31")
        assert list(w.values) == [120, 90, 108] and w.index[0] == pd.Timestamp("2018-12-31")
        m = lb.window_metrics(w)
        assert m["total"] == pytest.approx(-0.1) and m["mdd"] == pytest.approx(-0.25)
        assert m["worst_year"] == pytest.approx(-0.1) and m["pct_years_pos"] == 0
        assert lb.window_slice(eq, "2016-01-01", "2016-12-31").empty

    def test_trade_stats(self):
        s = lb.trade_stats(np.array([100.0, -50, 300, -50, 10, 20, 30]), np.array([1, 2, 3, 4, 5, 6, 7.0]))
        assert s["trades"] == 7 and s["pf"] == pytest.approx(460 / 100)
        assert s["pnl_ex_top5"] == pytest.approx(360 - (300 + 100 + 30 + 20 + 10))
        assert 0 <= s["boot_pos"] <= 1 and s["avg_hold"] == 4
        e = lb.trade_stats(np.array([]), np.array([]))
        assert e["trades"] == 0 and np.isnan(e["pf"]) and np.isnan(e["boot_pos"])

    def _wm(self, full_cagr=0.15, sub_cagr=(0.05, 0.1, 0.02, 0.3), mdd=-0.2, pf=(1.5, 1.3, 1.25, 2.0)):
        wm = {lb.FULL_WINDOW: {"cagr": full_cagr, "mdd": mdd, "pf": 1.5}}
        for (n, _, _), c, p in zip(lb.SUB_WINDOWS, sub_cagr, pf):
            wm[n] = {"cagr": c, "pf": p}
        return wm

    def test_pass_rule(self):
        assert lb.pass_checks(self._wm(), -0.3)["pass"]
        r = lb.pass_checks(self._wm(full_cagr=0.09), -0.3)
        assert not r["pass"] and not r["chk_cagr"] and r["chk_sub_cagr"] and r["chk_mdd"] and r["chk_sub_pf"]
        assert not lb.pass_checks(self._wm(sub_cagr=(0.05, -0.01, 0.1, 0.1)), -0.3)["chk_sub_cagr"]
        assert not lb.pass_checks(self._wm(mdd=-0.35), -0.3)["chk_mdd"]
        assert not lb.pass_checks(self._wm(), np.nan)["chk_mdd"]          # 0050抓不到 → 無法判斷 → ✗
        assert not lb.pass_checks(self._wm(pf=(1.5, 1.2, 1.3, 1.3)), -0.3)["chk_sub_pf"]   # 要 >1.2
        assert lb.pass_checks(self._wm(pf=(np.inf, 1.3, 1.3, 1.3)), -0.3)["chk_sub_pf"]
        assert not lb.pass_checks(self._wm(pf=(np.nan, 1.3, 1.3, 1.3)), -0.3)["chk_sub_pf"]   # 沒有交易
        assert not lb.pass_checks(self._wm(sub_cagr=(np.nan, 0.1, 0.1, 0.1)), -0.3)["chk_sub_cagr"]


# ---------------------------------------------------------------------------
# 整體回測(合成資料)
# ---------------------------------------------------------------------------
class TestBacktest:
    def test_shapes(self, backtest):
        v, w = backtest["variants"], backtest["windows"]
        assert len(v) == 38
        assert len(w) == 38 * 5 + 2 * 5
        assert set(w["window"]) == {lb.FULL_WINDOW} | {n for n, _, _ in lb.SUB_WINDOWS}
        eq = backtest["equity"]
        assert eq.shape[1] == 40 and eq.index[0] < pd.Timestamp("2018-01-01") <= eq.index[1]
        assert (eq.iloc[0, :38] == 200_000).all()
        assert backtest["axis"]["n"].groupby(backtest["axis"]["axis"]).sum().eq(36).all()

    def test_trades_consistent(self, backtest):
        t = backtest["trades"]
        assert len(t) > 0
        assert (t["entry_date"] > t["signal_date"]).all() and (t["exit_date"] >= t["entry_date"]).all()
        assert (t["shares"] >= 1).all() and (t["shares"] == t["shares"].round()).all()
        assert (t["entry_date"] >= pd.Timestamp("2018-01-01")).all()
        for name, g in t.groupby("variant"):
            k = int(name.split("｜")[2][1:])
            # 任何一天同時持有不超過K檔
            days = pd.bdate_range("2018-01-01", "2025-06-30")
            held = np.zeros(len(days), int)
            for _, r in g.iterrows():
                a = days.searchsorted(r["entry_date"])
                b = days.searchsorted(r["exit_date"])
                held[a:b] += 1
            assert held.max() <= k, name
        assert set(t.loc[t["entry_rule"] == "E_CRASH", "industry"]) <= set(lb.ELEC_INDUSTRIES)

    def test_final_equity_matches_trades(self, backtest):
        t = backtest["trades"]
        v = backtest["variants"].set_index("variant")
        for name in ["E_A｜X2｜K1｜無濾網", "E_C｜X3｜K3｜大盤濾網"]:
            g = t[t["variant"] == name]
            closed = g[~g["open_at_end"]]["pnl"].sum()
            op = g[g["open_at_end"]]["pnl"].sum()
            assert v.loc[name, "final_equity"] == pytest.approx(200_000 + closed + op, rel=1e-9)

    def test_filter_reduces_entries(self, backtest, synth):
        t = backtest["trades"]
        uni, prices, rev, taiex, _ = synth
        mf = lb.market_features(taiex)
        f = t[t["variant"].str.endswith("大盤濾網")]
        ok = mf["filter_ok"].reindex(pd.DatetimeIndex(f["signal_date"])).to_numpy()
        assert ok.all()

    def test_references(self, backtest):
        eq = backtest["equity"]
        assert eq[lb.REF_TWII].iloc[0] == 200_000 and eq[lb.REF_0050].notna().all()


# ---------------------------------------------------------------------------
# main 端到端(下載函式全部換成合成資料)
# ---------------------------------------------------------------------------
def _patch(monkeypatch, synth, uni_empty=False, rev_ok=True):
    uni, prices, rev, taiex, ref = synth
    calls = {"prices": []}

    def fake_prices(universe, dl_start, dl_end, cache_dir, refresh, debug_dir):
        calls["prices"].append((list(universe["code"]), dl_start, dl_end, debug_dir))
        if list(universe["code"]) == ["0050"]:
            return {"0050": ref}, pd.DataFrame(columns=["代號"])
        return {c: prices[c] for c in universe["code"]}, pd.DataFrame(columns=["代號"])

    def fake_rev(start_month, end_month, cache_dir, refresh, debug_dir):
        calls["rev"] = (start_month, end_month)
        return (rev, True) if rev_ok else (pd.DataFrame(), False)
    monkeypatch.setattr(ws, "load_universe", lambda c, r, d, t: ((uni.iloc[0:0] if uni_empty else uni.copy()), []))
    monkeypatch.setattr(ws, "load_prices", fake_prices)
    monkeypatch.setattr(ws, "load_revenue", fake_rev)
    monkeypatch.setattr(ws, "load_taiex", lambda *a, **k: taiex)
    return calls


class TestMain:
    def test_outputs(self, monkeypatch, tmp_path, synth):
        calls = _patch(monkeypatch, synth)
        out = tmp_path / "results_leader"
        rc = lb.main(["--output-dir", str(out), "--end", "2025-06-30", "--cache-dir", str(tmp_path / "c"),
                      "--capital", "300000"])
        assert rc == 0
        for fn in ["summary_leader.txt", "leader_variants.csv", "leader_windows.csv", "leader_trades.csv",
                   "leader_equity.csv", "leader_yearly.csv"]:
            assert (out / fn).exists(), fn
        assert (out / "debug").is_dir()
        assert calls["prices"][0][1:3] == ("2016-06-01", "2025-07-01")
        assert calls["prices"][1][0] == ["0050"] and calls["prices"][1][3].endswith("ref_0050")
        assert calls["rev"][0] == pd.Timestamp("2016-06-01")
        text = (out / "summary_leader.txt").read_text(encoding="utf-8")
        assert text.index("【警告(請先讀)】") < text.index("【績效：全期】")
        for s in ["存活者偏誤", "樣本內", "多重比較", "零股", "漲停", "【事先登錄的PASS檢查】", "【設計軸檢視】",
                  "通過的變體", "E_D｜X3｜K1｜大盤濾網", "【績效：2024-最新】", lb.REF_TWII, lb.REF_0050,
                  "不含股利", "執行時間", "【期末未平倉部位】", "E_CRASH 觸發日"]:
            assert s in text, s
        assert "✓" in text or "✗" in text
        v = pd.read_csv(out / "leader_variants.csv", encoding="utf-8-sig")
        assert len(v) == 38 and "PASS" in v.columns and "bootstrap正報酬比例" in v.columns
        assert (v["期末權益"] > 0).all()
        eq = pd.read_csv(out / "leader_equity.csv", encoding="utf-8-sig", index_col=0)
        assert eq.iloc[0, 0] == 300000
        w = pd.read_csv(out / "leader_windows.csv", encoding="utf-8-sig")
        assert {"年化報酬CAGR", "最大回撤(日MTM)", "最差日曆年", "正報酬年比例", "已平倉筆數", "勝率", "獲利因子PF",
                "平均持有交易日", "在場時間比例", "平均持股檔數", "扣除前5大獲利後損益", "bootstrap正報酬比例"} <= set(w.columns)
        tr = pd.read_csv(out / "leader_trades.csv", encoding="utf-8-sig")
        assert "期末未平倉" in tr.columns and tr["變體"].nunique() > 1
        assert (out / "leader_yearly.csv").exists()

    def test_universe_empty_fails_closed(self, monkeypatch, tmp_path, synth):
        _patch(monkeypatch, synth, uni_empty=True)
        out = tmp_path / "r"
        assert lb.main(["--output-dir", str(out), "--cache-dir", str(tmp_path / "c")]) == 1
        assert "fail closed" in (out / "summary_leader.txt").read_text(encoding="utf-8")

    def test_revenue_missing(self, monkeypatch, tmp_path, synth):
        _patch(monkeypatch, synth, rev_ok=False)
        out = tmp_path / "r"
        assert lb.main(["--output-dir", str(out), "--end", "2025-06-30", "--cache-dir", str(tmp_path / "c")]) == 0
        text = (out / "summary_leader.txt").read_text(encoding="utf-8")
        assert "月營收(MOPS)完全抓不到" in text
        v = pd.read_csv(out / "leader_variants.csv", encoding="utf-8-sig")
        assert (v.loc[v["進場"].isin(["E_B", "E_D"]), "訊號數"] == 0).all()


# ---------------------------------------------------------------------------
# 其他：不依賴scipy、workflow接線
# ---------------------------------------------------------------------------
def test_no_scipy_import():
    src = open(os.path.join(HERE, "leader_breakout.py"), encoding="utf-8").read()
    assert not re.search(r"^\s*(import|from)\s+scipy", src, re.M)
    code = "import sys, leader_breakout; print('scipy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "False", out.stderr


class TestWorkflow:
    def _wf(self):
        yaml = pytest.importorskip("yaml")
        with open(WORKFLOW, encoding="utf-8") as fh:
            text = fh.read()
        return text, yaml.safe_load(text)

    def test_parses_and_inputs(self):
        text, wf = self._wf()
        on = wf.get("on", wf.get(True))
        inputs = on["workflow_dispatch"]["inputs"]
        assert str(inputs["capital"]["default"]) == "200000"
        assert inputs["refresh_data"]["type"] == "boolean" and inputs["refresh_data"]["default"] is False
        job = next(iter(wf["jobs"].values()))
        assert job["timeout-minutes"] == 120
        runs = [s.get("run", "") or "" for s in job["steps"]]
        i_test = next(i for i, r in enumerate(runs) if "pytest" in r and "test_leader_breakout.py" in r)
        i_run = next(i for i, r in enumerate(runs) if "python leader_breakout.py" in r)
        assert i_test < i_run
        up = next(s for s in job["steps"] if "upload-artifact" in s.get("uses", ""))
        assert up["with"]["path"].rstrip("/") == "results_leader"
        caches = [s for s in job["steps"] if "actions/cache" in s.get("uses", "")]
        assert caches and all(s["with"]["path"] == "leader_cache" for s in caches)
        assert job["steps"][1]["with"]["python-version"] == "3.10"

    @pytest.mark.parametrize("capital,refresh,expected", [
        ("200000", "false", "--output-dir results_leader --cache-dir leader_cache --start 2018-01-01 --capital 200000"),
        ("500000", "true", "--output-dir results_leader --cache-dir leader_cache --start 2018-01-01 --capital 500000 --refresh"),
        ("", "false", "--output-dir results_leader --cache-dir leader_cache --start 2018-01-01 --capital 200000"),
    ])
    def test_run_script_maps_inputs(self, capital, refresh, expected):
        _, wf = self._wf()
        job = next(iter(wf["jobs"].values()))
        step = next(s for s in job["steps"] if "python leader_breakout.py" in s.get("run", ""))
        assert step["env"]["CAPITAL"] == "${{ github.event.inputs.capital }}"
        assert step["env"]["REFRESH"] == "${{ github.event.inputs.refresh_data }}"
        script = step["run"].replace("python leader_breakout.py", "echo ARGS=")
        env = {"PATH": os.environ.get("PATH", ""), "CAPITAL": capital, "REFRESH": refresh}
        proc = subprocess.run(["bash", "-e", "-c", script], capture_output=True, text=True, env=env, timeout=30)
        assert proc.returncode == 0, proc.stderr
        line = [l for l in proc.stdout.splitlines() if l.startswith("ARGS=")][0]
        assert line[len("ARGS="):].strip() == expected

    def test_restore_skipped_on_refresh(self):
        _, wf = self._wf()
        job = next(iter(wf["jobs"].values()))
        restore = next(s for s in job["steps"] if "actions/cache/restore" in s.get("uses", ""))
        assert "refresh_data" in restore["if"]
