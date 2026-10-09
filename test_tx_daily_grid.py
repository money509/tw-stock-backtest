"""
test_tx_daily_grid.py —— 網格模式(tx_daily_engine網格擴充 + tx_daily_grid + compare_tx_daily --grid)
離線測試：12個單一訊號(手工序列、穿越方向、不偷看未來、Panama平移不變)、混搭三天視窗、
出場網格(固定停利成交/跳空/同日停損停利/移動停損/停損倍數)、變體數1872、選擇只看訓練期、
高原鄰格(網格邊界)、Spearman(不用scipy)對手算、家族/出場彙總、8變體模式輸出不變、
合成23年日線的執行時間、scipy被擋+不連網的端對端。
"""
import itertools
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
from mean_reversion_engine import compute_rsi
from momentum_breakout_engine import compute_macd
from test_tx_daily_data import make_index, make_futures_daily_truth, daily_to_minutes, make_yf_frame
from test_tx_daily_engine import mk

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_SIG_COLS = [f"sig_{r['rule']}" for r in eng.grid_rule_list()]


def manual_cross(a, b):
    a = list(a)
    b = list(b) if hasattr(b, "__len__") else [b] * len(a)
    out = [0] * len(a)
    for i in range(1, len(a)):
        if any(pd.isna(x) for x in (a[i], b[i], a[i - 1], b[i - 1])):
            continue
        if a[i] > b[i] and a[i - 1] <= b[i - 1]:
            out[i] = 1
        elif a[i] < b[i] and a[i - 1] >= b[i - 1]:
            out[i] = -1
    return out


def updown(n_flat=70, down=30, up=60, base=100.0):
    return ([base] * n_flat + [base - i for i in range(1, down)]
            + [base - down + 1 + 3 * i for i in range(1, up)])


# ---------------------------------------------------------------------------
class TestSingleSignals:
    def test_registry(self):
        assert eng.GRID_SIGNAL_NAMES == ["MA_5_20", "MA_20_60", "MACD", "MACD_ZERO", "BODY", "RSI50",
                                         "DONCHIAN20", "DONCHIAN55", "BB_BREAK", "KDJ_CROSS",
                                         "SQZ_RELEASE", "SQZ_KDJ"]

    def test_ma_5_20(self):
        df = mk(updown())
        ind = eng.compute_grid_indicators(df)
        exp = manual_cross(df["Close"].rolling(5).mean(), df["Close"].rolling(20).mean())
        assert 1 in exp and -1 in exp
        assert list(ind["sig_MA_5_20"]) == exp

    def test_existing_aliases(self):
        df = make_index("2005-01-01", "2010-12-31", seed=3)
        ind = eng.compute_grid_indicators(df)
        assert list(ind["sig_MA_20_60"]) == list(ind["sig_MA_CROSS"])
        assert list(ind["sig_DONCHIAN20"]) == list(ind["sig_DONCHIAN"])
        base = eng.compute_daily_indicators(df)
        for s in ("SQZ_RELEASE", "SQZ_KDJ", "DONCHIAN", "MA_CROSS"):
            assert list(ind[f"sig_{s}"]) == list(base[f"sig_{s}"]), s

    def test_macd_line_vs_signal(self):
        df = mk(updown() + [200.0 - 2 * i for i in range(40)])
        ind = eng.compute_grid_indicators(df)
        m, s, _ = compute_macd(df["Close"], 12, 26, 9)
        exp = manual_cross(m, s)
        exp[:eng.MACD_WARMUP] = [0] * eng.MACD_WARMUP
        assert 1 in exp and -1 in exp
        assert list(ind["sig_MACD"]) == exp

    def test_macd_zero(self):
        df = mk(updown() + [200.0 - 2 * i for i in range(60)])
        ind = eng.compute_grid_indicators(df)
        m, _, _ = compute_macd(df["Close"], 12, 26, 9)
        exp = manual_cross(m, 0.0)
        exp[:eng.MACD_WARMUP] = [0] * eng.MACD_WARMUP
        assert 1 in exp and -1 in exp
        assert list(ind["sig_MACD_ZERO"]) == exp

    def test_histogram_zero_cross_is_identical_to_macd_signal_cross(self):
        """改名理由：柱狀圖穿越0 跟 MACD線穿越訊號線 每天都完全一樣。"""
        df = make_index("2005-01-01", "2012-12-31", seed=8)
        m, s, h = compute_macd(df["Close"], 12, 26, 9)
        assert eng.cross_signal(h, 0.0).tolist() == eng.cross_signal(m, s).tolist()

    def test_macd_warmup_masked(self):
        df = make_index("2005-01-01", "2006-12-31", seed=1)
        ind = eng.compute_grid_indicators(df)
        assert (ind["sig_MACD"].iloc[:eng.MACD_WARMUP] == 0).all()
        assert (ind["sig_MACD_ZERO"].iloc[:eng.MACD_WARMUP] == 0).all()

    def test_body(self):
        n = 30
        o = [100.0] * n
        c = [100.5] * n
        h = [101.0] * n
        l = [99.5] * n          # 平常全距1.5
        # 第20天：大紅K收最高 → 多
        o[20], h[20], l[20], c[20] = 100.0, 110.0, 99.8, 109.8
        # 第22天：大黑K收最低 → 空
        o[22], h[22], l[22], c[22] = 110.0, 110.2, 100.0, 100.2
        # 第24天：紅K實體0.6但收盤不在上面25%(上影線太長) → 0
        o[24], h[24], l[24], c[24] = 100.0, 110.0, 99.0, 106.6
        # 第26天：實體比例夠、收在頂，但全距 < 1×ATR → 0
        o[26], h[26], l[26], c[26] = 100.0, 100.5, 99.98, 100.48
        df = mk(c, opens=o, highs=h, lows=l)
        ind = eng.compute_grid_indicators(df)
        sig = ind["sig_BODY"]
        assert sig.iloc[20] == 1 and sig.iloc[22] == -1
        assert sig.iloc[24] == 0 and sig.iloc[26] == 0
        # 手算第24天：實體6.6/全距11=0.6 >= 0.6，但收盤106.6 < 110−0.25×11=107.25
        assert (106.6 - 100.0) / 11.0 == pytest.approx(0.6)
        assert ind["ATR"].iloc[26] > 0.52  # 全距0.52 < ATR
        assert set(np.unique(sig)) <= {-1, 0, 1}

    def test_rsi50(self):
        df = mk(updown())
        ind = eng.compute_grid_indicators(df)
        exp = manual_cross(compute_rsi(df["Close"], 14), 50.0)
        exp[:eng.RSI_WARMUP] = [0] * eng.RSI_WARMUP
        assert 1 in exp and -1 in exp
        assert list(ind["sig_RSI50"]) == exp

    def test_donchian55(self):
        closes = [100.0] * 30 + [130.0] + [100.0] * 30 + [120.0] + [100.0] * 5 + [140.0]
        df = mk(closes)
        ind = eng.compute_grid_indicators(df)
        # 第61天收120：突破前20天高(101)，但前55天最高是131(第30天) → 55日不觸發
        assert ind["sig_DONCHIAN20"].iloc[61] == 1 and ind["sig_DONCHIAN55"].iloc[61] == 0
        assert ind["sig_DONCHIAN55"].iloc[67] == 1  # 140 > 前55天最高
        assert ind["sig_DONCHIAN55"].iloc[:55].eq(0).all()  # 暖身
        df2 = mk([100.0] * 60 + [70.0])
        assert eng.compute_grid_indicators(df2)["sig_DONCHIAN55"].iloc[60] == -1

    def test_bb_break(self):
        wiggle = [100.0, 100.5] * 20
        closes = wiggle + [110.0, 111.0] + wiggle[:30] + [85.0]
        df = mk(closes)
        ind = eng.compute_grid_indicators(df)
        from mean_reversion_engine import compute_bollinger
        _, u, lo = compute_bollinger(df["Close"], 20, 2.0)
        up = [1 if x == 1 else 0 for x in manual_cross(df["Close"], u)]
        dn = [-1 if x == -1 else 0 for x in manual_cross(df["Close"], lo)]
        exp = [a + b for a, b in zip(up, dn)]
        assert list(ind["sig_BB_BREAK"]) == exp
        assert ind["sig_BB_BREAK"].iloc[40] == 1      # 突破那天
        assert ind["sig_BB_BREAK"].iloc[41] == 0      # 隔天還在上軌外，但不是「穿越」
        assert ind["sig_BB_BREAK"].iloc[len(closes) - 1] == -1

    def test_kdj_cross_matches_manual(self):
        df = make_index("2005-01-01", "2012-12-31", seed=5)
        ind = eng.compute_grid_indicators(df)
        c, h, l = df["Close"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy()
        k = d = 50.0
        ks, ds_ = [], []
        for i in range(len(df)):
            lo, hi = l[max(0, i - 8):i + 1].min(), h[max(0, i - 8):i + 1].max()
            rsv = (c[i] - lo) / (hi - lo) * 100 if hi > lo else 50.0
            k = 2 / 3 * k + rsv / 3
            d = 2 / 3 * d + k / 3
            ks.append(k)
            ds_.append(d)
        np.testing.assert_allclose(ind["KDJ_K9"], ks)
        np.testing.assert_allclose(ind["KDJ_D9"], ds_)
        exp = [0] * len(df)
        for i in range(eng.KDJ_WARMUP, len(df)):
            if ks[i] > ds_[i] and ks[i - 1] <= ds_[i - 1] and ks[i] < 30:
                exp[i] = 1
            elif ks[i] < ds_[i] and ks[i - 1] >= ds_[i - 1] and ks[i] > 70:
                exp[i] = -1
        assert 1 in exp and -1 in exp
        assert list(ind["sig_KDJ_CROSS"]) == exp

    def test_no_lookahead_all_grid_signals(self):
        df = make_index("2008-01-01", "2012-12-31", seed=13)
        full = eng.compute_grid_indicators(df)
        for cut in (300, 777):
            part = eng.compute_grid_indicators(df.iloc[:cut])
            for col in ALL_SIG_COLS + ["gate_long", "gate_short"]:
                assert list(part[col]) == list(full[col].iloc[:cut]), (cut, col)

    def test_shift_invariance_panama(self):
        df = make_index("2005-01-01", "2010-12-31", seed=2)
        shifted = df.copy()
        for c in ("Open", "High", "Low", "Close"):
            shifted[c] = shifted[c] - 20000.0
        a, b = eng.compute_grid_indicators(df), eng.compute_grid_indicators(shifted)
        for col in ALL_SIG_COLS:
            assert list(a[col]) == list(b[col]), col

    def test_every_single_signal_fires_both_ways_on_random_data(self):
        df = make_index("2001-01-01", "2012-12-31", seed=21)
        ind = eng.compute_grid_indicators(df)
        for s in eng.GRID_SIGNAL_NAMES:
            v = ind[f"sig_{s}"]
            assert (v == 1).any(), s
            if s in eng.LONG_ONLY_SIGNALS:
                assert not (v == -1).any(), s
            else:
                assert (v == -1).any(), s


# ---------------------------------------------------------------------------
class TestPairComposition:
    def test_window_logic(self):
        a = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        b = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        # b在第3天、a在第5天：差2天(在[t−2, t]內) → 第5天成立
        assert eng.compose_pair(a, b).tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        assert eng.compose_pair(b, a).tolist() == eng.compose_pair(a, b).tolist()  # 對稱
        b3 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 差3天 → 不成立
        assert eng.compose_pair(a, b3).tolist() == [0] * 12
        same = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        assert eng.compose_pair(a, same).tolist()[5] == 1  # 同一天
        # 先a後b也一樣(第t天是b、a在t−1)
        a2 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        b2 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        assert eng.compose_pair(a2, b2).tolist()[5] == 1 and eng.compose_pair(a2, b2).tolist()[4] == 0

    def test_direction_must_match(self):
        a = [0, 0, 1, 0, 0, -1, 0]
        b = [0, 0, -1, 0, 0, 0, -1]
        # 第2天a多b空 → 不成立；第6天b空、a在第5天空 → 空
        assert eng.compose_pair(a, b).tolist() == [0, 0, 0, 0, 0, 0, -1]

    def test_conflict_gives_zero(self):
        a = [1, 0, -1]
        b = [0, 1, -1]
        # 第2天：做空成立(兩個都空)；做多也成立(b在第1天多…但第2天沒有人發多) → 只有空
        assert eng.compose_pair(a, b).tolist() == [0, 1, -1]
        a = [1, -1]
        b = [-1, 1]
        # 第1天：b多、a在第0天多 → 多；a空、b在第0天空 → 空 → 衝突 → 0
        assert eng.compose_pair(a, b).tolist() == [0, 0]

    def test_sqz_kdj_pairs_long_only_and_columns(self):
        df = make_index("2001-01-01", "2012-12-31", seed=4)
        ind = eng.compute_grid_indicators(df)
        for r in eng.grid_rule_list():
            if len(r["components"]) == 2:
                a, b = r["components"]
                assert list(ind[f"sig_{r['rule']}"]) == eng.compose_pair(ind[f"sig_{a}"], ind[f"sig_{b}"]).tolist()
                if "SQZ_KDJ" in r["components"]:
                    assert not (ind[f"sig_{r['rule']}"] == -1).any(), r["rule"]

    def test_rules_and_variant_count(self):
        rules = eng.grid_rule_list()
        assert len(rules) == 78
        pairs = [r for r in rules if len(r["components"]) == 2]
        assert len(pairs) == 66
        assert [r["components"] for r in pairs] == list(itertools.combinations(eng.GRID_SIGNAL_NAMES, 2))
        vs = eng.grid_variant_list()
        assert len(vs) == 1872 == (12 + 66) * 2 * 12
        assert len({v["name"] for v in vs}) == 1872
        assert vs[0]["name"] == "MA_5_20｜停損1.0/停利2.0"
        assert vs[11]["name"] == "MA_5_20｜停損2.0/移動3.0"
        assert vs[12]["name"] == "MA_5_20+動能門檻｜停損1.0/停利2.0"


# ---------------------------------------------------------------------------
def arrs_for(df, sig, atr=10.0, rule="X"):
    ind = pd.DataFrame(index=df.index)
    ind["ATR"] = atr
    full = np.zeros(len(df), int)
    for i, v in sig.items():
        full[i] = v
    ind[f"sig_{rule}"] = full
    ind["gate_long"] = True
    ind["gate_short"] = True
    return eng.prepare_grid_arrays(df, ind), full


def grun(df, sig, stop=2.0, kind="trail", m=3.0, atr=10.0, **kw):
    arrs, full = arrs_for(df, sig, atr)
    kw.setdefault("commission_per_side", 0.0)
    kw.setdefault("tax_rate", 0.0)
    return eng.run_tx_daily_grid_backtest(arrs, full, full, stop, kind, m, **kw)


class TestExitGrid:
    def test_configs(self):
        cfg = eng.grid_exit_configs()
        assert len(cfg) == 12
        assert [(c["stop_atr"], c["exit_kind"], c["exit_mult"]) for c in cfg[:4]] == [
            (1.0, "target", 2.0), (1.0, "target", 3.0), (1.0, "target", 4.0), (1.0, "trail", 3.0)]
        assert [(c["stop_idx"], c["exit_idx"]) for c in cfg] == [(s, e) for s in range(3) for e in range(4)]

    def test_fixed_target_intraday_fill_no_slippage(self):
        closes = [100.0] * 12
        highs = [101.0] * 12
        highs[6] = 125.0
        df = mk(closes, highs=highs)
        tr = grun(df, {2: 1}, stop=1.0, kind="target", m=2.0)
        # 進場101，停利 = 101 + 2×10 = 121，第6天高125觸及 → 121成交(限價，不扣滑價)
        t = tr[0]
        assert t["exit_reason"] == "停利" and t["exit_price"] == 121.0 and t["target"] == 121.0
        assert t["initial_stop"] == pytest.approx(91.0)
        assert t["exit_date"] == df.index[6] and t["points"] == pytest.approx(20.0)

    def test_target_gap_through_fills_at_open(self):
        closes = [100.0] * 12
        opens = [100.0] * 12
        opens[6] = 140.0
        df = mk(closes, opens=opens)
        t = grun(df, {2: 1}, stop=1.0, kind="target", m=3.0)[0]
        assert t["exit_reason"] == "停利" and t["exit_price"] == 140.0  # 開盤價、不扣滑價

    def test_short_target(self):
        lows = [99.0] * 12
        lows[5] = 70.0
        df = mk([100.0] * 12, lows=lows)
        t = grun(df, {2: -1}, stop=1.5, kind="target", m=4.0)[0]
        # 進場99，停利 = 99 − 40 = 59 沒碰到；停損 = 99 + 15 = 114
        assert t["exit_reason"] == "期末平倉" and t["initial_stop"] == pytest.approx(114.0)
        t = grun(df, {2: -1}, stop=1.5, kind="target", m=2.0)[0]
        assert t["exit_reason"] == "停利" and t["exit_price"] == pytest.approx(79.0)

    def test_same_day_stop_and_target_counts_as_stop(self):
        highs = [101.0] * 12
        lows = [99.0] * 12
        highs[5], lows[5] = 130.0, 80.0
        df = mk([100.0] * 12, highs=highs, lows=lows)
        t = grun(df, {2: 1}, stop=1.0, kind="target", m=2.0)[0]
        assert t["exit_reason"] == "停損" and t["exit_price"] == pytest.approx(90.0)  # 91 − 1滑價

    def test_stop_gap_beats_target(self):
        opens = [100.0] * 12
        opens[5] = 80.0
        highs = [101.0] * 12
        highs[5] = 130.0
        df = mk([100.0] * 12, opens=opens, highs=highs)
        t = grun(df, {2: 1}, stop=1.0, kind="target", m=2.0)[0]
        assert t["exit_reason"] == "停損" and t["exit_price"] == 79.0

    def test_stop_multipliers(self):
        df = mk([100.0] * 12)
        for s, exp in ((1.0, 91.0), (1.5, 86.0), (2.0, 81.0)):
            t = grun(df, {2: 1}, stop=s, kind="target", m=3.0)[0]
            assert t["initial_stop"] == pytest.approx(exp)
        lows = [99.0] * 12
        lows[6] = 88.0
        df2 = mk([100.0] * 12, lows=lows)
        assert grun(df2, {2: 1}, stop=1.0, kind="target", m=3.0)[0]["exit_reason"] == "停損"
        assert grun(df2, {2: 1}, stop=1.5, kind="target", m=3.0)[0]["exit_reason"] == "期末平倉"

    def test_fixed_target_does_not_trail(self):
        closes = [100.0] * 3 + [110.0, 118.0, 119.0, 112.0, 95.0, 95.0, 95.0]
        lows = [c - 1 for c in closes]
        df = mk(closes, lows=lows)
        # 停利4×10=40 → 141 從沒碰到；固定停損81不移動 → 一路抱到期末(移動停損版本會在回落時出場)
        t = grun(df, {2: 1}, stop=2.0, kind="target", m=4.0)[0]
        assert t["exit_reason"] == "期末平倉"
        t2 = grun(df, {2: 1}, stop=2.0, kind="trail", m=1.0)[0]
        assert t2["exit_reason"] == "移動停損"

    def test_trailing_matches_basic_engine(self):
        df = make_index("2001-01-01", "2012-12-31", seed=6)
        df["Offset"] = -1000.0
        df["RollDay"] = False
        df.iloc[::21, df.columns.get_loc("RollDay")] = True
        ind = eng.compute_grid_indicators(df)
        arrs = eng.prepare_grid_arrays(df, ind)
        for old, new in (("DONCHIAN", "DONCHIAN20"), ("MA_CROSS", "MA_20_60"), ("SQZ_KDJ", "SQZ_KDJ"),
                         ("SQZ_RELEASE", "SQZ_RELEASE")):
            for gate in (False, True):
                a = eng.run_tx_daily_backtest(df, ind, old, gate, start="2003-01-01", end="2010-12-31")
                es, rs = eng.grid_entry_signals(arrs, new, gate)
                b = eng.run_tx_daily_grid_backtest(arrs, es, rs, 2.0, "trail", 3.0,
                                                   start="2003-01-01", end="2010-12-31")
                assert len(a) == len(b), (old, gate)
                for x, y in zip(a, b):
                    for k in x:
                        if k == "variant":
                            continue
                        assert x[k] == y[k] or (isinstance(x[k], float) and x[k] == pytest.approx(y[k])), (old, k)

    def test_opposite_raw_signal_exit_and_gate(self):
        df = mk([100.0] * 15)
        arrs, full = arrs_for(df, {2: 1, 6: -1})
        arrs["gate_short"] = np.zeros(15, bool)
        es, rs = eng.grid_entry_signals(arrs, "X", True)
        assert es.tolist()[6] == 0 and rs.tolist()[6] == -1
        tr = eng.run_tx_daily_grid_backtest(arrs, es, rs, 1.0, "target", 4.0, commission_per_side=0, tax_rate=0)
        assert len(tr) == 1 and tr[0]["exit_reason"] == "反向訊號" and tr[0]["exit_date"] == df.index[7]

    def test_last_day_signal_and_period_bounds(self):
        df = mk([100.0] * 30)
        assert grun(df, {20: 1}, end=df.index[20]) == []
        tr = grun(df, {2: 1, 12: 1}, start=df.index[10], end=df.index[20])
        assert len(tr) == 1 and tr[0]["entry_date"] == df.index[13] and tr[0]["exit_reason"] == "期末平倉"


# ---------------------------------------------------------------------------
def fake_stats(variants, train=None, test=None, default=(1.0, 50, 0.0)):
    train, test = train or {}, test or {}
    stats = {}
    for v in variants:
        for p, src in (("train", train), ("test", test), ("recent", {})):
            pf, n, pnl = src.get(v["name"], default)
            stats[(v["name"], p)] = {"profit_factor": pf, "trade_count": n, "total_pnl_ntd": pnl}
    return stats


class TestSelectionAndPlateau:
    def test_train_only(self):
        vs = eng.grid_variant_list()
        a, b = vs[100]["name"], vs[1500]["name"]
        stats = fake_stats(vs, train={a: (2.0, 60, 1000.0), b: (1.9, 60, 1e6)})
        assert grid.select_grid_variant(vs, stats) == a
        for v in vs:
            stats[(v["name"], "test")] = {"profit_factor": 0.1, "trade_count": 50, "total_pnl_ntd": -1e9}
        stats[(b, "test")] = {"profit_factor": 99.0, "trade_count": 500, "total_pnl_ntd": 1e12}
        assert grid.select_grid_variant(vs, stats) == a
        assert grid.train_ranking(vs, stats)[0]["name"] == a
        assert grid.train_ranking(vs, stats)[1]["name"] == b

    def test_min_trades_and_ties(self):
        vs = eng.grid_variant_list()
        a, b, c = vs[5]["name"], vs[7]["name"], vs[9]["name"]
        stats = fake_stats(vs, train={a: (5.0, 39, 1e6), b: (1.5, 40, 10.0), c: (1.5, 40, 10.0)})
        assert grid.select_grid_variant(vs, stats) == b  # a不夠40筆；b、c同PF同損益 → 登記順序
        stats = fake_stats(vs, train={b: (1.5, 40, 10.0), c: (1.5, 40, 20.0)})
        assert grid.select_grid_variant(vs, stats) == c
        stats = fake_stats(vs, default=(3.0, 10, 1.0))
        assert grid.select_grid_variant(vs, stats) is None

    def test_top_n_order(self):
        vs = eng.grid_variant_list()
        train = {v["name"]: (1.0 + k / 10000.0, 50, 0.0) for k, v in enumerate(vs)}
        stats = fake_stats(vs, train=train)
        top = grid.train_ranking(vs, stats)[:20]
        assert [v["name"] for v in top] == [v["name"] for v in vs[::-1][:20]]

    @pytest.mark.parametrize("cell,expected", [
        ((0, 0), [(0, 1), (1, 0), (1, 1)]),
        ((2, 3), [(1, 2), (1, 3), (2, 2)]),
        ((0, 3), [(0, 2), (1, 2), (1, 3)]),
        ((2, 0), [(1, 0), (1, 1), (2, 1)]),
        ((0, 1), [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]),
        ((1, 0), [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)]),
        ((1, 3), [(0, 2), (0, 3), (1, 2), (2, 2), (2, 3)]),
        ((1, 1), [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]),
    ])
    def test_neighbor_cells(self, cell, expected):
        assert grid.exit_neighbor_cells(*cell) == expected

    def test_neighbor_mean_same_rule_gate_and_cap(self):
        vs = eng.grid_variant_list()
        idx = grid._cell_index(vs)
        sel = idx[("BODY&RSI50", True, 0, 0)]
        nb = grid.neighbor_names(sel, idx)
        assert nb == ["BODY&RSI50+動能門檻｜停損1.0/停利3.0", "BODY&RSI50+動能門檻｜停損1.5/停利2.0",
                      "BODY&RSI50+動能門檻｜停損1.5/停利3.0"]
        stats = fake_stats(vs, test={nb[0]: (float("inf"), 3, 1.0), nb[1]: (0.5, 50, 0.0),
                                     nb[2]: (0.4, 50, 0.0), sel["name"]: (50.0, 50, 0.0)})
        m, names = grid.neighbor_mean_pf(sel, idx, stats, "test")
        assert names == nb
        assert m == pytest.approx((10.0 + 0.5 + 0.4) / 3)  # ∞截在10、不含自己

    def test_grid_verdict_requires_plateau(self):
        st = {"profit_factor": 1.5, "bootstrap_pct_positive": 95.0, "pnl_excl_top3_ntd": 10.0}
        ok, checks = grid.grid_verdict(st, 1.2)
        assert ok and len(checks) == 4
        ok, _ = grid.grid_verdict(st, 0.99)
        assert not ok
        ok, _ = grid.grid_verdict(st, float("nan"))
        assert not ok
        ok, _ = grid.grid_verdict(dict(st, profit_factor=0.9), 2.0)
        assert not ok


class TestSpearman:
    def test_hand_calc(self):
        # y排名：[1, 2, 3.5, 5, 3.5]；x排名1..5 → Σd·d = 8，Σx² = 10，Σy² = 9.5 → 8/√95
        assert grid.rankdata_avg([5, 6, 7, 8, 7]).tolist() == [1.0, 2.0, 3.5, 5.0, 3.5]
        assert grid.spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(8 / np.sqrt(95))
        assert grid.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
        assert grid.spearman([1, 2, 3], [1, 1, 1]) != grid.spearman([1, 2, 3], [1, 1, 1])  # NaN
        # ∞(只賺不賠)排在最後一名，照常計算
        assert grid.rankdata_avg([1.0, float("inf"), 0.5]).tolist() == [2.0, 3.0, 1.0]

    def test_matches_scipy_if_available(self):
        stats_mod = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        x = rng.integers(0, 20, 200).astype(float)
        y = x + rng.integers(0, 15, 200)
        assert grid.spearman(x, y) == pytest.approx(stats_mod.spearmanr(x, y)[0])
        np.testing.assert_allclose(grid.rankdata_avg(y), stats_mod.rankdata(y))


class TestAggregation:
    def _stats(self):
        vs = eng.grid_variant_list()
        good = {v["name"]: (1.5, 50, 1.0) for v in vs if "BODY" in v["components"]}
        stats = fake_stats(vs, train=good, test=good, default=(0.5, 50, -1.0))
        return vs, stats

    def test_family(self):
        vs, stats = self._stats()
        fam = {r["訊號"]: r for r in grid.family_rows(vs, stats)}
        assert list(fam) == eng.GRID_SIGNAL_NAMES
        assert fam["BODY"]["變體數(全部)"] == 288 and fam["BODY"]["兩段PF都>1比例(%)"] == 100.0
        assert fam["BODY"]["單獨使用：兩段PF>1比例(%)"] == 100.0
        # 其他訊號：288個裡只有跟BODY混搭的24個(1組 × 2門檻 × 12出場)
        assert fam["RSI50"]["兩段PF都>1的數量"] == 24
        assert fam["RSI50"]["兩段PF都>1比例(%)"] == round(24 / 288 * 100, 1)
        assert fam["RSI50"]["單獨使用：兩段PF>1比例(%)"] == 0.0
        assert fam["RSI50"]["混搭：兩段PF>1比例(%)"] == round(24 / 264 * 100, 1)

    def test_family_uses_only_eligible(self):
        vs, stats = self._stats()
        body_single = [v for v in vs if v["rule"] == "BODY"]
        for v in body_single:
            stats[(v["name"], "train")]["trade_count"] = 10
        fam = {r["訊號"]: r for r in grid.family_rows(vs, stats)}
        assert fam["BODY"]["合格變體數(訓練>=40筆)"] == 288 - 24
        assert np.isnan(fam["BODY"]["單獨使用：兩段PF>1比例(%)"])

    def test_exits_and_gates(self):
        vs, stats = self._stats()
        ex = grid.exit_rows(vs, stats)
        assert len(ex) == 12
        for r in ex:
            assert r["變體數(全部)"] == 156
            assert r["兩段PF都>1的數量"] == 24  # 12條含BODY的規則 × 2門檻
        g = grid.gate_rows(vs, stats)
        assert [r["兩段PF都>1的數量"] for r in g] == [144, 144]

    def test_diagnostics(self):
        vs, stats = self._stats()
        d = grid.overfit_diagnostics(vs, stats)
        assert d["n_variants"] == 1872 and d["n_eligible"] == 1872
        assert d["n_top"] == 20 and d["top_test_gt1"] == 20  # 前20名(全是含BODY的)測試PF都>1
        assert d["top_median_test_pf"] == 1.5 and d["all_median_test_pf"] == 0.5
        assert d["spearman"] == pytest.approx(1.0)
        assert d["test_pct_gt1"] == pytest.approx(288 / 1872 * 100)


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


# 8變體模式在HEAD(d5e3097，加網格之前)用同一份合成資料跑出來的(交易數, 總損益) —— 證明輸出沒變
BASIC_FINGERPRINT = {
    'DONCHIAN|train': (117, 91586), 'DONCHIAN+動能門檻|train': (58, 105439), 'MA_CROSS|train': (55, -98490),
    'MA_CROSS+動能門檻|train': (38, -75780), 'SQZ_KDJ|train': (11, -6592), 'SQZ_KDJ+動能門檻|train': (0, 0),
    'SQZ_RELEASE|train': (24, -28557), 'SQZ_RELEASE+動能門檻|train': (2, 7366), '買進持有(參考)|train': (1, 43602),
    'DONCHIAN|test': (106, 138714), 'DONCHIAN+動能門檻|test': (53, 102414), 'MA_CROSS|test': (48, 344263),
    'MA_CROSS+動能門檻|test': (21, 244902), 'SQZ_KDJ|test': (8, -145725), 'SQZ_KDJ+動能門檻|test': (0, 0),
    'SQZ_RELEASE|test': (28, 108033), 'SQZ_RELEASE+動能門檻|test': (6, -27541), '買進持有(參考)|test': (1, 527305),
    'DONCHIAN|recent': (20, 960875), 'DONCHIAN+動能門檻|recent': (17, 329542), 'MA_CROSS|recent': (10, 492689),
    'MA_CROSS+動能門檻|recent': (6, 151982), 'SQZ_KDJ|recent': (2, -36010), 'SQZ_KDJ+動能門檻|recent': (0, 0),
    'SQZ_RELEASE|recent': (8, 403461), 'SQZ_RELEASE+動能門檻|recent': (3, 85556), '買進持有(參考)|recent': (1, 315429),
}


class TestBasicModeUnchanged:
    def test_fingerprint_and_grid_consistency(self, synth_market):
        index_df, bars = synth_market
        ds = tdd.build_tx_daily_dataset(load_minute_fn=lambda: (bars, {"periods": {}}),
                                        load_index_fn=lambda: (index_df, {"reason": "ok"}))
        res = ctd.run_all(ds, eng.variant_list(), n_boot=10)
        got = {f"{n}|{p}": (s["trade_count"], s["total_pnl_ntd"]) for (n, p), s in res["stats"].items()}
        assert set(got) == set(BASIC_FINGERPRINT)
        for k, (n, pnl) in BASIC_FINGERPRINT.items():
            assert got[k][0] == n, k
            assert got[k][1] == pytest.approx(pnl, abs=1.0), k
        # 網格裡「停損2.0/移動3.0、同訊號同門檻」那一格 = 8變體模式的結果
        vs = eng.grid_variant_list()
        sub = [v for v in vs if v["rule"] in ("DONCHIAN20", "MA_20_60", "SQZ_KDJ", "SQZ_RELEASE")
               and v["exit_id"] == "S2.0_TR3.0"]
        g = grid.run_grid(ds, sub)
        old = {"DONCHIAN20": "DONCHIAN", "MA_20_60": "MA_CROSS", "SQZ_KDJ": "SQZ_KDJ", "SQZ_RELEASE": "SQZ_RELEASE"}
        for v in sub:
            oname = old[v["rule"]] + ("+動能門檻" if v["gate"] else "")
            for p in grid.PERIOD_KEYS:
                assert g["stats"][(v["name"], p)]["trade_count"] == res["stats"][(oname, p)]["trade_count"]
                assert g["stats"][(v["name"], p)]["total_pnl_ntd"] == pytest.approx(
                    res["stats"][(oname, p)]["total_pnl_ntd"])

    def test_basic_main_writes_no_grid_files(self, synth_market, tmp_path, monkeypatch):
        index_df, bars = synth_market
        _patch_sources(monkeypatch, index_df, bars)
        out = tmp_path / "res"
        assert ctd.main(["--results-dir", str(out), "--n-bootstrap", "20",
                         "--index-cache-dir", str(tmp_path / "ic")]) == 0
        files = set(os.listdir(out))
        assert "summary.txt" in files and not any(f.startswith("tx_grid") or f == "summary_grid.txt" for f in files)

    def test_grid_with_variants_rejected(self, tmp_path):
        assert ctd.main(["--grid", "--results-dir", str(tmp_path), "--variants", "DONCHIAN"]) == 2


class TestGridEndToEnd:
    def test_futures_mode(self, synth_market, tmp_path, monkeypatch, capsys):
        index_df, bars = synth_market
        _patch_sources(monkeypatch, index_df, bars)
        out = tmp_path / "res"
        rc = ctd.main(["--grid", "--results-dir", str(out), "--n-bootstrap", "100",
                       "--index-cache-dir", str(tmp_path / "ic")])
        assert rc == 0
        for f in ("summary_grid.txt", "tx_grid_all.csv", "tx_grid_top20.csv", "tx_grid_family.csv",
                  "tx_grid_exits.csv", "tx_grid_selected_trades.csv", "data_parse_report.txt",
                  "data_validation.json", "tx_daily_bars_sample.csv", "tx_roll_adjustments.csv"):
            assert (out / f).exists(), f
        assert not (out / "summary.txt").exists()
        s = (out / "summary_grid.txt").read_text(encoding="utf-8")
        assert s.splitlines()[0].startswith("價格來源：台指期")
        assert s.index("預先登記的規則") < s.index("== 選擇與判定") < s.index("過度擬合診斷(請先看")
        assert s.index("過度擬合診斷(請先看") < s.index("被選中變體完整明細")
        for key in ("1872", "Spearman", "家族檢視", "出場檢視", "不要從裡面挑", "判定：", "一定偏高",
                    "高原", "逐年", "出場原因", "訓練前20名", "執行時間"):
            assert key in s, key
        printed = capsys.readouterr().out
        assert printed.index("預先登記的規則") < printed.index("過度擬合診斷")
        allv = pd.read_csv(out / "tx_grid_all.csv", encoding="utf-8-sig")
        assert len(allv) == 1872 and {"變體", "訓練PF", "測試PF", "近期PF", "動能門檻", "相鄰出場格測試PF平均"} <= set(allv.columns)
        top = pd.read_csv(out / "tx_grid_top20.csv", encoding="utf-8-sig")
        assert len(top) == 20 and (top["訓練交易數"] >= 40).all()
        assert list(top["訓練PF"]) == sorted(top["訓練PF"], reverse=True)
        fam = pd.read_csv(out / "tx_grid_family.csv", encoding="utf-8-sig")
        assert list(fam["訊號"]) == eng.GRID_SIGNAL_NAMES and (fam["變體數(全部)"] == 288).all()
        ex = pd.read_csv(out / "tx_grid_exits.csv", encoding="utf-8-sig")
        assert len(ex) == 12 and (ex["變體數(全部)"] == 156).all()
        tr = pd.read_csv(out / "tx_grid_selected_trades.csv", encoding="utf-8-sig")
        assert len(tr) > 0 and tr["變體"].nunique() == 1 and tr["變體"].iloc[0] == top["變體"].iloc[0]
        assert (pd.to_datetime(tr[tr["期間"] == "train"]["出場日"]) <= "2012-12-31").all()
        assert (pd.to_datetime(tr[tr["期間"] == "test"]["進場日"]) >= "2013-01-01").all()

    def test_fallback_and_no_data(self, synth_market, tmp_path, monkeypatch):
        index_df, bars = synth_market
        bad = bars.copy()
        bad[["Open", "High", "Low", "Close"]] *= 1.5
        _patch_sources(monkeypatch, index_df, bad)
        out = tmp_path / "res"
        assert ctd.main(["--grid", "--results-dir", str(out), "--n-bootstrap", "20",
                         "--index-cache-dir", str(tmp_path / "ic")]) == 0
        s = (out / "summary_grid.txt").read_text(encoding="utf-8")
        assert s.splitlines()[0].startswith("價格來源：加權指數(期貨資料驗證失敗：原因")
        monkeypatch.setattr(tdd, "_yf_download", lambda *a: pd.DataFrame())
        monkeypatch.setattr(tdd.time, "sleep", lambda s: None)
        out2 = tmp_path / "res2"
        assert ctd.main(["--grid", "--results-dir", str(out2), "--index-cache-dir", str(tmp_path / "ic2")]) == 1
        assert "回測沒有執行" in (out2 / "summary_grid.txt").read_text(encoding="utf-8")


class TestRuntime:
    def test_runtime_smoke_23_years(self):
        """合成23年日線(2001~2023) + 近期段：1872變體×3期間的回測時間(印出來；寬鬆上限)。"""
        idx = make_index("1998-01-05", "2026-09-30", seed=17)
        ds = {"source": "index", "main": tdd._index_frame(idx), "index": tdd._index_frame(idx)}
        t0 = time.perf_counter()
        res = grid.run_grid(ds, eng.grid_variant_list())
        dt = time.perf_counter() - t0
        print(f"\n[runtime] 1872變體×3期間 = {res['timing']['n_runs']}次回測：{dt:.1f}秒 "
              f"(指標{res['timing']['indicators_sec']:.2f}秒)")
        assert res["timing"]["n_runs"] == 1872 * 3
        assert len(res["stats"]) == 1872 * 3
        assert dt < 300


class TestWithoutScipyAndNetwork:
    def test_grid_e2e_scipy_blocked_no_network(self, tmp_path):
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
            rc = ctd.main(["--grid", "--results-dir", {str(tmp_path / 'res')!r}, "--n-bootstrap", "100",
                           "--index-cache-dir", {str(tmp_path / 'ic')!r}])
            assert rc == 0
            assert "scipy" not in sys.modules or sys.modules["scipy"] is None
            print("GRID_E2E_OK")
        """)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_DIR)
        assert proc.returncode == 0, proc.stderr[-3000:]
        assert "GRID_E2E_OK" in proc.stdout
        assert (tmp_path / "res" / "summary_grid.txt").exists()


class TestWorkflowMode:
    def test_mode_input(self):
        txt = open(os.path.join(REPO_DIR, ".github", "workflows", "tx_daily_backtest.yml"), encoding="utf-8").read()
        assert "mode:" in txt and "- basic" in txt and "- grid" in txt and "default: basic" in txt
        assert '"${{ github.event.inputs.mode }}" = "grid"' in txt and "--grid" in txt
        assert "test_tx_daily_grid.py" in txt
        assert "path: results_tx_daily/" in txt
