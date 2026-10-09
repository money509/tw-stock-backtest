"""
test_tx_daily_engine.py —— tx_daily_engine.py離線測試：訊號觸發(不偷看未來)、動能門檻、
出場(初始停損/移動停損只收緊/跳空/反向訊號/最長持有/期末)、成本(手續費/稅/滑價/換月)、
小台/大台金額、買進持有參考。
"""
import numpy as np
import pandas as pd
import pytest

import tx_daily_engine as eng
from squeeze_kdj_signal import compute_squeeze_kdj_features
from test_tx_daily_data import make_index


def mk(closes, opens=None, highs=None, lows=None, offset=0.0, roll_idx=()):
    n = len(closes)
    closes = np.asarray(closes, float)
    opens = closes.copy() if opens is None else np.asarray(opens, float)
    highs = np.maximum(opens, closes) + 1 if highs is None else np.asarray(highs, float)
    lows = np.minimum(opens, closes) - 1 if lows is None else np.asarray(lows, float)
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes},
                      index=pd.bdate_range("2020-01-01", periods=n))
    df["Offset"] = offset
    df["RollDay"] = False
    for i in roll_idx:
        df.iloc[i, df.columns.get_loc("RollDay")] = True
    return df


def mk_ind(df, sig, name="DONCHIAN", atr=10.0, gate_long=True, gate_short=True):
    n = len(df)
    ind = pd.DataFrame(index=df.index)
    ind["ATR"] = atr
    for s in eng.SIGNAL_NAMES:
        ind[f"sig_{s}"] = 0
    full = np.zeros(n, int)
    for i, v in sig.items():
        full[i] = v
    ind[f"sig_{name}"] = full
    ind["gate_long"] = gate_long
    ind["gate_short"] = gate_short
    return ind


def run(df, ind, name="DONCHIAN", gate=False, **kw):
    kw.setdefault("commission_per_side", 0.0)
    kw.setdefault("tax_rate", 0.0)
    return eng.run_tx_daily_backtest(df, ind, name, gate, **kw)


# ---------------------------------------------------------------------------
class TestSignals:
    def test_donchian_trigger_and_no_lookahead(self):
        closes = [100.0] * 25 + [120.0] + [121.0] * 5
        df = mk(closes)
        ind = eng.compute_daily_indicators(df)
        assert ind["sig_DONCHIAN"].iloc[25] == 1          # 收盤120 > 前20天最高101
        assert (ind["sig_DONCHIAN"].iloc[:25] == 0).all()  # 觸發前一天不知道明天會突破
        # 只用<=t的資料：把t之後的資料截掉，t以前的訊號完全一樣
        for col in ["sig_DONCHIAN", "sig_MA_CROSS", "sig_SQZ_KDJ", "sig_SQZ_RELEASE", "gate_long", "gate_short"]:
            ind_trunc = eng.compute_daily_indicators(df.iloc[:26])
            assert list(ind_trunc[col]) == list(ind[col].iloc[:26]), col

    def test_no_lookahead_random_series(self):
        df = make_index("2010-01-01", "2012-12-31", seed=3)
        full = eng.compute_daily_indicators(df)
        cut = 500
        part = eng.compute_daily_indicators(df.iloc[:cut])
        for col in ["sig_DONCHIAN", "sig_MA_CROSS", "sig_SQZ_KDJ", "sig_SQZ_RELEASE", "gate_long", "gate_short"]:
            assert list(part[col]) == list(full[col].iloc[:cut]), col

    def test_donchian_short(self):
        closes = [100.0] * 25 + [80.0]
        ind = eng.compute_daily_indicators(mk(closes))
        assert ind["sig_DONCHIAN"].iloc[25] == -1

    def test_ma_cross(self):
        closes = [100.0] * 70 + [100.0 - i for i in range(1, 30)] + [80.0 + 3 * i for i in range(1, 60)]
        df = mk(closes)
        ind = eng.compute_daily_indicators(df)
        ma20, ma60 = df["Close"].rolling(20).mean(), df["Close"].rolling(60).mean()
        ups = [i for i in range(1, len(df)) if ma20.iloc[i] > ma60.iloc[i] and ma20.iloc[i - 1] <= ma60.iloc[i - 1]]
        dns = [i for i in range(1, len(df)) if ma20.iloc[i] < ma60.iloc[i] and ma20.iloc[i - 1] >= ma60.iloc[i - 1]]
        assert ups and dns
        assert list(np.where(ind["sig_MA_CROSS"] == 1)[0]) == ups
        assert list(np.where(ind["sig_MA_CROSS"] == -1)[0]) == dns

    def test_sqz_kdj_is_exact_entry_flag(self):
        df = make_index("2005-01-01", "2010-12-31", seed=7)
        ind = eng.compute_daily_indicators(df)
        flag = compute_squeeze_kdj_features(df)["EntryFlag"].to_numpy(bool)
        assert flag.sum() > 0
        assert list(ind["sig_SQZ_KDJ"] == 1) == list(flag)
        assert (ind["sig_SQZ_KDJ"] >= 0).all()  # 只做多

    def test_ttm_momentum_formula(self):
        df = mk([float(x) for x in range(1, 31)])
        mom = eng.compute_ttm_momentum(df, 20)
        t = 29
        hh, ll = df["High"].iloc[10:30].max(), df["Low"].iloc[10:30].min()
        sma = df["Close"].iloc[10:30].mean()
        assert mom.iloc[t] == pytest.approx(df["Close"].iloc[t] - ((hh + ll) / 2 + sma) / 2)

    def test_sqz_release_direction(self):
        df = make_index("2005-01-01", "2012-12-31", seed=11)
        ind = eng.compute_daily_indicators(df)
        sq = ind["Squeeze"].astype(bool)
        rel = sq.shift(1, fill_value=False) & ~sq
        assert rel.sum() > 0
        exp = np.where(rel & (ind["Momentum"] > 0), 1, np.where(rel & (ind["Momentum"] < 0), -1, 0))
        assert list(ind["sig_SQZ_RELEASE"]) == list(exp)
        assert (ind["sig_SQZ_RELEASE"][~rel] == 0).all()

    def test_shift_invariance_for_panama(self):
        """Panama調整等於整段加常數：所有訊號/門檻都不受影響(早年負價格也沒關係)。"""
        df = make_index("2005-01-01", "2010-12-31", seed=2)
        shifted = df.copy()
        for c in ("Open", "High", "Low", "Close"):
            shifted[c] = shifted[c] - 20000.0
        a, b = eng.compute_daily_indicators(df), eng.compute_daily_indicators(shifted)
        for col in ["sig_DONCHIAN", "sig_MA_CROSS", "sig_SQZ_KDJ", "sig_SQZ_RELEASE", "gate_long", "gate_short"]:
            assert list(a[col]) == list(b[col]), col
        np.testing.assert_allclose(a["ATR"].dropna(), b["ATR"].dropna())


class TestGate:
    def test_gate_logic(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        ind = pd.DataFrame({"sig_DONCHIAN": [1, 1, -1, -1], "gate_long": [True, False, True, False],
                            "gate_short": [False, False, True, True]}, index=dates)
        assert list(eng.entry_signal_series(ind, "DONCHIAN", False)) == [1, 1, -1, -1]
        assert list(eng.entry_signal_series(ind, "DONCHIAN", True)) == [1, 0, -1, -1]

    def test_gate_definition(self):
        closes = [100.0 + i for i in range(120)]  # 一路上漲：ADX高、多頭排列
        ind = eng.compute_daily_indicators(mk(closes))
        last = ind.iloc[-1]
        assert last["ADX"] >= 20 and last["MA5"] > last["MA20"] > last["MA60"]
        assert bool(last["gate_long"]) and not bool(last["gate_short"])
        assert not ind["gate_long"].iloc[:59].any()  # MA60還沒算出來不會放行

    def test_gated_variant_skips_entry_but_exit_uses_raw(self):
        df = mk([100.0] * 12)
        ind = mk_ind(df, {2: 1, 5: -1}, gate_long=False, gate_short=False)
        assert run(df, ind, gate=True) == []
        ind = mk_ind(df, {2: 1, 5: -1}, gate_long=True, gate_short=False)
        tr = run(df, ind, gate=True)
        assert len(tr) == 1 and tr[0]["exit_reason"] == "反向訊號"


class TestExits:
    def test_entry_next_open_with_slippage(self):
        df = mk([100.0] * 10, opens=[100.0] * 3 + [105.0] + [100.0] * 6)
        ind = mk_ind(df, {2: 1})
        tr = run(df, ind)
        assert tr[0]["signal_date"] == df.index[2] and tr[0]["entry_date"] == df.index[3]
        assert tr[0]["entry_price"] == 106.0
        assert tr[0]["initial_stop"] == pytest.approx(86.0)

    def test_initial_stop_intraday(self):
        closes = [100.0] * 10
        lows = [99.0] * 10
        lows[5] = 80.0
        df = mk(closes, lows=lows)
        tr = run(df, mk_ind(df, {2: 1}))
        # 進場101，停損101−20=81，第5天低點80觸及 → 81−1滑價
        assert tr[0]["exit_reason"] == "停損" and tr[0]["exit_price"] == 80.0
        assert tr[0]["exit_date"] == df.index[5] and tr[0]["hold_days"] == 3
        assert tr[0]["points"] == pytest.approx(-21.0)

    def test_gap_through_stop_fills_at_open(self):
        closes = [100.0] * 10
        opens = [100.0] * 10
        opens[5] = 70.0
        df = mk(closes, opens=opens)
        tr = run(df, mk_ind(df, {2: 1}))
        assert tr[0]["exit_price"] == 69.0 and tr[0]["exit_reason"] == "停損"

    def test_short_stop_and_gap(self):
        opens = [100.0] * 10
        opens[6] = 130.0
        df = mk([100.0] * 10, opens=opens)
        tr = run(df, mk_ind(df, {2: -1}))
        assert tr[0]["side"] == "short" and tr[0]["entry_price"] == 99.0
        assert tr[0]["initial_stop"] == pytest.approx(119.0)
        assert tr[0]["exit_price"] == 131.0
        assert tr[0]["points"] == pytest.approx(-32.0)

    def test_trailing_only_tightens(self):
        closes = [100.0] * 3 + [110, 130, 150, 140, 135, 145, 138] + [138.0] * 5
        df = mk(closes)
        ind = mk_ind(df, {2: 1})
        stops = []
        best = None
        stop = 101 - 20
        for i in range(3, len(closes)):
            best = closes[i] if best is None else max(best, closes[i])
            stop = max(stop, best - 30)
            stops.append(stop)
        assert all(b >= a for a, b in zip(stops, stops[1:]))
        tr = run(df, ind)
        # 最高收盤150 → 移動停損120；之後收盤回落不會讓停損下降，也沒觸及 → 期末平倉
        assert tr[0]["exit_reason"] == "期末平倉"
        lows = [c - 1 for c in closes]
        lows[12] = 119.0
        df2 = mk(closes, lows=lows)
        tr2 = run(df2, mk_ind(df2, {2: 1}))
        assert tr2[0]["exit_reason"] == "移動停損" and tr2[0]["exit_price"] == 119.0

    def test_opposite_signal_exits_next_open_no_reverse(self):
        opens = [100.0] * 12
        opens[7] = 103.0
        df = mk([100.0] * 12, opens=opens)
        ind = mk_ind(df, {2: 1, 6: -1, 9: -1})
        tr = run(df, ind)
        assert tr[0]["exit_reason"] == "反向訊號" and tr[0]["exit_date"] == df.index[7]
        assert tr[0]["exit_price"] == 102.0
        # 第6天的反向訊號只平倉、不反手；第9天的新訊號才會進場做空
        assert len(tr) == 2 and tr[1]["side"] == "short" and tr[1]["entry_date"] == df.index[10]

    def test_max_hold(self):
        n = 80
        df = mk([100.0 + 0.1 * i for i in range(n)])
        tr = run(df, mk_ind(df, {2: 1}, atr=5.0), max_hold_days=60)
        assert tr[0]["exit_reason"] == "最長持有" and tr[0]["hold_days"] == 60
        assert tr[0]["exit_date"] == df.index[63]

    def test_period_bounds_and_forced_close(self):
        df = mk([100.0] * 30)
        ind = mk_ind(df, {2: 1, 12: 1})
        tr = run(df, ind, start=df.index[10], end=df.index[20])
        assert len(tr) == 1 and tr[0]["entry_date"] == df.index[13]
        assert tr[0]["exit_reason"] == "期末平倉" and tr[0]["exit_date"] == df.index[20]
        assert tr[0]["exit_price"] == 99.0
        # 最後一天的訊號不會進場
        assert run(df, mk_ind(df, {20: 1}), start=df.index[10], end=df.index[20]) == []


class TestCosts:
    def test_mini_vs_big_and_costs(self):
        closes = [10000.0] * 3 + [10100.0] * 6
        opens = [10000.0] * 3 + [10000.0] + [10100.0] * 5
        df = mk(closes, opens=opens, offset=-2000.0)  # 未調整價格 = 調整後 − (−2000)
        ind = mk_ind(df, {2: 1}, atr=100.0)
        for contract, mult in (("mini", 50), ("big", 200)):
            tr = eng.run_tx_daily_backtest(df, ind, "DONCHIAN", False, contract=contract,
                                           commission_per_side=50.0)
            t = tr[0]
            assert t["entry_price"] == 10001.0 and t["exit_price"] == 10099.0
            assert t["raw_entry_price"] == 12001.0
            tax = (12001.0 + 12099.0) * mult * 0.00002
            assert t["cost_ntd"] == pytest.approx(100.0 + tax)
            assert t["pnl_ntd"] == pytest.approx(98.0 * mult - 100.0 - tax)

    def test_roll_cost_only_while_holding(self):
        df = mk([10000.0] * 12, roll_idx=(1, 5, 9))
        ind = mk_ind(df, {2: 1}, atr=100.0)
        tr = eng.run_tx_daily_backtest(df, ind, "DONCHIAN", False, contract="mini",
                                       commission_per_side=50.0, end=df.index[8])
        t = tr[0]
        assert t["n_rolls"] == 1  # 第1天換月時還沒進場；第9天在區間外
        roll_cost = 100.0 + 2 * 10000.0 * 50 * 0.00002
        assert t["roll_cost_ntd"] == pytest.approx(roll_cost)
        base = 100.0 + (10001.0 + 9999.0) * 50 * 0.00002
        assert t["cost_ntd"] == pytest.approx(base + roll_cost)

    def test_no_roll_cost_when_exiting_on_roll_open(self):
        df = mk([10000.0] * 10, roll_idx=(5,))
        ind = mk_ind(df, {2: 1, 4: -1}, atr=100.0)
        tr = eng.run_tx_daily_backtest(df, ind, "DONCHIAN", False, commission_per_side=50.0)
        assert tr[0]["exit_date"] == df.index[5] and tr[0]["n_rolls"] == 0

    def test_buy_and_hold(self):
        df = mk([100.0 + i for i in range(20)], opens=[100.0 + i for i in range(20)], roll_idx=(0, 4, 15))
        bh = eng.run_buy_and_hold(df, contract="mini", commission_per_side=50.0, tax_rate=0.0)
        t = bh[0]
        assert t["n_rolls"] == 2 and t["entry_price"] == 101.0 and t["exit_price"] == 118.0
        assert t["pnl_ntd"] == pytest.approx(17.0 * 50 - 100.0 - 2 * 100.0)
        assert t["variant"] == "買進持有(參考)"

    def test_summarize_mr_compatible(self):
        from mean_reversion_engine import summarize_mr
        from robustness_analysis import bootstrap_resample_pnl
        df = make_index("2005-01-01", "2008-12-31", seed=4)
        df["Offset"] = 0.0
        df["RollDay"] = False
        ind = eng.compute_daily_indicators(df)
        tr = eng.run_tx_daily_backtest(df, ind, "DONCHIAN", False)
        assert len(tr) > 5
        s = summarize_mr(tr, eng.STARTING_CAPITAL)
        assert s["trade_count"] == len(tr)
        assert len(bootstrap_resample_pnl(tr, 10)) == 10
        # 一次只有一個部位：前一筆出場日 <= 下一筆進場日
        for a, b in zip(tr, tr[1:]):
            assert a["exit_date"] <= b["entry_date"]

    def test_variant_list(self):
        names = [v["name"] for v in eng.variant_list()]
        assert len(names) == 8 and names[0] == "DONCHIAN" and names[1] == "DONCHIAN+動能門檻"
