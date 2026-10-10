"""
test_leader_signal.py —— 強勢股(領頭股突破 E_D｜X2｜K3｜無濾網)每日實盤訊號 leader_signal.py 的離線測試
(全部合成資料、不連網)：凍結設定 = leader_breakout 變體、重播 = 回測逐筆相同、下載視窗夠暖機、
空位/排序、股數與零股顯示、移動停損只上移且等於回測、今天停損出場、資料沒更新 → 沒有買進指示、
PAPER_START、紙上紀錄冪等、Telegram長度與截斷、不偷看未來、沒有scipy也能跑、workflow接線。
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import leader_breakout as lb
import leader_signal as s
import winners_study as ws

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
WF_PATH = os.path.join(REPO_DIR, ".github", "workflows", "leader_daily_signal.yml")
END = "2026-12-31"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("LEADER_SIGNALS_LINK", raising=False)


# ---------------------------------------------------------------------------
# 合成資料：30檔；10檔在 PAPER_START 之後跳空突破(6檔同一天 → 候選比空位多)，接著上漲25天、再下跌20天(觸發移動停損)
# ---------------------------------------------------------------------------
def _ohlc(close, dates, volume, rng):
    close = np.asarray(close, float)
    prev = np.r_[close[0], close[:-1]]
    op = prev * (1 + rng.normal(0, 0.003, len(close)))
    spread = np.abs(rng.normal(0.01, 0.004, len(close)))
    hi = np.maximum(op, close) * (1 + spread)
    lo = np.minimum(op, close) * (1 - spread)
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Adj Close": close,
                         "Volume": np.full(len(close), float(volume))}, index=pd.DatetimeIndex(dates))


def build_synthetic(n=30, seed=5, start="2025-05-01", end=END):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    inds = ["半導體業", "電子零組件業", "航運業", "食品工業", "光電業", "資訊服務業"]
    p0 = dates.searchsorted(pd.Timestamp(s.PAPER_START))
    rows, prices, rev = [], {}, []
    for i in range(n):
        code = f"{3000 + i}"
        rows.append({"code": code, "name": f"強勢{i}", "market": "上市" if i % 3 else "上櫃", "industry": inds[i % 6]})
        r = -0.0003 + rng.normal(0, 0.012, T)
        if i < 10:
            d = p0 + (5 if i < 6 else 16)
            r[:d] = rng.normal(0, 0.006, d)
            r[d - 25:d] -= 0.003
            r[d] = 0.30 + 0.01 * i
            r[d + 1:d + 26] += 0.01
            r[d + 26:d + 46] -= 0.02
        close = (20 + 5 * i) * np.exp(np.cumsum(r))
        prices[code] = _ohlc(close, dates, 2_000_000, rng)
        for y in (2024, 2025, 2026):
            for m in range(1, 13):
                rev.append({"code": code, "name": "", "industry": "", "revenue": 1000.0, "revenue_ly": 1000.0,
                            "mom_pct": 0.0, "yoy_pct": rng.normal(5 if i % 5 == 4 else 40, 5), "cum_yoy_pct": 0.0,
                            "year": y, "month": m, "market": "上市"})
    return pd.DataFrame(rows), prices, pd.DataFrame(rev)


@pytest.fixture(scope="module")
def synth():
    return build_synthetic()


@pytest.fixture(scope="module")
def full_replay(synth):
    uni, prices, rev = synth
    P = lb.build_panels(uni, prices, rev, pd.Series(dtype=float))
    return P, s.replay(P)


def ev(synth, as_of, **kw):
    uni, prices, rev = synth
    return s.evaluate(uni, prices, rev, as_of, **kw)


def trading_days(synth, a, b):
    d = synth[1]["3000"].index
    return [x for x in d if pd.Timestamp(a) <= x <= pd.Timestamp(b)]


def _trade_key(rows):
    return [(r["code"], pd.Timestamp(r["entry_date"]), pd.Timestamp(r["exit_date"]), round(float(r["entry_price"]), 9),
             round(float(r["exit_price"]), 9), int(r["shares"]), round(float(r["pnl"]), 6), r["exit_reason"])
            for r in rows]


# ---------------------------------------------------------------------------
class TestSettings:
    def test_matches_leader_breakout_variant(self):
        v = s.live_variant()
        assert v["name"] == "E_D｜X2｜K3｜無濾網" == lb.variant_name("E_D", "X2", 3, False)
        assert (v["entry"], v["exit"], v["K"], v["filter"], v["rank"]) == ("E_D", "X2", 3, False, "RS")
        st = s.LEADER_LIVE_SETTINGS
        assert st["risk_pct"] == 0.015 and st["trail_pct"] == 0.20 and st["init_stop_atr"] == 2.0
        assert st["rs_top"] == 0.80 and st["rev_yoy_min"] == 20.0 and st["frozen"] == "2026-10"
        assert s.PAPER_START == "2026-10-12" and s.PAPER_CAPITAL == 200_000

    def test_mismatch_raises(self, monkeypatch):
        monkeypatch.setitem(s.LEADER_LIVE_SETTINGS, "K", 1)
        with pytest.raises(ValueError):
            s.live_variant()

    def test_frozen_comment_present(self):
        src = open(s.__file__, encoding="utf-8").read()
        assert "CAGR 16.1%" in src and "−31.4%" in src and "不要依最近的結果調整" in src


# ---------------------------------------------------------------------------
class TestReplayEquivalence:
    def test_matches_run_backtest_trade_for_trade(self, synth):
        uni, prices, rev = synth
        as_of = "2026-12-24"
        bt = lb.run_backtest(uni, s.truncate_prices(prices, as_of), rev, pd.Series(dtype=float), None,
                             s.PAPER_START, as_of, 200_000, variants=[s.live_variant()])
        tdf = bt["trades"]
        closed_bt = tdf[~tdf["open_at_end"].astype(bool)].to_dict("records")
        open_bt = tdf[tdf["open_at_end"].astype(bool)]
        res = ev(synth, as_of)
        assert len(closed_bt) >= 3
        assert _trade_key(res["closed_trades"]) == _trade_key(closed_bt)
        assert sorted(open_bt["code"]) == sorted(h["code"] for h in res["holdings"])
        for h in res["holdings"]:
            r = open_bt[open_bt["code"] == h["code"]].iloc[0]
            assert r["shares"] == h["shares"] and abs(r["entry_price"] - h["entry_price"]) < 1e-9
        assert abs(bt["equity"][s.LEADER_LIVE_SETTINGS["name"]].iloc[-1] - res["account"]["equity"]) < 1e-6

    def test_state_out_does_not_change_simulation(self, full_replay):
        P, rp = full_replay
        v = s.live_variant()
        a = lb.simulate_portfolio(P, rp["sig"], rp["i0"], rp["i1"], "X2", 3, 200_000)
        st = {}
        b = lb.simulate_portfolio(P, rp["sig"], rp["i0"], rp["i1"], "X2", 3, 200_000, state_out=st)
        assert a["trades"] == b["trades"] and a["open"] == b["open"] and a["counts"] == b["counts"]
        np.testing.assert_array_equal(a["equity"], b["equity"])
        assert {"positions", "pending", "cash", "equity"} <= set(st)
        assert abs(st["equity"] - a["equity"][-1]) < 1e-9 and abs(st["cash"] - a["cash"]) < 1e-9
        assert v["K"] == 3

    def test_pending_is_what_the_next_day_buys(self, synth):
        """as_of 的計畫(依序補空位) = 下一天重播實際進場的股票與股數(開盤價不同，股數可能略有差 → 只比代號)。"""
        res = ev(synth, "2026-10-19")
        assert res["status"] == "ok" and res["plan"]["buys"]
        nxt = ev(synth, "2026-10-20")
        bought = [h["code"] for h in nxt["holdings"] if h["entered_today"]]
        assert bought == [b["code"] for b in res["plan"]["buys"]]

    def test_download_window_warmup_is_enough(self, synth):
        """只用 data_windows() 的下載起點(PAPER_START−420天)跟用完整歷史，重播結果完全相同。"""
        uni, prices, rev = synth
        as_of = "2026-12-24"
        dl_start, _, rev_start, _ = s.data_windows(pd.Timestamp(as_of).date())
        cut = {c: df[df.index >= pd.Timestamp(dl_start)] for c, df in prices.items()}
        rev_cut = rev[(rev["year"] * 12 + rev["month"]) >= rev_start.year * 12 + rev_start.month]
        a, b = ev(synth, as_of), s.evaluate(uni, cut, rev_cut, as_of)
        assert _trade_key(a["closed_trades"]) == _trade_key(b["closed_trades"])
        assert [(h["code"], h["shares"], round(h["stop"], 9)) for h in a["holdings"]] == \
               [(h["code"], h["shares"], round(h["stop"], 9)) for h in b["holdings"]]
        assert s.render_telegram(a) == s.render_telegram(b)

    def test_data_windows(self):
        import datetime
        a, e, r0, r1 = s.data_windows(datetime.date(2026, 11, 2))
        assert a == "2025-08-18" and e == "2026-11-03"          # PAPER_START − 420天、as_of+1(yfinance end不含)
        assert r0 == pd.Timestamp("2025-07-01") and r1 == pd.Timestamp("2026-11-02")
        a2, _, r02, _ = s.data_windows(datetime.date(2027, 12, 1))
        assert a2 == a and r02 == r0                              # 一年後仍從紙上交易起點往回暖機


# ---------------------------------------------------------------------------
class TestEntries:
    def test_limited_by_free_slots_and_ranked(self, synth, full_replay):
        res = ev(synth, "2026-10-19")
        plan = res["plan"]
        assert len(res["holdings"]) == 1 and plan["free"] == 2
        assert len(plan["buys"]) == 2 and len(plan["runners"]) >= 1
        P, _ = full_replay
        t = P["cal"].get_loc(pd.Timestamp("2026-10-19"))
        A = P["A"]
        cand = [b["j"] for b in plan["buys"] + plan["runners"]]
        expect = [int(j) for j in lb.rank_order(np.array(cand), A["rs"][t, cand], A["turnover20"][t, cand])]
        assert cand == expect
        assert all(b["rs"] >= r["rs"] for b in plan["buys"] for r in plan["runners"])
        assert [b["rank"] for b in plan["buys"]] == [1, 2]
        held = {h["code"] for h in res["holdings"]}
        assert not held & {x["code"] for x in plan["buys"] + plan["runners"]}
        md = s.render_markdown(res)
        assert "## 1. 明天開盤要買" in md and "候補" in md and "開盤買進 2 檔" in md

    def test_no_slots_means_no_buys_but_runners(self, full_replay):
        P, rp = full_replay
        st = dict(rp["state"])
        t = rp["i1"]
        st["positions"] = [{"j": 90 + i} for i in range(3)]
        A = P["A"]
        e = np.flatnonzero(A["elig"][t])[:4]
        st["pending"] = e
        plan = s.plan_entries(P, st, t, 3)
        assert plan["free"] == 0 and plan["buys"] == [] and [r["j"] for r in plan["runners"]] == list(e)

    def test_runners_capped_at_five(self, full_replay):
        P, rp = full_replay
        t = rp["i1"]
        st = {"positions": [], "cash": 2e5, "equity": 2e5, "pending": np.flatnonzero(P["A"]["elig"][t])}
        assert len(st["pending"]) > 8
        plan = s.plan_entries(P, st, t, 3)
        assert len(plan["buys"]) == 3 and len(plan["runners"]) == 5


class TestSizing:
    def test_sequential_sizing_matches_position_shares(self, synth, full_replay):
        res = ev(synth, "2026-10-19")
        P, _ = full_replay
        cash, eq = res["account"]["cash"], res["account"]["equity"]
        for b in res["plan"]["buys"]:
            n = lb.position_shares(eq, cash, b["close"], b["atr"], 3)
            assert b["shares"] == n >= 1
            assert abs(b["cost"] - (n * b["close"] + lb.buy_fee(n * b["close"]))) < 1e-6
            assert abs(b["approx_stop"] - (b["close"] - 2 * b["atr"])) < 1e-9
            cash -= b["cost"]
        assert abs(res["plan"]["cash_after"] - cash) < 1e-6

    @pytest.mark.parametrize("n,txt", [(1, "零股1股"), (999, "零股999股"), (1000, "1張"), (1226, "1張+零股226股"),
                                       (3000, "3張")])
    def test_lots_text(self, n, txt):
        assert s.lots_text(n) == txt

    def test_odd_lot_display_in_outputs(self, synth):
        res = ev(synth, "2026-10-19")
        for b in res["plan"]["buys"]:
            b["shares"] = 640
        md, tg = s.render_markdown(res), s.render_telegram(res)
        assert "640股(零股640股)" in md and "零股640股" in tg


# ---------------------------------------------------------------------------
class TestStops:
    def test_trailing_stop_never_decreases_and_matches_backtest(self, synth):
        days = trading_days(synth, "2026-10-20", "2026-12-24")
        hist = {}
        for d in days:
            res = ev(synth, d)
            for h in res["holdings"]:
                assert abs(h["stop"] - max(h["init_stop"], 0.8 * h["highest_close"])) < 1e-9
                hist.setdefault((h["code"], h["entry_date"]), []).append((d, h["stop"], h["stop_moved_up"],
                                                                          h["prev_stop"]))
            for x in res["exits_today"]:
                key = (x["code"], x["entry_date"])
                if key in hist and "跳空" not in x["exit_reason"]:
                    assert abs(x["exit_price"] - hist[key][-1][1]) < 1e-9   # 以昨天報的停損價出場
        assert any(any(m for _, _, m, _ in v) for v in hist.values())
        for v in hist.values():
            stops = [x[1] for x in v]
            assert all(b >= a - 1e-12 for a, b in zip(stops, stops[1:]))
            for (_, s0, _, _), (_, s1, moved, prev) in zip(v, v[1:]):
                assert moved == (s1 > s0 + 1e-9)
                if moved:
                    assert abs(prev - s0) < 1e-9

    def test_stopped_out_today(self, synth, full_replay):
        P, rp = full_replay
        tr = rp["sim"]["trades"][0]
        exit_day = P["cal"][tr["exit_idx"]]
        code = P["codes"][tr["code_idx"]]
        res = ev(synth, exit_day)
        assert [x["code"] for x in res["exits_today"]] == [code]
        assert code not in [h["code"] for h in res["holdings"]]
        assert abs(res["exits_today"][0]["pnl"] - tr["pnl"]) < 1e-9
        md, tg = s.render_markdown(res), s.render_telegram(res)
        assert "## 3. 今天出場" in md and code in md.split("## 3. 今天出場")[1].split("## 4.")[0]
        assert "＝今天出場＝" in tg
        before = ev(synth, P["cal"][tr["exit_idx"] - 1])
        assert code in [h["code"] for h in before["holdings"]] and not before["exits_today"]


# ---------------------------------------------------------------------------
class TestStaleAndStart:
    def test_weekend_is_stale_no_buys(self, synth):
        res = ev(synth, "2026-10-24")          # 星期六
        assert res["status"] == "stale" and res["plan"] is None and not res["data_fresh"]
        assert res["holdings"]                 # 持股照列
        md, tg = s.render_markdown(res), s.render_telegram(res)
        assert "⚠️ 資料還沒更新" in md and "開盤買進" not in md
        assert "⚠️" in tg and "今天不給買進指示" in tg
        assert s.paper_log_row(res)["new_buys"].startswith("資料未更新")

    def test_partial_update_is_stale(self, synth):
        uni, prices, rev = synth
        d = pd.Timestamp("2026-10-19")
        part = {c: (df[df.index < d] if i % 2 == 0 else df) for i, (c, df) in enumerate(prices.items())}
        res = s.evaluate(uni, part, rev, d)
        assert res["latest_date"] == d and res["status"] == "stale" and res["plan"] is None
        assert "資料可能還沒更新完" in res["stale_reason"]

    def test_revenue_missing_no_buys(self, synth):
        res = ev(synth, "2026-10-19", info={"revenue_ok": False})
        assert res["status"] == "no_revenue" and res["plan"] is None
        assert "月營收" in s.render_telegram(res)

    def test_before_paper_start(self, synth):
        res = ev(synth, "2026-10-09")
        assert res["status"] == "before_start" and res["plan"] is None
        assert res["holdings"] == [] and res["account"]["equity"] == 200_000
        assert "紙上交易尚未開始" in s.render_telegram(res)

    def test_first_day_no_positions_and_entries_after_start(self, synth):
        res = ev(synth, s.PAPER_START)
        assert res["status"] == "ok" and res["holdings"] == [] and res["closed_trades"] == []
        late = ev(synth, "2026-12-24")
        start = pd.Timestamp(s.PAPER_START)
        assert all(t["entry_date"] > start for t in late["closed_trades"])
        assert all(h["entry_date"] > start for h in late["holdings"])
        df = s.trades_frame(late)
        assert len(df) == len(late["closed_trades"]) and "進場日" in df.columns

    def test_capital_override(self, synth):
        a, b = ev(synth, "2026-10-19"), ev(synth, "2026-10-19", capital=1_000_000)
        assert b["account"]["equity"] > 900_000
        assert sum(x["shares"] for x in b["plan"]["buys"]) > sum(x["shares"] for x in a["plan"]["buys"])


class TestNoLookahead:
    @pytest.mark.parametrize("as_of", ["2026-10-19", "2026-11-10"])
    def test_future_bars_do_not_change_output(self, synth, as_of):
        uni, prices, rev = synth
        cut = {c: df[df.index <= pd.Timestamp(as_of)] for c, df in prices.items()}
        a = s.evaluate(uni, cut, rev, as_of)
        b = s.evaluate(uni, prices, rev, as_of)   # 含之後的K棒
        assert s.render_markdown(a) == s.render_markdown(b)
        assert s.render_telegram(a) == s.render_telegram(b)
        assert s.paper_log_row(a) == s.paper_log_row(b)


# ---------------------------------------------------------------------------
class TestOutputs:
    def test_paper_log_idempotent(self, synth, tmp_path):
        r1 = ev(synth, "2026-10-19")
        r2 = ev(synth, "2026-10-20")
        path = str(tmp_path / "log.csv")
        s.update_paper_log(path, s.paper_log_row(r2))
        s.update_paper_log(path, s.paper_log_row(r1))
        s.update_paper_log(path, s.paper_log_row(r1))
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        assert list(df.columns) == s.PAPER_LOG_COLUMNS
        assert list(df["date"]) == ["2026-10-19", "2026-10-20"]
        assert df.loc[0, "new_buys"] == "、".join(f"{b['code']}×{b['shares']}" for b in r1["plan"]["buys"])

    def test_write_outputs_files(self, synth, tmp_path):
        res = ev(synth, "2026-11-10")
        out = s.write_outputs(res, str(tmp_path))
        for k in ("latest", "history", "telegram", "paper_log", "trades"):
            assert os.path.exists(out[k]), k
        assert out["history"].endswith(os.path.join("leader_history", "2026-11-10.md"))
        tg = open(out["telegram"], encoding="utf-8").read()
        assert tg.splitlines()[0] == "【強勢股訊號】2026-11-10"
        s.write_outputs(res, str(tmp_path))
        assert len(pd.read_csv(out["paper_log"])) == 1

    def test_markdown_sections(self, synth):
        md = s.render_markdown(ev(synth, "2026-11-10"))
        for sec in ("## 1. 明天開盤要買", "## 2. 目前持股", "## 3. 今天出場", "## 4. 帳戶", "## 5. 注意事項"):
            assert sec in md
        assert "零股多數券商不支援停損單" in md and "盤中碰到停損就以停損價成交" in md and "漲停" in md

    def test_telegram_limit_truncates_runners_first(self, synth, monkeypatch):
        res = ev(synth, "2026-10-19")
        r0 = res["plan"]["runners"][0]
        res["plan"]["runners"] = [dict(r0, code=f"R{i}", name="很長的名字" * 6) for i in range(5)]
        full = s.render_telegram(res)
        assert full.splitlines()[0] == "【強勢股訊號】2026-10-19"
        assert all(f"R{i} " in full for i in range(5))
        no_runner = s.render_telegram(dict(res, plan=dict(res["plan"], runners=[])))
        limit = (len(full) + len(no_runner)) // 2              # 放得下持股/買進，放不下全部候補
        monkeypatch.setattr(s, "TELEGRAM_MAX_CHARS", limit)
        tg = s.render_telegram(res, link="https://example.com/x")
        assert len(tg) < limit and "…(其餘看完整版)" not in tg
        kept = [i for i in range(5) if f"R{i} " in tg]
        assert 0 < len(kept) < 5 and kept == list(range(len(kept)))   # 從最後一個候補開始刪
        assert "example.com" not in tg                                 # 連結先拿掉
        for line in no_runner.splitlines():                            # 買進/持股/帳戶都完整保留
            assert line in tg
        monkeypatch.setattr(s, "TELEGRAM_MAX_CHARS", 3500)
        h0 = dict(res["holdings"][0])
        res["holdings"] = [dict(h0, name="持股名稱" * 300) for _ in range(3)]
        tg2 = s.render_telegram(res)
        assert len(tg2) < 3500 and tg2.endswith("…(其餘看完整版)") and "候補" not in tg2
        assert tg2.splitlines()[0] == "【強勢股訊號】2026-10-19"

    def test_telegram_link_appended_when_short(self, synth):
        tg = s.render_telegram(ev(synth, "2026-10-19"), link="https://example.com/leader_latest.md")
        assert tg.endswith("完整說明：https://example.com/leader_latest.md")


# ---------------------------------------------------------------------------
def _patch_loaders_code(as_of_data_end):
    return textwrap.dedent(f"""
        import pandas as pd
        import winners_study as ws
        from test_leader_signal import build_synthetic
        uni, prices, rev = build_synthetic(end={as_of_data_end!r})
        calls = {{}}
        def fake_universe(cache_dir, refresh, debug_dir, today_str):
            calls["universe_key"] = today_str
            return uni.copy(), []
        def fake_prices(universe, dl_start, dl_end, cache_dir, refresh, debug_dir):
            calls["prices"] = (dl_start, dl_end, refresh)
            out = {{c: df[(df.index >= dl_start) & (df.index < dl_end)] for c, df in prices.items()}}
            return out, pd.DataFrame(columns=["代號"])
        def fake_revenue(start_month, end_month, cache_dir, refresh, debug_dir):
            calls["revenue"] = (start_month, end_month, refresh)
            return rev, True
        ws.load_universe, ws.load_prices, ws.load_revenue = fake_universe, fake_prices, fake_revenue
    """)


class TestCli:
    def test_main_end_to_end(self, tmp_path, monkeypatch):
        uni, prices, rev = build_synthetic()
        calls = {}

        def fake_prices(universe, dl_start, dl_end, cache_dir, refresh, debug_dir):
            calls["prices"] = (dl_start, dl_end, refresh)
            return ({c: df[(df.index >= dl_start) & (df.index < dl_end)] for c, df in prices.items()},
                    pd.DataFrame(columns=["代號"]))
        monkeypatch.setattr(ws, "load_universe", lambda c, r, d, key: (calls.setdefault("key", key), (uni, []))[1])
        monkeypatch.setattr(ws, "load_prices", fake_prices)
        monkeypatch.setattr(ws, "load_revenue", lambda a, b, c, r, d: (calls.setdefault("rev", (a, b)), (rev, True))[1])
        out = tmp_path / "signals"
        rc = s.main(["--output-dir", str(out), "--as-of", "2026-11-10", "--capital", "300000",
                     "--cache-dir", str(tmp_path / "c"), "--debug-dir", str(tmp_path / "d")])
        assert rc == 0
        assert calls["key"] == "2026-11"                                     # 股票清單每月一份快取
        assert calls["prices"] == ("2025-08-18", "2026-11-11", True)          # 股價每次重抓
        assert calls["rev"][0] == pd.Timestamp("2025-07-01")
        for f in ("leader_latest.md", "telegram_leader.txt", "leader_history/2026-11-10.md", "leader_paper_log.csv",
                  "leader_trades_paper.csv"):
            assert (out / f).exists(), f
        log = pd.read_csv(out / "leader_paper_log.csv")
        assert log.loc[0, "equity"] != 200000 and "NT$300,000" in (out / "leader_latest.md").read_text("utf-8")

    def test_main_fails_closed_without_universe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ws, "load_universe", lambda *a: (pd.DataFrame(columns=["code"]), ["x"]))
        assert s.main(["--output-dir", str(tmp_path), "--as-of", "2026-11-10",
                       "--debug-dir", str(tmp_path / "d")]) == 1
        assert not (tmp_path / "telegram_leader.txt").exists()


class TestWithoutScipy:
    def test_module_does_not_import_scipy(self):
        src = open(s.__file__, encoding="utf-8").read()
        assert "import scipy" not in src and "from scipy" not in src

    def test_end_to_end_with_scipy_blocked(self, tmp_path):
        code = textwrap.dedent(f"""
            import sys
            class _Block:
                def find_spec(self, name, path=None, target=None):
                    if name == "scipy" or name.startswith("scipy."):
                        raise ImportError("scipy blocked for test")
                    return None
            sys.meta_path.insert(0, _Block())
            sys.path.insert(0, {REPO_DIR!r})
        """) + _patch_loaders_code(END) + textwrap.dedent(f"""
            import leader_signal as s
            rc = s.main(["--output-dir", {str(tmp_path)!r}, "--as-of", "2026-11-10",
                         "--debug-dir", {str(tmp_path / "debug")!r}, "--cache-dir", {str(tmp_path / "cache")!r}])
            assert rc == 0
            assert "scipy" not in sys.modules or sys.modules["scipy"] is None
            print("OK")
        """)
        env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_STEP_SUMMARY", "LEADER_SIGNALS_LINK")}
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_DIR, env=env,
                              timeout=300)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
        assert "OK" in proc.stdout
        for name in ("leader_latest.md", "telegram_leader.txt", "leader_paper_log.csv", "leader_trades_paper.csv",
                     "leader_history/2026-11-10.md"):
            assert (tmp_path / name).exists(), name


# ---------------------------------------------------------------------------
class TestWorkflow:
    def _wf(self):
        yaml = pytest.importorskip("yaml")
        with open(WF_PATH, encoding="utf-8") as f:
            text = f.read()
        return text, yaml.safe_load(text)

    def _job(self, wf):
        return wf["jobs"]["signal"]

    def _step(self, wf, name):
        return next(st for st in self._job(wf)["steps"] if st.get("name") == name)

    def test_parses_schedule_and_inputs(self):
        text, wf = self._wf()
        on = wf.get("on", wf.get(True))
        assert on["schedule"][0]["cron"] == "20 11 * * 1-5"
        inputs = on["workflow_dispatch"]["inputs"]
        assert str(inputs["capital"]["default"]) == "200000" and "as_of" in inputs
        assert wf["concurrency"]["group"] == "leader-daily-signal"
        assert self._job(wf)["timeout-minutes"] == 45
        assert wf["permissions"]["contents"] == "write"
        header = text.split("on:")[0]
        assert "19:20" in header and "2026-10-12" in header and "紙上交易" in header

    def test_preflight_runs_only_own_tests_before_run(self):
        _, wf = self._wf()
        names = [st.get("name") for st in self._job(wf)["steps"]]
        assert self._step(wf, "Pre-flight test")["run"].strip() == "python -m pytest -q test_leader_signal.py"
        assert names.index("Pre-flight test") < names.index("Run leader signal")
        assert self._job(wf)["steps"][1]["with"]["python-version"] == "3.10"

    def test_cache_steps(self):
        _, wf = self._wf()
        steps = self._job(wf)["steps"]
        caches = [st for st in steps if "actions/cache" in st.get("uses", "")]
        assert len(caches) == 2 and all(c["with"]["path"] == "leader_signal_cache" for c in caches)
        assert "steps.date.outputs.month" in caches[0]["with"]["key"]

    @pytest.mark.parametrize("as_of,capital,expected", [
        ("", "", "--output-dir signals --cache-dir leader_signal_cache --debug-dir leader_signal_debug --capital 200000"),
        ("2026-11-02", "500000", "--output-dir signals --cache-dir leader_signal_cache --debug-dir leader_signal_debug "
                                 "--capital 500000 --as-of 2026-11-02"),
    ])
    def test_run_step_args_and_stale_telegram_deleted(self, as_of, capital, expected, tmp_path):
        _, wf = self._wf()
        step = self._step(wf, "Run leader signal")
        assert step["continue-on-error"] is True and step["id"] == "run"
        assert "rm -f signals/telegram_leader.txt" in step["run"]
        (tmp_path / "signals").mkdir()
        (tmp_path / "signals" / "telegram_leader.txt").write_text("舊的", encoding="utf-8")
        script = step["run"].replace("python leader_signal.py", "echo ARGS=")
        proc = subprocess.run(["bash", "-e", "-c", script], capture_output=True, text=True, cwd=str(tmp_path),
                              env={"PATH": os.environ.get("PATH", ""), "AS_OF": as_of, "CAPITAL": capital}, timeout=30)
        assert proc.returncode == 0, proc.stderr
        line = [x for x in proc.stdout.splitlines() if x.startswith("ARGS=")][0]
        assert line[len("ARGS="):].strip() == expected
        assert not (tmp_path / "signals" / "telegram_leader.txt").exists()

    def test_secrets_and_failure_notice(self, tmp_path):
        text, wf = self._wf()
        step = self._step(wf, "Send Telegram message")
        assert step["env"]["TELEGRAM_BOT_TOKEN"] == "${{ secrets.TELEGRAM_BOT_TOKEN }}"
        assert step["env"]["TELEGRAM_CHAT_ID"] == "${{ secrets.TELEGRAM_CHAT_ID }}"
        assert step["env"]["RUN_OUTCOME"] == "${{ steps.run.outcome }}"
        run = step["run"]
        (tmp_path / "signals").mkdir()
        bindir = tmp_path / "bin"
        bindir.mkdir()
        log = tmp_path / "curl.log"
        (bindir / "curl").write_text(textwrap.dedent(f"""\
            #!/bin/bash
            for a in "$@"; do
              case "$a" in
                text@*) f="${{a#text@}}"; echo "FILE:$(cat "$f")" >> {log} ;;
                text=*) echo "TEXT:${{a#text=}}" >> {log} ;;
              esac
            done
            echo '{{"ok":true}}'
        """), encoding="utf-8")
        os.chmod(bindir / "curl", 0o755)
        base = {"PATH": f"{bindir}:{os.environ.get('PATH', '')}", "TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}

        def go(outcome):
            if log.exists():
                log.unlink()
            p = subprocess.run(["bash", "-c", run], capture_output=True, text=True, cwd=str(tmp_path),
                               env=dict(base, RUN_OUTCOME=outcome), timeout=30)
            return p, (log.read_text(encoding="utf-8") if log.exists() else "")
        p, sent = go("failure")                                  # 失敗、沒有檔案
        assert p.returncode == 0 and "TEXT:強勢股訊號產生失敗，請看Actions紀錄" in sent
        (tmp_path / "signals" / "telegram_leader.txt").write_text("【強勢股訊號】OK", encoding="utf-8")
        p, sent = go("failure")                                  # 有舊檔但步驟失敗 → 仍送失敗通知
        assert "產生失敗" in sent and "FILE:" not in sent
        p, sent = go("success")
        assert p.returncode == 0 and "FILE:【強勢股訊號】OK" in sent and "產生失敗" not in sent
        p = subprocess.run(["bash", "-c", run], capture_output=True, text=True, cwd=str(tmp_path),
                           env={"PATH": base["PATH"], "RUN_OUTCOME": "success"}, timeout=30)   # 沒有secrets → 略過
        assert p.returncode == 0 and "略過" in p.stdout

    def test_commit_only_leader_files_with_retries(self):
        _, wf = self._wf()
        step = self._step(wf, "Commit signals back to repo")
        run = step["run"]
        assert "steps.run.outcome == 'success'" in step["if"]
        assert "git pull --rebase" in run and "for i in" in run and "git push" in run
        add = run[run.index("git add"):run.index("if git diff")]
        assert "signals/leader_latest.md" in add and "signals/telegram_leader.txt" in add
        assert "signals/leader_paper_log.csv" in add and "signals/leader_trades_paper.csv" in add
        assert "signals/leader_history/" in add and "git add signals/\n" not in run

    def test_upload_and_mark_failure(self):
        _, wf = self._wf()
        steps = self._job(wf)["steps"]
        up = next(st for st in steps if "upload-artifact" in st.get("uses", ""))
        assert up["if"] == "always()" and "signals/leader_*" in up["with"]["path"]
        last = steps[-1]
        assert "steps.run.outcome != 'success'" in last["if"] and "exit 1" in last["run"]
