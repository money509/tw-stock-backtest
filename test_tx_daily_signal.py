"""
test_tx_daily_signal.py —— 台指期(小台)每日實盤訊號(tx_daily_signal.py)離線測試(合成資料、不連網)：
凍結設定 = --refine的變體、重播交易 = 引擎交易、只做多、4種狀態(進場訊號/持有且停損上移/出場條件/今天停損出場)、
移動停損只上移且等於引擎隔天用的停損、不偷看未來、資料沒更新 → 警告且沒有進場指示、紙上紀錄冪等/PAPER_START、
Telegram長度、沒有scipy也能跑、workflow接線。
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import tx_daily_data as tdd
import tx_daily_engine as eng
import tx_daily_refine as rf
import tx_daily_signal as s
from test_tx_daily_data import make_index, make_yf_frame

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def _no_step_summary(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("TX_SIGNALS_LINK", raising=False)


# ---------------------------------------------------------------------------
# 合成資料
# ---------------------------------------------------------------------------
def _ohlc(closes, index, half, low_extra=None):
    c = np.asarray(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) + half
    l = np.minimum(o, c) - half
    if low_extra is not None:
        l = l - low_extra
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=index)


def build(n_pre=150, jump=500.0, n_up=80, up=15.0, half=20.0, drop_at=None, drop=0.0, low_extra_at=None,
          low_extra=0.0):
    """慢慢下跌150天(沒有任何訊號) → 第J天跳漲(MA5上穿MA20 + 突破20日高同一天 → 進場訊號)
    → 每天+15穩定上漲80天(移動停損每天上移、第60天觸發最長持有)。
    drop_at：從J+drop_at起整段下移drop點(製造反向訊號或停損)；low_extra_at：那天盤中最低點再往下low_extra點。"""
    closes = [10000.0 - 3 * i for i in range(n_pre)]
    closes.append(closes[-1] + jump)
    for _ in range(n_up):
        closes.append(closes[-1] + up)
    c = np.array(closes)
    if drop_at is not None:
        c[n_pre + drop_at:] -= drop
    idx = pd.bdate_range("2024-01-01", periods=len(c))
    extra = None
    if low_extra_at is not None:
        extra = np.where(np.arange(len(c)) == n_pre + low_extra_at, low_extra, 0.0)
    return _ohlc(c, idx, half, extra), n_pre


def day(df, J, k):
    return df.index[J + k]


def _norm(trades):
    """trade dict比較用：NaN(移動停損沒有停利價)換成None，NaN != NaN。"""
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in t.items()} for t in trades]


@pytest.fixture(scope="module")
def rand_index():
    return make_index("2023-01-02", "2026-10-09", seed=3, base=17000.0)


# ---------------------------------------------------------------------------
class TestSettings:
    def test_matches_refine_variant(self):
        v = s.live_variant()
        st = s.TX_LIVE_SETTINGS
        assert st["name"] == "MA_5_20&DONCHIAN20｜只做多｜停損1.0/移動3.0" == v["name"]
        assert v["rule"] == st["rule"] == "MA_5_20&DONCHIAN20"
        assert v["components"] == st["components"] == ("MA_5_20", "DONCHIAN20")
        assert v["direction"] == st["direction"] == "long"
        assert (v["stop_atr"], v["exit_kind"], v["exit_mult"]) == (st["stop_atr"], st["exit_kind"], st["exit_mult"]) \
            == (1.0, "trail", 3.0)
        assert v["macd"] is None and not v.get("gate") and st["gate"] is False
        assert st["pair_window_days"] == eng.PAIR_WINDOW_DAYS == 3
        assert st["max_hold_days"] == eng.MAX_HOLD_DAYS == 60
        assert st["atr_period"] == eng.ATR_PERIOD == 14
        assert st["contract"] == "mini" and eng.CONTRACT_MULTIPLIER["mini"] == 50
        assert s.REPLAY_START == "2024-01-01" and s.PAPER_START == "2026-10-12"

    def test_frozen_comment_in_source(self):
        with open(s.__file__, encoding="utf-8") as f:
            src = f.read()
        assert "2026-10" in src and "每年1月重跑一次 --refine" in src

    def test_mismatch_is_rejected(self, monkeypatch):
        monkeypatch.setitem(s.TX_LIVE_SETTINGS, "stop_atr", 1.5)
        with pytest.raises(ValueError):
            s.live_variant()


def _engine_trades_refine_path(frame, end):
    """--refine的原路徑：compute_refine_indicators + refine_entry_signals + 網格引擎(期末強制平倉)。"""
    v = next(x for x in rf.refine_variant_list() if x["name"] == s.TX_LIVE_SETTINGS["name"])
    arrs = eng.prepare_grid_arrays(frame, rf.compute_refine_indicators(frame))
    es, rsig = rf.refine_entry_signals(arrs, v)
    return eng.run_tx_daily_grid_backtest(arrs, es, rsig, v["stop_atr"], v["exit_kind"], v["exit_mult"],
                                          start=s.REPLAY_START, end=end, variant_name=v["name"])


class TestReplayEqualsEngine:
    @pytest.mark.parametrize("cut", [-1, -40, -123, -301])
    def test_trades_equal(self, rand_index, cut):
        df = rand_index.iloc[:len(rand_index) + cut + 1] if cut != -1 else rand_index
        frame = s.engine_frame(df)
        rp = s.replay(frame)
        ref = _engine_trades_refine_path(frame, frame.index[-1])
        pos = rp["state"]["position"]
        if pos is not None:
            assert ref[-1]["exit_reason"] == "期末平倉"
            assert ref[-1]["entry_date"] == pos["entry_date"]
            assert ref[-1]["entry_price"] == pos["entry_price"] and ref[-1]["initial_stop"] == pos["initial_stop"]
            ref = ref[:-1]
        assert len(rp["trades"]) == len(ref) > 0
        assert _norm(rp["trades"]) == _norm(ref)

    def test_long_only(self, rand_index):
        for seed in range(4):
            df = make_index("2023-01-02", "2026-10-09", seed=seed, base=17000.0)
            rp = s.replay(s.engine_frame(df))
            assert rp["trades"] and all(t["side"] == "long" for t in rp["trades"])
            assert rp["state"]["pending_entry"] in (0, 1)
            if rp["state"]["position"] is not None:
                assert rp["state"]["position"]["dir"] == 1
        # 原始規則確實有空訊號(只是不進場)
        frame = s.engine_frame(rand_index)
        arrs = eng.prepare_grid_arrays(frame, eng.compute_grid_indicators(frame))
        assert (arrs["ind"]["sig_MA_5_20&DONCHIAN20"] == -1).any()

    def test_engine_default_unchanged_by_state_out(self, rand_index):
        """state_out=None(預設)時跟以前一樣期末強制平倉；有state_out時只差最後那一筆。"""
        frame = s.engine_frame(rand_index)
        v = s.live_variant()
        arrs = eng.prepare_grid_arrays(frame, eng.compute_grid_indicators(frame))
        es, rsig = rf.refine_entry_signals(arrs, v)
        a = eng.run_tx_daily_grid_backtest(arrs, es, rsig, 1.0, "trail", 3.0, start="2024-01-01")
        st = {}
        b = eng.run_tx_daily_grid_backtest(arrs, es, rsig, 1.0, "trail", 3.0, start="2024-01-01", state_out=st)
        if st["position"] is None:
            assert _norm(a) == _norm(b)
        else:
            assert _norm(a[:-1]) == _norm(b) and a[-1]["exit_reason"] == "期末平倉"


# ---------------------------------------------------------------------------
class TestStatuses:
    def test_entry_signal_today(self):
        df, J = build()
        r = s.evaluate(df, day(df, J, 0))
        assert r["status"] == "entry" and r["data_fresh"]
        e = r["entry_plan"]
        assert e["stop_distance"] == pytest.approx(1.0 * r["atr"])
        assert e["approx_stop"] == pytest.approx(r["close"] - e["stop_distance"])
        md, tg = s.render_markdown(r), s.render_telegram(r)
        assert "開盤市價買進小台1口" in md and "停損距離" in md and "移動停損" in md
        assert "開盤市價買進小台1口" in tg
        # 隔天實際進場：引擎的初始停損 = 進場價 − 同一個停損距離
        nxt = s.evaluate(df, day(df, J, 1))
        p = nxt["position"]
        assert nxt["status"] == "hold" and p["entry_date"] == day(df, J, 1)
        assert p["entry_price"] - p["initial_stop"] == pytest.approx(e["stop_distance"])
        assert p["entry_price"] == pytest.approx(df["Open"].iloc[J + 1] + 1.0)  # 開盤+1點滑價

    def test_flat_no_signal(self):
        df, J = build()
        r = s.evaluate(df, day(df, J, -5))
        assert r["status"] == "flat" and r["position"] is None and r["entry_plan"] is None
        assert "不用下單" in s.render_markdown(r) and "買進" not in s.render_telegram(r)

    def test_holding_with_stop_moved_up(self):
        df, J = build()
        r = s.evaluate(df, day(df, J, 20))
        p = r["position"]
        assert r["status"] == "hold"
        assert p["stop_moved_up"] and p["stop"] > p["prev_stop"]
        assert p["prev_stop"] == pytest.approx(s.evaluate(df, day(df, J, 19))["position"]["stop"])
        assert p["days_held"] == 20
        assert p["best_close"] == pytest.approx(df["Close"].iloc[J + 20])
        assert p["unrealized_ntd"] == pytest.approx((r["close"] - p["entry_price"]) * 50)
        assert p["stop"] == pytest.approx(max(p["prev_stop"], p["best_close"] - 3.0 * r["atr"]))
        md = s.render_markdown(r)
        assert "停損單掛在這個價位" in md and "今天上移" in md and "20 天 / 60" in md
        assert "紙上交易開始" in md and "不要追" in s.render_telegram(r)  # 2024的部位 < PAPER_START
        early = s.evaluate(df, day(df, J, 5))["position"]
        assert not early["stop_moved_up"] and "今天沒有變動" in s.render_markdown(s.evaluate(df, day(df, J, 5)))

    def test_exit_by_max_hold(self):
        df, J = build()
        r59 = s.evaluate(df, day(df, J, 59))
        r = s.evaluate(df, day(df, J, 60))
        assert r59["status"] == "hold"
        assert r["status"] == "exit" and r["exit_reason"] == "最長持有" and r["position"]["days_held"] == 60
        assert "明天開盤平倉" in s.render_telegram(r) and "最長持有" in s.render_markdown(r)
        nxt = s.evaluate(df, day(df, J, 61))
        assert nxt["exits_today"] and nxt["exits_today"][0]["exit_reason"] == "最長持有"

    def test_exit_by_opposite_signal(self):
        df, J = build(half=200.0, drop_at=45, drop=800.0)
        r = s.evaluate(df, day(df, J, 45))
        assert r["status"] == "exit" and r["exit_reason"] == "反向訊號"
        frame = s.engine_frame(df)
        arrs = eng.prepare_grid_arrays(frame, eng.compute_grid_indicators(frame))
        assert arrs["ind"]["sig_MA_5_20&DONCHIAN20"].iloc[J + 45] == -1  # 原始空訊號
        nxt = s.evaluate(df, day(df, J, 46))
        assert nxt["status"] == "flat"
        t = nxt["exits_today"][0]
        assert t["exit_reason"] == "反向訊號" and t["exit_price"] == pytest.approx(df["Open"].iloc[J + 46] - 1.0)
        assert "最近一次出場(今天)：反向訊號" in s.render_markdown(nxt)

    def test_stopped_out_today(self):
        df, J = build(drop_at=25, drop=0.0, low_extra_at=25, low_extra=400.0)
        before = s.evaluate(df, day(df, J, 24))
        r = s.evaluate(df, day(df, J, 25))
        assert before["status"] == "hold"
        assert r["status"] == "flat" and r["position"] is None
        t = r["exits_today"][0]
        assert t["exit_reason"] == "移動停損"
        assert t["exit_price"] == pytest.approx(before["position"]["stop"] - 1.0)
        md, tg = s.render_markdown(r), s.render_telegram(r)
        assert "最近一次出場(今天)：移動停損" in md and "最近一次出場(今天)：移動停損" in tg


# ---------------------------------------------------------------------------
class TestTrailingStop:
    def _next_day_engine(self, df, i, low):
        """把第i+1天的K棒換成：開盤高於停損、盤中最低=low，再用引擎(期末強制平倉)跑到i+1。"""
        d2 = df.iloc[:i + 2].copy()
        prev_c = d2["Close"].iloc[i]
        d2.iloc[i + 1] = [prev_c, prev_c + 10, low, prev_c]
        return _engine_trades_refine_path(s.engine_frame(d2), d2.index[-1])

    @pytest.mark.parametrize("k", [5, 12, 20, 40])
    def test_replay_stop_is_engine_next_day_stop(self, k):
        df, J = build()
        i = J + k
        p = s.evaluate(df, df.index[i])["position"]
        stop = p["stop"]
        hit = self._next_day_engine(df, i, stop - 0.5)
        assert hit[-1]["exit_date"] == df.index[i + 1]
        assert hit[-1]["exit_reason"] == ("停損" if stop == p["initial_stop"] else "移動停損")
        assert hit[-1]["exit_price"] == pytest.approx(stop - 1.0)
        miss = self._next_day_engine(df, i, stop + 0.5)
        assert miss[-1]["exit_reason"] == "期末平倉"

    def test_never_decreases_random(self, rand_index):
        frame = s.engine_frame(rand_index)
        prev = None
        n_checked = 0
        for d in rand_index.index[-260:]:
            p = s.replay(frame, end=d)["state"]["position"]
            if p is not None and prev is not None and prev["entry_date"] == p["entry_date"]:
                assert p["stop"] >= prev["stop"] - 1e-9
                n_checked += 1
            prev = p
        assert n_checked > 10

    def test_random_stop_matches_engine_next_day(self, rand_index):
        """隨機資料：每個持倉日的重播停損，隔天如果盤中最低<停損而開盤>停損 → 引擎在停損價出場。"""
        frame = s.engine_frame(rand_index)
        n = 0
        for i in range(len(rand_index) - 200, len(rand_index) - 1, 7):
            p = s.replay(frame, end=rand_index.index[i])["state"]["position"]
            if p is None:
                continue
            if s.replay(frame, end=rand_index.index[i])["state"]["pending_exit"]:
                continue
            hit = self._next_day_engine(rand_index, i, p["stop"] - 0.5)
            assert hit[-1]["exit_date"] == rand_index.index[i + 1]
            assert hit[-1]["exit_price"] == pytest.approx(p["stop"] - 1.0)
            n += 1
        assert n >= 1


class TestNoLookahead:
    def test_appending_bars_does_not_change_result(self, rand_index):
        for d in rand_index.index[-200::23]:
            full = s.evaluate(rand_index, d)
            cut = s.evaluate(rand_index[rand_index.index <= d], d)
            assert s.render_markdown(full) == s.render_markdown(cut)
            assert s.render_telegram(full) == s.render_telegram(cut)
            assert s.paper_log_row(full) == s.paper_log_row(cut)

    def test_constructed_days(self):
        df, J = build(half=200.0, drop_at=45, drop=800.0)
        for k in (0, 20, 45, 46):
            d = day(df, J, k)
            assert s.render_markdown(s.evaluate(df, d)) == s.render_markdown(s.evaluate(df[df.index <= d], d))


# ---------------------------------------------------------------------------
class TestStale:
    def test_last_bar_older_than_as_of(self):
        df, J = build()
        d = day(df, J, 0)                      # 這天有進場訊號
        nxt = df.index[J + 1]
        r = s.evaluate(df[df.index <= d], nxt)  # 隔天的資料還沒出來
        assert r["status"] == "stale" and not r["data_fresh"] and r["entry_plan"] is None
        md, tg = s.render_markdown(r), s.render_telegram(r)
        assert "⚠️ 資料還沒更新" in md and "⚠️ 資料還沒更新" in tg
        assert "買進" not in md.split("## 規則重點")[0] and "買進" not in tg
        assert s.paper_log_row(r)["status"] == "資料未更新"

    def test_stale_while_holding_shows_reference_stop_only(self):
        df, J = build()
        d = day(df, J, 20)
        r = s.evaluate(df[df.index <= d], df.index[J + 21])
        assert r["status"] == "stale"
        assert "僅供參考" in s.render_markdown(r) and "停損" in s.render_telegram(r)
        assert "開盤市價" not in s.render_telegram(r)

    def test_weekend_as_of_is_stale(self):
        df, J = build()
        sat = df.index[J + 20] + pd.offsets.Day(1)
        while sat.weekday() != 5:
            sat += pd.offsets.Day(1)
        assert s.evaluate(df[df.index < sat], sat)["status"] == "stale"

    def test_download_failed(self, tmp_path):
        r = s.evaluate(pd.DataFrame(columns=["Open", "High", "Low", "Close"]), "2026-10-12",
                       {"reason": "download_empty"})
        assert r["status"] == "stale" and "download_empty" in r["stale_reason"]
        out = s.write_outputs(r, str(tmp_path))
        assert "⚠️ 資料還沒更新" in out["markdown"] and (tmp_path / "telegram_tx.txt").exists()

    def test_cli_download_failure_still_writes_files(self, monkeypatch, tmp_path):
        def boom(*a, **k):
            raise ConnectionError("no network in test")
        monkeypatch.setattr(tdd, "_yf_download", boom)
        monkeypatch.setattr(tdd.time, "sleep", lambda *_: None)
        out = s.main(["--output-dir", str(tmp_path), "--as-of", "2026-10-12"])
        assert out["result"]["status"] == "stale"
        tg = (tmp_path / "telegram_tx.txt").read_text(encoding="utf-8")
        assert tg.startswith("【台指期訊號】2026-10-12") and "⚠️ 資料還沒更新" in tg
        assert (tmp_path / "tx_history" / "2026-10-12.md").exists()


# ---------------------------------------------------------------------------
class TestPaperLog:
    def test_idempotent_per_date(self, tmp_path):
        df, J = build()
        for k in (20, 21, 20, 20):
            s.write_outputs(s.evaluate(df, day(df, J, k)), str(tmp_path))
        log = pd.read_csv(tmp_path / "tx_paper_log.csv", dtype=str, encoding="utf-8-sig")
        assert list(log.columns) == s.PAPER_LOG_COLUMNS
        assert list(log["date"]) == [day(df, J, 20).date().isoformat(), day(df, J, 21).date().isoformat()]
        row = log.iloc[0]
        r = s.evaluate(df, day(df, J, 20))
        assert row["status"] == "持有多單" and float(row["stop"]) == pytest.approx(round(r["position"]["stop"], 1))
        assert float(row["close"]) == pytest.approx(round(r["close"], 1)) and row["position"].startswith("多1口@")
        # 用舊日期重跑一次 → 依日期排序、不重複
        s.write_outputs(s.evaluate(df, day(df, J, 5)), str(tmp_path))
        log = pd.read_csv(tmp_path / "tx_paper_log.csv", dtype=str, encoding="utf-8-sig")
        assert list(log["date"]) == sorted(log["date"]) and len(log) == 3
        assert (tmp_path / "tx_history" / f"{day(df, J, 21).date()}.md").exists()
        assert (tmp_path / "tx_latest.md").read_text(encoding="utf-8").startswith(
            f"# 台指期(小台)每日訊號 {day(df, J, 5).date()}")

    def test_paper_start_filter(self, rand_index, monkeypatch):
        d = rand_index.index[-1]
        all_trades = s.replay(s.engine_frame(rand_index))["trades"]
        assert len(all_trades) >= 4
        cut = pd.Timestamp(all_trades[len(all_trades) // 2]["entry_date"])
        monkeypatch.setattr(s, "PAPER_START", cut.date().isoformat())
        r = s.evaluate(rand_index, d)
        exp = [t for t in all_trades if pd.Timestamp(t["entry_date"]) >= cut]
        assert _norm(r["paper_trades"]) == _norm(exp) and 0 < len(exp) < len(all_trades)
        assert r["paper_realized_ntd"] == pytest.approx(sum(t["pnl_ntd"] for t in exp))
        assert r["ref"]["n"] == len(all_trades)
        md = s.render_markdown(r)
        assert "回測參考，不是實際紀錄" in md and f"進場日 ≥ {cut.date()}" in md

    def test_default_paper_start_has_no_trades_before(self):
        df, J = build()  # 全部在2024
        r = s.evaluate(df, day(df, J, 70))
        assert r["paper_trades"] == [] and r["paper_open"] is None
        assert f"紙上交易從 {s.PAPER_START} 開始，目前還沒有交易" in s.render_markdown(r)

    def test_costs_same_as_backtest(self):
        df, J = build()
        t = s.evaluate(df, day(df, J, 62))["exits_today"] or s.replay(s.engine_frame(df))["trades"]
        t = t[0]
        exp_cost = 2 * 50 + (abs(t["entry_price"]) + abs(t["exit_price"])) * 50 * eng.FUTURES_TAX_RATE + t["roll_cost_ntd"]
        assert t["cost_ntd"] == pytest.approx(exp_cost)


# ---------------------------------------------------------------------------
class TestRendering:
    @pytest.mark.parametrize("k,status", [(-5, "flat"), (0, "entry"), (20, "hold"), (60, "exit")])
    def test_telegram_short_with_date_and_status(self, k, status):
        df, J = build()
        r = s.evaluate(df, day(df, J, k))
        assert r["status"] == status
        tg = s.render_telegram(r, link="https://github.com/x/y/blob/main/signals/tx_latest.md")
        first, second = tg.split("\n")[:2]
        assert first == f"【台指期訊號】{s.fmt_date(day(df, J, k))}"
        assert second == f"狀態：{s.STATUS_LABELS[status]}"
        assert len(tg) < 1500
        md = s.render_markdown(r)
        assert s.CAPITAL_NOTE in md and "NT$400,000" in md and "NT$18萬" in md
        assert "小台實際價格 = 指數 + 基差" in md and "停損距離(點)" in md
        assert s.TX_LIVE_SETTINGS["name"] in md

    def test_telegram_truncation(self, monkeypatch):
        df, J = build()
        r = s.evaluate(df, day(df, J, 20))
        r["exits_today"] = [dict(entry_date=day(df, J, 1), exit_date=day(df, J, 20), entry_price=1.0, exit_price=2.0,
                                 points=1.0, pnl_ntd=1.0, exit_reason="測試" * 30)] * 30
        assert len(s.render_telegram(r)) < 1500


class TestWithoutScipy:
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
            import tx_daily_data as tdd
            from test_tx_daily_data import make_index, make_yf_frame
            df = make_index("2023-01-02", "2026-10-12", seed=5, base=17000.0)
            tdd._yf_download = lambda symbol, start, end: make_yf_frame(df[(df.index >= start) & (df.index < end)])
            import tx_daily_signal as s
            out = s.main(["--output-dir", {str(tmp_path)!r}, "--as-of", "2026-10-12"])
            assert out["result"]["data_fresh"], out["result"]["stale_reason"]
            assert "scipy" not in sys.modules or sys.modules["scipy"] is None
            print("OK", out["result"]["status"])
        """)
        env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_STEP_SUMMARY", "TX_SIGNALS_LINK")}
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_DIR, env=env)
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout
        for name in ("tx_latest.md", "telegram_tx.txt", "tx_paper_log.csv", "tx_history/2026-10-12.md"):
            assert (tmp_path / name).exists(), name

    def test_module_does_not_import_scipy(self):
        with open(s.__file__, encoding="utf-8") as f:
            src = f.read()
        assert "import scipy" not in src and "from scipy" not in src


# ---------------------------------------------------------------------------
class TestWorkflow:
    WF_PATH = os.path.join(REPO_DIR, ".github", "workflows", "daily_squeeze_signals.yml")

    def _wf(self):
        yaml = pytest.importorskip("yaml")
        with open(self.WF_PATH, encoding="utf-8") as f:
            text = f.read()
        return text, yaml.safe_load(text)

    def _step(self, wf, name):
        return next(st for st in wf["jobs"]["scan"]["steps"] if st.get("name") == name)

    def test_preflight_includes_new_test(self):
        _, wf = self._wf()
        run = self._step(wf, "Pre-flight test")["run"]
        assert "test_tx_daily_signal.py" in run and "test_daily_squeeze_signals.py" in run

    def test_tx_step_after_scan_and_continue_on_error(self):
        _, wf = self._wf()
        names = [st.get("name") for st in wf["jobs"]["scan"]["steps"]]
        assert names.index("Scan TX signal") == names.index("Scan signals") + 1
        assert names.index("Send Telegram messages") > names.index("Scan TX signal")
        step = self._step(wf, "Scan TX signal")
        assert step["continue-on-error"] is True
        assert "python tx_daily_signal.py" in step["run"] and "--output-dir signals" in step["run"]

    @pytest.mark.parametrize("as_of_env", ["", "2026-10-12"])
    def test_tx_step_passes_as_of(self, as_of_env, tmp_path):
        _, wf = self._wf()
        step = self._step(wf, "Scan TX signal")
        script = step["run"].replace("python tx_daily_signal.py", "echo")
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=str(tmp_path),
                              env={"PATH": os.environ.get("PATH", ""), "AS_OF": as_of_env}, timeout=30)
        assert proc.returncode == 0, proc.stderr
        args = proc.stdout.split()
        if as_of_env:
            assert args[args.index("--as-of") + 1] == as_of_env
        else:
            assert "--as-of" not in args

    def test_telegram_step_sends_three_files_and_failure_message(self):
        text, wf = self._wf()
        run = self._step(wf, "Send Telegram messages")["run"]
        for f in ("signals/telegram_futures.txt", "signals/telegram_stock.txt", "signals/telegram_tx.txt"):
            assert f in run
        assert "台指期訊號產生失敗，請看Actions紀錄" in run
        assert "telegram_tx.txt" in text.split("on:")[0]  # 檔頭註解有提到第三則

    def test_telegram_failure_fallback_script(self, tmp_path):
        """把Telegram步驟用假的curl跑：telegram_tx.txt不存在 → 改送失敗訊息；前兩則照舊送。"""
        _, wf = self._wf()
        run = self._step(wf, "Send Telegram messages")["run"]
        (tmp_path / "signals").mkdir()
        (tmp_path / "signals" / "telegram_futures.txt").write_text("期貨", encoding="utf-8")
        (tmp_path / "signals" / "telegram_stock.txt").write_text("現股", encoding="utf-8")
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
        env = {"PATH": f"{bindir}:{os.environ.get('PATH', '')}", "TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c",
               "TX_OUTCOME": "failure"}
        proc = subprocess.run(["bash", "-c", run], capture_output=True, text=True, cwd=str(tmp_path), env=env,
                              timeout=30)
        sent = log.read_text(encoding="utf-8")
        assert "FILE:期貨" in sent and "FILE:現股" in sent
        assert "台指期訊號產生失敗，請看Actions紀錄" in sent
        # 檔案存在且步驟成功 → 送檔案內容
        (tmp_path / "signals" / "telegram_tx.txt").write_text("【台指期訊號】OK", encoding="utf-8")
        log.unlink()
        env["TX_OUTCOME"] = "success"
        proc = subprocess.run(["bash", "-c", run], capture_output=True, text=True, cwd=str(tmp_path), env=env,
                              timeout=30)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        sent = log.read_text(encoding="utf-8")
        assert "FILE:【台指期訊號】OK" in sent and "產生失敗" not in sent
