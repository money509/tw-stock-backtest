"""
test_tx_daily_data.py —— tx_daily_data.py的離線測試(不連Google Drive/Yahoo)。
合成資料：加權指數隨機漫步 + 期貨 = 指數 + 近月合約基差；7/8月結算的合約有150點
逆價差(模擬除息季)，其他月份0 → 已知的換月日/價差，用來驗證換月偵測與Panama調整。
"""
import os

import numpy as np
import pandas as pd
import pytest

import tx_daily_data as tdd
from settlement_calendar import third_wednesday

DIV_SPREAD = 150.0


# ---------------------------------------------------------------------------
# 合成資料產生器(其他測試檔也會import)
# ---------------------------------------------------------------------------
def make_index(start="1998-01-05", end="2026-09-30", seed=0, base=6000.0):
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.012, len(dates))
    close = base * np.exp(np.cumsum(rets))
    prev = np.concatenate([[base], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.003, len(dates)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, len(dates))))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=pd.DatetimeIndex(dates, name="Date"))


def _contract_basis(d: pd.Timestamp, settle_pos_month):
    """回傳當天近月合約的「到期月」與該合約基差。到期月 = 當月(若還沒過結算日)或下個月。"""
    tw = pd.Timestamp(third_wednesday(d.year, d.month))
    if d <= tw:
        y, m = d.year, d.month
    else:
        y, m = (d.year + (d.month == 12), 1 if d.month == 12 else d.month + 1)
    return (y, m), (-DIV_SPREAD if m in (7, 8) else 0.0)


def make_futures_daily_truth(index_df, start="2001-01-01", end="2023-12-31", noise=2.0, seed=1):
    """回傳(真實期貨日線[Open/High/Low/Close/Close1330]，已知換月日{date: gap})。
    換月日 = 第一個 > 第三個星期三 的交易日(結算日當天仍是舊合約)。"""
    rng = np.random.default_rng(seed)
    idx = index_df.loc[start:end]
    rows, contracts = [], []
    for d, r in idx.iterrows():
        (cm, b) = _contract_basis(d, None)
        contracts.append(cm)
        nb = rng.normal(0, noise)
        o = r["Open"] + b + nb
        c1330 = r["Close"] + b + rng.normal(0, noise)
        c = c1330 + rng.normal(0, 3)
        hi = max(r["High"] + b + nb, o, c, c1330)
        lo = min(r["Low"] + b + nb, o, c, c1330)
        rows.append((o, hi, lo, c, c1330))
    fut = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Close1330"], index=idx.index)
    rolls = {}
    for i in range(1, len(contracts)):
        if contracts[i] != contracts[i - 1]:
            _, b_new = _contract_basis(idx.index[i], None)
            _, b_old = _contract_basis(idx.index[i - 1], None)
            rolls[idx.index[i]] = b_new - b_old
    return fut, rolls


def daily_to_minutes(fut_daily, night=True):
    """每天產生幾根「分鐘」K棒(08:46/10:00/12:00/13:30/13:45)，2017-05-15以後另外加
    夜盤K棒(15:01/23:00/隔天02:00)，夜盤價格故意偏離+500點，用來確認會被丟掉。"""
    recs = []
    for d, r in fut_daily.iterrows():
        o, h, l, c, c1330 = r["Open"], r["High"], r["Low"], r["Close"], r["Close1330"]
        mid = (o + c1330) / 2
        pts = [
            ("08:46", o, max(o, mid), min(o, mid), mid),
            ("10:00", mid, h, mid, mid),
            ("12:00", mid, mid, l, mid),
            ("13:30", mid, max(mid, c1330), min(mid, c1330), c1330),
            ("13:45", c1330, max(c1330, c), min(c1330, c), c),
        ]
        for t, oo, hh, ll, cc in pts:
            recs.append((pd.Timestamp(f"{d.date()} {t}"), oo, hh, ll, cc, 10.0))
        if night and d >= pd.Timestamp("2017-05-15"):
            for t in ("15:01", "23:00"):
                recs.append((pd.Timestamp(f"{d.date()} {t}"), c + 500, c + 510, c + 490, c + 500, 1.0))
            nd = d + pd.Timedelta(days=1)
            recs.append((pd.Timestamp(f"{nd.date()} 02:00"), c + 500, c + 510, c + 490, c + 500, 1.0))
    df = pd.DataFrame(recs, columns=["BarStart", "Open", "High", "Low", "Close", "Volume"]).set_index("BarStart")
    return df.sort_index()


def raw_futures_from_truth(fut_daily, rolls):
    """真實期貨日線本身就是「未調整」序列(換月跳價)。"""
    return fut_daily


def make_yf_frame(index_df, multiindex=True):
    df = index_df.copy()
    df["Adj Close"] = df["Close"]
    df["Volume"] = 0
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["^TWII"]])
    return df


@pytest.fixture(scope="module")
def synth():
    index_df = make_index()
    fut, rolls = make_futures_daily_truth(index_df)
    return index_df, fut, rolls


# ---------------------------------------------------------------------------
class TestSessions:
    def test_night_rows_skipped_and_counted(self):
        idx = make_index("2017-05-10", "2017-05-31")
        fut, _ = make_futures_daily_truth(idx, "2017-05-10", "2017-05-31")
        bars = daily_to_minutes(fut)
        daily, stats = tdd.minute_bars_to_daily(bars)
        assert stats["rows_night"] > 0
        assert stats["days_with_night_rows"] > 0
        assert stats["rows_day"] == 5 * len(fut)
        # 夜盤+500點的價格完全沒有進到日線
        np.testing.assert_allclose(daily["High"].to_numpy(), fut["High"].to_numpy())
        np.testing.assert_allclose(daily["Close"].to_numpy(), fut["Close"].to_numpy())
        np.testing.assert_allclose(daily["Close1330"].to_numpy(), fut["Close1330"].to_numpy())
        np.testing.assert_allclose(daily["Open"].to_numpy(), fut["Open"].to_numpy())
        # 週六凌晨02:00的夜盤K棒不會變成一個「交易日」
        assert all(d.weekday() < 5 for d in daily.index)

    def test_classify_boundaries(self):
        ix = pd.DatetimeIndex(["2020-01-02 08:44", "2020-01-02 08:45", "2020-01-02 13:45",
                               "2020-01-02 13:46", "2020-01-02 15:00", "2020-01-03 05:00",
                               "2020-01-03 06:00"])
        assert list(tdd.classify_sessions(ix)) == ["other", "day", "day", "other", "night", "night", "other"]

    def test_empty(self):
        d, s = tdd.minute_bars_to_daily(pd.DataFrame(columns=["Open", "High", "Low", "Close"]))
        assert d.empty and s["rows_total"] == 0


class TestRollAdjustment:
    def test_back_adjustment_known_rolls(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        raw = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
                            "Close1330": 100.0}, index=dates)
        raw.iloc[4:, :] -= 30.0   # 第4天換到新合約，新合約比舊合約低30
        raw.iloc[8:, :] += 10.0   # 第8天再換，高10
        rolls = [{"roll_date": dates[4], "gap": -30.0}, {"roll_date": dates[8], "gap": 10.0}]
        adj = tdd.apply_back_adjustment(raw, rolls)
        # 調整後完全平滑；最後一段不動
        assert np.allclose(adj["Close"], adj["Close"].iloc[-1])
        assert adj["Close"].iloc[-1] == raw["Close"].iloc[-1]
        np.testing.assert_allclose(adj["RawClose"], raw["Close"])
        assert list(adj["Offset"]) == [-20.0] * 4 + [10.0] * 4 + [0.0] * 2
        assert list(adj["RollDay"]) == [False] * 4 + [True] + [False] * 3 + [True, False]
        np.testing.assert_allclose(adj["Close"] - adj["Offset"], adj["RawClose"])

    def test_detect_rolls_finds_known_gaps(self, synth):
        index_df, fut, rolls = synth
        detected, info = tdd.detect_rolls_vs_index(fut, index_df["Close"])
        sig = {r["roll_date"]: r["gap"] for r in detected if r["significant"]}
        true_sig = {d: g for d, g in rolls.items() if abs(g) > 0}
        assert set(sig) == set(true_sig)
        for d, g in true_sig.items():
            assert abs(sig[d] - g) < 15
        # 不顯著的月份價差一律0
        assert all(r["gap"] == 0.0 for r in detected if not r["significant"])
        assert info["n_rolls"] >= 270

    def test_adjusted_series_removes_roll_jumps(self, synth):
        index_df, fut, rolls = synth
        detected, _ = tdd.detect_rolls_vs_index(fut, index_df["Close"])
        adj = tdd.apply_back_adjustment(fut, detected)
        basis_adj = (adj["Close1330"] - index_df["Close"].reindex(adj.index)).diff()
        # 換月日的基差跳動被拿掉(剩雜訊)
        for d in [d for d, g in rolls.items() if g != 0]:
            assert abs(basis_adj.loc[d]) < 15
        assert adj["RawClose"].equals(fut["Close"].astype(float))

    def test_calendar_roll_days(self):
        dates = pd.bdate_range("2023-01-01", "2023-03-31")
        flag = tdd.calendar_roll_days(dates)
        assert list(flag[flag].index.date.astype(str)) == ["2023-01-19", "2023-02-16", "2023-03-16"]


class TestValidation:
    def _adj(self, synth):
        index_df, fut, _ = synth
        detected, _ = tdd.detect_rolls_vs_index(fut, index_df["Close"])
        return tdd.apply_back_adjustment(fut, detected)

    def test_pass(self, synth):
        index_df = synth[0]
        ok, stats, fails = tdd.validate_against_index(self._adj(synth), index_df)
        assert ok, fails
        assert stats["coverage_pct"] == 100.0
        assert stats["return_corr"] > 0.99
        assert stats["median_abs_basis_pct"] < 1

    def test_fail_garbage_prices(self, synth):
        index_df = synth[0]
        adj = self._adj(synth)
        bad = adj.copy()
        rng = np.random.default_rng(5)
        noise = rng.normal(0, 300, len(bad))
        for c in ("Close1330", "RawClose1330"):
            bad[c] = bad[c] * 1.2 + noise
        ok, stats, fails = tdd.validate_against_index(bad, index_df)
        assert not ok
        assert any("相關" in f for f in fails) and any("中位數" in f for f in fails)

    def test_fail_coverage_and_ohlc(self, synth):
        index_df = synth[0]
        adj = self._adj(synth)
        thin = adj.iloc[::3].copy()
        thin["RawHigh"] = thin["RawLow"] - 1
        ok, stats, fails = tdd.validate_against_index(thin, index_df)
        assert not ok
        assert any("覆蓋率" in f for f in fails) and any("OHLC" in f for f in fails)


class TestIndexLoader:
    def test_multiindex_and_cache(self, tmp_path, monkeypatch):
        idx = make_index("2020-01-01", "2020-03-31")
        calls = []

        def fake(symbol, start, end):
            calls.append((symbol, start, end))
            return make_yf_frame(idx)
        monkeypatch.setattr(tdd, "_yf_download", fake)
        df, info = tdd.load_index_ohlc("^TWII", "2020-01-01", "2020-04-01", cache_dir=str(tmp_path))
        assert info["reason"] == "ok" and len(df) == len(idx)
        np.testing.assert_allclose(df["Close"], idx["Close"])
        df2, info2 = tdd.load_index_ohlc("^TWII", "2020-01-01", "2020-04-01", cache_dir=str(tmp_path))
        assert info2["reason"] == "cached" and len(calls) == 1

    def test_repairs_inconsistent_rows(self, tmp_path, monkeypatch):
        idx = make_index("2020-01-01", "2020-01-31")
        raw = make_yf_frame(idx, multiindex=False)
        raw.iloc[3, raw.columns.get_loc("High")] = raw.iloc[3]["Low"] - 5
        raw.iloc[4, raw.columns.get_loc("Open")] = 0.0
        monkeypatch.setattr(tdd, "_yf_download", lambda *a: raw)
        df, info = tdd.load_index_ohlc("^TWII", "2020-01-01", "2020-02-01", cache_dir=str(tmp_path))
        assert info["n_repaired"] >= 1
        assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
        assert df["Open"].iloc[4] == pytest.approx(df["Close"].iloc[3])

    def test_failure_returns_empty(self, tmp_path, monkeypatch):
        def boom(*a):
            raise RuntimeError("no network")
        monkeypatch.setattr(tdd, "_yf_download", boom)
        monkeypatch.setattr(tdd.time, "sleep", lambda s: None)
        df, info = tdd.load_index_ohlc("^TWII", "2020-01-01", "2020-02-01", cache_dir=str(tmp_path))
        assert df.empty and info["reason"].startswith("download_failed")


# ---------------------------------------------------------------------------
def _ds_with(synth, minute_fn):
    index_df = synth[0]
    return tdd.build_tx_daily_dataset(load_minute_fn=minute_fn,
                                      load_index_fn=lambda: (index_df, {"reason": "ok"}))


class TestBuildDataset:
    def test_futures_pass(self, synth):
        index_df, fut, _ = synth
        bars = daily_to_minutes(fut)
        ds = _ds_with(synth, lambda: (bars, {"periods": {k: {"reason": "ok"} for k in tdd.FUTURES_PERIOD_KEYS}}))
        assert ds["source"] == "futures", ds["validation_fails"]
        assert ds["source_line"].startswith("價格來源：台指期")
        assert ds["main"].index.min() >= pd.Timestamp("2001-01-01")
        assert ds["main"].index.max() <= pd.Timestamp("2023-12-31")
        assert ds["session_stats"]["rows_night"] > 0
        assert ds["index"]["RollDay"].any()
        assert {"Offset", "RollDay", "RawClose"} <= set(ds["main"].columns)

    def test_validation_fail_falls_back_to_index(self, synth):
        index_df, fut, _ = synth
        bad = fut.copy() * 1.5
        bars = daily_to_minutes(bad)
        ds = _ds_with(synth, lambda: (bars, {"periods": {}}))
        assert ds["source"] == "index"
        assert ds["source_line"].startswith("價格來源：加權指數(期貨資料驗證失敗")
        assert ds["main"] is ds["index"]
        assert ds["main"].index.min() < pd.Timestamp("1999-01-01")  # 整段1998~最新
        assert ds["main"].index.max() > pd.Timestamp("2026-01-01")

    def test_download_fail_falls_back(self, synth):
        ds = _ds_with(synth, lambda: (None, {"periods": {"2001_2010": {"reason": "download_timeout"}}}))
        assert ds["source"] == "index" and "download_timeout" in ds["source_line"]

    def test_loader_exception_falls_back(self, synth):
        def boom():
            raise ValueError("parse exploded")
        ds = _ds_with(synth, boom)
        assert ds["source"] == "index" and "parse exploded" in ds["fallback_reason"]

    def test_index_fail_no_data(self):
        ds = tdd.build_tx_daily_dataset(load_minute_fn=lambda: (None, {}),
                                        load_index_fn=lambda: (pd.DataFrame(), {"reason": "download_empty"}))
        assert ds["main"] is None and "無法執行" in ds["source_line"]

    def test_default_loader_offline_gdown_blocked(self, synth, tmp_path, monkeypatch):
        """真正走taifex_history_loader的下載路徑，但gdown被擋(不連網) → 退回加權指數。"""
        import taifex_history_loader as thl

        def no_net(file_id, output_path):
            raise RuntimeError("network blocked in test")
        monkeypatch.setattr(thl, "_gdown_download", no_net)
        monkeypatch.setattr(thl, "RETRY_BACKOFF_SECONDS", 0)
        index_df = synth[0]
        ds = tdd.build_tx_daily_dataset(futures_cache_dir=str(tmp_path / "c"),
                                        load_index_fn=lambda: (index_df, {"reason": "ok"}))
        assert ds["source"] == "index"
        assert "下載/解析失敗" in ds["fallback_reason"]

    def test_diagnostics_written(self, synth, tmp_path):
        index_df, fut, _ = synth
        bars = daily_to_minutes(fut)
        ds = _ds_with(synth, lambda: (bars, {"periods": {}}))
        files = tdd.write_diagnostics(ds, str(tmp_path))
        names = {os.path.basename(f) for f in files}
        assert {"data_parse_report.txt", "data_validation.json", "tx_daily_bars_sample.csv",
                "tx_roll_adjustments.csv"} <= names
        rep = (tmp_path / "data_parse_report.txt").read_text(encoding="utf-8")
        assert "日盤/夜盤統計" in rep and "換月" in rep
        sample = pd.read_csv(tmp_path / "tx_daily_bars_sample.csv", encoding="utf-8-sig")
        assert "收(未調整)" in sample.columns and "加權指數收盤" in sample.columns


class TestDescribeRawFormat:
    def test_plain_csv(self, tmp_path):
        p = tmp_path / "x.raw"
        p.write_bytes(b"Date,Time,Open,High,Low,Close,Volume\r\n"
                      + b"".join(f"2011/01/03,08:{46 + i:02d}:00,9000,9008,8995,9006,1340\r\n".encode()
                                 for i in range(10)))
        out = tdd.describe_raw_format(str(p))
        assert out["detected"]["date"] == "Date" and out["detected"]["time"] == "Time"
        # 成交量1340落在價格範圍內也會被當成「價格欄」(loader只取前4個當OHLC，見taifex_history_loader)
        assert out["detected"]["price_cols"][:4] == ["Open", "High", "Low", "Close"]
        assert out["contract_like_columns"] == []
