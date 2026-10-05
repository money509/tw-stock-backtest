"""daily_squeeze_signals.py(每日訊號掃描器)的測試。全部離線：資料下載一律monkeypatch。

最重要的是 TestConsistencyWithBacktest：在長段合成資料上，對每一個訊號日d，用「只看得到d以前
(含d)、而且只抓預設回看窗口」的資料跑掃描器，結果必須跟回測(precompute_squeeze_kdj_backtest_arrays，
用完整歷史算)在下一個交易日的候選事件完全一致——代號、排名順序、限價(觸發K棒收盤+1檔)、ATR。"""
import datetime
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import daily_squeeze_signals as dss
from squeeze_kdj_signal import (
    compute_squeeze_kdj_features, precompute_squeeze_kdj_features_by_code,
    precompute_squeeze_kdj_backtest_arrays, taiwan_tick_size,
)
from taifex_universe import STOCK_FUTURES_UNIVERSE, estimate_margin

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_market(n_stocks=10, n_days=700, seed=0, common=0.85):
    """合成多檔股票OHLCV：波動度每40天在低/高之間切換(容易擠壓、再跌破下軌反彈)，加上共同的
    「大盤」因子讓訊號常常同一天出現好幾檔(才測得到排名)。價格從約25到1000元，涵蓋多個tick級距。
    代號取自STOCK_FUTURES_UNIVERSE(合約乘數/保證金查表需要真實代號)。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    codes = list(STOCK_FUTURES_UNIVERSE.keys())[:n_stocks]
    regime = np.where((np.arange(n_days) // 40) % 2 == 0, 0.006, 0.025)
    mkt = rng.normal(0, 1, n_days)
    price_data = {}
    for code in codes:
        vol = regime * rng.uniform(0.7, 1.3)
        z = common * mkt + np.sqrt(1 - common ** 2) * rng.normal(0, 1, n_days)
        close = 50 * rng.uniform(0.5, 20) * np.exp(np.cumsum(0.0002 + vol * z))
        open_ = close * (1 + rng.normal(0, 0.004, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        volume = rng.lognormal(8, 0.5, n_days)
        price_data[code] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
    universe = {c: STOCK_FUTURES_UNIVERSE[c] for c in codes}
    return price_data, universe, idx


def _fake_loader(full_data, calls=None):
    """模擬data_loader.load_price_data：依[start, end)切資料(end跟yfinance一樣不含)。"""
    def loader(universe, start, end, refresh=False, cache_dir=None):
        if calls is not None:
            calls.append({"start": start, "end": end, "refresh": refresh, "cache_dir": cache_dir})
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        out = {}
        for code in universe:
            df = full_data.get(code)
            if df is None:
                continue
            sub = df[(df.index >= s) & (df.index < e)]
            if not sub.empty:
                out[code] = sub.copy()
        return out
    return loader


def _run_main(monkeypatch, tmp_path, full_data, as_of, universe_codes, extra=(), calls=None):
    monkeypatch.setattr(dss, "load_price_data", _fake_loader(full_data, calls))
    monkeypatch.setattr(dss, "build_universe",
                        lambda max_stocks=0: {c: STOCK_FUTURES_UNIVERSE[c] for c in universe_codes})
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    argv = ["--as-of", pd.Timestamp(as_of).date().isoformat(), "--output-dir", str(tmp_path), *extra]
    return dss.main(argv)


def _find_signal(price_data, min_idx=300):
    """找一個(code, 觸發K棒位置t)，t之後還有資料。"""
    for code, df in price_data.items():
        flags = compute_squeeze_kdj_features(df)["EntryFlag"].to_numpy()
        for t in np.flatnonzero(flags):
            if min_idx <= t < len(df) - 2:
                return code, int(t)
    raise AssertionError("合成資料裡找不到訊號")


@pytest.fixture(scope="module")
def market():
    return _make_market()


# ----------------------------------------------------------------------------
class TestConsistencyWithBacktest:
    def test_every_day_matches_backtest_candidates_ranking_and_limit(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        codes = list(universe)
        features = precompute_squeeze_kdj_features_by_code(price_data, universe)
        pre = precompute_squeeze_kdj_backtest_arrays(price_data, features, atr_period=14)
        events = pre["events_by_date"]

        n_signal_days = n_multi = 0
        first = 300  # 400個日曆天回看窗口 ≈ 286個交易日，從這裡開始窗口才是完整的
        for k in range(first, len(idx) - 1):
            d, next_day = idx[k], idx[k + 1]
            out = _run_main(monkeypatch, tmp_path, price_data, d, codes)
            got = out["result"]["signals"]
            assert out["result"]["data_fresh"]

            expected = []
            for code, i in events.get(next_day, []):
                arrs = pre["per_code"][code]
                expected.append((code, arrs["trigger_strength"][i], arrs["trigger_close"][i], arrs["atr"][i]))
            expected.sort(key=lambda e: e[1], reverse=True)  # 回測的穩定排序

            assert [r["code"] for r in got] == [e[0] for e in expected], f"{d.date()} 候選/排名不一致"
            for r, (code, strength, trig_close, atr) in zip(got, expected):
                assert r["trigger_return"] == pytest.approx(strength, rel=1e-12)
                assert r["close"] == pytest.approx(trig_close, rel=1e-12)
                assert r["limit_raw"] == pytest.approx(trig_close + taiwan_tick_size(trig_close), rel=1e-12)
                # 掛單價 = 同一個價格取到合法檔位(真實資料的收盤本來就在檔位上，兩者相同；合成資料不在檔位上)
                assert r["limit_price"] == dss.round_to_tick(r["limit_raw"])
                assert r["atr"] == pytest.approx(atr, rel=1e-9)
                assert r["entry_date"] == next_day.date().isoformat()
            n_signal_days += bool(expected)
            n_multi += len(expected) > 1

        # 測試本身要有內容：夠多訊號日，而且有同一天多檔(才測得到排名)
        assert n_signal_days >= 10
        assert n_multi >= 2


# ----------------------------------------------------------------------------
class TestSignalDetection:
    def test_signal_on_last_bar_is_listed(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        code, t = _find_signal(price_data)
        out = _run_main(monkeypatch, tmp_path, price_data, idx[t], list(universe))
        assert code in [r["code"] for r in out["result"]["signals"]]
        assert "今天沒有訊號" not in out["markdown"]

    def test_signal_on_earlier_bar_is_not_listed(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        code, t = _find_signal(price_data)
        out = _run_main(monkeypatch, tmp_path, price_data, idx[t + 1], list(universe))
        assert code not in [r["code"] for r in out["result"]["signals"]]

    def test_requests_as_of_plus_one_day_and_refresh(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        calls = []
        _run_main(monkeypatch, tmp_path, price_data, idx[400], list(universe), calls=calls)
        assert len(calls) == 1
        assert calls[0]["end"] == (idx[400] + pd.Timedelta(days=1)).date().isoformat()  # yfinance end不含
        assert calls[0]["start"] == (idx[400] - pd.Timedelta(days=dss.DEFAULT_LOOKBACK_DAYS)).date().isoformat()
        assert calls[0]["refresh"] is True
        assert calls[0]["cache_dir"] is not None and not os.path.exists(calls[0]["cache_dir"])  # 暫存快取用完就刪

    def test_download_window(self):
        assert dss.download_window(datetime.date(2026, 10, 5), 400) == ("2025-08-31", "2026-10-06")


class TestStaleData:
    def test_last_bar_not_as_of_gives_warning_and_no_actionable_list(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        code, t = _find_signal(price_data)
        # 資料只到t(有訊號)，但as_of是t的下一個交易日 → 資料沒更新
        truncated = {c: df[df.index <= idx[t]] for c, df in price_data.items()}
        out = _run_main(monkeypatch, tmp_path, truncated, idx[t + 1], list(universe))
        res, md = out["result"], out["markdown"]
        assert res["data_fresh"] is False
        assert res["signals"] == []  # 最後一根不是as_of → 不算訊號
        assert "資料還沒更新到今天，這份清單不可信，請晚點重跑" in md
        assert "今天沒有訊號" not in md  # 不能讓人誤以為「確定沒訊號」

    def test_weekend_as_of_is_stale(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        sat = idx[400] + pd.offsets.Week(weekday=5)
        out = _run_main(monkeypatch, tmp_path, price_data, sat, list(universe))
        assert out["result"]["data_fresh"] is False
        assert "請晚點重跑" in out["markdown"]

    def test_failed_downloads_reported(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        codes = list(universe)
        partial = {c: df for c, df in price_data.items() if c != codes[1]}
        out = _run_main(monkeypatch, tmp_path, partial, idx[400], codes)
        assert out["result"]["failed_codes"] == [codes[1]]
        assert "1 檔失敗" in out["markdown"]

    def test_reference_stock_fresh_but_one_stock_stale_is_not_listed(self, market):
        price_data, universe, idx = market
        code, t = _find_signal(price_data)
        data = dict(price_data)
        data = {c: df[df.index <= idx[t]] for c, df in data.items()}
        # 這一檔停牌：沒有t那根
        data[code] = data[code][data[code].index < idx[t]]
        res = dss.scan(data, universe, idx[t])
        assert res["data_fresh"] is True
        assert code in res["stale_codes"]
        assert code not in [r["code"] for r in res["signals"]]


class TestNaNRows:
    def test_nan_rows_dropped_same_as_manually_removed(self, market):
        price_data, universe, idx = market
        code, t = _find_signal(price_data)
        d = idx[t]
        clean = {c: df[df.index <= d] for c, df in price_data.items()}
        dirty = {c: df.copy() for c, df in clean.items()}
        # 在每檔中間插入「有成交量、價格是NaN」的列(yfinance偶爾會這樣)，日期是週六，不影響其他列
        for c, df in dirty.items():
            extra_day = idx[t - 50] + pd.Timedelta(days=1) if idx[t - 50].weekday() == 4 else idx[t - 50] + pd.offsets.Week(weekday=5)
            nan_row = pd.DataFrame({"Open": [np.nan], "High": [np.nan], "Low": [np.nan], "Close": [np.nan],
                                    "Volume": [1000.0]}, index=[extra_day])
            df = pd.concat([df, nan_row]).sort_index()
            df.iloc[t - 120, df.columns.get_loc("Close")] = np.nan  # 單一欄位NaN也要整列丟掉
            dirty[c] = df
        for c in clean:
            clean[c] = clean[c].drop(clean[c].index[t - 120])
        a = dss.scan(clean, universe, d)
        b = dss.scan(dirty, universe, d)
        assert [r["code"] for r in a["signals"]] == [r["code"] for r in b["signals"]]
        for ra, rb in zip(a["signals"], b["signals"]):
            assert ra["limit_price"] == rb["limit_price"] and ra["atr"] == pytest.approx(rb["atr"])

    def test_nan_last_row_means_stale_for_that_stock(self, market):
        price_data, universe, idx = market
        code, t = _find_signal(price_data)
        data = {c: df[df.index <= idx[t]].copy() for c, df in price_data.items()}
        data[code].iloc[-1, data[code].columns.get_loc("Open")] = np.nan
        res = dss.scan(data, universe, idx[t])
        assert code not in [r["code"] for r in res["signals"]]
        assert code in res["stale_codes"]


# ----------------------------------------------------------------------------
class TestTickRounding:
    @pytest.mark.parametrize("price,expected", [
        (9.987, 9.98), (10.04, 10.0), (49.99, 49.95), (50.07, 50.0), (99.97, 99.9),
        (100.3, 100.0), (499.9, 499.5), (512.7, 512.0), (999.9, 999.0), (1003.0, 1000.0),
        (1009.99, 1005.0), (123.5, 123.5), (55.3, 55.3),
    ])
    def test_floor_to_tick(self, price, expected):
        assert dss.floor_to_tick(price) == pytest.approx(expected, abs=1e-9)
        assert dss.floor_to_tick(price) <= price + 1e-9

    @pytest.mark.parametrize("close,limit", [(9.99, 10.0), (49.95, 50.0), (99.9, 100.0), (499.5, 500.0),
                                             (999.0, 1000.0), (123.5, 124.0), (1005.0, 1010.0), (35.2, 35.25)])
    def test_limit_is_close_plus_one_tick(self, close, limit):
        assert dss.round_to_tick(close + taiwan_tick_size(close)) == pytest.approx(limit, abs=1e-9)

    def test_stop_and_target_rounded_down_from_limit_fill(self, market):
        price_data, universe, idx = market
        code, t = _find_signal(price_data)
        res = dss.scan({c: df[df.index <= idx[t]] for c, df in price_data.items()}, universe, idx[t])
        r = next(r for r in res["signals"] if r["code"] == code)
        raw_stop = r["limit_price"] - 1.0 * r["atr"]
        raw_target = r["limit_price"] + 3.0 * r["atr"]
        assert r["stop_if_fill_at_limit"] == dss.floor_to_tick(raw_stop)
        assert r["target_if_fill_at_limit"] == dss.floor_to_tick(raw_target)
        assert raw_stop - taiwan_tick_size(raw_stop) < r["stop_if_fill_at_limit"] <= raw_stop
        assert raw_target - taiwan_tick_size(raw_target) < r["target_if_fill_at_limit"] <= raw_target


def _single_stock_signal_df(code_price=100.0):
    """拿合成市場裡的一個訊號，把價格整體縮放到指定水準(訊號只看相對關係，縮放不改變EntryFlag)。"""
    price_data, universe, idx = _make_market(seed=0)
    code, t = _find_signal(price_data)
    df = price_data[code][price_data[code].index <= idx[t]].copy()
    scale = code_price / df["Close"].iloc[-1]
    df[["Open", "High", "Low", "Close"]] *= scale
    return df


class TestContractAndMargin:
    def test_mini_contract_100_shares(self):
        df = _single_stock_signal_df(100.0)
        row = dss.compute_signal_row("2330", df, capital=200_000)  # 2330有小型契約
        assert row is not None
        assert row["multiplier"] == 100 and row["contract"] == "小型"
        assert row["stop_dist_ntd_per_lot"] == pytest.approx(row["atr"] * 100)
        assert row["target_dist_ntd_per_lot"] == pytest.approx(3 * row["atr"] * 100)
        assert row["margin_est_1lot"] == pytest.approx(estimate_margin("2330", row["limit_price"], 1))

    def test_standard_contract_2000_shares_and_margin_flag(self):
        df = _single_stock_signal_df(300.0)
        row = dss.compute_signal_row("1101", df, capital=200_000)  # 1101沒有小型契約
        assert row["multiplier"] == 2000 and row["contract"] == "標準"
        # 300 x 2000 x 20% = 約12萬 > 20萬 x 35% = 7萬
        assert row["margin_over_cap"] is True and row["backtest_would_skip"] is True
        res = {"as_of": pd.Timestamp(row["signal_date"]), "capital": 200_000, "universe_size": 1,
               "downloaded": 1, "failed_codes": [], "usable": 1, "stale_codes": [],
               "latest_date": pd.Timestamp(row["signal_date"]), "latest_source": "x", "data_fresh": True,
               "signals": [{**row, "rank": 1}], "entry_date": pd.Timestamp(row["entry_date"])}
        md = dss.render_markdown(res)
        assert "超過單筆保證金上限，回測會略過，不建議下單" in md
        assert "⛔" in md

    def test_margin_flag_off_with_big_capital(self):
        df = _single_stock_signal_df(300.0)
        row = dss.compute_signal_row("1101", df, capital=10_000_000)
        assert row["margin_over_cap"] is False and row["backtest_would_skip"] is False


class TestExpiryDate:
    def test_friday_signal_enters_monday_and_expires_20th_trading_day(self):
        signal = pd.Timestamp("2026-10-02")  # 週五
        entry = dss.next_business_day(signal)
        assert entry == pd.Timestamp("2026-10-05")
        assert dss.time_exit_date(entry) == pd.Timestamp("2026-10-30")  # 進場日算第1天，第20個交易日
        assert len(pd.bdate_range(entry, dss.time_exit_date(entry))) == 20

    def test_row_dates(self):
        df = _single_stock_signal_df(100.0)
        row = dss.compute_signal_row("2330", df, capital=200_000)
        entry = pd.Timestamp(row["entry_date"])
        assert entry == pd.Timestamp(row["signal_date"]) + pd.offsets.BDay(1)
        assert len(pd.bdate_range(entry, row["time_exit_date"])) == 20


# ----------------------------------------------------------------------------
class TestOutputs:
    def test_no_signal_message_and_files(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        features = precompute_squeeze_kdj_features_by_code(price_data, universe)
        flags = pd.DataFrame({c: f["EntryFlag"] for c, f in features.items()})
        quiet = [d for d in idx[300:] if not flags.loc[d].any()][0]
        summary = tmp_path / "summary.md"
        monkeypatch.setattr(dss, "load_price_data", _fake_loader(price_data))
        monkeypatch.setattr(dss, "build_universe", lambda max_stocks=0: dict(universe))
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        out = dss.main(["--as-of", quiet.date().isoformat(), "--output-dir", str(tmp_path / "sig")])
        md = out["markdown"]
        assert "今天沒有訊號，明天不用下單" in md
        assert "每日操作規則" in md and "PF≈0.67" in md and "NT$30,000" in md
        day = quiet.date().isoformat()
        assert (tmp_path / "sig" / "latest.md").read_text(encoding="utf-8") == md
        assert (tmp_path / "sig" / f"{day}.md").read_text(encoding="utf-8") == md
        csv = pd.read_csv(tmp_path / "sig" / f"{day}.csv")
        assert list(csv.columns) == dss.CSV_COLUMNS and len(csv) == 0
        assert md in summary.read_text(encoding="utf-8")

    def test_signal_files_contain_ranked_rows(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        features = precompute_squeeze_kdj_features_by_code(price_data, universe)
        flags = pd.DataFrame({c: f["EntryFlag"] for c, f in features.items()})
        busy = flags.index[(flags.sum(axis=1) >= 2) & (flags.index >= idx[300])][0]
        out = _run_main(monkeypatch, tmp_path, price_data, busy, list(universe))
        sigs = out["result"]["signals"]
        assert len(sigs) >= 2
        assert [r["rank"] for r in sigs] == list(range(1, len(sigs) + 1))
        assert all(a["trigger_return"] >= b["trigger_return"] for a, b in zip(sigs, sigs[1:]))
        csv = pd.read_csv(tmp_path / f"{busy.date().isoformat()}.csv", dtype={"code": str})
        assert list(csv["code"]) == [r["code"] for r in sigs]
        assert "### 1. " in out["markdown"] and "### 2. " in out["markdown"]
        assert "今天沒有訊號" not in out["markdown"]


class TestWithoutScipy:
    def test_scan_runs_with_scipy_blocked(self, tmp_path):
        """GitHub Actions沒有scipy：在子行程裡把scipy整個擋掉，import+掃描+輸出都要能跑。"""
        code = textwrap.dedent(f"""
            import sys
            class _Block:
                def find_spec(self, name, path=None, target=None):
                    if name == "scipy" or name.startswith("scipy."):
                        raise ImportError("scipy blocked for test")
                    return None
            sys.meta_path.insert(0, _Block())
            sys.path.insert(0, {REPO_DIR!r})
            import test_daily_squeeze_signals as t
            import daily_squeeze_signals as dss
            price_data, universe, idx = t._make_market(n_stocks=4, n_days=420)
            res = dss.scan({{c: df[df.index <= idx[-1]] for c, df in price_data.items()}}, universe, idx[-1])
            out = dss.write_outputs(res, {str(tmp_path)!r})
            assert "scipy" not in sys.modules or sys.modules["scipy"] is None
            print("OK", len(res["signals"]))
        """)
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_STEP_SUMMARY"}
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_DIR, env=env)
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout
        assert (tmp_path / "latest.md").exists()


# ----------------------------------------------------------------------------
def _result_with(rows, fresh=True, as_of="2026-10-05"):
    as_of = pd.Timestamp(as_of)
    return {"as_of": as_of, "capital": 200_000, "universe_size": 249, "downloaded": 249, "failed_codes": [],
            "usable": 249, "stale_codes": [], "latest_date": as_of if fresh else as_of - pd.Timedelta(days=3),
            "latest_source": "2330", "data_fresh": fresh, "signals": rows,
            "entry_date": dss.next_business_day(as_of)}


class TestTelegramMessage:
    def _row(self, code, price, capital=200_000, rank=1):
        row = dss.compute_signal_row(code, _single_stock_signal_df(price), capital=capital)
        return {**row, "rank": rank}

    def test_no_signal(self):
        txt = dss.render_telegram(_result_with([]), link="https://example.com/latest.md")
        assert "今天沒有訊號" in txt and "2026-10-06(二) 不用下單" in txt
        assert txt.endswith("https://example.com/latest.md")

    def test_stale_data_has_no_prices(self):
        r = self._row("2330", 100.0)
        txt = dss.render_telegram(_result_with([r], fresh=False))
        assert "資料還沒更新到今天" in txt and "不要下單" in txt
        assert "限價買" not in txt

    def test_actionable_signal_numbers_match_markdown_row(self):
        r = self._row("2330", 100.0)
        txt = dss.render_telegram(_result_with([r]))
        assert f"限價買 {dss.fmt_price(r['limit_price'])}" in txt
        assert f"停損 {dss.fmt_price(r['stop_if_fill_at_limit'])}" in txt
        assert f"停利 {dss.fmt_price(r['target_if_fill_at_limit'])}" in txt
        assert "1檔訊號(可下單1檔)" in txt and "小型契約" in txt

    def test_blocked_signal_says_do_not_trade(self):
        r = self._row("1101", 300.0)  # 標準契約，保證金超過20萬x35%
        txt = dss.render_telegram(_result_with([r]))
        assert "⛔ 不要下(保證金超過資金35%)" in txt and "不用下單" in txt
        assert "限價買" not in txt

    def test_truncated_under_telegram_limit(self):
        base = self._row("2330", 100.0)
        rows = [{**base, "rank": i} for i in range(1, 80)]
        txt = dss.render_telegram(_result_with(rows), link="https://example.com/latest.md")
        assert len(txt) <= 4096
        assert "其餘請看完整版" in txt and txt.endswith("https://example.com/latest.md")

    def test_write_outputs_writes_telegram_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SIGNALS_LINK", "https://example.com/x.md")
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        dss.write_outputs(_result_with([]), str(tmp_path))
        assert (tmp_path / "telegram.txt").read_text(encoding="utf-8").endswith("https://example.com/x.md")
