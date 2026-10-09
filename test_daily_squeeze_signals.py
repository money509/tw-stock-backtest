"""daily_squeeze_signals.py(每日訊號掃描器)的測試。全部離線：資料下載一律monkeypatch。

最重要的是 TestConsistencyWithBacktest：在長段合成資料上，對每一個訊號日d，用「只看得到d以前
(含d)、而且只抓預設回看窗口」的資料跑掃描器，結果必須跟回測(precompute_squeeze_kdj_backtest_arrays，
用完整歷史算)在下一個交易日的候選事件完全一致——代號、排名順序、限價(觸發K棒收盤+2檔，跟回測
execution_model="limit_ticks"、entry_limit_ticks=2同一個函式)、ATR。
TestLimitMatchesBacktestAcrossTickBoundaries再用真的回測引擎在價位級距邊界上確認：掃描器的限價/停損/停利
跟回測實際用的價格一模一樣。"""
import datetime
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import requests

import daily_squeeze_signals as dss
import taifex_futures_loader as tfl
from test_taifex_futures_loader import (
    fake_liquidity, fake_codes, install_fake_http, make_stock_lists_html, make_fut_csv_bytes,
)
from squeeze_kdj_signal import (
    compute_squeeze_kdj_features, precompute_squeeze_kdj_features_by_code,
    precompute_squeeze_kdj_backtest_arrays, taiwan_tick_size, taiwan_add_ticks,
    squeeze_kdj_entry_limit_price, run_squeeze_kdj_capital_constrained_backtest,
)
import compare_breakout as cb
from taifex_universe import STOCK_FUTURES_UNIVERSE, estimate_margin

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """全部離線：真實HTTP一律失敗；main()裡的期交所loader預設換成「全部流動性都夠」的假結果
    (個別測試要測真的loader時，自己把dss.load_futures_liquidity換回tfl.load_futures_liquidity + 假HTTP)。"""
    def boom(*a, **k):
        raise AssertionError("測試不可以連網路")
    monkeypatch.setattr(requests.Session, "request", boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(dss, "load_futures_liquidity",
                        lambda as_of, universe, **kw: fake_liquidity(list(universe)))


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
                # 回測的限價(limit_ticks, N=2)：同一個函式、同一個輸入 → 逐位元相同
                backtest_limit = squeeze_kdj_entry_limit_price(trig_close, "limit_ticks", entry_limit_ticks=2)
                assert r["limit_raw"] == backtest_limit
                assert r["limit_raw"] == taiwan_add_ticks(trig_close, 2)
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

    @pytest.mark.parametrize("close,limit", [(9.98, 10.0), (9.99, 10.05), (49.95, 50.1), (99.9, 100.5),
                                             (499.5, 501.0), (999.0, 1005.0), (123.5, 124.5), (1005.0, 1015.0),
                                             (35.2, 35.3)])
    def test_limit_is_close_plus_two_ticks_walked(self, close, limit):
        """逐檔往上走2檔：跨級距時第2檔用新級距的tick(49.95 → 50.0 → 50.1，不是49.95+2x0.05)。"""
        raw = dss.entry_limit_raw(close)
        assert raw == squeeze_kdj_entry_limit_price(close, "limit_ticks", entry_limit_ticks=2)
        assert dss.round_to_tick(raw) == pytest.approx(limit, abs=1e-9)

    def test_stop_and_target_rounded_down_from_limit_fill(self, market):
        price_data, universe, idx = market
        code, t = _find_signal(price_data)
        res = dss.scan({c: df[df.index <= idx[t]] for c, df in price_data.items()}, universe, idx[t])
        r = next(r for r in res["signals"] if r["code"] == code)
        raw_stop = r["limit_price"] - 1.5 * r["atr"]
        raw_target = r["limit_price"] + 4.0 * r["atr"]  # 預設positions=1
        assert r["stop_if_fill_at_limit"] == dss.floor_to_tick(raw_stop)
        assert r["target_if_fill_at_limit"] == dss.floor_to_tick(raw_target)
        assert raw_stop - taiwan_tick_size(raw_stop) < r["stop_if_fill_at_limit"] <= raw_stop
        assert raw_target - taiwan_tick_size(raw_target) < r["target_if_fill_at_limit"] <= raw_target


_SIGNAL_MARKET = None


def _single_stock_signal_df(code_price=100.0):
    """拿合成市場裡的一個訊號，把價格整體縮放到指定水準(訊號只看相對關係，縮放不改變EntryFlag)。
    最後一根收盤直接設成code_price(避免縮放的浮點誤差讓收盤不剛好落在級距邊界上)。"""
    global _SIGNAL_MARKET
    if _SIGNAL_MARKET is None:
        price_data, universe, idx = _make_market(seed=0)
        _SIGNAL_MARKET = (price_data, idx, _find_signal(price_data))
    price_data, idx, (code, t) = _SIGNAL_MARKET
    df = price_data[code][price_data[code].index <= idx[t]].copy()
    scale = code_price / df["Close"].iloc[-1]
    df[["Open", "High", "Low", "Close"]] *= scale
    df.iloc[-1, df.columns.get_loc("Close")] = code_price
    df.iloc[-1, df.columns.get_loc("High")] = max(df["High"].iloc[-1], code_price)
    df.iloc[-1, df.columns.get_loc("Low")] = min(df["Low"].iloc[-1], code_price)
    return df


class TestContractAndMargin:
    def test_mini_contract_100_shares(self):
        df = _single_stock_signal_df(100.0)
        row = dss.compute_signal_row("2330", df, capital=200_000)  # 2330有小型契約
        assert row is not None
        assert row["multiplier"] == 100 and row["contract"] == "小型"
        assert row["stop_dist_ntd_per_lot"] == pytest.approx(1.5 * row["atr"] * 100)
        assert row["target_dist_ntd_per_lot"] == pytest.approx(4.0 * row["atr"] * 100)  # 預設positions=1
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
        res = dss.build_tracks(res, fake_liquidity(["1101"]))
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
        assert "每日操作規則" in md and "NT$30,000" in md and "PF<1" in md
        assert "PF≈0.67" not in md and "+2檔" in md
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
            res2 = dss.scan({{c: df[df.index <= idx[-1]] for c, df in price_data.items()}}, universe, idx[-1],
                            positions=2)
            assert res2["positions"] == 2 and "停利3.0倍ATR" in dss.render_telegram(res2)
            assert "scipy" not in sys.modules or sys.modules["scipy"] is None
            print("OK", len(res["signals"]))
        """)
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_STEP_SUMMARY"}
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_DIR, env=env)
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout
        assert (tmp_path / "latest.md").exists()
        assert "設定：同時最多1檔｜停損1.5倍ATR｜停利4.0倍ATR｜限價收盤+2檔" in (tmp_path / "latest.md").read_text(encoding="utf-8")
        assert "設定：同時最多1檔" in (tmp_path / "telegram.txt").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------
def _result_with(rows, fresh=True, as_of="2026-10-05", liq="ok", **track_kw):
    """liq="ok"：訊號的期貨流動性全部夠；None：沒有期貨資料(fail closed)；或直接給一份loader結果。
    現股成交值門檻預設0(合成資料的成交量很小)。"""
    as_of = pd.Timestamp(as_of)
    res = {"as_of": as_of, "capital": 200_000, "universe_size": 249, "downloaded": 249, "failed_codes": [],
           "usable": 249, "stale_codes": [], "latest_date": as_of if fresh else as_of - pd.Timedelta(days=3),
           "latest_source": "2330", "data_fresh": fresh, "signals": rows,
           "entry_date": dss.next_business_day(as_of)}
    if liq == "ok":
        liq = fake_liquidity(sorted({r["code"] for r in rows}))
    track_kw.setdefault("stock_min_turnover", 0)
    return dss.build_tracks(res, liq, **track_kw)


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


# ----------------------------------------------------------------------------
# 實盤設定(SQUEEZE_KDJ_LIVE_SETTINGS)：--positions 1/2
# ----------------------------------------------------------------------------
BOUNDARY_CLOSES = (9.99, 49.95, 99.9, 499.5, 999.0)


class TestLimitMatchesBacktestAcrossTickBoundaries:
    """在價位級距邊界上，用真的回測引擎(execution_model="limit_ticks"、entry_limit_ticks=2、實盤設定)
    確認掃描器算的限價就是回測實際用的限價：開盤=掃描器限價 → 回測剛好在限價成交；開盤=限價+1檔 → 回測不追。
    同時確認回測的停損/停利倍數跟掃描器一致。"""

    def _backtest(self, df, positions, next_bar):
        code = "2330"
        next_day = df.index[-1] + pd.offsets.BDay(1)
        full = pd.concat([df, pd.DataFrame([next_bar], index=[next_day])])
        price_data = {code: full}
        universe = {code: STOCK_FUTURES_UNIVERSE[code]}
        features = precompute_squeeze_kdj_features_by_code(price_data, universe)
        pre = precompute_squeeze_kdj_backtest_arrays(price_data, features, atr_period=14)
        i = len(full) - 1
        assert (code, i) in pre["events_by_date"][next_day]
        assert pre["per_code"][code]["trigger_close"][i] == df["Close"].iloc[-1]  # 回測的參考價 = 掃描器的收盤
        st = cb.SQUEEZE_KDJ_LIVE_SETTINGS[positions]
        trades, diag = run_squeeze_kdj_capital_constrained_backtest(
            price_data, universe, pd.DatetimeIndex([next_day]), 10_000_000, variant=st["variant"],
            lots=st["lots"], top_n=st["top_n"], max_concurrent_positions=st["max_concurrent_positions"],
            atr_stop_mult=st["atr_stop_mult"], atr_target_mult=st["atr_target_mult"], atr_period=st["atr_period"],
            max_hold_days=st["max_hold_days"], ranking_rule=st["ranking_rule"], entry_filter=st["entry_filter"],
            features_by_code=features, precomputed=pre, return_diagnostics=True,
            execution_model=st["execution_model"], entry_limit_ticks=st["entry_limit_ticks"])
        return trades, diag, pre["per_code"][code]["trigger_close"][i]

    @pytest.mark.parametrize("close", BOUNDARY_CLOSES)
    @pytest.mark.parametrize("positions", [1, 2])
    def test_fill_at_scanner_limit_and_target_matches(self, close, positions):
        df = _single_stock_signal_df(close)
        row = dss.compute_signal_row("2330", df, capital=10_000_000, positions=positions)
        assert row is not None and row["close"] == close
        lim, atr = row["limit_price"], row["atr"]
        # 開盤剛好=掃描器限價、當天漲很多 → 回測在限價成交、打到停利
        trades, diag, trig_close = self._backtest(
            df, positions, {"Open": lim, "High": lim + 10 * atr, "Low": lim, "Close": lim, "Volume": 1e4})
        assert row["limit_raw"] == squeeze_kdj_entry_limit_price(trig_close, "limit_ticks", entry_limit_ticks=2)
        assert diag["skipped_limit_not_filled"] == 0
        assert len(trades) == 1 and trades[0]["exit_reason"] == "target"
        assert trades[0]["e_price"] == pytest.approx(lim, abs=1e-9)
        target_mult = {1: 4.0, 2: 3.0}[positions]
        assert trades[0]["exit_price"] == pytest.approx(lim + target_mult * atr, rel=1e-9)
        assert row["target_dist"] == pytest.approx(target_mult * atr)

    @pytest.mark.parametrize("close", BOUNDARY_CLOSES)
    def test_stop_matches_backtest(self, close):
        df = _single_stock_signal_df(close)
        row = dss.compute_signal_row("2330", df, capital=10_000_000, positions=1)
        lim, atr = row["limit_price"], row["atr"]
        trades, diag, _ = self._backtest(
            df, 1, {"Open": lim, "High": lim, "Low": max(lim - 10 * atr, 0.01), "Close": lim, "Volume": 1e4})
        assert len(trades) == 1 and trades[0]["exit_reason"] == "stop"
        raw_stop = lim - 1.5 * atr
        # 回測：停損是市價型出場，多付1檔
        assert trades[0]["exit_price"] == pytest.approx(raw_stop - taiwan_tick_size(raw_stop), rel=1e-9)
        assert row["stop_dist"] == pytest.approx(1.5 * atr)

    @pytest.mark.parametrize("close", BOUNDARY_CLOSES)
    def test_open_one_tick_above_scanner_limit_is_not_chased(self, close):
        df = _single_stock_signal_df(close)
        row = dss.compute_signal_row("2330", df, capital=10_000_000)
        above = dss.round_to_tick(row["limit_price"] + taiwan_tick_size(row["limit_price"]))
        trades, diag, _ = self._backtest(
            df, 1, {"Open": above, "High": above, "Low": above, "Close": above, "Volume": 1e4})
        assert trades == [] and diag["skipped_limit_not_filled"] == 1


class TestLiveSettings:
    def test_live_settings_constant(self):
        s1, s2 = cb.SQUEEZE_KDJ_LIVE_SETTINGS[1], cb.SQUEEZE_KDJ_LIVE_SETTINGS[2]
        assert set(cb.SQUEEZE_KDJ_LIVE_SETTINGS) == {1, 2}
        for s in (s1, s2):
            assert s["atr_stop_mult"] == 1.5 and s["entry_limit_ticks"] == 2 and s["execution_model"] == "limit_ticks"
            assert s["variant"] == "B" and s["atr_period"] == 14 and s["max_hold_days"] == 20
            assert s["ranking_rule"] == "trigger_return" and s["entry_filter"] is None
            assert s["top_n"] == 3 and s["lots"] == 1
        assert (s1["atr_target_mult"], s1["max_concurrent_positions"]) == (4.0, 1)
        assert (s2["atr_target_mult"], s2["max_concurrent_positions"]) == (3.0, 2)
        # 舊模式依賴的舊設定沒動
        assert cb.SQUEEZE_KDJ_FIXED_SETTING["atr_stop_mult"] == 1.0
        assert cb.SQUEEZE_KDJ_FIXED_SETTING["atr_target_mult"] == 3.0
        assert cb.SQUEEZE_KDJ_FIXED_SETTING["max_concurrent_positions"] == 3

    def test_positions_1_target_4_stop_1_5(self):
        row = dss.compute_signal_row("2330", _single_stock_signal_df(100.0), capital=200_000, positions=1)
        assert row["atr_target_mult"] == 4.0 and row["atr_stop_mult"] == 1.5
        assert row["target_dist"] == pytest.approx(4.0 * row["atr"])
        assert row["stop_dist"] == pytest.approx(1.5 * row["atr"])
        assert row["target_if_fill_at_limit"] == dss.floor_to_tick(row["limit_price"] + 4.0 * row["atr"])
        assert row["stop_if_fill_at_limit"] == dss.floor_to_tick(row["limit_price"] - 1.5 * row["atr"])
        assert row["target_dist_ntd_per_lot"] == pytest.approx(4.0 * row["atr"] * row["multiplier"])

    def test_positions_2_target_3(self):
        df = _single_stock_signal_df(100.0)
        r1 = dss.compute_signal_row("2330", df, capital=200_000, positions=1)
        r2 = dss.compute_signal_row("2330", df, capital=200_000, positions=2)
        assert r2["atr_target_mult"] == 3.0 and r2["atr_stop_mult"] == 1.5
        assert r2["target_dist"] == pytest.approx(3.0 * r2["atr"])
        assert r2["target_if_fill_at_limit"] == dss.floor_to_tick(r2["limit_price"] + 3.0 * r2["atr"])
        # 限價/停損/ATR跟positions無關
        for k in ("limit_price", "stop_if_fill_at_limit", "atr", "stop_dist"):
            assert r1[k] == r2[k]

    def test_default_is_one_position(self):
        df = _single_stock_signal_df(100.0)
        assert dss.compute_signal_row("2330", df, capital=200_000)["atr_target_mult"] == 4.0
        assert dss.parse_args([]).positions == 1

    @pytest.mark.parametrize("bad", [0, 3, -1, True, "1", 1.5])
    def test_invalid_positions_rejected(self, bad, market):
        with pytest.raises(ValueError):
            dss.live_setting(bad)
        with pytest.raises(ValueError):
            dss.compute_signal_row("2330", _single_stock_signal_df(100.0), capital=200_000, positions=bad)
        price_data, universe, idx = market
        with pytest.raises(ValueError):
            dss.scan(price_data, universe, idx[400], positions=bad)

    @pytest.mark.parametrize("bad", ["0", "3", "x"])
    def test_invalid_positions_rejected_by_cli(self, bad):
        with pytest.raises(SystemExit):
            dss.parse_args(["--positions", bad])

    def test_cli_positions_2_reaches_rows_and_csv(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        features = precompute_squeeze_kdj_features_by_code(price_data, universe)
        flags = pd.DataFrame({c: f["EntryFlag"] for c, f in features.items()})
        busy = flags.index[(flags.sum(axis=1) >= 1) & (flags.index >= idx[300])][0]
        out = _run_main(monkeypatch, tmp_path, price_data, busy, list(universe), extra=("--positions", "2"))
        res = out["result"]
        assert res["positions"] == 2 and res["signals"]
        assert all(r["atr_target_mult"] == 3.0 for r in res["signals"])
        assert "設定：同時最多2檔｜停損1.5倍ATR｜停利3.0倍ATR｜限價收盤+2檔" in out["markdown"]
        csv = pd.read_csv(tmp_path / f"{busy.date().isoformat()}.csv", dtype={"code": str})
        assert set(csv["positions"]) == {2} and set(csv["atr_target_mult"]) == {3.0}
        assert set(csv["atr_stop_mult"]) == {1.5} and set(csv["entry_limit_ticks"]) == {2}
        tg = (tmp_path / "telegram.txt").read_text(encoding="utf-8")
        assert "設定：同時最多2檔｜停損1.5倍ATR｜停利3.0倍ATR｜限價收盤+2檔" in tg


class TestHeadersAndRules:
    def _row(self, positions):
        row = dss.compute_signal_row("2330", _single_stock_signal_df(100.0), capital=200_000, positions=positions)
        return {**row, "rank": 1}

    @pytest.mark.parametrize("positions,label", [
        (1, "設定：同時最多1檔｜停損1.5倍ATR｜停利4.0倍ATR｜限價收盤+2檔"),
        (2, "設定：同時最多2檔｜停損1.5倍ATR｜停利3.0倍ATR｜限價收盤+2檔"),
    ])
    def test_headers_show_setting(self, positions, label):
        assert dss.setting_label(positions) == label
        res = {**_result_with([self._row(positions)]), "positions": positions}
        md = dss.render_markdown(res)
        assert md.splitlines()[2] == f"**{label}**"  # 標題下面第一行
        tg = dss.render_telegram(res)
        assert tg.splitlines()[1] == label
        target = {1: "4", 2: "3"}[positions]
        assert f"({target}×ATR)" in tg and "(1.5×ATR)" in tg and "收盤+2檔" in tg

    def test_stale_and_no_signal_telegram_still_show_setting(self):
        assert dss.render_telegram(_result_with([])).splitlines()[1].startswith("設定：同時最多1檔")
        assert dss.render_telegram(_result_with([], fresh=False)).splitlines()[1].startswith("設定：")

    @pytest.mark.parametrize("positions", [1, 2])
    def test_rules_text(self, positions):
        rules = "\n".join(dss._rules_section(positions))
        assert "PF≈0.67" not in rules and "2018–2023" not in rules
        assert "+2檔" in rules and "開盤高於限價就不追" in rules
        assert "1.5×ATR" in rules and f"{ {1: 4, 2: 3}[positions] }×ATR" in rules
        assert f"空位上限={positions}檔(你設定的N檔)" in rules
        assert f"{dss.COOLDOWN_CALENDAR_DAYS}個日曆天" in rules and "NT$30,000" in rules
        assert "30筆" in rules and "PF<1" in rules and "最長持有到期日" in rules
        assert "從來沒有通過專案的正式驗證" in rules and "約28個" in rules
        assert "1.33" in rules and "1.09" in rules and "2026" in rules and "部位保持最小" in rules
        assert f"資金的{ {1: 35, 2: 70}[positions] }%" in rules

    def test_signal_block_shows_two_ticks(self):
        md = dss.render_markdown({**_result_with([self._row(1)]), "positions": 1})
        assert "+2檔)，開盤高於此價不追" in md and "(4×ATR)" in md

    def test_old_mode_reports_no_longer_claim_scanner_uses_fixed_setting(self):
        note = cb._squeeze_kdj_live_scanner_note()
        assert "SQUEEZE_KDJ_LIVE_SETTINGS" in note and "停利4.0倍ATR" in note and "停利3.0倍ATR" in note
        with open(cb.__file__, encoding="utf-8") as f:
            assert "仍然沿用SQUEEZE_KDJ_FIXED_SETTING" not in f.read()


class TestWorkflowPositionsInput:
    WF_PATH = os.path.join(REPO_DIR, ".github", "workflows", "daily_squeeze_signals.yml")

    def _text(self):
        with open(self.WF_PATH, encoding="utf-8") as f:
            return f.read()

    def test_text_wiring(self):
        text = self._text()
        assert "POSITIONS: ${{ github.event.inputs.positions || '1' }}" in text
        assert '--positions ${POSITIONS:-1}' in text
        assert "同時最多持有幾檔：1=停利4倍ATR、2=停利3倍ATR" in text
        assert "default改成'2'" in text  # 提醒長期改2檔要改這裡

    def _scan_step(self):
        yaml = pytest.importorskip("yaml")
        wf = yaml.safe_load(self._text())
        on = wf.get("on", wf.get(True))
        steps = wf["jobs"]["scan"]["steps"]
        # 注意：Pre-flight步驟的run裡也有「test_daily_squeeze_signals.py」，要比對完整的「python daily_squeeze_signals.py」
        return on, next(s for s in steps if "python daily_squeeze_signals.py" in s.get("run", ""))

    def test_dispatch_input_is_choice_default_1(self):
        on, _ = self._scan_step()
        inp = on["workflow_dispatch"]["inputs"]["positions"]
        assert inp["type"] == "choice" and inp["options"] == ["1", "2"] and inp["default"] == "1"
        assert "schedule" in on

    @pytest.mark.parametrize("positions_env,expected", [("", "1"), ("1", "1"), ("2", "2")])
    def test_run_script_passes_positions(self, positions_env, expected, tmp_path):
        """把Scan signals的run腳本拿來實際用bash跑(python換成echo)：排程沒有inputs時POSITIONS是空字串 → 1。"""
        _, step = self._scan_step()
        script = step["run"].replace("python daily_squeeze_signals.py", "echo")
        env = {"PATH": os.environ.get("PATH", ""), "AS_OF": "", "CAPITAL": "", "POSITIONS": positions_env}
        assert "pytest" not in script
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)
        assert proc.returncode == 0, proc.stderr
        args = proc.stdout.split()
        assert args[args.index("--positions") + 1] == expected
        # env裡的GitHub表達式：沒有inputs(排程)時 || '1' 生效
        assert "|| '1'" in step["env"]["POSITIONS"]


# ----------------------------------------------------------------------------
# 兩個清單：個股期貨清單 / 現股清單
# ----------------------------------------------------------------------------
def _busy_day(market, min_signals=2):
    price_data, universe, idx = market
    features = precompute_squeeze_kdj_features_by_code(price_data, universe)
    flags = pd.DataFrame({c: f["EntryFlag"] for c, f in features.items()})
    return flags.index[(flags.sum(axis=1) >= min_signals) & (flags.index >= idx[300])][0]


def _sig_row(code, price, rank, turnover=1e9, capital=10_000_000, trigger=None):
    row = dss.compute_signal_row(code, _single_stock_signal_df(price), capital=capital)
    row = {**row, "rank": rank, "turnover_20d": turnover}
    if trigger is not None:
        row["trigger_return"] = trigger
    return row


def _futures_part(md):
    return md.split("## 📈 個股期貨清單")[1].split("## 🧾 現股清單")[0]


def _stock_part(md):
    return md.split("## 🧾 現股清單")[1].split("## 每日操作規則")[0]


class TestTwoListsEndToEnd:
    """真的loader(taifex_futures_loader)+ 假HTTP(stockLists HTML、Big5 CSV)，從main()一路跑到輸出檔。"""

    def _setup(self, market, monkeypatch, csv_status=200, illiquid_top=True):
        price_data, universe, idx = market
        busy = _busy_day(market)
        sigs = dss.scan({c: df[df.index <= busy] for c, df in price_data.items()}, universe, busy)["signals"]
        top = sigs[0]["code"]
        contracts = {}
        for c in universe:
            std, mini = fake_codes(c)
            fut = mini if STOCK_FUTURES_UNIVERSE[c]["has_mini"] else std
            low = illiquid_top and c == top
            contracts[fut] = {"near": 5 if low else 300, "far": 1, "night": 2, "oi": 900}
        dates = pd.bdate_range(end=busy, periods=7)
        install_fake_http(monkeypatch, html=make_stock_lists_html(list(universe)),
                          csv=make_fut_csv_bytes(contracts, dates), csv_status=csv_status)
        monkeypatch.setattr(dss, "load_futures_liquidity", tfl.load_futures_liquidity)
        return busy, sigs, top

    def test_liquid_only_in_futures_list_and_all_outputs(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        busy, sigs, top = self._setup(market, monkeypatch)
        out = _run_main(monkeypatch, tmp_path, price_data, busy, list(universe), extra=("--stock-min-turnover", "0"))
        res, md = out["result"], out["markdown"]
        ft, stt = res["futures_track"], res["stock_track"]
        assert ft["ok"], ft["error"]
        # 期貨清單：去掉流動性不夠的第1名，其餘照原本順序、自己重新排名
        assert [r["code"] for r in ft["rows"]] == [s["code"] for s in sigs[1:]]
        assert [r["rank"] for r in ft["rows"]] == list(range(1, len(sigs)))
        assert [e["code"] for e in ft["excluded"]] == [top]
        assert ft["rows"][0]["fut_avg_volume"] == pytest.approx(303) and ft["rows"][0]["fut_near_oi"] == 900
        assert ft["rows"][0]["fut_spread_ticks"] >= 1
        # 現股清單：同一批訊號、同樣順序(門檻設0)
        assert [r["code"] for r in stt["rows"]] == [s["code"] for s in sigs]
        # 限價/停損/停利跟原本的訊號一模一樣
        by = {s["code"]: s for s in sigs}
        for r in ft["rows"] + stt["rows"]:
            for k in ("limit_price", "stop_if_fill_at_limit", "target_if_fill_at_limit", "atr", "trigger_return"):
                assert r[k] == by[r["code"]][k]
        # markdown兩個區塊 + 排除的代碼只在md
        assert "## 📈 個股期貨清單(主要)" in md and "## 🧾 現股清單(參考)" in md
        assert md.index("## 📈 個股期貨清單") < md.index("## 🧾 現股清單")
        assert "另有1檔訊號因期貨成交量不足未列出" in md and f"：{top}" in md
        assert "期貨流動性：" in _futures_part(md) and "5日均量 303口" in _futures_part(md)
        # Telegram兩則
        tg_f = (tmp_path / "telegram_futures.txt").read_text(encoding="utf-8")
        tg_s = (tmp_path / "telegram_stock.txt").read_text(encoding="utf-8")
        assert tg_f.startswith("【個股期貨清單】") and tg_s.startswith("【現股清單】")
        assert tg_f.splitlines()[1] == tg_s.splitlines()[1] == dss.setting_label(1)
        assert "另有1檔訊號因期貨成交量不足未列出" in tg_f
        assert not any(line.startswith(f"{i}. {top}") for line in tg_f.splitlines() for i in range(1, 20))
        assert any(line.startswith(f"1. {top}") for line in tg_s.splitlines())
        assert len(tg_f) <= 4096 and len(tg_s) <= 4096
        assert (tmp_path / "telegram.txt").read_text(encoding="utf-8") == tg_f
        # 診斷檔
        assert set(os.listdir(tmp_path / "debug")) == set(tfl.DEBUG_FILES)
        # CSV
        csv = pd.read_csv(tmp_path / f"{busy.date().isoformat()}.csv", dtype={"code": str})
        assert list(csv.columns) == dss.CSV_COLUMNS
        assert dict(zip(csv["code"], csv["fut_listed"]))[top] == False  # noqa: E712
        assert csv["stock_listed"].all()

    def test_fail_closed_futures_list_empty_stock_list_unaffected(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        busy, sigs, top = self._setup(market, monkeypatch, csv_status=500)
        out = _run_main(monkeypatch, tmp_path, price_data, busy, list(universe), extra=("--stock-min-turnover", "0"))
        res, md = out["result"], out["markdown"]
        assert res["futures_track"]["ok"] is False and res["futures_track"]["rows"] == []
        assert "HTTP 500" in res["futures_track"]["error"]
        fut = _futures_part(md)
        assert tfl.FAILURE_MESSAGE in fut and "### " not in fut and "限價買進" not in fut
        assert len(res["stock_track"]["rows"]) == len(sigs) and "### 1. " in _stock_part(md)
        tg_f = (tmp_path / "telegram_futures.txt").read_text(encoding="utf-8")
        tg_s = (tmp_path / "telegram_stock.txt").read_text(encoding="utf-8")
        assert tfl.FAILURE_MESSAGE in tg_f and "限價買" not in tg_f and "今天沒有訊號" not in tg_f
        assert "限價買" in tg_s
        rep = (tmp_path / "debug" / "parse_report.json").read_text(encoding="utf-8")
        assert '"ok": false' in rep

    def test_loader_crash_is_fail_closed(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        busy = _busy_day(market)

        def crash(*a, **k):
            raise RuntimeError("boom")
        monkeypatch.setattr(dss, "load_price_data", _fake_loader(price_data))
        monkeypatch.setattr(dss, "build_universe", lambda max_stocks=0: dict(universe))
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        monkeypatch.setattr(dss, "load_futures_liquidity", crash)
        out = dss.main(["--as-of", busy.date().isoformat(), "--output-dir", str(tmp_path)])
        assert out["result"]["futures_track"]["ok"] is False
        assert "boom" in out["result"]["futures_track"]["error"]
        assert tfl.FAILURE_MESSAGE in out["markdown"]

    def test_no_futures_data_given_means_fail_closed(self, market):
        price_data, universe, idx = market
        busy = _busy_day(market)
        res = dss.scan({c: df[df.index <= busy] for c, df in price_data.items()}, universe, busy)
        assert res["signals"] and res["futures_track"]["ok"] is False and res["futures_track"]["rows"] == []
        assert tfl.FAILURE_MESSAGE in dss.render_telegram_futures(res)

    def test_cli_thresholds_reach_tracks(self, market, monkeypatch, tmp_path):
        price_data, universe, idx = market
        busy = _busy_day(market)
        out = _run_main(monkeypatch, tmp_path, price_data, busy, list(universe),
                        extra=("--fut-min-volume", "2000", "--stock-amount", "50000", "--stock-min-turnover", "0"))
        ft, stt = out["result"]["futures_track"], out["result"]["stock_track"]
        assert ft["min_volume"] == 2000 and ft["rows"] == [] and len(ft["excluded"]) == len(out["result"]["signals"])
        assert "期貨清單今天沒有可下單的標的" in out["markdown"]
        assert stt["amount"] == 50000 and all(r["stock_amount"] == 50000 for r in stt["rows"])


class TestIndependentRanking:
    def test_each_list_filters_and_reranks_independently(self):
        a = _sig_row("2330", 100.0, 1)
        b = _sig_row("1101", 100.0, 2, turnover=1e6)   # 現股成交值不夠
        c = _sig_row("2303", 100.0, 3)
        liq = fake_liquidity(["2330", "1101", "2303"], overrides={"2330": {"avg_volume": 10}})  # 2330期貨量不夠
        res = _result_with([a, b, c], liq=liq, stock_min_turnover=dss.DEFAULT_STOCK_MIN_TURNOVER)
        ft, stt = res["futures_track"], res["stock_track"]
        assert [(r["code"], r["rank"], r["signal_rank"]) for r in ft["rows"]] == [("1101", 1, 2), ("2303", 2, 3)]
        assert [(r["code"], r["rank"], r["signal_rank"]) for r in stt["rows"]] == [("2330", 1, 1), ("2303", 2, 3)]
        assert [e["code"] for e in stt["excluded"]] == ["1101"]
        md = dss.render_markdown(res)
        assert "另有1檔訊號因現股20日均成交值不足未列出：1101" in md
        assert "另有1檔訊號因期貨成交量不足未列出" in md and "2330" in _futures_part(md).split("另有")[1]
        assert "另有1檔訊號因現股20日均成交值不足未列出" in dss.render_telegram_stock(res)
        # 原本的訊號列不被改動
        assert [a["rank"], b["rank"], c["rank"]] == [1, 2, 3]


class TestStockTrack:
    @pytest.mark.parametrize("shares,text", [(2663, "2張663股"), (3000, "3張"), (800, "800股(零股)"),
                                             (1000, "1張"), (1, "1股(零股)"), (0, "0股")])
    def test_fmt_shares(self, shares, text):
        assert dss.fmt_shares(shares) == text

    def test_sizing_and_costs(self):
        r = {**_sig_row("2330", 100.0, 1), "limit_price": 37.55}
        s = dss.stock_sizing(r, 100_000)
        assert s["stock_shares"] == 2663 and s["stock_lots"] == 2 and s["stock_odd_shares"] == 663
        assert s["stock_cost"] == pytest.approx(2663 * 37.55)
        assert s["stock_fee_buy"] == pytest.approx(2663 * 37.55 * 0.001425)
        assert s["stock_tax"] == pytest.approx(2663 * 37.55 * 0.003)
        assert s["stock_roundtrip_cost"] == pytest.approx(2663 * 37.55 * (0.001425 * 2 + 0.003))
        assert s["stock_stop_ntd"] == pytest.approx(r["stop_dist"] * 2663)
        assert s["stock_target_ntd"] == pytest.approx(r["target_dist"] * 2663)
        assert s["stock_blocked"] is False

    def test_exact_multiple_not_lost_to_float(self):
        r = {**_sig_row("2330", 100.0, 1), "limit_price": 0.1}
        assert dss.stock_sizing(r, 100_000)["stock_shares"] == 1_000_000

    def test_less_than_one_share_is_blocked(self):
        r = {**_sig_row("2330", 100.0, 1), "limit_price": 150_000.0}
        res = _result_with([r])
        row = res["stock_track"]["rows"][0]
        assert row["stock_shares"] == 0 and row["stock_blocked"] and "買不到1股" in row["stock_block_reason"]
        md = dss.render_markdown(res)
        assert "### 1. 2330 台積電 ⛔" in _stock_part(md) and "不要下單" in _stock_part(md)
        tg = dss.render_telegram_stock(res)
        assert "⛔ 不要下(NT$100,000買不到1股)" in tg and "限價買" not in tg and "不用下單" in tg

    def test_turnover_filter_and_average(self):
        df = _single_stock_signal_df(100.0)
        exp = (df["Close"] * df["Volume"]).tail(20).mean()
        assert dss.average_turnover(df) == pytest.approx(exp)
        assert np.isnan(dss.average_turnover(df.drop(columns=["Volume"])))
        row = dss.compute_signal_row("2330", df, capital=200_000)
        assert row["turnover_20d"] == pytest.approx(exp)
        lo = _result_with([{**row, "rank": 1}], stock_min_turnover=exp + 1)
        hi = _result_with([{**row, "rank": 1}], stock_min_turnover=exp)
        assert lo["stock_track"]["rows"] == [] and len(hi["stock_track"]["rows"]) == 1
        nan = _result_with([{**row, "rank": 1, "turnover_20d": np.nan}])
        assert nan["stock_track"]["rows"] == []  # 沒有成交量資料 → 不列(fail closed)

    def test_markdown_stock_section_text(self):
        r = {**_sig_row("2330", 100.0, 1), "limit_price": 37.55}
        md = dss.render_markdown(_result_with([r]))
        part = _stock_part(md)
        assert "PF≈1.12" in part and "PF≈1.02" in part and "約1.0" in part and "大約只是打平" in part
        assert "盤中零股" in part and "0.1425%" in part and "0.3%" in part and "很多券商有折扣" in part
        assert "股數 **2張663股**" in part and "來回成本約 NT$" in part and "停損約 −NT$" in part
        tg = dss.render_telegram_stock(_result_with([r]))
        assert "2張663股" in tg and "零股" in tg and "PF≈1.0–1.1" in tg

    def test_invalid_stock_amount(self):
        with pytest.raises(ValueError):
            _result_with([], stock_amount=0)
        with pytest.raises(SystemExit):
            dss.parse_args(["--stock-amount", "0"])

    def test_cli_defaults(self):
        a = dss.parse_args([])
        assert (a.stock_amount, a.stock_min_turnover, a.fut_min_volume, a.fut_min_oi) == (100_000, 50_000_000, 100, 300)


class TestFuturesTrackRendering:
    def test_futures_block_shows_liquidity_and_existing_info(self):
        res = _result_with([_sig_row("2330", 100.0, 1)])
        md = dss.render_markdown(res)
        fut = _futures_part(md)
        assert "小型契約 100股" in fut and "保證金約" in fut
        assert "5日均量 1,000口" in fut and "近月(202610)未平倉 5,000口" in fut and "價差 1檔" in fut
        tg = dss.render_telegram_futures(res)
        assert f"期貨{fake_codes('2330')[1]}：5日均量1,000口｜近月未平倉5,000口｜買賣價差1檔" in tg

    def test_stale_futures_data_warns(self):
        liq = fake_liquidity(["2330"], latest_date="2026-10-02")
        res = _result_with([_sig_row("2330", 100.0, 1)], liq=liq)
        assert "期貨資料最新日期是 2026-10-02" in dss.render_markdown(res)
        assert "期貨流動性資料最新是2026-10-02" in dss.render_telegram_futures(res)

    def test_failure_message_in_both_outputs(self):
        res = _result_with([_sig_row("2330", 100.0, 1)], liq=tfl.failed_liquidity("HTTP 503"))
        md, tg = dss.render_markdown(res), dss.render_telegram_futures(res)
        assert tfl.FAILURE_MESSAGE == "期貨流動性資料抓取/解析失敗，今天期貨清單不列任何標的(不是沒有訊號)"
        assert tfl.FAILURE_MESSAGE in md and "原因：HTTP 503" in md and "期貨清單：資料失敗、不列" in md
        assert tfl.FAILURE_MESSAGE in tg and "現股清單見另一則" in tg
        assert "限價買" in dss.render_telegram_stock(res)

    def test_both_telegrams_truncated_under_limit(self):
        base = _sig_row("2330", 37.55, 1)
        rows = [{**base, "rank": i} for i in range(1, 80)]
        res = _result_with(rows)
        for fn in (dss.render_telegram_futures, dss.render_telegram_stock):
            txt = fn(res, link="https://example.com/latest.md")
            assert len(txt) <= 4096 and "其餘請看完整版" in txt and txt.endswith("https://example.com/latest.md")

    def test_rules_mention_per_list_slots_and_double_exposure(self):
        rules = "\n".join(dss._rules_section(1))
        assert "每個清單各自計算" in rules and "曝險加倍" in rules

    def test_write_outputs_writes_both_telegram_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SIGNALS_LINK", "https://example.com/x.md")
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        out = dss.write_outputs(_result_with([_sig_row("2330", 100.0, 1)]), str(tmp_path))
        for name in ("telegram_futures.txt", "telegram_stock.txt", "telegram.txt"):
            txt = (tmp_path / name).read_text(encoding="utf-8")
            assert txt.endswith("https://example.com/x.md") and len(txt) <= 4096
        assert out["telegram_futures"].endswith("telegram_futures.txt")


class TestWorkflowTwoLists:
    WF_PATH = TestWorkflowPositionsInput.WF_PATH

    def _wf(self):
        yaml = pytest.importorskip("yaml")
        with open(self.WF_PATH, encoding="utf-8") as f:
            wf = yaml.safe_load(f.read())
        return wf.get("on", wf.get(True)), wf["jobs"]["scan"]["steps"]

    def test_dispatch_inputs_defaults(self):
        on, _ = self._wf()
        inp = on["workflow_dispatch"]["inputs"]
        assert inp["stock_amount"]["default"] == "100000"
        assert inp["fut_min_volume"]["default"] == "100"
        assert inp["fut_min_oi"]["default"] == "300"
        assert inp["positions"]["default"] == "1"

    @pytest.mark.parametrize("env,expected", [
        ({}, {"--stock-amount": "100000", "--fut-min-volume": "100", "--fut-min-oi": "300"}),
        ({"STOCK_AMOUNT": "50000", "FUT_MIN_VOLUME": "250", "FUT_MIN_OI": "1000"},
         {"--stock-amount": "50000", "--fut-min-volume": "250", "--fut-min-oi": "1000"}),
    ])
    def test_scan_step_passes_args(self, env, expected):
        _, steps = self._wf()
        step = next(s for s in steps if "python daily_squeeze_signals.py" in s.get("run", ""))
        for k in ("STOCK_AMOUNT", "FUT_MIN_VOLUME", "FUT_MIN_OI"):
            assert "github.event.inputs" in step["env"][k] and "||" in step["env"][k]
        script = step["run"].replace("python daily_squeeze_signals.py", "echo")
        full_env = {"PATH": os.environ.get("PATH", ""), "AS_OF": "", "CAPITAL": "", "POSITIONS": "",
                    "STOCK_AMOUNT": "", "FUT_MIN_VOLUME": "", "FUT_MIN_OI": "", **env}
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=full_env, timeout=30)
        assert proc.returncode == 0, proc.stderr
        args = proc.stdout.split()
        for flag, val in expected.items():
            assert args[args.index(flag) + 1] == val
        dss.parse_args(args)  # CLI真的吃得下這些參數

    def test_preflight_runs_loader_tests(self):
        _, steps = self._wf()
        pre = next(s for s in steps if s.get("name") == "Pre-flight test")
        assert "test_taifex_futures_loader.py" in pre["run"] and "test_daily_squeeze_signals.py" in pre["run"]

    def _telegram_step(self):
        _, steps = self._wf()
        return next(s for s in steps if "api.telegram.org" in s.get("run", ""))

    def test_telegram_step_sends_both_files_futures_first(self):
        step = self._telegram_step()
        assert step["continue-on-error"] is True
        run = step["run"]
        assert run.index("signals/telegram_futures.txt") < run.index("signals/telegram_stock.txt")

    @pytest.mark.parametrize("responses,exit_code", [(["ok", "ok"], 0), (["fail", "ok"], 1), (["ok", "fail"], 1)])
    def test_telegram_script_one_failure_does_not_stop_other(self, tmp_path, responses, exit_code):
        """用假的curl實際跑Telegram步驟的bash腳本：兩個檔案都要送；任一失敗→這一步exit 1
        (workflow設了continue-on-error，所以整個job不會失敗)。"""
        run = self._telegram_step()["run"]
        sig = tmp_path / "signals"
        sig.mkdir()
        (sig / "telegram_futures.txt").write_text("期貨", encoding="utf-8")
        (sig / "telegram_stock.txt").write_text("現股", encoding="utf-8")
        bindir = tmp_path / "bin"
        bindir.mkdir()
        log = tmp_path / "curl.log"
        state = tmp_path / "n"
        state.write_text("0")
        (bindir / "curl").write_text(
            "#!/bin/bash\n"
            f'echo "$@" >> {log}\n'
            f"n=$(cat {state}); echo $((n+1)) > {state}\n"
            f"if [ \"$n\" = 0 ]; then r={responses[0]}; else r={responses[1]}; fi\n"
            'if [ "$r" = ok ]; then echo \'{"ok":true}\'; else echo \'{"ok":false}\'; exit 22; fi\n')
        (bindir / "curl").chmod(0o755)
        env = {"PATH": f"{bindir}:{os.environ.get('PATH', '')}", "TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}
        proc = subprocess.run(["bash", "-eo", "pipefail", "-c", run], capture_output=True, text=True,
                              env=env, cwd=tmp_path, timeout=30)
        assert proc.returncode == exit_code, proc.stdout + proc.stderr
        sent = log.read_text().splitlines()
        assert len(sent) == 2
        assert "text@signals/telegram_futures.txt" in sent[0] and "text@signals/telegram_stock.txt" in sent[1]

    def test_telegram_skipped_without_secrets(self, tmp_path):
        run = self._telegram_step()["run"]
        proc = subprocess.run(["bash", "-eo", "pipefail", "-c", run], capture_output=True, text=True,
                              env={"PATH": os.environ.get("PATH", "")}, cwd=tmp_path, timeout=30)
        assert proc.returncode == 0 and "略過" in proc.stdout

    def test_debug_dir_committed_and_uploaded(self):
        _, steps = self._wf()
        commit = next(s for s in steps if "git commit" in s.get("run", ""))
        assert "git add signals/" in commit["run"]
        upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact"))
        assert upload["with"]["path"] == "signals/"
