"""
winners_study.py 的測試——全部用合成資料/手寫HTML，不連網路。
"""
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import winners_study as ws

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW = os.path.join(HERE, ".github", "workflows", "winners_study.yml")


# ---------------------------------------------------------------------------
# 合成資料
# ---------------------------------------------------------------------------
def _ohlc_from_close(close, dates, volume, rng=None, adj_factor=None):
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
    adj = close * (adj_factor if adj_factor is not None else 1.0)
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Adj Close": adj,
                         "Volume": np.full(len(close), float(volume))}, index=pd.DatetimeIndex(dates))


def build_synthetic(n=40, seed=11, start="2021-01-01", end="2024-12-31"):
    """n檔股票：前6檔在2023~2024大漲(強勢+營收高成長)，第6檔流動性很差，其餘隨機漫步。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    win_start = dates.searchsorted(pd.Timestamp("2023-01-02"))
    rows, prices = [], {}
    for i in range(n):
        code = f"{1100 + i}"
        market = "上市" if i % 3 else "上櫃"
        industry = ["半導體業", "電子零組件業", "航運業", "食品工業"][i % 4]
        rows.append({"code": code, "name": f"測試{i}", "market": market, "industry": industry})
        drift = np.full(len(dates), rng.normal(0.0001, 0.0002))
        if i < 6:
            drift[win_start - 60:] = 0.0035      # 起點前已經開始轉強，區間內漲超過1倍
        rets = drift + rng.normal(0, 0.015, len(dates))
        close = (20 + 10 * i % 200) * np.exp(np.cumsum(rets))
        vol = 2_000_000 if i != 5 else 1_000   # 第5檔：成交金額太低 → 流動性篩選排除
        adj = np.where(dates < pd.Timestamp("2022-07-01"), 0.95, 1.0) if i == 7 else None
        prices[code] = _ohlc_from_close(close, dates, vol, rng, adj)
    uni = pd.DataFrame(rows)
    rev = []
    for i in range(n):
        code = f"{1100 + i}"
        for y in range(2021, 2025):
            for m in range(1, 13):
                hot = i < 6 and (y, m) >= (2022, 9)
                yoy = rng.normal(60 if hot else 5, 10)
                rev.append({"code": code, "name": f"測試{i}", "industry": "", "revenue": 1000.0 * (1 + yoy / 100),
                            "revenue_ly": 1000.0, "mom_pct": 0.0, "yoy_pct": yoy, "cum_yoy_pct": yoy,
                            "year": y, "month": m, "market": "上市"})
    rev = pd.DataFrame(rev)
    taiex = pd.Series(np.linspace(14000, 22000, len(dates)), index=dates, name="Close")
    return uni, prices, rev, taiex


def build_synthetic_extra(uni, prices, start="2023-01-01", seed=3, fail=()):
    """模擬 winners_extra.load_all_extra() 的回傳：前6檔(設計成大漲的)籌碼偏多、季報EPS成長高。"""
    import winners_extra as wx
    rng = np.random.default_rng(seed)
    cal = ws.build_calendar(prices)
    d0 = cal[cal.searchsorted(pd.Timestamp(start))]
    dates = wx.needed_dates(cal, d0)
    codes = list(uni["code"])
    hot = {f"{1100 + i}" for i in range(6)}
    chip, loaded, report = {}, {}, []
    for source, ds in dates.items():
        chip[source], loaded[source] = {}, set()
        for d in ds:
            ok = source not in fail
            report += [{"來源": wx.SOURCE_NAMES[source], "市場": m, "狀態": "ok" if ok else "HTTP 500"} for m in ("上市", "上櫃")]
            if not ok:
                continue
            loaded[source] |= {(d, "上市"), (d, "上櫃")}
            n = len(codes)
            h = np.array([c in hot for c in codes])
            if source == "insti":
                df = pd.DataFrame({"code": codes, "foreign_net": rng.normal(0, 50_000, n) + h * 150_000,
                                   "trust_net": rng.normal(0, 20_000, n) + h * 60_000})
                df["total_net"] = df["foreign_net"] + df["trust_net"]
            elif source == "margin":
                df = pd.DataFrame({"code": codes, "margin_bal": rng.uniform(1000, 5000, n), "short_bal": rng.uniform(0, 800, n)})
            elif source == "qfii":
                df = pd.DataFrame({"code": codes, "shares_issued": np.full(n, 5e7), "foreign_pct": rng.uniform(1, 60, n)})
            else:
                df = pd.DataFrame({"code": codes, "pe": rng.uniform(5, 40, n), "pb": rng.uniform(0.5, 5, n), "dy": rng.uniform(0, 6, n)})
            chip[source][d] = df
    qrows = []
    for i, c in enumerate(codes):
        g = 0.12 if c in hot else 0.0
        for y in range(2020, 2025):
            cum = np.zeros(5)
            for q in range(1, 5):
                k = (y - 2020) * 4 + q
                rev = 1000 * (1 + g) ** k * (1 + rng.normal(0, 0.03))
                gross = rev * (0.25 + (0.004 * k if c in hot else 0))
                op = gross * 0.5
                ni = op * 0.8
                cum += np.array([rev, gross, op, ni, ni / 100])
                qrows.append({"code": c, "year": y, "q": q, "market": "上市", "revenue": cum[0], "gross": cum[1],
                              "op_income": cum[2], "net_income": cum[3], "eps": cum[4], "equity": 20_000.0})
    qraw = pd.DataFrame(qrows) if "t163sb04" not in fail else pd.DataFrame()
    for kind in ("t163sb04", "t163sb05"):
        report += [{"來源": wx.SOURCE_NAMES[kind], "市場": m, "狀態": "HTTP 500" if kind in fail else "ok"}
                   for m in ("上市", "上櫃") for _ in range(17)]
    return {"chip_daily": chip, "loaded": loaded, "quarterly_raw": qraw, "report": report, "errors": [],
            "dates": dates, "estimated_requests": 120, "actual_requests": 130}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """保險：任何測試都不准真的連網路。"""
    import winners_extra as wx

    def boom(*a, **k):
        raise RuntimeError("測試中不准連網路")
    monkeypatch.setattr(wx, "_http", boom)
    monkeypatch.setattr(ws, "_http_get", boom)


@pytest.fixture(scope="module")
def synth():
    return build_synthetic()


@pytest.fixture(scope="module")
def study(synth):
    uni, prices, rev, taiex = synth
    return ws.run_study(uni, prices, rev, taiex, "2023-01-01", "2024-12-31", extra=build_synthetic_extra(uni, prices))


# ---------------------------------------------------------------------------
# ISIN 解析
# ---------------------------------------------------------------------------
ISIN_HTML = """<html><head><meta http-equiv="Content-Type" content="text/html; charset=MS950"></head><body>
<table class='h4' align=center cellSpacing=3 cellPadding=2 width=750 border=0>
<tr align=center><td bgcolor=#D5FFD5>有價證券代號及名稱 </td><td bgcolor=#D5FFD5>國際證券辨識號碼(ISIN Code)</td>
<td bgcolor=#D5FFD5>上市日</td><td bgcolor=#D5FFD5>市場別</td><td bgcolor=#D5FFD5>產業別</td>
<td bgcolor=#D5FFD5>CFICode</td><td bgcolor=#D5FFD5>備註</td></tr>
<tr><td bgcolor=#FAFAD2 colspan=7 ><B> 股票 <B> </td></tr>
<tr><td bgcolor=#FAFAD2>1101　台泥</td><td bgcolor=#FAFAD2>TW0001101004</td><td bgcolor=#FAFAD2>1962/02/09</td>
<td bgcolor=#FAFAD2>上市</td><td bgcolor=#FAFAD2>水泥工業</td><td bgcolor=#FAFAD2>ESVUFR</td><td bgcolor=#FAFAD2></td></tr>
<tr><td bgcolor=#FAFAD2>2330　台積電</td><td bgcolor=#FAFAD2>TW0002330008</td><td bgcolor=#FAFAD2>1994/09/05</td>
<td bgcolor=#FAFAD2>上市</td><td bgcolor=#FAFAD2>半導體業</td><td bgcolor=#FAFAD2>ESVUFR</td><td bgcolor=#FAFAD2></td></tr>
<tr><td bgcolor=#FAFAD2>2881A　富邦特</td><td bgcolor=#FAFAD2>TW0002881A08</td><td bgcolor=#FAFAD2>2016/12/22</td>
<td bgcolor=#FAFAD2>上市</td><td bgcolor=#FAFAD2>金融保險業</td><td bgcolor=#FAFAD2>EPNRAR</td><td bgcolor=#FAFAD2></td></tr>
<tr><td bgcolor=#FAFAD2>9999　怪怪股</td><td bgcolor=#FAFAD2>TW0009999000</td><td bgcolor=#FAFAD2>2020/01/01</td>
<td bgcolor=#FAFAD2>上市</td><td bgcolor=#FAFAD2>其他業</td><td bgcolor=#FAFAD2>EPNRAR</td><td bgcolor=#FAFAD2></td></tr>
<tr><td bgcolor=#FAFAD2 colspan=7 ><B> 上市認購(售)權證 <B> </td></tr>
<tr><td bgcolor=#FAFAD2>030001　台積電元大4A購01</td><td bgcolor=#FAFAD2>TW15Z0300011</td><td bgcolor=#FAFAD2>2024/01/02</td>
<td bgcolor=#FAFAD2>上市</td><td bgcolor=#FAFAD2></td><td bgcolor=#FAFAD2>RWSCCE</td><td bgcolor=#FAFAD2></td></tr>
<tr><td bgcolor=#FAFAD2 colspan=7 ><B> ETF <B> </td></tr>
<tr><td bgcolor=#FAFAD2>0050　元大台灣50</td><td bgcolor=#FAFAD2>TW0000050004</td><td bgcolor=#FAFAD2>2003/06/30</td>
<td bgcolor=#FAFAD2>上市</td><td bgcolor=#FAFAD2></td><td bgcolor=#FAFAD2>CEOGEU</td><td bgcolor=#FAFAD2></td></tr>
<tr><td bgcolor=#FAFAD2 colspan=7 ><B> 臺灣存託憑證(TDR) <B> </td></tr>
<tr><td bgcolor=#FAFAD2>9105　泰金寶-DR</td><td bgcolor=#FAFAD2>TW0009105004</td><td bgcolor=#FAFAD2>1999/01/01</td>
<td bgcolor=#FAFAD2>上市</td><td bgcolor=#FAFAD2>電子零組件業</td><td bgcolor=#FAFAD2>EDSDDR</td><td bgcolor=#FAFAD2></td></tr>
<tr><td bgcolor=#FAFAD2 colspan=7 ><B> 特別股 <B> </td></tr>
<tr><td bgcolor=#FAFAD2>1312A　國喬特</td><td bgcolor=#FAFAD2>TW0001312A03</td><td bgcolor=#FAFAD2>2000/01/01</td>
<td bgcolor=#FAFAD2>上市</td><td bgcolor=#FAFAD2>塑膠工業</td><td bgcolor=#FAFAD2>EPNRAR</td><td bgcolor=#FAFAD2></td></tr>
</table></body></html>"""


def _roundtrip_cp950(html):
    return html.encode("cp950").decode("cp950")


class TestIsinParsing:
    def test_keeps_only_common_stocks(self):
        df, rep = ws.parse_isin_html(_roundtrip_cp950(ISIN_HTML), "上市")
        assert list(df["code"]) == ["1101", "2330"]
        assert list(df["industry"]) == ["水泥工業", "半導體業"]
        assert list(df["name"]) == ["台泥", "台積電"]
        assert set(df["market"]) == {"上市"}
        assert rep["kept"] == 2
        # 權證/ETF/TDR/特別股都被排除，排除原因有記錄
        assert sum(rep["excluded"].values()) == 6
        assert "股票" in rep["sections"] and "ETF" in rep["sections"]

    def test_cfi_absent_is_ok(self):
        html = ISIN_HTML.replace("ESVUFR", "")
        df, _ = ws.parse_isin_html(html, "上櫃")
        assert list(df["code"]) == ["1101", "2330"]
        assert set(df["market"]) == {"上櫃"}

    def test_empty_html(self):
        df, rep = ws.parse_isin_html("<html>維護中</html>", "上市")
        assert df.empty and rep["kept"] == 0


# ---------------------------------------------------------------------------
# MOPS 月營收解析
# ---------------------------------------------------------------------------
def _mops_table(industry, rows):
    head = ("<table width=100% border=0><tr><th align=left class='tt'>產業別：" + industry + "</th></tr>"
            "<tr><td><table bordercolor=#FF6600 class='hasBorder' border=1 width=100%>"
            "<tr><th class='tt' rowspan=2>公司<br>代號</th><th class='tt' rowspan=2>公司名稱</th>"
            "<th class='tt' colspan=5>營業收入</th><th class='tt' colspan=3>累計營業收入</th>"
            "<th class='tt' rowspan=2>備註</th></tr>"
            "<tr><th>當月營收</th><th>上月營收</th><th>去年當月營收</th><th>上月比較<br>增減(%)</th>"
            "<th>去年同月<br>增減(%)</th><th>當月累計營收</th><th>去年累計營收</th><th>前期比較<br>增減(%)</th></tr>")
    body = "".join("<tr align=right>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    total = "<tr><th colspan=2>合計</th><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td></td></tr>"
    return head + body + total + "</table></td></tr></table>"


MOPS_HTML = ("<html><body><center>本資料由各公司申報<br>"
             + _mops_table("水泥工業", [
                 ["1101", "台泥", "10,000,000", "9,000,000", "8,000,000", "11.11", "25.00", "20,000,000", "17,000,000", "17.65", "-"],
                 ["1102", "亞泥", "5,000", "6,000", "10,000", "-16.67", "-50.00", "11,000", "20,000", "-45.00", ""],
             ])
             + _mops_table("半導體業", [
                 ["2330", "台積電", "300,000,000", "250,000,000", "200,000,000", "20.00", "", "550,000,000", "400,000,000", "37.50", "說明"],
                 ["2303", "聯電", "20,000,000", "21,000,000", "22,000,000", "-4.76", "-9.09", "41,000,000", "44,000,000", "-6.82", ""],
             ])
             + "</center></body></html>")


class TestMopsParsing:
    def test_two_industry_tables(self):
        df, rep = ws.parse_mops_revenue_html(_roundtrip_cp950(MOPS_HTML))
        assert rep["tables_with_data"] == 2 and rep["header_tables"] == 2 and rep["positional_tables"] == 0
        df = df.set_index("code")
        assert list(df.index) == ["1101", "1102", "2330", "2303"]   # 合計列不算
        assert df.loc["1101", "revenue"] == 10_000_000
        assert df.loc["1101", "revenue_ly"] == 8_000_000
        assert df.loc["1101", "yoy_pct"] == pytest.approx(25.0)
        assert df.loc["1101", "mom_pct"] == pytest.approx(11.11)
        assert df.loc["1102", "yoy_pct"] == pytest.approx(-50.0)      # 負的百分比
        assert df.loc["1102", "cum_yoy_pct"] == pytest.approx(-45.0)
        assert df.loc["2330", "yoy_pct"] == pytest.approx(50.0)       # 空白 → 用營收自己算
        assert df.loc["1101", "industry"] == "水泥工業"
        assert df.loc["2303", "industry"] == "半導體業"

    def test_positional_fallback_when_header_unrecognised(self):
        html = MOPS_HTML.replace("公司<br>代號", "代碼X").replace("當月營收", "本月")
        df, rep = ws.parse_mops_revenue_html(html)
        assert rep["positional_tables"] == 2
        assert df.set_index("code").loc["1102", "yoy_pct"] == pytest.approx(-50.0)

    def test_no_table(self):
        df, rep = ws.parse_mops_revenue_html("<html>查無資料</html>")
        assert df.empty and rep["rows"] == 0

    def test_num(self):
        assert ws._num("1,234,567") == 1234567
        assert ws._num("-12.34") == pytest.approx(-12.34)
        for s in ["", "-", "--", "N/A", None]:
            assert np.isnan(ws._num(s))


# ---------------------------------------------------------------------------
# point-in-time 營收
# ---------------------------------------------------------------------------
class TestRevenuePointInTime:
    def _rev(self):
        rows = []
        for (y, m, yoy) in [(2023, 9, 10), (2023, 10, 30), (2023, 11, 40), (2023, 12, 50), (2024, 1, 80)]:
            rows.append({"code": "1101", "year": y, "month": m, "yoy_pct": yoy})
        return ws.build_revenue_features(pd.DataFrame(rows))

    def test_available_date(self):
        assert ws.revenue_available_date(2024, 1) == pd.Timestamp("2024-02-11")
        assert ws.revenue_available_date(2023, 12) == pd.Timestamp("2024-01-11")

    def test_month_not_visible_before_11th(self):
        rf = self._rev()["1101"]
        r = ws.revenue_features_on_dates(rf, pd.to_datetime(["2024-02-10", "2024-02-11"]))
        # 2/10：最新只看得到2023-12(50)；2/11起才看得到2024-01(80)
        assert r["yoy"].iloc[0] == 50 and r["rev_month"].iloc[0] == pd.Timestamp("2023-12-01")
        assert r["yoy"].iloc[1] == 80 and r["rev_month"].iloc[1] == pd.Timestamp("2024-01-01")
        assert r["yoy3"].iloc[0] == pytest.approx((30 + 40 + 50) / 3)
        assert r["yoy3"].iloc[1] == pytest.approx((40 + 50 + 80) / 3)
        assert r["consec20"].iloc[1] == 4
        # 加速 = 最新3月均 − 前3月均(2023-9..11缺7、8月 → 前一段只有9~11月中可用的)
        assert np.isfinite(r["accel"].iloc[1])

    def test_before_any_revenue_and_stale(self):
        rf = self._rev()["1101"]
        r = ws.revenue_features_on_dates(rf, pd.to_datetime(["2023-10-10", "2024-12-31"]))
        assert np.isnan(r["yoy"].iloc[0])         # 9月營收10/11才公告
        assert np.isnan(r["yoy"].iloc[1])         # 最新已知是2024-01，太舊 → 不用

    def test_future_months_do_not_change_past(self):
        rows = [{"code": "1101", "year": 2023, "month": m, "yoy_pct": float(m)} for m in range(1, 13)]
        base = ws.build_revenue_features(pd.DataFrame(rows))["1101"]
        more = rows + [{"code": "1101", "year": 2024, "month": m, "yoy_pct": 999.0} for m in range(1, 6)]
        ext = ws.build_revenue_features(pd.DataFrame(more))["1101"]
        d = pd.to_datetime(["2023-06-15", "2024-01-05"])
        pd.testing.assert_frame_equal(ws.revenue_features_on_dates(base, d), ws.revenue_features_on_dates(ext, d))


# ---------------------------------------------------------------------------
# 價格整理、區間報酬、指標
# ---------------------------------------------------------------------------
def _flat_px(n=300, price=10.0, start="2022-01-03"):
    dates = pd.bdate_range(start, periods=n)
    return _ohlc_from_close(np.full(n, price), dates, 1_000_000)


class TestPricesAndReturns:
    def test_prepare_drops_bad_rows_and_adjusts(self):
        raw = _flat_px(5)
        raw.iloc[1, raw.columns.get_loc("Close")] = 0.0
        raw.iloc[2, raw.columns.get_loc("Open")] = np.nan
        raw["Adj Close"] = raw["Close"] * 0.5
        px = ws.prepare_price_frame(raw)
        assert len(px) == 3
        assert px["close"].iloc[0] == pytest.approx(5.0) and px["raw_close"].iloc[0] == 10.0
        assert px["open"].iloc[0] == pytest.approx(raw["Open"].iloc[0] * 0.5)
        assert px["turnover"].iloc[0] == pytest.approx(10.0 * 1_000_000)

    def test_window_return_and_max_gain(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        close = [10, 12, 25, 18, 15, 14]
        ind = ws.compute_indicators(ws.prepare_price_frame(_ohlc_from_close(close, dates, 1e6)))
        R, M, p0, p1 = ws.window_return(ind, dates[0], dates[-1])
        assert R == pytest.approx(0.4) and M == pytest.approx(1.5) and (p0, p1) == (10, 14)
        assert ws.window_return(ind, pd.Timestamp("2023-12-29"), dates[-1]) is None

    def test_entry_a_first_new_high_after_20_days(self):
        n = 330
        close = np.full(n, 10.0)
        close[280] = 11.0             # 第一次創252日新高 → 訊號
        close[281:] = 11.0
        close[282] = 11.5             # 前20天內已經有新高 → 不算
        close[282:] = 11.5
        close[310] = 12.0             # 距上次新高28天 → 再一次訊號
        close[311:] = 12.0
        dates = pd.bdate_range("2022-01-03", periods=n)
        ind = ws.compute_indicators(ws.prepare_price_frame(_ohlc_from_close(close, dates, 1e6)))
        assert list(np.flatnonzero(ind["entry_a"].to_numpy())) == [280, 310]
        assert ind["new_high"].iloc[282] == 1


# ---------------------------------------------------------------------------
# lift
# ---------------------------------------------------------------------------
class TestLift:
    def test_lift_math(self):
        cond = pd.Series([True] * 10 + [False] * 90)
        win = pd.Series([True] * 4 + [False] * 6 + [True] * 6 + [False] * 84)
        r = ws.lift_row(cond, win)
        assert r["n_cond"] == 10 and r["winners_cond"] == 4
        assert r["rate_cond"] == pytest.approx(0.4)
        assert r["base_rate"] == pytest.approx(0.10)
        assert r["lift"] == pytest.approx(4.0)
        assert r["coverage"] == pytest.approx(0.4)
        assert r["prevalence"] == pytest.approx(0.10)

    def test_common_feature_with_no_lift(self):
        cond = pd.Series([True] * 90 + [False] * 10)
        win = pd.Series(([True] + [False] * 9) * 10)
        r = ws.lift_row(cond, win)
        assert r["coverage"] == pytest.approx(0.9) and r["lift"] == pytest.approx(1.0)

    def test_nan_and_empty(self):
        r = ws.lift_row(pd.Series([np.nan, False]), pd.Series([True, False]))
        assert r["n_cond"] == 0 and np.isnan(r["lift"])


# ---------------------------------------------------------------------------
# 贏家路徑
# ---------------------------------------------------------------------------
class TestPathStats:
    def test_hand_built_path(self):
        pre = [10.0] * 260
        win = [10, 9, 8, 9, 10.5, 12, 15, 20, 25, 22, 20, 21, 26, 30, 28]
        close = np.array(pre + win, dtype=float)
        dates = pd.bdate_range("2022-01-03", periods=len(close))
        ind = ws.compute_indicators(ws.prepare_price_frame(_ohlc_from_close(close, dates, 1e6)))
        d0, d1 = dates[260], dates[-1]
        ps = ws.path_stats(ind, d0, d1)
        assert ps["low_date"] == dates[262]
        assert ps["move_start_date"] == dates[264]     # 10.5 > 前252日最高10
        assert ps["days_to_move_start"] == 4
        assert ps["days_to_double"] == 7               # 20 ≥ 2×10
        assert ps["max_drawdown"] == pytest.approx(min(8 / 10, 20 / 25) - 1)
        atr_peak25 = ind["atr14"].loc[dates[268]]
        atr_start = ind["atr14"].loc[dates[260]]
        exp = max((25 - 20) / atr_peak25, (10 - 8) / atr_start)
        assert ps["max_pullback_atr"] == pytest.approx(exp)
        # 起漲後MA60仍在10附近、收盤都>MA60
        assert ps["below_ma60_after_start"] == 0.0

    def test_below_ma60_after_start(self):
        close = np.array([10.0] * 260 + [11, 12, 13, 5, 6], dtype=float)
        dates = pd.bdate_range("2022-01-03", periods=len(close))
        ind = ws.compute_indicators(ws.prepare_price_frame(_ohlc_from_close(close, dates, 1e6)))
        ps = ws.path_stats(ind, dates[259], dates[-1])
        assert ps["move_start_date"] == dates[260]
        assert ps["below_ma60_after_start"] == 1.0
        assert np.isnan(ps["days_to_double"])


# ---------------------------------------------------------------------------
# 規則模擬
# ---------------------------------------------------------------------------
def _arr(o, h, l, c, atr=1.0, ma60=None):
    n = len(c)
    return {"open": np.array(o, float), "high": np.array(h, float), "low": np.array(l, float),
            "close": np.array(c, float), "atr14": np.full(n, atr, float),
            "ma60": np.array(ma60 if ma60 is not None else [np.nan] * n, float)}


class TestRuleSimulation:
    def test_costs(self):
        pnl, ret = ws.trade_pnl(100, 110)
        exp = 1000 * 110 * (1 - 0.001425 - 0.003) - 100000 * 1.001425
        assert pnl == pytest.approx(exp) and ret == pytest.approx(exp / 100000)
        pnl0, _ = ws.trade_pnl(50, 50)
        assert pnl0 == pytest.approx(-100000 * (0.001425 * 2 + 0.003))

    def test_x1_trailing_atr(self):
        # 進場價10、ATR=1：收盤15 → 移動停損12；隔天最低11.5 → 停損價12出
        a = _arr(o=[9, 10, 11, 13, 14.5, 13], h=[9, 11, 13, 15, 15, 13], l=[9, 9.5, 10.8, 12.8, 11.5, 12],
                 c=[9, 10.5, 12.5, 15, 13, 12.5])
        x, px, why = ws.simulate_exit(a, 1, 5, 1.0, "X1")
        assert (x, px, why) == (4, 12.0, "移動停損")

    def test_initial_stop_and_gap(self):
        a = _arr(o=[10, 10, 9], h=[10, 10.2, 9], l=[10, 8.5, 7], c=[10, 9, 7.5])
        assert ws.simulate_exit(a, 1, 2, 1.0, "X1") == (2, 8.0, "初始停損")   # 進場10−2ATR=8，盤中碰到
        a2 = _arr(o=[10, 10, 7.5], h=[10, 10.2, 7.8], l=[10, 9.5, 7], c=[10, 9, 7.5])
        x, px, why = ws.simulate_exit(a2, 1, 2, 1.0, "X1")
        assert (x, px, why) == (2, 7.5, "初始停損")                           # 開盤跳空低於停損 → 開盤價

    def test_x2_trailing_pct(self):
        # 收盤20 → 停損16(>初始8)；隔天最低15 → 16出
        a = _arr(o=[10, 10, 14, 19, 18], h=[10, 14, 19, 20.5, 18], l=[10, 9.8, 13.5, 18.5, 15],
                 c=[10, 13.5, 18.5, 20, 15.5])
        x, px, why = ws.simulate_exit(a, 1, 4, 1.0, "X2")
        assert (x, px, why) == (4, 16.0, "移動停損")

    def test_x3_close_below_ma60_exit_next_open(self):
        a = _arr(o=[10, 10, 10.5, 10.2, 9.7], h=[10, 10.8, 10.6, 10.3, 9.9], l=[10, 9.9, 10, 9.6, 9.5],
                 c=[10, 10.5, 10.1, 9.7, 9.8], ma60=[9, 9.5, 9.9, 9.9, 9.9])
        x, px, why = ws.simulate_exit(a, 1, 4, 1.0, "X3")
        assert (x, px, why) == (4, 9.7, "跌破MA60")

    def test_x4_target_and_stop(self):
        a = _arr(o=[10, 10, 12, 13.5], h=[10, 11, 13, 14.5], l=[10, 9.6, 11.8, 13.2], c=[10, 10.8, 12.9, 14])
        assert ws.simulate_exit(a, 1, 3, 1.0, "X4") == (3, 14.0, "停利")       # 10+4ATR
        a_gap = _arr(o=[10, 10, 15], h=[10, 11, 16], l=[10, 9.6, 14.8], c=[10, 10.8, 15.5])
        assert ws.simulate_exit(a_gap, 1, 2, 1.0, "X4") == (2, 15.0, "停利")   # 跳空開在停利之上 → 開盤價
        a_stop = _arr(o=[10, 10, 9.5], h=[10, 10.4, 9.6], l=[10, 9.6, 8.4], c=[10, 10.1, 8.6])
        assert ws.simulate_exit(a_stop, 1, 2, 1.0, "X4") == (2, 8.5, "初始停損")  # 10−1.5ATR
        a_both = _arr(o=[10, 10, 10], h=[10, 10.4, 14.5], l=[10, 9.6, 8.0], c=[10, 10.1, 12])
        assert ws.simulate_exit(a_both, 1, 2, 1.0, "X4")[2] == "初始停損"     # 同一天都碰到 → 保守算停損

    def test_window_end_close_out(self):
        a = _arr(o=[10, 10, 10.5, 11], h=[10, 10.6, 11, 11.5], l=[10, 9.8, 10.3, 10.8], c=[10, 10.5, 10.9, 11.2])
        for kind in ["X1", "X2", "X3", "X4"]:
            assert ws.simulate_exit(a, 1, 3, 1.0, kind) == (3, 11.2, "期末平倉")

    def test_simulate_rule_entries(self):
        n = 10
        c = [10.0] * n
        a = _arr(o=c, h=[x * 1.01 for x in c], l=[x * 0.99 for x in c], c=c)
        sig = np.zeros(n, bool)
        sig[[0, 2, 9]] = True   # 2：持倉中(單一部位) → 略過；9：最後一天的訊號沒有隔天可進場
        trades = ws.simulate_rule(a, sig, 0, 9, "X1")
        assert len(trades) == 1
        t = trades[0]
        assert t["entry_idx"] == 1 and t["exit_idx"] == 9 and t["exit_reason"] == "期末平倉"
        assert t["hold_days"] == 8
        assert t["pnl"] == pytest.approx(ws.trade_pnl(10, 10)[0])

    def test_reentry_after_exit(self):
        o = [10, 10, 7, 7, 7, 7]
        a = _arr(o=o, h=[x + 0.1 for x in o], l=[x - 0.1 for x in o], c=o)
        sig = np.zeros(6, bool)
        sig[[0, 2, 3]] = True
        trades = ws.simulate_rule(a, sig, 0, 5, "X1")
        assert [t["entry_idx"] for t in trades] == [1, 3]   # 第2天停損出場，當天收盤訊號 → 第3天進場
        assert trades[0]["exit_reason"] == "初始停損" and trades[0]["exit_price"] == 7


# ---------------------------------------------------------------------------
# 整體研究(合成資料)
# ---------------------------------------------------------------------------
class TestRunStudy:
    def test_liquidity_filter_and_counts(self, study):
        m = study["meta"]
        assert m["excluded"]["流動性不足"] == 1
        assert "1105" not in set(study["features"]["code"])
        assert m["included"] == 39

    def test_winners_found(self, study):
        f = study["features"]
        assert f["w100"].sum() >= 4
        # 設計成大漲的前5檔(第6檔1105流動性不足被排除)都要是漲1倍贏家
        assert {"1100", "1101", "1102", "1103", "1104"} <= set(f.loc[f["w100"], "code"])
        assert (f["M"] >= f["R"] - 1e-12).all()

    def test_lift_table_shape(self, study):
        lift = study["lift"]
        assert set(lift["tier"]) == {"漲1倍以上", "漲2倍以上", "曾經漲1倍"}
        assert (lift["n_cond"] >= lift["winners_cond"]).all()
        rev = lift[(lift["condition"] == "營收3月均YoY>20%") & (lift["tier"] == "漲1倍以上")].iloc[0]
        assert rev["lift"] > 1

    def test_rules_and_trades(self, study):
        rules = study["rules"]
        assert len(rules) == 16
        assert rules["trades"].sum() == len(study["trades"])
        tr = study["trades"]
        assert (tr["entry_date"] > tr["signal_date"]).all()
        assert (tr["exit_date"] >= tr["entry_date"]).all()
        assert tr["exit_date"].max() <= study["meta"]["d1"]
        assert tr["signal_date"].min() >= study["meta"]["d0"]

    def test_features_at_start_have_no_lookahead(self, synth, study):
        uni, prices, rev, taiex = synth
        d0 = study["meta"]["d0"]
        cut = {c: df[df.index <= d0] for c, df in prices.items()}
        # 把股價截到起點、再接上一段完全不同的未來K棒 → 起點特徵必須一模一樣
        rng = np.random.default_rng(99)
        fut_dates = pd.bdate_range(d0 + pd.Timedelta(days=1), periods=40)
        ext = {}
        for c, df in cut.items():
            fut = _ohlc_from_close(df["Close"].iloc[-1] * np.exp(np.cumsum(rng.normal(0, 0.1, 40))), fut_dates, 5e6, rng)
            ext[c] = pd.concat([df, fut])
        rev_cut = rev[(rev["year"] < 2023)]
        res2 = ws.run_study(uni, ext, rev_cut, taiex, d0, fut_dates[-1])
        cols = ["ret_3m", "ret_6m", "ret_12m", "above_ma60", "above_ma240", "dist_52w_high", "new_high_20d",
                "vol_60d", "turnover_20d", "turnover_trend", "price", "rev_yoy_3m", "rev_yoy_latest", "rs_rank"]
        a = study["features"].set_index("code")[cols].sort_index()
        b = res2["features"].set_index("code")[cols].sort_index()
        pd.testing.assert_frame_equal(a, b)

    def test_extra_features_at_start_have_no_lookahead(self, synth, study):
        """籌碼/估值/季報/進階技術面：起點之後的K棒與季報全部換掉，起點特徵不變。"""
        uni, prices, rev, taiex = synth
        d0 = study["meta"]["d0"]
        extra = build_synthetic_extra(uni, prices)
        rng = np.random.default_rng(5)
        ext = {}
        for c, df in prices.items():
            cut = df[df.index <= d0]
            fut_dates = pd.bdate_range(d0 + pd.Timedelta(days=1), periods=60)
            fut = _ohlc_from_close(cut["Close"].iloc[-1] * np.exp(np.cumsum(rng.normal(0, 0.1, 60))), fut_dates, 9e6, rng)
            ext[c] = pd.concat([cut, fut])
        q = extra["quarterly_raw"].copy()
        later = (q["year"] >= 2023) | ((q["year"] == 2022) & (q["q"] == 4))   # 起點(2023-01-02)時還沒公告的季：亂改
        q.loc[later, ["revenue", "gross", "op_income", "net_income", "eps", "equity"]] *= 7.0
        extra2 = dict(extra, quarterly_raw=q)
        a = ws.run_study(uni, prices, rev, taiex, "2023-01-01", "2024-12-31", extra=extra, n_perm=5)
        b = ws.run_study(uni, ext, rev, taiex, d0, ext["1100"].index[-1], extra=extra2, n_perm=5)
        import winners_extra as wx
        cols = wx.TECH_COLS + [c for c in wx.CHIP_COLS] + wx.Q_FEAT_COLS + ["quarter_at_start", "rev_high12"]
        x = a["features"].set_index("code")[cols].sort_index()
        y = b["features"].set_index("code")[cols].sort_index()
        pd.testing.assert_frame_equal(x, y)
        assert (x["quarter_at_start"] == "2022Q3").all()        # 2023-01-02 時最新已公告季是2022Q3

    def test_indicator_rows_unchanged_by_future(self, synth):
        _, prices, _, _ = synth
        df = prices["1100"]
        d = df.index[500]
        full = ws.compute_indicators(ws.prepare_price_frame(df))
        part = ws.compute_indicators(ws.prepare_price_frame(df[df.index <= d]))
        assert ws.features_at(full, d) == pytest.approx(ws.features_at(part, d), nan_ok=True)

    def test_no_revenue_still_runs(self, synth):
        uni, prices, _, taiex = synth
        res = ws.run_study(uni, prices, pd.DataFrame(), taiex, "2023-01-01", "2024-12-31")
        r = res["rules"].set_index("rule")
        assert r.loc["E_B+X1", "trades"] == 0 and r.loc["E_D+X3", "trades"] == 0
        assert r.loc["E_A+X1", "trades"] > 0


# ---------------------------------------------------------------------------
# main 端到端(下載函式全部換成合成資料)
# ---------------------------------------------------------------------------
def _patch_loaders(monkeypatch, synth, rev_ok=True, uni_empty=False, extra_fail=()):
    uni, prices, rev, taiex = synth
    calls = {}

    def fake_extra(cal, d0, d1, cache_dir, refresh, debug_dir):
        calls["extra"] = (d0, d1)
        if extra_fail == "all":
            raise RuntimeError("模擬整體失敗")
        return build_synthetic_extra(uni, prices, start=str(d0.date()), fail=extra_fail)
    monkeypatch.setattr(ws, "load_extra", fake_extra)

    def fake_uni(cache_dir, refresh, debug_dir, today_str):
        calls["refresh"] = refresh
        return (uni.iloc[0:0] if uni_empty else uni.copy()), []

    def fake_prices(universe, dl_start, dl_end, cache_dir, refresh, debug_dir):
        calls["dl"] = (dl_start, dl_end)
        return {c: prices[c] for c in universe["code"]}, pd.DataFrame(columns=["代號"])

    def fake_rev(start_month, end_month, cache_dir, refresh, debug_dir):
        calls["rev"] = (start_month, end_month)
        return (rev, True) if rev_ok else (pd.DataFrame(), False)

    monkeypatch.setattr(ws, "load_universe", fake_uni)
    monkeypatch.setattr(ws, "load_prices", fake_prices)
    monkeypatch.setattr(ws, "load_revenue", fake_rev)
    monkeypatch.setattr(ws, "load_taiex", lambda *a, **k: taiex)
    return calls


class TestMainEndToEnd:
    def test_outputs(self, monkeypatch, tmp_path, synth):
        calls = _patch_loaders(monkeypatch, synth)
        out = tmp_path / "results_winners"
        rc = ws.main(["--output-dir", str(out), "--start", "2023-01-01", "--end", "2024-12-31",
                      "--cache-dir", str(tmp_path / "cache")])
        assert rc == 0
        for fn in ["summary_winners.txt", "winners_list.csv", "features_lift.csv", "industry_lift.csv",
                   "rules_compare.csv", "rule_trades.csv"]:
            assert (out / fn).exists(), fn
        assert (out / "debug").is_dir()
        assert calls["dl"] == ("2021-11-27", "2025-01-01")
        assert calls["rev"][0] == pd.Timestamp("2021-10-01")
        text = (out / "summary_winners.txt").read_text(encoding="utf-8")
        for sec in ["【資料與偏誤說明】", "【贏家名單概況】", "【特徵lift表】", "【產業表】", "【贏家路徑統計】",
                    "【簡單規則比較表】", "【結論提示】", "存活者偏誤", "樣本內", "執行時間"]:
            assert sec in text, sec
        wl = pd.read_csv(out / "winners_list.csv", encoding="utf-8-sig", dtype={"代號": str})
        for col in ["代號", "名稱", "市場", "產業別", "區間報酬R", "區間最大漲幅M", "起點還原收盤", "最大回檔ATR倍數"]:
            assert col in wl.columns
        rc_df = pd.read_csv(out / "rules_compare.csv", encoding="utf-8-sig")
        assert len(rc_df) == 16 and "扣除前5大獲利後損益" in rc_df.columns

    def test_extra_sections_and_columns(self, monkeypatch, tmp_path, synth):
        _patch_loaders(monkeypatch, synth, extra_fail=("margin",))
        out = tmp_path / "rx"
        assert ws.main(["--output-dir", str(out), "--start", "2023-01-01", "--end", "2024-12-31",
                        "--cache-dir", str(tmp_path / "c")]) == 0
        text = (out / "summary_winners.txt").read_text(encoding="utf-8")
        assert text.index("【資料來源狀態】") < text.index("【資料與偏誤說明】")
        for sec in ["── 技術面", "── 籌碼面", "── 基本面", "── 產業", "【雙條件組合】", "【多重比較警告】",
                    "洗牌基準", "集保大戶持股", "⚠️⚠️ 完全失敗  融資融券", "OK  三大法人買賣超", "預估至少 120 次",
                    "起漲時最新已公告季EPS年增率"]:
            assert sec in text, sec
        assert (out / "features_pairs.csv").exists() and (out / "debug" / "shuffle_baseline.csv").exists()
        sb = pd.read_csv(out / "debug" / "shuffle_baseline.csv", encoding="utf-8-sig")
        assert len(sb) == 200
        lift = pd.read_csv(out / "features_lift.csv", encoding="utf-8-sig")
        assert set(lift["面向"]) == {"技術面", "籌碼面", "基本面", "產業"}
        assert lift.loc[lift["條件"].str.contains("融資餘額"), "符合檔數"].max() == 0   # 融資失敗 → 沒有股票符合
        for fn in ["winners_list.csv", "debug/all_stocks_features.csv"]:
            df = pd.read_csv(out / fn, encoding="utf-8-sig")
            for c in ["外資20日買超/成交量(%)", "投信連續買超天數(最多20)", "EPS年增率(%)", "ROE近四季(%)", "本益比",
                      "均線多頭排列(MA5>20>60>120)", "對加權指數beta(250日)", "電子相關產業", "月營收創近12個月新高"]:
                assert c in df.columns, (fn, c)
        wl = pd.read_csv(out / "winners_list.csv", encoding="utf-8-sig")
        assert "起漲時EPS年增率(%)" in wl.columns and wl["起漲時RSI14"].notna().any()

    def test_extra_total_failure_still_runs(self, monkeypatch, tmp_path, synth):
        _patch_loaders(monkeypatch, synth, extra_fail="all")
        out = tmp_path / "rf"
        assert ws.main(["--output-dir", str(out), "--start", "2023-01-01", "--end", "2024-12-31",
                        "--cache-dir", str(tmp_path / "c")]) == 0
        text = (out / "summary_winners.txt").read_text(encoding="utf-8")
        assert "籌碼/估值/季報整體失敗" in text and text.count("⚠️⚠️ 完全失敗") == 6
        assert "沒有條件符合≥30檔" in text      # 籌碼面整個空掉

    def test_skip_extra_flag(self, monkeypatch, tmp_path, synth):
        calls = _patch_loaders(monkeypatch, synth)
        out = tmp_path / "rs"
        assert ws.main(["--output-dir", str(out), "--start", "2023-01-01", "--end", "2024-12-31",
                        "--cache-dir", str(tmp_path / "c"), "--skip-extra"]) == 0
        assert "extra" not in calls
        assert "本次沒有抓取" in (out / "summary_winners.txt").read_text(encoding="utf-8")

    def test_revenue_missing_warns(self, monkeypatch, tmp_path, synth):
        _patch_loaders(monkeypatch, synth, rev_ok=False)
        out = tmp_path / "r"
        assert ws.main(["--output-dir", str(out), "--start", "2023-01-01", "--end", "2024-12-31",
                        "--cache-dir", str(tmp_path / "c")]) == 0
        text = (out / "summary_winners.txt").read_text(encoding="utf-8")
        assert "月營收(MOPS)完全抓不到" in text

    def test_universe_empty_fails_closed(self, monkeypatch, tmp_path, synth):
        _patch_loaders(monkeypatch, synth, uni_empty=True)
        out = tmp_path / "r"
        assert ws.main(["--output-dir", str(out), "--cache-dir", str(tmp_path / "c")]) == 1
        assert "fail closed" in (out / "summary_winners.txt").read_text(encoding="utf-8")

    def test_window_preset_and_default_two_years(self, monkeypatch, tmp_path, synth):
        calls = _patch_loaders(monkeypatch, synth)
        # 2018-2019 合成資料沒有 → run_study 會丟錯；這裡只檢查參數換算
        monkeypatch.setattr(ws, "run_study", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
        with pytest.raises(RuntimeError):
            ws.main(["--output-dir", str(tmp_path / "x"), "--window", "2018-2019", "--cache-dir", str(tmp_path / "c")])
        assert calls["dl"] == ((pd.Timestamp("2018-01-01") - pd.Timedelta(days=400)).date().isoformat(), "2020-01-01")
        s, e = ws.resolve_window("", "2024-06-30")
        assert (s, e) == (pd.Timestamp("2022-06-30"), pd.Timestamp("2024-06-30"))


# ---------------------------------------------------------------------------
# 其他：不依賴scipy、workflow接線
# ---------------------------------------------------------------------------
def test_no_scipy_import():
    src = open(os.path.join(HERE, "winners_study.py"), encoding="utf-8").read()
    assert not re.search(r"^\s*(import|from)\s+scipy", src, re.M)
    code = "import sys, winners_study; print('scipy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "False", out.stderr


class TestWorkflow:
    def _wf(self):
        yaml = pytest.importorskip("yaml")
        with open(WORKFLOW, encoding="utf-8") as fh:
            text = fh.read()
        return text, yaml.safe_load(text)

    def _run_step(self, wf):
        steps = wf["jobs"]["study"]["steps"]
        return next(s for s in steps if "python winners_study.py" in s.get("run", ""))

    def test_parses_and_inputs(self):
        text, wf = self._wf()
        on = wf.get("on", wf.get(True))
        inputs = on["workflow_dispatch"]["inputs"]
        assert inputs["window"]["type"] == "choice"
        assert inputs["window"]["options"] == list(ws.WINDOW_PRESETS)
        assert inputs["window"]["default"] == "最近兩年"
        assert inputs["start"]["default"] == "" and inputs["end"]["default"] == ""
        assert inputs["refresh_data"]["type"] == "boolean"
        job = wf["jobs"]["study"]
        assert 80 <= job["timeout-minutes"] <= 100
        names = [s.get("name", "") for s in job["steps"]]
        runs = [s.get("run", "") or "" for s in job["steps"]]
        i_test = next(i for i, r in enumerate(runs) if "pytest" in r and "test_winners_study.py" in r
                      and "test_winners_extra.py" in r)
        i_run = next(i for i, r in enumerate(runs) if "python winners_study.py" in r)
        assert i_test < i_run
        up = next(s for s in job["steps"] if "upload-artifact" in s.get("uses", ""))
        assert up["with"]["path"].rstrip("/") == "results_winners"
        assert any("cache" in s.get("uses", "") for s in job["steps"])
        assert "Upload results" in names

    @pytest.mark.parametrize("window,start,end,refresh,expected", [
        ("最近兩年", "", "", "false", "--output-dir results_winners --cache-dir winners_cache"),
        ("2022-2023", "", "", "false", "--output-dir results_winners --cache-dir winners_cache --start 2022-01-01 --end 2023-12-31"),
        ("2020-2021", "", "", "true", "--output-dir results_winners --cache-dir winners_cache --start 2020-01-01 --end 2021-12-31 --refresh"),
        ("2018-2019", "2018-03-01", "", "false", "--output-dir results_winners --cache-dir winners_cache --start 2018-03-01 --end 2019-12-31"),
        ("最近兩年", "", "2025-06-30", "false", "--output-dir results_winners --cache-dir winners_cache --end 2025-06-30"),
    ])
    def test_run_script_maps_window(self, window, start, end, refresh, expected):
        _, wf = self._wf()
        step = self._run_step(wf)
        assert step["env"]["WINDOW"] == "${{ github.event.inputs.window }}"
        assert step["env"]["REFRESH"] == "${{ github.event.inputs.refresh_data }}"
        script = step["run"].replace("python winners_study.py", "echo ARGS=")
        env = {"PATH": os.environ.get("PATH", ""), "WINDOW": window, "START": start, "END": end, "REFRESH": refresh}
        proc = subprocess.run(["bash", "-e", "-c", script], capture_output=True, text=True, env=env, timeout=30)
        assert proc.returncode == 0, proc.stderr
        line = [l for l in proc.stdout.splitlines() if l.startswith("ARGS=")][0]
        assert line[len("ARGS="):].strip() == expected

    def test_presets_match_workflow_case(self):
        text, _ = self._wf()
        for name, (s, e) in ws.WINDOW_PRESETS.items():
            if s:
                assert f'"{name}") S="{s}"; E="{e}"' in text


# ---------------------------------------------------------------------------
# 下載函式(網路層換成假的回應；檢查快取、debug報告、失敗紀錄)
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.content = text.encode("cp950")


class TestLoaders:
    def test_load_revenue_cache_and_report(self, monkeypatch, tmp_path):
        calls = []

        def fake_get(url):
            calls.append(url)
            if "mopsov" in url and "_1_0" in url:
                return _Resp(200, MOPS_HTML)
            return _Resp(404)
        monkeypatch.setattr(ws, "_http_get", fake_get)
        monkeypatch.setattr(ws, "MOPS_DELAY", 0)
        dbg = tmp_path / "debug"
        rev, ok = ws.load_revenue(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-15"), str(tmp_path / "c"), False, str(dbg))
        assert ok
        assert set(rev["market"]) == {"上市", "上櫃"} and set(rev["month"]) == {1}
        assert calls[0] == "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_113_1_0.html"
        assert "https://mopsov.twse.com.tw/nas/t21/otc/t21sc03_113_1_0.html" in calls
        rep = pd.read_csv(dbg / "revenue_parse_report.csv", encoding="utf-8-sig")
        assert len(rep) == 8 and set(rep["狀態"]) == {"ok", "HTTP 404"}   # 2個月 × 2市場 × 國內/KY
        assert set(rep.loc[rep["狀態"] == "ok", "種類"]) == {"國內"}
        assert "https://mops.twse.com.tw/nas/t21/sii/t21sc03_113_1_1.html" in calls   # KY頁：兩個網域都試
        assert (dbg / "mops_sample_sii0_ok.html").exists() and (dbg / "revenue_samples.txt").exists()
        n_calls = len(calls)
        rev2, ok2 = ws.load_revenue(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"), str(tmp_path / "c"), False, str(dbg))
        new_calls = calls[n_calls:]
        assert ok2 and not any(u.endswith("_113_1_0.html") for u in new_calls)   # 一月國內頁已有快取，不再連線
        assert any(u.endswith("_113_1_1.html") for u in new_calls)              # 沒抓到的(KY頁)不快取、會重試
        assert len(rev2) == len(rev)

    def test_load_revenue_total_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ws, "_http_get", lambda url: _Resp(200, "<html>系統維護</html>"))
        monkeypatch.setattr(ws, "MOPS_DELAY", 0)
        rev, ok = ws.load_revenue(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"), str(tmp_path / "c"), False, str(tmp_path / "d"))
        assert not ok and rev.empty
        assert not os.listdir(tmp_path / "c" / "revenue")   # 失敗不寫快取

    def test_load_universe(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ws, "_http_get", lambda url: _Resp(200, ISIN_HTML))
        uni, warns = ws.load_universe(str(tmp_path / "c"), False, str(tmp_path / "d"), "2026-10-10")
        assert warns == []
        assert list(uni["code"]) == ["1101", "2330"]    # 兩個市場同代號 → 去重
        assert (tmp_path / "d" / "universe_report.txt").exists()

    def test_load_universe_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ws, "_http_get", lambda url: _Resp(500))
        monkeypatch.setattr(ws.time, "sleep", lambda s: None)
        uni, warns = ws.load_universe(str(tmp_path / "c"), False, str(tmp_path / "d"), "2026-10-10")
        assert uni.empty and len(warns) == 2

    def test_load_prices_records_failures(self, monkeypatch, tmp_path):
        import types
        dates = pd.bdate_range("2024-01-01", periods=5)
        seen = []

        def fake_download(tickers, **kw):
            seen.append(list(tickers))
            assert kw["auto_adjust"] is False
            frames = {}
            for t in tickers:
                if t.startswith("2330"):
                    continue          # 永遠抓不到
                frames[t] = _ohlc_from_close([10, 11, 12, 13, 14], dates, 1e6)
            return pd.concat(frames, axis=1) if frames else pd.DataFrame()
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
        monkeypatch.setattr(ws, "PRICE_BATCH_DELAY", 0)
        uni = pd.DataFrame({"code": ["1101", "2330", "6488"], "market": ["上市", "上市", "上櫃"]})
        out, fail = ws.load_prices(uni, "2024-01-01", "2024-01-08", str(tmp_path / "c"), False, str(tmp_path / "d"))
        assert set(out) == {"1101", "6488"}
        assert seen[0] == ["1101.TW", "2330.TW", "6488.TWO"] and seen[1] == ["2330.TW"]   # 第二輪只重抓失敗的
        assert list(fail["代號"]) == ["2330"]
        assert "Adj Close" in out["1101"].columns
        out2, _ = ws.load_prices(uni, "2024-01-01", "2024-01-08", str(tmp_path / "c"), False, str(tmp_path / "d"))
        assert seen[-1] == ["2330.TW"] and set(out2) == {"1101", "6488"}               # 其他讀快取
