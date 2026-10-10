"""
winners_extra.py 的測試(籌碼/估值/季報解析、季報point-in-time、進階技術面、籌碼特徵、雙條件lift、洗牌基準)。
全部是手寫的fixture，不連網路。fixture的JSON/HTML格式是「推測」的真實格式，第一次在Actions執行後要用debug/對照。
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import winners_extra as wx
import winners_study as ws
from test_winners_study import _ohlc_from_close


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("測試中不准連網路")
    monkeypatch.setattr(wx, "_http", boom)
    monkeypatch.setattr(ws, "_http_get", boom)


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------
T86 = {"stat": "OK", "date": "20240102", "fields": [
    "證券代號", "證券名稱", "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)", "外陸資買賣超股數(不含外資自營商)",
    "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數", "投信買進股數", "投信賣出股數", "投信買賣超股數",
    "自營商買賣超股數", "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)",
    "自營商買進股數(避險)", "自營商賣出股數(避險)", "自營商買賣超股數(避險)", "三大法人買賣超股數"],
    "data": [["2330", "台積電", "10,000", "4,000", "6,000", "0", "0", "-50", "1,000", "0", "1,000", "-200",
              "0", "0", "0", "0", "0", "0", "6,800"],
             ["1101", "台泥", "0", "3,000", "-3,000", "0", "0", "0", "0", "500", "-500", "0", "0", "0", "0", "0", "0", "0", "-3,500"],
             ["030001", "權證", "1", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1"]]}

TPEX_INSTI_OLD = {"reportDate": "113/01/02", "iTotalRecords": 1, "aaData": [
    ["6488", "環球晶", "5,000", "1,000", "4,000", "0", "0", "0", "5,000", "1,000", "4,000",
     "300", "100", "200", "0", "0", "0", "0", "0", "0", "0", "0", "-10", "4,190"]]}

TPEX_INSTI_NEW = {"stat": "ok", "tables": [{"title": "三大法人", "fields": [
    "代號", "名稱", "外資及陸資(不含外資自營商)-買進股數", "外資及陸資(不含外資自營商)-賣出股數",
    "外資及陸資(不含外資自營商)-買賣超股數", "投信-買賣超股數", "自營商-買賣超股數", "三大法人買賣超股數合計"],
    "data": [["6488", "環球晶", "5,000", "1,000", "4,000", "200", "-10", "4,190"]]}]}


class TestJsonParsers:
    def test_t86_by_field_name(self):
        df, method = wx.parse_insti(T86, "上市")
        assert method == "欄位名稱"
        d = df.set_index("code")
        assert list(d.index) == ["2330", "1101"]                    # 6位數權證排除
        assert d.loc["2330", "foreign_net"] == 6000                  # 不含外資自營商那一欄
        assert d.loc["2330", "trust_net"] == 1000
        assert d.loc["2330", "total_net"] == 6800
        assert d.loc["1101", "trust_net"] == -500

    def test_tpex_insti_old_positional(self):
        df, method = wx.parse_insti(TPEX_INSTI_OLD, "上櫃")
        assert method == "位置(猜測)"
        r = df.iloc[0]
        assert (r["code"], r["foreign_net"], r["trust_net"], r["total_net"]) == ("6488", 4000, 200, 4190)

    def test_tpex_insti_new_tables(self):
        df, method = wx.parse_insti(TPEX_INSTI_NEW, "上櫃")
        assert method == "欄位名稱"
        r = df.iloc[0]
        assert (r["foreign_net"], r["trust_net"], r["total_net"]) == (4000, 200, 4190)

    def test_margin_twse_tables_two_balances(self):
        payload = {"stat": "OK", "tables": [
            {"title": "信用交易統計", "fields": ["項目", "買進", "賣出", "現金(券)償還", "前日餘額", "今日餘額"],
             "data": [["融資(交易單位)", "1", "2", "3", "4", "5"]]},
            {"title": "融資融券彙總", "fields": ["代號", "名稱", "買進", "賣出", "現金償還", "前日餘額", "今日餘額", "次一營業日限額",
                                                "買進", "賣出", "現券償還", "前日餘額", "今日餘額", "次一營業日限額", "資券互抵", "註記"],
             "data": [["2330", "台積電", "100", "50", "0", "9,000", "9,050", "x", "10", "5", "0", "500", "505", "x", "0", ""]]}]}
        df, method = wx.parse_margin(payload, "上市")
        r = df.iloc[0]
        assert method == "欄位名稱" and (r["code"], r["margin_bal"], r["short_bal"]) == ("2330", 9050, 505)

    def test_margin_tpex_old(self):
        payload = {"aaData": [["6488", "環球晶", "1,000", "10", "20", "0", "990", "0", "5.0", "x",
                               "100", "5", "2", "0", "103", "0", "1.0", "x", "0", ""]]}
        df, _ = wx.parse_margin(payload, "上櫃")
        assert (df.iloc[0]["margin_bal"], df.iloc[0]["short_bal"]) == (990, 103)

    def test_qfii(self):
        payload = {"stat": "OK", "fields": ["證券代號", "證券名稱", "國際證券編碼", "發行股數", "外資及陸資尚可投資股數",
                                            "全體外資及陸資持有股數", "外資及陸資尚可投資比率", "全體外資及陸資持股比率",
                                            "外資及陸資共用法令投資上限比率", "與前日異動原因", "最近一次上市公司申報外資持股異動日期"],
                   "data": [["2330", "台積電", "TW0002330008", "25,930,380,458", "x", "x", "26.45", "73.55", "100.00", "", ""]]}
        df, _ = wx.parse_qfii(payload, "上市")
        assert df.iloc[0]["foreign_pct"] == pytest.approx(73.55)
        assert df.iloc[0]["shares_issued"] == 25930380458

    def test_qfii_tpex_with_rank_column(self):
        payload = {"aaData": [["1", "6488", "環球晶", "435,000,000", "x", "x", "70.0", "30.00", "100"]]}
        df, method = wx.parse_qfii(payload, "上櫃")
        assert method == "位置(猜測)"
        assert df.iloc[0]["code"] == "6488" and df.iloc[0]["foreign_pct"] == pytest.approx(30.0)
        assert df.iloc[0]["shares_issued"] == 435_000_000

    def test_valuation_twse_and_blank_pe(self):
        payload = {"stat": "OK", "fields": ["證券代號", "證券名稱", "收盤價", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
                   "data": [["2330", "台積電", "593.00", "2.36", "112", "15.57", "4.21", "112/3"],
                            ["1101", "台泥", "32.00", "3.50", "112", "-", "1.10", "112/3"]]}
        df, _ = wx.parse_valuation(payload, "上市")
        d = df.set_index("code")
        assert d.loc["2330", "pe"] == pytest.approx(15.57) and d.loc["2330", "pb"] == pytest.approx(4.21)
        assert d.loc["2330", "dy"] == pytest.approx(2.36)
        assert np.isnan(d.loc["1101", "pe"])

    def test_valuation_tpex_old(self):
        payload = {"aaData": [["6488", "環球晶", "12.5", "15.0", "112", "3.10", "2.20"]]}
        df, _ = wx.parse_valuation(payload, "上櫃")
        r = df.iloc[0]
        assert (r["pe"], r["dy"], r["pb"]) == (12.5, 3.10, 2.20)

    def test_no_data(self):
        df, method = wx.parse_insti({"stat": "很抱歉，沒有符合條件的資料!"}, "上市")
        assert df.empty and method == "解析不到"


# ---------------------------------------------------------------------------
# MOPS 季財報 HTML
# ---------------------------------------------------------------------------
def _t163_table(header, rows, repeat_header=False):
    h = "<tr>" + "".join(f"<th>{x}</th>" for x in header) + "</tr>"
    body = ""
    for i, r in enumerate(rows):
        if repeat_header and i == 1:
            body += h
        body += "<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>"
    return f"<table class='hasBorder'>{h}{body}</table>"


T163SB04 = ("<html><body><h4>綜合損益表</h4>"
            + _t163_table(["公司代號", "公司名稱", "營業收入", "營業成本", "營業毛利（毛損）", "未實現銷貨（損）益",
                           "營業毛利（毛損）淨額", "營業費用", "營業利益（損失）", "本期淨利（淨損）",
                           "淨利（淨損）歸屬於母公司業主", "基本每股盈餘（元）"],
                          [["1101", "台泥", "10,000", "8,000", "2,000", "0", "1,900", "500", "1,400", "1,000", "900", "0.50"],
                           ["2330", "台積電", "500,000", "250,000", "250,000", "0", "250,000", "50,000", "200,000", "180,000", "180,000", "6.90"]],
                          repeat_header=True)
            + _t163_table(["公司代號", "公司名稱", "利息淨收益", "利息以外淨損益", "本期稅後淨利（淨損）", "基本每股盈餘（元）"],
                          [["2881", "富邦金", "30,000", "10,000", "20,000", "1.50"]])
            + "</body></html>")

T163SB05 = ("<html><body>"
            + _t163_table(["公司代號", "公司名稱", "流動資產", "資產總計", "負債總計", "股本", "歸屬於母公司業主之權益合計",
                           "權益總計", "負債及權益總計", "每股參考淨值"],
                          [["1101", "台泥", "1", "100,000", "40,000", "1", "55,000", "60,000", "100,000", "30.1"]])
            + "</body></html>")


class TestT163:
    def test_income(self):
        df, rep = wx.parse_t163(T163SB04, "t163sb04")
        d = df.set_index("code")
        assert rep["tables_with_data"] == 2
        assert d.loc["1101", "revenue"] == 10000
        assert d.loc["1101", "gross"] == 1900          # 優先用「淨額」
        assert d.loc["1101", "op_income"] == 1400
        assert d.loc["1101", "net_income"] == 1000     # 不是「歸屬於母公司」那欄
        assert d.loc["1101", "eps"] == pytest.approx(0.5)
        assert d.loc["2330", "eps"] == pytest.approx(6.9)   # 重複表頭列不影響
        assert np.isnan(d.loc["2881", "revenue"]) and d.loc["2881", "eps"] == pytest.approx(1.5)
        assert d.loc["2881", "net_income"] == 20000

    def test_balance(self):
        df, _ = wx.parse_t163(T163SB05, "t163sb05")
        assert df.set_index("code").loc["1101", "equity"] == 60000   # 權益總計(不是歸屬母公司、不是負債及權益總計)

    def test_empty(self):
        df, rep = wx.parse_t163("<html>查詢無資料</html>", "t163sb04")
        assert df.empty and rep["rows"] == 0


# ---------------------------------------------------------------------------
# 季報 point-in-time
# ---------------------------------------------------------------------------
class TestQuarterPIT:
    def test_deadlines(self):
        assert wx.quarter_available(2024, 1) == pd.Timestamp("2024-05-16")
        assert wx.quarter_available(2024, 2) == pd.Timestamp("2024-08-15")
        assert wx.quarter_available(2024, 3) == pd.Timestamp("2024-11-15")
        assert wx.quarter_available(2023, 4) == pd.Timestamp("2024-04-01")

    @pytest.mark.parametrize("date,expected", [
        ("2024-05-15", (2023, 4)), ("2024-05-16", (2024, 1)), ("2024-03-31", (2023, 3)), ("2024-04-01", (2023, 4)),
        ("2024-08-14", (2024, 1)), ("2024-08-15", (2024, 2)), ("2024-12-31", (2024, 3)),
    ])
    def test_latest_published(self, date, expected):
        assert wx.latest_published_quarter(date) == expected

    def test_quarter_helpers(self):
        assert wx.shift_quarter((2024, 1), -1) == (2023, 4)
        assert wx.shift_quarter((2023, 4), 5) == (2025, 1)
        assert wx.quarters_between((2023, 3), (2024, 2)) == [(2023, 3), (2023, 4), (2024, 1), (2024, 2)]

    def _raw(self, cumulative: bool, codes=("1101", "1102", "1103", "1104", "1105")):
        """單季營收100,110,120,130(2023)、200,220,240,260(2024)；毛利率25%/30%；EPS=淨利/100。
        5家公司一樣(格式判斷要至少5家同時有Q1、Q2)。"""
        rows = []
        for code, y, base, gmr in [(c, *t) for c in codes for t in [(2023, 100, 0.25), (2024, 200, 0.30)]]:
            singles = [base, base * 1.1, base * 1.2, base * 1.3]
            cum = np.cumsum(singles)
            for q in range(1, 5):
                rev = cum[q - 1] if (cumulative or q == 4) else singles[q - 1]
                rows.append({"code": code, "year": y, "q": q, "market": "上市", "revenue": rev, "gross": rev * gmr,
                             "op_income": rev * gmr / 2, "net_income": rev * 0.1, "eps": rev * 0.1 / 100,
                             "equity": 1000.0})
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("cumulative", [True, False])
    def test_single_quarter_derivation_and_q4(self, cumulative):
        raw = self._raw(cumulative)
        det = wx.detect_cumulative(raw)
        assert det[(2023, "上市")] is cumulative and det[(2024, "上市")] is cumulative
        qf = wx.build_quarterly_features(raw)["1101"]
        q4_2024 = qf.loc[pd.Timestamp("2024-12-31")]
        assert q4_2024["eps"] == pytest.approx(260 * 0.1 / 100)            # Q4單季 = 全年−Q1~Q3
        assert q4_2024["eps_yoy"] == pytest.approx(100.0)                  # 2.6/1.3−1
        assert q4_2024["gm"] == pytest.approx(30.0)
        assert q4_2024["gm_yoy"] == pytest.approx(5.0)
        assert q4_2024["om_yoy"] == pytest.approx(2.5)
        assert q4_2024["eps4q"] == pytest.approx((200 + 220 + 240 + 260) * 0.1 / 100)
        assert q4_2024["roe"] == pytest.approx((200 + 220 + 240 + 260) * 0.1 / 1000 * 100)
        q2_2023 = qf.loc[pd.Timestamp("2023-06-30")]
        assert q2_2023["eps"] == pytest.approx(110 * 0.1 / 100)
        assert np.isnan(q2_2023["eps_yoy"]) and np.isnan(q2_2023["eps4q"])

    def test_detect_undecided_year_follows_majority(self):
        raw = self._raw(False)
        raw = pd.concat([raw, raw[raw["year"] == 2024].assign(year=2022)])
        raw = raw[~((raw["year"] == 2022) & raw["q"].isin([1, 2]))]     # 2022只有Q3/Q4 → 跟其他年份一樣是單季
        assert wx.detect_cumulative(raw)[(2022, "上市")] is False

    def test_on_dates_point_in_time(self):
        qf = wx.build_quarterly_features(self._raw(True))["1101"]
        r = wx.quarterly_on_dates(qf, pd.to_datetime(["2025-03-31", "2025-04-01", "2024-02-01", "2023-01-05"]))
        assert r["qlabel"].tolist()[:3] == ["2024Q3", "2024Q4", "2023Q3"]
        assert r["eps"].iloc[0] == pytest.approx(240 * 0.1 / 100)   # 3/31當天Q4還看不到
        assert r["eps"].iloc[1] == pytest.approx(260 * 0.1 / 100)
        assert np.isnan(r["eps"].iloc[3])

    def test_future_quarters_do_not_change_past(self):
        raw = self._raw(True)
        extra = raw[raw["year"] == 2024].copy()
        extra["year"] = 2025
        a = wx.quarterly_on_dates(wx.build_quarterly_features(raw)["1101"], ["2025-04-02"])
        b = wx.quarterly_on_dates(wx.build_quarterly_features(pd.concat([raw, extra]))["1101"], ["2025-04-02"])
        pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# 進階技術面
# ---------------------------------------------------------------------------
def _ind(close, volume=1000.0, start="2022-01-03"):
    dates = pd.bdate_range(start, periods=len(close))
    raw = _ohlc_from_close(close, dates, 1)
    raw["Volume"] = volume
    return ws.compute_indicators(ws.prepare_price_frame(raw))


class TestTechExtra:
    def test_uptrend(self):
        c = np.arange(100.0, 400.0)
        ind = _ind(c)
        d = ind.index[-1]
        t = wx.tech_extra_at(ind, d)
        assert t["bull_align"] == 1.0
        assert t["ma60_slope"] == pytest.approx(c[-60:].mean() / c[-80:-20].mean() - 1)
        assert t["ma120_slope"] == pytest.approx(c[-120:].mean() / c[-140:-20].mean() - 1)
        assert t["bias60"] == pytest.approx(c[-1] / c[-60:].mean() - 1)
        h, l = ind["high"].to_numpy(), ind["low"].to_numpy()
        assert t["range60"] == pytest.approx((h[-60:].max() - l[-60:].min()) / c[-1])
        assert t["rsi14"] == pytest.approx(100.0)
        assert 70 < t["k9"] < 100          # 收盤一路貼近9日高點(高點=收盤×1.01，所以不會到100)
        assert t["dist_52w_low"] == pytest.approx(c[-1] / c[-252] - 1)
        assert t["days_since_high"] == 0 and t["max_dd_1y"] == 0
        assert t["atr_pct"] == pytest.approx(ind["atr14"].iloc[-1] / c[-1] * 100)
        assert t["gap_ups_60"] == 0

    def test_downtrend_rsi_kd(self):
        t = wx.tech_extra_at(_ind(np.arange(400.0, 100.0, -1)), _ind(np.arange(400.0, 100.0, -1)).index[-1])
        assert t["rsi14"] == pytest.approx(0.0) and t["k9"] < 30 and t["bull_align"] == 0.0

    def test_drawdown_and_days_since_high(self):
        c = np.r_[np.linspace(100, 200, 150), np.linspace(199, 120, 60), np.linspace(121, 150, 60)]
        ind = _ind(c)
        t = wx.tech_extra_at(ind, ind.index[-1])
        assert t["max_dd_1y"] == pytest.approx(120 / 200 - 1)
        assert t["days_since_high"] == 120          # 最高點在倒數第121根

    def test_bbw_squeeze_percentile(self):
        rng = np.random.default_rng(0)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
        c[-25:] = c[-26]                             # 最後一段完全不動 → 帶寬=0 → 百分位最低
        ind = _ind(c)
        t = wx.tech_extra_at(ind, ind.index[-1])
        assert t["bbw_pct"] <= 5

    def test_beta(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.01, 300)
        dates = pd.bdate_range("2022-01-03", periods=300)
        mkt = pd.Series(10000 * np.cumprod(1 + r), index=dates)
        stock = 50 * np.cumprod(1 + 2 * r)
        ind = _ind(stock)
        t = wx.tech_extra_at(ind, ind.index[-1], mkt)
        assert t["beta"] == pytest.approx(2.0, rel=1e-6)

    def test_volume_burst_and_gaps(self):
        c = np.full(100, 50.0)
        dates = pd.bdate_range("2022-01-03", periods=100)
        raw = _ohlc_from_close(c, dates, 1000)
        raw.loc[raw.index[-5:], "Volume"] = 3000
        for k in (-30, -10):                          # 兩次向上跳空：當天最低 > 前一天最高
            raw.iloc[k:, raw.columns.get_indexer(["Open", "High", "Low", "Close", "Adj Close"])] *= 1.1
        ind = ws.compute_indicators(ws.prepare_price_frame(raw))
        t = wx.tech_extra_at(ind, ind.index[-1])
        assert t["vol_burst"] == pytest.approx(3000 / ((55 * 1000 + 5 * 3000) / 60))
        assert t["gap_ups_60"] == 2

    def test_no_lookahead(self):
        rng = np.random.default_rng(2)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
        dates = pd.bdate_range("2022-01-03", periods=400)
        mkt = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 400)), index=dates)
        full = _ind(c)
        part = _ind(c[:320])
        d = full.index[319]
        a, b = wx.tech_extra_at(full, d, mkt), wx.tech_extra_at(part, d, mkt[:320])
        assert a == pytest.approx(b, nan_ok=True)


# ---------------------------------------------------------------------------
# 籌碼特徵
# ---------------------------------------------------------------------------
class TestChipFeatures:
    def _setup(self, n_loaded=20):
        dates = pd.bdate_range("2024-01-01", periods=90)
        ind = _ind(np.full(90, 50.0), volume=1000.0, start="2024-01-01")
        cal = pd.DatetimeIndex(dates)
        d0 = cal[-1]
        nd = wx.needed_dates(cal, d0)
        insti = {}
        for k, d in enumerate(nd["insti"][:n_loaded]):
            rows = [{"code": "9999", "foreign_net": 1, "trust_net": 1, "total_net": 2}]
            if k not in (16,):   # 第17天這檔不在表裡 → 當天視為0
                rows.append({"code": "1101", "foreign_net": 100.0, "trust_net": 50.0 if k >= 17 else -10.0,
                             "total_net": 150.0})
            insti[d] = pd.DataFrame(rows)
        chip_daily = {
            "insti": insti,
            "margin": {nd["margin"][0]: pd.DataFrame([{"code": "1101", "margin_bal": 1000.0, "short_bal": 50.0}]),
                       nd["margin"][1]: pd.DataFrame([{"code": "1101", "margin_bal": 1200.0, "short_bal": 120.0}])},
            "qfii": {nd["qfii"][0]: pd.DataFrame([{"code": "1101", "shares_issued": 1e6, "foreign_pct": 20.0}]),
                     nd["qfii"][1]: pd.DataFrame([{"code": "1101", "shares_issued": 1e6, "foreign_pct": 23.5}])},
            "valuation": {nd["valuation"][0]: pd.DataFrame([{"code": "1101", "pe": np.nan, "pb": 1.5, "dy": 3.0}])},
        }
        loaded = {s: {(d, "上市") for d in v} for s, v in chip_daily.items()}
        return ind, wx.prepare_chip(chip_daily), loaded, nd

    def test_features(self):
        ind, chip, loaded, nd = self._setup()
        f = wx.chip_features("1101", "上市", ind, chip, loaded, nd)
        assert len(nd["insti"]) == 20 and nd["insti"][-1] == ind.index[-1]
        assert f["foreign_ratio20"] == pytest.approx(19 * 100 / 20000 * 100)
        assert f["trust_ratio20"] == pytest.approx((16 * -10 + 3 * 50) / 20000 * 100)
        assert f["trust_pct_shares20"] == pytest.approx((16 * -10 + 3 * 50) / 1e6 * 100)
        assert f["insti_ratio20"] == pytest.approx(19 * 150 / 20000 * 100)
        assert f["trust_streak"] == 3           # 最後3天買超，第17天(不在表)=0中斷
        assert f["foreign_streak"] == 3
        assert f["margin_chg20"] == pytest.approx(20.0)
        assert f["short_margin_ratio"] == pytest.approx(10.0)
        assert f["foreign_pct"] == pytest.approx(23.5) and f["foreign_pct_chg60"] == pytest.approx(3.5)
        assert np.isnan(f["pe"]) and f["pe_blank"] == 1.0 and f["pb"] == 1.5

    def test_too_few_days_and_wrong_market(self):
        ind, chip, loaded, nd = self._setup(n_loaded=14)
        f = wx.chip_features("1101", "上市", ind, chip, loaded, nd)
        assert np.isnan(f["foreign_ratio20"]) and np.isnan(f["trust_streak"])
        ind, chip, loaded, nd = self._setup()
        f = wx.chip_features("1101", "上櫃", ind, chip, loaded, nd)   # 上櫃那天沒抓到 → 不能當0
        assert np.isnan(f["foreign_ratio20"])

    def test_empty_chip(self):
        ind = _ind(np.full(90, 50.0))
        nd = wx.needed_dates(ind.index, ind.index[-1])
        f = wx.chip_features("1101", "上市", ind, wx.prepare_chip({}), {}, nd)
        assert all(np.isnan(v) for v in f.values())


# ---------------------------------------------------------------------------
# 雙條件 lift、洗牌基準
# ---------------------------------------------------------------------------
class TestPairsAndShuffle:
    def _data(self):
        n = 200
        w = np.zeros(n)
        w[:20] = 1
        C = np.zeros((n, 4), bool)
        C[:15, 0] = True; C[40:85, 0] = True      # A：60檔、15贏家
        C[5:20, 1] = True; C[60:110, 1] = True    # B：65檔、15贏家(和A不同特徵)
        C[:15, 2] = True; C[40:80, 2] = True      # A'：和A同一個特徵 → 不能和A一起被選
        C[150:190, 3] = True                      # D：0贏家
        return C, w, ["fa", "fb", "fa", "fd"]

    def test_select_distinct_features(self):
        C, w, feats = self._data()
        ch = wx.select_top_singles(C, w, feats, n_min=30, cov_min=0.1)
        assert ch == [2, 1]                       # A'(55檔15贏家，lift較高)勝過A；同特徵只取一個；D覆蓋率0不選

    def test_pair_math(self):
        C, w, feats = self._data()
        pl = wx.pair_lifts(C, w, [0, 1], n_min=10)
        i, j, n, wc, rate, lift, cov = pl[0]
        assert (i, j, n, wc) == (0, 1, 10 + 25, 10)       # 5~14 + 60~84
        assert rate == pytest.approx(10 / 35)
        assert lift == pytest.approx((10 / 35) / 0.1)
        assert cov == pytest.approx(0.5)
        assert wx.pair_lifts(C, w, [0, 1], n_min=40) == []

    def test_shuffle_deterministic(self):
        C, w, feats = self._data()
        a = wx.shuffle_baseline(C, w, feats, n_min=30, n_perm=50, seed=42)
        b = wx.shuffle_baseline(C, w, feats, n_min=30, n_perm=50, seed=42)
        c = wx.shuffle_baseline(C, w, feats, n_min=30, n_perm=50, seed=7)
        assert a["single_p95"] == b["single_p95"] and np.array_equal(a["singles"], b["singles"])
        assert not np.array_equal(a["singles"], c["singles"])
        assert a["single_p95"] > 1.0             # 隨機下最大lift仍然>1(多重比較)
        real = wx._lifts(C, w, C.sum(axis=0))[0]
        assert real[2] > a["single_p95"]          # 這組手寫資料的A'真的比運氣好


# ---------------------------------------------------------------------------
# 下載流程(假的HTTP)
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.content = text.encode("utf-8")


class TestLoaders:
    def test_daily_sources_tpex_fallback_cache_and_report(self, monkeypatch, tmp_path):
        calls = []

        def fake(method, url, data=None):
            calls.append(url)
            if "twse" in url and "T86" in url:
                return _Resp(200, json.dumps(T86, ensure_ascii=False))
            if "3itrade_hedge_result.php" in url:
                return _Resp(200, "<html>error</html>")          # 舊站壞了 → 試新站
            if "insti/dailyTrade" in url:
                return _Resp(200, json.dumps(TPEX_INSTI_NEW, ensure_ascii=False))
            return _Resp(404)
        monkeypatch.setattr(wx, "_http", fake)
        d = pd.Timestamp("2024-01-02")
        dates = {"insti": [d]}
        chip, loaded, rep = wx.load_daily_sources(dates, str(tmp_path / "c"), False, str(tmp_path / "dbg"),
                                                  today="2024-06-01", sleep=lambda s: None)
        assert set(chip["insti"][d]["code"]) == {"2330", "1101", "6488"}
        assert loaded["insti"] == {(d, "上市"), (d, "上櫃")}
        assert calls[0] == "https://www.twse.com.tw/rwd/zh/fund/T86?date=20240102&selectType=ALLBUT0999&response=json"
        assert "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=json&se=EW&t=D&d=113/01/02" in calls
        assert any("date=2024/01/02" in u for u in calls)
        assert os.path.exists(tmp_path / "dbg" / "extra_sample_insti_sii.txt")
        assert [r["狀態"] for r in rep] == ["ok", "ok"]
        n = len(calls)
        chip2, _, rep2 = wx.load_daily_sources(dates, str(tmp_path / "c"), False, str(tmp_path / "dbg"),
                                               today="2024-06-01", sleep=lambda s: None)
        assert len(calls) == n and [r["狀態"] for r in rep2] == ["快取", "快取"]
        assert wx.source_status(rep) == [("三大法人買賣超", 2, 2)]

    def test_daily_sources_failure_not_cached(self, monkeypatch, tmp_path):
        monkeypatch.setattr(wx, "_http", lambda m, u, data=None: _Resp(500))
        chip, loaded, rep = wx.load_daily_sources({"margin": [pd.Timestamp("2024-01-02")]}, str(tmp_path / "c"), False,
                                                  str(tmp_path / "d"), sleep=lambda s: None)
        assert chip["margin"] == {} and loaded["margin"] == set()
        assert wx.source_status(rep) == [("融資融券", 0, 2)]
        assert os.listdir(tmp_path / "c" / "extra") == []

    def test_quarterly_post_and_merge(self, monkeypatch, tmp_path):
        forms = []

        def fake(method, url, data=None):
            assert method == "POST"
            forms.append((url, dict(data)))
            if "mopsov" in url:
                return _Resp(500)                                 # 第一個網域壞 → 換 mops
            return _Resp(200, T163SB04 if "t163sb04" in url else T163SB05)
        monkeypatch.setattr(wx, "_http", fake)
        raw, rep = wx.load_quarterly([(2024, 1)], str(tmp_path / "c"), False, str(tmp_path / "d"),
                                     today="2024-12-31", sleep=lambda s: None)
        assert forms[0][0] == "https://mopsov.twse.com.tw/mops/web/ajax_t163sb04"
        f = forms[0][1]
        assert (f["TYPEK"], f["year"], f["season"], f["isQuery"], f["step"]) == ("sii", "113", "01", "Y", "1")
        assert any(u == "https://mops.twse.com.tw/mops/web/ajax_t163sb05" for u, _ in forms)
        r = raw[(raw["code"] == "1101") & (raw["market"] == "上市")].iloc[0]
        assert r["equity"] == 60000 and r["revenue"] == 10000 and (r["year"], r["q"]) == (2024, 1)
        assert all(x["狀態"] == "ok" for x in rep)

    def test_load_all_extra_fail_soft(self, monkeypatch, tmp_path):
        monkeypatch.setattr(wx, "REQUEST_DELAY", 0)
        monkeypatch.setattr(wx.time, "sleep", lambda s: None)
        monkeypatch.setattr(wx, "_http", lambda m, u, data=None: _Resp(503))
        cal = pd.bdate_range("2023-01-02", "2024-12-31")
        res = wx.load_all_extra(cal, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"), str(tmp_path / "c"), False,
                                str(tmp_path / "d"))
        assert res["quarterly_raw"].empty and all(not v for v in res["chip_daily"].values())
        assert all(ok == 0 for _, ok, _ in wx.source_status(res["report"]))
        assert os.path.exists(tmp_path / "d" / "extra_report.csv")

    def test_request_budget(self):
        cal = pd.bdate_range("2022-06-01", "2026-10-09")
        d0, d1 = cal[cal.searchsorted(pd.Timestamp("2024-10-09"))], cal[-1]
        dates = wx.needed_dates(cal, d0)
        assert [len(dates[k]) for k in ("insti", "margin", "qfii", "valuation")] == [20, 2, 2, 1]
        assert dates["margin"][0] == cal[cal.get_loc(d0) - 20] and dates["qfii"][0] == cal[cal.get_loc(d0) - 60]
        qs = wx.quarters_between(wx.shift_quarter(wx.latest_published_quarter(d0), -wx.QUARTERS_BACK),
                                 wx.latest_published_quarter(d1))
        assert wx.estimate_requests(dates, len(qs)) < 250
