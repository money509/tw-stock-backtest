"""taifex_futures_loader.py的測試。全部離線：requests.get/post一律換成假的(真的網路呼叫會直接讓測試失敗)。

假資料(fixture)是照「目前對期交所格式的理解」做的：stockLists的HTML表格(欄位順序打亂、多餘欄位、
小型/標準代碼)、futDataDown的Big5(cp950)CSV(多個到期月份、價差列、一般/盤後兩個時段、'-'值、行尾逗號)。
真實格式要等第一次在GitHub Actions跑完、看signals/debug/才能確認。"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import requests

import taifex_futures_loader as tfl
from taifex_universe import STOCK_FUTURES_UNIVERSE, get_contract_multiplier


# ----------------------------------------------------------------------------
# fixture產生器(test_daily_squeeze_signals.py也會用)
# ----------------------------------------------------------------------------
def fake_codes(code: str):
    """每檔股票一組假的期貨代碼：標準=S+兩個字母、小型=M+同樣兩個字母(跟真實代碼一樣是2~4個大寫字母)。"""
    i = list(STOCK_FUTURES_UNIVERSE).index(code)
    tail = chr(65 + i // 26 % 26) + chr(65 + i % 26)
    return "S" + tail, "M" + tail


def make_stock_lists_html(codes, drop_mini=(), extra_rows=()):
    """仿stockLists頁面：前面一個不相干的表格，再來是真正的表格(欄位順序不是我們預期的順序、有多餘欄位)。"""
    head = ("<tr><th>序號</th><th>證券名稱</th><th>上市/上櫃</th><th>股票期貨英文代碼</th>"
            "<th>股票選擇權英文代碼</th><th>證券代號</th><th>標準型證券股數</th><th>小型股票期貨英文代碼</th></tr>")
    body = []
    for n, code in enumerate(codes, start=1):
        std, mini = fake_codes(code)
        has_mini = STOCK_FUTURES_UNIVERSE[code]["has_mini"] and code not in drop_mini
        body.append(f"<tr><td>{n}</td><td>名稱{code}</td><td>上市</td><td> {std} </td><td>OPT</td>"
                    f"<td>{code}</td><td>2,000</td><td>{mini if has_mini else '-'}</td></tr>")
    body += list(extra_rows)
    return ("<html><head><meta charset='utf-8'><title>股票期貨標的</title></head><body>"
            "<table><tr><td>選單</td><td>首頁</td></tr></table>"
            "<table class='table_c'>" + head + "".join(body) + "</table></body></html>")


CSV_HEADER = ("交易日期,契約,到期月份(週別),開盤價,最高價,最低價,收盤價,漲跌價,漲跌%,成交量,結算價,"
              "未沖銷契約數,最後最佳買價,最後最佳賣價,歷史最高價,歷史最低價,是否因訊息面暫停交易,交易時段,"
              "價差對單式委託成交量")


def csv_row(date, contract, month, volume, oi, bid, ask, session="一般", close="100", spread_vol="0"):
    return (f"{date},{contract} ,{month} ,{close},{close},{close},{close},0,0%,{volume},{close},{oi},{bid},{ask},"
            f"{close},{close},,{session},{spread_vol},")  # 行尾多一個逗號(期交所CSV常見)


def make_fut_csv_bytes(contracts, dates, near="202610", far="202611", extra_rows=(), encoding="cp950"):
    """contracts: {期貨代碼: {"near": 一般時段近月每日量, "far": 一般時段次月每日量, "night": 盤後近月每日量,
    "oi": 近月未平倉, "bid":, "ask":}}。每天每個契約：近月(一般)、次月(一般，買賣價'-')、近月(盤後，未平倉'-')、
    一列價差委託(到期月份含'/'，量很大，必須被丟掉)。"""
    lines = [CSV_HEADER]
    for d in dates:
        ds = pd.Timestamp(d).strftime("%Y/%m/%d")
        for c, p in contracts.items():
            lines.append(csv_row(ds, c, near, p["near"], p["oi"], p.get("bid", "99.5"), p.get("ask", "100"),
                                 spread_vol="7"))
            lines.append(csv_row(ds, c, far, p["far"], p.get("far_oi", 10), "-", "-"))
            lines.append(csv_row(ds, c, near, p["night"], "-", "-", "-", session="盤後"))
            lines.append(csv_row(ds, c, f"{near}/{far}", 99999, "-", "-", "-"))
    lines += list(extra_rows)
    return ("\r\n".join(lines) + "\r\n").encode(encoding)


class FakeResp:
    def __init__(self, content=b"", status=200):
        self.content = content
        self.status_code = status


def install_fake_http(monkeypatch, html=None, csv=None, html_status=200, csv_status=200, calls=None,
                      html_exc=None, csv_exc=None):
    """把requests.get/post換成假的：GET stockLists → html；POST futDataDown → csv。其他網址直接報錯。"""
    calls = calls if calls is not None else []

    def fake_get(url, **kw):
        calls.append(("GET", url, kw.get("data"), kw.get("headers"), kw.get("timeout")))
        assert url == tfl.STOCK_LISTS_URL, url
        if html_exc:
            raise html_exc
        return FakeResp(html.encode("utf-8") if isinstance(html, str) else (html or b""), html_status)

    def fake_post(url, data=None, **kw):
        calls.append(("POST", url, data, kw.get("headers"), kw.get("timeout")))
        assert url == tfl.FUT_DATA_DOWN_URL, url
        if csv_exc:
            raise csv_exc
        return FakeResp(csv or b"", csv_status)

    monkeypatch.setattr(tfl.requests, "get", fake_get)
    monkeypatch.setattr(tfl.requests, "post", fake_post)
    monkeypatch.setattr(tfl, "RETRY_BACKOFF_SECONDS", 0)
    return calls


def fake_liquidity(codes, volume=1000.0, oi=5000.0, spread=1.0, overrides=None, latest_date=None):
    """直接造一份load_futures_liquidity()的成功結果(給掃描器的測試用，不經過HTTP/解析)。
    overrides: {股票代號: {"avg_volume":..., "near_oi":...}}；值是None代表行情裡沒有這個契約。"""
    overrides = overrides or {}
    traded, stats = {}, {}
    for c in codes:
        std, mini = fake_codes(c)
        fut = mini if get_contract_multiplier(c, 0) == 100 else std
        traded[c] = fut
        ov = overrides.get(c, {})
        if ov is None:
            continue
        stats[fut] = {"avg_volume": ov.get("avg_volume", volume), "days": 5, "near_month": "202610",
                      "near_oi": ov.get("near_oi", oi), "bid": 99.5, "ask": 100.0, "spread_ticks": spread,
                      "latest_date": latest_date}
    return {"ok": True, "error": None, "mapping": {}, "traded": traded, "stats": stats,
            "latest_date": latest_date, "trading_dates": [], "report": {"ok": True}}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """保險：任何沒被換掉的真實HTTP呼叫都直接失敗。"""
    def boom(*a, **k):
        raise AssertionError("測試不可以連網路")
    monkeypatch.setattr(requests.Session, "request", boom)


DATES7 = pd.bdate_range("2026-09-28", periods=7)  # 9/28(一) ~ 10/6(二)
AS_OF = DATES7[-1]
UNIV = ["1101", "1102", "2330", "1477", "2303"]  # 2330/1477有小型契約，其他沒有


def _contracts_for(codes, **per):
    out = {}
    for c in codes:
        std, mini = fake_codes(c)
        fut = mini if STOCK_FUTURES_UNIVERSE[c]["has_mini"] else std
        out[fut] = {"near": 100, "far": 20, "night": 10, "oi": 800, **per.get(c, {})}
        other = std if fut == mini else None
        if other:  # 有小型的股票，標準契約也在行情裡(量不同)，確認不會拿錯
            out[other] = {"near": 5, "far": 1, "night": 0, "oi": 50}
    return out


# ----------------------------------------------------------------------------
class TestColumnDetection:
    def test_csv_header_columns(self):
        header = CSV_HEADER.split(",")
        assert tfl.find_column(header, ("契約", "商品代號"), exclude_any=("未沖銷", "未平倉", "數", "名稱")) == "契約"
        assert tfl.find_column(header, ("成交量",), exclude_any=("價差",)) == "成交量"
        assert tfl.find_column(header, ("未沖銷", "未平倉")) == "未沖銷契約數"
        assert tfl.find_column(header, ("到期月份",)) == "到期月份(週別)"
        assert tfl.find_column(header, ("最佳買價",)) == "最後最佳買價"
        assert tfl.find_column(header, ("不存在",)) is None

    def test_stock_list_header_detection(self):
        row = ["序號", "證券名稱", "股票期貨英文代碼", "股票選擇權英文代碼", "證券代號", "小型股票期貨英文代碼", "股票期貨標的"]
        cols = tfl._detect_stock_list_header(row)
        assert cols == {"code": "證券代號", "std": "股票期貨英文代碼", "mini": "小型股票期貨英文代碼", "name": "證券名稱"}

    def test_mini_flag_column_is_not_a_code_column(self):
        cols = tfl._detect_stock_list_header(["證券代號", "股票期貨英文代碼", "小型股票期貨標的"])
        assert cols["mini"] is None and cols["std"] == "股票期貨英文代碼"


class TestStockListMapping:
    def test_parse_and_mini_vs_standard_matches_multiplier(self):
        mapping, info = tfl.parse_stock_list_html(make_stock_lists_html(UNIV))
        assert info["tables_found"] == 2 and info["table_index"] == 1
        assert mapping["2330"]["std"] == fake_codes("2330")[0] and mapping["2330"]["mini"] == fake_codes("2330")[1]
        assert mapping["1101"]["mini"] is None
        traded = tfl.map_universe(mapping, UNIV)
        for c in UNIV:
            std, mini = fake_codes(c)
            assert traded[c] == (mini if get_contract_multiplier(c, 100.0) == 100 else std)
        assert traded["2330"] == fake_codes("2330")[1] and traded["1101"] == fake_codes("1101")[0]

    def test_missing_mini_code_means_unmapped_not_standard(self):
        mapping, _ = tfl.parse_stock_list_html(make_stock_lists_html(UNIV, drop_mini=("2330",)))
        assert tfl.map_universe(mapping, ["2330"])["2330"] is None  # 不會退回用標準契約的量

    def test_invalid_rows_skipped_and_conflicts_fail_closed(self):
        extra = ["<tr><td>x</td><td>合計</td><td></td><td>-</td><td></td><td>總計</td><td></td><td></td></tr>",
                 "<tr><td>9</td><td>重複</td><td>上市</td><td>ZZZ</td><td></td><td>1101</td><td></td><td>-</td></tr>"]
        mapping, info = tfl.parse_stock_list_html(make_stock_lists_html(UNIV, extra_rows=extra))
        assert info["skipped_rows"] == 1
        assert "1101" in info["conflicting_codes"] and mapping["1101"]["std"] is None

    def test_no_table_raises(self):
        with pytest.raises(tfl.LiquidityDataError):
            tfl.parse_stock_list_html("<html><body>系統維護中</body></html>")


class TestFutCsv:
    def test_parse_cp950_spread_rows_sessions_and_dashes(self):
        content = make_fut_csv_bytes(_contracts_for(["1101"]), DATES7[:2])
        df, info = tfl.parse_fut_csv(content)
        assert info["encoding"] == "cp950"
        assert info["spread_rows_dropped"] == 2
        assert info["detected_columns"]["volume"] == "成交量" and info["detected_columns"]["contract"] == "契約"
        assert not df["month"].str.contains("/").any()
        assert set(df["session"]) == {"一般", "盤後"}
        night = df[df["session"] == "盤後"]
        assert night["oi"].isna().all() and night["bid"].isna().all()  # '-' → NaN
        assert df["contract"].str.strip().eq(df["contract"]).all()

    def test_utf8_bom_also_works(self):
        content = make_fut_csv_bytes(_contracts_for(["1101"]), DATES7[:1], encoding="utf-8-sig")
        df, info = tfl.parse_fut_csv(content)
        assert info["encoding"] == "utf-8-sig" and len(df) == 3

    def test_html_instead_of_csv_raises(self):
        with pytest.raises(tfl.LiquidityDataError):
            tfl.parse_fut_csv("<html><body>查無資料</body></html>".encode("utf-8"))

    def test_missing_required_column_raises(self):
        with pytest.raises(tfl.LiquidityDataError):
            tfl.parse_fut_csv("交易日期,契約,成交量\n2026/10/06,ABF,5\n".encode("cp950"))

    def test_five_day_average_and_near_month(self):
        codes = ["1101", "2330"]
        contracts = _contracts_for(codes, **{"1101": {"near": 100, "far": 20, "night": 10, "oi": 800,
                                                      "bid": "48.5", "ask": "48.65"}})
        # 第一天(只在第1~2天)量特別大：最後5天以外的日子不能算進去
        extra = [csv_row(DATES7[0].strftime("%Y/%m/%d"), fake_codes("1101")[0], "202612", 100000, 5, "-", "-")]
        # as_of之後的資料也不能用
        after = pd.Timestamp(AS_OF) + pd.offsets.BDay(1)
        extra.append(csv_row(after.strftime("%Y/%m/%d"), fake_codes("1101")[0], "202610", 77777, 1, "1", "2"))
        df, _ = tfl.parse_fut_csv(make_fut_csv_bytes(contracts, DATES7, extra_rows=extra))
        wanted = [fake_codes("1101")[0], fake_codes("2330")[1]]
        stats, info = tfl.compute_contract_stats(df, AS_OF, wanted)
        assert info["trading_dates_used"] == [d.date().isoformat() for d in DATES7[-5:]]
        s = stats[fake_codes("1101")[0]]
        assert s["avg_volume"] == pytest.approx(100 + 20 + 10)  # 全部月份、兩個時段，不含價差列
        assert s["near_month"] == "202610" and s["near_oi"] == 800
        assert s["spread_ticks"] == 3  # 48.5 → 48.65，tick 0.05
        assert s["latest_date"] == AS_OF.date().isoformat()
        m = stats[fake_codes("2330")[1]]
        assert m["avg_volume"] == pytest.approx(130) and m["near_oi"] == 800

    def test_missing_days_count_as_zero(self):
        c = fake_codes("1101")[0]
        content = make_fut_csv_bytes({c: {"near": 50, "far": 0, "night": 0, "oi": 400}}, DATES7[-2:])
        filler = make_fut_csv_bytes({"ZZ": {"near": 1, "far": 0, "night": 0, "oi": 1}}, DATES7[-5:-2])
        df1, _ = tfl.parse_fut_csv(content)
        df2, _ = tfl.parse_fut_csv(filler)
        stats, _ = tfl.compute_contract_stats(pd.concat([df1, df2]), AS_OF, [c])
        assert stats[c]["avg_volume"] == pytest.approx(50 * 2 / 5)

    def test_too_few_days_raises(self):
        df, _ = tfl.parse_fut_csv(make_fut_csv_bytes(_contracts_for(["1101"]), DATES7[-2:]))
        with pytest.raises(tfl.LiquidityDataError):
            tfl.compute_contract_stats(df, AS_OF, [fake_codes("1101")[0]])

    @pytest.mark.parametrize("bid,ask,ticks", [(99.9, 100.0, 1), (99.5, 100.0, 5), (49.95, 50.1, 2), (100, 100, 0), (500, 501, 1),
                                               (np.nan, 100, np.nan), (101, 100, np.nan)])
    def test_spread_ticks(self, bid, ask, ticks):
        got = tfl.spread_in_ticks(bid, ask)
        assert (np.isnan(got) and np.isnan(ticks)) or got == ticks


# ----------------------------------------------------------------------------
class TestThresholds:
    def _liq(self):
        return fake_liquidity(["1101", "1102", "2330", "1477"],
                              overrides={"1102": {"avg_volume": 99.9}, "2330": {"near_oi": 299}, "1477": None})

    def test_pass_and_fail_reasons(self):
        liq = self._liq()
        ok = tfl.evaluate_code(liq, "1101", 100, 300)
        assert ok["passed"] and ok["reason"] is None and ok["fut_code"] == fake_codes("1101")[0]
        v = tfl.evaluate_code(liq, "1102", 100, 300)
        assert not v["passed"] and "5日均量" in v["reason"]
        o = tfl.evaluate_code(liq, "2330", 100, 300)
        assert not o["passed"] and "近月未平倉" in o["reason"]
        m = tfl.evaluate_code(liq, "1477", 100, 300)
        assert not m["passed"] and m["avg_volume"] == 0
        u = tfl.evaluate_code(liq, "2303", 100, 300)
        assert not u["passed"] and "對照表找不到標準契約代碼" in u["reason"]

    def test_threshold_is_inclusive(self):
        liq = fake_liquidity(["1101"], volume=100, oi=300)
        assert tfl.evaluate_code(liq, "1101", 100, 300)["passed"]
        assert not tfl.evaluate_code(liq, "1101", 100.5, 300)["passed"]

    def test_failed_data_never_passes(self):
        assert not tfl.evaluate_code(tfl.failed_liquidity("x"), "1101", 0, 0)["passed"]
        assert not tfl.evaluate_code(None, "1101", 0, 0)["passed"]


# ----------------------------------------------------------------------------
class TestLoadEndToEnd:
    def test_ok_requests_and_diagnostics(self, monkeypatch, tmp_path):
        calls = install_fake_http(monkeypatch, html=make_stock_lists_html(UNIV),
                                  csv=make_fut_csv_bytes(_contracts_for(UNIV), DATES7))
        dbg = tmp_path / "debug"
        liq = tfl.load_futures_liquidity(AS_OF, UNIV, debug_dir=str(dbg))
        assert liq["ok"], liq["error"]
        assert liq["traded"]["2330"] == fake_codes("2330")[1]
        assert liq["stats"][fake_codes("2330")[1]]["avg_volume"] == pytest.approx(130)
        assert liq["latest_date"] == AS_OF.date().isoformat()
        # 送出的請求
        get = [c for c in calls if c[0] == "GET"]
        post = [c for c in calls if c[0] == "POST"]
        assert len(get) == 1 and len(post) == 1
        assert post[0][2] == {"down_type": "1", "commodity_id": "all", "commodity_id2": "",
                              "queryStartDate": "2026/09/22", "queryEndDate": "2026/10/06"}
        assert "Mozilla" in post[0][3]["User-Agent"] and post[0][4] == tfl.HTTP_TIMEOUT_SECONDS
        # 診斷檔
        assert set(os.listdir(dbg)) == set(tfl.DEBUG_FILES)
        rep = json.loads((dbg / "parse_report.json").read_text(encoding="utf-8"))
        assert rep["ok"] is True and rep["coverage"]["mapped"] == len(UNIV)
        assert rep["fut_data_down"]["detected_columns"]["oi"] == "未沖銷契約數"
        assert rep["fut_data_down"]["spread_rows_dropped"] > 0
        assert "交易日期" in (dbg / "futdatadown_head.csv").read_text(encoding="utf-8").splitlines()[0]
        assert (dbg / "stocklists_head.html").read_text(encoding="utf-8").startswith("<table")

    def test_debug_head_is_limited(self, monkeypatch, tmp_path):
        dates = pd.bdate_range("2026-09-01", periods=25)
        csv = make_fut_csv_bytes(_contracts_for(UNIV), dates)
        install_fake_http(monkeypatch, html=make_stock_lists_html(UNIV), csv=csv)
        tfl.load_futures_liquidity(dates[-1], UNIV, debug_dir=str(tmp_path))
        assert len((tmp_path / "futdatadown_head.csv").read_text(encoding="utf-8").splitlines()) == tfl.DEBUG_HEAD_LINES

    def test_cache_used_when_not_refresh(self, monkeypatch, tmp_path):
        install_fake_http(monkeypatch, html=make_stock_lists_html(UNIV),
                          csv=make_fut_csv_bytes(_contracts_for(UNIV), DATES7))
        assert tfl.load_futures_liquidity(AS_OF, UNIV, cache_dir=str(tmp_path), verbose=False)["ok"]
        calls = install_fake_http(monkeypatch, html_status=500, csv_status=500)
        assert tfl.load_futures_liquidity(AS_OF, UNIV, cache_dir=str(tmp_path), refresh=False, verbose=False)["ok"]
        assert calls == []


class TestFailClosed:
    def _assert_failed(self, liq, dbg, needle=None):
        assert liq["ok"] is False and liq["error"]
        assert liq["stats"] == {}
        if needle:
            assert needle in liq["error"]
        rep = json.loads((dbg / "parse_report.json").read_text(encoding="utf-8"))
        assert rep["ok"] is False and rep["error"] == liq["error"]
        for c in UNIV:
            assert not tfl.evaluate_code(liq, c, 0, 0)["passed"]

    def test_http_error_with_retries(self, monkeypatch, tmp_path):
        calls = install_fake_http(monkeypatch, html=make_stock_lists_html(UNIV), csv_status=500)
        liq = tfl.load_futures_liquidity(AS_OF, UNIV, debug_dir=str(tmp_path))
        self._assert_failed(liq, tmp_path, "HTTP 500")
        assert len([c for c in calls if c[0] == "POST"]) == 1 + tfl.MAX_RETRIES

    def test_connection_error(self, monkeypatch, tmp_path):
        install_fake_http(monkeypatch, html_exc=requests.ConnectionError("no route"))
        liq = tfl.load_futures_liquidity(AS_OF, UNIV, debug_dir=str(tmp_path))
        self._assert_failed(liq, tmp_path, "stockLists抓取失敗")
        assert not (tmp_path / "stocklists_head.html").exists()

    def test_garbage_csv(self, monkeypatch, tmp_path):
        install_fake_http(monkeypatch, html=make_stock_lists_html(UNIV),
                          csv="<html><body>系統忙碌中</body></html>".encode("utf-8"))
        liq = tfl.load_futures_liquidity(AS_OF, UNIV, debug_dir=str(tmp_path))
        self._assert_failed(liq, tmp_path, "表頭")
        assert "系統忙碌中" in (tmp_path / "futdatadown_head.csv").read_text(encoding="utf-8")

    def test_garbage_html(self, monkeypatch, tmp_path):
        install_fake_http(monkeypatch, html="<html>maintenance</html>",
                          csv=make_fut_csv_bytes(_contracts_for(UNIV), DATES7))
        self._assert_failed(tfl.load_futures_liquidity(AS_OF, UNIV, debug_dir=str(tmp_path)), tmp_path)

    def test_low_mapping_coverage(self, monkeypatch, tmp_path):
        install_fake_http(monkeypatch, html=make_stock_lists_html(UNIV[:2]),
                          csv=make_fut_csv_bytes(_contracts_for(UNIV), DATES7))
        liq = tfl.load_futures_liquidity(AS_OF, UNIV, debug_dir=str(tmp_path))
        self._assert_failed(liq, tmp_path, "對照表只對照到2/5")

    def test_csv_without_stock_futures(self, monkeypatch, tmp_path):
        install_fake_http(monkeypatch, html=make_stock_lists_html(UNIV),
                          csv=make_fut_csv_bytes({"TX": {"near": 100000, "far": 1, "night": 1, "oi": 1}}, DATES7))
        liq = tfl.load_futures_liquidity(AS_OF, UNIV, debug_dir=str(tmp_path))
        self._assert_failed(liq, tmp_path, "只找到0/5")

    def test_unexpected_exception_is_caught(self, monkeypatch, tmp_path):
        install_fake_http(monkeypatch, html=make_stock_lists_html(UNIV),
                          csv=make_fut_csv_bytes(_contracts_for(UNIV), DATES7))
        monkeypatch.setattr(tfl, "compute_contract_stats", lambda *a, **k: 1 / 0)
        liq = tfl.load_futures_liquidity(AS_OF, UNIV, debug_dir=str(tmp_path))
        self._assert_failed(liq, tmp_path, "未預期錯誤")

    def test_old_debug_files_removed_on_failure(self, monkeypatch, tmp_path):
        install_fake_http(monkeypatch, html=make_stock_lists_html(UNIV),
                          csv=make_fut_csv_bytes(_contracts_for(UNIV), DATES7))
        tfl.load_futures_liquidity(AS_OF, UNIV, debug_dir=str(tmp_path))
        assert (tmp_path / "futdatadown_head.csv").exists()
        install_fake_http(monkeypatch, html_status=503)
        tfl.load_futures_liquidity(AS_OF, UNIV, debug_dir=str(tmp_path))
        assert not (tmp_path / "futdatadown_head.csv").exists()  # 不留昨天的檔案誤導
        assert not (tmp_path / "stocklists_head.html").exists()
