"""
taifex_futures_loader.py —— 股票期貨「本身」的流動性資料(日成交量、近月未平倉量、最後最佳買賣價)。

給daily_squeeze_signals.py的「個股期貨清單」用：只有實際要交易的那一種契約(有小型契約的股票用
小型100股、沒有的用標準2000股，跟taifex_universe.get_contract_multiplier()同一個規則)流動性夠，
訊號才會列在期貨清單裡。

⚠️ 誠實聲明：開發環境連不到taifex.com.tw，下面的網址、表單欄位、欄位名稱都是「目前最好的理解」，
**沒有用真實回應驗證過**。第一次在GitHub Actions跑完之後，請看 signals/debug/ 底下的原始回應與
parse_report.json 確認格式(見README/回報說明)。所以這支模組的設計原則是：

  **失敗就關閉(fail closed)**：對照表或成交量資料抓不到、解析不出來、或看起來不對(例如找不到任何
  股票期貨契約、對照表涵蓋不到一半的標的)，回傳ok=False，期貨清單今天一檔都不列，並顯示大大的警告。
  **絕對不會**退回「假設流動性沒問題」。

資料來源：
1. 股票 ↔ 期貨契約代碼對照：期交所「股票期貨、選擇權商品標的」頁面
     GET https://www.taifex.com.tw/cht/2/stockLists
   HTML表格。用標準函式庫html.parser解析(不用pandas.read_html：它需要lxml/bs4，
   requirements.txt沒有列，裝不裝得到取決於yfinance版本的相依套件，不想讓每日掃描賭這個)。
   欄位用「子字串」比對(不寫死欄位位置)：
     股票代號：含「證券代號」(或含「代號」但不含「英文/期貨/選擇權」)
     標準契約：含「期貨」且含「英文/代碼/代號」，不含「小型/選擇權」
     小型契約：含「小型」且含「期貨」，且含「英文/代碼/代號」
   值也會檢查格式(股票代號=4~6位數字可帶1個英文字母；期貨代碼=2~4個大寫英數字)，不合格的當作沒有。
   同一檔股票出現兩次而且代碼不同 → 視為對照不明確，那一檔不對照(fail closed)。
2. 每日行情：期交所「期貨每日交易行情下載」
     POST https://www.taifex.com.tw/cht/3/futDataDown
     表單：down_type=1, commodity_id=all, commodity_id2=, queryStartDate=YYYY/MM/DD, queryEndDate=YYYY/MM/DD
   回傳(預期)Big5/cp950編碼的CSV，欄位(預期)：交易日期, 契約, 到期月份(週別), 開盤價, 最高價, 最低價,
   收盤價, 漲跌價, 漲跌%, 成交量, 結算價, 未沖銷契約數, 最後最佳買價, 最後最佳賣價, 歷史最高價,
   歷史最低價, 是否因訊息面暫停交易, 交易時段, 價差對單式委託成交量。
   解析規則：
     - 編碼依序試 utf-8-sig(嚴格) → cp950 → big5hkscs，最後cp950(errors=replace)。
     - 表頭=前20行裡第一個同時含「契約」與「成交量」的行；欄位用子字串比對(「契約」排除「未沖銷」，
       「成交量」排除「價差」——「價差對單式委託成交量」不是單式成交量)。
     - 每格去空白；'-'、''、'--' 當NaN；數字去掉千分位逗號。
     - 「到期月份」含'/'的是價差(跨月)委託列，整列丟掉(不算成交量、不算未平倉)。
     - 交易時段：含「盤後」=盤後，其他(含「一般」)=一般；沒有這個欄位時全部當一般。
   查詢區間=as_of往前14個日曆天(長假也拿得到5個交易日)。

算法(每個期貨契約代碼)：
  - 交易日 = 資料裡「一般時段」出現過的交易日期(≤as_of)，取最後5個(AVG_DAYS)。不到3個(MIN_DAYS)→失敗。
  - 5日均量 = 這5個交易日裡，這個契約「所有到期月份、一般+盤後兩個時段」的成交量加總 / 交易日數
    (某天沒有這個契約的列=當天0口)。不含價差委託(價差列整列丟掉、「價差對單式委託成交量」不加)。
  - 近月 = 最後一個交易日、一般時段裡，到期月份(YYYYMM)最小的那一個；近月未平倉 = 那一列的未沖銷契約數
    (一般時段那列是NaN時，才用同一天同月份其他時段那列的數字)。
  - 最後最佳買/賣價 = 同一列；買賣價差換算成幾檔(用股票的升降單位表逐檔走，跨級距時tick跟著變)。

快取：cache_dir不是None時，原始回應存在那裡(refresh=False時直接讀)。daily_squeeze_signals.py傳暫存資料夾、
refresh=True(跟股價資料一樣：每天一定重新抓，用完就刪)。

診斷檔(debug_dir，每天覆寫，保持很小)：
  stocklists_head.html   ：stockLists頁面從第一個<table起的前200行(沒有<table就是整頁前200行)
  futdatadown_head.csv   ：行情CSV解碼後的前200行(存成UTF-8，方便在GitHub上直接看)
  parse_report.json      ：偵測到的欄位、列數、日期、對照涵蓋率、失敗原因
"""
import csv
import datetime
import io
import json
import os
import re
import time
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import requests

from network_utils import run_with_hard_timeout
from taifex_universe import get_contract_multiplier

TAIFEX_BASE = "https://www.taifex.com.tw"
STOCK_LISTS_URL = TAIFEX_BASE + "/cht/2/stockLists"
FUT_DATA_DOWN_URL = TAIFEX_BASE + "/cht/3/futDataDown"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": TAIFEX_BASE + "/cht/3/futDataDown",
}
HTTP_TIMEOUT_SECONDS = 30
HARD_TIMEOUT_SECONDS = HTTP_TIMEOUT_SECONDS + 10
MAX_RETRIES = 2              # 失敗後最多再試2次(共3次)
RETRY_BACKOFF_SECONDS = 3

LOOKBACK_CALENDAR_DAYS = 14
AVG_DAYS = 5
MIN_DAYS = 3
MIN_MAPPING_COVERAGE = 0.5      # 對照表要涵蓋至少一半的標的(實際交易的那種契約)，不然當作解析失敗
MIN_CSV_CONTRACT_SHARE = 0.2    # 行情CSV裡至少要找得到20%標的的契約，不然當作「資料不對」(例如只回指數期貨)
DEBUG_HEAD_LINES = 200

STOCK_CODE_RE = re.compile(r"^\d{4,6}[A-Z]?$")
FUT_CODE_RE = re.compile(r"^[A-Z0-9]{2,4}$")
MONTH_RE = re.compile(r"^\d{6}(W\d)?$")
NA_TOKENS = {"", "-", "--", "—", "N/A", "NA", "nan"}

FAILURE_MESSAGE = "期貨流動性資料抓取/解析失敗，今天期貨清單不列任何標的(不是沒有訊號)"


class LiquidityDataError(Exception):
    """資料抓不到/解析不出來/看起來不對 → 期貨清單fail closed。"""


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
def _http(method: str, url: str, data=None):
    """回傳(content_bytes, None)或(None, 錯誤說明)。timeout + 硬性逾時 + 最多重試MAX_RETRIES次。"""
    last = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            fn = requests.post if method == "POST" else requests.get
            kwargs = {"headers": HEADERS, "timeout": HTTP_TIMEOUT_SECONDS}
            if data is not None:
                kwargs["data"] = data
            resp = run_with_hard_timeout(fn, args=(url,), kwargs=kwargs, timeout=HARD_TIMEOUT_SECONDS)
            if resp.status_code == 200 and resp.content:
                return resp.content, None
            last = f"HTTP {resp.status_code}" + ("(內容是空的)" if resp.status_code == 200 else "")
        except Exception as e:  # 連線錯誤/逾時/HardTimeout
            last = f"{type(e).__name__}: {e}"
        print(f"[taifex_futures_loader] {method} {url} 第{attempt + 1}次失敗：{last}", flush=True)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS)
    return None, last


def _cached_fetch(method, url, data, cache_dir, cache_name, refresh):
    path = os.path.join(cache_dir, cache_name) if cache_dir else None
    if path and not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            content = f.read()
        if content:
            return content, None
    content, err = _http(method, url, data)
    if content is not None and path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
    return content, err


# ----------------------------------------------------------------------------
# 解碼/小工具
# ----------------------------------------------------------------------------
def decode_bytes(content: bytes, prefer=("utf-8-sig", "cp950", "big5hkscs")):
    """回傳(text, encoding)。嚴格依序試；全部失敗才用cp950(errors=replace)。"""
    for enc in prefer:
        try:
            return content.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("cp950", errors="replace"), "cp950(replace)"


def _norm(s) -> str:
    return "".join(str(s).split())


def _to_float(x) -> float:
    s = _norm(x).replace(",", "")
    if s in NA_TOKENS:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def find_column(columns, include_any, exclude_any=(), require_any=()):
    """欄位名稱子字串比對：名稱(去空白後)含include_any任一個、含require_any任一個(有給的話)、
    不含exclude_any任何一個。完全相等的優先。找不到回傳None。"""
    cands = []
    for col in columns:
        c = _norm(col)
        if not c:
            continue
        if not any(k in c for k in include_any):
            continue
        if any(k in c for k in exclude_any):
            continue
        if require_any and not any(k in c for k in require_any):
            continue
        cands.append(col)
    for col in cands:
        if _norm(col) in include_any:
            return col
    return cands[0] if cands else None


def spread_in_ticks(bid: float, ask: float) -> float:
    """最佳買賣價差是幾檔：從買價逐檔往上走到賣價(跨價位級距時tick跟著變)。資料不合理回傳NaN。"""
    from squeeze_kdj_signal import taiwan_tick_size  # 延遲import：loader本身保持輕量
    if not (np.isfinite(bid) and np.isfinite(ask)) or bid <= 0 or ask < bid:
        return np.nan
    n, p = 0, bid
    while p < ask - 1e-9 and n < 10_000:
        p = round(p + taiwan_tick_size(p), 4)
        n += 1
    return float(n)


# ----------------------------------------------------------------------------
# 1. stockLists：股票代號 ↔ 期貨契約代碼
# ----------------------------------------------------------------------------
class _TableParser(HTMLParser):
    """極簡HTML表格解析：收集每個<table>的列(每列=各格文字)。容忍沒關的<td>/<tr>。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self._stack = []
        self._row = None
        self._cell = None

    def _flush_cell(self):
        if self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
        self._cell = None

    def _flush_row(self):
        self._flush_cell()
        if self._row is not None and self._stack and self._row:
            self._stack[-1].append(self._row)
        self._row = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._stack.append([])
        elif tag == "tr" and self._stack:
            self._flush_row()
            self._row = []
        elif tag in ("td", "th") and self._stack:
            if self._row is None:
                self._row = []
            self._flush_cell()
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._flush_cell()
        elif tag == "tr":
            self._flush_row()
        elif tag == "table" and self._stack:
            self._flush_row()
            self.tables.append(self._stack.pop())

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _detect_stock_list_header(row):
    code = find_column(row, ("證券代號",)) or find_column(row, ("代號",), exclude_any=("英文", "期貨", "選擇權"))
    std = find_column(row, ("期貨",), exclude_any=("小型", "選擇權"), require_any=("英文", "代碼", "代號"))
    mini = find_column(row, ("小型",), require_any=("期貨",))
    if mini is not None and not any(k in _norm(mini) for k in ("英文", "代碼", "代號")):
        mini = None  # 「小型股票期貨標的(○)」這種旗標欄不是代碼
    name = find_column(row, ("證券名稱",)) or find_column(row, ("名稱",), exclude_any=("英文", "期貨", "選擇權"))
    return {"code": code, "std": std, "mini": mini, "name": name}


def parse_stock_list_html(html_text: str):
    """回傳(mapping, info)。mapping = {股票代號: {"std": 標準契約代碼或None, "mini": 小型契約代碼或None, "name": 名稱}}。
    找不到可用的表格 → LiquidityDataError。"""
    parser = _TableParser()
    parser.feed(html_text)
    parser.close()
    info = {"tables_found": len(parser.tables), "table_sizes": [len(t) for t in parser.tables][:20]}
    for ti, table in enumerate(parser.tables):
        for hi in range(min(3, len(table))):
            header = table[hi]
            cols = _detect_stock_list_header(header)
            if cols["code"] is None or cols["std"] is None:
                continue
            idx = {k: header.index(v) if v is not None else None for k, v in cols.items()}
            mapping, conflicts, bad_rows = {}, set(), 0
            for row in table[hi + 1:]:
                def cell(key):
                    i = idx[key]
                    return row[i].strip() if i is not None and i < len(row) else ""
                code = _norm(cell("code"))
                if not STOCK_CODE_RE.match(code):
                    bad_rows += 1
                    continue
                std = _norm(cell("std")).upper()
                mini = _norm(cell("mini")).upper()
                entry = {"std": std if FUT_CODE_RE.match(std) else None,
                         "mini": mini if FUT_CODE_RE.match(mini) else None,
                         "name": cell("name")}
                if code in mapping:
                    old = mapping[code]
                    for k in ("std", "mini"):
                        if old[k] and entry[k] and old[k] != entry[k]:
                            conflicts.add(code)
                        old[k] = old[k] or entry[k]
                else:
                    mapping[code] = entry
            for code in conflicts:  # 同一檔股票出現不同代碼 → 不知道哪個才對，fail closed
                mapping[code] = {"std": None, "mini": None, "name": mapping[code]["name"], "conflict": True}
            info.update({
                "table_index": ti, "header_row_index": hi, "header": header,
                "detected_columns": cols, "data_rows": len(table) - hi - 1, "skipped_rows": bad_rows,
                "stocks": len(mapping), "with_std_code": sum(1 for m in mapping.values() if m["std"]),
                "with_mini_code": sum(1 for m in mapping.values() if m["mini"]),
                "conflicting_codes": sorted(conflicts),
                "sample_rows": table[hi + 1: hi + 6],
            })
            if not mapping:
                continue
            return mapping, info
    raise LiquidityDataError(
        f"stockLists頁面找不到「證券代號＋股票期貨代碼」的表格(找到{len(parser.tables)}個表格)")


def traded_contract_kind(code: str) -> str:
    """實際交易的契約：跟get_contract_multiplier()同一個規則(有小型契約→小型100股，否則標準2000股)。"""
    return "mini" if get_contract_multiplier(code, 0.0) == 100 else "std"


def map_universe(mapping: dict, universe) -> dict:
    """{股票代號: 實際交易的期貨契約代碼或None}。"""
    out = {}
    for code in universe:
        m = mapping.get(code)
        out[code] = m.get(traded_contract_kind(code)) if m else None
    return out


# ----------------------------------------------------------------------------
# 2. futDataDown：每日行情CSV
# ----------------------------------------------------------------------------
def parse_fut_csv(content: bytes):
    """回傳(df, info)。df欄位：date, contract, month, session('一般'/'盤後'), volume, oi, bid, ask, close。
    價差列已丟掉。欄位偵測不到 → LiquidityDataError。"""
    text, enc = decode_bytes(content)
    rows = list(csv.reader(io.StringIO(text)))
    info = {"encoding": enc, "raw_lines": len(rows)}
    hdr_i = None
    for i, r in enumerate(rows[:20]):
        joined = "".join(_norm(c) for c in r)
        if "契約" in joined and "成交量" in joined:
            hdr_i = i
            break
    if hdr_i is None:
        info["first_line"] = (text.splitlines() or [""])[0][:200]
        raise LiquidityDataError("行情CSV找不到表頭(前20行沒有同時含「契約」與「成交量」的行)，可能回傳的是網頁或錯誤訊息")
    header = [_norm(c) for c in rows[hdr_i]]
    cols = {
        "date": find_column(header, ("交易日期", "日期")),
        "contract": find_column(header, ("契約", "商品代號"), exclude_any=("未沖銷", "未平倉", "數", "名稱")),
        "month": find_column(header, ("到期月份", "契約月份")),
        "volume": find_column(header, ("成交量",), exclude_any=("價差",)),
        "oi": find_column(header, ("未沖銷", "未平倉")),
        "bid": find_column(header, ("最佳買價",)),
        "ask": find_column(header, ("最佳賣價",)),
        "session": find_column(header, ("交易時段",)),
        "close": find_column(header, ("收盤價",)),
    }
    info["header"] = header
    info["detected_columns"] = cols
    missing = [k for k in ("date", "contract", "month", "volume", "oi") if cols[k] is None]
    if missing:
        raise LiquidityDataError(f"行情CSV缺少必要欄位{missing}(表頭={header})")
    pos = {k: header.index(v) if v is not None else None for k, v in cols.items()}

    recs, bad_date, spread_rows, short_rows = [], 0, 0, 0
    for r in rows[hdr_i + 1:]:
        if not any(_norm(c) for c in r):
            continue
        if len(r) <= max(p for p in pos.values() if p is not None):
            short_rows += 1
            continue

        def g(k):
            p = pos[k]
            return r[p] if p is not None else ""
        month = _norm(g("month"))
        contract = _norm(g("contract")).upper()
        if "/" in month or "/" in contract:
            spread_rows += 1
            continue
        d = pd.to_datetime(_norm(g("date")), errors="coerce")
        if pd.isna(d):
            bad_date += 1
            continue
        sess = _norm(g("session"))
        recs.append({
            "date": d.normalize(), "contract": contract, "month": month,
            "session": "盤後" if "盤後" in sess else "一般",
            "volume": _to_float(g("volume")), "oi": _to_float(g("oi")),
            "bid": _to_float(g("bid")), "ask": _to_float(g("ask")), "close": _to_float(g("close")),
        })
    df = pd.DataFrame(recs, columns=["date", "contract", "month", "session", "volume", "oi", "bid", "ask", "close"])
    info.update({
        "data_rows": len(df), "spread_rows_dropped": spread_rows, "bad_date_rows": bad_date,
        "short_rows": short_rows,
        "sessions": df["session"].value_counts().to_dict() if len(df) else {},
        "raw_session_values": sorted({_norm(r[pos["session"]]) for r in rows[hdr_i + 1:]
                                      if pos["session"] is not None and len(r) > pos["session"]})[:10],
        "dates": sorted({d.date().isoformat() for d in df["date"]}) if len(df) else [],
        "contracts": int(df["contract"].nunique()) if len(df) else 0,
    })
    if df.empty:
        raise LiquidityDataError("行情CSV有表頭但沒有任何可用的資料列")
    return df, info


def compute_contract_stats(df: pd.DataFrame, as_of, contracts) -> tuple:
    """回傳(stats_by_contract, info)。stats：avg_volume, days, near_month, near_oi, bid, ask, spread_ticks, latest_date。"""
    as_of = pd.Timestamp(as_of).normalize()
    df = df[df["date"] <= as_of]
    reg_dates = sorted(df.loc[df["session"] == "一般", "date"].unique())
    if not reg_dates:  # 沒有一般時段的列(例如沒有交易時段欄位但全部被歸類到盤後——理論上不會發生)
        reg_dates = sorted(df["date"].unique())
    dates = [pd.Timestamp(d) for d in reg_dates[-AVG_DAYS:]]
    info = {"trading_dates_used": [d.date().isoformat() for d in dates]}
    if len(dates) < MIN_DAYS:
        raise LiquidityDataError(f"行情資料只有{len(dates)}個交易日(≤訊號日)，至少要{MIN_DAYS}個")
    latest = dates[-1]
    info["latest_date"] = latest.date().isoformat()
    win = df[df["date"].isin(dates)]
    groups = {c: g for c, g in win.groupby("contract")}
    stats = {}
    for c in contracts:
        g = groups.get(c)
        if g is None:
            continue
        avg_vol = float(np.nansum(g["volume"].to_numpy())) / len(dates)
        last = g[(g["date"] == latest) & g["month"].str.match(MONTH_RE)]
        reg = last[last["session"] == "一般"]
        base = reg if not reg.empty else last
        near_month, near_oi, bid, ask = None, np.nan, np.nan, np.nan
        if not base.empty:
            near_month = sorted(base["month"].unique())[0]
            row = base[base["month"] == near_month].iloc[0]
            near_oi, bid, ask = float(row["oi"]), float(row["bid"]), float(row["ask"])
            if not np.isfinite(near_oi):
                alt = last[(last["month"] == near_month)]["oi"].dropna()
                if len(alt):
                    near_oi = float(alt.iloc[0])
        stats[c] = {"avg_volume": avg_vol, "days": len(dates), "near_month": near_month, "near_oi": near_oi,
                    "bid": bid, "ask": ask, "spread_ticks": spread_in_ticks(bid, ask),
                    "latest_date": latest.date().isoformat()}
    return stats, info


# ----------------------------------------------------------------------------
# 3. 對外：抓 + 解析 + 檢查 + 診斷
# ----------------------------------------------------------------------------
def fut_data_down_form(as_of) -> dict:
    end = pd.Timestamp(as_of).date()
    start = end - datetime.timedelta(days=LOOKBACK_CALENDAR_DAYS)
    return {"down_type": "1", "commodity_id": "all", "commodity_id2": "",
            "queryStartDate": start.strftime("%Y/%m/%d"), "queryEndDate": end.strftime("%Y/%m/%d")}


def _head_lines(text: str, start_marker: str = None, n: int = DEBUG_HEAD_LINES) -> str:
    if start_marker:
        i = text.lower().find(start_marker)
        if i >= 0:
            text = text[i:]
    return "\n".join(text.splitlines()[:n]) + "\n"


DEBUG_FILES = ("stocklists_head.html", "futdatadown_head.csv", "parse_report.json")


def write_diagnostics(debug_dir, html_text, csv_text, report):
    os.makedirs(debug_dir, exist_ok=True)
    for name in DEBUG_FILES:  # 每天覆寫：先刪掉昨天的，避免抓失敗時留下舊檔誤導
        p = os.path.join(debug_dir, name)
        if os.path.exists(p):
            os.remove(p)
    if html_text is not None:
        with open(os.path.join(debug_dir, "stocklists_head.html"), "w", encoding="utf-8") as f:
            f.write(_head_lines(html_text, "<table"))
    if csv_text is not None:
        with open(os.path.join(debug_dir, "futdatadown_head.csv"), "w", encoding="utf-8") as f:
            f.write(_head_lines(csv_text))
    with open(os.path.join(debug_dir, "parse_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


def format_report(report: dict) -> str:
    lines = ["===== 期貨流動性資料解析報告(taifex_futures_loader) ====="]
    lines.append(f"結果：{'OK' if report.get('ok') else '失敗 → ' + str(report.get('error'))}")
    sl = report.get("stock_lists", {})
    lines.append(f"[stockLists] {sl.get('url')} 錯誤={sl.get('fetch_error')} 表格數={sl.get('tables_found')} "
                 f"偵測欄位={sl.get('detected_columns')} 股票數={sl.get('stocks')} "
                 f"有標準代碼={sl.get('with_std_code')} 有小型代碼={sl.get('with_mini_code')}")
    cov = report.get("coverage", {})
    lines.append(f"[對照涵蓋] 標的{cov.get('universe')}檔，實際交易契約對照到{cov.get('mapped')}檔"
                 f"({cov.get('mapped_share')})；有小型但找不到小型代碼{cov.get('mini_missing')}檔；"
                 f"未對照(前20)={cov.get('unmapped_codes', [])[:20]}")
    fd = report.get("fut_data_down", {})
    lines.append(f"[futDataDown] {fd.get('url')} 表單={fd.get('form')} 錯誤={fd.get('fetch_error')} "
                 f"編碼={fd.get('encoding')} 行數={fd.get('raw_lines')} 資料列={fd.get('data_rows')} "
                 f"價差列丟掉={fd.get('spread_rows_dropped')} 時段={fd.get('sessions')}")
    lines.append(f"  偵測欄位={fd.get('detected_columns')}")
    lines.append(f"  使用交易日={report.get('stats', {}).get('trading_dates_used')} "
                 f"行情裡找到的標的契約={cov.get('contracts_in_csv')}")
    return "\n".join(lines)


def load_futures_liquidity(as_of, universe, cache_dir=None, refresh=True, debug_dir=None, verbose=True) -> dict:
    """抓+解析，回傳：
      {"ok": bool, "error": str|None, "mapping": {股票: {"std","mini","name"}}, "traded": {股票: 期貨代碼|None},
       "stats": {期貨代碼: {...}}, "latest_date": "YYYY-MM-DD"|None, "trading_dates": [...], "report": {...}}
    任何一步失敗都回傳ok=False(不丟例外)。呼叫端要fail closed：ok=False時期貨清單一檔都不列。"""
    as_of = pd.Timestamp(as_of).normalize()
    universe = list(universe)
    form = fut_data_down_form(as_of)
    report = {"as_of": as_of.date().isoformat(), "ok": False, "error": None,
              "stock_lists": {"url": STOCK_LISTS_URL, "method": "GET"},
              "fut_data_down": {"url": FUT_DATA_DOWN_URL, "method": "POST", "form": form},
              "coverage": {}, "stats": {},
              "thresholds": {"MIN_MAPPING_COVERAGE": MIN_MAPPING_COVERAGE,
                             "MIN_CSV_CONTRACT_SHARE": MIN_CSV_CONTRACT_SHARE,
                             "AVG_DAYS": AVG_DAYS, "MIN_DAYS": MIN_DAYS}}
    out = {"ok": False, "error": None, "mapping": {}, "traded": {c: None for c in universe}, "stats": {},
           "latest_date": None, "trading_dates": [], "report": report}
    html_text = csv_text = None
    try:
        # --- 對照表 ---
        content, err = _cached_fetch("GET", STOCK_LISTS_URL, None, cache_dir, "stocklists.html", refresh)
        report["stock_lists"]["fetch_error"] = err
        if content is None:
            raise LiquidityDataError(f"stockLists抓取失敗：{err}")
        html_text, enc = decode_bytes(content, prefer=("utf-8", "cp950", "big5hkscs"))
        report["stock_lists"]["encoding"] = enc
        report["stock_lists"]["bytes"] = len(content)
        mapping, sl_info = parse_stock_list_html(html_text)
        report["stock_lists"].update(sl_info)
        out["mapping"] = mapping
        traded = map_universe(mapping, universe)
        out["traded"] = traded
        mapped = [c for c in universe if traded[c]]
        mini_missing = [c for c in universe if traded_contract_kind(c) == "mini"
                        and mapping.get(c) and not mapping[c].get("mini")]
        share = len(mapped) / len(universe) if universe else 0.0
        report["coverage"] = {"universe": len(universe), "mapped": len(mapped), "mapped_share": round(share, 3),
                              "mini_missing": len(mini_missing), "mini_missing_codes": mini_missing[:50],
                              "unmapped_codes": [c for c in universe if not traded[c]][:100]}
        if share < MIN_MAPPING_COVERAGE:
            raise LiquidityDataError(f"對照表只對照到{len(mapped)}/{len(universe)}檔標的的實際交易契約"
                                     f"(<{MIN_MAPPING_COVERAGE:.0%})，看起來解析錯了")

        # --- 行情 ---
        content, err = _cached_fetch("POST", FUT_DATA_DOWN_URL, form, cache_dir,
                                     f"futdatadown_{form['queryStartDate'].replace('/', '')}_"
                                     f"{form['queryEndDate'].replace('/', '')}.csv", refresh)
        report["fut_data_down"]["fetch_error"] = err
        if content is None:
            raise LiquidityDataError(f"futDataDown抓取失敗：{err}")
        report["fut_data_down"]["bytes"] = len(content)
        csv_text, _ = decode_bytes(content)
        df, fd_info = parse_fut_csv(content)
        report["fut_data_down"].update(fd_info)
        wanted = sorted({traded[c] for c in mapped})
        present = set(df["contract"].unique()) & set(wanted)
        report["coverage"]["contracts_in_csv"] = len(present)
        report["coverage"]["traded_contracts"] = len(wanted)
        if len(present) < max(1, MIN_CSV_CONTRACT_SHARE * len(wanted)):
            raise LiquidityDataError(f"行情CSV裡只找到{len(present)}/{len(wanted)}個標的的股票期貨契約，"
                                     "看起來不是完整的股票期貨行情")
        stats, st_info = compute_contract_stats(df, as_of, wanted)
        report["stats"] = st_info
        out["stats"] = stats
        out["latest_date"] = st_info["latest_date"]
        out["trading_dates"] = st_info["trading_dates_used"]
        report["stats"]["sample"] = {c: stats[c] for c in list(stats)[:5]}
        out["ok"] = True
        report["ok"] = True
    except LiquidityDataError as e:
        out["error"] = str(e)
    except Exception as e:  # 任何沒料到的狀況一樣fail closed，不讓整個每日掃描掛掉
        out["error"] = f"未預期錯誤 {type(e).__name__}: {e}"
    report["error"] = out["error"]
    if not out["ok"]:
        out["stats"] = {}
    if debug_dir:
        try:
            write_diagnostics(debug_dir, html_text, csv_text, report)
        except Exception as e:
            print(f"[taifex_futures_loader] 寫診斷檔失敗：{e}", flush=True)
    if verbose:
        print(format_report(report), flush=True)
    return out


def failed_liquidity(error: str) -> dict:
    """fail-closed用的空結果(例如呼叫端根本沒有抓資料)。"""
    return {"ok": False, "error": error, "mapping": {}, "traded": {}, "stats": {}, "latest_date": None,
            "trading_dates": [], "report": {"ok": False, "error": error}}


def evaluate_code(liq: dict, code: str, min_volume: float, min_oi: float) -> dict:
    """單一股票的期貨流動性判斷。回傳{"passed": bool, "reason": str|None, "fut_code", "avg_volume", "near_oi",
    "near_month", "spread_ticks", "bid", "ask"}。資料失敗/對照不到/行情沒有這個契約 → passed=False。"""
    res = {"passed": False, "reason": None, "fut_code": None, "avg_volume": np.nan, "near_oi": np.nan,
           "near_month": None, "spread_ticks": np.nan, "bid": np.nan, "ask": np.nan}
    if not liq or not liq.get("ok"):
        res["reason"] = "期貨流動性資料失敗"
        return res
    fut = (liq.get("traded") or {}).get(code)
    if not fut:
        kind = "小型" if traded_contract_kind(code) == "mini" else "標準"
        res["reason"] = f"對照表找不到{kind}契約代碼"
        return res
    res["fut_code"] = fut
    st = liq["stats"].get(fut)
    if st is None:
        res["reason"] = f"行情裡沒有{fut}(5日均量視為0口)"
        res["avg_volume"] = 0.0
        return res
    for k in ("avg_volume", "near_oi", "near_month", "spread_ticks", "bid", "ask"):
        res[k] = st[k]
    reasons = []
    if not (np.isfinite(st["avg_volume"]) and st["avg_volume"] >= min_volume):
        reasons.append(f"5日均量{st['avg_volume']:,.0f}口<{min_volume:,.0f}")
    if not (np.isfinite(st["near_oi"]) and st["near_oi"] >= min_oi):
        oi = "無資料" if not np.isfinite(st["near_oi"]) else f"{st['near_oi']:,.0f}口"
        reasons.append(f"近月未平倉{oi}<{min_oi:,.0f}")
    res["passed"] = not reasons
    res["reason"] = "、".join(reasons) or None
    return res
