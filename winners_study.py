"""
winners_study.py — 描述性研究：「最近兩年大漲股的共同點」+「哪些簡單規則當時抓得到它們」

這不是策略回測、也不是選股建議，是一份「事後回頭看」的描述性研究，三個部分：
  分析1  起點(區間第一個交易日)當下就看得到的特徵(point-in-time)，跟「之後兩年漲1倍/2倍」的關係(lift)。
         重點：贏家常見的特徵，只有在「有這個特徵的股票贏家率明顯高於整體贏家率」時才有用——
         很多輸家可能也有同樣的特徵(例如「站上年線」在多頭市場裡大部分股票都有)。
  分析2  贏家的「路徑」：什麼時候開始漲、中途最多回檔多少(從高點回落%、幾倍ATR)，
         用來估計移動停損要放多寬才抱得住。
  分析3  事先登錄的幾個簡單進出場規則，對「全部股票」(不是只有贏家)逐檔模擬，看抓到多少贏家、賺多少。

資料來源(真實下載只在GitHub Actions上跑，sandbox沒有網路)：
  1. 股票清單+產業別：TWSE ISIN 頁面 https://isin.twse.com.tw/isin/C_public.jsp?strMode=2(上市)/4(上櫃)，
     cp950(Big5)編碼的HTML表格；只留「股票」分類、4位數代號、CFICode以ES開頭(有提供時)的列，
     排除ETF/權證/特別股/TDR。repo裡沒有現成的「全市場上市櫃+產業別」清單工具
     (taifex_universe.py只有股票期貨標的)，所以這裡新寫一個解析器。
     ⚠️ 只有「目前仍上市櫃」的股票 → 存活者偏誤(下市/被併購的股票不在清單裡)，報告裡會寫明。
  2. 股價：yfinance日K(.TW/.TWO後綴，沿用data_loader._symbol_for)，auto_adjust=False一次拿到
     原始價+Adj Close，用 Adj Close/Close 當還原因子算出還原OHLC(等同auto_adjust=True)，
     同時保留原始收盤與成交量(算成交金額)。批次下載+重試，失敗清單寫進debug/。
     收盤/開高低為NaN或<=0的列視為缺資料(整列丟掉)。
  3. 月營收：MOPS每月營收彙總表(一個市場一個月一頁，Big5 HTML、每個產業一個表格)；
     網址 t21sc03_{民國年}_{月}_0.html 是國內公司，_1.html 是外國(KY)公司，兩種都抓；
     上櫃的路徑用 /otc/t21sc03_...(既有revenue_data_loader.py用的是t21sc04，未經驗證)。
     網域先試 mopsov.twse.com.tw 再試 mops.twse.com.tw。
     ⚠️ 既有的revenue_data_loader.py只取第一個找得到「公司代號」的表格(=只有第一個產業)，
     也沒保留營收金額，所以這裡另寫一個逐表格解析器(用Python內建html.parser，不依賴lxml/bs4)，
     只沿用network_utils的硬性逾時。
     point-in-time規則：M月營收視為M+1月11日(含)起才看得到(法定公告期限是10日)。
     MOPS完全抓不到 → 不使用營收特徵繼續跑，報告最上面會寫清楚。

用法：
  python winners_study.py --output-dir results_winners [--start YYYY-MM-DD] [--end YYYY-MM-DD]
  python winners_study.py --window 2020-2021      # 預設區間(見WINDOW_PRESETS)
start留空 = end往前2年；end留空 = 最新資料(今天，台北時間)。
"""
import argparse
import datetime
import os
import re
import sys
import time
from html.parser import HTMLParser

import numpy as np
import pandas as pd

from network_utils import run_with_hard_timeout, HardTimeout
import winners_extra as wx

# ---------------------------------------------------------------------------
# 參數(事先登錄，不依結果調整)
# ---------------------------------------------------------------------------
WINDOW_PRESETS = {
    "最近兩年": ("", ""),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2020-2021": ("2020-01-01", "2021-12-31"),
    "2018-2019": ("2018-01-01", "2019-12-31"),
}
WARMUP_CALENDAR_DAYS = 400          # 52週高/MA240的暖機
REVENUE_LOOKBACK_MONTHS = 15        # 月營收從start往前抓幾個月
MIN_TURNOVER = 10_000_000           # 流動性門檻：起點20日平均成交金額(新台幣)
TRADE_NOTIONAL = 100_000            # 規則模擬：每筆固定金額
FEE_RATE = 0.001425                 # 手續費(買賣各一次，牌告費率)
TAX_RATE = 0.003                    # 證交稅(賣出)
TIER_DEFS = [                       # (欄位key, 中文名)
    ("w100", "漲1倍以上"),
    ("w200", "漲2倍以上"),
    ("m100", "曾經漲1倍"),
]
ENTRY_RULES = {
    "E_A": "收盤創252日新高(距上次新高≥20日)",
    "E_B": "E_A且營收3個月平均YoY>20%",
    "E_C": "E_A且RS前20%",
    "E_D": "E_B且RS前20%",
}
EXIT_RULES = {
    "X1": "移動停損3×ATR(自最高收盤)",
    "X2": "移動停損20%(自最高收盤)",
    "X3": "收盤跌破MA60→隔日開盤出",
    "X4": "固定停利4×ATR+停損1.5×ATR",
}
INITIAL_STOP_ATR = 2.0
X4_STOP_ATR = 1.5
X4_TARGET_ATR = 4.0
X1_TRAIL_ATR = 3.0
X2_TRAIL_PCT = 0.20
RS_TOP = 0.80
REV_YOY_MIN = 20.0

ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
ISIN_MODES = {"上市": 2, "上櫃": 4}
MOPS_DOMAINS = ["https://mopsov.twse.com.tw", "https://mops.twse.com.tw"]
# 檔名最後一碼：0=國內公司、1=外國公司(-KY)。題目只指定_0，這裡兩種都抓(KY股在贏家裡不少)，
# _1抓不到不影響整體(只是KY股沒有營收特徵)。
MOPS_PATH = {"上市": "/nas/t21/sii/t21sc03_{roc}_{m}_{kind}.html",
             "上櫃": "/nas/t21/otc/t21sc03_{roc}_{m}_{kind}.html"}
MOPS_KINDS = {0: "國內", 1: "外國(KY)"}
HTTP_HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")}
HTTP_TIMEOUT = 30
PRICE_BATCH_SIZE = 50
PRICE_BATCH_DELAY = 2.0
MOPS_DELAY = 2.0

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winners_cache")


# ===========================================================================
# HTML表格解析(內建html.parser，支援巢狀表格、colspan/rowspan、沒關閉的td)
# ===========================================================================
class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []          # 依開啟順序：{"rows": [[cell,...],...], "heading": str}
        self._stack = []          # 開啟中的表格(最內層在最後)
        self.last_heading = ""    # 最近一個「產業別：xxx」之類的標題文字

    def _cur(self):
        return self._stack[-1] if self._stack else None

    def _close_cell(self, t):
        if t is not None and t["cell"] is not None:
            cell = t["cell"]
            # 空白/換行/&nbsp;/全形空白都壓成一個半形空白(ISIN用全形空白分隔代號與名稱，正規式兩種都接受)
            cell["text"] = re.sub(r"\s+", " ", cell["text"]).strip()
            if t["row"] is None:
                t["row"] = []
            t["row"].append(cell)
            if "產業別" in cell["text"]:
                self.last_heading = cell["text"]
            t["cell"] = None

    def _close_row(self, t):
        if t is None:
            return
        self._close_cell(t)
        if t["row"] is not None:
            if t["row"]:
                t["table"]["rows"].append(t["row"])
            t["row"] = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        a = dict(attrs)
        if tag == "table":
            table = {"rows": [], "heading": self.last_heading}
            self.tables.append(table)
            self._stack.append({"table": table, "row": None, "cell": None})
        elif tag == "tr":
            t = self._cur()
            if t is not None:
                self._close_row(t)
                t["row"] = []
        elif tag in ("td", "th"):
            t = self._cur()
            if t is None:
                return
            self._close_cell(t)
            if t["row"] is None:
                t["row"] = []

            def _int(v):
                try:
                    return max(1, int(str(v).strip()))
                except (TypeError, ValueError):
                    return 1
            t["cell"] = {"text": "", "th": tag == "th",
                         "colspan": _int(a.get("colspan", 1)), "rowspan": _int(a.get("rowspan", 1))}
        elif tag == "br":
            t = self._cur()
            if t is not None and t["cell"] is not None:
                t["cell"]["text"] += " "

    def handle_endtag(self, tag):
        tag = tag.lower()
        t = self._cur()
        if t is None:
            return
        if tag in ("td", "th"):
            self._close_cell(t)
        elif tag == "tr":
            self._close_row(t)
        elif tag == "table":
            self._close_row(t)
            self._stack.pop()

    def handle_data(self, data):
        t = self._cur()
        if t is not None and t["cell"] is not None:
            t["cell"]["text"] += data
        elif "產業別" in data:
            self.last_heading = data.strip()

    def close(self):
        super().close()
        while self._stack:
            self._close_row(self._stack[-1])
            self._stack.pop()


def parse_html_tables(html: str) -> list:
    p = _TableParser()
    p.feed(html)
    p.close()
    return p.tables


def _expand_header(rows: list) -> list:
    """把含colspan/rowspan的表頭列攤平成每一欄一個名稱(上下層文字串起來)。"""
    grid = {}
    ncols = 0
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    grid[(r + dr, c + dc)] = cell["text"]
            c += cell["colspan"]
        ncols = max(ncols, c)
    names = []
    for c in range(ncols):
        parts = []
        for r in range(len(rows)):
            txt = grid.get((r, c), "")
            if txt and (not parts or parts[-1] != txt):
                parts.append(txt)
        names.append(re.sub(r"\s+", "", "".join(parts)))
    return names


def _num(s) -> float:
    """'1,234,567' → 1234567.0；'-12.34' → -12.34；''/'-'/'--'/'N/A' → NaN。"""
    if s is None:
        return float("nan")
    s = str(s).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "--", "N/A", "nan", "不適用"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# ISIN(上市/上櫃股票清單+產業別)
# ---------------------------------------------------------------------------
_CODE_NAME_RE = re.compile(r"^([0-9A-Za-z]+)[　\s]+(.+)$")
_EXCLUDED_SECTION_WORDS = ("特別", "存託", "權證", "受益", "ETF", "ETN", "債")


def parse_isin_html(html: str, market: str):
    """
    回傳 (DataFrame[code,name,market,industry,isin,listed_date,cfi], report dict)。
    保留：分類列(單一儲存格那種)是「股票」、代號是4位數字、CFICode有值時以ES開頭、名稱不含DR。
    """
    rows_out = []
    report = {"market": market, "rows_seen": 0, "kept": 0, "sections": {}, "excluded": {}}
    tables = parse_html_tables(html)
    for table in tables:
        idx = {"code_name": 0, "isin": 1, "listed": 2, "mkt": 3, "industry": 4, "cfi": 5}
        section = ""
        for row in table["rows"]:
            texts = [c["text"].strip() for c in row]
            nonempty = [t for t in texts if t]
            if len(nonempty) == 1 and (len(texts) == 1 or row[0]["colspan"] > 1):
                section = nonempty[0].replace(" ", "")
                report["sections"].setdefault(section, 0)
                continue
            joined = "".join(texts)
            if "有價證券代號" in joined or ("產業別" in joined and "CFICode" in joined.replace(" ", "")):
                for i, t in enumerate(texts):
                    t2 = t.replace(" ", "")
                    if "有價證券代號" in t2:
                        idx["code_name"] = i
                    elif "ISIN" in t2:
                        idx["isin"] = i
                    elif "上市日" in t2 or "上櫃日" in t2 or "掛牌日" in t2:
                        idx["listed"] = i
                    elif "市場別" in t2:
                        idx["mkt"] = i
                    elif "產業別" in t2:
                        idx["industry"] = i
                    elif "CFI" in t2:
                        idx["cfi"] = i
                continue
            if len(texts) < 3:
                continue
            m = _CODE_NAME_RE.match(texts[idx["code_name"]]) if idx["code_name"] < len(texts) else None
            if not m:
                continue
            report["rows_seen"] += 1
            if section:
                report["sections"][section] = report["sections"].get(section, 0) + 1
            code, name = m.group(1).strip(), m.group(2).strip()

            def _get(k):
                i = idx[k]
                return texts[i].strip() if i < len(texts) else ""
            cfi = _get("cfi")
            reason = None
            if "股票" not in section or any(w in section for w in _EXCLUDED_SECTION_WORDS):
                reason = f"分類:{section or '無'}"
            elif not re.fullmatch(r"\d{4}", code):
                reason = "代號非4位數字"
            elif cfi and not cfi.upper().startswith("ES"):
                reason = f"CFI:{cfi[:2]}"
            elif "-DR" in name.upper():
                reason = "DR"
            if reason:
                report["excluded"][reason] = report["excluded"].get(reason, 0) + 1
                continue
            rows_out.append({"code": code, "name": name, "market": market,
                             "industry": _get("industry") or "未分類",
                             "isin": _get("isin"), "listed_date": _get("listed"), "cfi": cfi})
    df = pd.DataFrame(rows_out, columns=["code", "name", "market", "industry", "isin", "listed_date", "cfi"])
    df = df.drop_duplicates("code", keep="first").reset_index(drop=True)
    report["kept"] = int(len(df))
    return df, report


# ---------------------------------------------------------------------------
# MOPS 月營收彙總表
# ---------------------------------------------------------------------------
_REV_COLS = ["code", "name", "industry", "revenue", "revenue_ly", "mom_pct", "yoy_pct", "cum_yoy_pct"]


def _map_revenue_columns(names: list) -> dict:
    m = {}
    for i, n in enumerate(names):
        n = n.replace("營業收入", "", 1) if n.startswith("營業收入") and len(n) > 4 else n
        if "公司代號" in n or n == "代號":
            m.setdefault("code", i)
        elif "公司名稱" in n:
            m.setdefault("name", i)
        elif "去年當月營收" in n:
            m.setdefault("revenue_ly", i)
        elif "當月營收" in n and "累計" not in n and "去年" not in n:
            m.setdefault("revenue", i)
        elif "上月比較增減" in n:
            m.setdefault("mom_pct", i)
        elif "去年同月增減" in n:
            m.setdefault("yoy_pct", i)
        elif "前期比較增減" in n:
            m.setdefault("cum_yoy_pct", i)
    return m


# 表頭解析不出來時的位置備援(MOPS常見11欄：代號,名稱,當月,上月,去年當月,上月增減%,去年同月增減%,當月累計,去年累計,前期增減%,備註)
_POSITIONAL = {"code": 0, "name": 1, "revenue": 2, "revenue_ly": 4, "mom_pct": 5, "yoy_pct": 6, "cum_yoy_pct": 9}


def parse_mops_revenue_html(html: str):
    """
    回傳 (DataFrame[_REV_COLS], report dict)。每個產業一個表格，全部都讀(不是只讀第一個)。
    YoY優先用表格自己的「去年同月增減(%)」，該欄空白時用 當月營收/去年當月營收 自己算。
    """
    tables = parse_html_tables(html)
    out = []
    report = {"tables": len(tables), "tables_with_data": 0, "header_tables": 0, "positional_tables": 0}
    for table in tables:
        rows = table["rows"]
        first_data = None
        for r, row in enumerate(rows):
            if row and re.fullmatch(r"\d{4,6}[A-Z]?", row[0]["text"].strip()):
                first_data = r
                break
        if first_data is None:
            continue
        header_rows = rows[:first_data]
        # 只取從含「公司代號」那一列開始的表頭(前面可能有產業別標題列)
        start_h = None
        for r, row in enumerate(header_rows):
            if any("公司代號" in c["text"].replace(" ", "") or c["text"].strip() == "代號" for c in row):
                start_h = r
                break
        colmap = {}
        if start_h is not None:
            colmap = _map_revenue_columns(_expand_header(header_rows[start_h:]))
        if "code" in colmap and "revenue" in colmap:
            report["header_tables"] += 1
        else:
            colmap = dict(_POSITIONAL)
            report["positional_tables"] += 1
        industry = ""
        h = table.get("heading", "") or ""
        for row in header_rows:  # 產業別標題列如果在同一個表格裡，以它為準
            for c in row:
                if "產業別" in c["text"]:
                    h = c["text"]
        if "產業別" in h:
            industry = re.split(r"[:：]", h, maxsplit=1)[-1].strip()
        n_before = len(out)
        for row in rows[first_data:]:
            texts = [c["text"].strip() for c in row]
            if not texts or not re.fullmatch(r"\d{4,6}[A-Z]?", texts[0]):
                continue
            if len(texts) <= max(colmap.get("code", 0), colmap.get("revenue", 2)):
                continue

            def _g(k):
                i = colmap.get(k)
                return texts[i] if (i is not None and i < len(texts)) else ""
            rec = {"code": _g("code"), "name": _g("name"), "industry": industry,
                   "revenue": _num(_g("revenue")), "revenue_ly": _num(_g("revenue_ly")),
                   "mom_pct": _num(_g("mom_pct")), "yoy_pct": _num(_g("yoy_pct")),
                   "cum_yoy_pct": _num(_g("cum_yoy_pct"))}
            if not np.isfinite(rec["yoy_pct"]) and np.isfinite(rec["revenue"]) \
                    and np.isfinite(rec["revenue_ly"]) and rec["revenue_ly"] > 0:
                rec["yoy_pct"] = (rec["revenue"] / rec["revenue_ly"] - 1) * 100
            out.append(rec)
        if len(out) > n_before:
            report["tables_with_data"] += 1
    df = pd.DataFrame(out, columns=_REV_COLS)
    df = df.drop_duplicates("code", keep="last").reset_index(drop=True)
    report["rows"] = int(len(df))
    return df, report


def revenue_available_date(year: int, month: int) -> pd.Timestamp:
    """M月營收從M+1月11日(含)起才視為已知(法定期限10日，保守多留一天)。"""
    y, m = (year + 1, 1) if month == 12 else (year, month + 1)
    return pd.Timestamp(datetime.date(y, m, 11))


def build_revenue_features(rev_long: pd.DataFrame) -> dict:
    """
    rev_long：[code, year, month, yoy_pct, ...] → {code: DataFrame(index=所屬月份月初,
    columns=[yoy, yoy3, yoy3_prev, accel, consec20, avail_date])}。
    每一列只用到「那個月(含)以前」的營收，再用avail_date決定哪天起看得到。
    """
    out = {}
    if rev_long is None or rev_long.empty:
        return out
    df = rev_long.copy()
    df["month_start"] = pd.to_datetime(dict(year=df["year"].astype(int), month=df["month"].astype(int), day=1))
    if "revenue" not in df.columns:
        df["revenue"] = np.nan
    for code, g in df.groupby("code"):
        g = g.drop_duplicates("month_start", keep="last").set_index("month_start").sort_index()
        s = g["yoy_pct"].astype(float)
        full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
        s = s.reindex(full)
        amt = pd.to_numeric(g["revenue"], errors="coerce").reindex(full)
        # 營收創新高：當月營收 ≥ 近12個月(含當月)最高，需要連續12個月都有資料
        high12 = (amt >= amt.rolling(12, min_periods=12).max()).astype(float)
        high12[amt.rolling(12, min_periods=12).max().isna()] = np.nan
        yoy3 = s.rolling(3, min_periods=2).mean()
        yoy3_prev = yoy3.shift(3)
        accel = yoy3 - yoy3_prev
        consec = []
        run = 0
        for v in s.values:
            run = run + 1 if (np.isfinite(v) and v > 20) else 0
            consec.append(run)
        f = pd.DataFrame({"yoy": s, "yoy3": yoy3, "yoy3_prev": yoy3_prev, "accel": accel,
                          "consec20": np.array(consec, dtype=float), "high12": high12}, index=full)
        f["avail_date"] = [revenue_available_date(d.year, d.month) for d in full]
        out[str(code)] = f
    return out


_REV_FEAT_COLS = ["yoy", "yoy3", "accel", "consec20", "high12"]
REVENUE_STALE_MONTHS = 4


def revenue_features_on_dates(rf: pd.DataFrame, dates) -> pd.DataFrame:
    """對每個日期d取「avail_date<=d」的最新一個月；最新已知月份太舊(>4個月前)視為沒資料。"""
    dates = pd.DatetimeIndex(dates)
    res = pd.DataFrame(np.nan, index=dates, columns=_REV_FEAT_COLS + ["rev_month"])
    res["rev_month"] = pd.NaT
    if rf is None or rf.empty or len(dates) == 0:
        return res
    avail = pd.DatetimeIndex(rf["avail_date"]).values
    pos = np.searchsorted(avail, dates.values, side="right") - 1
    ok = pos >= 0
    if ok.any():
        sub = rf.iloc[pos[ok]]
        months = pd.DatetimeIndex(sub.index)
        d_ok = dates[ok]
        age = (d_ok.year - months.year) * 12 + (d_ok.month - months.month)
        vals = sub[_REV_FEAT_COLS].to_numpy(dtype=float).copy()
        vals[np.asarray(age) > REVENUE_STALE_MONTHS] = np.nan
        res.loc[d_ok, _REV_FEAT_COLS] = vals
        res.loc[d_ok, "rev_month"] = months.values
    return res


# ===========================================================================
# 價格/指標
# ===========================================================================
def prepare_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    raw：yfinance auto_adjust=False 的 [Open,High,Low,Close,(Adj Close),Volume]。
    回傳 [open,high,low,close(還原),raw_close,volume,turnover]；NaN或<=0價格的列丟掉。
    還原因子 = Adj Close / Close(沒有Adj Close欄時=1，等於不還原)。
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "raw_close", "volume", "turnover"])
    df = raw.copy()
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    adj_col = "Adj Close" if "Adj Close" in df.columns else ("AdjClose" if "AdjClose" in df.columns else None)
    num = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    adj = pd.to_numeric(df[adj_col], errors="coerce") if adj_col else num["Close"]
    valid = (num[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (adj > 0)
    num = num[valid]
    adj = adj[valid]
    factor = adj / num["Close"]
    out = pd.DataFrame({
        "open": num["Open"] * factor, "high": num["High"] * factor, "low": num["Low"] * factor,
        "close": adj, "raw_close": num["Close"], "volume": num["Volume"].fillna(0.0),
    }, index=num.index)
    out["turnover"] = out["raw_close"] * out["volume"]
    return out


def compute_indicators(px: pd.DataFrame) -> pd.DataFrame:
    """全部都是「往回看」的滾動指標：某一天的值只用到那天(含)以前的資料。"""
    from mean_reversion_engine import compute_atr_correct  # 沿用repo既有ATR(TR的14日簡單平均)
    df = px.copy()
    c = df["close"]
    df["ma60"] = c.rolling(60, min_periods=60).mean()
    df["ma120"] = c.rolling(120, min_periods=120).mean()
    df["ma240"] = c.rolling(240, min_periods=240).mean()
    df["hh252"] = c.rolling(252, min_periods=252).max()                      # 含當天
    df["hh252_prev"] = c.shift(1).rolling(252, min_periods=252).max()        # 不含當天
    df["new_high"] = (c > df["hh252_prev"]).astype(float)
    df.loc[df["hh252_prev"].isna(), "new_high"] = np.nan
    df["at_high"] = (c >= df["hh252"]).astype(float)
    df.loc[df["hh252"].isna(), "at_high"] = np.nan
    df["atr14"] = compute_atr_correct(df.rename(columns={"high": "High", "low": "Low", "close": "Close"}), 14)
    df["turnover20"] = df["turnover"].rolling(20, min_periods=20).mean()
    df["turnover120"] = df["turnover"].rolling(120, min_periods=120).mean()
    df["ret63"] = c / c.shift(63) - 1
    df["ret126"] = c / c.shift(126) - 1
    df["ret252"] = c / c.shift(252) - 1
    lr = np.log(c / c.shift(1))
    df["vol60"] = lr.rolling(60, min_periods=60).std() * np.sqrt(252)
    df["high_in_20"] = df["at_high"].rolling(20, min_periods=1).max()
    # E_A：今天創新高，且前20天(不含今天)都沒有創新高
    prior_nh = df["new_high"].shift(1).rolling(20, min_periods=20).sum()
    df["entry_a"] = ((df["new_high"] == 1) & (prior_nh == 0)).astype(bool)
    return df


def features_at(ind: pd.DataFrame, date) -> dict:
    """起點(date收盤)當下的特徵，只讀date那一列(全部是往回看的指標)。"""
    if date not in ind.index:
        return {}
    r = ind.loc[date]
    c = r["close"]

    def _above(ma):
        v = r[ma]
        return float(c > v) if np.isfinite(v) else np.nan
    t120 = r["turnover120"]
    return {
        "ret_3m": r["ret63"], "ret_6m": r["ret126"], "ret_12m": r["ret252"],
        "above_ma60": _above("ma60"), "above_ma120": _above("ma120"), "above_ma240": _above("ma240"),
        "dist_52w_high": (c / r["hh252"] - 1) if np.isfinite(r["hh252"]) else np.nan,
        "new_high_20d": r["high_in_20"] if np.isfinite(r["hh252"]) else np.nan,
        "vol_60d": r["vol60"],
        "turnover_20d": r["turnover20"],
        "turnover_trend": (r["turnover20"] / t120) if (np.isfinite(t120) and t120 > 0) else np.nan,
        "price": r["raw_close"],
    }


def window_return(ind: pd.DataFrame, d0, d1):
    """回傳(R, M, 起點還原收盤, 終點還原收盤)；起點或終點那天沒有有效價格 → None。"""
    if d0 not in ind.index or d1 not in ind.index:
        return None
    c = ind["close"]
    p0, p1 = float(c.loc[d0]), float(c.loc[d1])
    if not (p0 > 0 and p1 > 0):
        return None
    w = c.loc[d0:d1]
    return p1 / p0 - 1, float(w.max()) / p0 - 1, p0, p1


def path_stats(ind: pd.DataFrame, d0, d1, rev_feat: pd.DataFrame = None) -> dict:
    """贏家路徑：低點日、起漲日(區間內第一次收盤>前252日最高收盤)、到+100%的天數、
    自區間內滾動高點的最大回檔%、最大回檔/高點當天ATR的倍數、起漲後是否曾收在MA60下方、起漲時營收3月均YoY。"""
    w = ind.loc[d0:d1]
    c = w["close"].to_numpy(dtype=float)
    dates = w.index
    out = {"low_date": dates[int(np.nanargmin(c))] if len(c) else pd.NaT}
    nh = w["new_high"].to_numpy()
    idx_ms = np.flatnonzero(nh == 1)
    ms = dates[idx_ms[0]] if len(idx_ms) else pd.NaT
    out["move_start_date"] = ms
    out["days_to_move_start"] = int(idx_ms[0]) if len(idx_ms) else np.nan
    dbl = np.flatnonzero(c >= 2 * c[0])
    out["days_to_double"] = int(dbl[0]) if len(dbl) else np.nan
    runmax = np.maximum.accumulate(c)
    dd = c / runmax - 1
    out["max_drawdown"] = float(dd.min())
    atr = w["atr14"].to_numpy(dtype=float)
    peak_idx = np.zeros(len(c), dtype=int)
    p = 0
    for i in range(len(c)):
        if c[i] >= c[p]:
            p = i
        peak_idx[i] = p
    atr_at_peak = atr[peak_idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        mult = (runmax - c) / atr_at_peak
    mult = mult[np.isfinite(mult)]
    out["max_pullback_atr"] = float(mult.max()) if len(mult) else np.nan
    if pd.notna(ms):
        after = w.loc[w.index > ms]
        valid = after["ma60"].notna()
        out["below_ma60_after_start"] = float((after.loc[valid, "close"] < after.loc[valid, "ma60"]).any())
        if rev_feat is not None and not rev_feat.empty:
            out["rev_yoy3_at_move_start"] = float(revenue_features_on_dates(rev_feat, [ms])["yoy3"].iloc[0])
        else:
            out["rev_yoy3_at_move_start"] = np.nan
    else:
        out["below_ma60_after_start"] = np.nan
        out["rev_yoy3_at_move_start"] = np.nan
    return out


# ===========================================================================
# 規則模擬
# ===========================================================================
def trade_pnl(entry_price: float, exit_price: float, notional: float = TRADE_NOTIONAL):
    """固定金額買進(可零股)：買進付手續費，賣出付手續費+證交稅。回傳(pnl, ret)。"""
    shares = notional / entry_price
    proceeds = shares * exit_price * (1 - FEE_RATE - TAX_RATE)
    cost = notional * (1 + FEE_RATE)
    pnl = proceeds - cost
    return pnl, pnl / notional


def simulate_exit(arr: dict, e: int, i_end: int, atr: float, exit_kind: str):
    """
    從第e根K棒開盤進場，往後逐日檢查出場，回傳(exit_idx, exit_price, reason)。
    - 停損是盤中停損單：開盤就低於停損 → 開盤價出(跳空)；盤中最低碰到 → 停損價出。進場當天也算。
    - X1/X2/X3 的初始停損 = 進場價−2×ATR(訊號日ATR)；移動停損只上不下，收盤後更新、隔天生效。
    - X3：收盤<MA60 → 隔天開盤出。
    - X4：停損1.5×ATR、停利4×ATR；同一天停損停利都碰到 → 保守當作停損。
    - 撐到區間最後一天 → 當天收盤平倉。
    """
    o, h, l, c = arr["open"], arr["high"], arr["low"], arr["close"]
    atr_s, ma60 = arr["atr14"], arr["ma60"]
    pe = o[e]
    init_stop = pe - (X4_STOP_ATR if exit_kind == "X4" else INITIAL_STOP_ATR) * atr
    stop = init_stop
    target = pe + X4_TARGET_ATR * atr if exit_kind == "X4" else np.inf
    hc = -np.inf
    pending_open_exit = False
    for j in range(e, i_end + 1):
        if pending_open_exit:
            return j, o[j], "跌破MA60"
        stop_reason = "初始停損" if stop <= init_stop + 1e-12 else "移動停損"
        if o[j] <= stop:
            return j, o[j], stop_reason
        if l[j] <= stop:
            return j, stop, stop_reason
        if exit_kind == "X4":
            if o[j] >= target:
                return j, o[j], "停利"
            if h[j] >= target:
                return j, target, "停利"
        if j == i_end:
            return j, c[j], "期末平倉"
        hc = max(hc, c[j])
        if exit_kind == "X1" and np.isfinite(atr_s[j]):
            stop = max(stop, hc - X1_TRAIL_ATR * atr_s[j])
        elif exit_kind == "X2":
            stop = max(stop, hc * (1 - X2_TRAIL_PCT))
        elif exit_kind == "X3" and np.isfinite(ma60[j]) and c[j] < ma60[j]:
            pending_open_exit = True
    return i_end, c[i_end], "期末平倉"


def simulate_rule(arr: dict, signal: np.ndarray, i0: int, i1: int, exit_kind: str) -> list:
    """單一股票、同時最多一個部位：訊號日t收盤成立 → t+1開盤進場；t必須在[i0, i1-1]。"""
    trades = []
    sig_idx = np.flatnonzero(signal[i0:i1]) + i0
    free_from = i0
    for t in sig_idx:
        if t < free_from:
            continue
        e = t + 1
        if e > i1:
            break
        atr = arr["atr14"][t]
        pe = arr["open"][e]
        if not (np.isfinite(atr) and atr > 0 and np.isfinite(pe) and pe > 0):
            continue
        x, px, reason = simulate_exit(arr, e, i1, atr, exit_kind)
        pnl, ret = trade_pnl(pe, px)
        trades.append({"signal_idx": int(t), "entry_idx": int(e), "exit_idx": int(x),
                       "entry_price": float(pe), "exit_price": float(px), "exit_reason": reason,
                       "hold_days": int(x - e), "pnl": float(pnl), "ret": float(ret)})
        free_from = x
    return trades


def rule_signals(ind: pd.DataFrame, rs: pd.Series, rev_on_dates: pd.DataFrame) -> dict:
    a = ind["entry_a"].to_numpy(dtype=bool)
    rs_ok = (rs.reindex(ind.index).to_numpy(dtype=float) >= RS_TOP)
    if rev_on_dates is not None and len(rev_on_dates):
        rev_ok = rev_on_dates["yoy3"].reindex(ind.index).to_numpy(dtype=float) > REV_YOY_MIN
    else:
        rev_ok = np.zeros(len(ind), dtype=bool)
    return {"E_A": a, "E_B": a & rev_ok, "E_C": a & rs_ok, "E_D": a & rev_ok & rs_ok}


# ===========================================================================
# lift
# ===========================================================================
def lift_row(cond: pd.Series, winners: pd.Series) -> dict:
    cond = pd.Series(np.where(pd.isna(cond), False, cond), index=cond.index).astype(bool)
    winners = pd.Series(np.where(pd.isna(winners), False, winners), index=winners.index).astype(bool)
    n = len(cond)
    n_c = int(cond.sum())
    w = int(winners.sum())
    w_c = int((cond & winners).sum())
    base = w / n if n else np.nan
    rate = w_c / n_c if n_c else np.nan
    lift = rate / base if (n_c and base and base > 0) else np.nan
    cov = w_c / w if w else np.nan
    return {"n_cond": n_c, "winners_cond": w_c, "rate_cond": rate, "base_rate": base,
            "lift": lift, "coverage": cov, "prevalence": n_c / n if n else np.nan}


ELEC_INDUSTRIES = ("半導體業", "電腦及週邊設備業", "電子零組件業", "通信網路業", "光電業", "其他電子業")
ASPECTS = ["技術面", "籌碼面", "基本面", "產業"]

# (欄位, 中文名, 面向)：做五分位條件的連續特徵
QUINTILE_FEATURES = [
    ("ret_3m", "3個月報酬", "技術面"), ("ret_6m", "6個月報酬", "技術面"), ("ret_12m", "12個月報酬", "技術面"),
    ("dist_52w_high", "距52週高點", "技術面"), ("vol_60d", "60日波動率", "技術面"),
    ("turnover_20d", "20日均成交金額", "技術面"), ("turnover_trend", "成交金額趨勢(20d/120d)", "技術面"),
    ("price", "股價", "技術面"), ("bias60", "MA60乖離率", "技術面"), ("range60", "60日盤整幅度", "技術面"),
    ("atr_pct", "ATR%", "技術面"), ("beta", "對加權指數beta", "技術面"), ("vol_burst", "量能爆發(5日/60日均量)", "技術面"),
    ("foreign_ratio20", "外資20日買超/成交量", "籌碼面"), ("trust_ratio20", "投信20日買超/成交量", "籌碼面"),
    ("insti_ratio20", "三大法人20日買超/成交量", "籌碼面"), ("margin_chg20", "融資餘額20日變化", "籌碼面"),
    ("foreign_pct", "外資持股比例", "籌碼面"), ("foreign_pct_chg60", "外資持股60日變化", "籌碼面"),
    ("rev_yoy_3m", "營收3月均YoY", "基本面"), ("pe", "本益比", "基本面"), ("pb", "股價淨值比", "基本面"),
    ("dy", "殖利率", "基本面"), ("eps_yoy", "EPS年增率", "基本面"), ("gm", "毛利率", "基本面"),
    ("om", "營益率", "基本面"), ("roe", "ROE", "基本面"),
]
# 同一個東西換個說法(例如RS百分位就是6個月報酬排序)，雙條件組合時視為同一個特徵
_FEATURE_ALIAS = {"rs_rank": "ret_6m"}


def build_conditions(f: pd.DataFrame) -> list:
    """
    事先登錄的條件清單：[(面向, 群組, 特徵, 條件名, bool Series)]。NaN一律視為不符合
    (所以資料來源失敗時，相關條件的符合檔數會是0或很少，在報告裡會被「≥30檔」門檻自然濾掉)。
    """
    conds = []

    def col(c):
        return f[c].astype(float) if c in f.columns else pd.Series(np.nan, index=f.index)

    def add(aspect, group, feature, name, s):
        s = pd.Series(np.where(pd.isna(s), False, s), index=f.index).astype(bool)
        conds.append((aspect, group, _FEATURE_ALIAS.get(feature, feature), name, s))
    gt = lambda c, v: col(c) > v
    ge = lambda c, v: col(c) >= v
    lt = lambda c, v: col(c) < v
    le = lambda c, v: col(c) <= v
    eq = lambda c, v: col(c) == v
    T, C, F, I = ASPECTS
    # ---- 技術面
    add(T, "相對強度", "rs_rank", "RS前20%(6個月報酬百分位≥80)", ge("rs_rank", 0.8))
    add(T, "相對強度", "rs_rank", "RS後20%(6個月報酬百分位<20)", lt("rs_rank", 0.2))
    add(T, "報酬", "ret_3m", "3個月報酬>20%", gt("ret_3m", 0.20))
    add(T, "報酬", "ret_3m", "3個月報酬<0", lt("ret_3m", 0.0))
    add(T, "報酬", "ret_12m", "12個月報酬>50%", gt("ret_12m", 0.50))
    add(T, "均線", "above_ma60", "站上MA60", eq("above_ma60", 1))
    add(T, "均線", "above_ma120", "站上MA120", eq("above_ma120", 1))
    add(T, "均線", "above_ma240", "站上MA240(年線)", eq("above_ma240", 1))
    add(T, "均線", "above_ma240", "跌破MA240(年線)", eq("above_ma240", 0))
    add(T, "均線", "ma_all", "同時站上MA60/120/240", eq("above_ma60", 1) & eq("above_ma120", 1) & eq("above_ma240", 1))
    add(T, "均線", "bull_align", "均線多頭排列(MA5>20>60>120)", eq("bull_align", 1))
    add(T, "均線", "ma60_slope", "季線(MA60)20日斜率>0", gt("ma60_slope", 0))
    add(T, "均線", "ma120_slope", "半年線(MA120)20日斜率>0", gt("ma120_slope", 0))
    add(T, "乖離/盤整", "bias60", "MA60乖離>20%", gt("bias60", 0.20))
    add(T, "乖離/盤整", "bias60", "MA60乖離<−10%", lt("bias60", -0.10))
    add(T, "乖離/盤整", "range60", "60日盤整幅度<20%(整理)", lt("range60", 0.20))
    add(T, "乖離/盤整", "bbw_pct", "布林帶寬位於過去250日最低20%(壓縮)", le("bbw_pct", 20))
    add(T, "乖離/盤整", "bbw_pct", "布林帶寬位於過去250日最高20%(擴張)", ge("bbw_pct", 80))
    add(T, "動能指標", "rsi14", "RSI14>70", gt("rsi14", 70))
    add(T, "動能指標", "rsi14", "RSI14<30", lt("rsi14", 30))
    add(T, "動能指標", "k9", "KD K值>80", gt("k9", 80))
    add(T, "動能指標", "k9", "KD K值<20", lt("k9", 20))
    add(T, "52週高低", "dist_52w_high", "距52週高點10%以內", ge("dist_52w_high", -0.10))
    add(T, "52週高低", "dist_52w_high", "距52週高點超過30%", lt("dist_52w_high", -0.30))
    add(T, "52週高低", "new_high_20d", "近20日內創52週新高", eq("new_high_20d", 1))
    add(T, "52週高低", "dist_52w_low", "距52週低點>50%", gt("dist_52w_low", 0.50))
    add(T, "52週高低", "dist_52w_low", "距52週低點<10%", lt("dist_52w_low", 0.10))
    add(T, "52週高低", "days_since_high", "52週新高後≤5天", le("days_since_high", 5))
    add(T, "52週高低", "days_since_high", "52週新高後≥120天", ge("days_since_high", 120))
    add(T, "風險", "max_dd_1y", "過去一年最大回撤>40%", lt("max_dd_1y", -0.40))
    add(T, "風險", "max_dd_1y", "過去一年最大回撤<20%", gt("max_dd_1y", -0.20))
    add(T, "風險", "beta", "beta>1.2", gt("beta", 1.2))
    add(T, "風險", "beta", "beta<0.8", lt("beta", 0.8))
    add(T, "量能", "turnover_trend", "成交金額趨勢>1.5(量增)", gt("turnover_trend", 1.5))
    add(T, "量能", "turnover_trend", "成交金額趨勢<0.8(量縮)", lt("turnover_trend", 0.8))
    add(T, "量能", "vol_burst", "5日均量>2倍60日均量", gt("vol_burst", 2))
    add(T, "量能", "gap_ups_60", "60日內向上跳空≥2次", ge("gap_ups_60", 2))
    add(T, "股價", "price", "股價<50元", lt("price", 50))
    add(T, "股價", "price", "股價>500元", gt("price", 500))
    # ---- 籌碼面
    add(C, "外資", "foreign_ratio20", "外資20日買超>0", gt("foreign_ratio20", 0))
    add(C, "外資", "foreign_ratio20", "外資20日買超>成交量5%", gt("foreign_ratio20", 5))
    add(C, "外資", "foreign_streak", "外資連續買超≥3天", ge("foreign_streak", 3))
    add(C, "外資", "foreign_streak", "外資連續買超≥5天", ge("foreign_streak", 5))
    add(C, "外資", "foreign_pct", "外資持股>30%", gt("foreign_pct", 30))
    add(C, "外資", "foreign_pct", "外資持股<5%", lt("foreign_pct", 5))
    add(C, "外資", "foreign_pct_chg60", "外資持股60日增加>1個百分點", gt("foreign_pct_chg60", 1))
    add(C, "外資", "foreign_pct_chg60", "外資持股60日減少>1個百分點", lt("foreign_pct_chg60", -1))
    add(C, "投信", "trust_ratio20", "投信20日買超>0", gt("trust_ratio20", 0))
    add(C, "投信", "trust_ratio20", "投信20日買超>成交量2%", gt("trust_ratio20", 2))
    add(C, "投信", "trust_pct_shares20", "投信20日買超>發行股數0.5%", gt("trust_pct_shares20", 0.5))
    add(C, "投信", "trust_streak", "投信連續買超≥3天", ge("trust_streak", 3))
    add(C, "投信", "trust_streak", "投信連續買超≥5天", ge("trust_streak", 5))
    add(C, "三大法人", "insti_ratio20", "三大法人20日買超>0", gt("insti_ratio20", 0))
    add(C, "三大法人", "insti_ratio20", "三大法人20日買超>成交量5%", gt("insti_ratio20", 5))
    add(C, "融資融券", "margin_chg20", "融資餘額20日增加>10%", gt("margin_chg20", 10))
    add(C, "融資融券", "margin_chg20", "融資餘額20日減少>10%", lt("margin_chg20", -10))
    add(C, "融資融券", "short_margin_ratio", "券資比>10%", gt("short_margin_ratio", 10))
    add(C, "融資融券", "short_margin_ratio", "券資比>30%", gt("short_margin_ratio", 30))
    # ---- 基本面
    add(F, "營收", "rev_yoy_latest", "最新月營收YoY>20%", gt("rev_yoy_latest", 20))
    add(F, "營收", "rev_yoy_3m", "營收3月均YoY>20%", gt("rev_yoy_3m", 20))
    add(F, "營收", "rev_yoy_3m", "營收3月均YoY>50%", gt("rev_yoy_3m", 50))
    add(F, "營收", "rev_yoy_3m", "營收3月均YoY<0", lt("rev_yoy_3m", 0))
    add(F, "營收", "rev_yoy_accel", "營收YoY加速(>0)", gt("rev_yoy_accel", 0))
    add(F, "營收", "rev_yoy_accel", "營收YoY加速>10個百分點", gt("rev_yoy_accel", 10))
    add(F, "營收", "rev_consec_20", "連續≥3個月YoY>20%", ge("rev_consec_20", 3))
    add(F, "營收", "rev_high12", "月營收創近12個月新高", eq("rev_high12", 1))
    add(F, "估值", "pe", "本益比<15", lt("pe", 15))
    add(F, "估值", "pe", "本益比15~30", ge("pe", 15) & le("pe", 30))
    add(F, "估值", "pe", "本益比>30", gt("pe", 30))
    add(F, "估值", "pe_blank", "本益比空白(多為虧損)", eq("pe_blank", 1))
    add(F, "估值", "pb", "股價淨值比<1", lt("pb", 1))
    add(F, "估值", "pb", "股價淨值比>3", gt("pb", 3))
    add(F, "估值", "dy", "殖利率>4%", gt("dy", 4))
    add(F, "估值", "dy", "殖利率<1%", lt("dy", 1))
    add(F, "季財報", "eps", "最新季EPS>0", gt("eps", 0))
    add(F, "季財報", "eps_yoy", "EPS年增率>20%", gt("eps_yoy", 20))
    add(F, "季財報", "eps_yoy", "EPS年增率>50%", gt("eps_yoy", 50))
    add(F, "季財報", "eps_yoy", "EPS年增率<0", lt("eps_yoy", 0))
    add(F, "季財報", "gm_yoy", "毛利率年增>2個百分點", gt("gm_yoy", 2))
    add(F, "季財報", "gm_yoy", "毛利率年減>2個百分點", lt("gm_yoy", -2))
    add(F, "季財報", "om_yoy", "營益率年增>2個百分點", gt("om_yoy", 2))
    add(F, "季財報", "om", "營益率>10%", gt("om", 10))
    add(F, "季財報", "eps4q", "近四季EPS合計>5元", gt("eps4q", 5))
    add(F, "季財報", "roe", "ROE(近四季)>15%", gt("roe", 15))
    add(F, "季財報", "roe", "ROE(近四季)<5%", lt("roe", 5))
    # ---- 產業
    add(I, "市場", "market", "上市", f["market"] == "上市")
    add(I, "市場", "market", "上櫃", f["market"] == "上櫃")
    add(I, "族群", "elec", "電子相關(半導體/電腦週邊/電子零組件/通信網路/光電/其他電子)", f["industry"].isin(ELEC_INDUSTRIES))
    counts = f["industry"].value_counts()
    for ind_name in counts[counts >= 30].index:
        add(I, "產業別", "industry", f"產業：{ind_name}", f["industry"] == ind_name)
    # ---- 五分位
    for c, label, aspect in QUINTILE_FEATURES:
        s = col(c)
        valid = s.dropna()
        if valid.nunique() < 5:
            continue
        try:
            q = pd.qcut(valid, 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        nq = int(q.max()) + 1
        for k in range(nq):
            tag = "最低" if k == 0 else ("最高" if k == nq - 1 else "")
            m = pd.Series(False, index=f.index)
            m.loc[q.index[q == k]] = True
            add(aspect, f"五分位:{label}", c, f"{label} Q{k + 1}{tag}", m)
    return conds


def analyze_lift(f: pd.DataFrame, n_min: int = 30, n_perm: int = 200, seed: int = 42) -> dict:
    """單一條件lift(三種贏家定義)、雙條件lift(漲1倍)、洗牌基準。"""
    conds = build_conditions(f)
    rows = []
    for tier, tier_name in TIER_DEFS:
        for aspect, group, feat, name, s in conds:
            rows.append({"aspect": aspect, "group": group, "feature": feat, "condition": name, "tier": tier_name,
                         **lift_row(s, f[tier])})
    lift_df = pd.DataFrame(rows)
    C = np.column_stack([s.to_numpy(bool) for *_, s in conds]) if conds else np.zeros((len(f), 0), bool)
    feats = [c[2] for c in conds]
    w = f["w100"].to_numpy(dtype=float)
    chosen = wx.select_top_singles(C, w, feats, n_min=n_min)
    pairs = []
    for i, j, n, wc, rate, lift, cov in wx.pair_lifts(C, w, chosen, n_min=n_min):
        pairs.append({"cond_a": conds[i][3], "aspect_a": conds[i][0], "cond_b": conds[j][3], "aspect_b": conds[j][0],
                      "n_cond": n, "winners_cond": wc, "rate_cond": rate, "base_rate": w.mean(), "lift": lift,
                      "coverage": cov})
    pairs_df = pd.DataFrame(pairs, columns=["cond_a", "aspect_a", "cond_b", "aspect_b", "n_cond", "winners_cond",
                                            "rate_cond", "base_rate", "lift", "coverage"])
    pairs_df = pairs_df.sort_values("lift", ascending=False).reset_index(drop=True)
    null = wx.shuffle_baseline(C, w, feats, n_min=n_min, n_perm=n_perm, seed=seed)
    n_single_eligible = int((C.sum(axis=0) >= n_min).sum()) if C.shape[1] else 0
    return {"lift": lift_df, "pairs": pairs_df, "null": null, "n_conditions": len(conds),
            "n_single_eligible": n_single_eligible, "n_pairs_tested": len(chosen) * (len(chosen) - 1) // 2,
            "chosen": [conds[i][3] for i in chosen]}


# ===========================================================================
# 主研究流程(純計算，方便用合成資料測試)
# ===========================================================================
def _taipei_today() -> datetime.date:
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)).date()


def resolve_window(start: str, end: str, today: datetime.date = None):
    today = today or _taipei_today()
    end_ts = pd.Timestamp(end) if end else pd.Timestamp(today)
    start_ts = pd.Timestamp(start) if start else end_ts - pd.DateOffset(years=2)
    if start_ts >= end_ts:
        raise ValueError(f"start({start_ts.date()})必須早於end({end_ts.date()})")
    return start_ts.normalize(), end_ts.normalize()


def build_calendar(inds: dict) -> pd.DatetimeIndex:
    counts = {}
    for ind in inds.values():
        for d in ind.index:
            counts[d] = counts.get(d, 0) + 1
    if not counts:
        return pd.DatetimeIndex([])
    mx = max(counts.values())
    return pd.DatetimeIndex(sorted(d for d, n in counts.items() if n >= 0.3 * mx))


MS_COLS = ["ms_ret_3m", "ms_bias60", "ms_range60", "ms_atr_pct", "ms_vol_burst", "ms_bull_align", "ms_rsi14",
           "ms_bbw_pct", "ms_eps_yoy", "ms_quarter"]


def move_start_snapshot(ind: pd.DataFrame, ms, q_feat: pd.DataFrame = None, mkt_close=None) -> dict:
    """起漲日當天(收盤)看得到的價格特徵 + 最新已公告季EPS年增率(point-in-time)。"""
    out = {k: np.nan for k in MS_COLS}
    out["ms_quarter"] = ""
    if ms is None or pd.isna(ms) or ms not in ind.index:
        return out
    t = wx.tech_extra_at(ind, ms, mkt_close)
    out.update({"ms_ret_3m": ind.loc[ms, "ret63"], "ms_bias60": t["bias60"], "ms_range60": t["range60"],
                "ms_atr_pct": t["atr_pct"], "ms_vol_burst": t["vol_burst"], "ms_bull_align": t["bull_align"],
                "ms_rsi14": t["rsi14"], "ms_bbw_pct": t["bbw_pct"]})
    qv = wx.quarterly_on_dates(q_feat, [ms]).iloc[0]
    out["ms_eps_yoy"] = qv["eps_yoy"]
    out["ms_quarter"] = qv["qlabel"]
    return out


def run_study(universe: pd.DataFrame, raw_prices: dict, revenue_long: pd.DataFrame,
              taiex: pd.Series, start, end, min_turnover: float = MIN_TURNOVER, extra: dict = None,
              n_perm: int = 200) -> dict:
    """extra：winners_extra.load_all_extra() 的結果(籌碼/估值/季報)；None=都沒有(相關特徵全部空值)。"""
    meta = {"universe": int(len(universe)), "price_ok": 0, "no_price": 0}
    uni = universe.set_index("code")
    inds = {}
    for code in uni.index:
        px = prepare_price_frame(raw_prices.get(code))
        if len(px) < 30:
            meta["no_price"] += 1
            continue
        inds[code] = compute_indicators(px)
    meta["price_ok"] = len(inds)
    cal = build_calendar(inds)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    in_win = cal[(cal >= start) & (cal <= end)]
    if len(in_win) < 2:
        raise ValueError("區間內交易日不足(資料可能沒抓到)")
    d0, d1 = in_win[0], in_win[-1]
    meta.update({"d0": d0, "d1": d1, "trading_days": int(len(in_win))})

    rev_feats = build_revenue_features(revenue_long)
    meta["revenue_codes"] = len(rev_feats)
    extra = extra or {}
    chip = wx.prepare_chip(extra.get("chip_daily"))
    loaded = extra.get("loaded") or {}
    chip_dates = wx.needed_dates(cal, d0)
    q_raw = extra.get("quarterly_raw")
    q_cum = wx.detect_cumulative(q_raw) if q_raw is not None and len(q_raw) else {}
    q_feats = wx.build_quarterly_features(q_raw, q_cum) if q_cum else {}
    meta["quarterly_cumulative"] = q_cum
    meta["quarterly_codes"] = len(q_feats)
    mkt_close = taiex.dropna() if (taiex is not None and len(taiex)) else None

    rows = []
    excl = {"起點或終點沒有價格": 0, "流動性不足": 0}
    for code, ind in inds.items():
        wr = window_return(ind, d0, d1)
        if wr is None:
            excl["起點或終點沒有價格"] += 1
            continue
        R, M, p0, p1 = wr
        feat = features_at(ind, d0)
        t20 = feat["turnover_20d"]
        liquid = bool(np.isfinite(t20) and t20 >= min_turnover)
        if not liquid:
            excl["流動性不足"] += 1
        rv = revenue_features_on_dates(rev_feats.get(code), [d0]).iloc[0]
        info = uni.loc[code]
        tech = wx.tech_extra_at(ind, d0, mkt_close)
        chipf = wx.chip_features(code, info["market"], ind, chip, loaded, chip_dates)
        qv = wx.quarterly_on_dates(q_feats.get(code), [d0]).iloc[0]
        qd = {k: qv[k] for k in wx.Q_FEAT_COLS}
        rows.append({"code": code, "name": info["name"], "market": info["market"], "industry": info["industry"],
                     "R": R, "M": M, "start_price": p0, "end_price": p1,
                     "start_raw_close": float(ind.loc[d0, "raw_close"]), "included": liquid, **feat,
                     "rev_yoy_latest": rv["yoy"], "rev_yoy_3m": rv["yoy3"], "rev_yoy_accel": rv["accel"],
                     "rev_consec_20": rv["consec20"], "rev_high12": rv["high12"], "rev_month_at_start": rv["rev_month"],
                     **tech, **chipf, **qd, "quarter_at_start": qv["qlabel"],
                     "elec": float(info["industry"] in ELEC_INDUSTRIES)})
    meta["excluded"] = excl
    allf = pd.DataFrame(rows)
    if allf.empty:
        raise ValueError("沒有任何股票在起點與終點都有價格")
    f = allf[allf["included"]].copy().reset_index(drop=True)
    if f.empty:
        raise ValueError("流動性篩選後沒有股票")
    f["rs_rank"] = f["ret_6m"].rank(pct=True)
    f["w100"] = f["R"] >= 1.0
    f["w200"] = f["R"] >= 2.0
    f["m100"] = f["M"] >= 1.0
    meta["included"] = int(len(f))
    meta["revenue_coverage_at_start"] = float(f["rev_yoy_3m"].notna().mean())
    meta["coverage"] = {k: float(f[c].notna().mean()) for k, c in [
        ("三大法人", "foreign_ratio20"), ("融資融券", "short_margin_ratio"), ("外資持股", "foreign_pct"),
        ("估值", "pb"), ("季財報", "eps"), ("ROE", "roe"), ("月營收", "rev_yoy_3m"), ("beta", "beta")]}

    # ---- 分析1：lift(單一條件、雙條件、洗牌基準)
    la = analyze_lift(f, n_perm=n_perm)
    lift_df = la["lift"]
    ind_rows = []
    for industry, g in f.groupby("industry"):
        r = {"industry": industry, "count": int(len(g))}
        for tier, tier_name in TIER_DEFS:
            lr = lift_row(f["industry"] == industry, f[tier])
            r[f"{tier}_winners"] = lr["winners_cond"]
            r[f"{tier}_rate"] = lr["rate_cond"]
            r[f"{tier}_lift"] = lr["lift"]
        ind_rows.append(r)
    industry_df = pd.DataFrame(ind_rows).sort_values(["w100_winners", "count"], ascending=[False, False])

    # ---- 分析2：贏家路徑(漲1倍以上 或 曾經漲1倍)
    wl = f[f["w100"] | f["m100"]].copy()
    path_rows = []
    for code in wl["code"]:
        ps = path_stats(inds[code], d0, d1, rev_feats.get(code))
        ps.update(move_start_snapshot(inds[code], ps.get("move_start_date"), q_feats.get(code), mkt_close))
        path_rows.append({"code": code, **ps})
    path_df = pd.DataFrame(path_rows)
    if not path_df.empty:
        wl = wl.merge(path_df, on="code", how="left")
    wl = wl.sort_values("R", ascending=False).reset_index(drop=True)

    # ---- 分析3：規則模擬(全部納入的股票)
    codes = list(f["code"])
    ret6 = pd.DataFrame({c: inds[c]["ret126"].reindex(cal) for c in codes})
    rs_panel = ret6.rank(axis=1, pct=True)
    winners_set = set(f.loc[f["w100"], "code"])
    trades = []
    for code in codes:
        ind = inds[code]
        idx = ind.index
        i0 = int(idx.searchsorted(d0))
        i1 = int(idx.searchsorted(d1, side="right")) - 1
        if i1 <= i0:
            continue
        rev_d = revenue_features_on_dates(rev_feats.get(code), idx) if code in rev_feats else None
        sigs = rule_signals(ind, rs_panel[code], rev_d)
        arr = {k: ind[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "atr14", "ma60")}
        for ek, sig in sigs.items():
            if not sig[i0:i1].any():
                continue
            for xk in EXIT_RULES:
                for t in simulate_rule(arr, sig, i0, i1, xk):
                    t.update({"rule": f"{ek}+{xk}", "entry_rule": ek, "exit_rule": xk, "code": code,
                              "signal_date": idx[t["signal_idx"]], "entry_date": idx[t["entry_idx"]],
                              "exit_date": idx[t["exit_idx"]], "is_winner": code in winners_set})
                    trades.append(t)
    tcols = ["rule", "entry_rule", "exit_rule", "code", "signal_date", "entry_date", "entry_price",
             "exit_date", "exit_price", "exit_reason", "hold_days", "ret", "pnl", "is_winner"]
    trades_df = pd.DataFrame(trades, columns=tcols + ["signal_idx", "entry_idx", "exit_idx"])[tcols]
    rules_df = summarize_rules(trades_df, f)

    # 參考：全部納入股票等權重買進持有、加權指數
    bh_net = [trade_pnl(1.0, 1.0 + r)[1] for r in f["R"]]
    taiex_ret = np.nan
    if taiex is not None and len(taiex):
        tw = taiex.dropna()
        tw = tw[(tw.index >= d0) & (tw.index <= d1)]
        if len(tw) >= 2:
            taiex_ret = float(tw.iloc[-1] / tw.iloc[0] - 1)
    meta.update({"bh_mean_R": float(f["R"].mean()), "bh_median_R": float(f["R"].median()),
                 "bh_mean_net": float(np.mean(bh_net)), "bh_total_pnl": float(np.sum(bh_net) * TRADE_NOTIONAL),
                 "taiex_ret": taiex_ret,
                 "n_w100": int(f["w100"].sum()), "n_w200": int(f["w200"].sum()), "n_m100": int(f["m100"].sum()),
                 "n_all_with_price": int(len(allf)),
                 "n_w100_excluded_illiquid": int(((allf["R"] >= 1.0) & ~allf["included"]).sum())})
    return {"meta": meta, "features": f, "all_features": allf, "lift": lift_df, "industry": industry_df,
            "pairs": la["pairs"], "null": la["null"], "lift_meta": {k: la[k] for k in
                                                                     ("n_conditions", "n_single_eligible", "n_pairs_tested", "chosen")},
            "winners": wl, "trades": trades_df, "rules": rules_df}


def summarize_rules(trades_df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    winners = f.loc[f["w100"], ["code", "R"]]
    n_w = len(winners)
    rows = []
    for ek in ENTRY_RULES:
        for xk in EXIT_RULES:
            rule = f"{ek}+{xk}"
            t = trades_df[trades_df["rule"] == rule]
            n = len(t)
            gp = float(t.loc[t["pnl"] > 0, "pnl"].sum())
            gl = float(-t.loc[t["pnl"] < 0, "pnl"].sum())
            top5 = float(t["pnl"].nlargest(5).sum()) if n else 0.0
            tw = t[t["is_winner"]]
            per_w = tw.groupby("code")["ret"].apply(lambda r: float(np.prod(1 + r.to_numpy()) - 1)) if len(tw) else pd.Series(dtype=float)
            total = float(t["pnl"].sum())
            rows.append({
                "rule": rule, "entry": ENTRY_RULES[ek], "exit": EXIT_RULES[xk], "trades": n,
                "win_rate": float((t["pnl"] > 0).mean()) if n else np.nan,
                "pf": gp / gl if gl > 0 else (np.inf if gp > 0 else np.nan),
                "total_pnl": total, "avg_pnl": total / n if n else np.nan,
                "pnl_ex_top5": total - top5,
                "avg_hold_days": float(t["hold_days"].mean()) if n else np.nan,
                "winners_traded_pct": (tw["code"].nunique() / n_w) if n_w else np.nan,
                "winner_median_realized": float(per_w.median()) if len(per_w) else np.nan,
                "winner_median_R": float(winners["R"].median()) if n_w else np.nan,
                # 總損益<=0時「來自贏家的比例」沒有意義(會變成負數或除以0)，留空
                "winner_pnl_share": (float(tw["pnl"].sum()) / total) if total > 0 else np.nan,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# 輸出
# ===========================================================================
FEATURE_LABELS = {
    "code": "代號", "name": "名稱", "market": "市場", "industry": "產業別",
    "R": "區間報酬R", "M": "區間最大漲幅M", "start_price": "起點還原收盤", "end_price": "終點還原收盤",
    "start_raw_close": "起點原始收盤", "included": "納入主分析",
    "ret_3m": "3個月報酬", "ret_6m": "6個月報酬", "ret_12m": "12個月報酬", "rs_rank": "RS百分位(6個月報酬)",
    "above_ma60": "站上MA60", "above_ma120": "站上MA120", "above_ma240": "站上MA240",
    "dist_52w_high": "距52週高點", "new_high_20d": "近20日創52週新高", "vol_60d": "60日年化波動率",
    "turnover_20d": "20日均成交金額", "turnover_trend": "成交金額趨勢(20d/120d)", "price": "起點股價",
    "rev_yoy_latest": "最新月營收YoY(%)", "rev_yoy_3m": "營收3月均YoY(%)", "rev_yoy_accel": "營收YoY加速(百分點)",
    "rev_consec_20": "連續YoY>20%月數", "rev_month_at_start": "起點已知最新營收月份",
    "w100": "漲1倍以上", "w200": "漲2倍以上", "m100": "曾經漲1倍",
    "low_date": "區間低點日", "move_start_date": "起漲日(首次收盤>前252日高)", "days_to_move_start": "起點到起漲交易日數",
    "days_to_double": "起點到+100%交易日數", "max_drawdown": "區間最大回檔(自滾動高點)",
    "max_pullback_atr": "最大回檔ATR倍數", "below_ma60_after_start": "起漲後曾收在MA60下",
    "rev_yoy3_at_move_start": "起漲時營收3月均YoY(%)",
    "rev_high12": "月營收創近12個月新高", "elec": "電子相關產業",
    # 進階技術面
    "bull_align": "均線多頭排列(MA5>20>60>120)", "ma60_slope": "MA60_20日斜率", "ma120_slope": "MA120_20日斜率",
    "bias60": "MA60乖離率", "range60": "60日盤整幅度", "bbw_pct": "布林帶寬250日百分位", "atr_pct": "ATR%",
    "rsi14": "RSI14", "k9": "KD_K值(9,3,3)", "dist_52w_low": "距52週低點", "days_since_high": "52週新高後天數",
    "max_dd_1y": "過去一年最大回撤", "beta": "對加權指數beta(250日)", "vol_burst": "量能爆發(5日/60日均量)",
    "gap_ups_60": "60日內向上跳空次數",
    # 籌碼面
    "foreign_ratio20": "外資20日買超/成交量(%)", "trust_ratio20": "投信20日買超/成交量(%)",
    "trust_pct_shares20": "投信20日買超/發行股數(%)", "trust_streak": "投信連續買超天數(最多20)",
    "foreign_streak": "外資連續買超天數(最多20)", "insti_ratio20": "三大法人20日買超/成交量(%)",
    "margin_chg20": "融資餘額20日變化(%)", "short_margin_ratio": "券資比(%)", "foreign_pct": "外資持股比例(%)",
    "foreign_pct_chg60": "外資持股60日變化(百分點)",
    # 基本面
    "pe": "本益比", "pb": "股價淨值比", "dy": "殖利率(%)", "pe_blank": "本益比空白(多為虧損)",
    "eps": "最新季EPS(單季)", "eps_yoy": "EPS年增率(%)", "gm": "毛利率(%)", "gm_yoy": "毛利率年增(百分點)",
    "om": "營益率(%)", "om_yoy": "營益率年增(百分點)", "eps4q": "近四季EPS合計", "roe": "ROE近四季(%)",
    "quarter_at_start": "起點已知最新季",
    # 起漲日快照
    "ms_ret_3m": "起漲時3個月報酬", "ms_bias60": "起漲時MA60乖離", "ms_range60": "起漲時60日盤整幅度",
    "ms_atr_pct": "起漲時ATR%", "ms_vol_burst": "起漲時量能爆發", "ms_bull_align": "起漲時均線多頭排列",
    "ms_rsi14": "起漲時RSI14", "ms_bbw_pct": "起漲時布林帶寬百分位", "ms_eps_yoy": "起漲時EPS年增率(%)",
    "ms_quarter": "起漲時已知最新季",
}
LIFT_LABELS = {"aspect": "面向", "feature": "特徵", "group": "特徵群組", "condition": "條件", "tier": "贏家定義", "n_cond": "符合檔數",
               "winners_cond": "符合中贏家數", "rate_cond": "符合者贏家率", "base_rate": "整體贏家率",
               "lift": "lift(倍)", "coverage": "贏家覆蓋率", "prevalence": "全體符合比例"}
RULE_LABELS = {"rule": "規則", "entry": "進場", "exit": "出場", "trades": "交易筆數", "win_rate": "勝率",
               "pf": "獲利因子PF", "total_pnl": "總損益", "avg_pnl": "平均損益", "pnl_ex_top5": "扣除前5大獲利後損益",
               "avg_hold_days": "平均持有交易日", "winners_traded_pct": "漲1倍贏家有交易比例",
               "winner_median_realized": "贏家個股實現報酬中位數", "winner_median_R": "贏家區間報酬中位數",
               "winner_pnl_share": "總損益來自贏家比例"}
PAIR_LABELS = {"cond_a": "條件A", "aspect_a": "A面向", "cond_b": "條件B", "aspect_b": "B面向", "n_cond": "同時符合檔數",
               "winners_cond": "符合中贏家數", "rate_cond": "符合者贏家率", "base_rate": "整體贏家率",
               "lift": "lift(倍)", "coverage": "贏家覆蓋率"}
TRADE_LABELS = {"rule": "規則", "entry_rule": "進場規則", "exit_rule": "出場規則", "code": "代號",
                "signal_date": "訊號日", "entry_date": "進場日", "entry_price": "進場價(還原)",
                "exit_date": "出場日", "exit_price": "出場價(還原)", "exit_reason": "出場原因",
                "hold_days": "持有交易日", "ret": "報酬率(含成本)", "pnl": "損益", "is_winner": "是否漲1倍贏家"}
INDUSTRY_LABELS = {"industry": "產業別", "count": "檔數",
                   "w100_winners": "漲1倍檔數", "w100_rate": "漲1倍比例", "w100_lift": "漲1倍lift",
                   "w200_winners": "漲2倍檔數", "w200_rate": "漲2倍比例", "w200_lift": "漲2倍lift",
                   "m100_winners": "曾漲1倍檔數", "m100_rate": "曾漲1倍比例", "m100_lift": "曾漲1倍lift"}


def _pct(v, d=1):
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:.{d}f}%"


def _f(v, d=2):
    if v is None:
        return "—"
    try:
        if np.isinf(v):
            return "∞"
        return "—" if not np.isfinite(v) else f"{v:.{d}f}"
    except TypeError:
        return str(v)


def _money(v):
    return "—" if v is None or not np.isfinite(v) else f"{v:,.0f}"


def build_summary(res: dict, ctx: dict) -> str:
    meta, f = res["meta"], res["features"]
    L = []
    L.append("最近兩年大漲股的共同點 — 描述性研究(不是選股建議)")
    L.append("=" * 70)
    L.append(f"研究區間：{meta['d0'].date()} ~ {meta['d1'].date()}(要求 {ctx.get('start')} ~ {ctx.get('end')}，"
             f"{meta['trading_days']} 個交易日)")
    L.append("")
    L.append("【資料來源狀態】(每個來源都是可有可無；失敗的來源，相關特徵一律空值、條件不會被列出)")
    if not ctx.get("revenue_ok", False):
        L.append("  ⚠️⚠️ 月營收(MOPS)：完全失敗")
    xs = ctx.get("extra_status")
    if xs is None:
        L.append("  ⚠️⚠️ 籌碼面/估值/季財報：本次沒有抓取(相關特徵全部空值)")
    else:
        for name, ok, req in xs:
            mark = "⚠️⚠️ 完全失敗" if ok == 0 else ("⚠️ 部分失敗" if ok < req else "OK")
            L.append(f"  {mark}  {name}：成功 {ok}/{req}(日期或季×市場)")
        for e in ctx.get("extra_errors", []):
            L.append(f"  ⚠️⚠️ {e}")
        if ctx.get("extra_requests"):
            est, act = ctx["extra_requests"]
            L.append(f"  ・籌碼/估值/季報請求數：預估至少 {est} 次，實際 {act} 次(含重試與TPEx/MOPS備援網址；快取不算)")
    cov = meta.get("coverage", {})
    if cov:
        L.append("  ・納入股票中有資料的比例：" + "、".join(f"{k} {_pct(v, 0)}" for k, v in cov.items()))
    L.append("  ・集保大戶持股：沒有納入——歷史資料只能逐股逐週查詢(上千檔×多週，請求量遠超預算)，這次不抓。")
    qc = meta.get("quarterly_cumulative") or {}
    if qc:
        n_cum = sum(1 for v in qc.values() if v)
        L.append(f"  ・季財報：依同公司Q2/Q1營收比判斷格式，{n_cum}/{len(qc)} 個(年,市場)是「年初累計」→ 單季=本季累計−上季累計；"
                 "其餘視為單季，Q4(年報)=全年−Q1~Q3。EPS用相減得到是近似值。")
    L.append("  ・季財報point-in-time：Q1→5/15、Q2→8/14、Q3→11/14、Q4→隔年3/31 之後才算看得到(金融業期限不同，未特別處理)。")
    L.append("")
    L.append("【資料與偏誤說明】")
    if not ctx.get("revenue_ok", False):
        L.append("  ⚠️⚠️ 月營收(MOPS)完全抓不到或解析不出來：本次所有營收特徵都是空的、E_B/E_D規則不會有交易。"
                 "請看 debug/revenue_parse_report.csv 與 debug/mops_sample_*.html。")
    L.append(f"  ・股票清單：TWSE ISIN 頁面的上市+上櫃普通股 {meta['universe']} 檔(排除ETF/權證/特別股/TDR)。")
    L.append("  ・⚠️ 存活者偏誤：清單只有「目前仍上市櫃」的股票，區間內下市/被併購/轉上市櫃的股票不在裡面，"
             "整體贏家率與規則績效都會偏樂觀。")
    for w in ctx.get("warnings", []):
        L.append(f"  ⚠️ {w}")
    L.append(f"  ・有股價資料 {meta['price_ok']} 檔、沒有/太少 {meta['no_price']} 檔；"
             f"起點或終點沒有價格 {meta['excluded']['起點或終點沒有價格']} 檔(排除)。")
    L.append(f"  ・流動性門檻：起點20日平均成交金額 ≥ NT${MIN_TURNOVER / 1e6:.0f}百萬；"
             f"排除 {meta['excluded']['流動性不足']} 檔，主分析納入 {meta['included']} 檔。"
             f"(被排除的低流動性股票中有 {meta['n_w100_excluded_illiquid']} 檔漲1倍以上)")
    L.append(f"  ・價格：yfinance 還原權值價(Adj Close)算報酬；成交金額用原始收盤×成交股數。")
    L.append(f"  ・月營收：M月營收視為M+1月11日起才看得到(point-in-time)；起點有營收3月均YoY的股票比例 "
             f"{_pct(meta['revenue_coverage_at_start'])}。")
    L.append("  ・⚠️ 樣本內：研究區間就是被研究的這段行情(約當AI多頭)，這裡看起來有效的條件/規則，"
             "一定要用 2018–2023 等其他區間重跑(workflow的window選項)再判斷，不能直接拿來用。")
    L.append("")
    L.append("【贏家名單概況】")
    n = meta["included"]
    L.append(f"  納入 {n} 檔：漲1倍以上 {meta['n_w100']} 檔({_pct(meta['n_w100'] / n)})、"
             f"漲2倍以上 {meta['n_w200']} 檔({_pct(meta['n_w200'] / n)})、曾經漲1倍 {meta['n_m100']} 檔({_pct(meta['n_m100'] / n)})")
    L.append(f"  全部納入股票 R 中位數 {_pct(meta['bh_median_R'])}、平均 {_pct(meta['bh_mean_R'])}；"
             f"加權指數(^TWII)區間報酬 {_pct(meta['taiex_ret'])}")
    wl = res["winners"]
    top = wl[wl["w100"]].head(15) if len(wl) else wl
    if len(top):
        L.append("  漲幅前15名(只是描述，不是推薦)：")
        for _, r in top.iterrows():
            L.append(f"    {r['code']} {r['name']:<8} {r['market']} {str(r['industry'])[:8]:<8} R={_pct(r['R'], 0)} M={_pct(r['M'], 0)}")
    ind_counts = f[f["w100"]].groupby("industry").size().sort_values(ascending=False).head(8)
    if len(ind_counts):
        L.append("  漲1倍贏家最多的產業：" + "、".join(f"{k}({v})" for k, v in ind_counts.items()))
    L.append("")
    L.append("【特徵lift表】(起點當下看得到的特徵；lift=符合者贏家率/整體贏家率；只列符合≥30檔的條件)")
    L.append("  ★ 重點：贏家常見的特徵，只有在「符合者贏家率明顯高於整體贏家率(lift明顯>1)」時才有用；")
    L.append("    覆蓋率高但lift≈1的特徵，代表輸家也一樣常有，不能拿來選股。")
    lift = res["lift"]
    null = res.get("null") or {}
    p95 = null.get("single_p95", np.nan)
    pair95 = null.get("pair_p95", np.nan)
    lm = res.get("lift_meta") or {}
    t = lift[(lift["tier"] == "漲1倍以上") & (lift["n_cond"] >= 30)]
    base = t["base_rate"].iloc[0] if len(t) else np.nan
    L.append(f"  漲1倍以上：整體贏家率 {_pct(base)}；洗牌基準(單一條件最大lift的95百分位) = {_f(p95)}，"
             f"lift超過它的條件標 ◎(比純運氣好)")
    hdr = f"     {'條件':<30}{'符合檔數':>7}{'贏家率':>8}{'lift':>7}{'覆蓋率':>8}"
    for aspect in ASPECTS:
        ta = t[t["aspect"] == aspect].sort_values("lift", ascending=False)
        L.append(f"  ── {aspect}(共 {len(ta)} 個條件符合≥30檔，列lift前12)")
        if ta.empty:
            L.append("     (沒有條件符合≥30檔；如果這個面向的資料來源失敗，這裡會是空的)")
            continue
        L.append(hdr)
        for _, r in ta.head(12).iterrows():
            mark = "◎" if (np.isfinite(p95) and np.isfinite(r["lift"]) and r["lift"] > p95) else " "
            L.append(f"   {mark} {r['condition'][:28]:<30}{r['n_cond']:>7}{_pct(r['rate_cond']):>8}{_f(r['lift']):>7}{_pct(r['coverage']):>8}")
    low = t[(t["coverage"] >= 0.5) & (t["lift"] < 1.3)].sort_values("coverage", ascending=False).head(8)
    L.append("  贏家很常有、但lift偏低(<1.3)的特徵(輸家也很常有 → 單獨使用沒什麼篩選力)：")
    if len(low):
        for _, r in low.iterrows():
            L.append(f"     [{r['aspect']}] {r['condition'][:28]:<30} 覆蓋率{_pct(r['coverage'])} 但lift只有{_f(r['lift'])}")
    else:
        L.append("     (無)")
    for tier, tier_name in TIER_DEFS[1:]:
        tt = lift[(lift["tier"] == tier_name) & (lift["n_cond"] >= 30) & (lift["winners_cond"] >= 3)]
        b2 = tt["base_rate"].iloc[0] if len(tt) else np.nan
        L.append(f"  {tier_name}(整體贏家率 {_pct(b2)})：lift前8(不分面向；未做洗牌基準)")
        for _, r in tt.sort_values("lift", ascending=False).head(8).iterrows():
            L.append(f"     [{r['aspect']}] {r['condition'][:28]:<30}{r['n_cond']:>7}{_pct(r['rate_cond']):>8}{_f(r['lift']):>7}{_pct(r['coverage']):>8}")
    L.append("")
    L.append("【雙條件組合】(漲1倍以上；從lift前12、覆蓋率≥10%、不同特徵的單一條件中兩兩組合，同時符合≥30檔)")
    L.append(f"  候選條件：{'、'.join(lm.get('chosen', [])) or '(無)'}")
    pairs = res.get("pairs")
    if pairs is not None and len(pairs):
        L.append(f"  洗牌基準(重挑前12後，雙條件最大lift的95百分位) = {_f(pair95)}，超過的標 ◎")
        L.append(f"     {'條件A':<24}{'條件B':<24}{'檔數':>6}{'贏家率':>8}{'lift':>7}{'覆蓋率':>8}")
        for _, r in pairs.head(15).iterrows():
            mark = "◎" if (np.isfinite(pair95) and r["lift"] > pair95) else " "
            L.append(f"   {mark} {r['cond_a'][:22]:<24}{r['cond_b'][:22]:<24}{r['n_cond']:>6}{_pct(r['rate_cond']):>8}"
                     f"{_f(r['lift']):>7}{_pct(r['coverage']):>8}")
    else:
        L.append("  (沒有同時符合≥30檔的組合)")
    L.append("")
    L.append("【多重比較警告】")
    L.append(f"  這次一共檢驗了 {lm.get('n_conditions', 0)} 個單一條件 × 3 種贏家定義，加上 {lm.get('n_pairs_tested', 0)} 個雙條件組合；")
    L.append("  條件一多，光靠運氣就會有幾個lift看起來很高。洗牌基準做法：把「漲1倍」標籤隨機打亂 "
             f"{null.get('n_perm', 0)} 次(seed={null.get('seed', '')})，每次記錄最大lift，取95百分位。")
    L.append(f"  → 單一條件：運氣能做到的最大lift約 {_f(p95)}；雙條件：約 {_f(pair95)}。低於這個水準的lift，很可能只是雜訊。")
    L.append("")
    L.append("【產業表】(依漲1倍檔數排序，前15)")
    L.append(f"   {'產業別':<12}{'檔數':>6}{'漲1倍':>7}{'比例':>8}{'lift':>7}")
    for _, r in res["industry"].head(15).iterrows():
        L.append(f"   {str(r['industry'])[:10]:<12}{r['count']:>6}{r['w100_winners']:>7}{_pct(r['w100_rate']):>8}{_f(r['w100_lift']):>7}")
    L.append("   (產業檔數少時，比例與lift很不穩定，看的時候請一起看檔數)")
    L.append("")
    L.append("【贏家路徑統計】(漲1倍以上的股票；中位數 [25%, 75%])")
    w = wl[wl["w100"]] if len(wl) else wl
    if len(w):
        def q(col, fmt):
            s = pd.to_numeric(w[col], errors="coerce").dropna()
            if s.empty:
                return "—"
            return f"{fmt(s.median())} [{fmt(s.quantile(0.25))}, {fmt(s.quantile(0.75))}]"
        L.append(f"   起點到起漲(首次收盤>前252日高)交易日數：{q('days_to_move_start', lambda v: f'{v:.0f}')}")
        L.append(f"   起點到+100%交易日數：{q('days_to_double', lambda v: f'{v:.0f}')}")
        L.append(f"   區間最大回檔(自滾動高點，收盤對收盤)：{q('max_drawdown', lambda v: _pct(v))}")
        L.append(f"   最大回檔換算ATR倍數(回檔/高點當天ATR14)：{q('max_pullback_atr', lambda v: f'{v:.1f}')}")
        bm = pd.to_numeric(w["below_ma60_after_start"], errors="coerce").dropna()
        L.append(f"   起漲後(區間內)曾收在MA60下方的比例：{_pct(bm.mean()) if len(bm) else '—'}")
        L.append(f"   起漲時營收3月均YoY(point-in-time)：{q('rev_yoy3_at_move_start', lambda v: f'{v:.0f}%')}")
        L.append(f"   起漲時最新已公告季EPS年增率(point-in-time)：{q('ms_eps_yoy', lambda v: f'{v:.0f}%')}")
        L.append(f"   起漲時3個月報酬：{q('ms_ret_3m', lambda v: _pct(v, 0))}；MA60乖離：{q('ms_bias60', lambda v: _pct(v, 0))}")
        L.append(f"   起漲時60日盤整幅度：{q('ms_range60', lambda v: _pct(v, 0))}；布林帶寬百分位：{q('ms_bbw_pct', lambda v: f'{v:.0f}')}")
        L.append(f"   起漲時ATR%：{q('ms_atr_pct', lambda v: f'{v:.1f}%')}；量能爆發(5日/60日)：{q('ms_vol_burst', lambda v: f'{v:.1f}')}；"
                 f"RSI14：{q('ms_rsi14', lambda v: f'{v:.0f}')}")
        ba = pd.to_numeric(w["ms_bull_align"], errors="coerce").dropna() if "ms_bull_align" in w.columns else pd.Series(dtype=float)
        L.append(f"   起漲時均線多頭排列的比例：{_pct(ba.mean()) if len(ba) else '—'}(起漲日的籌碼快照需要額外下載，這次沒有做)")
        L.append("   解讀：如果多數贏家途中回檔超過 X ATR，移動停損設比 X 緊就會在中途被洗出場。")
    else:
        L.append("   (本區間沒有漲1倍以上的股票)")
    L.append("")
    L.append("【簡單規則比較表】(全部納入股票逐檔模擬；每筆NT$100,000、隔日開盤進場、成本0.1425%×2+0.3%稅)")
    L.append(f"   參考：全部納入股票等權重買進持有(含成本)平均報酬 {_pct(meta['bh_mean_net'])}、"
             f"每檔投入NT$100,000的總損益 {_money(meta['bh_total_pnl'])}；加權指數 {_pct(meta['taiex_ret'])}")
    L.append(f"   {'規則':<7}{'筆數':>6}{'勝率':>7}{'PF':>6}{'總損益':>13}{'扣前5大':>13}{'持有日':>7}{'抓到贏家':>8}{'贏家實現中位':>11}{'損益來自贏家':>11}")
    for _, r in res["rules"].iterrows():
        L.append(f"   {r['rule']:<7}{r['trades']:>6}{_pct(r['win_rate'], 0):>7}{_f(r['pf']):>6}{_money(r['total_pnl']):>13}"
                 f"{_money(r['pnl_ex_top5']):>13}{_f(r['avg_hold_days'], 0):>7}{_pct(r['winners_traded_pct'], 0):>8}"
                 f"{_pct(r['winner_median_realized'], 0):>11}{_pct(r['winner_pnl_share'], 0):>11}")
    L.append("   進場：" + "；".join(f"{k}={v}" for k, v in ENTRY_RULES.items()))
    L.append("   出場：" + "；".join(f"{k}={v}" for k, v in EXIT_RULES.items()) + "；X1~X3另有初始停損2×ATR；期末一律平倉")
    L.append("   抓到贏家=漲1倍贏家中至少有一筆交易的比例；贏家實現中位=這些贏家個股所有交易複利後報酬的中位數"
             "(對照：贏家區間報酬R中位數 " + (_pct(res["rules"]["winner_median_R"].iloc[0], 0) if len(res["rules"]) else "—")
             + ")；損益來自贏家=贏家個股損益/總損益(總損益<=0時留空)。")
    L.append("   注意：每檔獨立模擬、不受總資金限制，總損益≠一個真實帳戶的報酬；漲停買不到、流動性衝擊都沒有模擬。")
    L.append("")
    L.append("【結論提示】(中性描述，請自行判斷)")
    t1 = lift[(lift["tier"] == "漲1倍以上") & (lift["n_cond"] >= 30) & (lift["winners_cond"] >= 3)]
    if len(t1):
        best = t1.sort_values("lift", ascending=False).iloc[0]
        beat = int((t1["lift"] > p95).sum()) if np.isfinite(p95) else 0
        L.append(f"  ・本區間lift最高的單一條件是「{best['condition']}」(lift {_f(best['lift'])}、覆蓋 {_pct(best['coverage'])} 的贏家)；"
                 f"lift超過洗牌基準 {_f(p95)} 的單一條件有 {beat} 個。")
    rules = res["rules"]
    if len(rules) and rules["trades"].sum() > 0:
        rb = rules.sort_values("total_pnl", ascending=False).iloc[0]
        L.append(f"  ・總損益最高的規則是 {rb['rule']}({_money(rb['total_pnl'])}，扣除前5大後 {_money(rb['pnl_ex_top5'])})；"
                 "如果扣除前5大後大幅縮水，代表結果靠少數幾檔撐起來。")
    L.append("  ・這些都是「同一段行情回頭看」的樣本內結果：請用 window=2022-2023 / 2020-2021 / 2018-2019 重跑，"
             "條件或規則在不同區間都站得住，才有參考價值。")
    L.append("  ・本報告不構成任何個股推薦。")
    L.append("")
    L.append(f"執行時間：{ctx.get('runtime_text', '')}")
    return "\n".join(L) + "\n"


def write_outputs(res: dict, out_dir: str, ctx: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    wl = res["winners"].copy()
    keep = [c for c in FEATURE_LABELS if c in wl.columns]
    wl[keep].rename(columns=FEATURE_LABELS).to_csv(os.path.join(out_dir, "winners_list.csv"),
                                                   index=False, encoding="utf-8-sig")
    lift = res["lift"].sort_values(["tier", "lift"], ascending=[True, False])
    lift.rename(columns=LIFT_LABELS).to_csv(os.path.join(out_dir, "features_lift.csv"), index=False, encoding="utf-8-sig")
    pairs = res.get("pairs")
    if pairs is not None:
        pairs.rename(columns=PAIR_LABELS).to_csv(os.path.join(out_dir, "features_pairs.csv"), index=False, encoding="utf-8-sig")
    null = res.get("null") or {}
    if "singles" in null:
        os.makedirs(os.path.join(out_dir, "debug"), exist_ok=True)
        pd.DataFrame({"單一條件最大lift": null["singles"], "雙條件最大lift": null["pairs"]}).to_csv(
            os.path.join(out_dir, "debug", "shuffle_baseline.csv"), index=False, encoding="utf-8-sig")
    res["industry"].rename(columns=INDUSTRY_LABELS).to_csv(os.path.join(out_dir, "industry_lift.csv"),
                                                            index=False, encoding="utf-8-sig")
    res["rules"].rename(columns=RULE_LABELS).to_csv(os.path.join(out_dir, "rules_compare.csv"), index=False, encoding="utf-8-sig")
    res["trades"].rename(columns=TRADE_LABELS).to_csv(os.path.join(out_dir, "rule_trades.csv"), index=False, encoding="utf-8-sig")
    allf = res["all_features"]
    keep = [c for c in FEATURE_LABELS if c in allf.columns]
    os.makedirs(os.path.join(out_dir, "debug"), exist_ok=True)
    allf[keep].rename(columns=FEATURE_LABELS).to_csv(os.path.join(out_dir, "debug", "all_stocks_features.csv"),
                                                     index=False, encoding="utf-8-sig")
    text = build_summary(res, ctx)
    with open(os.path.join(out_dir, "summary_winners.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


# ===========================================================================
# 下載(只在GitHub Actions上真正執行；測試全部monkeypatch)
# ===========================================================================
def _http_get(url: str):
    import requests
    return run_with_hard_timeout(requests.get, args=(url,),
                                 kwargs={"headers": HTTP_HEADERS, "timeout": HTTP_TIMEOUT}, timeout=HTTP_TIMEOUT + 10)


def _write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def load_universe(cache_dir: str, refresh: bool, debug_dir: str, today_str: str):
    """回傳(DataFrame, warnings)。兩個市場都抓不到 → 空DataFrame(呼叫端fail closed)。"""
    frames, warnings, lines = [], [], []
    for market, mode in ISIN_MODES.items():
        cache_path = os.path.join(cache_dir, "isin", f"isin_{mode}_{today_str}.csv")
        if not refresh and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, dtype=str).fillna("")
                if len(df):
                    frames.append(df)
                    lines.append(f"{market}: 讀快取 {len(df)} 檔")
                    continue
            except Exception:
                pass
        html, err = None, None
        for attempt in range(3):
            try:
                resp = _http_get(ISIN_URL.format(mode=mode))
                if resp.status_code == 200:
                    html = resp.content.decode("cp950", errors="replace")
                    break
                err = f"HTTP {resp.status_code}"
            except (Exception, HardTimeout) as e:
                err = str(e)[:200]
            time.sleep(3)
        if html is None:
            warnings.append(f"{market}股票清單(ISIN strMode={mode})下載失敗：{err}；本次不含{market}股票")
            lines.append(f"{market}: 下載失敗 {err}")
            continue
        _write_text(os.path.join(debug_dir, f"isin_sample_{mode}.html"), html[:8000])
        df, rep = parse_isin_html(html, market)
        lines.append(f"{market}: 解析 {rep['rows_seen']} 列、保留 {rep['kept']} 檔；分類={rep['sections']}；排除={rep['excluded']}")
        if df.empty:
            warnings.append(f"{market}股票清單解析結果是0檔(格式可能變了，請看debug/isin_sample_{mode}.html)")
            continue
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        frames.append(df)
    _write_text(os.path.join(debug_dir, "universe_report.txt"), "\n".join(lines) + "\n")
    if not frames:
        return pd.DataFrame(columns=["code", "name", "market", "industry"]), warnings
    uni = pd.concat(frames, ignore_index=True).drop_duplicates("code").sort_values("code").reset_index(drop=True)
    uni["code"] = uni["code"].astype(str)
    return uni, warnings


def _extract_ticker_frame(raw: pd.DataFrame, sym: str, single: bool):
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(raw.columns.get_level_values(0))
        lv1 = set(raw.columns.get_level_values(1))
        if sym in lv0:
            sub = raw[sym]
        elif sym in lv1:
            sub = raw.xs(sym, axis=1, level=1)
        else:
            return None
    else:
        sub = raw if single else None
    if sub is None:
        return None
    sub = sub.dropna(how="all")
    if sub.empty or "Close" not in sub.columns:
        return None
    return sub


def load_prices(universe: pd.DataFrame, dl_start: str, dl_end: str, cache_dir: str, refresh: bool, debug_dir: str):
    """回傳({code: raw DataFrame}, 失敗DataFrame)。yfinance end不含當天，呼叫端已+1天。"""
    import yfinance as yf
    from data_loader import _symbol_for
    pdir = os.path.join(cache_dir, "prices")
    os.makedirs(pdir, exist_ok=True)
    out, todo = {}, []
    for _, r in universe.iterrows():
        code = str(r["code"])
        cp = os.path.join(pdir, f"{code}_{dl_start}_{dl_end}.csv")
        if not refresh and os.path.exists(cp):
            try:
                df = pd.read_csv(cp, index_col=0, parse_dates=True)
                if len(df):
                    out[code] = df
                    continue
            except Exception:
                pass
        todo.append((code, _symbol_for(code, {"otc": r["market"] == "上櫃"})))
    print(f"股價：{len(out)} 檔讀快取，{len(todo)} 檔需要下載(每批{PRICE_BATCH_SIZE}檔)", flush=True)
    failures = {}
    for pass_no in range(2):  # 第二輪只重抓第一輪失敗的
        if not todo:
            break
        retry = []
        for b in range(0, len(todo), PRICE_BATCH_SIZE):
            batch = todo[b:b + PRICE_BATCH_SIZE]
            syms = [s for _, s in batch]
            raw = None
            for attempt in range(3):
                try:
                    raw = yf.download(tickers=syms, start=dl_start, end=dl_end, interval="1d", group_by="ticker",
                                      threads=True, progress=False, auto_adjust=False)
                    break
                except Exception as e:
                    print(f"  批次下載例外(第{attempt + 1}次)：{e}", flush=True)
                    time.sleep(5 * (attempt + 1))
            for code, sym in batch:
                sub = _extract_ticker_frame(raw, sym, single=len(syms) == 1)
                if sub is None:
                    retry.append((code, sym))
                    failures[code] = (sym, "沒有資料" if raw is not None else "整批下載失敗")
                    continue
                cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in sub.columns]
                df = sub[cols].copy()
                idx = pd.DatetimeIndex(df.index)
                df.index = idx.tz_localize(None) if idx.tz is not None else idx
                df.to_csv(os.path.join(pdir, f"{code}_{dl_start}_{dl_end}.csv"))
                out[code] = df
                failures.pop(code, None)
            print(f"  第{pass_no + 1}輪 {min(b + PRICE_BATCH_SIZE, len(todo))}/{len(todo)}，累計成功 {len(out)} 檔", flush=True)
            time.sleep(PRICE_BATCH_DELAY)
        todo = retry
    fail_df = pd.DataFrame([{"代號": c, "yfinance代號": s, "原因": why} for c, (s, why) in sorted(failures.items())],
                           columns=["代號", "yfinance代號", "原因"])
    os.makedirs(debug_dir, exist_ok=True)
    fail_df.to_csv(os.path.join(debug_dir, "price_failures.csv"), index=False, encoding="utf-8-sig")
    _write_text(os.path.join(debug_dir, "price_download_report.txt"),
                f"下載區間 {dl_start} ~ {dl_end}(end不含)\n要求 {len(universe)} 檔，成功 {len(out)} 檔，失敗 {len(fail_df)} 檔\n")
    return out, fail_df


def load_taiex(dl_start: str, dl_end: str, cache_dir: str, refresh: bool) -> pd.Series:
    try:
        from data_loader import load_index_series
        return load_index_series("^TWII", dl_start, dl_end, refresh=refresh, cache_dir=os.path.join(cache_dir, "index"))
    except Exception as e:
        print(f"加權指數下載失敗：{e}", flush=True)
        return pd.Series(dtype=float)


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> list:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_mops_month(market: str, year: int, month: int, kind: int = 0):
    """回傳(DataFrame或None, info dict)。依序試MOPS_DOMAINS。"""
    info = {"market": market, "year": year, "month": month, "url": "", "status": "", "rows": 0,
            "header_tables": 0, "positional_tables": 0, "html": None}
    path = MOPS_PATH[market].format(roc=year - 1911, m=month, kind=kind)
    for domain in MOPS_DOMAINS:
        url = domain + path
        info["url"] = url
        for attempt in range(2):
            try:
                resp = _http_get(url)
            except (Exception, HardTimeout) as e:
                info["status"] = f"連線失敗:{str(e)[:80]}"
                time.sleep(2)
                continue
            if resp.status_code != 200:
                info["status"] = f"HTTP {resp.status_code}"
                break
            html = resp.content.decode("cp950", errors="replace")
            info["html"] = html
            df, rep = parse_mops_revenue_html(html)
            info.update({"rows": rep["rows"], "header_tables": rep["header_tables"],
                         "positional_tables": rep["positional_tables"]})
            if rep["rows"] > 0:
                info["status"] = "ok"
                return df, info
            info["status"] = f"解析0列(表格{rep['tables']}個)"
            break
    return None, info


def load_revenue(start_month: pd.Timestamp, end_month: pd.Timestamp, cache_dir: str, refresh: bool, debug_dir: str):
    """回傳(rev_long DataFrame, ok)。成功的月份才寫快取(沒公告/失敗的下次會重抓)。"""
    rdir = os.path.join(cache_dir, "revenue")
    os.makedirs(rdir, exist_ok=True)
    frames, report, samples = [], [], []
    saved_sample = set()
    months = months_between(start_month, end_month)
    jobs = [(y, m, market, kind) for (y, m) in months for market in MOPS_PATH for kind in MOPS_KINDS]
    print(f"月營收：{len(months)} 個月 × 2 市場 × 國內/KY = {len(jobs)} 頁", flush=True)
    for (y, m, market, kind) in jobs:
        mk = "sii" if market == "上市" else "otc"
        cp = os.path.join(rdir, f"rev_{mk}{kind}_{y}{m:02d}.csv")
        if not refresh and os.path.exists(cp):
            try:
                df = pd.read_csv(cp, dtype={"code": str})
                if len(df):
                    df["year"], df["month"], df["market"] = y, m, market
                    frames.append(df)
                    report.append({"年": y, "月": m, "市場": market, "種類": MOPS_KINDS[kind], "狀態": "快取",
                                   "列數": len(df), "網址": ""})
                    continue
            except Exception:
                pass
        df, info = fetch_mops_month(market, y, m, kind)
        html = info.pop("html", None)
        key = (market, kind, info["status"] == "ok")
        if html and key not in saved_sample:
            saved_sample.add(key)
            tag = "ok" if key[2] else "fail"
            _write_text(os.path.join(debug_dir, f"mops_sample_{mk}{kind}_{tag}.html"),
                        html[:20000])
        report.append({"年": y, "月": m, "市場": market, "種類": MOPS_KINDS[kind], "狀態": info["status"], "列數": info["rows"],
                       "表頭解析表格數": info["header_tables"], "位置備援表格數": info["positional_tables"],
                       "網址": info["url"]})
        if df is not None and len(df):
            # 最近2個月的公告可能還沒收齊(每月10日截止、會更正)，不寫快取，下次重抓
            _now = pd.Timestamp.today()
            if (_now.year * 12 + _now.month) - (y * 12 + m) > 2:
                df.to_csv(cp, index=False)
            samples.append(f"{y}-{m:02d} {market}{MOPS_KINDS[kind]}：" + " | ".join(
                f"{r.code} {r.name} 營收{r.revenue:,.0f} YoY{r.yoy_pct:.2f}%" for r in df.head(3).itertuples()))
            df = df.copy()
            df["year"], df["month"], df["market"] = y, m, market
            frames.append(df)
        time.sleep(MOPS_DELAY)
    os.makedirs(debug_dir, exist_ok=True)
    pd.DataFrame(report).to_csv(os.path.join(debug_dir, "revenue_parse_report.csv"), index=False, encoding="utf-8-sig")
    _write_text(os.path.join(debug_dir, "revenue_samples.txt"), "\n".join(samples) + "\n")
    if not frames:
        return pd.DataFrame(columns=_REV_COLS + ["year", "month", "market"]), False
    rev = pd.concat(frames, ignore_index=True)
    rev["code"] = rev["code"].astype(str)
    return rev, True


def load_extra(cal, d0, d1, cache_dir: str, refresh: bool, debug_dir: str) -> dict:
    """籌碼/估值/季報(winners_extra)；測試時整個換掉。"""
    return wx.load_all_extra(cal, d0, d1, cache_dir, refresh, debug_dir)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="最近兩年大漲股的共同點(描述性研究)")
    p.add_argument("--output-dir", default="results_winners")
    p.add_argument("--start", default="", help="區間起點YYYY-MM-DD，留空=end往前2年")
    p.add_argument("--end", default="", help="區間終點YYYY-MM-DD，留空=最新資料")
    p.add_argument("--window", default="", choices=[""] + list(WINDOW_PRESETS),
                   help="預設區間(--start/--end有給就以它們為準)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--refresh", action="store_true", help="忽略快取重新下載")
    p.add_argument("--max-stocks", type=int, default=0, help="只取前N檔(測試用，0=全部)")
    p.add_argument("--min-turnover", type=float, default=MIN_TURNOVER)
    p.add_argument("--skip-extra", action="store_true", help="不抓籌碼/估值/季報(相關特徵全部空值)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    t0 = time.time()
    args = parse_args(argv)
    start, end = args.start, args.end
    if args.window:
        ws, we = WINDOW_PRESETS[args.window]
        start, end = start or ws, end or we
    start_ts, end_ts = resolve_window(start, end)
    out_dir = args.output_dir
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    today_str = _taipei_today().isoformat()
    dl_start = (start_ts - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)).date().isoformat()
    dl_end = (end_ts + pd.Timedelta(days=1)).date().isoformat()
    print(f"研究區間 {start_ts.date()} ~ {end_ts.date()}；股價下載 {dl_start} ~ {dl_end}", flush=True)

    universe, warnings = load_universe(args.cache_dir, args.refresh, debug_dir, today_str)
    if universe.empty:
        msg = "⚠️ 上市/上櫃股票清單都抓不到(ISIN頁面)，無法進行研究(fail closed)。請看 debug/universe_report.txt。\n"
        _write_text(os.path.join(out_dir, "summary_winners.txt"), msg)
        print(msg, flush=True)
        return 1
    if args.max_stocks and args.max_stocks > 0:
        universe = universe.head(args.max_stocks).reset_index(drop=True)
    print(f"股票清單 {len(universe)} 檔(上市 {int((universe['market'] == '上市').sum())}、上櫃 {int((universe['market'] == '上櫃').sum())})", flush=True)

    prices, fail_df = load_prices(universe, dl_start, dl_end, args.cache_dir, args.refresh, debug_dir)
    if len(prices) == 0:
        msg = "⚠️ 股價全部下載失敗(yfinance)，無法進行研究(fail closed)。請看 debug/price_failures.csv。\n"
        _write_text(os.path.join(out_dir, "summary_winners.txt"), msg)
        print(msg, flush=True)
        return 1
    fail_ratio = len(fail_df) / max(1, len(universe))
    if fail_ratio > 0.05:
        warnings.append(f"股價下載失敗 {len(fail_df)} 檔({fail_ratio:.0%})，清單見 debug/price_failures.csv")
    taiex = load_taiex(dl_start, dl_end, args.cache_dir, args.refresh)
    if taiex is None or len(taiex) == 0:
        warnings.append("加權指數(^TWII)下載失敗，報告中的指數報酬會是空的")

    rev_start = (start_ts - pd.DateOffset(months=REVENUE_LOOKBACK_MONTHS)).normalize()
    revenue, rev_ok = load_revenue(rev_start, end_ts, args.cache_dir, args.refresh, debug_dir)
    if not rev_ok:
        print("⚠️ 月營收完全抓不到，本次不使用營收特徵繼續執行", flush=True)

    extra, extra_ctx = None, {"extra_status": None}
    if not args.skip_extra:
        cal = build_calendar(prices)
        in_win = cal[(cal >= start_ts) & (cal <= end_ts)]
        if len(in_win) >= 2:
            try:
                extra = load_extra(cal, in_win[0], in_win[-1], args.cache_dir, args.refresh, debug_dir)
                extra_ctx = {"extra_status": wx.source_status(extra.get("report", [])),
                             "extra_errors": extra.get("errors", []),
                             "extra_requests": (extra.get("estimated_requests"), extra.get("actual_requests"))}
                if not extra_ctx["extra_status"]:
                    extra_ctx["extra_status"] = [(n, 0, 1) for n in wx.SOURCE_NAMES.values()]
            except Exception as e:  # 任何沒料到的錯誤都不能讓研究中斷
                extra = None
                extra_ctx = {"extra_status": [(n, 0, 1) for n in wx.SOURCE_NAMES.values()],
                             "extra_errors": [f"籌碼/估值/季報整體失敗：{str(e)[:200]}"]}

    res = run_study(universe, prices, revenue, taiex, start_ts, end_ts, args.min_turnover, extra=extra)
    elapsed = time.time() - t0
    ctx = {"start": start_ts.date(), "end": end_ts.date(), "revenue_ok": rev_ok, "warnings": warnings,
           "runtime_text": f"{elapsed / 60:.1f} 分鐘({elapsed:.0f} 秒)", **extra_ctx}
    text = write_outputs(res, out_dir, ctx)
    print(text, flush=True)
    print(f"完成，總執行時間 {elapsed:.1f} 秒，結果在 {out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
