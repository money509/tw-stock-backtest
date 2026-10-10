"""
winners_extra.py — winners_study.py 的「額外特徵」：籌碼面、估值、季財報、進階技術面、雙條件lift與洗牌基準。

每一個資料來源都是「可有可無」(fail soft)：抓不到/解析不出來 → 那些特徵是空值，
summary最上面會列出失敗的來源，研究照常跑完。原始回應的前幾千字存到 debug/extra_sample_*.txt，
逐日/逐季的請求結果存到 debug/extra_report.csv，第一次在GitHub Actions跑完請先看這兩個檔案。

只抓「起點那天需要的日期」(不是整個區間)：
  三大法人  起點(含)往前20個交易日 × 上市/上櫃
  融資融券  起點、起點前20個交易日 × 上市/上櫃
  外資持股  起點、起點前60個交易日 × 上市/上櫃
  估值      起點 × 上市/上櫃
  季財報    綜合損益(t163sb04)+資產負債(t163sb05)，從「起點時最新已公告季」往前8季，到「終點時最新已公告季」× 上市/上櫃
集保大戶持股：歷史資料只能逐股查詢(集保結算所)，上千檔×多個日期的請求量太大，這裡**不抓**，報告裡會寫明。

⚠️ 端點格式說明(沒有網路的sandbox裡無法驗證，全部要看第一次執行的debug/)：
  TWSE rwd JSON(T86/MI_MARGN/MI_QFIIS/BWIBBU_d)：假設是 {"stat":"OK","fields":[...],"data":[[...]]}
    或 {"stat":"OK","tables":[{"fields":[...],"data":[...]}]}，欄位一律「用名稱找位置」。
  TPEx：舊站 .php?l=zh-tw&o=json&d=民國年/MM/DD 回傳 {"aaData":[[...]]}(沒有欄位名稱 → 用位置，位置是猜的)；
    舊站失敗再試新站 /www/zh-tw/... ?response=json(假設是 {"tables":[{"fields","data"}]}，網址是猜的)。
  MOPS t163sb04/05：POST表單回傳HTML，每個產業格式一個表格；表頭含「公司代號」。
"""
import datetime
import json
import os
import re
import time

import numpy as np
import pandas as pd

from network_utils import run_with_hard_timeout, HardTimeout

HTTP_HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")}
HTTP_TIMEOUT = 30
REQUEST_DELAY = 3.0        # TWSE/TPEx/MOPS 每次請求間隔(這些端點對頻繁請求很敏感)
MAX_ATTEMPTS = 3

SOURCE_NAMES = {"insti": "三大法人買賣超", "margin": "融資融券", "qfii": "外資持股比例", "valuation": "本益比/淨值比/殖利率",
                "t163sb04": "季財報(綜合損益)", "t163sb05": "季財報(資產負債)"}
TWSE_URLS = {
    "insti": "https://www.twse.com.tw/rwd/zh/fund/T86?date={ymd}&selectType=ALLBUT0999&response=json",
    "margin": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={ymd}&selectType=ALL&response=json",
    "qfii": "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date={ymd}&selectType=ALLBUT0999&response=json",
    "valuation": "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={ymd}&selectType=ALL&response=json",
}
# TPEx：先試舊站(.php，aaData)，失敗再試新站(網址是猜的，靠debug驗證)
TPEX_URLS = {
    "insti": ["https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=json&se=EW&t=D&d={roc}",
              "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=EW&date={slash}&response=json"],
    "margin": ["https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php?l=zh-tw&o=json&d={roc}",
               "https://www.tpex.org.tw/www/zh-tw/margin/balance?date={slash}&response=json"],
    "qfii": ["https://www.tpex.org.tw/web/stock/3insti/qfii/qfii_result.php?l=zh-tw&o=json&d={roc}",
             "https://www.tpex.org.tw/www/zh-tw/insti/qfii?date={slash}&response=json"],
    "valuation": ["https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera_result.php?l=zh-tw&o=json&d={roc}",
                  "https://www.tpex.org.tw/www/zh-tw/afterTrading/peQryDate?date={slash}&response=json"],
}
MOPS_DOMAINS = ["https://mopsov.twse.com.tw", "https://mops.twse.com.tw"]
MOPS_T163_PATH = "/mops/web/ajax_{kind}"
QUARTERS_BACK = 8

_CODE_RE = re.compile(r"^\d{4}$")


# ===========================================================================
# 共用小工具
# ===========================================================================
def _clean(v) -> str:
    return re.sub(r"<[^>]+>", "", str(v)).replace("　", " ").strip()


def _code(v):
    s = _clean(v).split(" ")[0] if v is not None else ""
    return s if _CODE_RE.match(s) else None


def _num(v) -> float:
    if v is None:
        return float("nan")
    s = _clean(v).replace(",", "").replace("%", "").replace("＋", "+").replace("－", "-")
    if s in ("", "-", "--", "---", "—", "N/A", "nan", "None", "不適用"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _norm(s) -> str:
    return re.sub(r"\s+", "", _clean(s)).replace("（", "(").replace("）", ")")


def _find(fields, include, exclude=(), nth=0):
    """回傳第nth個「名稱包含include全部子字串、不含exclude任何子字串」的欄位位置。"""
    hits = [i for i, f in enumerate(fields)
            if all(x in _norm(f) for x in include) and not any(x in _norm(f) for x in exclude)]
    return hits[nth] if len(hits) > nth else None


def json_tables(payload) -> list:
    """把各種格式的JSON回應整理成 [(fields或None, rows)]。"""
    out = []
    if not isinstance(payload, dict):
        return out
    for t in payload.get("tables") or []:
        if isinstance(t, dict) and isinstance(t.get("data"), list) and t["data"]:
            out.append((t.get("fields"), t["data"]))
    if isinstance(payload.get("data"), list) and payload["data"]:
        out.append((payload.get("fields"), payload["data"]))
    if isinstance(payload.get("aaData"), list) and payload["aaData"]:
        out.append((payload.get("fields"), payload["aaData"]))
    return out


def _code_offset(rows) -> int:
    """沒有欄位名稱時：找第一列裡第一個像股票代號的位置(有些表前面多一個「排行」欄)。"""
    for row in rows[:5]:
        for i, v in enumerate(row[:3]):
            if _code(v):
                return i
    return 0


def _rows_to_df(rows, colpos: dict, cols: list) -> pd.DataFrame:
    recs = []
    ci = colpos["code"]
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) <= ci:
            continue
        c = _code(row[ci])
        if not c:
            continue
        rec = {"code": c}
        for k in cols:
            p = colpos.get(k)
            rec[k] = _num(row[p]) if (p is not None and p < len(row)) else float("nan")
        recs.append(rec)
    df = pd.DataFrame(recs, columns=["code"] + cols)
    return df.drop_duplicates("code", keep="last").reset_index(drop=True)


def _parse_generic(payload, cols: list, by_name, positional: dict):
    """by_name(fields) → {欄位key: 位置} 或 None；positional：沒有欄位名稱時的位置(相對代號欄)。"""
    for fields, rows in json_tables(payload):
        if fields:
            pos = by_name(fields)
            if pos and pos.get("code") is not None and any(pos.get(k) is not None for k in cols):
                df = _rows_to_df(rows, pos, cols)
                if len(df):
                    return df, "欄位名稱"
    for fields, rows in json_tables(payload):
        if not fields:
            off = _code_offset(rows)
            pos = {"code": off, **{k: (v + off) for k, v in positional.items()}}
            df = _rows_to_df(rows, pos, cols)
            if len(df):
                return df, "位置(猜測)"
    return pd.DataFrame(columns=["code"] + cols), "解析不到"


# ===========================================================================
# 各端點解析
# ===========================================================================
INSTI_COLS = ["foreign_net", "trust_net", "total_net"]


def parse_insti(payload, market: str):
    """三大法人買賣超(股)。TWSE T86 / TPEx 3itrade_hedge。"""
    def by_name(f):
        foreign = _find(f, ("外", "買賣超", "不含"))
        if foreign is None:
            foreign = _find(f, ("外", "買賣超"), ("自營",))
        return {"code": _find(f, ("代號",)), "foreign_net": foreign,
                "trust_net": _find(f, ("投信", "買賣超")),
                "total_net": _find(f, ("三大法人", "買賣超")) if _find(f, ("三大法人", "買賣超")) is not None
                else _find(f, ("合計", "買賣超"))}
    # TPEx舊站aaData：0代號 1名稱 2-4外資(不含自營)買/賣/超 … 11-13投信 … 23三大法人合計；TWSE：4外資超 10投信超 最後=合計
    pos = {"foreign_net": 4, "trust_net": 13, "total_net": 23} if market == "上櫃" else \
        {"foreign_net": 4, "trust_net": 10, "total_net": 18}
    return _parse_generic(payload, INSTI_COLS, by_name, pos)


MARGIN_COLS = ["margin_bal", "short_bal"]


def parse_margin(payload, market: str):
    """融資/融券今日餘額(張)。TWSE MI_MARGN 有兩個「今日餘額」(第1個=融資、第2個=融券)。"""
    def by_name(f):
        m = _find(f, ("今日餘額",), nth=0)
        s = _find(f, ("今日餘額",), nth=1)
        if m is None:
            m = _find(f, ("資餘額",), ("前",))
            s = _find(f, ("券餘額",), ("前",))
        return {"code": _find(f, ("代號",)), "margin_bal": m, "short_bal": s}
    # TPEx舊站：0代號 1名稱 2前資餘額 3資買 4資賣 5現償 6資餘額 … 10前券餘額 11券賣 12券買 13券償 14券餘額
    pos = {"margin_bal": 6, "short_bal": 14} if market == "上櫃" else {"margin_bal": 6, "short_bal": 12}
    return _parse_generic(payload, MARGIN_COLS, by_name, pos)


QFII_COLS = ["shares_issued", "foreign_pct"]


def parse_qfii(payload, market: str):
    """外資持股比例(%)與發行股數。"""
    def by_name(f):
        pct = _find(f, ("全體", "持股比率"))
        if pct is None:
            pct = _find(f, ("持股比率",), ("尚可", "上限"))
        return {"code": _find(f, ("代號",)), "shares_issued": _find(f, ("發行股數",)), "foreign_pct": pct}
    # TWSE：0代號 1名稱 2ISIN 3發行股數 … 7全體外資持股比率；TPEx舊站(猜)：0代號 1名稱 2發行股數 … 6持股比率
    pos = {"shares_issued": 2, "foreign_pct": 6} if market == "上櫃" else {"shares_issued": 3, "foreign_pct": 7}
    return _parse_generic(payload, QFII_COLS, by_name, pos)


VAL_COLS = ["pe", "pb", "dy"]


def parse_valuation(payload, market: str):
    """本益比/股價淨值比/殖利率(%)。本益比空白(通常是虧損)→NaN。"""
    def by_name(f):
        return {"code": _find(f, ("代號",)), "pe": _find(f, ("本益比",)), "pb": _find(f, ("股價淨值比",)),
                "dy": _find(f, ("殖利率",))}
    # TWSE：0代號 1名稱 2收盤 3殖利率 4股利年度 5本益比 6淨值比；TPEx舊站：0代號 1名稱 2本益比 3每股股利 4股利年度 5殖利率 6淨值比
    pos = {"pe": 2, "pb": 6, "dy": 5} if market == "上櫃" else {"pe": 5, "pb": 6, "dy": 3}
    return _parse_generic(payload, VAL_COLS, by_name, pos)


PARSERS = {"insti": parse_insti, "margin": parse_margin, "qfii": parse_qfii, "valuation": parse_valuation}


# ---------------------------------------------------------------------------
# MOPS 季財報彙總(HTML)
# ---------------------------------------------------------------------------
T163_COLS = {"t163sb04": ["revenue", "gross", "op_income", "net_income", "eps"], "t163sb05": ["equity"]}


def _t163_colmap(names: list, kind: str) -> dict:
    n = [_norm(x) for x in names]

    def first(include, exclude=(), prefer=None):
        hits = [i for i, x in enumerate(n) if all(s in x for s in include) and not any(s in x for s in exclude)]
        if prefer:
            pref = [i for i in hits if prefer in n[i]]
            if pref:
                return pref[0]
        return hits[0] if hits else None
    m = {"code": first(("公司代號",)) if first(("公司代號",)) is not None else first(("代號",))}
    if kind == "t163sb04":
        m.update({"revenue": first(("營業收入",)), "gross": first(("營業毛利",), prefer="淨額"),
                  "op_income": first(("營業利益",)), "net_income": first(("本期", "淨利"), ("歸屬", "其他", "繼續營業")),
                  "eps": first(("基本每股盈餘",))})
    else:
        m.update({"equity": first(("權益總",), ("歸屬", "負債及權益", "負債與權益"))})
    return m


def parse_t163(html: str, kind: str):
    """回傳(DataFrame[code + T163_COLS[kind]], report)。表頭可能每N列重複一次，遇到就重讀。"""
    from winners_study import parse_html_tables
    cols = T163_COLS[kind]
    out = []
    rep = {"tables": 0, "tables_with_data": 0}
    for table in parse_html_tables(html):
        rep["tables"] += 1
        colmap = None
        n_before = len(out)
        for row in table["rows"]:
            texts = [c["text"] for c in row]
            if any("公司代號" in _norm(t) for t in texts):
                colmap = _t163_colmap(texts, kind)
                continue
            if colmap is None or colmap.get("code") is None or not texts:
                continue
            c = _code(texts[colmap["code"]]) if colmap["code"] < len(texts) else None
            if not c:
                continue
            rec = {"code": c}
            for k in cols:
                p = colmap.get(k)
                rec[k] = _num(texts[p]) if (p is not None and p < len(texts)) else float("nan")
            out.append(rec)
        if len(out) > n_before:
            rep["tables_with_data"] += 1
    df = pd.DataFrame(out, columns=["code"] + cols).drop_duplicates("code", keep="last").reset_index(drop=True)
    rep["rows"] = int(len(df))
    return df, rep


# ===========================================================================
# 季報 point-in-time
# ===========================================================================
def quarter_deadline(year: int, q: int) -> pd.Timestamp:
    """法定公告期限(一般業)：Q1→5/15、Q2→8/14、Q3→11/14、Q4(年報)→隔年3/31。"""
    return {1: pd.Timestamp(year, 5, 15), 2: pd.Timestamp(year, 8, 14), 3: pd.Timestamp(year, 11, 14),
            4: pd.Timestamp(year + 1, 3, 31)}[q]


def quarter_available(year: int, q: int) -> pd.Timestamp:
    """期限過了才算看得到：期限隔天(含)起。"""
    return quarter_deadline(year, q) + pd.Timedelta(days=1)


def latest_published_quarter(date) -> tuple:
    d = pd.Timestamp(date)
    y, q = d.year, 4
    for _ in range(12):
        if quarter_available(y, q) <= d:
            return y, q
        y, q = (y - 1, 4) if q == 1 else (y, q - 1)
    return y, q


def quarters_between(a: tuple, b: tuple) -> list:
    out = []
    y, q = a
    while (y, q) <= b:
        out.append((y, q))
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def shift_quarter(yq: tuple, n: int) -> tuple:
    idx = yq[0] * 4 + (yq[1] - 1) + n
    return idx // 4, idx % 4 + 1


def detect_cumulative(raw: pd.DataFrame) -> dict:
    """
    每個(年, 市場)判斷損益表是「年初累計」還是「單季」：比較同公司 Q2營收/Q1營收 的中位數，
    >1.5 視為累計(Q2≈2×Q1)。只有Q1或沒有Q2時預設「累計」(MOPS彙總表常見格式)。
    回傳 {(year, market): True/False}。
    """
    res, undecided = {}, []
    if raw is None or raw.empty:
        return res
    for (y, mk), g in raw.groupby(["year", "market"]):
        p = g.pivot_table(index="code", columns="q", values="revenue", aggfunc="last")
        r = pd.Series(dtype=float)
        if 1 in p.columns and 2 in p.columns:
            r = (p[2] / p[1]).replace([np.inf, -np.inf], np.nan)
            r = r[(p[1] > 0) & (p[2] > 0)].dropna()
        if len(r) >= 5:
            res[(int(y), mk)] = bool(r.median() > 1.5)
        else:
            undecided.append((int(y), mk))
    for (y, mk) in undecided:  # 判斷不了的年份(例如只抓到Q3、Q4)：跟同市場其他年份的多數決，都沒有就預設累計
        votes = [v for (yy, m2), v in res.items() if m2 == mk]
        res[(y, mk)] = (sum(votes) * 2 >= len(votes)) if votes else True
    return res


FLOW_COLS = ["revenue", "gross", "op_income", "net_income", "eps"]
Q_FEAT_COLS = ["eps", "eps_yoy", "gm", "gm_yoy", "om", "om_yoy", "eps4q", "roe"]


def build_quarterly_features(raw: pd.DataFrame, cumulative: dict = None) -> dict:
    """
    raw：[code, year, q, market, revenue, gross, op_income, net_income, eps, equity]
    → {code: DataFrame(index=季末日, columns=Q_FEAT_COLS + [avail_date, qlabel])}，全部是單季數字。
    累計格式：單季 = 本季累計 − 上季累計；單季格式：Q1~Q3原樣，Q4 = 全年 − (Q1+Q2+Q3)。
    EPS用相減得到的單季值是近似(期中股數可能變動)。
    """
    out = {}
    if raw is None or raw.empty:
        return out
    cumulative = cumulative if cumulative is not None else detect_cumulative(raw)
    for code, g in raw.groupby("code"):
        g = g.sort_values(["year", "q"]).drop_duplicates(["year", "q"], keep="last").set_index(["year", "q"])
        recs = []
        for (y, q), r in g.iterrows():
            cum = cumulative.get((int(y), r.get("market")), True)
            single = {}
            for c in FLOW_COLS:
                v = r.get(c, np.nan)
                if q == 1:
                    single[c] = v
                elif cum:
                    prev = g[c].get((y, q - 1), np.nan) if (y, q - 1) in g.index else np.nan
                    single[c] = v - prev
                elif q == 4:
                    prev3 = [g[c].get((y, k), np.nan) if (y, k) in g.index else np.nan for k in (1, 2, 3)]
                    single[c] = v - np.sum(prev3)
                else:
                    single[c] = v
            recs.append({"year": int(y), "q": int(q), **single, "equity": r.get("equity", np.nan)})
        df = pd.DataFrame(recs)
        df["qend"] = [pd.Timestamp(y, q * 3, 1) + pd.offsets.MonthEnd(0) for y, q in zip(df["year"], df["q"])]
        df = df.set_index("qend")
        full = pd.date_range(df.index.min(), df.index.max(), freq="QE" if _pd_has_qe() else "Q")
        df = df.reindex(full)
        with np.errstate(invalid="ignore", divide="ignore"):
            rev = df["revenue"].where(df["revenue"] > 0)
            f = pd.DataFrame(index=full)
            f["eps"] = df["eps"]
            prev_eps = df["eps"].shift(4)
            f["eps_yoy"] = ((df["eps"] - prev_eps) / prev_eps.abs().where(prev_eps.abs() > 0)) * 100
            f["gm"] = df["gross"] / rev * 100
            f["gm_yoy"] = f["gm"] - f["gm"].shift(4)
            f["om"] = df["op_income"] / rev * 100
            f["om_yoy"] = f["om"] - f["om"].shift(4)
            f["eps4q"] = df["eps"].rolling(4, min_periods=4).sum()
            ni4 = df["net_income"].rolling(4, min_periods=4).sum()
            eq = df["equity"].where(df["equity"] > 0)
            f["roe"] = ni4 / eq * 100
        f["avail_date"] = [quarter_available(d.year, (d.month - 1) // 3 + 1) for d in full]
        f["qlabel"] = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in full]
        out[str(code)] = f
    return out


def _pd_has_qe() -> bool:
    try:
        pd.tseries.frequencies.to_offset("QE")
        return True
    except ValueError:
        return False


def quarterly_on_dates(qf: pd.DataFrame, dates) -> pd.DataFrame:
    """每個日期取「avail_date<=日期」的最新一季；最新一季的季末超過9個月前 → 空值(公司沒申報)。"""
    dates = pd.DatetimeIndex(dates)
    res = pd.DataFrame(np.nan, index=dates, columns=Q_FEAT_COLS)
    res["qlabel"] = ""
    if qf is None or qf.empty or len(dates) == 0:
        return res
    avail = pd.DatetimeIndex(qf["avail_date"]).values
    pos = np.searchsorted(avail, dates.values, side="right") - 1
    for k, (d, p) in enumerate(zip(dates, pos)):
        if p < 0:
            continue
        row = qf.iloc[p]
        if (d - qf.index[p]).days > 275:
            continue
        res.iloc[k, :len(Q_FEAT_COLS)] = row[Q_FEAT_COLS].to_numpy(dtype=float)
        res.iloc[k, len(Q_FEAT_COLS)] = row["qlabel"]
    return res


# ===========================================================================
# 進階技術面(只在指定日期算，只用到那天(含)以前的K棒)
# ===========================================================================
TECH_COLS = ["bull_align", "ma60_slope", "ma120_slope", "bias60", "range60", "bbw_pct", "atr_pct", "rsi14", "k9",
             "dist_52w_low", "days_since_high", "max_dd_1y", "beta", "vol_burst", "gap_ups_60"]


def _rsi(c: np.ndarray, n: int = 14) -> float:
    if len(c) < n + 1:
        return np.nan
    d = np.diff(c)
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    if dn == 0:
        return 100.0 if up > 0 else 50.0
    return float(100 - 100 / (1 + up / dn))


def _kd_k(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 9) -> float:
    """KD(9,3,3)的K值：RSV=(C−9日最低)/(9日最高−9日最低)×100，K=2/3×前K+1/3×RSV，起始50。"""
    if len(c) < n:
        return np.nan
    k = 50.0
    for i in range(n - 1, len(c)):
        hh, ll = h[i - n + 1:i + 1].max(), l[i - n + 1:i + 1].min()
        rsv = (c[i] - ll) / (hh - ll) * 100 if hh > ll else 50.0
        k = k * 2 / 3 + rsv / 3
    return float(k)


def tech_extra_at(ind: pd.DataFrame, date, mkt_close: pd.Series = None) -> dict:
    """ind：winners_study.compute_indicators 的結果。回傳 TECH_COLS。"""
    out = {k: np.nan for k in TECH_COLS}
    if date not in ind.index:
        return out
    i = int(ind.index.get_loc(date))
    h = ind["high"].to_numpy(float)[:i + 1]
    l = ind["low"].to_numpy(float)[:i + 1]
    c = ind["close"].to_numpy(float)[:i + 1]
    v = ind["volume"].to_numpy(float)[:i + 1]
    n = len(c)

    def ma(k, end=n):
        return c[end - k:end].mean() if end >= k else np.nan
    ma5, ma20, ma60, ma120 = ma(5), ma(20), ma(60), ma(120)
    if np.isfinite(ma120):
        out["bull_align"] = float(ma5 > ma20 > ma60 > ma120)
    if n >= 80:
        out["ma60_slope"] = ma60 / ma(60, n - 20) - 1
    if n >= 140:
        out["ma120_slope"] = ma120 / ma(120, n - 20) - 1
    if np.isfinite(ma60):
        out["bias60"] = c[-1] / ma60 - 1
        out["range60"] = (h[-60:].max() - l[-60:].min()) / c[-1]
    if n >= 20 + 249:
        bbw = []
        for e in range(n - 249, n + 1):
            w = c[e - 20:e]
            bbw.append(4 * w.std(ddof=0) / w.mean())
        bbw = np.asarray(bbw)
        out["bbw_pct"] = float((bbw <= bbw[-1]).mean() * 100)
    atr = ind["atr14"].iloc[i]
    if np.isfinite(atr):
        out["atr_pct"] = atr / c[-1] * 100
    out["rsi14"] = _rsi(c[-250:])
    out["k9"] = _kd_k(h[-120:], l[-120:], c[-120:])
    if n >= 252:
        w = c[-252:]
        out["dist_52w_low"] = c[-1] / w.min() - 1
        last_max = len(w) - 1 - int(np.argmax(w[::-1]))
        out["days_since_high"] = float(len(w) - 1 - last_max)
        out["max_dd_1y"] = float((w / np.maximum.accumulate(w) - 1).min())
    if n >= 65:
        out["vol_burst"] = v[-5:].mean() / v[-60:].mean() if v[-60:].mean() > 0 else np.nan
        out["gap_ups_60"] = float(np.sum(l[-60:] > h[-61:-1]))
    if mkt_close is not None and len(mkt_close):
        s = pd.Series(c, index=ind.index[:i + 1]).iloc[-251:]
        m = mkt_close.reindex(s.index)
        df = pd.concat([s / s.shift(1) - 1, m / m.shift(1) - 1], axis=1).dropna()
        if len(df) >= 120 and df.iloc[:, 1].var() > 0:
            out["beta"] = float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())
    return out


# ===========================================================================
# 籌碼特徵(起點)
# ===========================================================================
CHIP_COLS = ["foreign_ratio20", "trust_pct_shares20", "trust_ratio20", "trust_streak", "foreign_streak",
             "insti_ratio20", "margin_chg20", "short_margin_ratio", "foreign_pct", "foreign_pct_chg60",
             "pe", "pb", "dy", "pe_blank"]


def needed_dates(cal: pd.DatetimeIndex, d0) -> dict:
    i = int(cal.searchsorted(pd.Timestamp(d0)))
    i = min(i, len(cal) - 1)

    def at(k):
        return [cal[i - k]] if i - k >= 0 else []
    return {"insti": list(cal[max(0, i - 19):i + 1]), "margin": at(20) + [cal[i]], "qfii": at(60) + [cal[i]],
            "valuation": [cal[i]]}


def _stack(daily: dict) -> pd.DataFrame:
    frames = []
    for d, df in (daily or {}).items():
        if df is not None and len(df):
            x = df.copy()
            x["date"] = pd.Timestamp(d)
            frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def chip_features(code: str, market: str, ind: pd.DataFrame, chip: dict, loaded: dict, dates: dict) -> dict:
    """
    chip：{source: {date: DataFrame(code, …)}}(上市+上櫃合在一起)；
    loaded：{source: {(date, market)}} 哪些日期×市場確實抓到資料(抓到但這檔不在表裡=當天0買賣超)。
    20日買賣超需要至少15天有資料，否則空值。
    """
    out = {k: np.nan for k in CHIP_COLS}
    chip = chip or {}
    # 三大法人
    ins = chip.get("insti_by_code")
    days = [d for d in dates["insti"] if (d, market) in loaded.get("insti", set())]
    if ins is not None and len(days) >= 15:
        sub = ins.get(code)
        vals = {d: (0.0, 0.0, 0.0) for d in days}
        if sub is not None:
            for d, r in sub.iterrows():
                if d in vals:
                    vals[d] = (r["foreign_net"], r["trust_net"], r["total_net"])
        arr = np.array([vals[d] for d in sorted(days)], dtype=float)
        vol = ind["volume"].reindex(sorted(days)).to_numpy(float)
        vsum = np.nansum(vol)
        if vsum > 0:
            out["foreign_ratio20"] = np.nansum(arr[:, 0]) / vsum * 100
            out["trust_ratio20"] = np.nansum(arr[:, 1]) / vsum * 100
            out["insti_ratio20"] = np.nansum(arr[:, 2]) / vsum * 100

        def streak(col):
            k = 0
            for x in arr[::-1, col]:
                if x > 0:
                    k += 1
                else:
                    break
            return float(k)
        out["foreign_streak"] = streak(0)
        out["trust_streak"] = streak(1)
        q = _lookup(chip, "qfii", code, dates["qfii"][-1])
        if q is not None and np.isfinite(q.get("shares_issued", np.nan)) and q["shares_issued"] > 0:
            out["trust_pct_shares20"] = np.nansum(arr[:, 1]) / q["shares_issued"] * 100
    # 融資融券
    m_t = _lookup(chip, "margin", code, dates["margin"][-1])
    if m_t is not None:
        if m_t["margin_bal"] > 0:
            out["short_margin_ratio"] = m_t["short_bal"] / m_t["margin_bal"] * 100
        if len(dates["margin"]) == 2:
            m_0 = _lookup(chip, "margin", code, dates["margin"][0])
            if m_0 is not None and m_0["margin_bal"] > 0:
                out["margin_chg20"] = (m_t["margin_bal"] / m_0["margin_bal"] - 1) * 100
    # 外資持股
    q_t = _lookup(chip, "qfii", code, dates["qfii"][-1])
    if q_t is not None:
        out["foreign_pct"] = q_t["foreign_pct"]
        if len(dates["qfii"]) == 2:
            q_0 = _lookup(chip, "qfii", code, dates["qfii"][0])
            if q_0 is not None:
                out["foreign_pct_chg60"] = q_t["foreign_pct"] - q_0["foreign_pct"]
    # 估值
    v = _lookup(chip, "valuation", code, dates["valuation"][-1])
    if v is not None:
        pe = v["pe"]
        out["pe"] = pe if (np.isfinite(pe) and pe > 0) else np.nan
        out["pe_blank"] = float(not np.isfinite(out["pe"]))
        out["pb"] = v["pb"] if np.isfinite(v["pb"]) and v["pb"] > 0 else np.nan
        out["dy"] = v["dy"]
    return out


def _lookup(chip, source, code, date):
    idx = chip.get(f"{source}_idx")
    if idx is None:
        return None
    try:
        r = idx.loc[(pd.Timestamp(date), code)]
    except KeyError:
        return None
    if isinstance(r, pd.DataFrame):
        r = r.iloc[-1]
    return r.to_dict()


def prepare_chip(chip_daily: dict) -> dict:
    """{source: {date: df}} → 加上查詢用索引。"""
    out = dict(chip_daily or {})
    for src in ("margin", "qfii", "valuation"):
        st = _stack(out.get(src))
        if len(st):
            out[f"{src}_idx"] = st.set_index(["date", "code"]).sort_index()
    st = _stack(out.get("insti"))
    if len(st):
        out["insti_by_code"] = {c: g.set_index("date") for c, g in st.groupby("code")}
    return out


# ===========================================================================
# 雙條件 lift、洗牌基準
# ===========================================================================
def _lifts(C: np.ndarray, w: np.ndarray, n_c: np.ndarray):
    W = w.sum()
    base = W / len(w) if len(w) else np.nan
    wc = C.T @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        lift = (wc / n_c) / base if base > 0 else np.full(len(n_c), np.nan)
        cov = wc / W if W > 0 else np.full(len(n_c), np.nan)
    return lift, cov, wc


def select_top_singles(C, w, features, n_min=30, cov_min=0.10, k=12):
    """lift由高到低挑，每個特徵(feature)只取一個條件，最多k個。"""
    n_c = C.sum(axis=0)
    lift, cov, _ = _lifts(C, w, n_c)
    order = np.argsort(-np.nan_to_num(lift, nan=-1))
    chosen, used = [], set()
    for j in order:
        if n_c[j] < n_min or not np.isfinite(lift[j]) or not (cov[j] >= cov_min):
            continue
        if features[j] in used:
            continue
        chosen.append(int(j))
        used.add(features[j])
        if len(chosen) >= k:
            break
    return chosen


def pair_lifts(C, w, chosen, n_min=30):
    """回傳 [(i, j, n, winners, rate, lift, coverage)]，只留符合≥n_min檔的組合。"""
    W = w.sum()
    base = W / len(w) if len(w) else np.nan
    out = []
    for a in range(len(chosen)):
        for b in range(a + 1, len(chosen)):
            i, j = chosen[a], chosen[b]
            m = C[:, i] & C[:, j]
            n = int(m.sum())
            if n < n_min:
                continue
            wc = int(w[m].sum())
            rate = wc / n
            out.append((i, j, n, wc, rate, rate / base if base > 0 else np.nan, wc / W if W > 0 else np.nan))
    return out


def shuffle_baseline(C, w, features, n_min=30, n_perm=200, seed=42, cov_min=0.10, k=12):
    """
    把贏家標籤隨機打亂n_perm次(固定seed)：每次記錄「最大單一條件lift」與「挑前k名後最大雙條件lift」。
    回傳 dict(single_p95, pair_p95, single_max, pair_max 的陣列)。只看符合≥n_min檔的條件。
    """
    rng = np.random.default_rng(seed)
    n_c = C.sum(axis=0)
    ok = n_c >= n_min
    singles, pairs = [], []
    for _ in range(n_perm):
        wp = rng.permutation(w)
        lift, _, _ = _lifts(C, wp, n_c)
        singles.append(np.nanmax(np.where(ok, lift, np.nan)) if ok.any() else np.nan)
        ch = select_top_singles(C, wp, features, n_min, cov_min, k)
        pl = [p[5] for p in pair_lifts(C, wp, ch, n_min)]
        pairs.append(max(pl) if pl else np.nan)
    singles, pairs = np.asarray(singles, float), np.asarray(pairs, float)
    return {"single_p95": float(np.nanpercentile(singles, 95)) if np.isfinite(singles).any() else np.nan,
            "pair_p95": float(np.nanpercentile(pairs, 95)) if np.isfinite(pairs).any() else np.nan,
            "n_perm": n_perm, "seed": seed, "singles": singles, "pairs": pairs}


# ===========================================================================
# 下載(只在GitHub Actions上真正執行；測試把 _http 換掉)
# ===========================================================================
_REQUEST_COUNT = {"n": 0}


def _http(method: str, url: str, data=None):
    import requests
    _REQUEST_COUNT["n"] += 1
    kw = {"headers": HTTP_HEADERS, "timeout": HTTP_TIMEOUT}
    if method == "POST":
        return run_with_hard_timeout(requests.post, args=(url,), kwargs={**kw, "data": data}, timeout=HTTP_TIMEOUT + 10)
    return run_with_hard_timeout(requests.get, args=(url,), kwargs=kw, timeout=HTTP_TIMEOUT + 10)


def _decode(resp) -> str:
    raw = resp.content
    for enc in ("utf-8", "cp950"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _request(method, url, data=None, sleep=time.sleep):
    """回傳(text或None, 狀態)。連線失敗/逾時重試，HTTP非200不重試。"""
    status = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = _http(method, url, data)
        except (Exception, HardTimeout) as e:
            status = f"連線失敗:{str(e)[:60]}"
            sleep(REQUEST_DELAY * (attempt + 1))
            continue
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        return _decode(resp), "ok"
    return None, status


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _fmt_urls(source, market, d):
    d = pd.Timestamp(d)
    if market == "上市":
        return [TWSE_URLS[source].format(ymd=d.strftime("%Y%m%d"))]
    roc = f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"
    return [u.format(roc=roc, slash=d.strftime("%Y/%m/%d")) for u in TPEX_URLS[source]]


def estimate_requests(dates: dict, n_quarters: int) -> int:
    """最少請求數(每個日期/季×市場一次；失敗重試、TPEx/MOPS備援網址另計)。"""
    daily = sum(len(v) for v in dates.values()) * 2
    return daily + n_quarters * 2 * 2


def load_daily_sources(dates: dict, cache_dir: str, refresh: bool, debug_dir: str, today=None, sleep=time.sleep):
    """回傳 (chip_daily {source: {date: df}}, loaded {source: set((date, market))}, report list)。"""
    today = pd.Timestamp(today or datetime.date.today())
    cdir = os.path.join(cache_dir, "extra")
    os.makedirs(cdir, exist_ok=True)
    chip, loaded, report, sampled = {}, {}, [], set()
    for source, ds in dates.items():
        chip[source], loaded[source] = {}, set()
        for d in ds:
            d = pd.Timestamp(d)
            frames = []
            for market in ("上市", "上櫃"):
                mk = "sii" if market == "上市" else "otc"
                cp = os.path.join(cdir, f"{source}_{mk}_{d:%Y%m%d}.csv")
                if not refresh and os.path.exists(cp):
                    try:
                        df = pd.read_csv(cp, dtype={"code": str})
                        frames.append(df)
                        loaded[source].add((d, market))
                        report.append({"來源": SOURCE_NAMES[source], "市場": market, "日期": d.date(), "狀態": "快取",
                                       "列數": len(df), "解析方式": "", "網址": ""})
                        continue
                    except Exception:
                        pass
                df, status, method, used_url = None, "", "", ""
                for url in _fmt_urls(source, market, d):
                    used_url = url
                    text, status = _request("GET", url, sleep=sleep)
                    sleep(REQUEST_DELAY)
                    if text is None:
                        continue
                    try:
                        payload = json.loads(text)
                    except ValueError:
                        payload, status = None, "非JSON"
                    key = (source, market, payload is not None)
                    if key not in sampled:
                        sampled.add(key)
                        _write(os.path.join(debug_dir, f"extra_sample_{source}_{mk}.txt"), f"{url}\n{text[:5000]}")
                    if payload is None:
                        continue
                    df, method = PARSERS[source](payload, market)
                    if len(df):
                        status = "ok"
                        break
                    stat = str(payload.get("stat", "")) if isinstance(payload, dict) else ""
                    status = f"0列(stat={stat[:30]})"
                report.append({"來源": SOURCE_NAMES[source], "市場": market, "日期": d.date(), "狀態": status,
                               "列數": 0 if df is None else len(df), "解析方式": method, "網址": used_url})
                if df is not None and len(df):
                    frames.append(df)
                    loaded[source].add((d, market))
                    if d < today:
                        df.to_csv(cp, index=False)
            if frames:
                chip[source][d] = pd.concat(frames, ignore_index=True)
    return chip, loaded, report


def load_quarterly(quarters: list, cache_dir: str, refresh: bool, debug_dir: str, today=None, sleep=time.sleep):
    """回傳 (raw DataFrame[code, year, q, market, revenue…equity], report list)。"""
    today = pd.Timestamp(today or datetime.date.today())
    cdir = os.path.join(cache_dir, "extra")
    os.makedirs(cdir, exist_ok=True)
    per_kind = {k: [] for k in T163_COLS}
    report, sampled = [], set()
    for kind in T163_COLS:
        for (y, q) in quarters:
            for market, typek in (("上市", "sii"), ("上櫃", "otc")):
                cp = os.path.join(cdir, f"{kind}_{typek}_{y}Q{q}.csv")
                if not refresh and os.path.exists(cp):
                    try:
                        df = pd.read_csv(cp, dtype={"code": str})
                        df["year"], df["q"], df["market"] = y, q, market
                        per_kind[kind].append(df)
                        report.append({"來源": SOURCE_NAMES[kind], "市場": market, "季": f"{y}Q{q}", "狀態": "快取", "列數": len(df)})
                        continue
                    except Exception:
                        pass
                form = {"encodeURIComponent": "1", "step": "1", "firstin": "1", "off": "1", "isQuery": "Y",
                        "TYPEK": typek, "year": str(y - 1911), "season": f"{q:02d}"}
                df, status, url = None, "", ""
                for domain in MOPS_DOMAINS:
                    url = domain + MOPS_T163_PATH.format(kind=kind)
                    text, status = _request("POST", url, data=form, sleep=sleep)
                    sleep(REQUEST_DELAY)
                    if text is None:
                        continue
                    df, rep = parse_t163(text, kind)
                    key = (kind, typek, len(df) > 0)
                    if key not in sampled:
                        sampled.add(key)
                        _write(os.path.join(debug_dir, f"extra_sample_{kind}_{typek}.html"), f"<!-- {url} {form} -->\n{text[:20000]}")
                    if len(df):
                        status = "ok"
                        break
                    status = f"0列(表格{rep['tables']}個)"
                report.append({"來源": SOURCE_NAMES[kind], "市場": market, "季": f"{y}Q{q}", "狀態": status,
                               "列數": 0 if df is None else len(df), "網址": url})
                if df is not None and len(df):
                    if today > quarter_deadline(y, q) + pd.Timedelta(days=30):
                        df.to_csv(cp, index=False)
                    df = df.copy()
                    df["year"], df["q"], df["market"] = y, q, market
                    per_kind[kind].append(df)
    inc = pd.concat(per_kind["t163sb04"], ignore_index=True) if per_kind["t163sb04"] else pd.DataFrame()
    bal = pd.concat(per_kind["t163sb05"], ignore_index=True) if per_kind["t163sb05"] else pd.DataFrame()
    if inc.empty:
        return pd.DataFrame(), report
    if not bal.empty:
        inc = inc.merge(bal[["code", "year", "q", "equity"]], on=["code", "year", "q"], how="left")
    else:
        inc["equity"] = np.nan
    return inc, report


def source_status(report: list) -> list:
    """把逐日/逐季報告整理成每個來源一行：(名稱, 成功數, 要求數)。"""
    if not report:
        return []
    df = pd.DataFrame(report)
    out = []
    for name, g in df.groupby("來源", sort=False):
        ok = int(g["狀態"].isin(["ok", "快取"]).sum())
        out.append((name, ok, int(len(g))))
    return out


def load_all_extra(cal: pd.DatetimeIndex, d0, d1, cache_dir: str, refresh: bool, debug_dir: str):
    """main用：抓起點需要的籌碼/估值日期、與區間需要的季報。任何錯誤都吞掉(fail soft)。"""
    t0 = time.time()
    _REQUEST_COUNT["n"] = 0
    dates = needed_dates(cal, d0)
    q_first = shift_quarter(latest_published_quarter(d0), -QUARTERS_BACK)
    q_last = latest_published_quarter(d1)
    quarters = quarters_between(q_first, q_last)
    est = estimate_requests(dates, len(quarters))
    print(f"籌碼/估值/季報：{sum(len(v) for v in dates.values())} 個日期、{len(quarters)} 季，"
          f"預估至少 {est} 次請求(不含快取/重試/備援網址)，每次間隔{REQUEST_DELAY:.0f}秒", flush=True)
    result = {"chip_daily": {}, "loaded": {}, "quarterly_raw": pd.DataFrame(), "report": [], "errors": [],
              "dates": dates, "quarters": quarters, "estimated_requests": est}
    try:
        chip, loaded, rep = load_daily_sources(dates, cache_dir, refresh, debug_dir)
        result.update({"chip_daily": chip, "loaded": loaded})
        result["report"] += rep
    except Exception as e:
        result["errors"].append(f"籌碼/估值下載發生例外：{str(e)[:200]}")
    try:
        raw, rep = load_quarterly(quarters, cache_dir, refresh, debug_dir)
        result["quarterly_raw"] = raw
        result["report"] += rep
    except Exception as e:
        result["errors"].append(f"季財報下載發生例外：{str(e)[:200]}")
    result["actual_requests"] = _REQUEST_COUNT["n"]
    result["seconds"] = time.time() - t0
    os.makedirs(debug_dir, exist_ok=True)
    if result["report"]:
        pd.DataFrame(result["report"]).to_csv(os.path.join(debug_dir, "extra_report.csv"), index=False, encoding="utf-8-sig")
    print(f"籌碼/估值/季報完成：實際 {result['actual_requests']} 次請求，{result['seconds']:.0f} 秒", flush=True)
    return result
