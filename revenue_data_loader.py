"""
基本面資料：每月營收(月增率/年增率)。

現股波段策略([[equity_swing_engine]])用的三個新資料來源之一，信心程度：中等
(見下方「已知限制」)。

資料來源：公開資訊觀測站(MOPS)「上市/上櫃公司每月營業收入彙總表」——
https://mops.twse.com.tw/nas/t21/sii/t21sc03_{民國年}_{月}_0.html (上市)
https://mops.twse.com.tw/nas/t21/otc/t21sc04_{民國年}_{月}_0.html (上櫃)
這是HTML表格(不是JSON API)，用 requests 抓回HTML原始碼、再用 pandas.read_html()
解析表格內容，不是逐股查詢，是「一次拿到當月全部公司」的彙總表，所以用「逐月」
迴圈抓取，不是逐股迴圈——跟 chip_data_loader.py 的「逐日」抓T86是同一種設計精神
(一次請求涵蓋全市場，減少請求總數)。

已知限制(誠實列出，這一版都還沒有機會用GitHub Actions真實環境驗證，因為這個
sandbox對twse.com.tw/mops.twse.com.tw的對外連線被proxy allowlist擋掉)：
1. URL裡的「民國年」用西元年-1911換算，月份用整數(不補0)，這是根據公開資料社群
   常見引用的格式推斷，不是已經實測確認過的格式——第一次真正在GitHub Actions
   執行這支程式，才是這個URL格式對不對的真正驗證。
2. 每月營收公告日期通常落後所屬月份約10天(例如2026年3月的營收約在2026年4月10日
   前後公告)。回測比對訊號時間點時，一定要用「公告日」而不是「所屬月份」去對齊
   股價資料，否則會用到當時market還看不到的未來資料(look-ahead bias)。這支模組
   回傳的DataFrame index就是「近似公告日」(所屬月份的下個月10日)，不是所屬月份
   本身，呼叫端(equity_swing_engine.py)不需要再自己處理這個時間差。
3. 彙總表本身通常已經直接附「去年同月增減(%)」「上月比較增減(%)」這兩欄百分比
   字串(例如"12.34"或"-5.67")，這裡直接解析這兩欄字串轉成浮點數，不是拿原始營收
   金額重新計算——原因：直接採用彙總表自己算好、公司自己證實過的百分比，比自己
   重新計算更貼近官方公告的真實數字(避免計算基期、四捨五入方式跟官方不一致)。
4. 表格欄位名稱/欄位順序可能隨年份調整過(這份報表存在很多年，格式演變的可能性
   跟 chip_data_loader.py 的 T86 欄位演變是同一種風險)，這裡一律用「欄位名稱
   包含特定子字串」動態比對，不依賴固定欄位位置，降低格式微調造成整批解析失敗
   的風險，但沒辦法完全排除表格結構大幅改版(例如整個從單一表格改成分頁)的可能。
5. 停業/下市/當月剛好沒有公告的公司，該月不會出現在彙總表裡，屬於正常情況，
   不是抓取失敗。
"""
import os
import time
import datetime
import io
import requests
import pandas as pd

from network_utils import run_with_hard_timeout, HardTimeout

REVENUE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "revenue_cache")
HARD_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3
REQUEST_DELAY_SECONDS = 2.0  # 兩次請求(兩個月份/兩個市場)之間的間隔
THREAD_HARD_TIMEOUT_SECONDS = HARD_TIMEOUT_SECONDS + 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

MARKET_URL_TEMPLATE = {
    "sii": "https://mops.twse.com.tw/nas/t21/sii/t21sc03_{roc_year}_{month}_0.html",  # 上市
    "otc": "https://mops.twse.com.tw/nas/t21/otc/t21sc04_{roc_year}_{month}_0.html",  # 上櫃
}

CODE_COL_CANDIDATES = ("公司代號", "公司 代號", "代號")
YOY_COL_CANDIDATES = ("去年同月增減", "去年同月增減(%)")
MOM_COL_CANDIDATES = ("上月比較增減", "上月比較增減(%)")


class FetchFailed(Exception):
    """代表這次請求本身失敗(逾時/連線錯誤/HTML解析不出任何表格)，
    跟「查到這個月確實還沒公告」要分開處理，前者該重試，後者不該重試。"""
    pass


def _parse_pct(s) -> float:
    """把彙總表裡的百分比字串(可能含逗號、正負號、或空白代表無法計算)轉成浮點數，
    轉不出來(例如公司上個月沒有營收基期，欄位是空白或"--")回傳NaN，不是0——
    0代表「真的沒有增減」，NaN代表「這個月沒辦法算這個比較基準」，是兩件不同的事。"""
    if s is None:
        return float("nan")
    s = str(s).strip().replace(",", "")
    if s in ("", "--", "-", "N/A", "nan"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _find_col(columns, candidates):
    for col in columns:
        col_str = str(col).replace(" ", "")
        for cand in candidates:
            if cand.replace(" ", "") in col_str:
                return col
    return None


def fetch_revenue_month(year: int, month: int, market: str = "sii"):
    """
    抓取單一「西元年+月」、單一市場(sii=上市/otc=上櫃)的當月營收彙總表。
    回傳 dict {code: {"revenue_yoy_pct": float, "revenue_mom_pct": float}} 代表有資料；
    回傳 None 代表「確認這個月這個市場還沒有彙總表可查」(通常是月份還沒到公告時間，
    或月份本身就不合理，例如查到未來的月份)；request本身失敗或完全解析不出表格時
    丟出 FetchFailed，由呼叫端決定要不要重試。
    """
    roc_year = year - 1911
    url = MARKET_URL_TEMPLATE[market].format(roc_year=roc_year, month=month)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
        if resp.status_code == 404:
            return None  # 這個月份確認查無彙總表(通常是還沒公告)
        resp.encoding = resp.encoding or "utf-8"
        html_text = resp.text
    except Exception as e:
        raise FetchFailed(str(e))

    try:
        tables = pd.read_html(io.StringIO(html_text))
    except Exception:
        # 查無表格(常見於MOPS對「還沒到公告時間的月份」回傳一個沒有資料表的頁面)，
        # 視為「確認無資料」而不是失敗——跟T86的stat!=OK是同一種處理精神。
        return None

    target_df = None
    for tbl in tables:
        cols = list(tbl.columns)
        if _find_col(cols, CODE_COL_CANDIDATES) is not None:
            target_df = tbl
            break

    if target_df is None:
        return None  # 頁面存在但完全找不到預期欄位，視為這個月無有效資料

    code_col = _find_col(target_df.columns, CODE_COL_CANDIDATES)
    yoy_col = _find_col(target_df.columns, YOY_COL_CANDIDATES)
    mom_col = _find_col(target_df.columns, MOM_COL_CANDIDATES)

    result = {}
    for _, row in target_df.iterrows():
        code = str(row[code_col]).strip()
        if not code.isdigit():
            continue  # 跳過表格裡的小計/合計列等非股票代碼列
        result[code] = {
            "revenue_yoy_pct": _parse_pct(row[yoy_col]) if yoy_col is not None else float("nan"),
            "revenue_mom_pct": _parse_pct(row[mom_col]) if mom_col is not None else float("nan"),
        }
    if not result:
        return None
    return result


def _fetch_one_month_with_retry(year: int, month: int, market: str):
    for attempt in range(MAX_RETRIES + 1):
        try:
            data = run_with_hard_timeout(
                fetch_revenue_month, args=(year, month, market), timeout=THREAD_HARD_TIMEOUT_SECONDS)
            return data, True
        except (FetchFailed, HardTimeout):
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            return None, False
    return None, False


def _months_between(start: str, end: str) -> list:
    """列出start~end(YYYY-MM-DD)之間所有(year, month)，含頭尾月份。"""
    start_d = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    months = []
    y, m = start_d.year, start_d.month
    while (y, m) <= (end_d.year, end_d.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def load_revenue_data(start: str, end: str, universe_codes: set = None, refresh: bool = False,
                       markets: tuple = ("sii", "otc")) -> dict:
    """
    回傳 {code: DataFrame(index=announcement_date, columns=[revenue_yoy_pct, revenue_mom_pct])}。
    announcement_date 是「所屬月份的次月10日」的近似公告日(見模組docstring第2點)，
    嚴格早於這個日期的股價資料才看得到這筆營收，避免回測用到未來資料。

    每個月每個市場的原始資料快取成一個小檔案，重複執行不會重新打MOPS。
    universe_codes: 只保留這個集合裡的股票代碼(建議傳入 taifex_universe.STOCK_FUTURES_UNIVERSE 的 keys())。
    """
    os.makedirs(REVENUE_CACHE_DIR, exist_ok=True)
    months = _months_between(start, end)

    NO_DATA_MARKER_COLS = ["code", "revenue_yoy_pct", "revenue_mom_pct"]
    to_fetch = []
    for (y, m) in months:
        for market in markets:
            cache_path = os.path.join(REVENUE_CACHE_DIR, f"rev_{market}_{y}{m:02d}.csv")
            if not refresh and os.path.exists(cache_path):
                continue
            to_fetch.append((y, m, market))

    print(f"每月營收資料：{len(months)} 個月 x {len(markets)} 個市場，{len(to_fetch)} 筆需要下載"
          f"(其餘已有快取)，序列處理中(每次間隔{REQUEST_DELAY_SECONDS}秒) ...", flush=True)

    fetched_count = no_data_count = failed_count = 0
    for done_n, (y, m, market) in enumerate(to_fetch, start=1):
        data, success = _fetch_one_month_with_retry(y, m, market)
        cache_path = os.path.join(REVENUE_CACHE_DIR, f"rev_{market}_{y}{m:02d}.csv")

        if not success:
            failed_count += 1  # 留給下次重跑補抓，不寫快取
        elif data is None:
            no_data_count += 1
            pd.DataFrame(columns=NO_DATA_MARKER_COLS).to_csv(cache_path, index=False)
        else:
            fetched_count += 1
            rows = [{"code": c, **v} for c, v in data.items()]
            pd.DataFrame(rows).to_csv(cache_path, index=False)

        if done_n % 10 == 0 or done_n == len(to_fetch):
            print(f"  進度 {done_n}/{len(to_fetch)} "
                  f"(成功{fetched_count} 無資料{no_data_count} 失敗待補{failed_count}) ...", flush=True)
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"每月營收資料下載完成：本次新抓{fetched_count}筆，確認無資料{no_data_count}筆，"
          f"逾時待補{failed_count}筆{'（下次重跑會自動補抓）' if failed_count else ''}", flush=True)

    per_stock_records = {}
    for (y, m) in months:
        # 公告日近似：所屬月份的次月10日(見模組docstring第2點)
        ann_year, ann_month = (y + 1, 1) if m == 12 else (y, m + 1)
        ann_date = pd.Timestamp(datetime.date(ann_year, ann_month, 10))
        for market in markets:
            cache_path = os.path.join(REVENUE_CACHE_DIR, f"rev_{market}_{y}{m:02d}.csv")
            if not os.path.exists(cache_path):
                continue
            try:
                month_df = pd.read_csv(cache_path, dtype={"code": str})
            except Exception:
                continue
            if month_df.empty:
                continue
            for _, row in month_df.iterrows():
                code = str(row["code"])
                if universe_codes is not None and code not in universe_codes:
                    continue
                per_stock_records.setdefault(code, []).append({
                    "date": ann_date,
                    "revenue_yoy_pct": row["revenue_yoy_pct"],
                    "revenue_mom_pct": row["revenue_mom_pct"],
                })

    revenue_data = {}
    for code, records in per_stock_records.items():
        df = pd.DataFrame(records).set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        revenue_data[code] = df
    return revenue_data
