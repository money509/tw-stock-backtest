"""
validate_tx_history_accuracy.py
====================================
在真的拿taifex_history_loader.py(非官方、社群維護、crazyindicator.pixnet.net上的
Google Drive歷史資料)去跑任何回測之前，**先確認拿證交所給的日線去對比一下**
(使用者原話)——這支腳本就是做這件事，獨立CLI，不是taifex_history_loader.py的
一部分，因為它牽涉另一個完全獨立的資料來源(TAIFEX官方日線/data.gov.tw開放資料
備援)，混在loader模組裡會讓loader背負不該背的「還要懂怎麼抓官方資料」的責任。

⚠️⚠️ 跟repo其他所有loader一樣的誠實聲明：這個sandbox連taifex.com.tw/data.gov.tw
都連不到(proxy allowlist擋掉)，下面對TAIFEX「期貨每日交易行情下載」
(dlFutDailyMarketView)頁面的欄位名稱/查詢方式**全部是根據其他TAIFEX「資料下載」
頁面常見模式的推斷**，沒有機會實際看過這個頁面的表單欄位長什麼樣子——**第一次
在GitHub Actions真實環境執行，才是這個假設第一次被真正驗證**，失敗時會印出
明確的診斷(包含實際送出的查詢參數)，方便事後比對真實回應來源修正欄位名稱，
不是crash也不是假裝成功。

跟dlFutPrevious30DaysSalesData(taifex_intraday_loader.py用的那個、只有滾動30天
窗口)不同，dlFutDailyMarketView預期是「每日收盤/結算行情」報表，照TAIFEX其他
「資料下載」分類頁面的慣例，這類日報表通常支援帶起訖日期查詢(不像前者只能拿
「查詢當下」往回算30天的窗口)——但這同樣是根據命名與頁面歸類的推斷，不是已經
看過真實查詢表單欄位名稱後的結論。

驗證範圍(使用者沒有要求比對全部20多年歷史，這裡刻意只抽樣，見下方)：
1. 預設只驗證**2011_2020跟2021_2023**兩段(DEFAULT_PERIODS)，不含1998_2000/
   2001_2010——刻意選擇，不是遺漏：越早期的合約規格/交易時段/漲跌停制度跟現在
   差異越大，對「現代回測」的參考價值較低，而且驗證全部20多年要抓的官方比對
   資料量會大很多、拉長這支腳本的執行時間，這裡的目標是「先抓出這份社群資料
   跟官方口徑差多少」的概括判斷，不是逐年全面稽核，所以保守先驗證最近兩段
   (涵蓋2011~2023，對現代回測最有參考價值)就夠；需要驗證更早期資料時，可以用
   --periods加回1998_2000/2001_2010重跑。
2. 不會抓官方日線的「全部」歷史(數千個交易日)去逐一比對——那樣對一支「先抓個概括
   印象」的驗證腳本而言太誇張，這裡只抽樣DEFAULT_N_SAMPLE_DATES(預設30)天，
   平均分散在可比對的日期範圍裡(見select_sample_dates())，而不是只挑最前面/
   最後面幾天(那樣會漏掉中間年份可能存在的系統性落差)。

比對邏輯：
  1. 用taifex_history_loader.load_tx_history_bars()取得社群1分鐘K棒(bar_minutes=1，
     盡量保留原始細度)，直接在這裡(不透過loader)聚合成「每個日曆交易日」的
     OHLC(Open=當天第一根Open/High=當天最高/Low=當天最低/Close=當天最後一根Close)。
     只有tick-level來源才需要loader先挑近月合約(bar-level來源則假設Google Drive
     這份資料集本身已經是單一合約/連續的每日序列，沒有另外的合約欄位可以挑選——
     這點**沒有實測驗證過**，如果GitHub Actions真的跑出來發現同一天有重複/矛盾
     的K棒，很可能就是這個假設錯誤，代表bar-level來源其實混了多個合約)。
  2. 從聚合出來的交易日裡，挑DEFAULT_N_SAMPLE_DATES天出來(select_sample_dates())。
  3. fetch_official_daily_ohlc()嘗試抓這些日期對應的官方日線(先試TAIFEX端點，
     失敗再試data.gov.tw備援，兩者都是(data, reason)防禦性寫法，見下方)。
  4. 官方資料抓不到時：**明確印出「驗證沒有執行」，不是假裝比對過、也不是
     假裝資料是準的**，不產生看起來像結論但其實沒比對過的報告。
  5. 兩邊都有資料的日期，逐一計算Close(以及Open/High/Low，缺值時個別跳過)的
     絕對誤差跟百分比誤差，統計平均/中位數/最大值，以及「在TOLERANCE_PCT
     (預設0.1%)容忍範圍內」的天數比例，印出結論(中文，依實際數字生成，不是
     寫死的樂觀結論)。

輸出：
  - console列印完整比對結果
  - results_tx_history_validation/validation_report.txt(同樣內容存檔)
"""
import argparse
import datetime
import os
import re
import sys

import pandas as pd
import requests

from network_utils import run_with_hard_timeout, HardTimeout
from taifex_history_loader import load_tx_history_bars

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_tx_history_validation")
DEFAULT_PERIODS = ["2011_2020", "2021_2023"]
DEFAULT_N_SAMPLE_DATES = 30
DEFAULT_TOLERANCE_PCT = 0.1  # 百分比(0.1代表0.1%)

TAIFEX_BASE = "https://www.taifex.com.tw"
DAILY_VIEW_PATH = "/cht/3/dlFutDailyMarketView"
DATA_GOV_DATASET_ID = "11319"
DATA_GOV_API = f"https://data.gov.tw/api/v2/rest/dataset/{DATA_GOV_DATASET_ID}"

HARD_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

DISCLAIMER_HEADER = (
    "⚠️⚠️ 社群版台指期1分鐘K棒(crazyindicator.pixnet.net/Google Drive) vs "
    "TAIFEX官方日線 — 準確度驗證(抽樣比對，不是逐日全面稽核) ⚠️⚠️"
)

DATE_COL_CANDS = ("交易日期", "日期")
OPEN_COL_CANDS = ("開盤價", "開盤")
HIGH_COL_CANDS = ("最高價", "最高")
LOW_COL_CANDS = ("最低價", "最低")
CLOSE_COL_CANDS = ("收盤價", "結算價", "收盤")
PRODUCT_COL_CANDS = ("商品代號", "契約代號", "商品名稱")
CONTRACT_MONTH_COL_CANDS = ("到期月份(週別)", "到期月份", "契約月份")


class FetchFailed(Exception):
    """官方日線抓取連線層級失敗(逾時/連線錯誤)，由呼叫端決定要不要重試，跟
    「抓到了但格式不是預期的」(reason字串)是兩件不同的事，跟repo其他loader
    同樣的區分精神。"""
    pass


def _find_col(columns, candidates):
    for col in columns:
        col_str = str(col).replace(" ", "")
        for cand in candidates:
            if cand.replace(" ", "") in col_str:
                return col
    return None


def select_sample_dates(available_dates, n=DEFAULT_N_SAMPLE_DATES):
    """從available_dates(已排序的date物件list/Index)裡，平均分散挑出最多n個日期
    (不是只取最前面/最後面)。available_dates數量 <= n時，全部回傳。"""
    dates = sorted(available_dates)
    if len(dates) <= n:
        return dates
    if n <= 1:
        return dates[:1]
    step = (len(dates) - 1) / (n - 1)
    indices = sorted({round(i * step) for i in range(n)})
    return [dates[i] for i in indices]


def aggregate_minute_bars_to_daily(bars: pd.DataFrame):
    """把1分鐘(或任意細度)K棒DataFrame(index=時間戳記，欄位Open/High/Low/Close/
    Volume)聚合成「每個日曆交易日」的OHLC：Open=當天第一根Open/High=當天最高/
    Low=當天最低/Close=當天最後一根Close。回傳index是date物件的DataFrame，
    欄位Open/High/Low/Close。

    假設bars已經是單一合約(前面load_tx_history_bars()/taifex_history_loader.py
    的tick-level路徑已經挑過近月合約；bar-level路徑則假設來源資料集本身就是
    單一合約序列，見模組docstring第1點的誠實聲明)。
    """
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    sorted_bars = bars.sort_index()
    grouped = sorted_bars.groupby(sorted_bars.index.date)
    daily = grouped.agg(Open=("Open", "first"), High=("High", "max"),
                         Low=("Low", "min"), Close=("Close", "last"))
    daily.index.name = "Date"
    return daily


def _parse_official_table(raw_bytes):
    """把官方端點回傳的原始內容(猜測是CSV/HTML表格)解析成統一欄位
    (Date/Open/High/Low/Close)的DataFrame，回傳(df_or_None, reason)。跟
    taifex_intraday_loader.py的parse_tick_csv()同樣的「動態比對欄位名稱」精神。
    """
    df = None
    for encoding in ("utf-8", "big5", "cp950"):
        try:
            import io
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, dtype=str)
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        try:
            import io
            tables = pd.read_html(io.BytesIO(raw_bytes))
            df = next((t for t in tables if t.shape[1] >= 4), None)
            if df is not None:
                df = df.astype(str)
        except Exception:
            df = None
    if df is None or df.empty:
        return None, "no_table_parsed"

    date_col = _find_col(df.columns, DATE_COL_CANDS)
    open_col = _find_col(df.columns, OPEN_COL_CANDS)
    high_col = _find_col(df.columns, HIGH_COL_CANDS)
    low_col = _find_col(df.columns, LOW_COL_CANDS)
    close_col = _find_col(df.columns, CLOSE_COL_CANDS)
    product_col = _find_col(df.columns, PRODUCT_COL_CANDS)
    month_col = _find_col(df.columns, CONTRACT_MONTH_COL_CANDS)

    missing = [name for name, col in [
        ("date", date_col), ("close", close_col)] if col is None]
    if missing:
        print(f"[validate_tx_history_accuracy] 官方日線解析：缺少必要欄位{missing}，"
              f"目前解析出的欄位名稱={list(df.columns)}(供之後修正欄位名稱candidates用)")
        return None, "missing_columns:" + ",".join(missing)

    out = pd.DataFrame({"_date_raw": df[date_col].astype(str).str.strip()})
    out["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    out["Open"] = pd.to_numeric(df[open_col], errors="coerce") if open_col else float("nan")
    out["High"] = pd.to_numeric(df[high_col], errors="coerce") if high_col else float("nan")
    out["Low"] = pd.to_numeric(df[low_col], errors="coerce") if low_col else float("nan")
    if product_col is not None:
        out["ProductCode"] = df[product_col].astype(str).str.strip()
    if month_col is not None:
        out["ContractMonth"] = df[month_col].astype(str).str.strip()

    # 近月合約篩選：如果解析出合約月份欄，挑成交天數最多的月份代碼當近月(同
    # taifex_intraday_loader.py select_front_month()同樣的捷徑精神)，避免同一天
    # 出現多個合約月份的重複列造成比對時誤判。
    if "ContractMonth" in out.columns and out["ContractMonth"].nunique() > 1:
        front_month = out["ContractMonth"].value_counts().index[0]
        out = out[out["ContractMonth"] == front_month]

    date_parsed = pd.to_datetime(out["_date_raw"], errors="coerce")
    if date_parsed.isna().mean() > 0.5:
        date_parsed = pd.to_datetime(out["_date_raw"], format="%Y%m%d", errors="coerce")
    out["Date"] = date_parsed.dt.date
    out = out.dropna(subset=["Date", "Close"])
    if out.empty:
        return None, "empty_after_parse"
    out = out.drop_duplicates(subset=["Date"], keep="first").set_index("Date")
    return out[["Open", "High", "Low", "Close"]], "ok"


def _try_taifex_daily_endpoint(start_date, end_date, commodity_id="TXF"):
    """嘗試TAIFEX「期貨每日交易行情下載」(猜測端點，見模組docstring)。跟
    taifex_intraday_loader.py的_try_post_guess()同樣精神：對同一頁面路徑送POST，
    帶起訖日期跟商品代號欄位——**欄位名稱是猜測，沒有實測驗證過**，失敗時會印出
    實際送出的欄位，方便事後對照真實回應修正。回傳(content_bytes_or_None, reason)。
    """
    url = TAIFEX_BASE + DAILY_VIEW_PATH
    payload = {
        "firstDate": start_date.strftime("%Y/%m/%d"),
        "lastDate": end_date.strftime("%Y/%m/%d"),
        "queryStartDate": start_date.strftime("%Y/%m/%d"),
        "queryEndDate": end_date.strftime("%Y/%m/%d"),
        "commodity_id": commodity_id,
        "queryType": "2",
        "download": "csv",
    }
    try:
        resp = requests.post(url, data=payload, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
    except Exception as e:
        raise FetchFailed(str(e))
    if resp.status_code != 200:
        print(f"[validate_tx_history_accuracy] TAIFEX日線端點：HTTP {resp.status_code}，"
              f"送出的查詢參數={payload}(欄位名稱是猜測，供之後對照真實表單修正)")
        return None, f"taifex_daily_http_{resp.status_code}"
    content_type = resp.headers.get("Content-Type", "")
    text_head = resp.content[:200].decode("utf-8", errors="ignore").lstrip().lower()
    if "text/html" in content_type or text_head.startswith("<!doctype") or text_head.startswith("<html"):
        print(f"[validate_tx_history_accuracy] TAIFEX日線端點：回傳的是網頁本身不是資料，"
              f"代表猜測的欄位名稱/POST方式錯了。送出的查詢參數={payload}")
        return None, "taifex_daily_returned_html_not_data"
    return resp.content, "ok"


def _try_data_gov_fallback(start_date, end_date, commodity_id="TXF"):
    """備援來源：政府開放資料平台(data.gov.tw)「期貨每日交易行情」資料集，見模組
    docstring。做法：先打dataset API拿資源(resource)清單，找看起來像CSV下載連結的
    那個，再下載該連結內容——跟taifex_intraday_loader.py的_try_scrape_download_link()
    「先找連結再下載」同樣精神，因為data.gov.tw資源的實際下載網址會隨資料集/資源
    改版變動，不寫死。回傳(content_bytes_or_None, reason)。"""
    try:
        resp = requests.get(DATA_GOV_API, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
    except Exception as e:
        raise FetchFailed(str(e))
    if resp.status_code != 200:
        return None, f"data_gov_api_http_{resp.status_code}"
    try:
        payload = resp.json()
    except Exception:
        return None, "data_gov_api_not_json"

    resource_urls = re.findall(r'"(https?://[^"]+\.csv[^"]*)"', resp.text, flags=re.IGNORECASE)
    if not resource_urls:
        print(f"[validate_tx_history_accuracy] data.gov.tw備援：dataset API回應裡找不到csv資源連結，"
              f"回應頂層欄位={list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
              f"(供之後對照真實回應結構修正)")
        return None, "data_gov_no_csv_resource"

    download_url = resource_urls[0]
    try:
        dl_resp = requests.get(download_url, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
    except Exception as e:
        raise FetchFailed(str(e))
    if dl_resp.status_code != 200:
        return None, f"data_gov_download_http_{dl_resp.status_code}"
    return dl_resp.content, "ok"


def fetch_official_daily_ohlc(start_date, end_date, commodity_id="TXF"):
    """
    整合入口：依序嘗試TAIFEX官方端點(主要) -> data.gov.tw開放資料(備援)，兩者都是
    防禦性(data, reason)寫法。回傳(df_or_None, reason)，df的index是date物件，
    欄位Open/High/Low/Close：
      reason == "ok"
      reason == "taifex_daily_..."  TAIFEX端點失敗的具體原因
      reason == "data_gov_..."      備援也失敗的具體原因(兩者都失敗時，reason是
                                     備援的原因，因為它是最後執行的)
    連線層級失敗(兩個來源都是)時重試MAX_RETRIES次，重試後仍失敗才真正放棄。
    """
    for attempt in range(MAX_RETRIES + 1):
        last_reason = None
        try:
            content, reason = run_with_hard_timeout(
                _try_taifex_daily_endpoint, args=(start_date, end_date, commodity_id),
                timeout=HARD_TIMEOUT_SECONDS + 10)
            if reason == "ok":
                df, parse_reason = _parse_official_table(content)
                if parse_reason == "ok":
                    return df, "ok"
                last_reason = "taifex_daily_" + parse_reason
            else:
                last_reason = reason
        except (FetchFailed, HardTimeout):
            last_reason = "taifex_daily_connection_error"

        try:
            content, reason = run_with_hard_timeout(
                _try_data_gov_fallback, args=(start_date, end_date, commodity_id),
                timeout=HARD_TIMEOUT_SECONDS + 10)
            if reason == "ok":
                df, parse_reason = _parse_official_table(content)
                if parse_reason == "ok":
                    return df, "ok"
                last_reason = "data_gov_" + parse_reason
            else:
                last_reason = reason
        except (FetchFailed, HardTimeout):
            last_reason = "data_gov_connection_error"

        if attempt < MAX_RETRIES:
            import time
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue
        return None, last_reason
    return None, "unknown_failure"


def compare_daily_ohlc(community_daily, official_daily, tolerance_pct=DEFAULT_TOLERANCE_PCT):
    """
    比對社群資料(community_daily)跟官方資料(official_daily)，兩者皆為index=date、
    欄位Open/High/Low/Close的DataFrame。只比對兩邊都有的日期(inner join)。

    回傳(comparison_df, stats: dict)：
      comparison_df: 每個重疊日期一列，欄位 CommunityClose/OfficialClose/
        AbsErrorClose/PctErrorClose(以及Open/High/Low的同樣欄位，缺值時該欄NaN)
      stats: {
        "n_dates_compared": int,
        "n_within_tolerance": int,  # Close誤差百分比 <= tolerance_pct 的天數
        "pct_within_tolerance": float,
        "mean_abs_error_close": float, "median_abs_error_close": float,
        "max_abs_error_close": float,
        "mean_pct_error_close": float, "median_pct_error_close": float,
        "max_pct_error_close": float,
        "tolerance_pct": tolerance_pct,
      }
    comparison_df為空(沒有重疊日期)時，stats的數值欄位一律是None，
    n_dates_compared=0。
    """
    merged = community_daily.join(official_daily, how="inner", lsuffix="_community", rsuffix="_official")
    if merged.empty:
        return merged, {
            "n_dates_compared": 0, "n_within_tolerance": 0, "pct_within_tolerance": None,
            "mean_abs_error_close": None, "median_abs_error_close": None, "max_abs_error_close": None,
            "mean_pct_error_close": None, "median_pct_error_close": None, "max_pct_error_close": None,
            "tolerance_pct": tolerance_pct,
        }

    out = pd.DataFrame(index=merged.index)
    for field in ("Open", "High", "Low", "Close"):
        community_col = f"{field}_community"
        official_col = f"{field}_official"
        out[f"Community{field}"] = merged[community_col]
        out[f"Official{field}"] = merged[official_col]
        abs_err = (merged[community_col] - merged[official_col]).abs()
        pct_err = (abs_err / merged[official_col].abs()) * 100.0
        out[f"AbsError{field}"] = abs_err
        out[f"PctError{field}"] = pct_err

    close_abs = out["AbsErrorClose"].dropna()
    close_pct = out["PctErrorClose"].dropna()
    within_tolerance = close_pct <= tolerance_pct

    stats = {
        "n_dates_compared": int(len(out)),
        "n_within_tolerance": int(within_tolerance.sum()),
        "pct_within_tolerance": float(within_tolerance.mean() * 100.0) if len(close_pct) else None,
        "mean_abs_error_close": float(close_abs.mean()) if len(close_abs) else None,
        "median_abs_error_close": float(close_abs.median()) if len(close_abs) else None,
        "max_abs_error_close": float(close_abs.max()) if len(close_abs) else None,
        "mean_pct_error_close": float(close_pct.mean()) if len(close_pct) else None,
        "median_pct_error_close": float(close_pct.median()) if len(close_pct) else None,
        "max_pct_error_close": float(close_pct.max()) if len(close_pct) else None,
        "tolerance_pct": tolerance_pct,
    }
    return out, stats


def _verdict_sentence(stats):
    """依實際比對出來的數字生成中文結論句，不是寫死的樂觀結論(使用者明確要求)。"""
    if stats["n_dates_compared"] == 0:
        return "沒有任何重疊日期可以比對，驗證沒有執行，不能對這份社群資料的準確度下任何結論。"
    n = stats["n_dates_compared"]
    mean_pct = stats["mean_pct_error_close"]
    max_pct = stats["max_pct_error_close"]
    within = stats["pct_within_tolerance"]
    tol = stats["tolerance_pct"]
    verdict = (
        f"比對{n}天：收盤價平均誤差{mean_pct:.4f}%，最大誤差{max_pct:.4f}%，"
        f"{within:.1f}%的天數落在{tol}%容忍範圍內。"
    )
    if mean_pct <= tol and within >= 90:
        verdict += "數字上看起來跟官方日線相當接近，但這只是抽樣比對、不是逐日全面稽核，正式拿來回測前仍建議留意。"
    elif mean_pct <= tol * 3:
        verdict += "平均誤差不算大，但建議先看過comparison明細、確認沒有少數幾天誤差特別大的情況，再決定要不要信任這份資料。"
    else:
        verdict += "誤差明顯偏大，建議在信任這份資料前先看這個比對結果，不要直接拿去回測。"
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", nargs="+", default=DEFAULT_PERIODS,
                         help=f"要驗證的historical period key清單，預設{DEFAULT_PERIODS}"
                              "(刻意只驗證最近兩段，見模組docstring)")
    parser.add_argument("--n-sample-dates", type=int, default=DEFAULT_N_SAMPLE_DATES,
                         help="抽樣比對幾個交易日(平均分散，不是只取頭尾)")
    parser.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT,
                         help="收盤價誤差百分比容忍範圍，預設0.1(代表0.1%%)")
    parser.add_argument("--commodity-id", default="TXF")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "validation_report.txt")
    lines = [DISCLAIMER_HEADER, ""]
    printed_count = 0

    def _flush_new_lines():
        # 只印出自上次呼叫以來新增的行，避免跟最後寫檔前的內容重複印在console上
        # (console逐步印進度，檔案存完整版本，兩者不是同一份輸出，不該互相重印)。
        nonlocal printed_count
        for line in lines[printed_count:]:
            print(line, flush=True)
        printed_count = len(lines)

    def _write_report():
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    lines.append(f"驗證區間(periods)：{args.periods}(預設只含2011_2020/2021_2023，"
                 "理由見模組docstring)")
    _flush_new_lines()

    bars, diag = load_tx_history_bars(args.periods, bar_minutes=1, refresh=args.refresh_cache)
    if bars is None:
        lines += [
            "",
            "社群歷史資料下載/解析全部失敗，驗證沒有執行(不是資料準，是根本沒有資料可以比對)。",
            f"各period失敗原因：{diag.get('periods')}",
        ]
        _flush_new_lines()
        _write_report()
        print("社群歷史資料下載/解析全部失敗，驗證沒有執行。", file=sys.stderr, flush=True)
        sys.exit(0)  # 優雅結束：資料抓不到是預期中可能發生的情況，不是crash

    lines.append(f"社群資料下載/解析結果：{diag}")
    _flush_new_lines()

    daily = aggregate_minute_bars_to_daily(bars)
    lines.append(f"聚合成日線後，共{len(daily)}個交易日(社群資料涵蓋範圍："
                 f"{daily.index.min()}~{daily.index.max()})")
    _flush_new_lines()

    sample_dates = select_sample_dates(list(daily.index), n=args.n_sample_dates)
    lines.append(f"抽樣{len(sample_dates)}個交易日進行比對(平均分散在可用範圍內，非逐日全面稽核)")
    _flush_new_lines()

    if not sample_dates:
        lines.append("沒有可抽樣的交易日，驗證沒有執行。")
        _flush_new_lines()
        _write_report()
        sys.exit(0)

    start_date, end_date = sample_dates[0], sample_dates[-1]
    official_daily, fetch_reason = fetch_official_daily_ohlc(start_date, end_date, args.commodity_id)

    if official_daily is None:
        lines += [
            "",
            f"官方日線資料抓取失敗(reason={fetch_reason})，驗證沒有執行。",
            "這不代表社群資料不準，只代表這次沒有機會比對——TAIFEX官方端點的查詢欄位",
            "名稱是猜測(見模組docstring)，第一次在GitHub Actions真實環境執行，才是",
            "這個假設第一次被真正驗證；如果欄位名稱猜錯了，上面的console log會印出",
            "實際送出的查詢參數，方便之後對照真實回應修正。",
        ]
        _flush_new_lines()
        _write_report()
        print(f"官方日線資料抓取失敗(reason={fetch_reason})，驗證沒有執行。", file=sys.stderr, flush=True)
        sys.exit(0)

    sample_community = daily.loc[daily.index.isin(sample_dates)]
    comparison, stats = compare_daily_ohlc(sample_community, official_daily, tolerance_pct=args.tolerance_pct)

    lines.append("")
    lines.append(f"實際取得官方日線的交易日數：{len(official_daily)}；社群/官方重疊可比對的交易日數："
                 f"{stats['n_dates_compared']}")
    if stats["n_dates_compared"] > 0:
        lines.append(
            f"收盤價誤差 — 平均：{stats['mean_abs_error_close']:.4f}點"
            f"({stats['mean_pct_error_close']:.4f}%)，"
            f"中位數：{stats['median_abs_error_close']:.4f}點"
            f"({stats['median_pct_error_close']:.4f}%)，"
            f"最大：{stats['max_abs_error_close']:.4f}點"
            f"({stats['max_pct_error_close']:.4f}%)"
        )
        lines.append(
            f"落在容忍範圍(±{stats['tolerance_pct']}%)內的天數：{stats['n_within_tolerance']}/"
            f"{stats['n_dates_compared']}({stats['pct_within_tolerance']:.1f}%)"
        )
        lines.append("")
        lines.append("逐日明細：")
        lines.append(comparison.round(4).to_string())

    lines.append("")
    verdict = _verdict_sentence(stats)
    lines.append(f"結論：{verdict}")

    _flush_new_lines()
    _write_report()


if __name__ == "__main__":
    main()
