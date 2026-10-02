"""
validate_tx_history_accuracy.py
====================================
在真的拿taifex_history_loader.py(非官方、社群維護、crazyindicator.pixnet.net上的
Google Drive歷史資料)去跑任何回測之前，**先確認拿證交所給的日線去對比一下**
(使用者原話)——這支腳本就是做這件事，獨立CLI，不是taifex_history_loader.py的
一部分，因為它牽涉另一個完全獨立的資料來源(TAIFEX官方日線/data.gov.tw開放資料
備援)，混在loader模組裡會讓loader背負不該背的「還要懂怎麼抓官方資料」的責任。

⚠️⚠️ 跟repo其他所有loader一樣的誠實聲明：這個sandbox連taifex.com.tw/data.gov.tw
都連不到(proxy allowlist擋掉)。第一版對dlFutDailyMarketView(那其實是人看的
導覽/下載頁，不是查詢端點)送POST、猜firstDate/lastDate範圍查詢，第一次真實跑
失敗(回傳的是網頁本身)。這一版改用futDailyMarketReport，是交叉確認多篇獨立
公開爬蟲文章/repo的寫法得出的(跟taifex_intraday_loader.py改用daily zip URL
時同樣的交叉確認方式)：一次只能查**一天**(queryDate)，不支援起訖範圍查詢，
所以fetch_official_daily_ohlc()改成對sample_dates逐日查詢，不是查一個區間。
即使這版欄位名稱來源更可靠，仍然是**第一次在GitHub Actions真實環境用這組
參數打過**，失敗時一樣會印出明確診斷(包含實際送出的查詢參數)方便之後修正，
不是crash也不是假裝成功。

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
FUT_DAILY_MARKET_REPORT_PATH = "/cht/3/futDailyMarketReport"
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


def _parse_official_table(raw_bytes, fallback_date=None):
    """把官方端點回傳的原始內容(猜測是CSV/HTML表格)解析成統一欄位
    (Date/Open/High/Low/Close)的DataFrame，回傳(df_or_None, reason)。跟
    taifex_intraday_loader.py的parse_tick_csv()同樣的「動態比對欄位名稱」精神。

    fallback_date：單日查詢(futDailyMarketReport一次只能查一天，見
    _try_taifex_daily_endpoint docstring)時，表格裡常見的設計是日期只出現在
    查詢條件/頁面標題，表格本身欄位只有契約/開高低收，沒有重複列出日期欄——
    這種情況找不到date_col時，不要直接回報no_date_col，改用呼叫端已經知道
    的查詢日期(因為就是查這一天)當作Date欄，比死板要求表格裡一定要有日期欄
    更貼近單日查詢這個使用情境的實際資料形狀。"""
    # futDailyMarketReport確認回傳的是HTML表格，不是CSV(見_try_taifex_daily_endpoint
    # docstring)，所以內容看起來像HTML時優先走HTML解析，不要先試CSV——pd.read_csv()
    # 對HTML內容不一定會丟例外(可能把整個<table...>那行當成怪異的單欄位header硬解
    # 出一張「看起來像表格但其實不是」的1欄DataFrame)，導致真正的HTML表格解析
    # 永遠執行不到。data.gov.tw備援才真的可能是CSV，所以CSV優先順序留給那個情境。
    looks_like_html = raw_bytes.lstrip()[:1] == b"<"
    df = None

    def _try_csv():
        for encoding in ("utf-8", "big5", "cp950"):
            try:
                import io
                candidate = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, dtype=str)
                if candidate is not None and not candidate.empty and candidate.shape[1] > 1:
                    return candidate
            except Exception:
                continue
        return None

    def _try_html():
        import io
        for encoding in ("utf-8", "big5", "cp950"):
            try:
                tables = pd.read_html(io.BytesIO(raw_bytes), encoding=encoding)
                found = next((t for t in tables if t.shape[1] >= 4), None)
                if found is not None:
                    return found.astype(str)
            except Exception:
                continue
        return None

    if looks_like_html:
        df = _try_html()
        if df is None or df.empty:
            df = _try_csv()
    else:
        df = _try_csv()
        if df is None or df.empty:
            df = _try_html()

    if df is None or df.empty:
        return None, "no_table_parsed"

    date_col = _find_col(df.columns, DATE_COL_CANDS)
    open_col = _find_col(df.columns, OPEN_COL_CANDS)
    high_col = _find_col(df.columns, HIGH_COL_CANDS)
    low_col = _find_col(df.columns, LOW_COL_CANDS)
    close_col = _find_col(df.columns, CLOSE_COL_CANDS)
    product_col = _find_col(df.columns, PRODUCT_COL_CANDS)
    month_col = _find_col(df.columns, CONTRACT_MONTH_COL_CANDS)

    if date_col is None and fallback_date is None:
        print(f"[validate_tx_history_accuracy] 官方日線解析：缺少必要欄位['date']，"
              f"目前解析出的欄位名稱={list(df.columns)}(供之後修正欄位名稱candidates用)")
        return None, "missing_columns:date"
    if close_col is None:
        print(f"[validate_tx_history_accuracy] 官方日線解析：缺少必要欄位['close']，"
              f"目前解析出的欄位名稱={list(df.columns)}(供之後修正欄位名稱candidates用)")
        return None, "missing_columns:close"

    if date_col is not None:
        out = pd.DataFrame({"_date_raw": df[date_col].astype(str).str.strip()})
    else:
        out = pd.DataFrame({"_date_raw": [fallback_date.strftime("%Y%m%d")] * len(df)})
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


def _try_taifex_daily_endpoint(date_obj, commodity_id="TXF"):
    """嘗試TAIFEX「期貨每日交易行情查詢」。

    ⚠️這一輪改用真正查過的端點，不是第一版的猜測：第一次真實跑失敗
    (taifex_daily_returned_html_not_data)之後，查了多篇獨立的公開爬蟲文章/
    repo(交叉確認方式跟taifex_intraday_loader.py改用daily zip URL時一樣)，
    確認真正的端點是futDailyMarketReport(不是FUT_DAILY_VIEW_PATH / dlFut開頭
    那個下載頁路徑，那個是人看的導覽頁、不是資料查詢端點)，而且**一次只能查
    一天**(不支援起訖日期範圍查詢，這點第一版猜錯了)，欄位是：
        queryType='2', marketCode='0', commodity_id=<商品代號>,
        queryDate='YYYY/MM/DD'
    回傳表格是HTML(class通常是table_f)，不是直接下載CSV。

    即使這版欄位名稱是交叉確認過的，還是**第一次在GitHub Actions真實環境
    用這組參數打過**，仍然可能因為TAIFEX網站改版等原因失敗，失敗時一樣會
    印出實際送出的查詢參數方便之後對照修正。回傳(content_bytes_or_None, reason)。
    """
    url = TAIFEX_BASE + FUT_DAILY_MARKET_REPORT_PATH
    payload = {
        "queryType": "2",
        "marketCode": "0",
        "commodity_id": commodity_id,
        "queryDate": date_obj.strftime("%Y/%m/%d"),
    }
    try:
        resp = requests.post(url, data=payload, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
    except Exception as e:
        raise FetchFailed(str(e))
    if resp.status_code != 200:
        print(f"[validate_tx_history_accuracy] TAIFEX日線端點：HTTP {resp.status_code}，"
              f"送出的查詢參數={payload}")
        return None, f"taifex_daily_http_{resp.status_code}"
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


def _build_response_diagnostic(content):
    """這一輪新增：上一次真實跑看到的只是回應的前500 bytes，剛好全部落在
    <head>裡面(meta標籤之類)，完全看不出後面的<body>裡到底有沒有結果表格——
    看不出是(a)整頁回的是查詢表單本身(代表POST參數/流程錯了，例如這個頁面
    可能是ASP.NET WebForms、需要先GET拿__VIEWSTATE等隱藏欄位才能正確送出
    POST)，還是(b)表格其實在，只是pd.read_html()的欄位判斷邏輯不符
    (shape[1]>=4這個門檻，或欄位名稱候選詞對不上)，還是(c)結果由JavaScript
    動態載入、原始HTML裡根本没有<table>。這三種情況要修的地方完全不同，
    所以這裡明確回報"content裡到底有沒有<table>標籤"，並且在有的情況下，
    把*那個標籤附近*的內容摘出來(而不是永遠只看開頭的500 bytes)，才看得到
    真正該看的部分。回傳一段人看得懂的多行診斷字串。"""
    text_lower = content.lower()
    has_table_tag = b"<table" in text_lower
    has_viewstate = b"__viewstate" in text_lower
    table_idx = text_lower.find(b"<table")

    lines = [
        f"回應內容長度：{len(content)} bytes；是否含有<table>標籤：{has_table_tag}；"
        f"是否像ASP.NET WebForms(含__VIEWSTATE隱藏欄位)：{has_viewstate}",
    ]
    if has_table_tag:
        snippet = content[table_idx:table_idx + 500]
        lines.append(f"<table>標籤附近內容：{snippet!r}")
    else:
        lines.append(f"開頭500 bytes(供參考，但整份內容裡沒有<table>標籤)：{content[:500]!r}")
    return "\n".join(lines)


def _fetch_taifex_single_date(date_obj, commodity_id):
    """對單一日期查TAIFEX futDailyMarketReport，回傳(df_or_None, reason,
    content_preview_or_None)。連線層級失敗時重試MAX_RETRIES次。

    content_preview：HTTP本身成功、但_parse_official_table()解析失敗時，附上
    實際回應內容前500字(安全repr，不會因編碼問題再炸一次)——跟
    taifex_history_loader.py的_raw_content_preview()同樣的「解析失敗時留下
    診斷線索，不要只留一個reason字串」精神。只有解析失敗(HTTP本身成功)才有
    這個值，HTTP層級失敗(逾時/非200)時是None，因為那種情況看內容預覽沒意義。
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            content, reason = run_with_hard_timeout(
                _try_taifex_daily_endpoint, args=(date_obj, commodity_id),
                timeout=HARD_TIMEOUT_SECONDS + 10)
            if reason == "ok":
                df, parse_reason = _parse_official_table(content, fallback_date=date_obj)
                if parse_reason == "ok":
                    return df, "ok", None
                preview = _build_response_diagnostic(content)
                return None, "taifex_daily_" + parse_reason, preview
            return None, reason, None
        except (FetchFailed, HardTimeout):
            if attempt < MAX_RETRIES:
                import time
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            return None, "taifex_daily_connection_error", None
    return None, "taifex_daily_connection_error", None


def fetch_official_daily_ohlc(dates, commodity_id="TXF"):
    """
    整合入口：對傳入的每個日期(sample_dates，不是整個起訖範圍——
    futDailyMarketReport一次只能查一天，見_try_taifex_daily_endpoint docstring，
    所以這裡改成逐日查詢，只查真正要拿來比對的那幾十天，不是整個區間裡每一天)
    依序嘗試TAIFEX官方端點，某幾天失敗不影響其他天(跟taifex_history_loader.py
    單一period失敗不連累其他period同樣精神)。TAIFEX完全查不到任何一天時，
    退回data.gov.tw開放資料整批下載當保底，試著從裡面篩出需要的日期。

    回傳(df_or_None, reason, diag)，df的index是date物件，欄位Open/High/Low/Close：
      reason == "ok"：至少成功查到一天
      reason == "taifex_daily_..."/"data_gov_..."：兩者都完全沒拿到任何一天時的原因
    diag = {"taifex_reason": ..., "data_gov_reason": ..., "n_dates_ok": int,
            "n_dates_failed": int, "date_failures": {date_str: reason}}
    """
    diag = {"taifex_reason": None, "data_gov_reason": None,
            "n_dates_ok": 0, "n_dates_failed": 0, "date_failures": {},
            "sample_content_preview": None}
    dfs = []
    for date_obj in dates:
        df, reason, preview = _fetch_taifex_single_date(date_obj, commodity_id)
        if df is not None and reason == "ok":
            dfs.append(df)
            diag["n_dates_ok"] += 1
        else:
            diag["n_dates_failed"] += 1
            diag["date_failures"][date_obj.isoformat()] = reason
            if preview is not None and diag["sample_content_preview"] is None:
                # 只留第一筆預覽就好，30天大概率是同一種格式問題，不需要30份一樣的預覽。
                diag["sample_content_preview"] = preview

    if dfs:
        diag["taifex_reason"] = "ok"
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep="first")]
        return combined, "ok", diag

    # TAIFEX一天都沒查到，記下代表性的失敗原因(最常見的那個)，再試data.gov.tw保底。
    if diag["date_failures"]:
        reasons = list(diag["date_failures"].values())
        diag["taifex_reason"] = max(set(reasons), key=reasons.count)
    else:
        diag["taifex_reason"] = "no_dates_to_query"

    start_date, end_date = min(dates), max(dates)
    try:
        content, reason = run_with_hard_timeout(
            _try_data_gov_fallback, args=(start_date, end_date, commodity_id),
            timeout=HARD_TIMEOUT_SECONDS + 10)
        if reason == "ok":
            df, parse_reason = _parse_official_table(content)
            if parse_reason == "ok":
                diag["data_gov_reason"] = "ok"
                return df, "ok", diag
            diag["data_gov_reason"] = "data_gov_" + parse_reason
        else:
            diag["data_gov_reason"] = reason
    except (FetchFailed, HardTimeout):
        diag["data_gov_reason"] = "data_gov_connection_error"

    return None, diag["data_gov_reason"], diag


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
            f"各period失敗原因：{ {k: {kk: vv for kk, vv in v.items() if kk != 'raw_preview'} for k, v in diag.get('periods', {}).items()} }",
        ]
        for period_key, period_diag in diag.get("periods", {}).items():
            if "raw_preview" in period_diag:
                lines += [
                    "",
                    f"--- {period_key} 原始檔案內容預覽(解析失敗時的診斷用，方便下次直接修正解析邏輯) ---",
                    period_diag["raw_preview"],
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

    official_daily, fetch_reason, fetch_diag = fetch_official_daily_ohlc(
        sample_dates, args.commodity_id)

    if official_daily is None:
        lines += [
            "",
            f"官方日線資料抓取失敗(reason={fetch_reason})，驗證沒有執行。",
            f"兩個來源個別的失敗原因 — TAIFEX官方端點：{fetch_diag.get('taifex_reason')}"
            f"(逐日查詢：成功{fetch_diag.get('n_dates_ok')}天/失敗{fetch_diag.get('n_dates_failed')}天，"
            f"各天失敗原因={fetch_diag.get('date_failures')})；"
            f"data.gov.tw備援：{fetch_diag.get('data_gov_reason')}",
            "這不代表社群資料不準，只代表這次沒有機會比對——TAIFEX官方端點的查詢欄位",
            "名稱是猜測(見模組docstring)，第一次在GitHub Actions真實環境執行，才是",
            "這個假設第一次被真正驗證；如果欄位名稱猜錯了，上面的console log會印出",
            "實際送出的查詢參數，方便之後對照真實回應修正。",
        ]
        if fetch_diag.get("sample_content_preview"):
            lines += [
                "",
                "--- TAIFEX端點實際回應內容預覽(解析失敗時的診斷用，取其中一天的回應"
                "前500字，方便下次直接對照修正_parse_official_table的欄位解析邏輯) ---",
                fetch_diag["sample_content_preview"],
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
