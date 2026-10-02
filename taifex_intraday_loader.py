"""
台指期(TX/大台指期貨)逐筆成交資料 -> 1小時K棒 —— 資料來源信心程度：**這個repo目前
所有資料來源裡最低**，明確標記為**未經驗證(UNVERIFIED)**，比financial_statement_loader.py
(信心程度「最低」的既有記錄保持人)更不確定，原因見下方。

⚠️⚠️ 為什麼比financial_statement_loader.py還不確定：
financial_statement_loader.py好歹知道要打哪個查詢介面(MOPS t163sb04)、表單欄位名稱
是「根據既有認識推斷」。這裡連「這個頁面到底是不是表單、要用GET還是POST、欄位叫
什麼名字」都沒有機會實測——這個sandbox環境對外連線同樣被proxy allowlist擋掉
(跟financial_statement_loader.py/revenue_data_loader.py遇到的限制一樣)，WebFetch
只能看到taifex.com.tw頁面的高層結構(選單導覽)，看不到實際觸發下載的JS/表單細節。
**第一次在GitHub Actions真實環境執行這支loader，才是它的第一次真正整合測試**，
下面的實作是「防禦性猜測」，不是已驗證的行為，失敗是預期中可能發生的事，不代表設計錯誤。

資料來源(使用者要求)：臺灣期貨交易所(TAIFEX)「交易資訊 > 資料下載專區 > 期貨 >
前30個交易日期貨每筆成交資料」頁面：
  https://www.taifex.com.tw/cht/3/dlFutPrevious30DaysSalesData
這是TAIFEX**唯一免費**、不需要「交易歷史資料申請」付費/審核流程的逐筆成交資料來源，
但先天限制是官方只提供「最近30個交易日」的滾動窗口，查詢時間點之前更久的資料
一律拿不到(除非走付費申請)。使用者已經明確理解並接受這個限制：這支模組是給
squeeze_kdj_tx_hourly_preview.py用的「預覽」用途，不是要拿來做嚴謹回測。

下載機制的猜測與不確定性(誠實列出)：
1. 這個頁面用WebFetch只看得到選單導覽的HTML，看不到實際觸發下載的表單/JS行為，
   猜測沿用TAIFEX其他「資料下載」頁面常見的模式：對同一個頁面路徑送POST，帶
   commodity_id(商品代號)等欄位，直接拿回CSV/文字內容(不是先給一個下載連結、
   使用者手動點擊)。這裡完全是猜測，欄位名稱(commodity_id/queryType等)都沒有
   查證來源。
2. 為了不只賭「POST猜對」，這裡同時準備了第二種策略：對頁面做一次GET，用正規
   表示式在回傳的HTML裡找看看有沒有現成的下載連結(<a href=...>指向.csv/.zip/
   包含"Down"字樣的路徑)，找到就直接下載那個連結——這個策略比較穩，因為不用猜
   表單欄位名稱，只要頁面上真的有這種連結。
3. 兩種策略都失敗時，`fetch_raw_tick_data()`回傳(None, reason)，reason會明確
   說是哪一種失敗(見函式docstring列舉的原因)，呼叫端(load_tx_hourly_bars())
   把這個reason原封不動往上傳、印出診斷訊息，不會偽裝成「資料是空的」。
4. 就算成功抓到內容，欄位名稱/格式(商品代號欄叫什麼、到期月份欄叫什麼、時間格式
   是HHMMSS還是HH:MM:SS)也是根據TAIFEX其他逐筆成交資料集常見慣例推斷，一樣沒有
   實測驗證過，parse_tick_csv()用「欄位名稱包含特定子字串」動態比對(跟
   revenue_data_loader.py/financial_statement_loader.py同樣的容錯精神)，找不到
   預期欄位時一樣回傳明確原因，不會靜默吞掉。

近月合約(front month)判斷：不假設知道正確的到期日曆(TX是每週/每月都有商品到期，
精確計算「目前哪一口是近月」需要完整的到期日規則，這裡沒有實作)，改用一個較粗但
穩健的實務捷徑：抓回的30個交易日資料裡，每個「到期月份(週別)」代碼各自的成交筆數
加總，筆數最多的那個視為近月合約(近月合約流動性遠高於遠月/價差是業界共識，成交量
集中度是可靠的近月判斷依據，不需要精確到期日曆也能得到正確答案)。

盤別(session)假設(誠實列出，未逐一查證目前TAIFEX官方公告的確切時間，這裡採用
一般認知的現行時段)：
  日盤(day session)：08:45–13:45(台灣時間)
  夜盤(night/after-hours session)：約15:00–翌日05:00
這裡**只處理日盤資料，夜盤ticks會被明確過濾掉並在log裡回報筆數**，不是忘記處理，
是刻意簡化(見squeeze_kdj_tx_hourly_preview.py呼叫端說明)：夜盤橫跨兩個日曆日、
交易時數(約14小時)遠超過日盤(5小時)，混在同一套「1小時K棒」邏輯裡容易在日期邊界
出錯，這一版先只驗證日盤，日盤本身就有5根完整的1小時K棒(08:45-09:45/.../12:45-
13:45)，足夠給這個「預覽」腳本一個能看的樣本。之後如果要延伸到夜盤，建議先跟
使用者確認夜盤實際時段(以防官方公告時段跟這裡假設的不同)。

快取：原始下載內容存到 taifex_intraday_cache/ 目錄，用「查詢當天日期」當檔名
(不是用資料涵蓋的交易日，因為TAIFEX這個端點回傳的永遠是「以查詢當下往回算30個
交易日」的滾動窗口，同一個查詢日重跑不需要重新打網路)。
"""
import os
import re
import io
import datetime
import requests
import pandas as pd

from network_utils import run_with_hard_timeout, HardTimeout

TAIFEX_BASE = "https://www.taifex.com.tw"
LANDING_PATH = "/cht/3/dlFutPrevious30DaysSalesData"

TAIFEX_INTRADAY_CACHE_DIR = os.path.join(os.path.dirname(__file__), "taifex_intraday_cache")
HARD_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3
THREAD_HARD_TIMEOUT_SECONDS = HARD_TIMEOUT_SECONDS + 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# 日盤時段假設(未逐一查證官方最新公告，見模組docstring)。
DAY_SESSION_START = "08:45"
DAY_SESSION_END = "13:45"

# 欄位名稱候選(未實測驗證，見模組docstring第4點)。
PRODUCT_COL_CANDS = ("商品代號", "契約代號", "商品名稱")
CONTRACT_MONTH_COL_CANDS = ("到期月份(週別)", "到期月份", "契約月份")
DATE_COL_CANDS = ("成交日期", "交易日期")
TIME_COL_CANDS = ("成交時間",)
PRICE_COL_CANDS = ("成交價格", "成交價")
VOLUME_COL_CANDS = ("成交數量(B or S)", "成交數量", "數量")


class FetchFailed(Exception):
    """代表這次請求本身失敗(逾時/連線錯誤)，由呼叫端決定要不要重試。跟「查了但格式
    不是預期的」(那個用reason字串表達)是兩件不同的事。"""
    pass


def _find_col(columns, candidates):
    for col in columns:
        col_str = str(col).replace(" ", "")
        for cand in candidates:
            if cand.replace(" ", "") in col_str:
                return col
    return None


def _try_post_guess(commodity_id: str):
    """策略1(純猜測，見模組docstring第1點)：對頁面路徑本身送POST，帶commodity_id。
    回傳(content_bytes_or_None, reason)。"""
    url = TAIFEX_BASE + LANDING_PATH
    payload = {"commodity_id": commodity_id, "queryType": "1"}
    try:
        resp = requests.post(url, data=payload, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
    except Exception as e:
        raise FetchFailed(str(e))
    if resp.status_code != 200:
        return None, f"post_guess_http_{resp.status_code}"
    content_type = resp.headers.get("Content-Type", "")
    text_head = resp.content[:200].decode("utf-8", errors="ignore").lstrip().lower()
    if "text/html" in content_type or text_head.startswith("<!doctype") or text_head.startswith("<html"):
        # 回傳的還是網頁本身，代表猜的表單欄位/POST方式錯了，不是真的資料檔案。
        return None, "post_guess_returned_html_not_data"
    return resp.content, "ok"


def _try_scrape_download_link():
    """策略2(較穩，見模組docstring第2點)：GET頁面本身，用正規表示式找HTML裡
    看起來像下載連結的<a href>，找到就跟著下載。回傳(content_bytes_or_None, reason)。"""
    url = TAIFEX_BASE + LANDING_PATH
    try:
        resp = requests.get(url, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
    except Exception as e:
        raise FetchFailed(str(e))
    if resp.status_code != 200:
        return None, f"scrape_landing_http_{resp.status_code}"

    html_text = resp.content.decode("utf-8", errors="ignore")
    # 找看看頁面上有沒有指向.csv/.zip，或路徑帶"Down"字樣的連結。
    matches = re.findall(r'href=["\']([^"\']+\.(?:csv|zip))["\']', html_text, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r'href=["\']([^"\']*Down[^"\']*)["\']', html_text)
    if not matches:
        return None, "scrape_no_download_link_found"

    link = matches[0]
    if link.startswith("http"):
        download_url = link
    elif link.startswith("/"):
        download_url = TAIFEX_BASE + link
    else:
        download_url = TAIFEX_BASE + "/" + link

    try:
        dl_resp = requests.get(download_url, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
    except Exception as e:
        raise FetchFailed(str(e))
    if dl_resp.status_code != 200:
        return None, f"scrape_download_http_{dl_resp.status_code}"
    return dl_resp.content, "ok"


DAILY_ZIP_URL_TEMPLATE = TAIFEX_BASE + "/file/taifex/Dailydownload/DailydownloadCSV/Daily_{y:04d}_{m:02d}_{d:02d}.zip"
DAILY_ZIP_CACHE_DIR = os.path.join(os.path.dirname(__file__), "taifex_intraday_cache", "daily_zips")

# commodity_id對ProductCode欄位的別名猜測：使用者/呼叫端習慣傳"TXF"，但TAIFEX逐筆
# 成交檔案裡商品代號欄位實際內容可能是"TX"(不含F)。exact match優先，match不到才
# 退而求其次試這個別名表，兩種都試過還是找不到就誠實回報，不要猜更多。
COMMODITY_ALIASES = {"TXF": ["TX"], "TX": ["TXF"]}


def _try_direct_daily_zip(date_obj, commodity_id: str):
    """策略0(這一輪新增，根據多篇獨立的公開爬蟲文章/repo交叉確認過的真實下載機制，
    不是用猜的——見taifex_intraday_loader.py所在commit訊息)：TAIFEX把每個交易日的
    全商品逐筆成交資料包成固定命名的zip檔，直接用日期組URL就能下載，不需要猜表單
    欄位、也不用掃描頁面找連結：
        https://www.taifex.com.tw/file/taifex/Dailydownload/DailydownloadCSV/Daily_YYYY_MM_DD.zip
    這個策略一次只服務「一個交易日」，呼叫端(fetch_multi_day_ticks)負責往回疊代
    多個交易日湊出「近30個交易日」的資料，不像策略1/2是對「前30天」這個頁面整包要。

    回傳(ticks_df_or_None, reason)：
      reason == "ok"                 成功，ticks_df已經過parse_tick_csv()解析、
                                      且已經用commodity_id(或COMMODITY_ALIASES裡的
                                      別名)篩選過ProductCode
      reason == "http_404"           這天沒有這個檔案(非交易日/假日，正常情況，
                                      呼叫端不該當成錯誤看待，應該跳過繼續試前一天)
      reason == "http_<code>"        其他HTTP錯誤
      reason == "<parse_tick_csv的reason>"  下載成功但解析失敗，原因沿用
                                      parse_tick_csv()的reason定義
      reason == "commodity_not_found_in_day" 解析成功，但這天的資料裡找不到
                                      commodity_id(含別名)對應的ProductCode
    連線本身失敗(逾時/連線錯誤)時丟出FetchFailed，由呼叫端決定要不要重試這天。
    """
    url = DAILY_ZIP_URL_TEMPLATE.format(y=date_obj.year, m=date_obj.month, d=date_obj.day)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
    except Exception as e:
        raise FetchFailed(str(e))
    if resp.status_code == 404:
        return None, "http_404"
    if resp.status_code != 200:
        return None, f"http_{resp.status_code}"

    ticks, parse_reason = parse_tick_csv(resp.content)
    if parse_reason != "ok":
        return None, parse_reason

    candidates = [commodity_id] + COMMODITY_ALIASES.get(commodity_id, [])
    product_stripped = ticks["ProductCode"].str.strip()
    matched = None
    for cand in candidates:
        hit = ticks[product_stripped == cand]
        if not hit.empty:
            matched = hit
            break
    if matched is None:
        return None, "commodity_not_found_in_day"
    return matched.copy(), "ok"


def fetch_multi_day_ticks(commodity_id: str = "TXF", target_trading_days: int = 32,
                           max_lookback_calendar_days: int = 50, cache_dir=None, refresh: bool = False):
    """
    往回疊代日曆日(跳過週六週日)，對每個交易日呼叫_try_direct_daily_zip()，直到湊滿
    target_trading_days個「真的有資料」的交易日，或超過max_lookback_calendar_days
    (多留一點緩衝給國定假日)。每天的原始zip內容快取在DAILY_ZIP_CACHE_DIR/下，檔名
    是日期(過去的交易日資料永遠不會變，不像舊版「前30天」快取每天都要重抓)。

    回傳(ticks_df_or_None, diagnostics: dict)。diagnostics包含：
      "reason"                    "ok"或明確失敗原因
      "n_trading_days_found"      成功湊到幾個交易日
      "n_calendar_days_scanned"   實際掃了幾個日曆日(含假日/週末)
      "day_failures"              dict，日期字串->失敗reason，只記錄「有掃但沒抓到」
                                   的日子(http_404是正常的非交易日，不算失敗、不記錄)
    """
    cache_dir = cache_dir or DAILY_ZIP_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    all_ticks = []
    day_failures = {}
    n_found = 0
    n_scanned = 0
    today = datetime.date.today()

    for offset in range(1, max_lookback_calendar_days + 1):
        if n_found >= target_trading_days:
            break
        date_obj = today - datetime.timedelta(days=offset)
        if date_obj.weekday() >= 5:  # 週六=5, 週日=6
            continue
        n_scanned += 1

        cache_path = os.path.join(cache_dir, f"{date_obj.isoformat()}_{commodity_id}.csv")
        if not refresh and os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, parse_dates=["Timestamp"])
            if not cached.empty:
                all_ticks.append(cached)
                n_found += 1
            continue

        try:
            day_ticks, reason = run_with_hard_timeout(
                _try_direct_daily_zip, args=(date_obj, commodity_id), timeout=THREAD_HARD_TIMEOUT_SECONDS)
        except (FetchFailed, HardTimeout) as e:
            day_failures[date_obj.isoformat()] = f"connection_error:{e}"
            continue

        if reason == "http_404":
            continue  # 正常的非交易日，不記錄成失敗
        if reason != "ok":
            day_failures[date_obj.isoformat()] = reason
            continue

        day_ticks.to_csv(cache_path, index=False)
        all_ticks.append(day_ticks)
        n_found += 1

    if not all_ticks:
        return None, {
            "reason": "no_trading_days_found", "n_trading_days_found": 0,
            "n_calendar_days_scanned": n_scanned, "day_failures": day_failures,
        }

    combined = pd.concat(all_ticks, ignore_index=True)
    return combined, {
        "reason": "ok", "n_trading_days_found": n_found,
        "n_calendar_days_scanned": n_scanned, "day_failures": day_failures,
    }


def fetch_raw_tick_data(commodity_id: str = "TXF"):
    """
    嘗試抓取「前30個交易日期貨每筆成交資料」的原始內容(見模組docstring策略1/2)。
    回傳(content_bytes_or_None, reason)：
      reason == "ok"                       成功，content是原始下載內容(bytes)
      reason == "post_guess_..."           策略1(POST猜測)失敗的具體原因
      reason == "scrape_..."               策略2(掃描下載連結)失敗的具體原因，兩個
                                            策略都失敗時，reason是策略2的原因(較後執行)
    連線本身失敗(逾時/連線錯誤，兩個策略都是)時丟出FetchFailed，由呼叫端決定要不要重試。
    """
    fetch_errors = []
    try:
        content, reason = _try_post_guess(commodity_id)
        if reason == "ok":
            return content, "ok"
        last_reason = reason
    except FetchFailed as e:
        fetch_errors.append(str(e))
        last_reason = "post_guess_connection_error"

    try:
        content, reason = _try_scrape_download_link()
        if reason == "ok":
            return content, "ok"
        last_reason = reason
    except FetchFailed as e:
        fetch_errors.append(str(e))
        last_reason = "scrape_connection_error"

    if len(fetch_errors) == 2:
        raise FetchFailed("; ".join(fetch_errors))
    return None, last_reason


def _fetch_raw_with_retry(commodity_id: str):
    """回傳(content, success, reason)，跟revenue_data_loader.py的
    _fetch_one_month_with_retry()同樣精神：success=False代表逾時/連線失敗
    (該重試)，success=True時reason是"ok"或具體失敗原因(不該重試，代表格式/猜測錯誤)。"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            content, reason = run_with_hard_timeout(
                fetch_raw_tick_data, args=(commodity_id,), timeout=THREAD_HARD_TIMEOUT_SECONDS)
            return content, True, reason
        except (FetchFailed, HardTimeout):
            if attempt < MAX_RETRIES:
                import time
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            return None, False, "fetch_failed"
    return None, False, "fetch_failed"


def parse_tick_csv(raw_content: bytes):
    """
    把原始下載內容(猜測是CSV文字，也可能是zip，見下方)解析成統一格式的逐筆成交
    DataFrame，欄位：ProductCode/ContractMonth/Timestamp(合併日期+時間的
    pd.Timestamp)/Price(float)/Volume(float)。

    回傳(df_or_None, reason)：
      reason == "ok"
      reason == "zip_no_csv_member"     內容是zip但裡面找不到csv成員
      reason == "not_csv_or_zip"        內容既不是能解析的CSV文字、也不是zip
      reason == "no_table_parsed"       pandas.read_csv完全解析不出東西
      reason == "missing_columns:<...>" 解析出表格，但找不到必要欄位(見冒號後列出缺哪些)
      reason == "empty_after_filter"    解析出表格、欄位也找到了，但過濾完是空的
    """
    text = None
    if raw_content[:2] == b"PK":  # zip檔案的magic bytes
        import zipfile
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw_content))
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return None, "zip_no_csv_member"
            text = zf.read(csv_names[0])
        except Exception:
            return None, "zip_no_csv_member"
    else:
        text = raw_content

    df = None
    for encoding in ("utf-8", "big5", "cp950"):
        try:
            df = pd.read_csv(io.BytesIO(text) if isinstance(text, bytes) else io.StringIO(text),
                              encoding=encoding, dtype=str)
            break
        except Exception:
            df = None
            continue
    if df is None or df.empty:
        return None, "no_table_parsed"

    product_col = _find_col(df.columns, PRODUCT_COL_CANDS)
    month_col = _find_col(df.columns, CONTRACT_MONTH_COL_CANDS)
    date_col = _find_col(df.columns, DATE_COL_CANDS)
    time_col = _find_col(df.columns, TIME_COL_CANDS)
    price_col = _find_col(df.columns, PRICE_COL_CANDS)
    volume_col = _find_col(df.columns, VOLUME_COL_CANDS)

    missing = [name for name, col in [
        ("product", product_col), ("contract_month", month_col), ("date", date_col),
        ("time", time_col), ("price", price_col), ("volume", volume_col),
    ] if col is None]
    if missing:
        return None, "missing_columns:" + ",".join(missing)

    out = pd.DataFrame({
        "ProductCode": df[product_col].astype(str).str.strip(),
        "ContractMonth": df[month_col].astype(str).str.strip(),
        "_date_raw": df[date_col].astype(str).str.strip(),
        "_time_raw": df[time_col].astype(str).str.strip().str.zfill(6),
        "Price": pd.to_numeric(df[price_col], errors="coerce"),
        "Volume": pd.to_numeric(df[volume_col], errors="coerce"),
    })
    out = out.dropna(subset=["Price", "Volume"])
    if out.empty:
        return None, "empty_after_filter"

    # 日期欄猜測是YYYYMMDD純數字字串，時間欄猜測是HHMMSS(不含冒號，zfill(6)補足)，
    # 兩者格式都沒有實測驗證過(見模組docstring第4點)，解析不出來的列直接丟棄。
    ts = pd.to_datetime(out["_date_raw"] + out["_time_raw"], format="%Y%m%d%H%M%S", errors="coerce")
    out["Timestamp"] = ts
    out = out.dropna(subset=["Timestamp"]).drop(columns=["_date_raw", "_time_raw"])
    if out.empty:
        return None, "empty_after_filter"

    return out, "ok"


def select_front_month(ticks: pd.DataFrame):
    """
    從逐筆成交資料裡挑出「近月合約」的資料列：ContractMonth各自的成交筆數(不是
    成交量)加總，筆數最多的那個視為近月(見模組docstring「近月合約判斷」段落，
    這是刻意選的、不依賴到期日曆的粗略但穩健捷徑)。
    回傳(filtered_df, front_month_code)，ticks為空時回傳(空df, None)。
    """
    if ticks.empty:
        return ticks, None
    counts = ticks["ContractMonth"].value_counts()
    front_month = counts.index[0]
    return ticks[ticks["ContractMonth"] == front_month].copy(), front_month


def aggregate_ticks_to_bars(ticks: pd.DataFrame, bar_minutes: int = 60,
                             session_start: str = DAY_SESSION_START,
                             session_end: str = DAY_SESSION_END):
    """
    把逐筆成交(Timestamp/Price/Volume)聚合成任意分鐘數的OHLCV K棒，**只處理日盤時段**
    (見模組docstring「盤別假設」段落，夜盤ticks會被過濾掉)。這是
    aggregate_ticks_to_hourly()的通用版本(這一輪新增，見squeeze_kdj_tx_hourly_preview.py
    之後要延伸到台指期15分鐘K棒的tx_intraday_engine.py)——原本寫死60分鐘窗口的邏輯
    改成參數化，bar_minutes=60時行為跟舊版aggregate_ticks_to_hourly()完全一致。

    每根K棒：
      Open   = 這根K棒時間窗內第一筆成交價
      High   = 這根K棒時間窗內最高成交價
      Low    = 這根K棒時間窗內最低成交價
      Close  = 這根K棒時間窗內最後一筆成交價
      Volume = 這根K棒時間窗內成交量加總
    (用「最後一筆成交價」當Close、不是量加權平均價：這樣OHLC彼此之間維持
    High>=Open/Close>=Low的一致性，量加權平均價可能落在High/Low範圍外造成
    K棒本身不自洽，這裡選簡單、K棒定義自洽的做法，見模組docstring)。

    K棒窗口從session_start開始，每bar_minutes分鐘切一刀，直到session_end(含最後一段，
    即使不足bar_minutes分鐘)。

    回傳(bars_df, n_night_session_ticks_dropped)。bars_df的index是每根
    K棒的起始時間(pd.Timestamp，含日期)，欄位Open/High/Low/Close/Volume。
    """
    if ticks.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]), 0

    ticks = ticks.sort_values("Timestamp")
    time_of_day = ticks["Timestamp"].dt.strftime("%H:%M")
    in_session = (time_of_day >= session_start) & (time_of_day <= session_end)
    n_dropped = int((~in_session).sum())
    day_ticks = ticks[in_session].copy()
    if day_ticks.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]), n_dropped

    session_start_h, session_start_m = (int(x) for x in session_start.split(":"))

    def _bar_start(ts):
        session_open = ts.replace(hour=session_start_h, minute=session_start_m, second=0, microsecond=0)
        elapsed_minutes = (ts - session_open).total_seconds() / 60.0
        bar_index = int(elapsed_minutes // bar_minutes)
        return session_open + pd.Timedelta(minutes=bar_index * bar_minutes)

    day_ticks["_bar_start"] = day_ticks["Timestamp"].apply(_bar_start)

    rows = []
    for bar_start, grp in day_ticks.groupby("_bar_start"):
        rows.append({
            "BarStart": bar_start,
            "Open": float(grp["Price"].iloc[0]),
            "High": float(grp["Price"].max()),
            "Low": float(grp["Price"].min()),
            "Close": float(grp["Price"].iloc[-1]),
            "Volume": float(grp["Volume"].sum()),
        })
    bars = pd.DataFrame(rows).set_index("BarStart").sort_index()
    return bars, n_dropped


def aggregate_ticks_to_hourly(ticks: pd.DataFrame, session_start: str = DAY_SESSION_START,
                               session_end: str = DAY_SESSION_END):
    """
    向後相容包裝：1小時K棒版本，等價於 aggregate_ticks_to_bars(ticks, bar_minutes=60, ...)。
    保留這個函式名稱/簽名是因為squeeze_kdj_tx_hourly_preview.py(透過load_tx_hourly_bars())
    已經在用，不因為新增通用版本而破壞既有呼叫端。
    """
    return aggregate_ticks_to_bars(ticks, bar_minutes=60, session_start=session_start, session_end=session_end)


def load_tx_bars(commodity_id: str = "TXF", refresh: bool = False, bar_minutes: int = 60):
    """
    整合入口(通用版)：下載(或讀快取)「前30個交易日」TX逐筆成交資料 -> 過濾近月合約 ->
    聚合成bar_minutes分鐘K棒(僅日盤)。這一輪新增，把原本寫死1小時的load_tx_hourly_bars()
    邏輯參數化，讓tx_intraday_engine.py(15分鐘K棒版右側突破引擎)可以共用同一套下載/
    快取/近月合約邏輯，只是聚合的K棒週期不同——**下載/解析/近月合約判斷完全不變**，
    只有最後一步「聚合成K棒」這裡換成傳入的bar_minutes(見aggregate_ticks_to_bars())。
    快取用的是原始逐筆成交內容，跟bar_minutes無關，同一天內不管跑幾種K棒週期都只打一次網路。

    回傳(bars_df_or_None, diagnostics: dict)。diagnostics包含：
      "reason"              成功時"ok"，失敗時具體原因(見fetch_raw_tick_data/
                             parse_tick_csv的reason定義)
      "front_month"         成功時挑出的近月合約代碼
      "n_ticks"              成功時過濾近月後的逐筆成交筆數
      "n_night_session_dropped" 成功時被丟棄的非日盤ticks筆數
      "used_cache"          這次是不是直接用快取，沒有重新打網路

    失敗時(reason != "ok")，bars_df_or_None是None，呼叫端(squeeze_kdj_tx_hourly_preview.py/
    compare_tx_intraday.py)必須把這個當成「這次沒有資料可用」優雅處理，不能讓整個腳本崩潰
    (見squeeze_kdj_tx_hourly_preview.py)。
    """
    os.makedirs(TAIFEX_INTRADAY_CACHE_DIR, exist_ok=True)

    # 策略0(直接組每日zip URL，見fetch_multi_day_ticks docstring)優先，這是經過
    # 交叉確認過的真實下載機制，不是猜測；策略1/2(POST猜測/掃描下載連結，對應
    # 舊版「前30天」整包頁面)留著當保底，萬一TAIFEX哪天改了每日zip的命名規則或
    # 路徑，還有退路可以試。
    ticks, multi_day_diag = fetch_multi_day_ticks(commodity_id, refresh=refresh)
    used_cache = False  # fetch_multi_day_ticks內部逐日快取，這裡的used_cache欄位
                         # 保留給舊策略用，策略0成功時不透過這個欄位表達快取狀態，
                         # 完整快取細節看multi_day_diag本身。
    if ticks is not None and multi_day_diag["reason"] == "ok":
        front_ticks, front_month = select_front_month(ticks)
        if front_ticks.empty:
            return None, {"reason": "no_front_month_ticks", "used_cache": used_cache,
                           "strategy": "direct_daily_zip", "multi_day_diag": multi_day_diag}
        bars, n_dropped = aggregate_ticks_to_bars(front_ticks, bar_minutes=bar_minutes)
        if bars.empty:
            return None, {"reason": "no_day_session_bars", "used_cache": used_cache,
                           "front_month": front_month, "n_night_session_dropped": n_dropped,
                           "strategy": "direct_daily_zip", "multi_day_diag": multi_day_diag}
        return bars, {
            "reason": "ok", "used_cache": used_cache, "front_month": front_month,
            "n_ticks": int(len(front_ticks)), "n_night_session_dropped": n_dropped,
            "strategy": "direct_daily_zip", "multi_day_diag": multi_day_diag,
        }

    # 策略0完全沒湊到任何一天的資料，退回舊版策略1/2(整包「前30天」頁面猜測)。
    today_str = datetime.date.today().isoformat()
    cache_path = os.path.join(TAIFEX_INTRADAY_CACHE_DIR, f"raw_{commodity_id}_{today_str}.bin")

    if not refresh and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            content = f.read()
        used_cache = True
        fetch_reason = "ok" if content else "cached_empty"
    else:
        content, success, fetch_reason = _fetch_raw_with_retry(commodity_id)
        if success and fetch_reason == "ok" and content:
            with open(cache_path, "wb") as f:
                f.write(content)
        elif not success:
            return None, {"reason": "network_failed_after_retries", "used_cache": False,
                           "strategy": "legacy_fallback", "multi_day_diag": multi_day_diag}

    if not content or fetch_reason != "ok":
        return None, {"reason": fetch_reason, "used_cache": used_cache,
                       "strategy": "legacy_fallback", "multi_day_diag": multi_day_diag}

    ticks, parse_reason = parse_tick_csv(content)
    if parse_reason != "ok":
        return None, {"reason": parse_reason, "used_cache": used_cache,
                       "strategy": "legacy_fallback", "multi_day_diag": multi_day_diag}

    front_ticks, front_month = select_front_month(ticks)
    if front_ticks.empty:
        return None, {"reason": "no_front_month_ticks", "used_cache": used_cache}

    bars, n_dropped = aggregate_ticks_to_bars(front_ticks, bar_minutes=bar_minutes)
    if bars.empty:
        return None, {"reason": "no_day_session_bars", "used_cache": used_cache,
                       "front_month": front_month, "n_night_session_dropped": n_dropped}

    return bars, {
        "reason": "ok", "used_cache": used_cache, "front_month": front_month,
        "n_ticks": int(len(front_ticks)), "n_night_session_dropped": n_dropped,
    }


def load_tx_hourly_bars(commodity_id: str = "TXF", refresh: bool = False):
    """
    向後相容包裝：1小時K棒版本，等價於 load_tx_bars(commodity_id, refresh, bar_minutes=60)。
    保留這個函式名稱/簽名是因為squeeze_kdj_tx_hourly_preview.py已經在呼叫，不因為新增
    通用版本而破壞既有呼叫端。
    """
    return load_tx_bars(commodity_id, refresh, bar_minutes=60)


def load_tx_15min_bars(commodity_id: str = "TXF", refresh: bool = False):
    """
    15分鐘K棒版本，等價於 load_tx_bars(commodity_id, refresh, bar_minutes=15)。
    給tx_intraday_engine.py/compare_tx_intraday.py用——日盤5小時 = 20根15分鐘K棒/天，
    見tx_intraday_engine.py模組docstring的樣本數估算。
    """
    return load_tx_bars(commodity_id, refresh, bar_minutes=15)
