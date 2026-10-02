"""
台指期(TX/大台指期貨) 1分鐘K棒 —— 多年期歷史資料(非官方、社群維護) —— 資料來源
信心程度：**未經驗證(UNVERIFIED)**，跟taifex_intraday_loader.py同一個等級的不確定性，
原因見下方。這支模組是taifex_intraday_loader.py(官方免費「前30個交易日」逐筆成交)
的**另一個獨立資料來源**，用來補足官方端點「只有最近30個交易日」這個先天限制——
官方端點沒辦法拿到更久以前的歷史資料，這支模組改用一個涵蓋20多年的社群資料集，
但代價是「非官方、格式完全沒有實測過」，見下方`validate_tx_history_accuracy.py`
就是為了在真的拿這份資料做回測之前，先跟官方日線資料對比，看這份資料到底準不準。

⚠️⚠️ 為什麼是「未經驗證」：這個sandbox環境對外連線被proxy allowlist擋掉，連
Google Drive都連不到，所以連「下載下來的檔案長什麼樣子(欄位/分隔符號/編碼/
有沒有header/是不是壓縮檔)」都沒有機會實際看過一眼——下面的下載/解析邏輯是
「防禦性猜測」，不是已驗證的行為。**第一次在GitHub Actions真實環境執行這支
loader，才是它的第一次真正整合測試**，失敗是預期中可能發生的事，不代表設計錯誤。

資料來源(使用者指定)：crazyindicator.pixnet.net(一個長期經營的台股/期貨技術指標
部落格，同一個作者也發MT4/MT5指標文章，是一個持續在維護的部落格，不是單次隨便
找到的網站)上公開分享的台指期1分鐘K棒歷史資料，以檔案形式放在Google Drive上。
部落格上寫這份資料橫跨20多年、全部壓縮後總共約10MB——跟「純文字/CSV經壓縮」的
檔案大小量級吻合(不是什麼特殊的二進位格式)，這是唯一能在「完全沒看過檔案內容」
的情況下，拿來推斷格式的間接線索。

已知的Google Drive分享連結(見GDRIVE_FILES)，**刻意跳過**「調整/連續合約價差版」
那個檔案——使用者指定先只用「原始逐口合約版」就夠，連續合約版之後有需要再說。

由於完全不知道檔案內部格式，`parse_history_file()`用內容特徵(sniff)猜測，不是
寫死欄位位置/名稱：
  - 日期欄：多數非空值符合8位數字(YYYYMMDD)、且落在1990~2035年這個合理範圍
  - 時間欄：多數非空值符合3~6位數字、補零成6位後是合法的HHMMSS
  - 價格欄：數值欄位中位數落在3000~25000(台指期點數的合理範圍)
  - 成交量欄：數值欄位中位數是個偏小的非負整數(不在價格範圍內)
猜不出必要欄位(日期/時間/至少一個價格欄)時，回傳明確的reason，**絕不**硬著頭皮
把猜錯的欄位當成真的資料、產生看起來像樣但其實是垃圾的K棒——這是使用者在規格裡
明確要求的「寧可大聲失敗，也不要悄悄產生垃圾資料」原則，跟repo其他loader一樣。

這份資料的「資料細度」本身也不確定：如果偵測到剛好4個價格欄位，這裡假設它們
依照原始欄位的左到右順序就是開高低收(Open/High/Low/Close)——這是「1分鐘K棒」
類資料集最常見的欄位順序慣例，但**沒有實測驗證過**，如果GitHub Actions真的跑起來
之後發現K棒本身不自洽(例如High < Low)，代表這個順序假設可能是錯的，需要回來重看。
如果只偵測到1個價格欄位，則當成逐筆成交(tick-level)處理，沿用
taifex_intraday_loader.py的aggregate_ticks_to_bars()聚合成K棒(不重複實作聚合邏輯)。

安全檢查(`check_file_signature()`)：下載下來、要真的拿去解析之前，先看檔案開頭
幾KB有沒有看起來像執行檔(PE的MZ開頭/ELF開頭)、shebang腳本(#!開頭)、或HTML/JS
(代表Google Drive可能回傳了一個錯誤頁/登入頁，不是真正的檔案內容)。**誠實聲明
這個檢查能做到/不能做到什麼**：這只是一個「防禦性查看檔頭簽章」的基本檢查，
**不是**防毒掃描，偵測不到偽裝成純文字的惡意內容、不會掃描巨集/腳本注入、也不會
掃任何payload本身的惡意邏輯。真正的安全防線不是這個檢查，而是**這整個流程只在
GitHub Actions的ephemeral、用完即丟的runner裡執行**，不是在使用者自己的電腦上——
就算這個簡單簽章檢查漏放了什麼，影響範圍也只是一個跑完就銷毀的CI容器，不會碰到
使用者自己的機器/帳號，這才是這裡真正依靠的安全假設，這個函式只是順手多一層
「盡量不要因為自己的疏忽，把一個明顯不是資料的東西硬塞進parser」的基本檢查而已。

快取：下載下來的原始檔案存到cache_dir(預設taifex_history_cache/)，用period_key
當檔名，重跑時(refresh=False)直接用快取，不重新打Google Drive——這份資料是
「已經結束的歷史區間」，不像taifex_intraday_loader.py的「滾動30天窗口」每天
內容都會變，所以快取可以長期有效，不需要用日期當檔名。
"""
import os
import io
import zipfile
import datetime

import pandas as pd

from network_utils import run_with_hard_timeout, HardTimeout
from taifex_intraday_loader import aggregate_ticks_to_bars, select_front_month

HARD_TIMEOUT_SECONDS = 120  # 檔案可能有幾MB，比打一般API的timeout長一些
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 5
THREAD_HARD_TIMEOUT_SECONDS = HARD_TIMEOUT_SECONDS + 15

TAIFEX_HISTORY_CACHE_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "taifex_history_cache")

# Google Drive檔案ID(從使用者提供的分享連結擷取)，見模組docstring。刻意跳過
# 「調整/連續合約價差版」那個檔案(使用者指定先不用)。
GDRIVE_FILES = {
    "1998_2000": {"id": "1xB1bvwBDtaoUEAobmcRUTHNcyOZodTz0", "range": "1998.07.22~2000.12.31"},
    "2001_2010": {"id": "1762OrBEo7q5B6YgM2DqXKY0J2ykIxBsw", "range": "2001.01.01~2010.12.31"},
    "2011_2020": {"id": "1lsam29dX2n8oPOP25SfyZIgeMJAGKpeK", "range": "2011.01.01~2020.12.31"},
    "2021_2023": {"id": "1VOqnu11Tarn1IVZrO6Y6sdX7_YwG1ZEP", "range": "2021.01.01~2023.12.31"},
}

# 價格欄位猜測用的合理範圍(台指期點數)——寬一點，涵蓋1998年以來的歷史低點到近年高點。
PRICE_RANGE_MIN = 1000
PRICE_RANGE_MAX = 30000
# 日期欄位猜測用的合理年份範圍。
DATE_YEAR_MIN = 1990
DATE_YEAR_MAX = 2035


class FetchFailed(Exception):
    """代表這次下載本身失敗(逾時/連線錯誤/gdown丟例外)，由呼叫端決定要不要重試。跟
    「下載到東西了，但格式不是預期的/簽章看起來不安全」(那些用reason字串表達)是
    兩件不同的事，跟taifex_intraday_loader.py的FetchFailed同樣的區分精神。"""
    pass


def check_file_signature(local_path):
    """
    防禦性檔頭簽章檢查，**不是**防毒掃描(見模組docstring「安全檢查」段落，
    誠實列出這個函式能做到/不能做到什麼)。只看前4KB有沒有像：
      - PE執行檔(MZ開頭)
      - ELF執行檔(\\x7fELF開頭)
      - shebang腳本(#!開頭)
      - HTML/JS(代表可能拿到的是Google Drive的錯誤頁/登入頁，不是真正檔案)
    回傳(is_safe: bool, reason)，reason in {"ok", "empty_file",
    "unsafe_file_signature", "signature_check_io_error:<detail>"}。
    """
    try:
        with open(local_path, "rb") as f:
            head = f.read(4096)
    except Exception as e:
        return False, f"signature_check_io_error:{e}"

    if not head:
        return False, "empty_file"
    if head[:2] == b"MZ":
        return False, "unsafe_file_signature"
    if head[:4] == b"\x7fELF":
        return False, "unsafe_file_signature"
    if head.lstrip()[:2] == b"#!":
        return False, "unsafe_file_signature"
    head_lower = head.lower()
    if b"<html" in head_lower or b"<script" in head_lower or b"<!doctype html" in head_lower:
        # zip檔案(PK開頭)不受這條影響，b"<html"不會出現在一般zip二進位檔頭裡；
        # 這裡刻意沒有特別排除zip，因為zip的二進位內容本來就不太可能剛好包含這幾個
        # 字樣，若真的誤判，使用者可以從印出的診斷訊息看到是這條規則擋下來的。
        return False, "unsafe_file_signature"
    return True, "ok"


def _gdown_download(file_id, output_path):
    """實際呼叫gdown下載單一檔案(可能觸發Google Drive「檔案過大、無法掃描病毒」的
    確認頁，gdown會自動處理這個確認token流程，不用自己手刻)。
    回傳gdown回報的輸出路徑；下載失敗時gdown可能回傳None或直接丟例外，兩種情況
    呼叫端(_download_with_retry)都當作失敗處理。

    ⚠️修正記錄：原本多傳了fuzzy=True，這個參數是給「直接丟整個分享網址、讓gdown
    自己從網址裡猜file id」用的；這裡我們已經自己從網址解析出file_id、用id=直接
    指定，不需要fuzzy模式。而且目前requirements.txt裝的是較新版gdown(6.x)，
    這個版本的download()根本沒有fuzzy這個參數了(舊版才有)，傳了會直接丟
    TypeError: unexpected keyword argument，這是第一次在GitHub Actions
    真實環境跑才發現的，拿掉fuzzy就是正確用法。"""
    import gdown
    return gdown.download(id=file_id, output=output_path, quiet=False)


def _download_with_retry(file_id, tmp_path):
    """回傳(success: bool, reason)。success=False代表逾時/連線層級失敗(該重試)；
    success=True但tmp_path實際上沒有產生出檔案時，呼叫端會再判一次。"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = run_with_hard_timeout(
                _gdown_download, args=(file_id, tmp_path), timeout=THREAD_HARD_TIMEOUT_SECONDS)
            if result is None or not os.path.exists(tmp_path):
                last_reason = "download_error"
            else:
                return True, "ok"
        except HardTimeout:
            last_reason = "download_timeout"
        except Exception as e:
            last_reason = f"download_error:{e}"

        if attempt < MAX_RETRIES:
            import time
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue
        return False, last_reason
    return False, "download_error"


def download_period(period_key, cache_dir=TAIFEX_HISTORY_CACHE_DIR_DEFAULT, refresh=False):
    """
    下載(或讀快取)GDRIVE_FILES[period_key]對應的歷史資料檔案，回傳
    (local_path_or_None, reason)：
      reason == "ok"                     剛下載成功
      reason == "cached"                 直接用快取，沒有重新打Google Drive
      reason == "unknown_period"         period_key不在GDRIVE_FILES裡
      reason == "download_error"/"download_error:<detail>"/"download_timeout"
                                          下載失敗(見_download_with_retry)
      reason == "unsafe_file_signature"  下載到的內容沒通過check_file_signature()
                                          (見模組docstring，可能是Google Drive
                                          回傳了錯誤頁/登入頁而不是真正檔案)

    快取的原始檔案存在cache_dir/{period_key}.raw，見模組docstring「快取」段落。
    """
    if period_key not in GDRIVE_FILES:
        return None, "unknown_period"

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{period_key}.raw")

    if not refresh and os.path.exists(cache_path):
        is_safe, sig_reason = check_file_signature(cache_path)
        if not is_safe:
            print(f"[taifex_history_loader] {period_key}: 快取檔案未通過簽章檢查({sig_reason})，"
                  f"視為失敗(不會嘗試解析)")
            return None, sig_reason
        print(f"[taifex_history_loader] {period_key}: 使用快取 {cache_path}")
        return cache_path, "cached"

    file_id = GDRIVE_FILES[period_key]["id"]
    tmp_path = cache_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    success, reason = _download_with_retry(file_id, tmp_path)
    if not success:
        print(f"[taifex_history_loader] {period_key}: 下載失敗({reason})")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None, reason

    is_safe, sig_reason = check_file_signature(tmp_path)
    if not is_safe:
        print(f"[taifex_history_loader] {period_key}: 下載內容未通過簽章檢查({sig_reason})，"
              f"不會存入快取、不會嘗試解析")
        os.remove(tmp_path)
        return None, sig_reason

    os.replace(tmp_path, cache_path)
    print(f"[taifex_history_loader] {period_key}: 下載成功，存入快取 {cache_path}")
    return cache_path, "ok"


def _read_candidate(raw_bytes, encoding, sep, header):
    """嘗試用給定的(encoding, sep, header)組合把raw_bytes解析成DataFrame，
    失敗(任何例外，或解析出來不到2欄/0列)回傳None。"""
    try:
        df = pd.read_csv(
            io.BytesIO(raw_bytes), encoding=encoding, sep=sep,
            header=0 if header else None, dtype=str, engine="python",
        )
    except Exception:
        return None
    if df is None or df.shape[1] < 2 or df.shape[0] < 1:
        return None
    return df


def _sniff_table(raw_bytes):
    """對原始內容嘗試多組(編碼, 分隔符, 有無header)組合，回傳第一個「看起來像一張
    有意義表格」的DataFrame，或None(所有組合都失敗/都不像樣)。欄位名稱統一轉成
    字串(header=None時pandas給的是整數欄名，轉字串方便後面統一處理)。"""
    for encoding in ("utf-8", "big5", "cp950"):
        for sep in (",", "\t", r"\s+", ";"):
            for header in (True, False):
                df = _read_candidate(raw_bytes, encoding, sep, header)
                if df is not None:
                    df.columns = [str(c) for c in df.columns]
                    return df
    return None


def _col_matches_date(series):
    s = series.astype(str).str.strip().str.replace(r"[/-]", "", regex=True)
    digits8 = s.str.match(r"^\d{8}$")
    if digits8.mean() < 0.8:
        return False
    years = pd.to_numeric(s[digits8].str.slice(0, 4), errors="coerce")
    valid_year = years.between(DATE_YEAR_MIN, DATE_YEAR_MAX)
    return valid_year.mean() >= 0.8 if len(years) else False


def _col_matches_time(series):
    s = series.astype(str).str.strip()
    digits = s.str.match(r"^\d{3,6}$")
    if digits.mean() < 0.8:
        return False
    padded = s[digits].str.zfill(6)
    hh = pd.to_numeric(padded.str.slice(0, 2), errors="coerce")
    mm = pd.to_numeric(padded.str.slice(2, 4), errors="coerce")
    ss = pd.to_numeric(padded.str.slice(4, 6), errors="coerce")
    valid = hh.between(0, 23) & mm.between(0, 59) & ss.between(0, 59)
    return valid.mean() >= 0.8 if len(hh) else False


def _numeric_col(series):
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def _detect_columns(df):
    """在DataFrame的每一欄上套用內容特徵猜測，回傳dict：
      {"date": col_or_None, "time": col_or_None,
       "price_cols": [col, ...](依原始左到右順序), "volume": col_or_None}
    見模組docstring「sniff」段落列出的猜測規則。"""
    date_col = None
    time_col = None
    price_cols = []
    numeric_candidates = []

    for col in df.columns:
        series = df[col]
        if date_col is None and _col_matches_date(series):
            date_col = col
            continue
        if time_col is None and _col_matches_time(series):
            time_col = col
            continue
        numeric = _numeric_col(series)
        valid_ratio = numeric.notna().mean()
        if valid_ratio >= 0.8:
            numeric_candidates.append((col, numeric))

    volume_col = None
    for col, numeric in numeric_candidates:
        median = numeric.median()
        if PRICE_RANGE_MIN <= median <= PRICE_RANGE_MAX:
            price_cols.append(col)
        elif volume_col is None and median >= 0 and median < PRICE_RANGE_MIN:
            # 數值欄、但中位數不在價格合理範圍內、是非負值 -> 猜是成交量
            # (這是個偏弱的猜測，成交量欄是「可選」資訊，猜不到也不影響日期/時間/
            # 價格欄的判斷，見parse_history_file的missing-volume處理)。
            volume_col = col

    return {"date": date_col, "time": time_col, "price_cols": price_cols, "volume": volume_col}


def parse_history_file(local_path):
    """
    把download_period()下載下來的原始檔案解析成統一格式的DataFrame。完全不假設
    已知欄位名稱/分隔符號/編碼/是否有header——用內容特徵猜測(見模組docstring)。

    回傳(df_or_None, reason)：
      reason == "ok"
      reason == "empty_file"             檔案大小是0，或簽章檢查判定是空檔案
      reason == "unsafe_file_signature"  檔頭簽章檢查沒通過(見check_file_signature)
      reason == "unrecognized_format"    所有(編碼,分隔符,header)組合都解析不出
                                          一張像樣的表格，或價格欄位數量既不是1個
                                          (逐筆成交)也不是4個(OHLC)，無法判斷資料細度
      reason == "no_date_col"            解析出表格，但猜不出哪一欄是日期
      reason == "no_time_col"            猜不出哪一欄是時間
      reason == "no_price_col"           猜不出任何一欄是價格(更別說1個或4個)

    成功時，回傳的DataFrame有一個額外屬性df.attrs["source_level"]：
      "tick" —— 偵測到1個價格欄位，欄位為ProductCode/ContractMonth/Timestamp/
                 Price/Volume，跟taifex_intraday_loader.py的tick DataFrame同樣格式
                 (ProductCode/ContractMonth猜不到時填None，呼叫端
                 load_tx_history_bars()會視情況略過前月合約篩選)。
      "bar"  —— 偵測到4個價格欄位(假設左到右順序是開高低收，見模組docstring)，
                 欄位為BarStart(index)/Open/High/Low/Close/Volume，直接就是
                 1分鐘K棒，不需要再聚合。
    """
    try:
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            return None, "empty_file"
        with open(local_path, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        return None, f"io_error:{e}"

    is_safe, sig_reason = check_file_signature(local_path)
    if not is_safe:
        return None, sig_reason

    if raw_bytes[:2] == b"PK":  # zip檔案，取第一個成員內容來sniff(見模組docstring)
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
            members = [n for n in zf.namelist() if not n.endswith("/")]
            if not members:
                return None, "unrecognized_format"
            raw_bytes = zf.read(members[0])
        except Exception:
            return None, "unrecognized_format"

    df = _sniff_table(raw_bytes)
    if df is None:
        return None, "unrecognized_format"

    detected = _detect_columns(df)
    if detected["date"] is None:
        return None, "no_date_col"
    if detected["time"] is None:
        return None, "no_time_col"
    if not detected["price_cols"]:
        return None, "no_price_col"

    date_s = df[detected["date"]].astype(str).str.strip().str.replace(r"[/-]", "", regex=True)
    time_s = df[detected["time"]].astype(str).str.strip().str.zfill(6)
    timestamp = pd.to_datetime(date_s + time_s, format="%Y%m%d%H%M%S", errors="coerce")

    volume_col = detected["volume"]
    if volume_col is not None:
        volume = _numeric_col(df[volume_col])
    else:
        print("[taifex_history_loader] 警告：猜不出成交量欄位，Volume一律填0"
              "(不影響Open/High/Low/Close，但量能類訊號不能用這份資料算)")
        volume = pd.Series(0.0, index=df.index)

    n_price_cols = len(detected["price_cols"])
    if n_price_cols == 1:
        price = _numeric_col(df[detected["price_cols"][0]])
        out = pd.DataFrame({
            "ProductCode": "TXF",
            "ContractMonth": None,
            "Timestamp": timestamp,
            "Price": price,
            "Volume": volume,
        })
        out = out.dropna(subset=["Timestamp", "Price"])
        if out.empty:
            return None, "unrecognized_format"
        out.attrs["source_level"] = "tick"
        return out, "ok"

    if n_price_cols >= 4:
        open_c, high_c, low_c, close_c = detected["price_cols"][:4]
        out = pd.DataFrame({
            "BarStart": timestamp,
            "Open": _numeric_col(df[open_c]),
            "High": _numeric_col(df[high_c]),
            "Low": _numeric_col(df[low_c]),
            "Close": _numeric_col(df[close_c]),
            "Volume": volume,
        })
        out = out.dropna(subset=["BarStart", "Open", "High", "Low", "Close"])
        if out.empty:
            return None, "unrecognized_format"
        out = out.set_index("BarStart").sort_index()
        out.attrs["source_level"] = "bar"
        return out, "ok"

    # 價格欄位數量不是1也不是4(例如2或3個)，猜不出這是逐筆成交還是OHLC K棒，
    # 誠實回報格式無法辨識，不要亂猜("寧可大聲失敗"原則，見模組docstring)。
    return None, "unrecognized_format"


def _resample_bar_level(bars, bar_minutes):
    """把已經是bar-level(例如1分鐘)的OHLCV資料，重新聚合成bar_minutes分鐘的K棒。
    跟aggregate_ticks_to_bars()的K棒定義完全一致(Open=第一根的Open/High=期間最高/
    Low=期間最低/Close=最後一根的Close/Volume=期間加總)，只是輸入已經是K棒不是
    逐筆成交，所以用pandas resample而不是逐筆分組——bar_minutes==1時直接原樣回傳，
    不多做一次沒必要的resample。"""
    if bars.empty or bar_minutes == 1:
        return bars
    rule = f"{bar_minutes}min"
    agg = bars.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    })
    agg = agg.dropna(subset=["Open", "High", "Low", "Close"])
    return agg


def load_tx_history_bars(period_keys, bar_minutes=15, refresh=False,
                          cache_dir=TAIFEX_HISTORY_CACHE_DIR_DEFAULT):
    """
    整合入口：對每個period_key依序下載(或讀快取) -> 解析 -> (tick-level時)挑近月
    合約、聚合成K棒 / (bar-level時)視需要重新聚合成bar_minutes分鐘K棒，最後把所有
    period串接起來、按時間排序、去重複。

    單一period失敗(下載失敗/格式辨識不出來)**不會**讓整個函式中斷，只會跳過那個
    period、繼續處理其他period，並在diagnostics裡誠實記錄失敗原因——使用者要求的
    行為(某一段區間資料有問題，不該連累其他段已經能用的資料)。

    回傳(bars_df_or_None, diagnostics: dict)：
      diagnostics["periods"][period_key] = {"reason": ..., "n_rows": int,
                                             "source_level": "tick"/"bar"/None}
      diagnostics["n_periods_ok"] / diagnostics["n_periods_failed"]
      全部period都失敗時，回傳(None, diagnostics)，diagnostics裡仍然列出每個
      period個別失敗的原因，呼叫端可以印出來讓人判斷是哪裡出問題。
    """
    diagnostics = {"periods": {}, "n_periods_ok": 0, "n_periods_failed": 0}
    all_bars = []

    for period_key in period_keys:
        local_path, dl_reason = download_period(period_key, cache_dir=cache_dir, refresh=refresh)
        if local_path is None:
            print(f"[taifex_history_loader] {period_key}: 下載階段失敗({dl_reason})，跳過這段")
            diagnostics["periods"][period_key] = {
                "reason": dl_reason, "n_rows": 0, "source_level": None}
            diagnostics["n_periods_failed"] += 1
            continue

        parsed, parse_reason = parse_history_file(local_path)
        if parsed is None:
            print(f"[taifex_history_loader] {period_key}: 解析階段失敗({parse_reason})，跳過這段")
            diagnostics["periods"][period_key] = {
                "reason": parse_reason, "n_rows": 0, "source_level": None}
            diagnostics["n_periods_failed"] += 1
            continue

        source_level = parsed.attrs.get("source_level")
        try:
            if source_level == "tick":
                ticks = parsed
                if ticks["ContractMonth"].notna().any() and ticks["ContractMonth"].nunique() > 1:
                    ticks, _front_month = select_front_month(ticks)
                bars, _n_dropped = aggregate_ticks_to_bars(ticks, bar_minutes=bar_minutes,
                                                            session_start="00:00", session_end="23:59")
            else:  # "bar"
                bars = _resample_bar_level(parsed, bar_minutes)
        except Exception as e:
            print(f"[taifex_history_loader] {period_key}: 聚合階段發生例外({e})，跳過這段")
            diagnostics["periods"][period_key] = {
                "reason": f"aggregation_error:{e}", "n_rows": 0, "source_level": source_level}
            diagnostics["n_periods_failed"] += 1
            continue

        if bars is None or bars.empty:
            diagnostics["periods"][period_key] = {
                "reason": "empty_after_aggregation", "n_rows": 0, "source_level": source_level}
            diagnostics["n_periods_failed"] += 1
            continue

        diagnostics["periods"][period_key] = {
            "reason": "ok", "n_rows": int(len(bars)), "source_level": source_level}
        diagnostics["n_periods_ok"] += 1
        all_bars.append(bars)

    if not all_bars:
        return None, diagnostics

    combined = pd.concat(all_bars).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined, diagnostics
