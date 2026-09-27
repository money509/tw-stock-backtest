"""
基本面資料：每日個股本益比(PE)、殖利率(%)、股價淨值比(PB)。

現股波段策略([[equity_swing_engine]])用的三個新資料來源之一，信心程度：三個
新loader裡最高(這份報表是證交所官方JSON端點，欄位命名風格跟已經驗證過的
chip_data_loader.py的T86報表高度一致)，但仍然**沒有**在這個sandbox裡對真實
twse.com.tw端點實測過(對外連線被proxy allowlist擋掉)，第一次GitHub Actions
真實環境執行才是最終驗證。

資料來源：證交所官方「每日收盤價及月均價比、本益比、殖利率及股價淨值比」報表
https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU?date=YYYYMMDD&selectType=ALL&response=json
這個端點一次回傳「當天全部上市股票」的PE/PB/殖利率，跟T86同樣是「一次拿到
全市場」的彙總表，所以一樣用「逐日」迴圈抓取(不是逐股迴圈)。

已知限制：
1. 這是「逐日」bulk端點，不是「一次拿到全部歷史日期」的單一端點——已知證交所
   有一個「只回傳最新一天」的簡化版本(BWIBBU_ALL，不帶date參數)，但那個版本
   沒辦法回溯歷史，回測需要完整的歷史時間序列，所以這裡選擇跟T86同樣的
   「逐日帶date參數查詢」設計，犧牲一點下載時間，換取可以回測任意歷史區間。
   如果之後改成只需要「最新一天」的用途(例如純粹的選股掃描，不是回測)，
   可以改用不帶date的簡化端點，一次請求打完，不需要逐日迴圈。
2. 序列請求(不平行)、每次間隔REQUEST_DELAY_SECONDS秒——沿用chip_data_loader.py
   實測驗證過的教訓(這類TWSE報表端點對同時多連線極度敏感)，這裡還沒有機會
   對這個特定端點重新實測，是延用同一份經驗法則的保守假設，不是這個端點本身
   已經驗證過。
3. 虧損中的公司本益比欄位通常是空白/負值/"—"，代表「本益比沒有意義」不是
   資料缺失，這裡一律轉成NaN，不會當成0(0代表本益比剛好是0，語意上不對)。
4. stat != "OK" 代表確認當天非交易日，正常跳過，不是請求失敗；request本身
   逾時/連線錯誤才會觸發重試，兩者分開處理，理由跟chip_data_loader.py完全一樣。
5. 欄位名稱一律用「動態從當天payload的fields陣列查名字找位置」解析，不寫死
   index，避免報表格式微調時整批解析失敗或抓錯欄(同樣的教訓來自T86)。
"""
import os
import time
import datetime
import requests
import pandas as pd
import numpy as np

from network_utils import run_with_hard_timeout, HardTimeout

VALUATION_CACHE_DIR = os.path.join(os.path.dirname(__file__), "valuation_cache")
HARD_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3
REQUEST_DELAY_SECONDS = 1.5
THREAD_HARD_TIMEOUT_SECONDS = HARD_TIMEOUT_SECONDS + 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

CODE_COL_EXACT = "證券代號"
CLOSE_COL_CANDS = ("收盤價",)
PE_COL_CANDS = ("本益比",)
PB_COL_CANDS = ("股價淨值比",)
YIELD_COL_CANDS = ("殖利率",)


def _first_field_index(fields, exact=None, contains=None):
    if exact is not None:
        for i, f in enumerate(fields):
            if str(f).strip() == exact:
                return i
    if contains is not None:
        for i, f in enumerate(fields):
            if all(s in str(f) for s in contains):
                return i
    return None


def _resolve_column_indices(fields):
    code_i = _first_field_index(fields, exact=CODE_COL_EXACT, contains=("證券代號",))
    close_i = _first_field_index(fields, contains=CLOSE_COL_CANDS)
    pe_i = _first_field_index(fields, contains=PE_COL_CANDS)
    pb_i = _first_field_index(fields, contains=PB_COL_CANDS)
    yield_i = _first_field_index(fields, contains=YIELD_COL_CANDS)
    return {"code": code_i, "close": close_i, "pe": pe_i, "pb": pb_i, "yield": yield_i}


def _parse_float(s):
    """轉不出數字(空白/"—"/負值代表無意義的本益比欄位)一律回傳NaN，不是0。"""
    if s is None:
        return float("nan")
    s = str(s).strip().replace(",", "")
    if s in ("", "--", "-", "—", "N/A"):
        return float("nan")
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return v


class FetchFailed(Exception):
    """代表這次請求本身失敗(逾時/連線錯誤)，跟「查到stat!=OK的確認無交易」分開處理。"""
    pass


def fetch_valuation_day(date_str: str):
    """
    抓取單一天全市場的PE/PB/殖利率資料。
    回傳 dict {code: {close, pe, pb, dividend_yield}} 代表有資料；
    回傳 None 代表「確認當天無交易」(stat!=OK)；
    request本身失敗時丟出 FetchFailed，由呼叫端決定要不要重試。
    """
    url = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU"
    params = {"date": date_str, "selectType": "ALL", "response": "json"}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
        payload = resp.json()
    except Exception as e:
        raise FetchFailed(str(e))

    if payload.get("stat") != "OK":
        return None

    fields = payload.get("fields") or []
    col = _resolve_column_indices(fields) if fields else {
        "code": 0, "close": 2, "pe": 4, "pb": 6, "yield": 3,
    }

    def _safe_get(row, idx):
        if idx is None or idx >= len(row):
            return None
        return row[idx]

    result = {}
    for row in payload.get("data", []):
        code_raw = _safe_get(row, col["code"])
        if code_raw is None:
            continue
        code = str(code_raw).strip()
        if not code.isdigit():
            continue
        result[code] = {
            "close": _parse_float(_safe_get(row, col["close"])),
            "pe": _parse_float(_safe_get(row, col["pe"])),
            "pb": _parse_float(_safe_get(row, col["pb"])),
            "dividend_yield": _parse_float(_safe_get(row, col["yield"])),
        }
    return result


def _fetch_one_day_with_retry(day_str: str):
    for attempt in range(MAX_RETRIES + 1):
        try:
            data = run_with_hard_timeout(
                fetch_valuation_day, args=(day_str,), timeout=THREAD_HARD_TIMEOUT_SECONDS)
            return day_str, data, True
        except (FetchFailed, HardTimeout):
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            return day_str, None, False
    return day_str, None, False


def _trading_days_between(start: str, end: str) -> list:
    start_d = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    days = []
    cur = start_d
    while cur <= end_d:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y%m%d"))
        cur += datetime.timedelta(days=1)
    return days


def load_valuation_data(start: str, end: str, universe_codes: set = None, refresh: bool = False) -> dict:
    """
    回傳 {code: DataFrame(index=日期, columns=[close, pe, pb, dividend_yield,
    implied_eps, implied_book_value])}。

    implied_eps = close / pe (pe缺失/<=0時implied_eps為NaN，不強行計算)
    implied_book_value = close / pb (pb缺失/<=0時同樣為NaN)
    這兩個是這個模組額外提供的衍生欄位，方便量化EPS成長性/淨值成長性用，
    避免呼叫端各自重新處理除以零/缺值的邊界情況。

    每天的原始資料快取成一個小檔案，重複執行不會重新打API；序列處理+重試，
    邏輯跟chip_data_loader.load_chip_data()完全對稱(見兩份模組docstring)。
    """
    os.makedirs(VALUATION_CACHE_DIR, exist_ok=True)
    days = _trading_days_between(start, end)

    days_to_fetch = [
        d for d in days
        if refresh or not os.path.exists(os.path.join(VALUATION_CACHE_DIR, f"val_{d}.csv"))
    ]

    print(f"每日評價資料(PE/PB/殖利率)：{len(days)} 個平日，{len(days_to_fetch)} 天需要下載"
          f"(其餘已有快取)，序列處理中(每次間隔{REQUEST_DELAY_SECONDS}秒) ...", flush=True)

    fetched_count = no_data_count = failed_count = 0
    for done_n, day_str in enumerate(days_to_fetch, start=1):
        _, data, success = _fetch_one_day_with_retry(day_str)
        cache_path = os.path.join(VALUATION_CACHE_DIR, f"val_{day_str}.csv")

        if not success:
            failed_count += 1
        elif data is None:
            no_data_count += 1
            pd.DataFrame(columns=["code", "close", "pe", "pb", "dividend_yield"]).to_csv(cache_path, index=False)
        else:
            fetched_count += 1
            rows = [{"code": c, **v} for c, v in data.items()]
            pd.DataFrame(rows).to_csv(cache_path, index=False)

        if done_n % 50 == 0 or done_n == len(days_to_fetch):
            print(f"  進度 {done_n}/{len(days_to_fetch)} "
                  f"(成功{fetched_count} 無交易{no_data_count} 失敗待補{failed_count}) ...", flush=True)
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"每日評價資料下載完成：本次新抓{fetched_count}天，確認無交易{no_data_count}天，"
          f"逾時待補{failed_count}天{'（下次重跑會自動補抓這些天）' if failed_count else ''}", flush=True)

    per_stock_records = {}
    for day_str in days:
        cache_path = os.path.join(VALUATION_CACHE_DIR, f"val_{day_str}.csv")
        if not os.path.exists(cache_path):
            continue
        try:
            day_df = pd.read_csv(cache_path, dtype={"code": str})
        except Exception:
            continue
        if day_df.empty:
            continue
        date_ts = pd.Timestamp(datetime.datetime.strptime(day_str, "%Y%m%d"))
        for _, row in day_df.iterrows():
            code = str(row["code"])
            if universe_codes is not None and code not in universe_codes:
                continue
            per_stock_records.setdefault(code, []).append({
                "date": date_ts, "close": row["close"], "pe": row["pe"],
                "pb": row["pb"], "dividend_yield": row["dividend_yield"],
            })

    valuation_data = {}
    for code, records in per_stock_records.items():
        df = pd.DataFrame(records).set_index("date").sort_index()
        pe_valid = df["pe"].where(df["pe"] > 0)
        pb_valid = df["pb"].where(df["pb"] > 0)
        df["implied_eps"] = (df["close"] / pe_valid).replace([np.inf, -np.inf], np.nan)
        df["implied_book_value"] = (df["close"] / pb_valid).replace([np.inf, -np.inf], np.nan)
        valuation_data[code] = df
    return valuation_data
