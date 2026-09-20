"""
day_trading_loader.py
======================
台股「當日沖銷交易統計」(day-trading ratio) 資料下載模組。

資料來源（官方，已用 web_fetch 驗證過 JSON 結構，date=20260804）：
    https://www.twse.com.tw/exchangeReport/TWTB4U?response=json&date=YYYYMMDD&selectType=All

回傳 JSON 結構重點：
    res["tables"][0] -> 全市場加總（要跳過，不是個股資料）
    res["tables"][1] -> 個股資料，欄位（fields）:
        ["證券代號","證券名稱","暫停現股賣出後現款買進當沖註記",
         "當日沖銷交易成交股數","當日沖銷交易買進成交金額","當日沖銷交易賣出成交金額"]

    COL_DAY_TRADING_SHARES = 3  (當日沖銷交易成交股數，股數，非金額)

沿用本專案在 chip_data_loader.py 已證實有效的模式：
    - TWSE 這類「報表類」端點完全無法承受併發（3 條並發即 82~91% 失敗率），
      診斷腳本證實「序列請求 + 1.5~2 秒延遲」= 100% 成功率。
      因此本模組一律「序列處理」，不使用 ThreadPoolExecutor。
    - FetchFailed 例外代表「可重試」的失敗（逾時、連線錯誤、JSON 格式異常等）；
      若該日期是「確定沒有資料」（例如非交易日、stat != OK 且訊息明確表示無資料），
      回傳 None，不重試、不快取為失敗。
    - 每日結果快取為一個 CSV（day_trading_cache/YYYYMMDD.csv），下次執行時若已存在
      且不要求 refresh，直接讀快取，不重新打 API。
    - universe_codes 過濾：TWTB4U 的 selectType=All 理論上已經是個股層級，但為了與
      chip_data_loader.py 的行為一致、並避免萬一遇到權證/ETF 等非目標代碼混入，
      仍然提供 universe_codes 參數，可在載入時過濾成只保留目標期貨標的池。
"""

import os
import time
import random
import datetime
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 常數設定
# ---------------------------------------------------------------------------

TWTB4U_URL = "https://www.twse.com.tw/exchangeReport/TWTB4U"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.twse.com.tw/zh/trading/day-trading/twtb4u.html",
}

CACHE_DIR = "day_trading_cache"

REQUEST_DELAY_SEC = 1.5          # 序列請求之間的延遲（已證實 100% 成功率的節奏）
REQUEST_TIMEOUT_SEC = 15
MAX_RETRIES = 3                  # 每個日期最多重試次數（不含第一次嘗試）
RETRY_BACKOFF_BASE_SEC = 3.0     # 重試前的基礎延遲，每次重試遞增

# tables[1] 的欄位 index（已用實際回應驗證）
COL_CODE = 0
COL_NAME = 1
COL_DAY_TRADING_SHARES = 3       # 當日沖銷交易成交股數


class FetchFailed(Exception):
    """代表這次下載「可重試」的失敗（逾時、連線錯誤、非預期的回應格式等）。"""
    pass


# ---------------------------------------------------------------------------
# 單日下載
# ---------------------------------------------------------------------------

def _fetch_one_day(date_str):
    """
    下載單一交易日的當日沖銷交易統計。

    回傳：
        - list[dict]：該日各股票的當沖資料（成功且有資料）
        - []（空 list）：該日確定沒有資料（例如非交易日）——不會被視為錯誤
        - 拋出 FetchFailed：代表值得重試的失敗（逾時、連線錯誤、JSON 格式異常）
    """
    params = {
        "response": "json",
        "date": date_str,
        "selectType": "All",
    }
    try:
        resp = requests.get(
            TWTB4U_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT_SEC
        )
    except requests.exceptions.RequestException as e:
        raise FetchFailed(f"{date_str}: 連線錯誤 {e}")

    if resp.status_code != 200:
        raise FetchFailed(f"{date_str}: HTTP {resp.status_code}")

    try:
        data = resp.json()
    except ValueError:
        raise FetchFailed(f"{date_str}: 回應非合法 JSON")

    stat = data.get("stat", "")

    # 官方對「非交易日 / 無資料」通常回傳 stat 不是 OK，且訊息會提到查無資料。
    # 這種情況視為「確定沒有資料」，不重試。
    if stat != "OK":
        msg = str(data.get("stat", "")) + str(data.get("statMsg", data.get("msg", "")))
        no_data_markers = ("查無資料", "無此資料", "很抱歉", "OFF")
        if any(marker in msg for marker in no_data_markers) or stat == "":
            return []
        # 其餘未知的非 OK 狀態，保守起見視為可重試的失敗
        raise FetchFailed(f"{date_str}: stat={stat} msg={msg}")

    tables = data.get("tables", [])
    if len(tables) < 2:
        # 結構跟預期不符（tables[0]=全市場加總, tables[1]=個股），可能是暫時性異常
        raise FetchFailed(f"{date_str}: tables 數量異常 ({len(tables)})")

    per_stock_table = tables[1]
    rows = per_stock_table.get("data", [])

    results = []
    for row in rows:
        try:
            code = str(row[COL_CODE]).strip()
            name = str(row[COL_NAME]).strip()
            shares_raw = str(row[COL_DAY_TRADING_SHARES]).replace(",", "").strip()
            shares = int(shares_raw) if shares_raw not in ("", "--", "-") else 0
        except (IndexError, ValueError, TypeError):
            continue

        if not code:
            continue

        results.append({
            "date": date_str,
            "code": code,
            "name": name,
            "day_trading_shares": shares,
        })

    return results


def _fetch_one_day_with_retry(date_str):
    """
    對單一日期執行「序列請求 + 重試」邏輯。
    只有 FetchFailed（可重試錯誤）才會重試；「確定無資料」的 [] 直接回傳，不重試。
    """
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _fetch_one_day(date_str)
        except FetchFailed as e:
            last_err = e
            if attempt < MAX_RETRIES:
                backoff = RETRY_BACKOFF_BASE_SEC * (attempt + 1) + random.uniform(0, 1.0)
                print(f"  ⚠️  {date_str} 下載失敗（第 {attempt+1} 次）：{e}，{backoff:.1f}s 後重試...")
                time.sleep(backoff)
            else:
                print(f"  ❌ {date_str} 下載失敗，已達最大重試次數（{MAX_RETRIES}），放棄此日。")
    # 全部重試都失敗：拋出最後一次的錯誤，讓呼叫端決定如何處理
    raise last_err


# ---------------------------------------------------------------------------
# 快取存取
# ---------------------------------------------------------------------------

def _cache_path(date_str):
    return os.path.join(CACHE_DIR, f"{date_str}.csv")


def _load_from_cache(date_str):
    path = _cache_path(date_str)
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"code": str, "date": str})
        return df
    return None


def _save_to_cache(date_str, rows):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(date_str)
    df = pd.DataFrame(rows, columns=["date", "code", "name", "day_trading_shares"])
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# 主要對外介面
# ---------------------------------------------------------------------------

def get_trading_days(start_date, end_date):
    """回傳 start_date ~ end_date（含）之間的所有平日（週六日排除，假日未排除，
    交易所若回傳無資料會在 _fetch_one_day 內被判定為空清單並跳過）。"""
    days = []
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y%m%d"))
        cur += datetime.timedelta(days=1)
    return days


def load_day_trading_data(start_date, end_date, universe_codes=None, refresh=False):
    """
    下載（或讀取快取）start_date ~ end_date 之間，每日的當日沖銷交易統計。

    參數：
        start_date, end_date : datetime.date
        universe_codes        : 可選，set/list of str。若提供，只保留這些代碼的資料
                                 （沿用 chip_data_loader.py 的 universe 過濾模式，
                                   避免非目標代碼污染資料集）。
        refresh                : 若 True，忽略既有快取，強制重新下載所有日期。

    回傳：
        pandas.DataFrame，欄位：date, code, name, day_trading_shares
        （date 為字串 YYYYMMDD，code 為字串）
    """
    days = get_trading_days(start_date, end_date)
    universe_set = set(universe_codes) if universe_codes else None

    all_rows = []
    total = len(days)
    failed_days = []

    for idx, date_str in enumerate(days):
        if not refresh:
            cached = _load_from_cache(date_str)
            if cached is not None:
                all_rows.append(cached)
                continue

        print(f"[{idx+1}/{total}] 下載當沖統計 {date_str} ...")
        try:
            rows = _fetch_one_day_with_retry(date_str)
        except FetchFailed as e:
            print(f"  ❌ {date_str} 最終失敗，跳過（不快取）：{e}")
            failed_days.append(date_str)
            time.sleep(REQUEST_DELAY_SEC)
            continue

        # 空清單（確定無資料，例如非交易日）也快取起來，避免下次重複打 API
        _save_to_cache(date_str, rows)
        if rows:
            all_rows.append(pd.DataFrame(rows))

        time.sleep(REQUEST_DELAY_SEC)

    if failed_days:
        print(f"\n⚠️  共有 {len(failed_days)} 天下載失敗（已跳過，未快取）：{failed_days}")

    if not all_rows:
        return pd.DataFrame(columns=["date", "code", "name", "day_trading_shares"])

    df = pd.concat(all_rows, ignore_index=True)
    df["code"] = df["code"].astype(str)
    df["date"] = df["date"].astype(str)

    if universe_set is not None:
        before = len(df)
        df = df[df["code"].isin(universe_set)].reset_index(drop=True)
        after = len(df)
        print(f"✅ universe_codes 過濾：{before} -> {after} 筆")

    return df


def compute_day_trading_ratio(day_trading_df, volume_df):
    """
    計算「當沖比例」= 當日沖銷交易成交股數 / 該股當日總成交股數。

    參數：
        day_trading_df : load_day_trading_data() 的回傳結果 (date, code, name, day_trading_shares)
        volume_df      : 必須有 date, code, volume（股數，非張數）欄位的 DataFrame，
                          用來對齊當日總成交量

    回傳：
        DataFrame，欄位：date, code, day_trading_shares, volume, day_trading_ratio
        (day_trading_ratio 為 0~1 之間的浮點數；volume 為 0 或缺值時 ratio 設為 NaN)
    """
    merged = day_trading_df.merge(
        volume_df[["date", "code", "volume"]],
        on=["date", "code"],
        how="left",
    )
    merged["day_trading_ratio"] = merged.apply(
        lambda r: (r["day_trading_shares"] / r["volume"])
        if pd.notna(r["volume"]) and r["volume"] > 0
        else float("nan"),
        axis=1,
    )
    return merged
