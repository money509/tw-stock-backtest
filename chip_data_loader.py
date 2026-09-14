"""
籌碼面資料：三大法人(外資/投信/自營商)每日買賣超。

資料來源：證交所官方 T86 報表
https://www.twse.com.tw/rwd/zh/fund/T86?date=YYYYMMDD&selectType=ALL&response=json
這個端點一次回傳「當天全部上市股票」的三大法人買賣超，不是一檔一檔查，
所以用「逐日」迴圈抓取，不是像股價那樣「逐股」迴圈。

已知限制：
1. 這個端點目前只確認格式正確、有回傳資料，但受限於工具本身的快取問題，
   沒辦法在開發階段直接驗證「查詢歷史日期是否真的回傳那天的資料」，
   第一次正式執行時務必先檢查抓到的資料日期是否正確，不要照單全收。
2. 目前只涵蓋上市(TWSE)股票，上櫃(TPEx)的三大法人資料用的是不同端點，
   這版本還沒有實作，上櫃股票的籌碼面訊號會是空值。
3. 每日一次請求，遇到假日/非交易日會回傳 stat != "OK"，正常跳過即可，
   這跟「請求逾時/失敗」是兩件不同的事，必須分開處理(見下方重試邏輯)，
   否則逾時的交易日會被誤存成「沒有資料」，永久遺失那天的真實資料。
"""
import os
import time
import datetime
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

CHIP_CACHE_DIR = os.path.join(os.path.dirname(__file__), "chip_cache")
HARD_TIMEOUT_SECONDS = 15
MAX_WORKERS = 8          # 平行下載的執行緒數，加速用
MAX_RETRIES = 2          # 請求失敗(非"確認無交易")時的重試次數

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# T86 回傳欄位裡，我們關心的幾個索引位置 (對照官方fields順序)
COL_CODE = 0
COL_FOREIGN_NET = 4    # 外陸資買賣超股數(不含外資自營商)
COL_TRUST_NET = 10     # 投信買賣超股數
COL_DEALER_NET = 11    # 自營商買賣超股數(合計)
COL_TOTAL_NET = 18     # 三大法人買賣超股數(合計)


def _parse_int(s: str) -> int:
    try:
        return int(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return 0


class FetchFailed(Exception):
    """代表這次請求本身失敗(逾時/連線錯誤)，跟「查到stat!=OK的確認無交易」要分開處理，
    前者應該重試，後者不該重試(是正常的休市日)。"""
    pass


def fetch_t86_day(date_str: str):
    """
    抓取單一天的三大法人買賣超資料。
    回傳 dict {code: {...}} 代表有資料；回傳 None 代表「確認當天無交易」(stat!=OK)；
    request本身失敗時丟出 FetchFailed，由呼叫端決定要不要重試。
    """
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    params = {"date": date_str, "selectType": "ALL", "response": "json"}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
        payload = resp.json()
    except Exception as e:
        raise FetchFailed(str(e))

    if payload.get("stat") != "OK":
        return None  # 確認是非交易日/查無資料，不是請求失敗，不用重試

    result = {}
    for row in payload.get("data", []):
        code = str(row[COL_CODE]).strip()
        if not code.isdigit():
            continue
        result[code] = {
            "foreign_net": _parse_int(row[COL_FOREIGN_NET]),
            "trust_net": _parse_int(row[COL_TRUST_NET]),
            "dealer_net": _parse_int(row[COL_DEALER_NET]),
            "total_net": _parse_int(row[COL_TOTAL_NET]),
        }
    return result


def _fetch_one_day_with_retry(day_str: str):
    """幫單一天套上重試邏輯，回傳 (day_str, data_dict_or_None, success_bool)。
    success_bool=False 代表重試用盡仍失敗，呼叫端不該把這天當成「確認無交易」快取起來，
    應該留到下次重跑再試，避免真實資料被永久誤判成空白。"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            data = fetch_t86_day(day_str)
            return day_str, data, True
        except FetchFailed:
            if attempt < MAX_RETRIES:
                time.sleep(1.5)
                continue
            return day_str, None, False
    return day_str, None, False


def _trading_days_between(start: str, end: str) -> list:
    """粗略列出start~end之間所有平日(週一到週五)，週末先排除，
    國定假日靠T86自己回傳stat!=OK來跳過，不需要額外查假日清單。"""
    start_d = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    days = []
    cur = start_d
    while cur <= end_d:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y%m%d"))
        cur += datetime.timedelta(days=1)
    return days


def load_chip_data(start: str, end: str, refresh: bool = False) -> dict:
    """
    回傳 {code: DataFrame(foreign_net, trust_net, dealer_net, total_net, index=日期)}。
    每天的原始資料會快取成一個小檔案，重複執行不會重新打API。
    改用多執行緒平行下載加速；確認無交易(國定假日等)才會快取空白標記，
    請求逾時/失敗的日子不會被誤存成空白，會保留到下次重跑時自動補抓。
    """
    os.makedirs(CHIP_CACHE_DIR, exist_ok=True)
    days = _trading_days_between(start, end)

    per_stock_records = {}  # code -> list of (date, dict)
    days_to_fetch = []      # 還沒有快取(或明確標記過)、需要真的發請求的日期

    NO_DATA_MARKER = "__NO_DATA__"

    for day_str in days:
        cache_path = os.path.join(CHIP_CACHE_DIR, f"t86_{day_str}.csv")
        if not refresh and os.path.exists(cache_path):
            continue  # 已經有快取(不管是有資料還是確認無交易的標記檔)，不用重抓
        days_to_fetch.append(day_str)

    print(f"三大法人資料：{len(days)} 個平日，{len(days_to_fetch)} 天需要下載"
          f"(其餘已有快取)，使用 {MAX_WORKERS} 個執行緒平行下載 ...", flush=True)

    fetched_count = 0
    no_data_count = 0
    failed_count = 0

    if days_to_fetch:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_one_day_with_retry, d): d for d in days_to_fetch}
            done_n = 0
            for future in as_completed(futures):
                day_str, data, success = future.result()
                done_n += 1
                cache_path = os.path.join(CHIP_CACHE_DIR, f"t86_{day_str}.csv")

                if not success:
                    failed_count += 1
                    # 重試用盡仍失敗：不寫快取，留給下次重跑補抓，避免誤判成無交易
                elif data is None:
                    no_data_count += 1
                    pd.DataFrame(columns=["code", "foreign_net", "trust_net", "dealer_net", "total_net"]).to_csv(cache_path, index=False)
                else:
                    fetched_count += 1
                    rows = [{"code": c, **v} for c, v in data.items()]
                    pd.DataFrame(rows).to_csv(cache_path, index=False)

                if done_n % 50 == 0 or done_n == len(days_to_fetch):
                    print(f"  進度 {done_n}/{len(days_to_fetch)} "
                          f"(成功{fetched_count} 無交易{no_data_count} 失敗待補{failed_count}) ...", flush=True)

    print(f"三大法人資料下載完成：本次新抓{fetched_count}天，"
          f"確認無交易{no_data_count}天，逾時待補{failed_count}天"
          f"{'（下次重跑會自動補抓這些天）' if failed_count else ''}", flush=True)

    # ---- 讀取全部快取(含這次新抓的+之前就有的)，組成最終結果 ----
    for day_str in days:
        cache_path = os.path.join(CHIP_CACHE_DIR, f"t86_{day_str}.csv")
        if not os.path.exists(cache_path):
            continue  # 逾時待補的日子，這次先跳過
        try:
            day_df = pd.read_csv(cache_path, dtype={"code": str})
        except Exception:
            continue
        if day_df.empty:
            continue
        date_ts = pd.Timestamp(datetime.datetime.strptime(day_str, "%Y%m%d"))
        for _, row in day_df.iterrows():
            code = str(row["code"])
            per_stock_records.setdefault(code, []).append({
                "date": date_ts,
                "foreign_net": row["foreign_net"],
                "trust_net": row["trust_net"],
                "dealer_net": row["dealer_net"],
                "total_net": row["total_net"],
            })

    chip_data = {}
    for code, records in per_stock_records.items():
        df = pd.DataFrame(records).set_index("date").sort_index()
        chip_data[code] = df
    return chip_data


def compute_consecutive_net_buy_days(chip_df: pd.DataFrame, as_of_date, net_col: str = "foreign_net") -> int:
    """
    計算「as_of_date之前」，某檔股票連續buy(正值)的天數。
    連續賣超則回傳負數(例如連續3天賣超回傳-3)，方便同一個函式同時用在多空兩種訊號判斷。
    保留這個逐次計算版本作為對照/單元測試基準，正式回測請改用 precompute_chip_streak()
    預先算好整個序列，避免day-by-day迴圈裡重複計算(跟RSI/布林通道一樣的效能考量)。
    """
    hist = chip_df[chip_df.index < as_of_date][net_col]
    if hist.empty:
        return 0
    streak = 0
    sign = None
    for v in hist.iloc[::-1]:  # 從最近的日期往前數
        cur_sign = 1 if v > 0 else (-1 if v < 0 else 0)
        if cur_sign == 0:
            break
        if sign is None:
            sign = cur_sign
        if cur_sign != sign:
            break
        streak += 1
    return streak * (sign or 0)


def precompute_chip_streak(chip_df: pd.DataFrame, net_col: str = "foreign_net") -> pd.Series:
    """
    向量化版本：一次算好整個序列「連續買超/賣超天數」，取代逐日呼叫 compute_consecutive_net_buy_days()。
    正值=連續買超天數，負值=連續賣超天數(絕對值)，0=當天不買不賣或資料缺失。

    做法：用「正負號跟前一天不同就開新一段」的方式分組(sign變化點累加當作分組編號)，
    組內用cumcount()算出「這是本段第幾天」，乘上正負號就是最終結果。
    這個寫法在數學上等同對每一天呼叫 compute_consecutive_net_buy_days()，但是O(n)而不是O(n²)。
    """
    net = chip_df[net_col]
    sign = net.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    group_id = (sign != sign.shift()).cumsum()
    day_in_streak = sign.groupby(group_id).cumcount() + 1
    streak = day_in_streak * sign
    streak = streak.where(sign != 0, 0)
    return streak
