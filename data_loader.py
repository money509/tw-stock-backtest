"""
負責從 yfinance 下載歷史股價，並快取成本機 CSV 檔，避免每次跑回測都重新下載。
"""
import os
import time
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
BATCH_SIZE = 30                # 每批一次打包幾檔股票一起下載
BATCH_DELAY_SECONDS = 3        # 每批之間的間隔，避免仍然觸發限速

# 上櫃標的清單 (Yahoo Finance 需要用 .TWO 後綴)
# 已查證：目前 whitelist 中只有雙鴻(3324)、台燿(6274) 為上櫃股，其餘皆為上市股 (.TW)
OTC_STOCKS = {"3324", "6274"}

STOCK_FUTURES_WHITELIST = {
    "2330": "台積電", "2454": "聯發科", "2303": "聯電", "3711": "日月光投控",
    "2449": "京元電子", "3035": "智原", "3443": "創意", "3661": "世芯-KY",
    "2317": "鴻海", "2382": "廣達", "3231": "緯創", "6669": "緯穎",
    "2356": "英業達", "2376": "技嘉", "2357": "華碩", "2301": "光寶科",
    "3017": "奇鋐", "3324": "雙鴻", "2308": "台達電", "2421": "建準",
    "3013": "晟銘電", "8210": "勤誠", "8996": "高力",
    "2345": "智邦", "3450": "聯鈞", "4977": "眾達-KY", "6451": "訊芯-KY",
    "3008": "大立光", "3406": "玉晶光", "2327": "國巨", "2492": "華新科",
    "1519": "華城", "1503": "士電", "1513": "中興電", "1514": "亞力", "1609": "大亞",
    "2368": "金像電", "3037": "欣興", "8046": "南電", "3189": "景碩", "6274": "台燿", "2383": "台光電",
    "2359": "所羅門", "4566": "時碩工業", "1504": "東元",
    "2409": "友達", "3481": "群創", "2408": "南亞科", "2344": "華邦電", "2337": "旺宏",
    "2603": "長榮", "2609": "陽明", "2615": "萬海", "2618": "長榮航", "2610": "華航",
    "2881": "富邦金", "2882": "國泰金", "2377": "微星", "2379": "瑞昱",
}


def _symbol_for(code: str, entry=None) -> str:
    """
    判斷這個代碼該用哪個yfinance後綴。
    entry 可以是舊格式的字串(股票名稱，此時退回用OTC_STOCKS這份59檔清單判斷)，
    或新格式的dict(含"otc"欄位，來自taifex_universe.py的320檔全市場清單，直接讀取判斷)。
    """
    if isinstance(entry, dict) and "otc" in entry:
        suffix = ".TWO" if entry["otc"] else ".TW"
    else:
        suffix = ".TWO" if code in OTC_STOCKS else ".TW"
    return f"{code}{suffix}"


def load_price_data(whitelist: dict, start: str, end: str, refresh: bool = False) -> dict:
    """
    回傳 {code: DataFrame(Open, High, Low, Close, Volume, index=日期)}。
    有本機快取，重複執行不會一直打 yfinance。refresh=True 會強制重新下載。

    改進：不再一檔一檔分開請求 (那樣249檔就是249次個別請求，容易在請求200多次後
    被Yahoo判定為異常流量而卡住/被限速，之前兩次修正逾時的嘗試都沒解決這個根本問題)。
    改用 yf.download() 一次打包多檔一起下載，大幅減少總請求次數。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    price_data = {}
    to_download = []  # [(code, sym, entry), ...]

    for code, entry in whitelist.items():
        cache_path = os.path.join(CACHE_DIR, f"{code}_{start}_{end}.csv")
        if not refresh and os.path.exists(cache_path):
            try:
                cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            except Exception:
                cached_df = None
            if cached_df is not None and "Volume" in cached_df.columns and not cached_df.empty:
                price_data[code] = cached_df
                continue
        to_download.append((code, _symbol_for(code, entry), entry))

    if not to_download:
        print("全部標的都已有快取，不需要下載", flush=True)
        return price_data

    print(f"需要下載 {len(to_download)} 檔股票，每批 {BATCH_SIZE} 檔批次下載中 ...", flush=True)

    for batch_start in range(0, len(to_download), BATCH_SIZE):
        batch = to_download[batch_start: batch_start + BATCH_SIZE]
        batch_syms = [b[1] for b in batch]
        batch_no = batch_start // BATCH_SIZE + 1
        print(f"下載第 {batch_no} 批 ({len(batch_syms)}檔)：{batch_syms[0]} ~ {batch_syms[-1]} ...", flush=True)

        raw = None
        for attempt in range(2):  # 整批最多重試1次
            try:
                raw = yf.download(
                    tickers=batch_syms, start=start, end=end, interval="1d",
                    group_by="ticker", threads=True, progress=False, auto_adjust=False,
                )
                break
            except Exception as e:
                print(f"  第{attempt+1}次批次下載發生例外: {e}", flush=True)
                raw = None
                time.sleep(3)

        if raw is None or raw.empty:
            print(f"  第 {batch_no} 批整批失敗，略過這批 {len(batch_syms)} 檔", flush=True)
            time.sleep(BATCH_DELAY_SECONDS)
            continue

        for code, sym, entry in batch:
            try:
                if len(batch_syms) == 1:
                    sub = raw
                elif sym in set(raw.columns.get_level_values(0)):
                    sub = raw[sym]
                else:
                    print(f"  {sym} 這批結果裡找不到資料，略過", flush=True)
                    continue

                sub = sub.dropna(how="all")
                if sub.empty or "Close" not in sub.columns:
                    print(f"  {sym} 沒有抓到任何資料，略過", flush=True)
                    continue

                df = sub[["Open", "High", "Low", "Close", "Volume"]].copy()
                idx = pd.DatetimeIndex(df.index)
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                df.index = idx

                cache_path = os.path.join(CACHE_DIR, f"{code}_{start}_{end}.csv")
                df.to_csv(cache_path)
                price_data[code] = df
            except Exception as e:
                print(f"  處理 {sym} 資料時發生錯誤: {e}", flush=True)
                continue

        time.sleep(BATCH_DELAY_SECONDS)

    return price_data


def build_master_calendar(price_data: dict, reference_code: str = "2330") -> pd.DatetimeIndex:
    """
    用一檔流動性最好、幾乎天天有交易的股票 (預設台積電) 的日期序列，
    當作整個回測的「交易日曆」基準。
    """
    if reference_code in price_data:
        return price_data[reference_code].index
    # 保險起見，如果參考股票剛好沒抓到資料，改用所有股票日期的聯集
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index)
    return pd.DatetimeIndex(sorted(all_dates))
