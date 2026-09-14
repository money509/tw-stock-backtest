"""
負責從 yfinance 下載歷史股價，並快取成本機 CSV 檔，避免每次跑回測都重新下載。
"""
import os
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

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
    回傳 {code: DataFrame(Open, High, Low, Close, index=日期)}。
    有本機快取，重複執行不會一直打 yfinance。refresh=True 會強制重新下載。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    price_data = {}

    for code in whitelist:
        cache_path = os.path.join(CACHE_DIR, f"{code}_{start}_{end}.csv")

        df = None
        if not refresh and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if "Volume" not in df.columns:
                # 舊版快取沒有存 Volume 欄位 (量能加權分數需要用到)，強制重新下載
                print(f"  {code} 的快取是舊格式(缺Volume)，自動重新下載", flush=True)
                df = None

        if df is None:
            entry = whitelist[code]
            sym = _symbol_for(code, entry)
            display_name = entry if isinstance(entry, str) else code
            print(f"下載 {sym} ({display_name}) ...", flush=True)
            try:
                raw = yf.Ticker(sym).history(start=start, end=end, interval="1d")
            except Exception as e:
                print(f"  下載失敗: {e}", flush=True)
                continue
            if raw is None or raw.empty:
                print(f"  {sym} 沒有抓到任何資料，略過", flush=True)
                continue
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            # 統一時區資訊，避免跟後面計算日期時因為 tz-aware/naive 不一致而出錯
            df.index = df.index.tz_localize(None)
            df.to_csv(cache_path)

        if df.empty:
            continue
        price_data[code] = df

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
