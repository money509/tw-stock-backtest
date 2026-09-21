"""
us_market_loader.py
======================
抓美股大盤指數(預設 S&P500，代號 ^GSPC)的每日資料，算出「隔夜美股報酬率」，
用來當隔日衝策略的跳空風險濾網——你自己一開始就提過「台股跟美股連動，
半夜美股大跌，部位會直接跳空虧損」，這支模組把這句話變成可以檢查的資料。

時區對應說明（這是這個濾網能不能對齊日期的關鍵）：
    美股交易時段 9:30am~4:00pm 美東時間，換算成台北時間大約是
    21:30~04:00（夏令時間）或 22:30~05:00（冬令時間）。
    也就是說，「美股日期 X」的那個交易時段，剛好落在「台股日期 X 收盤後」到
    「台股日期 X+1 開盤前」這段空窗期——不管中間有沒有週末。
    所以要查「台股 T 日收盤到 T+1 開盤這段夜盤風險」，直接用「美股日期同樣是 T」
    的那筆報酬率去對齊即可，不需要額外處理日期位移，包括跨週末的情況
    （週五台股收盤到週一開盤之間，美股只有週五那個交易時段有開盤，
      週一的美股交易時段是在台股週一開盤"之後"才開始，不影響這段跳空風險）。

已知限制：
    - 用 yfinance 抓 ^GSPC，跟 data_loader.py 抓台股一樣的方式，本身不保證
      100% 對齊每個美股假日（美股假日行事曆跟台股不同），但兩邊都用日期字串
      merge，對不上的日期就是沒有濾網可用，程式會自然略過，不會硬湊。
    - 只抓收盤價，用「今天收盤 vs 昨天收盤」的報酬率當「隔夜美股表現」的代理值，
      不是精確到「台股收盤那個時間點」的即時報酬率，是近似值。
"""

import os
import time
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "us_market_cache")
DEFAULT_SYMBOL = "^GSPC"  # S&P 500


def load_us_market_returns(start: str, end: str, symbol: str = DEFAULT_SYMBOL,
                            refresh: bool = False) -> pd.DataFrame:
    """
    回傳 DataFrame[date(str YYYYMMDD), close, return_pct]。
    return_pct = (今天收盤 / 昨天收盤 - 1) * 100，第一筆一定是 NaN（沒有前一天可比較）。

    有本機快取（跟 data_loader.py 一樣的模式），重複執行不會一直打 yfinance。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_symbol = symbol.replace("^", "").replace("=", "")
    cache_path = os.path.join(CACHE_DIR, f"{safe_symbol}_{start}_{end}.csv")

    if not refresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype={"date": str})
            if not df.empty and "return_pct" in df.columns:
                return df
        except Exception:
            pass

    print(f"下載美股大盤資料 {symbol} ({start} ~ {end}) ...", flush=True)
    raw = None
    for attempt in range(2):
        try:
            raw = yf.download(
                tickers=symbol, start=start, end=end, interval="1d",
                progress=False, auto_adjust=False,
            )
            break
        except Exception as e:
            print(f"  第{attempt+1}次下載發生例外: {e}", flush=True)
            raw = None
            time.sleep(3)

    if raw is None or raw.empty:
        print(f"  {symbol} 下載失敗或無資料，回傳空的濾網資料(不會擋到任何交易)", flush=True)
        return pd.DataFrame(columns=["date", "close", "return_pct"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].copy()
    df = df.reset_index()
    df.columns = ["date"] + list(df.columns[1:])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    df = df.rename(columns={"Close": "close"})
    df["return_pct"] = (df["close"] / df["close"].shift(1) - 1.0) * 100.0
    df = df[["date", "close", "return_pct"]]

    df.to_csv(cache_path, index=False)
    return df
