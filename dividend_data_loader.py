"""
dividend_data_loader.py
==========================
抓每檔股票的除權息（含股票股利/現金股利/減資/分割）事件日期，用來讓
overnight_momentum_engine.py 在當天是某檔股票的除權息日時，直接把它從候選名單
剔除——因為 data_loader.py 抓的是「未還原」(auto_adjust=False)的股價，除權息當天
收盤價會因為公司配股配息機械性地往下跳一段，這段跳動跟真正的動能轉弱無關，
但會被 score_gain_pct（漲幅%評分）、score_rel_strength（相對大盤強弱）誤判成
「這檔轉弱了」，也會汙染 atr14（真實區間）的計算，甚至可能誤觸 gap_stop。

背景：未還原股價本身不是bug——期貨standholder也不會收到股息，所以期貨價格
同樣會在除權息時機械性下跳，未還原股價其實更貼近期貨的真實走勢。真正的問題
是「這個機械性下跳被評分邏輯誤讀成基本面轉弱」，解法不是換成還原股價
（那樣反而會讓進場價/出場價跟期貨實際報價脫節），而是「除權息當天乾脆不要
選這檔股票進場」——反正只是少一天的候選機會，不影響策略其他部分。

做法：用 yfinance 的 Ticker(sym).actions（涵蓋 Dividends 跟 Stock Splits 兩欄）
抓出每檔股票歷史上所有除權息/減資/分割的日期，整理成
{code: set(日期字串 YYYYMMDD)}，可以直接拿去跟 scan_candidates_for_date() 裡的
date_str 比對。跟其他 loader 一樣做本機 CSV 快取，重複執行不會一直打 yfinance；
單一標的失敗(找不到資料、下載出錯)只記錄並跳過，不會讓整個流程中斷
（沿用 us_market_loader.py 的「查不到就回傳空的，不會硬擋任何交易」原則）。
"""

import os
import time
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "dividend_cache")


def _symbol_for(code: str, entry=None) -> str:
    """跟 data_loader._symbol_for() 完全一樣的判斷邏輯，這裡獨立複製一份，
    避免這支模組去 import data_loader 造成不必要的耦合（這支只管除權息日期，
    不需要知道 data_loader 內部 OTC_STOCKS 以外的任何東西）。"""
    import data_loader
    if isinstance(entry, dict) and "otc" in entry:
        suffix = ".TWO" if entry["otc"] else ".TW"
    else:
        suffix = ".TWO" if code in data_loader.OTC_STOCKS else ".TW"
    return f"{code}{suffix}"


def load_dividend_events(whitelist: dict, start: str, end: str, refresh: bool = False) -> dict:
    """
    回傳 {code: set(日期字串 YYYYMMDD)}，代表這檔股票在這個區間內所有
    除權息/減資/分割（ex-dividend/ex-rights）的日期。

    whitelist: 跟 data_loader.load_price_data() 用同一份 {code: entry}。
    有本機快取（跟其他 loader 一樣），refresh=True 會強制重新下載。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    result = {}

    for code, entry in whitelist.items():
        cache_path = os.path.join(CACHE_DIR, f"{code}_{start}_{end}.csv")

        if not refresh and os.path.exists(cache_path):
            try:
                cached = pd.read_csv(cache_path, dtype={"date": str})
                result[code] = set(cached["date"].tolist())
                continue
            except Exception:
                pass

        sym = _symbol_for(code, entry)
        dates = set()
        for attempt in range(2):
            try:
                ticker = yf.Ticker(sym)
                actions = ticker.actions  # DataFrame, index=日期, columns 可能有 Dividends/Stock Splits
                if actions is not None and not actions.empty:
                    idx = pd.DatetimeIndex(actions.index)
                    if idx.tz is not None:
                        idx = idx.tz_localize(None)
                    # 只保留區間內、且 Dividends 或 Stock Splits 欄位非零的那幾天
                    mask = pd.Series(False, index=actions.index)
                    if "Dividends" in actions.columns:
                        mask = mask | (actions["Dividends"] != 0)
                    if "Stock Splits" in actions.columns:
                        mask = mask | (actions["Stock Splits"] != 0)
                    event_dates = idx[mask.to_numpy()]
                    event_dates = event_dates[(event_dates >= pd.Timestamp(start)) &
                                               (event_dates <= pd.Timestamp(end))]
                    dates = set(event_dates.strftime("%Y%m%d").tolist())
                break
            except Exception as e:
                print(f"  抓 {sym} 除權息資料第{attempt+1}次發生例外: {e}", flush=True)
                dates = set()
                time.sleep(2)

        result[code] = dates
        pd.DataFrame({"date": sorted(dates)}).to_csv(cache_path, index=False)
        time.sleep(0.3)  # 逐檔查詢，稍微間隔一下避免觸發限速

    return result
