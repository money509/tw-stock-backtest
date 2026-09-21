"""
futures_liquidity_checker.py
================================
股票期貨流動性檢查——目前整個回測完全沒有處理過這一塊，這支模組只能先把它
變成「可以檢查的資料」，還不能真的補上完整答案，以下說明白：

回測至今的「冷門股濾網」(overnight_momentum_engine.py 的 MIN_TURNOVER_VALUE，
當天成交金額 < 2000萬就排除)，檢查的是「現股」的成交金額——但隔日衝實際上
交易的是「股票期貨」這個獨立的衍生性商品，兩者的流動性不是同一件事：
一檔現股成交熱絡(法人、隔日衝當沖客都在買賣現股)，不代表它的股票期貨合約
也有足夠的未平倉量(open interest)、日成交量——台灣期交所249檔股票期貨裡，
有不少檔位是名符其實的「冷門合約」，現股流動性再好，期貨合約本身可能一天
只成交幾口甚至掛不到對手單，回測完全假設「算出來要買/要賣多少口，一定能
用理論價位成交」，跟現股的MIN_TURNOVER_VALUE濾網完全沒關係。

這支模組能做的：
  1. 提供資料結構跟flag邏輯，一旦有了「股票期貨本身」的日成交量/未平倉量
     資料(需要另外從期交所公開資訊觀測站抓，例如
     https://www.taifex.com.tw/cht/3/dlStockFOMarketDayReport 或期交所
     OpenAPI，這支模組目前沒有內建下載器，需要另外建一支
     futures_liquidity_loader.py 才能真正抓到這份資料並驗證格式)，
     就能直接把回測交易紀錄跟這份資料串起來檢查。
  2. 在還沒有那份資料的情況下，誠實回報「無法判斷」，而不是假裝現股濾網
     已經足夠涵蓋期貨流動性風險——這是刻意的設計，寧可讓使用者清楚知道
     這塊還沒補上，也不要用現股資料冒充期貨流動性資料，製造虛假的安全感。

⚠️ 還沒完成的部分：真正抓期交所股票期貨日成交量/未平倉量的下載器
(futures_liquidity_loader.py)還沒有建立，因為那份資料的實際API/報表格式
需要在有網路的環境(例如GitHub Actions)實際打過一次才能確認欄位、驗證
不會像chip_data_loader.py當初那樣因為格式假設錯誤而出錯——在還沒實際驗證過
真實資料格式之前，貿然寫死一套解析邏輯風險很高，這支模組先只處理「資料
串接+診斷」這一層，資料來源本身留給下一步。
"""

DEFAULT_MIN_DAILY_VOLUME = 50  # 口數/日，保守佔位值，未經真實資料校準，見下方annotate_trades_with_liquidity說明


def annotate_trades_with_liquidity(trades, futures_volume_by_code=None, min_daily_volume=None):
    """
    幫交易紀錄(list[dict]，需要有code欄位)加上流動性標記。

    futures_volume_by_code：dict，{code: 該檔股票期貨的日均成交量(口數)}。
    這份資料目前沒有內建下載器可以自動取得(見模組docstring)，需要由呼叫端
    自行提供——沒有提供的話，每筆交易的liquidity_known都會是False，
    liquidity_flag也會是None(代表「無法判斷」，不是「流動性沒問題」)。

    min_daily_volume：低於這個日均成交量(口)視為流動性偏低，預設None時
    使用DEFAULT_MIN_DAILY_VOLUME(見常數，目前是保守的佔位值，實際門檻
    應該等真正拿到期交所資料後，依統計分布重新校準，不是憑空定案的標準)。

    不修改原本的trades，回傳新的list，每筆多兩個欄位：
      liquidity_known: bool，這筆交易的code是否在futures_volume_by_code裡
      liquidity_flag: True(流動性偏低，回測用理論價成交的假設可能失真)/
                       False(流動性正常)/None(不知道，因為liquidity_known=False)
    """
    threshold = min_daily_volume if min_daily_volume is not None else DEFAULT_MIN_DAILY_VOLUME
    futures_volume_by_code = futures_volume_by_code or {}

    annotated = []
    for t in trades:
        new_t = dict(t)
        code = t["code"]
        if code in futures_volume_by_code:
            avg_vol = futures_volume_by_code[code]
            new_t["liquidity_known"] = True
            new_t["liquidity_flag"] = avg_vol < threshold
            new_t["futures_avg_daily_volume"] = avg_vol
        else:
            new_t["liquidity_known"] = False
            new_t["liquidity_flag"] = None
            new_t["futures_avg_daily_volume"] = None
        annotated.append(new_t)
    return annotated


def summarize_liquidity_coverage(trades, futures_volume_by_code=None, min_daily_volume=None):
    """
    彙總「這批交易裡，流動性資料涵蓋了多少比例」以及「已知資料裡，有多少比例
    偏低流動性」——如果coverage_pct很低，代表目前完全無法對這批交易做出
    任何流動性風險的結論，不能誤讀成「流動性沒問題」。
    """
    annotated = annotate_trades_with_liquidity(
        trades, futures_volume_by_code=futures_volume_by_code, min_daily_volume=min_daily_volume)
    total = len(annotated)
    if total == 0:
        return {
            "total_trades": 0, "known_count": 0, "coverage_pct": None,
            "low_liquidity_count": 0, "low_liquidity_pct_of_known": None,
            "low_liquidity_codes": [],
        }

    known = [t for t in annotated if t["liquidity_known"]]
    low_liq = [t for t in known if t["liquidity_flag"]]
    coverage_pct = len(known) / total * 100.0
    low_liq_pct = (len(low_liq) / len(known) * 100.0) if known else None

    return {
        "total_trades": total,
        "known_count": len(known),
        "coverage_pct": coverage_pct,
        "low_liquidity_count": len(low_liq),
        "low_liquidity_pct_of_known": low_liq_pct,
        "low_liquidity_codes": sorted({t["code"] for t in low_liq}),
    }
