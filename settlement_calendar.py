"""
settlement_calendar.py
==========================
股票期貨結算日計算：跟台指期貨一樣，是每個月的「第三個星期三」(查證時間點
的規則，交易所日後若調整規則，這裡要跟著更新；理論值沒有核對台股國定假日
行事曆，遇到假日實際上會怎麼順延，務必自行核對期交所公告的正式結算日期)。

為什麼要管這件事：現在的隔日衝策略完全沒有處理結算日——如果你在結算日
「當天」進場一檔股票期貨，持有的合約在結算日當天會被交易所強制以結算價
結算(不是你自己選擇要不要出場、用什麼價出場)，這跟策略原本「T+1收盤前
自己決定出場」的假設不一樣，回測完全沒模擬到這件事，可能讓實際結果
跟回測不同。

這支模組提供：
  1. 算出理論上的結算日期
  2. 判斷某一天是不是「結算日附近」(風險窗口)
  3. 幫既有的交易紀錄(list[dict])標記「這筆交易的進場日是不是在結算日附近」，
     方便你直接用手上已經跑出來的回測結果，診斷「結算日附近的交易，表現
     是不是特別不一樣」，不需要真的先把結算邏輯寫進引擎才能開始分析。
"""
import calendar
import datetime


def third_wednesday(year: int, month: int) -> datetime.date:
    """回傳某年某月「第三個星期三」的理論日期(未核對台股假日行事曆)。"""
    cal = calendar.Calendar()
    wednesdays = [
        d for d in cal.itermonthdates(year, month)
        if d.month == month and d.weekday() == 2  # Monday=0 ... Wednesday=2
    ]
    return wednesdays[2]


def _to_date(date) -> datetime.date:
    if isinstance(date, datetime.date):
        return date
    return datetime.datetime.strptime(str(date), "%Y%m%d").date()


def is_settlement_day(date) -> bool:
    """date可以是datetime.date或YYYYMMDD字串。回傳這天是不是理論上的結算日。"""
    d = _to_date(date)
    return d == third_wednesday(d.year, d.month)


def is_near_settlement(date, days_before=1, days_after=0) -> bool:
    """
    判斷date是不是在「結算日前days_before個日曆天 ~ 結算日後days_after個日曆天」
    這個風險窗口內。用日曆天(不是交易日)概算，會比實際的交易日窗口略寬鬆，
    屬於保守估計(寧可多排除幾天，也不要漏掉真正該注意的日子)。
    """
    d = _to_date(date)
    theoretical = third_wednesday(d.year, d.month)
    delta = (d - theoretical).days
    return -days_before <= delta <= days_after


def settlement_dates_in_range(start_year, start_month, end_year, end_month):
    """回傳[start_year-start_month ~ end_year-end_month]這段期間內，每個月
    理論上的結算日期list(datetime.date)。"""
    dates = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        dates.append(third_wednesday(y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def annotate_trades_with_settlement_proximity(trades, days_before=1, days_after=0):
    """
    幫交易紀錄(list[dict]，需要有entry_date欄位，YYYYMMDD字串)多加
    near_settlement欄位(bool)，標記這筆交易的進場日是不是在結算日附近。
    不修改原本的trades，回傳新的list。
    """
    annotated = []
    for t in trades:
        new_t = dict(t)
        new_t["near_settlement"] = is_near_settlement(
            t["entry_date"], days_before=days_before, days_after=days_after)
        annotated.append(new_t)
    return annotated


def summarize_settlement_proximity_impact(trades, days_before=1, days_after=0):
    """
    比較「結算日附近進場」vs「不在結算日附近進場」這兩組交易的平均損益/勝率，
    格式比照diagnose_open_gap.py的風格——只是分析，不改變策略邏輯本身。
    """
    annotated = annotate_trades_with_settlement_proximity(trades, days_before, days_after)
    groups = {True: [], False: []}
    for t in annotated:
        groups[t["near_settlement"]].append(t["pnl"])

    def _stats(pnls):
        if not pnls:
            return {"count": 0, "avg_pnl": 0.0, "total_pnl": 0.0, "win_rate": 0.0}
        wins = sum(1 for p in pnls if p > 0)
        return {
            "count": len(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "total_pnl": sum(pnls),
            "win_rate": wins / len(pnls) * 100.0,
        }

    return {
        "near_settlement": _stats(groups[True]),
        "not_near_settlement": _stats(groups[False]),
    }
