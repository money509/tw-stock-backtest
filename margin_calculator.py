"""
margin_calculator.py
========================
把 run_overnight_backtest() 產生的交易紀錄，換算成「這筆交易實際上需要多少
保證金」，以及「整個回測期間同時最多需要準備多少本金」——這是之前一直沒做
的事：run_overnight_backtest() 只算絕對損益(NT$)，完全沒有考慮股票期貨的
保證金機制，沒辦法回答「用多少本金可以做到」這個問題。

保證金資料來源：taifex_universe.py（期交所股票期貨標的清單 + 原始保證金
適用比例，查證時間點的數值，正式下單前務必以期交所或期貨商當下公告為準）。
"""
import taifex_universe as tu


def estimate_trade_margin(code, entry_price, lots=1):
    """
    估算單筆交易需要的原始保證金(新台幣)。
    code 不在 taifex_universe.STOCK_FUTURES_UNIVERSE 裡的話會丟出 ValueError——
    不該讓不明代碼靜靜地算出一個不可靠的保證金數字，寧可讓呼叫端知道要處理。
    """
    return tu.estimate_margin(code, entry_price, lots)


def annotate_trades_with_margin(trades, lots_per_trade=1):
    """
    幫每筆交易紀錄多加 margin_required 欄位(新台幣)。
    code 不在 taifex_universe 清單裡的交易，margin_required 設為 None
    (不是0)，避免被後面的加總誤算成「這筆不用本金」。
    """
    annotated = []
    for t in trades:
        code = t["code"]
        try:
            margin = estimate_trade_margin(code, t["entry_price"], lots=lots_per_trade)
        except ValueError:
            margin = None
        new_t = dict(t)
        new_t["margin_required"] = margin
        annotated.append(new_t)
    return annotated


def summarize_capital_requirement(trades, lots_per_trade=1, max_concurrent_positions=None):
    """
    算出整個回測期間，實際上需要準備多少本金，以及對應的報酬率。

    做法：先幫每筆交易算出所需保證金，再依「進場日」分組，找出「同一天最多
    同時有幾筆新進場的交易」，加總當天所有保證金，取整段期間裡「單日保證金
    需求最高」的那一天當作估算本金的基準。

    這是保守估計，沒有考慮「昨天進場的部位今天還沒出場、今天又有新的要進場」
    這種「部位重疊」需要的加碼保證金——因為目前策略是T日收盤進場、T+1日收盤前
    一定出清，理論上不會有連續多天疊倉的情況，所以「單日」是合理的估算基準；
    但如果你之後改了策略邏輯(例如允許多天不出場)，這裡的估算就不適用了。

    max_concurrent_positions：可選，如果你想手動限制「最多同時持有幾檔」
    (例如本金只夠買2檔，不是top_n=5全買)，可以傳進來——算出來的保證金只會
    取「這一天保證金最高的前N筆」加總，模擬「資金不夠、優先買分數最高的」
    這種情境；同時也應該只計入這幾筆的損益，不然報酬率會失真。
    """
    annotated = annotate_trades_with_margin(trades, lots_per_trade=lots_per_trade)
    valid = [t for t in annotated if t["margin_required"] is not None]
    skipped_codes = sorted({t["code"] for t in annotated if t["margin_required"] is None})

    by_day = {}
    for t in valid:
        by_day.setdefault(t["entry_date"], []).append(t)

    daily_margin_totals = {}
    counted_trades = []
    for day, day_trades in by_day.items():
        day_trades_sorted = sorted(day_trades, key=lambda t: t["margin_required"], reverse=True)
        if max_concurrent_positions is not None:
            day_trades_sorted = day_trades_sorted[:max_concurrent_positions]
        daily_margin_totals[day] = sum(t["margin_required"] for t in day_trades_sorted)
        counted_trades.extend(day_trades_sorted)

    if not daily_margin_totals:
        return {
            "peak_daily_margin": 0.0,
            "peak_day": None,
            "avg_daily_margin": 0.0,
            "total_trades_with_margin": 0,
            "total_pnl_of_counted_trades": 0.0,
            "return_on_peak_capital_pct": None,
            "skipped_codes_not_in_universe": skipped_codes,
        }

    peak_day = max(daily_margin_totals, key=daily_margin_totals.get)
    peak_margin = daily_margin_totals[peak_day]
    avg_margin = sum(daily_margin_totals.values()) / len(daily_margin_totals)

    total_pnl = sum(t["pnl"] for t in counted_trades)
    return_on_peak_capital = (total_pnl / peak_margin * 100.0) if peak_margin else None

    return {
        "peak_daily_margin": peak_margin,
        "peak_day": peak_day,
        "avg_daily_margin": avg_margin,
        "total_trades_with_margin": len(counted_trades),
        "total_pnl_of_counted_trades": total_pnl,
        "return_on_peak_capital_pct": return_on_peak_capital,
        "skipped_codes_not_in_universe": skipped_codes,
    }
