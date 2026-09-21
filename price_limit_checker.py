"""
price_limit_checker.py
==========================
台股漲跌停(目前規則：前一交易日收盤價的正負10%，查證時間點的規則，交易所
日後若調整漲跌幅限制，這裡要跟著更新)：ATR算出來的停損/停利價，理論上有
可能超出當天真正能成交的漲跌停範圍——尤其這個策略本身就偏好挑近期強勢股，
比一般股票更容易碰到漲停鎖死，回測完全假設「不管算出來多少，一定能用那個
價位成交」，這支模組把這件事變成可以檢查/修正的資料。

這支模組是單純的數學工具，不碰任何資料源，方便被回測分析或未來的即時
監控腳本共用。
"""

DEFAULT_LIMIT_PCT = 10.0


def price_limit_band(prev_close, limit_pct=DEFAULT_LIMIT_PCT):
    """回傳(跌停價, 漲停價)。prev_close必須是正數。"""
    if prev_close is None or prev_close <= 0:
        raise ValueError(f"prev_close必須是正數，收到: {prev_close}")
    lower = prev_close * (1 - limit_pct / 100.0)
    upper = prev_close * (1 + limit_pct / 100.0)
    return lower, upper


def is_within_limit_band(price, prev_close, limit_pct=DEFAULT_LIMIT_PCT):
    """判斷price是不是在prev_close算出來的漲跌停範圍內(含邊界)。"""
    lower, upper = price_limit_band(prev_close, limit_pct)
    return lower <= price <= upper


def clamp_to_limit_band(price, prev_close, limit_pct=DEFAULT_LIMIT_PCT):
    """
    把price夾到漲跌停範圍內——用來修正「ATR算出來的停利價超過漲停」這種
    不可能真的成交在那個價位的情況。回傳夾過的價格；如果price原本就在範圍
    內，原封不動回傳。
    """
    lower, upper = price_limit_band(prev_close, limit_pct)
    return max(lower, min(upper, price))


def check_trade_exposure(entry_price, prev_close, stop_price=None, target_price=None,
                          limit_pct=DEFAULT_LIMIT_PCT):
    """
    一次檢查一筆交易的進場價/停損價/停利價，各自有沒有超出當天(以prev_close
    為基準算出來的)漲跌停範圍。回傳dict，每個有給值的價位對應一個bool
    (True=在範圍內，正常；False=超出範圍，回測假設的那個出場價實際上不可能
    成交，績效可能因此被高估)；沒給值的價位(例如stop_price=None)回傳None
    (無法判斷，不是"在範圍內")。
    """
    lower, upper = price_limit_band(prev_close, limit_pct)

    def _check(price):
        if price is None:
            return None
        return lower <= price <= upper

    return {
        "limit_down": lower,
        "limit_up": upper,
        "entry_within_band": _check(entry_price),
        "stop_within_band": _check(stop_price),
        "target_within_band": _check(target_price),
    }
