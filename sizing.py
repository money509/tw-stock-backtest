"""
部位大小規則模組。

engine.py 產生的每筆交易，本身已經決定了「進場價/停損價/出場價」這些跟部位大小
完全無關的東西（因為停損停利是用ATR算的價位，不會因為你買幾口而改變）。
這裡要做的，是拿這些「原始交易紀錄」，套用不同的口數計算規則，重新算出：
  - 這筆交易實際上買了幾口
  - 換算成新台幣的實際損益
  - 以「部位名義本金」為基準的報酬率 (跟口數無關，方便橫向比較)
"""


def contract_multiplier(price: float) -> int:
    """複製 main.py 的契約規格判斷：進場價>=500用小型期貨(100股/口)，否則標準期貨(2000股/口)"""
    return 100 if price >= 500 else 2000


def compute_lots(trade: dict, sizing_mode: str, risk_per_trade_ntd: float = 20000) -> int:
    """
    sizing_mode == "fixed2":     固定2口 (跟production目前的做法一致)
    sizing_mode == "risk_based": 用固定風險金額反推口數，並且至少維持2口
    """
    if sizing_mode == "fixed2":
        return 2

    if sizing_mode == "risk_based":
        e_price = trade["e_price"]
        sl_price = trade["sl_price_initial"]
        mult = contract_multiplier(e_price)
        risk_per_lot = (e_price - sl_price) * mult
        if risk_per_lot <= 0:
            return 2  # 理論上不該發生（進場價應該要高於停損價），保險起見給預設值
        lots = round(risk_per_trade_ntd / risk_per_lot)
        return max(2, lots)

    raise ValueError(f"未知的 sizing mode: {sizing_mode}")


def apply_sizing(trades_raw: list, sizing_mode: str, risk_per_trade_ntd: float = 20000) -> list:
    """
    把一批 raw trades 套用指定的部位大小規則，回傳新的交易清單，
    每筆多了 lots / pnl_ntd / return_pct（依實際口數重新加權計算）欄位。
    """
    result = []
    for trade in trades_raw:
        lots = compute_lots(trade, sizing_mode, risk_per_trade_ntd)
        mult = contract_multiplier(trade["e_price"])
        e_price = trade["e_price"]

        if trade["leg1_hit"]:
            leg1_lots = lots // 2
            leg2_lots = lots - leg1_lots
            leg1_pnl = leg1_lots * (trade["leg1_exit_price"] - e_price) * mult
            leg2_pnl = leg2_lots * (trade["leg2_exit_price"] - e_price) * mult
        else:
            leg1_lots = 0
            leg2_lots = lots
            leg1_pnl = 0.0
            leg2_pnl = lots * (trade["leg2_exit_price"] - e_price) * mult

        pnl_ntd = leg1_pnl + leg2_pnl
        notional = lots * e_price * mult
        return_pct = pnl_ntd / notional if notional else 0.0

        new_trade = dict(trade)
        new_trade.update({
            "lots": lots,
            "leg1_lots": leg1_lots,
            "leg2_lots": leg2_lots,
            "pnl_ntd": pnl_ntd,
            "return_pct": return_pct,  # 覆蓋掉engine.py給的預設值，改用實際口數加權後的版本
        })
        result.append(new_trade)
    return result
