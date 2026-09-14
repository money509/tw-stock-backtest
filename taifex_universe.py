"""
期交所股票期貨標的資料表。

資料來源：臺灣期貨交易所官方標的清單 (https://www.taifex.com.tw/cht/2/stockLists)，
擷取日期對應官方網頁標示之「最新更新(生效)日期：2026年9月3日」版本。

重要提醒：
1. 這份清單會隨期交所公告新增/調整，建議每隔一段時間重新核對官方頁面。
2. 保證金比例 (MARGIN_RATIO_TABLE) 是「原始保證金適用比例」，這個數字期交所每季會依風險評估調整，
   這裡收錄的是查證當下的數值，正式下單前務必以期交所或您的期貨商當下公告為準。
3. 若某檔股票代碼不在 STOCK_FUTURES_UNIVERSE 裡，代表該股票沒有對應的股票期貨商品，
   不該出現在任何選股候選名單中。
"""

# 有股票期貨商品的標的代碼 -> {"otc": 是否為上櫃股, "has_mini": 是否有小型契約(100股)}
# 上市股用 .TW 後綴查yfinance歷史資料，上櫃股用 .TWO
STOCK_FUTURES_UNIVERSE = {
    "1101": {"otc": False, "has_mini": False}, "1102": {"otc": False, "has_mini": False},
    "1210": {"otc": False, "has_mini": False}, "1216": {"otc": False, "has_mini": False},
    "1301": {"otc": False, "has_mini": False}, "1303": {"otc": False, "has_mini": False},
    "1312": {"otc": False, "has_mini": False}, "1314": {"otc": False, "has_mini": False},
    "1319": {"otc": False, "has_mini": False}, "1326": {"otc": False, "has_mini": False},
    "1402": {"otc": False, "has_mini": False}, "1440": {"otc": False, "has_mini": False},
    "1476": {"otc": False, "has_mini": False}, "1477": {"otc": False, "has_mini": True},
    "1503": {"otc": False, "has_mini": False}, "1504": {"otc": False, "has_mini": False},
    "1513": {"otc": False, "has_mini": False}, "1519": {"otc": False, "has_mini": True},
    "1536": {"otc": False, "has_mini": False}, "1560": {"otc": False, "has_mini": False},
    "1565": {"otc": True, "has_mini": True}, "1590": {"otc": False, "has_mini": False},
    "1605": {"otc": False, "has_mini": False}, "1608": {"otc": False, "has_mini": False},
    "1609": {"otc": False, "has_mini": False}, "1717": {"otc": False, "has_mini": False},
    "1718": {"otc": False, "has_mini": False}, "1722": {"otc": False, "has_mini": False},
    "1795": {"otc": False, "has_mini": False}, "1802": {"otc": False, "has_mini": False},
    "1904": {"otc": False, "has_mini": False}, "1905": {"otc": False, "has_mini": False},
    "1907": {"otc": False, "has_mini": False}, "1909": {"otc": False, "has_mini": False},
    "2002": {"otc": False, "has_mini": False}, "2006": {"otc": False, "has_mini": False},
    "2027": {"otc": False, "has_mini": False}, "2049": {"otc": False, "has_mini": True},
    "2059": {"otc": False, "has_mini": True}, "2105": {"otc": False, "has_mini": False},
    "2201": {"otc": False, "has_mini": False}, "2231": {"otc": False, "has_mini": False},
    "2301": {"otc": False, "has_mini": False}, "2303": {"otc": False, "has_mini": False},
    "2308": {"otc": False, "has_mini": True}, "2312": {"otc": False, "has_mini": False},
    "2313": {"otc": False, "has_mini": False}, "2317": {"otc": False, "has_mini": False},
    "2323": {"otc": False, "has_mini": False}, "2324": {"otc": False, "has_mini": False},
    "2327": {"otc": False, "has_mini": True}, "2328": {"otc": False, "has_mini": False},
    "2329": {"otc": False, "has_mini": False}, "2330": {"otc": False, "has_mini": True},
    "2331": {"otc": False, "has_mini": False}, "2332": {"otc": False, "has_mini": False},
    "2337": {"otc": False, "has_mini": False}, "2338": {"otc": False, "has_mini": False},
    "2340": {"otc": False, "has_mini": False}, "2344": {"otc": False, "has_mini": False},
    "2345": {"otc": False, "has_mini": True}, "2347": {"otc": False, "has_mini": False},
    "2352": {"otc": False, "has_mini": False}, "2353": {"otc": False, "has_mini": False},
    "2354": {"otc": False, "has_mini": False}, "2355": {"otc": False, "has_mini": False},
    "2356": {"otc": False, "has_mini": False}, "2357": {"otc": False, "has_mini": True},
    "2360": {"otc": False, "has_mini": True}, "2367": {"otc": False, "has_mini": False},
    "2368": {"otc": False, "has_mini": True}, "2371": {"otc": False, "has_mini": False},
    "2376": {"otc": False, "has_mini": True}, "2377": {"otc": False, "has_mini": False},
    "2379": {"otc": False, "has_mini": True}, "2382": {"otc": False, "has_mini": False},
    "2383": {"otc": False, "has_mini": True}, "2385": {"otc": False, "has_mini": False},
    "2388": {"otc": False, "has_mini": False}, "2392": {"otc": False, "has_mini": False},
    "2393": {"otc": False, "has_mini": False}, "2395": {"otc": False, "has_mini": False},
    "2401": {"otc": False, "has_mini": False}, "2404": {"otc": False, "has_mini": True},
    "2408": {"otc": False, "has_mini": False}, "2409": {"otc": False, "has_mini": False},
    "2412": {"otc": False, "has_mini": False}, "2421": {"otc": False, "has_mini": False},
    "2439": {"otc": False, "has_mini": False}, "2441": {"otc": False, "has_mini": False},
    "2449": {"otc": False, "has_mini": True}, "2454": {"otc": False, "has_mini": True},
    "2455": {"otc": False, "has_mini": False}, "2457": {"otc": False, "has_mini": False},
    "2458": {"otc": False, "has_mini": False}, "2474": {"otc": False, "has_mini": False},
    "2481": {"otc": False, "has_mini": False}, "2485": {"otc": False, "has_mini": False},
    "2486": {"otc": False, "has_mini": False}, "2489": {"otc": False, "has_mini": False},
    "2492": {"otc": False, "has_mini": False}, "2498": {"otc": False, "has_mini": False},
    "2515": {"otc": False, "has_mini": False}, "2520": {"otc": False, "has_mini": False},
    "2542": {"otc": False, "has_mini": False}, "2548": {"otc": False, "has_mini": False},
    "2603": {"otc": False, "has_mini": False}, "2605": {"otc": False, "has_mini": False},
    "2606": {"otc": False, "has_mini": False}, "2609": {"otc": False, "has_mini": False},
    "2610": {"otc": False, "has_mini": False}, "2615": {"otc": False, "has_mini": False},
    "2618": {"otc": False, "has_mini": False}, "2633": {"otc": False, "has_mini": False},
    "2634": {"otc": False, "has_mini": False}, "2801": {"otc": False, "has_mini": False},
    "2834": {"otc": False, "has_mini": False}, "2880": {"otc": False, "has_mini": False},
    "2881": {"otc": False, "has_mini": False}, "2882": {"otc": False, "has_mini": False},
    "2883": {"otc": False, "has_mini": False}, "2884": {"otc": False, "has_mini": False},
    "2885": {"otc": False, "has_mini": False}, "2886": {"otc": False, "has_mini": False},
    "2887": {"otc": False, "has_mini": False}, "2890": {"otc": False, "has_mini": False},
    "2891": {"otc": False, "has_mini": False}, "2892": {"otc": False, "has_mini": False},
    "2913": {"otc": False, "has_mini": False}, "2915": {"otc": False, "has_mini": False},
    "3005": {"otc": False, "has_mini": False}, "3006": {"otc": False, "has_mini": False},
    "3008": {"otc": False, "has_mini": True}, "3017": {"otc": False, "has_mini": True},
    "3019": {"otc": False, "has_mini": False}, "3034": {"otc": False, "has_mini": True},
    "3035": {"otc": False, "has_mini": False}, "3036": {"otc": False, "has_mini": False},
    "3037": {"otc": False, "has_mini": False}, "3042": {"otc": False, "has_mini": False},
    "3044": {"otc": False, "has_mini": False}, "3045": {"otc": False, "has_mini": False},
    "3078": {"otc": True, "has_mini": False}, "3081": {"otc": True, "has_mini": False},
    "3105": {"otc": True, "has_mini": True}, "3152": {"otc": True, "has_mini": False},
    "3189": {"otc": False, "has_mini": False}, "3211": {"otc": True, "has_mini": True},
    "3227": {"otc": True, "has_mini": False}, "3231": {"otc": False, "has_mini": False},
    "3260": {"otc": True, "has_mini": False}, "3264": {"otc": True, "has_mini": False},
    "3293": {"otc": True, "has_mini": True}, "3324": {"otc": True, "has_mini": True},
    "3374": {"otc": True, "has_mini": False}, "3376": {"otc": False, "has_mini": False},
    "3380": {"otc": False, "has_mini": False}, "3406": {"otc": False, "has_mini": True},
    "3443": {"otc": False, "has_mini": True}, "3481": {"otc": False, "has_mini": False},
    "3529": {"otc": True, "has_mini": True}, "3532": {"otc": False, "has_mini": False},
    "3533": {"otc": False, "has_mini": True}, "3552": {"otc": True, "has_mini": False},
    "3653": {"otc": False, "has_mini": True}, "3661": {"otc": False, "has_mini": True},
    "3665": {"otc": False, "has_mini": True}, "3673": {"otc": False, "has_mini": False},
    "3680": {"otc": True, "has_mini": True}, "3691": {"otc": True, "has_mini": False},
    "3702": {"otc": False, "has_mini": False}, "3706": {"otc": False, "has_mini": False},
    "3711": {"otc": False, "has_mini": True}, "3714": {"otc": False, "has_mini": False},
    "3718": {"otc": True, "has_mini": False}, "4123": {"otc": True, "has_mini": False},
    "4128": {"otc": True, "has_mini": False}, "4162": {"otc": True, "has_mini": False},
    "4736": {"otc": False, "has_mini": False}, "4743": {"otc": True, "has_mini": False},
    "4904": {"otc": False, "has_mini": False}, "4919": {"otc": False, "has_mini": False},
    "4938": {"otc": False, "has_mini": False}, "4958": {"otc": False, "has_mini": False},
    "5009": {"otc": True, "has_mini": False}, "5269": {"otc": False, "has_mini": True},
    "5274": {"otc": True, "has_mini": True}, "5347": {"otc": True, "has_mini": False},
    "5388": {"otc": False, "has_mini": False}, "5425": {"otc": True, "has_mini": False},
    "5457": {"otc": True, "has_mini": False}, "5483": {"otc": True, "has_mini": False},
    "5534": {"otc": False, "has_mini": False}, "5871": {"otc": False, "has_mini": False},
    "5876": {"otc": False, "has_mini": False}, "5880": {"otc": False, "has_mini": False},
    "5904": {"otc": True, "has_mini": True}, "6005": {"otc": False, "has_mini": False},
    "6116": {"otc": False, "has_mini": False}, "6121": {"otc": True, "has_mini": False},
    "6139": {"otc": False, "has_mini": True}, "6147": {"otc": True, "has_mini": False},
    "6153": {"otc": False, "has_mini": False}, "6173": {"otc": True, "has_mini": False},
    "6176": {"otc": False, "has_mini": False}, "6182": {"otc": True, "has_mini": False},
    "6188": {"otc": True, "has_mini": False}, "6213": {"otc": False, "has_mini": False},
    "6223": {"otc": True, "has_mini": True}, "6239": {"otc": False, "has_mini": False},
    "6245": {"otc": True, "has_mini": False}, "6257": {"otc": False, "has_mini": False},
    "6269": {"otc": False, "has_mini": False}, "6271": {"otc": False, "has_mini": False},
    "6274": {"otc": True, "has_mini": False}, "6278": {"otc": False, "has_mini": False},
    "6279": {"otc": True, "has_mini": False}, "6282": {"otc": False, "has_mini": False},
    "6285": {"otc": False, "has_mini": False}, "6290": {"otc": True, "has_mini": False},
    "6414": {"otc": False, "has_mini": False}, "6443": {"otc": False, "has_mini": False},
    "6472": {"otc": False, "has_mini": True}, "6488": {"otc": True, "has_mini": True},
    "6505": {"otc": False, "has_mini": False}, "6510": {"otc": True, "has_mini": True},
    "6526": {"otc": False, "has_mini": True}, "6547": {"otc": True, "has_mini": False},
    "6669": {"otc": False, "has_mini": True}, "6757": {"otc": False, "has_mini": False},
    "6770": {"otc": False, "has_mini": False}, "8039": {"otc": False, "has_mini": False},
    "8044": {"otc": True, "has_mini": False}, "8046": {"otc": False, "has_mini": True},
    "8069": {"otc": True, "has_mini": False}, "8086": {"otc": True, "has_mini": False},
    "8112": {"otc": False, "has_mini": False}, "8150": {"otc": False, "has_mini": False},
    "8163": {"otc": False, "has_mini": False}, "8299": {"otc": True, "has_mini": True},
    "8358": {"otc": True, "has_mini": False}, "8436": {"otc": True, "has_mini": False},
    "8932": {"otc": True, "has_mini": False}, "9904": {"otc": False, "has_mini": False},
    "9914": {"otc": False, "has_mini": False}, "9938": {"otc": False, "has_mini": False},
    "9939": {"otc": False, "has_mini": False}, "9945": {"otc": False, "has_mini": False},
    "9958": {"otc": False, "has_mini": True},
}

# 原始保證金適用比例 (查證來源：期交所保證金公告，查證時間點的數值)。
# 只收錄有明確查證到的代碼；沒有查到的一律用 DEFAULT_MARGIN_RATIO 這個保守估計值，
# 這是刻意設計成「寧可高估風險、少選一些股票，也不要低估風險」。
MARGIN_RATIO_TABLE = {
    "2330": 0.135, "2454": 0.2025, "3008": 0.162, "3406": 0.162, "2327": 0.2025,
    "2357": 0.162, "2308": 0.162, "2345": 0.2025, "3711": 0.162, "2449": 0.2025,
    "6669": 0.2025, "2376": 0.2025, "3017": 0.2025, "2368": 0.2025, "8046": 0.216,
    "2383": 0.2025, "2379": 0.162, "3443": 0.2025, "3661": 0.2025, "1519": 0.2025,
    "3324": 0.405,  # 雙鴻，風險係數極高的代表
    "2303": 0.162, "2317": 0.162, "2382": 0.2025, "3231": 0.162, "2356": 0.162,
    "2301": 0.162, "2421": 0.162, "2492": 0.2025, "1503": 0.2025, "1513": 0.162,
    "1609": 0.162, "3037": 0.2025, "3189": 0.2025, "6274": 0.2025, "1504": 0.2025,
    "2409": 0.2025, "3481": 0.2025, "2408": 0.2295, "2344": 0.216, "2337": 0.216,
    "2603": 0.162, "2609": 0.162, "2615": 0.162, "2618": 0.135, "2610": 0.135,
    "2881": 0.135, "2882": 0.135, "2377": 0.135,
}
DEFAULT_MARGIN_RATIO = 0.20  # 沒有查到明確數字的代碼，保守用20%估算


def is_valid_futures_code(code: str) -> bool:
    """檢查這個股票代碼是否真的有股票期貨商品存在。"""
    return code in STOCK_FUTURES_UNIVERSE


def get_yf_symbol(code: str) -> str:
    """回傳這個代碼在 yfinance 該用的完整代號 (.TW 或 .TWO)。"""
    info = STOCK_FUTURES_UNIVERSE.get(code)
    if info is None:
        raise ValueError(f"{code} 不是有效的股票期貨標的")
    suffix = ".TWO" if info["otc"] else ".TW"
    return f"{code}{suffix}"


def get_contract_multiplier(code: str, price: float) -> int:
    """
    回傳這筆交易該用的契約股數 (100 或 2000)。
    有小型契約的股票，統一優先使用小型契約 (100股)，因為在小資金(如20萬)情境下，
    小型契約的保證金負擔遠低於標準契約，更適合小額交易。
    沒有小型契約的股票，只能用標準契約 (2000股)。
    """
    info = STOCK_FUTURES_UNIVERSE.get(code)
    if info is None:
        raise ValueError(f"{code} 不是有效的股票期貨標的")
    return 100 if info["has_mini"] else 2000


def get_margin_ratio(code: str) -> float:
    return MARGIN_RATIO_TABLE.get(code, DEFAULT_MARGIN_RATIO)


def estimate_margin(code: str, price: float, lots: int) -> float:
    """估算這筆交易(多少口)需要的原始保證金總額(新台幣)。"""
    mult = get_contract_multiplier(code, price)
    ratio = get_margin_ratio(code)
    return price * mult * ratio * lots
