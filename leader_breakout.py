"""
leader_breakout.py — 把「大漲股共同點研究」(winners_study)的發現，變成一個有資金限制的真實投資組合回測。

研究問題：如果真的拿一個小帳戶(預設NT$200,000、現金帳戶、可買零股)照事先登錄的規則交易
「創一年新高的強勢股」或「大跌後的高beta電子領頭股」，2018-01-01 到最新，能不能年化 >10%、
而且在 2018-2019 / 2020-2021 / 2022-2023 / 2024-最新 四段都站得住？

資料(全部沿用 winners_study.py 的下載器與快取，不重寫)：
  ・股票清單+產業別：TWSE ISIN(上市+上櫃普通股) → ws.load_universe
    ⚠️ 只有「目前仍上市櫃」的股票 → 存活者偏誤(下市股不在裡面，績效偏樂觀)。
  ・股價：yfinance 日K(還原權值) → ws.load_prices / ws.prepare_price_frame，從 2016-06-01 起抓(252日特徵暖機)。
  ・月營收：MOPS 彙總表 → ws.load_revenue / ws.build_revenue_features / ws.revenue_features_on_dates
    (point-in-time：M月營收從M+1月11日起才看得到)，從 2016-06 起抓。
  ・加權指數 ^TWII(價格指數，不含股利) → ws.load_taiex；0050.TW(還原，含股利) → ws.load_prices。
  ・ATR、252日新高、E_A訊號等個股指標 → ws.compute_indicators(沿用repo的compute_atr_correct)。
  ・bootstrap → robustness_analysis.bootstrap_resample_pnl(n=1000, seed=42，repo慣例)。

每天的特徵(point-in-time，只用到當天收盤(含)以前的資料；寬表 日期×代號、向量化計算)：
  252日高/低、ATR14、MA60、20日平均成交金額(原始收盤×成交量)、6個月報酬與當天在「合格股票」中的百分位(RS)、
  營收3個月平均YoY(PIT)、對^TWII的250日beta與當天的五分位、電子股旗標(產業別)。
  合格(eligible)：當天有有效價格、ATR有值、20日平均成交金額 ≥ NT$3,000萬。

用法：
  python leader_breakout.py --output-dir results_leader [--start 2018-01-01] [--end ''] [--capital 200000] [--refresh]
"""
import argparse
import os
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

import winners_study as ws
from robustness_analysis import bootstrap_resample_pnl, summarize_bootstrap

# ---------------------------------------------------------------------------
# 參數(事先登錄，不依結果調整)
# ---------------------------------------------------------------------------
DEFAULT_START = "2018-01-01"
WARMUP_MONTHS = 19                    # 2018-01-01 往前19個月 = 2016-06-01(股價與月營收都從這裡開始抓)
DEFAULT_CAPITAL = 200_000
RISK_PCT = 0.015                      # 每筆交易風險 = 目前權益的1.5%
MIN_FEE = 20.0                        # 每筆委託最低手續費(零股)
FEE_RATE = ws.FEE_RATE                # 0.1425%
TAX_RATE = ws.TAX_RATE                # 0.3%(賣出)
ELIG_TURNOVER = 30_000_000            # 合格：20日平均成交金額 ≥ NT$3,000萬
INITIAL_STOP_ATR = ws.INITIAL_STOP_ATR    # 初始停損 = 進場價 − 2×ATR14(訊號日)
X2_TRAIL_PCT = ws.X2_TRAIL_PCT            # 20%移動停損
RS_TOP = ws.RS_TOP                        # RS ≥ 第80百分位
REV_YOY_MIN = ws.REV_YOY_MIN              # 營收3個月平均YoY > 20%
BETA_WINDOW = 250
BETA_MIN_OBS = 120
LIMIT_UP_RATIO = 1.095                # 開盤 ≥ 前一日原始收盤×1.095 且 開=高=低 → 視為一價漲停鎖死，買不到
CRASH_DD = 0.15                       # E_CRASH：^TWII 自252日高點回落 ≥15%
CRASH_LOOKBACK = 60                   # ……發生在最近60個交易日內
CRASH_MA = 20                         # 今天收盤由下往上穿越MA20
CRASH_REARM_DAYS = 120                # 觸發後：^TWII 先創252日新高、或過120個交易日，才重新武裝
FILTER_MA = 60                        # 大盤濾網：^TWII 收盤 > MA60(只管新進場)
REF_ETF = "0050"
ELEC_INDUSTRIES = ("半導體業", "電腦及週邊設備業", "電子零組件業", "通信網路業", "光電業", "其他電子業",
                   "電子通路業", "資訊服務業")
ENTRY_RULES = {
    "E_A": "收盤 > 前252日最高收盤，且前20個交易日都沒有這種新高",
    "E_B": "E_A 且 營收3個月平均YoY(PIT) > 20%",
    "E_C": "E_A 且 RS(6個月報酬百分位) ≥ 80",
    "E_D": "E_B 且 RS ≥ 80",
    "E_CRASH": "大跌後買領頭股：^TWII 近60日內曾自252日高回落≥15%，今天收盤首次站上MA20 → 買高beta(第5分位)電子股中20日成交金額前N名",
}
EXIT_RULES = {
    "X2": "移動停損20%(自進場後最高收盤；停損單，跳空開低以開盤價成交)",
    "X3": "收盤 < MA60 → 隔日開盤出場",
}
SUB_WINDOWS = [("2018-2019", "2018-01-01", "2019-12-31"),
               ("2020-2021", "2020-01-01", "2021-12-31"),
               ("2022-2023", "2022-01-01", "2023-12-31"),
               ("2024-最新", "2024-01-01", None)]
FULL_WINDOW = "全期"
PASS_CAGR = 0.10
PASS_PF = 1.2
BOOT_N = 1000
BOOT_SEED = 42
REF_TWII = "^TWII買進持有(價格指數，不含股利)"
REF_0050 = "0050買進持有(還原，含股利)"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leader_cache")


# ===========================================================================
# 變體(事先登錄：4進場×2出場×2K×2濾網 + E_CRASH×2出場×2K + E_D/X3/K∈{1,3}改用成交金額排序 = 38)
# ===========================================================================
def variant_name(entry, exit_kind, k, filt=None, rank="RS"):
    parts = [entry, exit_kind, f"K{k}"]
    if filt is not None:
        parts.append("大盤濾網" if filt else "無濾網")
    if rank != "RS":
        parts.append("成交金額排序")
    return "｜".join(parts)


def build_variants() -> list:
    out = []
    for e in ["E_A", "E_B", "E_C", "E_D"]:
        for x in ["X2", "X3"]:
            for k in [1, 3]:
                for f in [False, True]:
                    out.append({"name": variant_name(e, x, k, f), "entry": e, "exit": x, "K": k,
                                "filter": f, "rank": "RS"})
    for x in ["X2", "X3"]:
        for k in [1, 3]:
            out.append({"name": variant_name("E_CRASH", x, k), "entry": "E_CRASH", "exit": x, "K": k,
                        "filter": None, "rank": "成交金額"})
    for k in [1, 3]:
        out.append({"name": variant_name("E_D", "X3", k, False, rank="成交金額"), "entry": "E_D", "exit": "X3",
                    "K": k, "filter": False, "rank": "成交金額"})
    return out


# ===========================================================================
# 特徵(寬表)
# ===========================================================================
_NUM_COLS = ["open", "high", "low", "close", "raw_close", "atr14", "ma60", "turnover20", "ret126",
             "hh252", "hh252_prev", "ll252"]


def stock_indicators(px: pd.DataFrame) -> pd.DataFrame:
    """沿用 ws.compute_indicators(往回看的滾動指標)，只補上252日最低收盤。"""
    ind = ws.compute_indicators(px)
    ind["ll252"] = ind["close"].rolling(252, min_periods=252).min()
    return ind


def market_features(twii: pd.Series) -> pd.DataFrame:
    """^TWII 在自己交易日上的特徵：MA20/MA60、252日高、回落幅度、穿越MA20、大盤濾網、E_CRASH 觸發日。"""
    cols = ["close", "ma20", "ma60", "hh252", "dd", "cross20", "filter_ok", "crash"]
    if twii is None or len(twii) == 0:
        return pd.DataFrame(columns=cols)
    c = pd.Series(twii, dtype=float).dropna().sort_index()
    c = c[~c.index.duplicated(keep="last")]
    df = pd.DataFrame({"close": c})
    df["ma20"] = c.rolling(CRASH_MA, min_periods=CRASH_MA).mean()
    df["ma60"] = c.rolling(FILTER_MA, min_periods=FILTER_MA).mean()
    df["hh252"] = c.rolling(252, min_periods=252).max()
    df["dd"] = c / df["hh252"] - 1
    above = c > df["ma20"]
    prev_above = above.shift(1, fill_value=True)
    df["cross20"] = above & ~prev_above & df["ma20"].notna() & df["ma20"].shift(1).notna()
    df["filter_ok"] = (c > df["ma60"]) & df["ma60"].notna()
    df["crash"] = crash_signal_days(df["dd"].to_numpy(float), df["cross20"].to_numpy(bool))
    return df


def crash_signal_days(dd: np.ndarray, cross: np.ndarray, dd_thr: float = CRASH_DD,
                      lookback: int = CRASH_LOOKBACK, rearm_days: int = CRASH_REARM_DAYS) -> np.ndarray:
    """
    E_CRASH 狀態機(逐日，只用到當天(含)以前)：
      武裝中 且 最近lookback個交易日內(含今天、且在上次重新武裝之後)曾經 dd ≤ −15% 且 今天收盤穿越MA20 → 觸發，解除武裝。
      解除武裝後：某天創252日新高(dd ≥ 0) 或 距觸發日滿rearm_days個交易日 → 重新武裝(從那天之後的回落才算數)。
    """
    n = len(dd)
    out = np.zeros(n, dtype=bool)
    armed = True
    rearm_idx = -1          # 只有這天之後的深跌才算新的一段
    fire_idx = None
    last_deep = -10 ** 9
    for t in range(n):
        if not armed:
            if (np.isfinite(dd[t]) and dd[t] >= 0) or (t - fire_idx >= rearm_days):
                armed = True
                rearm_idx = t
        if np.isfinite(dd[t]) and dd[t] <= -dd_thr and t > rearm_idx:
            last_deep = t
        if armed and cross[t] and last_deep > rearm_idx and t - last_deep < lookback:
            out[t] = True
            armed = False
            fire_idx = t
    return out


def build_panels(universe: pd.DataFrame, raw_prices: dict, revenue_long: pd.DataFrame, taiex: pd.Series) -> dict:
    """回傳 P：cal(DatetimeIndex)、codes、numpy 寬表(T×N) 與市場序列。全部 point-in-time。"""
    uni = universe.drop_duplicates("code").copy()
    uni["code"] = uni["code"].astype(str)
    uni = uni.set_index("code")
    pxs = {}
    for code in uni.index:
        px = ws.prepare_price_frame(raw_prices.get(code))
        if len(px) >= 30:
            pxs[code] = px
    if not pxs:
        raise ValueError("沒有任何股票有足夠的股價資料")
    cal = ws.build_calendar(pxs)
    codes = sorted(pxs)
    T, N = len(cal), len(codes)
    A = {c: np.full((T, N), np.nan) for c in _NUM_COLS}
    A["entry_a"] = np.zeros((T, N), dtype=bool)
    for j, code in enumerate(codes):
        ind = stock_indicators(pxs.pop(code))
        pos = cal.get_indexer(ind.index)
        ok = pos >= 0
        rows = pos[ok]
        for c in _NUM_COLS:
            A[c][rows, j] = ind[c].to_numpy(dtype=float)[ok]
        A["entry_a"][rows, j] = ind["entry_a"].to_numpy(dtype=bool)[ok]
    valid = np.isfinite(A["close"]) & np.isfinite(A["open"]) & (A["close"] > 0) & (A["open"] > 0)
    elig = valid & np.isfinite(A["atr14"]) & (A["atr14"] > 0) & (A["turnover20"] >= ELIG_TURNOVER)
    A["valid"] = valid
    A["elig"] = elig
    A["close_ff"] = pd.DataFrame(A["close"]).ffill().to_numpy()
    # RS：當天合格股票中，6個月報酬的百分位
    ret = np.where(elig, A["ret126"], np.nan)
    A["rs"] = pd.DataFrame(ret).rank(axis=1, pct=True).to_numpy()
    # 月營收3個月平均YoY(PIT)
    rev_feats = ws.build_revenue_features(revenue_long) if revenue_long is not None and len(revenue_long) else {}
    A["yoy3"] = np.full((T, N), np.nan)
    n_rev = 0
    for j, code in enumerate(codes):
        rf = rev_feats.get(code)
        if rf is not None and len(rf):
            A["yoy3"][:, j] = ws.revenue_features_on_dates(rf, cal)["yoy3"].to_numpy(dtype=float)
            n_rev += 1
    # 市場：^TWII 對齊交易日曆
    mkt = market_features(taiex)
    mk = mkt.reindex(cal)
    m_close = mk["close"].to_numpy(dtype=float)
    mret = m_close / np.r_[np.nan, m_close[:-1]] - 1
    A["beta"] = rolling_beta(A["close"], mret)
    beta_e = np.where(elig, A["beta"], np.nan)
    pct = pd.DataFrame(beta_e).rank(axis=1, pct=True).to_numpy()
    A["beta_q"] = np.where(np.isfinite(pct), np.clip(np.ceil(pct * 5), 1, 5), np.nan)
    elec = np.array([str(uni.loc[c, "industry"]) in ELEC_INDUSTRIES if "industry" in uni.columns else False
                     for c in codes], dtype=bool)
    info = uni.reindex(codes)
    return {"cal": cal, "codes": codes, "A": A, "elec": elec,
            "names": info["name"].fillna("").astype(str).tolist() if "name" in info.columns else [""] * N,
            "industries": info["industry"].fillna("").astype(str).tolist() if "industry" in info.columns else [""] * N,
            "mkt": mkt, "mkt_cal": mk,
            "filter_ok": mk["filter_ok"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "crash_day": mk["crash"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "n_revenue_codes": n_rev, "twii_ok": bool(len(mkt))}


def rolling_beta(close: np.ndarray, mret: np.ndarray, window: int = BETA_WINDOW, min_obs: int = BETA_MIN_OBS) -> np.ndarray:
    """向量化 beta = cov(個股日報酬, 大盤日報酬)/var(大盤)，過去window個交易日、兩者都有值的天數≥min_obs。"""
    prev = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        r = close / prev - 1
    M = np.broadcast_to(mret[:, None], r.shape)
    both = np.isfinite(r) & np.isfinite(M)
    X = np.where(both, r, 0.0)
    Y = np.where(both, M, 0.0)

    def rs(a):
        return pd.DataFrame(a).rolling(window, min_periods=1).sum().to_numpy()
    n = rs(both.astype(float))
    sx, sy, sxy, syy = rs(X), rs(Y), rs(X * Y), rs(Y * Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (sxy - sx * sy / n) / (n - 1)
        var = (syy - sy * sy / n) / (n - 1)
        beta = cov / var
    beta[(n < min_obs) | ~(var > 1e-14)] = np.nan
    return beta


def entry_signals(P: dict) -> dict:
    """{規則: T×N bool}；訊號在收盤t成立(只用t以前資料)，隔天開盤進場。"""
    A = P["A"]
    a = A["entry_a"] & A["elig"]
    with np.errstate(invalid="ignore"):
        rev_ok = A["yoy3"] > REV_YOY_MIN
        rs_ok = A["rs"] >= RS_TOP
        crash = (P["crash_day"][:, None] & A["elig"] & P["elec"][None, :] & (A["beta_q"] == 5))
    b = a & rev_ok
    return {"E_A": a, "E_B": b, "E_C": a & rs_ok, "E_D": b & rs_ok, "E_CRASH": crash}


def variant_signal(sigs: dict, P: dict, v: dict) -> np.ndarray:
    """大盤濾網：訊號日 ^TWII 收盤 > MA60 才允許新進場(E_CRASH 的 filter=None，不套)。"""
    sig = sigs[v["entry"]]
    if v.get("filter"):
        sig = sig & P["filter_ok"][:, None]
    return sig


# ===========================================================================
# 投資組合模擬
# ===========================================================================
def buy_fee(notional: float) -> float:
    return max(MIN_FEE, notional * FEE_RATE)


def sell_cost(notional: float) -> float:
    return max(MIN_FEE, notional * FEE_RATE) + notional * TAX_RATE


def position_shares(equity: float, cash: float, price: float, atr: float, k: int) -> int:
    """
    風險1.5%：股數 = floor(權益×1.5% / (2×ATR))(=訊號收盤到停損參考價的距離)；
    上限：部位金額 ≤ 權益/K、買進金額+手續費(最低20元) ≤ 現金。零股可，最少1股，不足回傳0。
    """
    if not (np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0 and equity > 0):
        return 0
    risk_per_share = INITIAL_STOP_ATR * atr
    n = int(np.floor(equity * RISK_PCT / risk_per_share))
    n = min(n, int(np.floor(equity / k / price)))
    n_cash = int(np.floor(max(cash, 0.0) / (price * (1 + FEE_RATE))))
    while n_cash > 0 and n_cash * price + buy_fee(n_cash * price) > cash + 1e-9:
        n_cash -= 1
    n = min(n, n_cash)
    return max(n, 0)


def is_limit_up_open(A: dict, t: int, j: int) -> bool:
    """進場日t：開盤(換回原始價) ≥ 前一日原始收盤×1.095 且 開=高=低(一價鎖漲停)。"""
    if t < 1:
        return False
    o, h, l, c = A["open"][t, j], A["high"][t, j], A["low"][t, j], A["close"][t, j]
    rc, prc = A["raw_close"][t, j], A["raw_close"][t - 1, j]
    if not all(np.isfinite(v) for v in (o, h, l, c, rc, prc)) or c <= 0 or prc <= 0:
        return False
    raw_open = o * rc / c
    tol = 1e-9 * max(abs(o), 1.0)
    return raw_open >= prc * LIMIT_UP_RATIO - 1e-9 and abs(h - o) <= tol and abs(l - o) <= tol


def rank_order(cols: np.ndarray, primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """依 primary 由大到小、同分看 secondary 由大到小；NaN 排最後。"""
    p = np.where(np.isfinite(primary), primary, -np.inf)
    s = np.where(np.isfinite(secondary), secondary, -np.inf)
    order = np.lexsort((-s, -p))
    return cols[order]


def simulate_portfolio(P: dict, sig: np.ndarray, i0: int, i1: int, exit_kind: str, k: int,
                       capital: float = DEFAULT_CAPITAL, rank: str = "RS") -> dict:
    """
    日迴圈(只在[i0, i1])：
      開盤：(1)前一天收盤跌破MA60的部位以開盤價出場；(2)開盤已低於停損 → 開盤價出場(跳空)；
            (3)前一天收盤的訊號依排序補滿空位，開盤價進場(一價漲停跳過，換下一檔)；
      盤中：最低價碰到停損 → 停損價出場(進場當天也算)；
      收盤：更新最高收盤/移動停損(隔天生效)、X3檢查、以收盤(沒有成交就用最近收盤)計算權益；
            產生今天的訊號(只在 t < i1)。
    權益 = 現金 + Σ股數×收盤(不扣未來賣出成本)。訊號日權益(=進場前一天收盤)拿來算部位大小。
    """
    A = P["A"]
    o, h, l, c, cff = A["open"], A["high"], A["low"], A["close"], A["close_ff"]
    atr, ma60, valid = A["atr14"], A["ma60"], A["valid"]
    if rank == "RS":
        prim, sec = A["rs"], A["turnover20"]
    else:
        prim, sec = A["turnover20"], A["rs"]
    cash = float(capital)
    positions = []
    trades = []
    n_days = i1 - i0 + 1
    equity = np.zeros(n_days)
    npos = np.zeros(n_days, dtype=int)
    invested = np.zeros(n_days)
    pending = None
    eq_prev = float(capital)
    cnt = {"signals": 0, "limit_up_skipped": 0, "size_skipped": 0, "slot_skipped": 0, "no_price_skipped": 0}

    def close_pos(p, t, px, reason):
        nonlocal cash
        notional = p["shares"] * px
        cost = sell_cost(notional)
        cash += notional - cost
        pnl = notional - cost - p["cost_basis"]
        trades.append({"code_idx": p["j"], "signal_idx": p["sig_idx"], "entry_idx": p["entry_idx"], "exit_idx": t,
                       "entry_price": p["entry_px"], "exit_price": float(px), "shares": p["shares"],
                       "buy_fee": p["buy_fee"], "sell_cost": float(cost), "pnl": float(pnl),
                       "ret": float(pnl / p["cost_basis"]), "exit_reason": reason,
                       "hold_days": int(t - p["entry_idx"]), "open_at_end": False, "stop": p["stop"],
                       "rank_value": p["rank_value"]})

    def stop_reason(p):
        return "初始停損" if p["stop"] <= p["init_stop"] + 1e-12 else "移動停損"

    for t in range(i0, i1 + 1):
        # ---- 開盤：MA60出場、跳空停損
        keep = []
        for p in positions:
            j = p["j"]
            if not valid[t, j]:
                keep.append(p)
                continue
            if p["pending_exit"]:
                close_pos(p, t, o[t, j], "跌破MA60")
            elif o[t, j] <= p["stop"]:
                close_pos(p, t, o[t, j], stop_reason(p) + "(跳空)")
            else:
                keep.append(p)
        positions = keep
        # ---- 開盤：進場
        if pending is not None and len(pending):
            held = {p["j"] for p in positions}
            free = k - len(positions)
            for j in pending:
                if j in held:
                    continue
                if free <= 0:
                    cnt["slot_skipped"] += 1
                    continue
                if not valid[t, j]:
                    cnt["no_price_skipped"] += 1
                    continue
                if is_limit_up_open(A, t, j):
                    cnt["limit_up_skipped"] += 1
                    continue
                a = atr[t - 1, j]
                px = o[t, j]
                n = position_shares(eq_prev, cash, px, a, k)
                if n < 1:
                    cnt["size_skipped"] += 1
                    continue
                notional = n * px
                fee = buy_fee(notional)
                cash -= notional + fee
                init_stop = px - INITIAL_STOP_ATR * a
                positions.append({"j": j, "shares": n, "entry_px": float(px), "entry_idx": t, "sig_idx": t - 1,
                                  "init_stop": init_stop, "stop": init_stop, "hc": -np.inf, "pending_exit": False,
                                  "cost_basis": notional + fee, "buy_fee": fee,
                                  "rank_value": float(prim[t - 1, j]) if np.isfinite(prim[t - 1, j]) else np.nan})
                held.add(j)
                free -= 1
        pending = None
        # ---- 盤中：停損
        keep = []
        for p in positions:
            j = p["j"]
            if valid[t, j] and l[t, j] <= p["stop"]:
                close_pos(p, t, p["stop"], stop_reason(p))
            else:
                keep.append(p)
        positions = keep
        # ---- 收盤：更新停損/MA60、權益
        for p in positions:
            j = p["j"]
            if not valid[t, j]:
                continue
            p["hc"] = max(p["hc"], c[t, j])
            if exit_kind == "X2":
                p["stop"] = max(p["stop"], p["hc"] * (1 - X2_TRAIL_PCT))
            elif exit_kind == "X3" and np.isfinite(ma60[t, j]) and c[t, j] < ma60[t, j]:
                p["pending_exit"] = True
        mv = sum(p["shares"] * cff[t, p["j"]] for p in positions)
        eq = cash + mv
        equity[t - i0] = eq
        npos[t - i0] = len(positions)
        invested[t - i0] = mv / eq if eq > 0 else 0.0
        eq_prev = eq
        # ---- 收盤：今天的訊號(明天開盤進場)
        if t < i1:
            row = np.flatnonzero(sig[t])
            if len(row):
                cnt["signals"] += len(row)
                pending = rank_order(row, prim[t, row], sec[t, row])
    open_pos = []
    for p in positions:
        j = p["j"]
        px = cff[i1, j]
        open_pos.append({"code_idx": j, "signal_idx": p["sig_idx"], "entry_idx": p["entry_idx"], "exit_idx": i1,
                         "entry_price": p["entry_px"], "exit_price": float(px), "shares": p["shares"],
                         "buy_fee": p["buy_fee"], "sell_cost": 0.0,
                         "pnl": float(p["shares"] * px - p["cost_basis"]),
                         "ret": float((p["shares"] * px - p["cost_basis"]) / p["cost_basis"]),
                         "exit_reason": "期末未平倉(以最後收盤估值)", "hold_days": int(i1 - p["entry_idx"]),
                         "open_at_end": True, "stop": p["stop"], "rank_value": p["rank_value"]})
    return {"equity": equity, "npos": npos, "invested": invested, "trades": trades, "open": open_pos,
            "counts": cnt, "cash": cash}


# ===========================================================================
# 績效
# ===========================================================================
def cagr(v0: float, v1: float, d0, d1) -> float:
    years = (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.25
    if not (v0 > 0 and v1 > 0) or years <= 0:
        return np.nan
    return (v1 / v0) ** (1 / years) - 1


def max_drawdown(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return np.nan
    v = eq.to_numpy(dtype=float)
    peak = np.maximum.accumulate(v)
    return float((v / peak - 1).min())


def yearly_returns(eq: pd.Series) -> pd.Series:
    """eq 第一列是基準(開始前一天)。每年 = 年底權益/前一年底(或基準)權益 − 1；最後一年可能是部分年度。"""
    if eq is None or len(eq) < 2:
        return pd.Series(dtype=float)
    out = {}
    base = eq.iloc[0]
    body = eq.iloc[1:]
    for y, g in body.groupby(body.index.year):
        out[int(y)] = g.iloc[-1] / base - 1
        base = g.iloc[-1]
    return pd.Series(out, dtype=float)


def window_slice(eq: pd.Series, w0, w1) -> pd.Series:
    """取 [w0, w1] 的權益，前面接上「w0之前最後一個收盤」當基準；區間內沒有資料 → 空。"""
    w0 = pd.Timestamp(w0)
    w1 = pd.Timestamp(w1) if w1 is not None else eq.index[-1]
    inside = eq[(eq.index >= w0) & (eq.index <= w1)]
    before = eq[eq.index < w0]
    if inside.empty or before.empty:
        return pd.Series(dtype=float)
    return pd.concat([before.iloc[-1:], inside])


def trade_stats(pnls: np.ndarray, holds: np.ndarray) -> dict:
    n = len(pnls)
    gp = float(pnls[pnls > 0].sum()) if n else 0.0
    gl = float(-pnls[pnls < 0].sum()) if n else 0.0
    total = float(pnls.sum()) if n else 0.0
    top5 = float(np.sort(pnls)[::-1][:5].sum()) if n else 0.0
    if n:
        boot = summarize_bootstrap(bootstrap_resample_pnl([{"pnl_ntd": float(p)} for p in pnls],
                                                          n_resamples=BOOT_N, seed=BOOT_SEED))["pct_positive"] / 100
    else:
        boot = np.nan
    return {"trades": n, "win_rate": float((pnls > 0).mean()) if n else np.nan,
            "pf": gp / gl if gl > 0 else (np.inf if gp > 0 else np.nan),
            "pnl": total, "pnl_ex_top5": total - top5 if n else np.nan,
            "avg_hold": float(np.mean(holds)) if n else np.nan, "boot_pos": boot}


def window_metrics(eq: pd.Series, npos: pd.Series = None, invested: pd.Series = None,
                   tr: pd.DataFrame = None) -> dict:
    """eq 已含基準列。tr = 這段期間內「出場」的已平倉交易(None = 參考指數，沒有交易統計)。"""
    if eq is None or len(eq) < 2:
        return {k: np.nan for k in ["cagr", "total", "mdd", "worst_year", "pct_years_pos", "exposure",
                                    "avg_pos", "avg_invested", "trades", "win_rate", "pf", "pnl", "pnl_ex_top5",
                                    "avg_hold", "boot_pos", "start", "end"]}
    yr = yearly_returns(eq)
    out = {"start": eq.index[1], "end": eq.index[-1],
           "cagr": cagr(eq.iloc[0], eq.iloc[-1], eq.index[0], eq.index[-1]),
           "total": eq.iloc[-1] / eq.iloc[0] - 1, "mdd": max_drawdown(eq),
           "worst_year": float(yr.min()) if len(yr) else np.nan,
           "pct_years_pos": float((yr > 0).mean()) if len(yr) else np.nan}
    if npos is not None:
        np_w = npos.reindex(eq.index[1:])
        out["exposure"] = float((np_w > 0).mean())
        out["avg_pos"] = float(np_w.mean())
        out["avg_invested"] = float(invested.reindex(eq.index[1:]).mean())
    else:
        out.update({"exposure": np.nan, "avg_pos": np.nan, "avg_invested": np.nan})
    if tr is not None:
        out.update(trade_stats(tr["pnl"].to_numpy(dtype=float), tr["hold_days"].to_numpy(dtype=float)))
    else:
        out.update({"trades": np.nan, "win_rate": np.nan, "pf": np.nan, "pnl": np.nan, "pnl_ex_top5": np.nan,
                    "avg_hold": np.nan, "boot_pos": np.nan})
    return out


def windows_for(cal_start, cal_end) -> list:
    """[(名稱, 起, 迄)]：全期 + 四段(超出回測範圍的段落會在metrics裡變成空值)。"""
    return [(FULL_WINDOW, pd.Timestamp(cal_start), pd.Timestamp(cal_end))] + \
           [(n, pd.Timestamp(a), pd.Timestamp(b) if b else pd.Timestamp(cal_end)) for n, a, b in SUB_WINDOWS]


def pass_checks(wm: dict, ref_mdd_0050: float) -> dict:
    """wm：{窗口名: metrics}。事先登錄的PASS規則(四項都要過)。"""
    subs = [n for n, _, _ in SUB_WINDOWS]
    full = wm.get(FULL_WINDOW, {})

    def ok(v, thr):
        return bool(np.isfinite(v) and v > thr) if v is not None else False
    c1 = ok(full.get("cagr", np.nan), PASS_CAGR)
    c2 = all(ok(wm.get(n, {}).get("cagr", np.nan), 0.0) for n in subs)
    m = full.get("mdd", np.nan)
    c3 = bool(np.isfinite(m) and np.isfinite(ref_mdd_0050) and m > ref_mdd_0050)
    c4 = all((lambda v: bool(v is not None and not pd.isna(v) and v > PASS_PF))(wm.get(n, {}).get("pf", np.nan))
             for n in subs)
    return {"chk_cagr": c1, "chk_sub_cagr": c2, "chk_mdd": c3, "chk_sub_pf": c4, "pass": c1 and c2 and c3 and c4}


def axis_view(vdf: pd.DataFrame) -> pd.DataFrame:
    """每個設計軸(進場/出場/K/濾網)的各選項：全期CAGR中位數、最差分段CAGR中位數(只看36個主變體)。"""
    main = vdf[vdf["rank"] == "RS"].copy() if "rank" in vdf.columns else vdf.copy()
    main = pd.concat([main, vdf[(vdf["entry"] == "E_CRASH")]]).drop_duplicates("variant")
    main["filter_label"] = main["filter"].map({True: "大盤濾網", False: "無濾網"}).fillna("不適用(E_CRASH)")
    rows = []
    for axis, col in [("進場", "entry"), ("出場", "exit"), ("K", "K"), ("濾網", "filter_label")]:
        for val, g in main.groupby(col, sort=True):
            rows.append({"axis": axis, "value": f"K{val}" if col == "K" else str(val), "n": int(len(g)),
                         "median_cagr": float(g["cagr"].median()),
                         "median_worst_sub_cagr": float(g["worst_sub_cagr"].median()),
                         "n_pass": int(g["pass"].sum())})
    return pd.DataFrame(rows)


# ===========================================================================
# 主流程(純計算，測試用合成資料)
# ===========================================================================
def run_backtest(universe, raw_prices, revenue_long, taiex, ref_prices, start, end,
                 capital: float = DEFAULT_CAPITAL, variants: list = None) -> dict:
    t_feat = time.time()
    P = build_panels(universe, raw_prices, revenue_long, taiex)
    cal, A = P["cal"], P["A"]
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    i0 = int(cal.searchsorted(start))
    i1 = int(cal.searchsorted(end, side="right")) - 1
    if i1 - i0 < 2 or i0 < 1:
        raise ValueError("回測區間內交易日不足，或區間前沒有暖機資料")
    sigs = entry_signals(P)
    feat_sec = time.time() - t_feat
    variants = variants if variants is not None else build_variants()
    eq_idx = cal[i0 - 1:i1 + 1]
    t_sim = time.time()
    results = {}
    for v in variants:
        sig = variant_signal(sigs, P, v)
        r = simulate_portfolio(P, sig, i0, i1, v["exit"], v["K"], capital, rank=v["rank"])
        r["equity_s"] = pd.Series(np.r_[capital, r["equity"]], index=eq_idx)
        r["npos_s"] = pd.Series(r["npos"], index=cal[i0:i1 + 1])
        r["inv_s"] = pd.Series(r["invested"], index=cal[i0:i1 + 1])
        results[v["name"]] = r
    sim_sec = time.time() - t_sim

    # ---- 參考：^TWII、0050
    refs = {}
    twc = P["mkt"]["close"] if P["twii_ok"] else pd.Series(dtype=float)
    ref_px = ws.prepare_price_frame(ref_prices) if ref_prices is not None else None
    for name, s in [(REF_TWII, twc), (REF_0050, ref_px["close"] if ref_px is not None and len(ref_px) else None)]:
        if s is None or len(s) == 0:
            refs[name] = None
            continue
        s = s.reindex(eq_idx).ffill()
        if not np.isfinite(s.iloc[0]):
            fv = s.first_valid_index()
            if fv is None:
                refs[name] = None
                continue
            s = s.bfill()
        refs[name] = capital * s / s.iloc[0]

    # ---- 交易表
    codes, names, inds = P["codes"], P["names"], P["industries"]
    trows = []
    for v in variants:
        r = results[v["name"]]
        for tr in r["trades"] + r["open"]:
            j = tr["code_idx"]
            trows.append({"variant": v["name"], "entry_rule": v["entry"], "exit_rule": v["exit"], "K": v["K"],
                          "code": codes[j], "name": names[j], "industry": inds[j],
                          "signal_date": cal[tr["signal_idx"]], "entry_date": cal[tr["entry_idx"]],
                          "exit_date": cal[tr["exit_idx"]], **{k: tr[k] for k in (
                              "entry_price", "exit_price", "shares", "buy_fee", "sell_cost", "pnl", "ret",
                              "exit_reason", "hold_days", "open_at_end", "rank_value")}})
    tcols = ["variant", "entry_rule", "exit_rule", "K", "code", "name", "industry", "signal_date", "entry_date",
             "entry_price", "exit_date", "exit_price", "shares", "buy_fee", "sell_cost", "pnl", "ret",
             "exit_reason", "hold_days", "open_at_end", "rank_value"]
    trades_df = pd.DataFrame(trows, columns=tcols)

    # ---- 各窗口績效
    wins = windows_for(cal[i0], cal[i1])
    ref_wm = {}
    for name, eq in refs.items():
        ref_wm[name] = {wn: window_metrics(window_slice(eq, a, b)) if eq is not None else window_metrics(None)
                        for wn, a, b in wins}
    ref_mdd_0050 = ref_wm[REF_0050][FULL_WINDOW]["mdd"]
    wrows, vrows = [], []
    for v in variants:
        r = results[v["name"]]
        tv = trades_df[(trades_df["variant"] == v["name"]) & ~trades_df["open_at_end"].astype(bool)]
        wm = {}
        for wn, a, b in wins:
            eqw = window_slice(r["equity_s"], a, b)
            tw = tv[(tv["exit_date"] >= a) & (tv["exit_date"] <= b)]
            wm[wn] = window_metrics(eqw, r["npos_s"], r["inv_s"], tw) if len(eqw) else window_metrics(None)
            wrows.append({"variant": v["name"], "window": wn, **wm[wn]})
        chk = pass_checks(wm, ref_mdd_0050)
        subs = [wm[n]["cagr"] for n, _, _ in SUB_WINDOWS]
        op = r["open"]
        vrows.append({"variant": v["name"], "entry": v["entry"], "exit": v["exit"], "K": v["K"],
                      "filter": v["filter"], "rank": v["rank"], **wm[FULL_WINDOW],
                      "worst_sub_cagr": float(np.nanmin(subs)) if np.isfinite(subs).any() else np.nan,
                      **{f"cagr_{n}": wm[n]["cagr"] for n, _, _ in SUB_WINDOWS},
                      **{f"pf_{n}": wm[n]["pf"] for n, _, _ in SUB_WINDOWS},
                      **chk, "open_positions": len(op), "open_unrealized": float(sum(x["pnl"] for x in op)),
                      "final_equity": float(r["equity_s"].iloc[-1]), **r["counts"]})
    for name in refs:
        for wn, _, _ in wins:
            wrows.append({"variant": name, "window": wn, **ref_wm[name][wn]})
    vdf = pd.DataFrame(vrows)
    wdf = pd.DataFrame(wrows)
    # ---- 權益、年度
    eq_df = pd.DataFrame({n: results[n]["equity_s"] for n in results})
    for name, eq in refs.items():
        eq_df[name] = eq if eq is not None else np.nan
    yrows = []
    for col in eq_df.columns:
        if eq_df[col].notna().sum() < 2:
            continue
        for y, rv in yearly_returns(eq_df[col]).items():
            yrows.append({"variant": col, "year": y, "return": rv})
    ydf = pd.DataFrame(yrows, columns=["variant", "year", "return"])
    # ---- 診斷
    elig_n = A["elig"][i0:i1 + 1].sum(axis=1)
    crash_dates = [cal[t] for t in np.flatnonzero(P["crash_day"][i0:i1]) + i0]
    meta = {"universe": int(len(universe)), "price_ok": len(P["codes"]), "d0": cal[i0], "d1": cal[i1],
            "trading_days": int(i1 - i0 + 1), "capital": capital, "feat_sec": feat_sec, "sim_sec": sim_sec,
            "elig_median": float(np.median(elig_n)) if len(elig_n) else 0.0,
            "elig_min": int(elig_n.min()) if len(elig_n) else 0,
            "revenue_codes": P["n_revenue_codes"], "twii_ok": P["twii_ok"],
            "ref_0050_ok": refs[REF_0050] is not None,
            "signal_counts": {k: int(s[i0:i1].sum()) for k, s in sigs.items()},
            "crash_dates": crash_dates,
            "filter_on_pct": float(P["filter_ok"][i0:i1 + 1].mean()),
            "rev_coverage": float(np.isfinite(np.where(A["elig"][i0:i1 + 1], A["yoy3"][i0:i1 + 1], np.nan)).sum()
                                  / max(1, A["elig"][i0:i1 + 1].sum()))}
    return {"meta": meta, "variants": vdf, "windows": wdf, "trades": trades_df, "equity": eq_df,
            "yearly": ydf, "axis": axis_view(vdf), "ref_windows": ref_wm, "variant_defs": variants}


# ===========================================================================
# 輸出
# ===========================================================================
def _dw(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(s))


def _pad(s, width: int, right: bool = False) -> str:
    s = str(s)
    gap = max(0, width - _dw(s))
    return (" " * gap + s) if right else (s + " " * gap)


_pct = ws._pct
_f = ws._f
_money = ws._money


VARIANT_LABELS = {"variant": "變體", "entry": "進場", "exit": "出場", "K": "最多同時持有K", "filter": "大盤濾網",
                  "rank": "排序方式", "start": "起", "end": "迄", "cagr": "年化報酬CAGR", "total": "總報酬",
                  "mdd": "最大回撤(日MTM)", "worst_year": "最差日曆年", "pct_years_pos": "正報酬年比例",
                  "exposure": "在場時間比例", "avg_pos": "平均持股檔數", "avg_invested": "平均持股比重(市值/權益)",
                  "trades": "已平倉筆數", "win_rate": "勝率", "pf": "獲利因子PF", "pnl": "已平倉總損益",
                  "pnl_ex_top5": "扣除前5大獲利後損益", "avg_hold": "平均持有交易日", "boot_pos": "bootstrap正報酬比例",
                  "worst_sub_cagr": "最差分段CAGR",
                  "chk_cagr": "檢查1_全期CAGR>10%", "chk_sub_cagr": "檢查2_四段CAGR>0",
                  "chk_mdd": "檢查3_全期回撤優於0050", "chk_sub_pf": "檢查4_四段PF>1.2", "pass": "PASS",
                  "open_positions": "期末未平倉檔數", "open_unrealized": "期末未平倉未實現損益",
                  "final_equity": "期末權益", "signals": "訊號數", "limit_up_skipped": "漲停買不到跳過",
                  "size_skipped": "資金不足跳過", "slot_skipped": "沒有空位跳過", "no_price_skipped": "進場日無價格跳過",
                  "window": "期間"}
for _n, _, _ in SUB_WINDOWS:
    VARIANT_LABELS[f"cagr_{_n}"] = f"CAGR_{_n}"
    VARIANT_LABELS[f"pf_{_n}"] = f"PF_{_n}"
TRADE_LABELS = {"variant": "變體", "entry_rule": "進場規則", "exit_rule": "出場規則", "K": "K", "code": "代號",
                "name": "名稱", "industry": "產業別", "signal_date": "訊號日", "entry_date": "進場日",
                "entry_price": "進場價(還原)", "exit_date": "出場日", "exit_price": "出場價(還原)", "shares": "股數",
                "buy_fee": "買進手續費", "sell_cost": "賣出手續費+稅", "pnl": "損益(含成本)", "ret": "報酬率(含成本)",
                "exit_reason": "出場原因", "hold_days": "持有交易日", "open_at_end": "期末未平倉",
                "rank_value": "排序值(RS或20日成交金額)"}


def _mark(b) -> str:
    return "✓" if b else "✗"


def build_summary(res: dict, ctx: dict) -> str:
    meta, vdf, wdf = res["meta"], res["variants"], res["windows"]
    rw = res["ref_windows"]
    L = []
    L.append("領頭股突破 — 有資金限制的投資組合回測(不是選股建議)")
    L.append("=" * 78)
    L.append(f"回測區間：{meta['d0'].date()} ~ {meta['d1'].date()}({meta['trading_days']} 個交易日)；"
             f"起始資金 NT${meta['capital']:,.0f}；變體 {len(vdf)} 個")
    L.append("")
    L.append("【警告(請先讀)】")
    L.append("  ⚠️ 存活者偏誤：股票清單只有「目前仍上市櫃」的普通股，2018年以來下市/被併購的股票不在裡面 → 所有績效都偏樂觀。")
    L.append("  ⚠️ 樣本內：2024-最新這一段就是「大漲股共同點研究」(winners_study)研究過的行情，規則是從那段行情得到靈感的，"
             "這一段的好成績不算驗證；請看 2018-2019 / 2020-2021 / 2022-2023 三段。")
    L.append(f"  ⚠️ 多重比較：一次比較 {len(vdf)} 個變體，光靠運氣就會有幾個看起來很好；"
             "請看【設計軸檢視】哪些設計「一致地」比較好，而不是挑單一最好的格子。")
    L.append("  ⚠️ 成本/漲停簡化：手續費0.1425%(最低20元)+證交稅0.3%，開盤價成交不加滑價、停損單以停損價成交；"
             "只有「開=高=低且≥前收×1.095」的一價漲停才當作買不到，盤中漲停鎖住、跌停賣不掉都沒有模擬。")
    L.append("  ⚠️ 零股流動性：小資金常買零股，零股實際成交量小、價差大(盤中零股撮合每分鐘一次)，真實成交會比回測差。")
    L.append("  ⚠️ 價格用還原權值價(股利視為再投入同一檔)，股數是用還原價算的近似值；^TWII 是價格指數不含股利，"
             "0050 用還原價(含股利)；兩個參考都未計交易成本。")
    L.append("  ・分段績效是從同一條連續權益曲線切出來的(以前一段最後收盤當基準)，部位會跨段延續；"
             "已平倉交易依「出場日」歸屬到段落。")
    for w in ctx.get("warnings", []):
        L.append(f"  ⚠️ {w}")
    L.append("")
    L.append("【資料來源狀態】")
    L.append(f"  ・股票清單 {meta['universe']} 檔，有股價 {meta['price_ok']} 檔；每天合格(20日均成交金額≥NT$3,000萬)股票數 "
             f"中位數 {meta['elig_median']:.0f}、最少 {meta['elig_min']}")
    if not ctx.get("revenue_ok", True):
        L.append("  ⚠️⚠️ 月營收(MOPS)完全抓不到：E_B/E_D 不會有任何訊號。請看 debug/revenue_parse_report.csv。")
    L.append(f"  ・月營收：{meta['revenue_codes']} 檔有資料；合格股票-日中有營收3月均YoY的比例 {_pct(meta['rev_coverage'])}"
             "(M月營收從M+1月11日起才算看得到)")
    L.append(f"  ・^TWII：{'OK' if meta['twii_ok'] else '⚠️⚠️ 抓不到(beta/大盤濾網/E_CRASH 全部無法計算，濾網變體不會進場)'}；"
             f"大盤濾網(收盤>MA60)開啟的天數比例 {_pct(meta['filter_on_pct'])}")
    L.append(f"  ・0050：{'OK' if meta['ref_0050_ok'] else '⚠️⚠️ 抓不到(PASS檢查3無法判斷，一律✗)'}")
    sc = meta["signal_counts"]
    L.append("  ・訊號數(股票-日)：" + "、".join(f"{k} {v}" for k, v in sc.items()))
    cd = meta["crash_dates"]
    L.append(f"  ・E_CRASH 觸發日({len(cd)} 次)：" + ("、".join(str(d.date()) for d in cd) if cd else "(無)"))
    L.append("")
    L.append("【規則】")
    for k, v in ENTRY_RULES.items():
        L.append(f"  {k}：{v}")
    for k, v in EXIT_RULES.items():
        L.append(f"  {k}：{v}")
    L.append(f"  兩種出場都另有初始停損 = 進場價 − {INITIAL_STOP_ATR:g}×ATR14(訊號日)，停損單(盤中碰到以停損價成交，開盤跳空以開盤價)；不設停利。")
    L.append(f"  部位：每筆風險 = 權益(前一日收盤市值)×{RISK_PCT:.1%}，股數 = floor(風險/(2×ATR))，"
             "上限為 權益/K 與 可用現金(含最低手續費)；零股可、最少1股。")
    L.append("  排序：空位不夠時 RS 高者優先、同分看20日成交金額；E_CRASH 依20日成交金額；另有 E_D｜X3 改用成交金額排序的對照組。")
    L.append("  大盤濾網：訊號日 ^TWII 收盤 > MA60 才進新部位(出場不受影響)；E_CRASH 不套濾網。")
    L.append("  注意：E_CRASH 進場時(大跌剛反彈)股價常仍在MA60之下，搭配X3時常常隔天開盤就出場，請一起看平均持有日。")
    L.append("")
    hdr = (_pad("變體", 34) + _pad("CAGR", 8, True) + _pad("總報酬", 9, True) + _pad("最大回撤", 9, True)
           + _pad("最差年", 8, True) + _pad("正年", 6, True) + _pad("筆數", 6, True) + _pad("勝率", 6, True)
           + _pad("PF", 6, True) + _pad("持有日", 7, True) + _pad("在場", 6, True) + _pad("持股數", 7, True)
           + _pad("扣前5大", 10, True) + _pad("boot", 6, True))

    def row_line(name, m):
        return (_pad(name[:30], 34) + _pad(_pct(m["cagr"]), 8, True) + _pad(_pct(m["total"], 0), 9, True)
                + _pad(_pct(m["mdd"]), 9, True) + _pad(_pct(m["worst_year"]), 8, True)
                + _pad(_pct(m["pct_years_pos"], 0), 6, True) + _pad(_f(m["trades"], 0), 6, True)
                + _pad(_pct(m["win_rate"], 0), 6, True) + _pad(_f(m["pf"]), 6, True) + _pad(_f(m["avg_hold"], 0), 7, True)
                + _pad(_pct(m["exposure"], 0), 6, True) + _pad(_f(m["avg_pos"], 1), 7, True)
                + _pad(_money(m["pnl_ex_top5"]), 10, True) + _pad(_pct(m["boot_pos"], 0), 6, True))
    for wn, _, _ in [(FULL_WINDOW, None, None)] + SUB_WINDOWS:
        L.append(f"【績效：{wn}】" + ("(2024-最新 = 樣本內)" if wn == "2024-最新" else ""))
        L.append("  " + hdr)
        sub = wdf[wdf["window"] == wn].set_index("variant")
        for name in list(vdf["variant"]) + [REF_TWII, REF_0050]:
            if name in sub.index:
                L.append("  " + row_line(name, sub.loc[name]))
        L.append("")
    L.append("  欄位：CAGR=權益年化複利；最大回撤=每日收盤市值；最差年/正年=日曆年(頭尾可能是部分年度)；筆數/勝率/PF/扣前5大/boot"
             "只算這段期間出場的已平倉交易；在場=有持股的交易日比例；持股數=平均同時持有檔數；"
             f"boot=已平倉損益bootstrap重抽{BOOT_N}次(seed={BOOT_SEED})總和>0的比例。")
    L.append("")
    L.append("【事先登錄的PASS檢查】(1)全期CAGR>10%  (2)四段CAGR都>0  (3)全期最大回撤優於0050  (4)四段PF都>1.2")
    m0050 = rw[REF_0050][FULL_WINDOW]["mdd"]
    L.append(f"  0050 全期最大回撤 {_pct(m0050)}")
    L.append("  注意：PF只算「這段期間出場」的已平倉交易；長抱跨段或期末未平倉的獲利會反映在CAGR，但不會反映在該段PF"
             "(所以X2這類長抱的變體，某一段PF可能很低甚至是空值，但CAGR是正的)。")
    L.append("  " + _pad("變體", 34) + "  (1) (2) (3) (4)  結果")
    for _, r in vdf.iterrows():
        L.append("  " + _pad(r["variant"][:30], 34) + f"  {_mark(r['chk_cagr'])}   {_mark(r['chk_sub_cagr'])}   "
                 f"{_mark(r['chk_mdd'])}   {_mark(r['chk_sub_pf'])}   {'PASS' if r['pass'] else '—'}")
    passed = list(vdf.loc[vdf["pass"], "variant"])
    L.append(f"  通過的變體({len(passed)}/{len(vdf)})：" + ("、".join(passed) if passed else "無"))
    L.append(f"  提醒：{len(vdf)} 個變體裡就算有幾個通過，也可能只是運氣(多重比較)；而且全部都帶有存活者偏誤。")
    L.append("")
    L.append("【設計軸檢視】(36個主變體；每個選項的全期CAGR中位數、最差分段CAGR中位數 → 看哪個設計一致地比較好)")
    L.append("  " + _pad("軸", 6) + _pad("選項", 18) + _pad("變體數", 7, True) + _pad("全期CAGR中位", 14, True)
             + _pad("最差分段CAGR中位", 18, True) + _pad("通過數", 7, True))
    for _, r in res["axis"].iterrows():
        L.append("  " + _pad(r["axis"], 6) + _pad(r["value"], 18) + _pad(r["n"], 7, True)
                 + _pad(_pct(r["median_cagr"]), 14, True) + _pad(_pct(r["median_worst_sub_cagr"]), 18, True)
                 + _pad(r["n_pass"], 7, True))
    L.append("")
    L.append("【排序方式對照】(E_D｜X3｜無濾網：RS排序 vs 20日成交金額排序)")
    for k in (1, 3):
        for nm in (variant_name("E_D", "X3", k, False), variant_name("E_D", "X3", k, False, rank="成交金額")):
            r = vdf[vdf["variant"] == nm]
            if len(r):
                r = r.iloc[0]
                L.append(f"  {_pad(nm, 36)} 全期CAGR {_pct(r['cagr'])}、最差分段 {_pct(r['worst_sub_cagr'])}、"
                         f"最大回撤 {_pct(r['mdd'])}、PF {_f(r['pf'])}")
    L.append("")
    L.append("【期末未平倉部位】(以最後收盤估值，計入權益但不計入勝率/PF；未扣賣出成本)")
    op = res["trades"][res["trades"]["open_at_end"].astype(bool)]
    if op.empty:
        L.append("  (無)")
    else:
        for _, r in op.iterrows():
            L.append(f"  {_pad(r['variant'][:30], 32)} {r['code']} {str(r['name'])[:6]} 進場 {r['entry_date'].date()} "
                     f"@{r['entry_price']:.2f} → {r['exit_price']:.2f}  {r['shares']}股  未實現 {_money(r['pnl'])}")
    L.append("")
    L.append("【跳過的訊號】(全期合計，各變體)")
    for _, r in vdf.iterrows():
        if r["limit_up_skipped"] or r["size_skipped"]:
            L.append(f"  {_pad(r['variant'][:30], 32)} 漲停買不到 {r['limit_up_skipped']}、資金/股數不足 {r['size_skipped']}")
    L.append("  (沒有空位而沒買的訊號很多是正常的，見 leader_variants.csv)")
    L.append("")
    L.append("  ・本報告不做自動挑選，也不構成任何個股推薦。")
    L.append("")
    L.append(f"執行時間：{ctx.get('runtime_text', '')}(特徵 {meta['feat_sec']:.1f} 秒、模擬 {meta['sim_sec']:.1f} 秒)")
    return "\n".join(L) + "\n"


def write_outputs(res: dict, out_dir: str, ctx: dict) -> str:
    os.makedirs(os.path.join(out_dir, "debug"), exist_ok=True)

    def save(df, fn, labels):
        df.rename(columns=labels).to_csv(os.path.join(out_dir, fn), index=False, encoding="utf-8-sig")
    save(res["variants"], "leader_variants.csv", VARIANT_LABELS)
    save(res["windows"], "leader_windows.csv", VARIANT_LABELS)
    save(res["trades"], "leader_trades.csv", TRADE_LABELS)
    eq = res["equity"].copy()
    eq.index.name = "日期"
    eq.to_csv(os.path.join(out_dir, "leader_equity.csv"), encoding="utf-8-sig")
    save(res["yearly"], "leader_yearly.csv", {"variant": "變體", "year": "年", "return": "報酬率"})
    save(res["axis"], os.path.join("debug", "axis_view.csv"),
         {"axis": "軸", "value": "選項", "n": "變體數", "median_cagr": "全期CAGR中位數",
          "median_worst_sub_cagr": "最差分段CAGR中位數", "n_pass": "通過數"})
    pd.DataFrame({"E_CRASH觸發日": [d.date() for d in res["meta"]["crash_dates"]]}).to_csv(
        os.path.join(out_dir, "debug", "crash_dates.csv"), index=False, encoding="utf-8-sig")
    text = build_summary(res, ctx)
    with open(os.path.join(out_dir, "summary_leader.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="領頭股突破：有資金限制的投資組合回測")
    p.add_argument("--output-dir", default="results_leader")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default="", help="留空=最新資料(今天，台北時間)")
    p.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--refresh", action="store_true", help="忽略快取重新下載")
    p.add_argument("--max-stocks", type=int, default=0, help="只取前N檔(測試用，0=全部)")
    return p.parse_args(argv)


def _fail(out_dir: str, msg: str) -> int:
    ws._write_text(os.path.join(out_dir, "summary_leader.txt"), msg)
    print(msg, flush=True)
    return 1


def main(argv=None) -> int:
    t0 = time.time()
    args = parse_args(argv)
    start_ts, end_ts = ws.resolve_window(args.start or DEFAULT_START, args.end)
    out_dir = args.output_dir
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    today_str = ws._taipei_today().isoformat()
    dl_start_ts = (start_ts - pd.DateOffset(months=WARMUP_MONTHS)).normalize()
    dl_start = dl_start_ts.date().isoformat()
    dl_end = (end_ts + pd.Timedelta(days=1)).date().isoformat()
    print(f"回測區間 {start_ts.date()} ~ {end_ts.date()}；股價/月營收從 {dl_start} 起抓", flush=True)

    universe, warnings = ws.load_universe(args.cache_dir, args.refresh, debug_dir, today_str)
    if universe.empty:
        return _fail(out_dir, "⚠️ 上市/上櫃股票清單都抓不到(ISIN頁面)，無法回測(fail closed)。請看 debug/universe_report.txt。\n")
    if args.max_stocks and args.max_stocks > 0:
        universe = universe.head(args.max_stocks).reset_index(drop=True)
    print(f"股票清單 {len(universe)} 檔", flush=True)
    prices, fail_df = ws.load_prices(universe, dl_start, dl_end, args.cache_dir, args.refresh, debug_dir)
    if len(prices) == 0:
        return _fail(out_dir, "⚠️ 股價全部下載失敗(yfinance)，無法回測(fail closed)。請看 debug/price_failures.csv。\n")
    fail_ratio = len(fail_df) / max(1, len(universe))
    if fail_ratio > 0.05:
        warnings.append(f"股價下載失敗 {len(fail_df)} 檔({fail_ratio:.0%})，清單見 debug/price_failures.csv")
    ref_uni = pd.DataFrame({"code": [REF_ETF], "market": ["上市"]})
    ref_prices, _ = ws.load_prices(ref_uni, dl_start, dl_end, args.cache_dir, args.refresh,
                                   os.path.join(debug_dir, "ref_0050"))
    taiex = ws.load_taiex(dl_start, dl_end, args.cache_dir, args.refresh)
    if taiex is None or len(taiex) == 0:
        warnings.append("加權指數(^TWII)下載失敗：beta、大盤濾網、E_CRASH 都無法計算")
    revenue, rev_ok = ws.load_revenue(dl_start_ts.replace(day=1), end_ts, args.cache_dir, args.refresh, debug_dir)
    if not rev_ok:
        print("⚠️ 月營收完全抓不到，E_B/E_D 不會有訊號", flush=True)
    t_dl = time.time() - t0
    t1 = time.time()
    res = run_backtest(universe, prices, revenue, taiex, ref_prices.get(REF_ETF), start_ts, end_ts, args.capital)
    compute = time.time() - t1
    elapsed = time.time() - t0
    ctx = {"revenue_ok": rev_ok, "warnings": warnings,
           "runtime_text": f"總計 {elapsed / 60:.1f} 分鐘({elapsed:.0f} 秒；下載/讀快取 {t_dl:.0f} 秒、計算 {compute:.0f} 秒)"}
    text = write_outputs(res, out_dir, ctx)
    print(text, flush=True)
    print(f"完成，總執行時間 {elapsed:.1f} 秒(計算 {compute:.1f} 秒)，結果在 {out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
