"""
tx_daily_engine.py
====================
台指期(預設小台MTX)「日線」單一部位回測引擎：一次最多1口、可多可空、不反手。

輸入日線DataFrame(由tx_daily_data.py產生)：
  Open/High/Low/Close  —— 訊號跟點數損益用的價格(期貨模式=Panama調整後；加權指數模式=指數本身)
  Offset               —— 調整後 − 未調整(期交稅用未調整價格算：未調整 = 調整後 − Offset)
  RollDay              —— 這天開盤時換到下一個合約(持有部位跨過這天就多收一次來回手續費+稅)

訊號(第t天收盤後計算、只用<=t的資料，第t+1天開盤進場)：
  SIG_DONCHIAN     收盤 > 前20天(不含今天)最高價 → 多；收盤 < 前20天最低價 → 空
  SIG_MA_CROSS     MA20由下往上穿越MA60 → 多；由上往下 → 空
  SIG_SQZ_KDJ      squeeze_kdj_signal.compute_squeeze_kdj_features()的EntryFlag(跟個股策略同規則)，只做多
  SIG_SQZ_RELEASE  昨天擠壓(BB完全在KC內)、今天解除 → 方向 = 動能符號
                   動能(TTM動能的簡化代理，**沒有**再做20日線性回歸平滑)：
                     mom_t = Close_t − ( (HH20_t + LL20_t)/2 + SMA20_t ) / 2
                   HH20/LL20 = 含今天的20日最高價/最低價；mom>0做多、<0做空、=0不動作。
「動能分數」門檻(選配)：第t天 ADX(14) >= 20 且 均線排列跟方向一致
  (多：MA5>MA20>MA60；空：MA5<MA20<MA60)。門檻只過濾「進場」；反向訊號出場用未過濾
  的原始訊號(門檻是進場品質過濾，出場規則所有變體一致)。

出場(所有變體相同)：
  初始停損 = 進場價 ∓ 2.0×ATR(14)(ATR取訊號日t的值)
  移動停損 = 進場後最高收盤(多)/最低收盤(空) ∓ 3.0×ATR(14)(每天收盤後用當天ATR更新)，只會收緊
  停損觸價：當天開盤已越過停損(跳空) → 以開盤價成交；否則盤中觸及 → 以停損價成交；兩者再各扣1點滑價
  反向訊號：持倉中出現反向原始訊號 → 隔天開盤市價出場(不反手，之後的新訊號才可能再進場)
  最長持有：持有滿60個交易日(含進場當天)的那天收盤後 → 隔天開盤出場(安全閥，不是主要出場邏輯)
  期末：回測期間最後一天仍有部位 → 以收盤價(扣1點滑價)平倉，exit_reason="期末平倉"

成本(每口每邊)：手續費NT$50(CLI可改) + 期貨交易稅 0.002% × 未調整價格 × 每點金額；
  市價類成交(開盤進出場/停損)一律1點滑價(對自己不利)；換月：持倉跨過RollDay那天開盤 →
  加收一次來回(2×手續費 + 2×稅，稅用換月日開盤未調整價格算)，不另計滑價。

每點金額：小台MTX NT$50(預設)、大台TX NT$200。
保證金：MARGIN_ESTIMATE是估計值(期交所會隨指數水位/波動調整，以公告為準)，只用來在報告
  裡跟最大回撤/起始資金對照，不會擋交易。

-----------------------------------------------------------------------------------------
網格模式(--grid，compare_tx_daily.py / tx_daily_grid.py)用的擴充(上面8變體的函式完全沒動)：

單一訊號登記表 GRID_SIGNAL_NAMES(12個，順序固定 = 平手時的最後判準)。全部在第t天收盤後
只用<=t的資料計算，+1多/−1空/0不動作，第t+1天開盤進場。「穿越」一律定義為
  上穿：a_t > b_t 且 a_(t−1) <= b_(t−1)；下穿：a_t < b_t 且 a_(t−1) >= b_(t−1)(暖身期NaN → 0)。
   1. MA_5_20     MA5上穿MA20 → 多；下穿 → 空
   2. MA_20_60    = 既有MA_CROSS(MA20上穿/下穿MA60)
   3. MACD        MACD線(EMA12−EMA26)上穿/下穿訊號線(MACD線的EMA9)
                  (重用momentum_breakout_engine.compute_macd；前35根K棒暖身不出訊號)
   4. MACD_ZERO   MACD線(EMA12−EMA26)上穿/下穿0軸(前35根暖身不出訊號)
                  ⚠️ 原本登記的是「MACD柱狀圖穿越0」，但柱狀圖 = MACD線 − 訊號線，「柱狀圖穿越0」
                  跟第3個「MACD線穿越訊號線」在數學上是**同一件事**(每天訊號完全相同)，留著只會
                  多出一整組重複變體、灌水家族統計；改成MACD另一個常見訊號「MACD線穿越0軸」。
   5. BODY        強勢K棒：實體/全距 >= 0.6 且 全距 >= 1.0×ATR(14)(ATR含當天)；
                  紅K(收>開)且收盤在當天全距最上面25%(收 >= 高 − 0.25×全距) → 多；
                  黑K(收<開)且收盤在最下面25%(收 <= 低 + 0.25×全距) → 空(每天獨立判斷，不是穿越)
   6. RSI50       RSI(14)上穿/下穿50(重用mean_reversion_engine.compute_rsi，簡單平均版RSI；
                  前15根暖身不出訊號)
   7. DONCHIAN20  = 既有DONCHIAN(收盤突破前20天高/低)
   8. DONCHIAN55  同上，55天
   9. BB_BREAK    收盤上穿布林上軌(20,2) → 多；收盤下穿布林下軌 → 空
                  (上穿：收_t > 上軌_t 且 收_(t−1) <= 上軌_(t−1)；下軌鏡射)
  10. KDJ_CROSS   KDJ(9,3,3)：RSV9 = (收−9日最低)/(9日最高−9日最低)×100；
                  K_t = 2/3·K_(t−1) + 1/3·RSV_t(K_0前一值=50，重用squeeze_kdj_signal.compute_kdj_k)；
                  D_t = 2/3·D_(t−1) + 1/3·K_t(D_0前一值=50)。
                  K上穿D 且 當天K<30 → 多；K下穿D 且 當天K>70 → 空(前12根暖身不出訊號)
  11. SQZ_RELEASE 既有(擠壓解除，方向=動能符號)
  12. SQZ_KDJ     既有(只做多)
混搭(變種)訊號：12個任取2個的66組(A&B，順序依登記表)。第t天方向d成立 ⇔
  其中一個在第t天發出d，且另一個在[t−2, t]任一天也發出d(三天視窗內兩個都同方向出現)。
  同一天多空同時成立(極少見) → 0。SQZ_KDJ本身只會發+1，所以含它的組合只會做多。
出場網格(12種)：初始停損 ∈ {1.0, 1.5, 2.0}×ATR(14) × 出場方式 ∈
  {固定停利2.0 / 3.0 / 4.0×ATR(14)(不移動停損)，移動停損3.0×ATR(14)(不設停利)}。
  停利是限價單：盤中觸及 → 以停利價成交、不扣滑價；開盤就跳空越過停利 → 以開盤價成交(不扣滑價)。
  同一天盤中同時碰到停損與停利(日線看不出先後) → 保守當作先停損。
  開盤跳空越過停損 → 開盤價(扣滑價)，優先於停利判斷。
  ATR一律取訊號日t的值；反向訊號 = 同一條進場規則的「未過濾」原始訊號(混搭就是混搭原始訊號)；
  最長持有/期末/成本/換月與上面8變體完全相同。
  「停損2.0 + 移動停損3.0」這一格就是上面8變體的出場規則(測試有驗證逐筆相同)。
"""
import itertools

import numpy as np
import pandas as pd

from mean_reversion_engine import compute_bollinger, compute_atr_correct, compute_rsi
from momentum_breakout_engine import compute_adx, compute_macd
from squeeze_kdj_signal import (compute_squeeze_kdj_features, compute_keltner_channel, compute_squeeze_flag,
                                compute_kdj_k)

CONTRACT_MULTIPLIER = {"mini": 50, "big": 200}
MARGIN_ESTIMATE = {"mini": 85_000, "big": 340_000}  # 估計值，見模組docstring
DEFAULT_COMMISSION_PER_SIDE = 50.0
FUTURES_TAX_RATE = 0.00002   # 0.002%
SLIPPAGE_POINTS = 1.0
STARTING_CAPITAL = 200_000

ATR_PERIOD = 14
INITIAL_STOP_ATR = 2.0
TRAILING_STOP_ATR = 3.0
MAX_HOLD_DAYS = 60
ADX_PERIOD = 14
ADX_GATE = 20.0
DONCHIAN_WINDOW = 20

SIGNAL_NAMES = ["DONCHIAN", "MA_CROSS", "SQZ_KDJ", "SQZ_RELEASE"]


def variant_list():
    """8個預先登記的變體，順序固定(選擇時平手的最後判準就是這個順序)。"""
    out = []
    for sig in SIGNAL_NAMES:
        out.append({"name": f"{sig}", "signal": sig, "gate": False})
        out.append({"name": f"{sig}+動能門檻", "signal": sig, "gate": True})
    return out


def compute_ttm_momentum(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """mom_t = Close_t − ((HH_t + LL_t)/2 + SMA_t)/2(見模組docstring)。"""
    hh = df["High"].rolling(period).max()
    ll = df["Low"].rolling(period).min()
    sma = df["Close"].rolling(period).mean()
    return df["Close"] - ((hh + ll) / 2.0 + sma) / 2.0


def compute_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一次算好所有指標/原始訊號(第t列只用到<=t的資料)。原始訊號欄位：
    sig_<NAME> ∈ {+1, −1, 0}；gate_long/gate_short ∈ bool。"""
    close, high, low = df["Close"], df["High"], df["Low"]
    ind = pd.DataFrame(index=df.index)
    ind["ATR"] = compute_atr_correct(df, period=ATR_PERIOD)
    ind["ADX"] = compute_adx(df, period=ADX_PERIOD)
    for p in (5, 20, 60):
        ind[f"MA{p}"] = close.rolling(p).mean()

    prior_hh = high.rolling(DONCHIAN_WINDOW).max().shift(1)
    prior_ll = low.rolling(DONCHIAN_WINDOW).min().shift(1)
    don = np.where(close > prior_hh, 1, np.where(close < prior_ll, -1, 0))
    ind["sig_DONCHIAN"] = don

    ma20, ma60 = ind["MA20"], ind["MA60"]
    up = (ma20 > ma60) & (ma20.shift(1) <= ma60.shift(1))
    dn = (ma20 < ma60) & (ma20.shift(1) >= ma60.shift(1))
    ind["sig_MA_CROSS"] = np.where(up, 1, np.where(dn, -1, 0))

    feats = compute_squeeze_kdj_features(df)
    ind["sig_SQZ_KDJ"] = np.where(feats["EntryFlag"].to_numpy(dtype=bool), 1, 0)

    _, bb_u, bb_l = compute_bollinger(close, period=20, num_std=2.0)
    _, kc_u, kc_l = compute_keltner_channel(df, period=20, atr_mult=1.5)
    sq = compute_squeeze_flag(bb_u, bb_l, kc_u, kc_l).astype(bool)
    mom = compute_ttm_momentum(df, 20)
    release = sq.shift(1, fill_value=False).astype(bool) & ~sq
    ind["Squeeze"] = sq
    ind["Momentum"] = mom
    ind["sig_SQZ_RELEASE"] = np.where(release & (mom > 0), 1, np.where(release & (mom < 0), -1, 0))

    adx_ok = ind["ADX"] >= ADX_GATE
    ind["gate_long"] = (adx_ok & (ind["MA5"] > ind["MA20"]) & (ind["MA20"] > ind["MA60"])).fillna(False).astype(bool)
    ind["gate_short"] = (adx_ok & (ind["MA5"] < ind["MA20"]) & (ind["MA20"] < ind["MA60"])).fillna(False).astype(bool)
    for s in SIGNAL_NAMES:
        ind[f"sig_{s}"] = ind[f"sig_{s}"].astype(int)
    return ind


def entry_signal_series(ind: pd.DataFrame, signal: str, gate: bool) -> np.ndarray:
    raw = ind[f"sig_{signal}"].to_numpy(dtype=int)
    if not gate:
        return raw
    gl = ind["gate_long"].to_numpy(dtype=bool)
    gs = ind["gate_short"].to_numpy(dtype=bool)
    return np.where((raw == 1) & gl, 1, np.where((raw == -1) & gs, -1, 0))


def side_tax(raw_price: float, multiplier: int, tax_rate: float = FUTURES_TAX_RATE) -> float:
    return abs(float(raw_price)) * multiplier * tax_rate


def run_tx_daily_backtest(df: pd.DataFrame, ind: pd.DataFrame, signal: str, gate: bool,
                          start=None, end=None, contract: str = "mini",
                          commission_per_side: float = DEFAULT_COMMISSION_PER_SIDE,
                          slippage_points: float = SLIPPAGE_POINTS,
                          tax_rate: float = FUTURES_TAX_RATE,
                          initial_stop_atr: float = INITIAL_STOP_ATR,
                          trailing_stop_atr: float = TRAILING_STOP_ATR,
                          max_hold_days: int = MAX_HOLD_DAYS,
                          starting_capital: float = STARTING_CAPITAL,
                          variant_name: str = None) -> list:
    """在[start, end]區間內獨立跑一次(期初空手、期末強制平倉)。df/ind可以含區間外的
    歷史(只用來暖身指標)。回傳trades(list of dict，summarize_mr/bootstrap相容)。"""
    mult = CONTRACT_MULTIPLIER[contract]
    dates = df.index
    lo = 0 if start is None else int(dates.searchsorted(pd.Timestamp(start), side="left"))
    hi = len(dates) - 1 if end is None else int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    if hi < lo:
        return []
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    off = df["Offset"].to_numpy(float) if "Offset" in df.columns else np.zeros(len(df))
    roll = df["RollDay"].to_numpy(bool) if "RollDay" in df.columns else np.zeros(len(df), bool)
    atr = ind["ATR"].to_numpy(float)
    entry_sig = entry_signal_series(ind, signal, gate)
    raw_sig = ind[f"sig_{signal}"].to_numpy(int)
    name = variant_name or signal

    trades = []
    pos = None
    pending_entry = 0           # +1/−1：明天開盤進場
    pending_exit = None         # 出場原因：明天開盤出場

    def close_trade(i, fill, reason, hold):
        d = pos["dir"]
        pts = (fill - pos["entry_price"]) * d
        raw_exit = fill - off[i]
        costs = (2 * commission_per_side + side_tax(pos["raw_entry"], mult, tax_rate)
                 + side_tax(raw_exit, mult, tax_rate) + pos["roll_cost"])
        pnl = pts * mult - costs
        trades.append({
            "variant": name, "side": "long" if d == 1 else "short",
            "signal_date": pos["signal_date"], "entry_date": pos["entry_date"],
            "exit_date": dates[i], "entry_price": pos["entry_price"], "exit_price": float(fill),
            "raw_entry_price": pos["raw_entry"], "raw_exit_price": float(raw_exit),
            "initial_stop": pos["initial_stop"], "points": float(pts),
            "n_rolls": pos["n_rolls"], "roll_cost_ntd": pos["roll_cost"],
            "cost_ntd": float(costs), "pnl_ntd": float(pnl),
            "return_pct": float(pnl / starting_capital),
            "hold_days": int(hold), "exit_reason": reason,
        })

    for i in range(lo, hi + 1):
        # ---- 開盤 ----
        if pos is not None and pending_exit is not None:
            fill = o[i] - slippage_points * pos["dir"]
            close_trade(i, fill, pending_exit, i - pos["entry_idx"])
            pos, pending_exit = None, None
        elif pos is not None and roll[i]:
            pos["n_rolls"] += 1
            pos["roll_cost"] += 2 * commission_per_side + 2 * side_tax(o[i] - off[i], mult, tax_rate)
        if pos is None and pending_entry != 0:
            d = pending_entry
            sig_i = i - 1
            a = atr[sig_i]
            if np.isfinite(a) and a > 0:
                fill = o[i] + slippage_points * d
                stop = fill - d * initial_stop_atr * a
                pos = {"dir": d, "entry_idx": i, "entry_date": dates[i], "signal_date": dates[sig_i],
                       "entry_price": float(fill), "raw_entry": float(fill - off[i]),
                       "stop": float(stop), "initial_stop": float(stop), "best_close": None,
                       "n_rolls": 0, "roll_cost": 0.0}
        pending_entry = 0

        # ---- 盤中停損 ----
        if pos is not None:
            d, stop = pos["dir"], pos["stop"]
            hit = None
            if d == 1:
                if o[i] <= stop:
                    hit = o[i]
                elif l[i] <= stop:
                    hit = stop
            else:
                if o[i] >= stop:
                    hit = o[i]
                elif h[i] >= stop:
                    hit = stop
            if hit is not None:
                reason = "停損" if stop == pos["initial_stop"] else "移動停損"
                close_trade(i, hit - slippage_points * d, reason, i - pos["entry_idx"] + 1)
                pos = None

        # ---- 收盤 ----
        if i == hi:
            if pos is not None:
                close_trade(i, c[i] - slippage_points * pos["dir"], "期末平倉", i - pos["entry_idx"] + 1)
                pos = None
            break
        if pos is not None:
            d = pos["dir"]
            bc = c[i] if pos["best_close"] is None else (max(pos["best_close"], c[i]) if d == 1
                                                         else min(pos["best_close"], c[i]))
            pos["best_close"] = bc
            if np.isfinite(atr[i]):
                trail = bc - d * trailing_stop_atr * atr[i]
                pos["stop"] = max(pos["stop"], trail) if d == 1 else min(pos["stop"], trail)
            if raw_sig[i] == -d:
                pending_exit = "反向訊號"
            elif (i - pos["entry_idx"] + 1) >= max_hold_days:
                pending_exit = "最長持有"
        else:
            if entry_sig[i] != 0:
                pending_entry = int(entry_sig[i])
    return trades


def run_buy_and_hold(df: pd.DataFrame, start=None, end=None, contract: str = "mini",
                     commission_per_side: float = DEFAULT_COMMISSION_PER_SIDE,
                     slippage_points: float = SLIPPAGE_POINTS, tax_rate: float = FUTURES_TAX_RATE,
                     starting_capital: float = STARTING_CAPITAL) -> list:
    """參考用：期間第一天開盤做多1口、一路換月持有到最後一天收盤。回傳1筆trade的list。"""
    mult = CONTRACT_MULTIPLIER[contract]
    dates = df.index
    lo = 0 if start is None else int(dates.searchsorted(pd.Timestamp(start), side="left"))
    hi = len(dates) - 1 if end is None else int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    if hi < lo:
        return []
    off = df["Offset"].to_numpy(float) if "Offset" in df.columns else np.zeros(len(df))
    roll = df["RollDay"].to_numpy(bool) if "RollDay" in df.columns else np.zeros(len(df), bool)
    o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    entry = o[lo] + slippage_points
    exit_ = c[hi] - slippage_points
    n_rolls = int(roll[lo + 1:hi + 1].sum())
    roll_cost = sum(2 * commission_per_side + 2 * side_tax(o[j] - off[j], mult, tax_rate)
                    for j in range(lo + 1, hi + 1) if roll[j])
    costs = (2 * commission_per_side + side_tax(entry - off[lo], mult, tax_rate)
             + side_tax(exit_ - off[hi], mult, tax_rate) + roll_cost)
    pts = exit_ - entry
    pnl = pts * mult - costs
    return [{"variant": "買進持有(參考)", "side": "long", "signal_date": dates[lo], "entry_date": dates[lo],
             "exit_date": dates[hi], "entry_price": float(entry), "exit_price": float(exit_),
             "raw_entry_price": float(entry - off[lo]), "raw_exit_price": float(exit_ - off[hi]),
             "initial_stop": float("nan"), "points": float(pts), "n_rolls": n_rolls,
             "roll_cost_ntd": float(roll_cost), "cost_ntd": float(costs), "pnl_ntd": float(pnl),
             "return_pct": float(pnl / starting_capital), "hold_days": int(hi - lo + 1),
             "exit_reason": "期末平倉"}]


# =========================================================================================
# 網格模式(--grid)：12個單一訊號 + 66組混搭 × 動能門檻有/無 × 12種出場 = 1872變體
# (定義見模組docstring下半段；上面8變體的函式完全沒動)
# =========================================================================================
GRID_SIGNAL_NAMES = ["MA_5_20", "MA_20_60", "MACD", "MACD_ZERO", "BODY", "RSI50",
                     "DONCHIAN20", "DONCHIAN55", "BB_BREAK", "KDJ_CROSS", "SQZ_RELEASE", "SQZ_KDJ"]
LONG_ONLY_SIGNALS = ("SQZ_KDJ",)
PAIR_WINDOW_DAYS = 3            # 混搭：另一個訊號要在[t−2, t]內出現過
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MACD_WARMUP = MACD_SLOW + MACD_SIGNAL      # 前35根不出訊號
RSI_PERIOD = 14
RSI_WARMUP = RSI_PERIOD + 1
KDJ_GRID_PERIOD = 9
KDJ_WARMUP = KDJ_GRID_PERIOD + 3
KDJ_LONG_MAX_K = 30.0
KDJ_SHORT_MIN_K = 70.0
BODY_MIN_RATIO = 0.6
BODY_MIN_RANGE_ATR = 1.0
BODY_CLOSE_ZONE = 0.25
DONCHIAN_LONG_WINDOW = 55

GRID_STOP_MULTS = (1.0, 1.5, 2.0)
GRID_EXIT_OPTIONS = (("target", 2.0), ("target", 3.0), ("target", 4.0), ("trail", 3.0))


def cross_signal(a: pd.Series, b) -> np.ndarray:
    """a上穿b → +1；a下穿b → −1；其他(含NaN暖身) → 0。b可以是Series或常數。"""
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=a.index)
    a_prev, b_prev = a.shift(1), b.shift(1)
    up = (a > b) & (a_prev <= b_prev)
    dn = (a < b) & (a_prev >= b_prev)
    return np.where(up.to_numpy(bool), 1, np.where(dn.to_numpy(bool), -1, 0)).astype(int)


def _mask_warmup(sig: np.ndarray, n: int) -> np.ndarray:
    out = np.asarray(sig, dtype=int).copy()
    out[:n] = 0
    return out


def compute_kdj_d(k: pd.Series) -> pd.Series:
    """D_t = 2/3·D_(t−1) + 1/3·K_t，D的前一值起始=50(跟compute_kdj_k的K起始方式一致)。"""
    kv = k.to_numpy(float)
    d = np.empty(len(kv))
    prev = 50.0
    for i, x in enumerate(kv):
        prev = (2.0 / 3.0) * prev + (1.0 / 3.0) * x
        d[i] = prev
    return pd.Series(d, index=k.index)


def body_signal(df: pd.DataFrame, atr: pd.Series) -> np.ndarray:
    """BODY強勢K棒訊號(定義見模組docstring)。"""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    rng = h - l
    ratio = (c - o).abs() / rng.where(rng > 0)
    big = (ratio >= BODY_MIN_RATIO) & (rng >= BODY_MIN_RANGE_ATR * atr)
    bull = big & (c > o) & (c >= h - BODY_CLOSE_ZONE * rng)
    bear = big & (c < o) & (c <= l + BODY_CLOSE_ZONE * rng)
    return np.where(bull.to_numpy(bool), 1, np.where(bear.to_numpy(bool), -1, 0)).astype(int)


def donchian_signal(df: pd.DataFrame, window: int) -> np.ndarray:
    prior_hh = df["High"].rolling(window).max().shift(1)
    prior_ll = df["Low"].rolling(window).min().shift(1)
    c = df["Close"]
    return np.where((c > prior_hh).to_numpy(bool), 1, np.where((c < prior_ll).to_numpy(bool), -1, 0)).astype(int)


def _recent(flag: np.ndarray, window: int) -> np.ndarray:
    """flag在[t−window+1, t]內任一天為True。"""
    out = flag.copy()
    for k in range(1, window):
        out[k:] |= flag[:-k]
    return out


def compose_pair(a, b, window: int = PAIR_WINDOW_DAYS) -> np.ndarray:
    """混搭訊號(見模組docstring)：第t天方向d ⇔ 一個在t發d、另一個在[t−window+1, t]發過d。"""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    fire = {}
    for d in (1, -1):
        fa, fb = a == d, b == d
        fire[d] = (fa & _recent(fb, window)) | (fb & _recent(fa, window))
    out = np.where(fire[1] & ~fire[-1], 1, np.where(fire[-1] & ~fire[1], -1, 0))
    return out.astype(int)


def grid_rule_list() -> list:
    """78條進場規則(順序固定)：12個單一訊號，接著66組混搭(itertools.combinations順序)。
    每條 = {"rule": 名稱, "components": (A,) 或 (A, B)}。"""
    rules = [{"rule": s, "components": (s,)} for s in GRID_SIGNAL_NAMES]
    for a, b in itertools.combinations(GRID_SIGNAL_NAMES, 2):
        rules.append({"rule": f"{a}&{b}", "components": (a, b)})
    return rules


def grid_exit_configs() -> list:
    """12種出場設定(停損在外圈、出場方式在內圈)。每個 = {"exit_id", "stop_atr", "exit_kind",
    "exit_mult", "stop_idx", "exit_idx", "label"}。"""
    out = []
    for si, s in enumerate(GRID_STOP_MULTS):
        for ei, (kind, m) in enumerate(GRID_EXIT_OPTIONS):
            tag = f"停利{m:.1f}" if kind == "target" else f"移動{m:.1f}"
            out.append({"exit_id": f"S{s:.1f}_{'T' if kind == 'target' else 'TR'}{m:.1f}",
                        "stop_atr": s, "exit_kind": kind, "exit_mult": m,
                        "stop_idx": si, "exit_idx": ei, "label": f"停損{s:.1f}/{tag}"})
    return out


def grid_variant_list() -> list:
    """(12 + 66) × {無門檻, 動能門檻} × 12出場 = 1872個變體，順序固定(= 平手的最後判準)。"""
    out = []
    for r in grid_rule_list():
        for gate in (False, True):
            for ex in grid_exit_configs():
                name = f"{r['rule']}{'+動能門檻' if gate else ''}｜{ex['label']}"
                v = {"name": name, "rule": r["rule"], "components": r["components"], "gate": gate}
                v.update(ex)
                out.append(v)
    return out


def compute_grid_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """compute_daily_indicators()全部欄位 + 12個網格單一訊號 + 66組混搭訊號(sig_<規則名>)。
    第t列只用到<=t的資料。"""
    ind = compute_daily_indicators(df)
    close = df["Close"]
    ind["sig_MA_5_20"] = cross_signal(ind["MA5"], ind["MA20"])
    ind["sig_MA_20_60"] = ind["sig_MA_CROSS"].to_numpy(int)

    macd, macd_sig, macd_hist = compute_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    ind["MACD"], ind["MACD_Signal"], ind["MACD_Hist"] = macd, macd_sig, macd_hist
    ind["sig_MACD"] = _mask_warmup(cross_signal(macd, macd_sig), MACD_WARMUP)
    ind["sig_MACD_ZERO"] = _mask_warmup(cross_signal(macd, 0.0), MACD_WARMUP)

    ind["sig_BODY"] = body_signal(df, ind["ATR"])

    rsi = compute_rsi(close, period=RSI_PERIOD)
    ind["RSI14"] = rsi
    ind["sig_RSI50"] = _mask_warmup(cross_signal(rsi, 50.0), RSI_WARMUP)

    ind["sig_DONCHIAN20"] = ind["sig_DONCHIAN"].to_numpy(int)
    ind["sig_DONCHIAN55"] = donchian_signal(df, DONCHIAN_LONG_WINDOW)

    _, bb_u, bb_l = compute_bollinger(close, period=20, num_std=2.0)
    ind["BB_Upper"], ind["BB_Lower"] = bb_u, bb_l
    up = cross_signal(close, bb_u) == 1
    dn = cross_signal(close, bb_l) == -1
    ind["sig_BB_BREAK"] = np.where(up, 1, np.where(dn, -1, 0)).astype(int)

    k = compute_kdj_k(df, period=KDJ_GRID_PERIOD)
    d = compute_kdj_d(k)
    ind["KDJ_K9"], ind["KDJ_D9"] = k, d
    kx = cross_signal(k, d)
    kv = k.to_numpy(float)
    kdj = np.where((kx == 1) & (kv < KDJ_LONG_MAX_K), 1, np.where((kx == -1) & (kv > KDJ_SHORT_MIN_K), -1, 0))
    ind["sig_KDJ_CROSS"] = _mask_warmup(kdj, KDJ_WARMUP)
    # sig_SQZ_RELEASE / sig_SQZ_KDJ 已經由compute_daily_indicators算好(SQZ_KDJ只有+1)

    pair_cols = {}
    for r in grid_rule_list():
        if len(r["components"]) == 2:
            a, b = r["components"]
            pair_cols[f"sig_{r['rule']}"] = compose_pair(ind[f"sig_{a}"].to_numpy(int),
                                                         ind[f"sig_{b}"].to_numpy(int))
    ind = pd.concat([ind, pd.DataFrame(pair_cols, index=ind.index)], axis=1)
    for s in GRID_SIGNAL_NAMES:
        ind[f"sig_{s}"] = ind[f"sig_{s}"].astype(int)
    return ind


def prepare_grid_arrays(df: pd.DataFrame, ind: pd.DataFrame) -> dict:
    """把價格/ATR/門檻轉成Python list(逐日迴圈用list索引比numpy純量快很多)，每個frame只做一次。"""
    n = len(df)
    off = df["Offset"].to_numpy(float) if "Offset" in df.columns else np.zeros(n)
    roll = df["RollDay"].to_numpy(bool) if "RollDay" in df.columns else np.zeros(n, bool)
    return {
        "dates": df.index, "n": n,
        "o": df["Open"].to_numpy(float).tolist(), "h": df["High"].to_numpy(float).tolist(),
        "l": df["Low"].to_numpy(float).tolist(), "c": df["Close"].to_numpy(float).tolist(),
        "off": off.tolist(), "roll": roll.tolist(), "atr": ind["ATR"].to_numpy(float).tolist(),
        "gate_long": ind["gate_long"].to_numpy(bool), "gate_short": ind["gate_short"].to_numpy(bool),
        "ind": ind,
    }


def grid_entry_signals(arrs: dict, rule: str, gate: bool):
    """回傳(entry_sig ndarray, raw_sig ndarray)：raw = 規則的原始訊號(反向出場用)，
    entry = 有門檻時只留門檻同向放行的進場訊號。"""
    raw = arrs["ind"][f"sig_{rule}"].to_numpy(int)
    if not gate:
        return raw, raw
    gl, gs = arrs["gate_long"], arrs["gate_short"]
    return np.where((raw == 1) & gl, 1, np.where((raw == -1) & gs, -1, 0)).astype(int), raw


def run_tx_daily_grid_backtest(arrs: dict, entry_sig: np.ndarray, raw_sig: np.ndarray,
                               initial_stop_atr: float, exit_kind: str, exit_mult: float,
                               start=None, end=None, contract: str = "mini",
                               commission_per_side: float = DEFAULT_COMMISSION_PER_SIDE,
                               slippage_points: float = SLIPPAGE_POINTS,
                               tax_rate: float = FUTURES_TAX_RATE,
                               max_hold_days: int = MAX_HOLD_DAYS,
                               starting_capital: float = STARTING_CAPITAL,
                               variant_name: str = "") -> list:
    """網格版單一部位回測。exit_kind="trail"：移動停損exit_mult×ATR、無停利(跟
    run_tx_daily_backtest同邏輯)；exit_kind="target"：固定停利exit_mult×ATR、停損不移動。
    其餘(進出場時點、跳空、滑價、成本、換月、最長持有、期末)與run_tx_daily_backtest相同。
    空手又沒有待進場時直接跳到下一個進場訊號日(結果不變，只是比較快)。"""
    if exit_kind not in ("trail", "target"):
        raise ValueError(exit_kind)
    mult = CONTRACT_MULTIPLIER[contract]
    dates = arrs["dates"]
    lo = 0 if start is None else int(dates.searchsorted(pd.Timestamp(start), side="left"))
    hi = len(dates) - 1 if end is None else int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    if hi < lo:
        return []
    o, h, l, c = arrs["o"], arrs["h"], arrs["l"], arrs["c"]
    off, roll, atr = arrs["off"], arrs["roll"], arrs["atr"]
    es = entry_sig.tolist() if isinstance(entry_sig, np.ndarray) else list(entry_sig)
    rs = raw_sig.tolist() if isinstance(raw_sig, np.ndarray) else list(raw_sig)
    nz = np.flatnonzero(np.asarray(entry_sig) != 0)
    trailing = exit_kind == "trail"
    slip = slippage_points
    tax_k = mult * tax_rate

    trades = []
    pos = None
    pending_entry = 0
    pending_exit = None

    def close_trade(i, fill, reason, hold):
        d = pos["dir"]
        pts = (fill - pos["entry_price"]) * d
        raw_exit = fill - off[i]
        costs = (2 * commission_per_side + abs(pos["raw_entry"]) * tax_k + abs(raw_exit) * tax_k
                 + pos["roll_cost"])
        pnl = pts * mult - costs
        trades.append({
            "variant": variant_name, "side": "long" if d == 1 else "short",
            "signal_date": pos["signal_date"], "entry_date": pos["entry_date"],
            "exit_date": dates[i], "entry_price": pos["entry_price"], "exit_price": float(fill),
            "raw_entry_price": pos["raw_entry"], "raw_exit_price": float(raw_exit),
            "initial_stop": pos["initial_stop"], "target": pos["target"], "points": float(pts),
            "n_rolls": pos["n_rolls"], "roll_cost_ntd": pos["roll_cost"],
            "cost_ntd": float(costs), "pnl_ntd": float(pnl),
            "return_pct": float(pnl / starting_capital),
            "hold_days": int(hold), "exit_reason": reason,
        })

    i = lo
    while i <= hi:
        if pos is None and pending_entry == 0:
            # 空手、沒有待辦：下一件會發生的事就是下一個進場訊號(訊號在最後一天不會進場)
            k = int(nz.searchsorted(i, side="left"))
            if k >= len(nz) or nz[k] >= hi:
                break
            j = int(nz[k])
            pending_entry = es[j]
            i = j + 1
            continue
        # ---- 開盤 ----
        if pos is not None and pending_exit is not None:
            close_trade(i, o[i] - slip * pos["dir"], pending_exit, i - pos["entry_idx"])
            pos, pending_exit = None, None
        elif pos is not None and roll[i]:
            pos["n_rolls"] += 1
            pos["roll_cost"] += 2 * commission_per_side + 2 * abs(o[i] - off[i]) * tax_k
        if pos is None and pending_entry != 0:
            d = pending_entry
            sig_i = i - 1
            a = atr[sig_i]
            if np.isfinite(a) and a > 0:
                fill = o[i] + slip * d
                stop = fill - d * initial_stop_atr * a
                tgt = None if trailing else fill + d * exit_mult * a
                pos = {"dir": d, "entry_idx": i, "entry_date": dates[i], "signal_date": dates[sig_i],
                       "entry_price": float(fill), "raw_entry": float(fill - off[i]),
                       "stop": float(stop), "initial_stop": float(stop),
                       "target": float(tgt) if tgt is not None else float("nan"), "tgt": tgt,
                       "best_close": None, "n_rolls": 0, "roll_cost": 0.0}
        pending_entry = 0

        # ---- 盤中：停損(含跳空) / 停利 ----
        if pos is not None:
            d, stop, tgt = pos["dir"], pos["stop"], pos["tgt"]
            oi, hi_i, lo_i = o[i], h[i], l[i]
            fill, reason = None, None
            if d == 1:
                if oi <= stop:
                    fill, reason = oi - slip, "stop"
                elif tgt is not None and oi >= tgt:
                    fill, reason = oi, "停利"
                elif lo_i <= stop:
                    fill, reason = stop - slip, "stop"
                elif tgt is not None and hi_i >= tgt:
                    fill, reason = tgt, "停利"
            else:
                if oi >= stop:
                    fill, reason = oi + slip, "stop"
                elif tgt is not None and oi <= tgt:
                    fill, reason = oi, "停利"
                elif hi_i >= stop:
                    fill, reason = stop + slip, "stop"
                elif tgt is not None and lo_i <= tgt:
                    fill, reason = tgt, "停利"
            if fill is not None:
                if reason == "stop":
                    reason = "停損" if stop == pos["initial_stop"] else "移動停損"
                close_trade(i, fill, reason, i - pos["entry_idx"] + 1)
                pos = None

        # ---- 收盤 ----
        if i == hi:
            if pos is not None:
                close_trade(i, c[i] - slip * pos["dir"], "期末平倉", i - pos["entry_idx"] + 1)
                pos = None
            break
        if pos is not None:
            d = pos["dir"]
            if trailing:
                ci = c[i]
                bc = pos["best_close"]
                bc = ci if bc is None else (max(bc, ci) if d == 1 else min(bc, ci))
                pos["best_close"] = bc
                ai = atr[i]
                if np.isfinite(ai):
                    trail = bc - d * exit_mult * ai
                    pos["stop"] = max(pos["stop"], trail) if d == 1 else min(pos["stop"], trail)
            if rs[i] == -d:
                pending_exit = "反向訊號"
            elif (i - pos["entry_idx"] + 1) >= max_hold_days:
                pending_exit = "最長持有"
        else:
            if es[i] != 0:
                pending_entry = int(es[i])
        i += 1
    return trades
