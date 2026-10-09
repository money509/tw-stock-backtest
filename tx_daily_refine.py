"""
tx_daily_refine.py
====================
台指期(預設小台MTX)日線策略 —— 「精煉模式」(compare_tx_daily.py --refine)。

背景：--grid(1872變體)看完之後，MACD類進場看起來最有希望。可是2001~2023跟2024~的結果都已經
看過了，再從同一批資料裡挑「最好的格子」只會重複過度擬合。所以這個模式要檢驗的是
「精煉這個動作本身」：用walk-forward(每年只用當時已知的歷史挑變體、然後拿下一年來算)當主要判定。

事先登記的變體(312個，順序固定 = 平手時的最後判準)：
  進場規則(5條；混搭定義跟網格完全相同：一個在第t天發出d、另一個在[t−2, t]內也發出過d)：
    MACD ; MA_5_20&MACD ; MACD&DONCHIAN20 ; MACD&RSI50 ; MA_5_20&DONCHIAN20
  MACD參數(只用在含MACD的規則；訊號 = MACD線上穿/下穿訊號線，前slow+signal根暖身不出訊號)：
    標準(12,26,9)、快(8,17,9)、慢(19,39,9)。標準參數的訊號跟網格的sig_MACD逐日相同(測試有驗證)。
  方向：多空 / 只做多(空訊號不進場；多單的「反向訊號出場」仍用原始空訊號，跟網格一樣)
  出場(12種)：初始停損{1.0, 1.5, 2.0}×ATR14 × {停利3.0、停利4.0、停利5.0、移動停損3.0}×ATR14；
    停利限價成交規則、反向訊號出場、60天安全閥、成本、換月 = 網格引擎(run_tx_daily_grid_backtest)原封不動
  無動能門檻。總數 = 4規則×3參數×2方向×12 + 1×2×12 = 288 + 24 = 312。

每個變體每段(訓練2001~2012、測試2013~2023、近期2024~指數代理；每段獨立重跑，跟網格一樣)：
  交易數、PF、勝率、損益、平倉最大回撤、逐日市值最大回撤(見mtm_equity_curve)、拿掉最賺3筆後損益、
  bootstrap正報酬比例(robustness_analysis，seed=42)、平均持有天數、在場時間比例、多/空損益、
  成本加壓PF(滑價1點改3點整個重跑)、獲利年度比例(依進場日歸年；該段有資料、但沒交易的年度算沒獲利)。
  買進持有參考(每段)：1口一路換月持有(tx_daily_engine.run_buy_and_hold)，另報逐日市值最大回撤與
  損益/逐日市值最大回撤。

Walk-forward(主要判定)：
  Y = 2006 ~ 資料最後一年。每年1/1，312個變體各自只用「出場日 < Y年1/1」的歷史交易評分(2001起累積；
  2001~2012取訓練段、2013~2023取測試段、2024~取近期段的獨立回測，依日期串起來)。
  ⚠️ 跟原始規格的差異：規格寫「進場日 < Y年1/1」。但12月進場、隔年1~2月才出場的交易，它的損益要用
  Y年的價格才算得出來，在Y年1/1其實還不知道 → 改成「出場日 < Y年1/1」(= 進場日 < Y 且已平倉)，
  完全不偷看未來。
  歷史交易數 >= 30 才合格；選歷史PF最高(平手 → 歷史損益 → 登記順序)；取該變體「進場日在Y年」的交易；
  逐年串接 = walk-forward交易清單。換變體時，前一年的部位(進場在Y−1、出場在Y)照原本規則出場，
  可能跟新變體的部位短暫重疊(都算進去)。
  對照(同樣2006~)：固定基準 = MACD標準 多空 停損1.0/停利3.0(永遠同一個)、買進持有。
  通過(全部成立)：WF PF>1；WF bootstrap正報酬比例>80%；WF拿掉最賺3筆後損益>0；
    WF 損益/逐日市值最大回撤 > 同期買進持有的同一比值；獲利年度比例 >= 55%。

診斷(一定印出)：維度檢視(MACD參數/方向/出場/進場規則 → 合格數、訓練測試PF都>1比例、測試PF中位數、
  近期PF中位數；合格 = 訓練交易數>=40，跟網格一樣)；配對比較(只做多 vs 多空、快/慢 vs 標準)；
  多重比較與「資料已經看過」警告。
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

import tx_daily_engine as eng
from tx_daily_engine import (
    CONTRACT_MULTIPLIER, DEFAULT_COMMISSION_PER_SIDE, FUTURES_TAX_RATE, SLIPPAGE_POINTS, STARTING_CAPITAL,
    MAX_HOLD_DAYS, GRID_STOP_MULTS,
)
import compare_tx_daily as ctd
import tx_daily_grid as grid
from robustness_analysis import bootstrap_resample_pnl, summarize_bootstrap

REFINE_RULES = [
    ("MACD", ("MACD",)),
    ("MA_5_20&MACD", ("MA_5_20", "MACD")),
    ("MACD&DONCHIAN20", ("MACD", "DONCHIAN20")),
    ("MACD&RSI50", ("MACD", "RSI50")),
    ("MA_5_20&DONCHIAN20", ("MA_5_20", "DONCHIAN20")),
]
MACD_PARAMS = [("標準", (12, 26, 9)), ("快", (8, 17, 9)), ("慢", (19, 39, 9))]
DIRECTIONS = [("both", "多空"), ("long", "只做多")]
REFINE_EXIT_OPTIONS = (("target", 3.0), ("target", 4.0), ("target", 5.0), ("trail", 3.0))
STRESS_SLIPPAGE_POINTS = 3.0
WF_START_YEAR = 2006
WF_MIN_HIST_TRADES = 30
WF_BOOTSTRAP_PASS_PCT = 80.0
WF_POS_YEAR_PCT = 55.0
PERIOD_KEYS = grid.PERIOD_KEYS
PERIOD_LABEL = grid.PERIOD_LABEL
MAIN_SEGMENT_PERIODS = ("train", "test")       # 用ds["main"]那份資料(期貨或備援指數)
RECENT_SEGMENT_PERIODS = ("recent",)          # 一律加權指數代理


def _macd_tag(pk):
    f, s, g = dict(MACD_PARAMS)[pk]
    return f"{pk}{f}/{s}/{g}"


# ---------------------------------------------------------------------------
# 變體 / 指標
# ---------------------------------------------------------------------------
def refine_exit_configs():
    return eng.grid_exit_configs(GRID_STOP_MULTS, REFINE_EXIT_OPTIONS)


def refine_variant_list() -> list:
    """312個變體，順序固定：規則 → MACD參數 → 方向 → 出場(停損外圈、出場方式內圈)。"""
    out = []
    for rule, comps in REFINE_RULES:
        params = [pk for pk, _ in MACD_PARAMS] if "MACD" in comps else [None]
        for pk in params:
            for dk, dl in DIRECTIONS:
                for ex in refine_exit_configs():
                    rule_lab = rule if pk is None else f"{rule}[{_macd_tag(pk)}]"
                    v = {"name": f"{rule_lab}｜{dl}｜{ex['label']}", "rule": rule, "components": comps,
                         "macd": pk, "direction": dk, "direction_label": dl,
                         "sig_col": f"sig_{rule}" if pk is None else f"sig_{rule}@{pk}",
                         "exit_label": ex["label"]}
                    v.update(ex)
                    out.append(v)
    return out


def fixed_baseline_name():
    return f"MACD[{_macd_tag('標準')}]｜多空｜停損1.0/停利3.0"


def compute_refine_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """compute_grid_indicators()全部欄位 + 每組MACD參數的sig_MACD@<參數> + 含MACD規則的
    sig_<規則>@<參數>(混搭用網格的compose_pair)。MA_5_20&DONCHIAN20直接用網格欄位。"""
    ind = eng.compute_grid_indicators(df)
    cols = {}
    for pk, (f, s, g) in MACD_PARAMS:
        cols[f"sig_MACD@{pk}"] = eng.macd_cross_signal(df["Close"], f, s, g)[0]
    for rule, comps in REFINE_RULES:
        if "MACD" not in comps:
            continue
        for pk, _ in MACD_PARAMS:
            parts = [cols[f"sig_MACD@{pk}"] if c == "MACD" else ind[f"sig_{c}"].to_numpy(int) for c in comps]
            cols[f"sig_{rule}@{pk}"] = parts[0] if len(parts) == 1 else eng.compose_pair(*parts)
    return pd.concat([ind, pd.DataFrame(cols, index=ind.index).astype(int)], axis=1)


def refine_entry_signals(arrs: dict, v: dict):
    """(entry_sig, raw_sig)：raw = 規則原始訊號(反向出場用)；只做多 → 進場只留+1。"""
    raw = arrs["ind"][v["sig_col"]].to_numpy(int)
    if v["direction"] == "both":
        return raw, raw
    return np.where(raw == 1, 1, 0).astype(int), raw


def refine_period_frames(ds):
    """{period: {"arrs", "df", "label"}}；訓練/測試共用ds["main"]，近期一律加權指數。"""
    main = ds["main"]
    arrs_main = eng.prepare_grid_arrays(main, compute_refine_indicators(main))
    if ds["source"] == "index":
        idx_df, arrs_index = main, arrs_main
    else:
        idx_df = ds["index"]
        arrs_index = eng.prepare_grid_arrays(idx_df, compute_refine_indicators(idx_df))
    main_label = "台指期(社群分鐘資料，換月調整)" if ds["source"] == "futures" else "加權指數(備援)"
    return {"train": {"arrs": arrs_main, "df": main, "label": main_label},
            "test": {"arrs": arrs_main, "df": main, "label": main_label},
            "recent": {"arrs": arrs_index, "df": idx_df, "label": "加權指數代理"}}


# ---------------------------------------------------------------------------
# 逐日市值(mark-to-market)權益曲線
# ---------------------------------------------------------------------------
def _bounds(dates, start, end):
    lo = 0 if start is None else int(dates.searchsorted(pd.Timestamp(start), side="left"))
    hi = len(dates) - 1 if end is None else int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    return lo, hi


def mtm_equity_curve(trades, arrs, start=None, end=None, contract="mini") -> pd.Series:
    """[start, end]每個交易日收盤的累積損益(NT$，起點0)：
      已平倉 = 出場當天起認列該筆pnl_ntd(含全部成本/換月成本)；
      持倉中 = (當天收盤 − 進場價)×方向×每點金額(調整後點數；成本等平倉那天才認列)。
    出場日收盤一定已經空手(一次1口、出場當天不會再進場)，所以平倉權益曲線的每個點都在這條曲線上，
    逐日市值最大回撤的絕對值一定 >= 平倉最大回撤。可以同時有好幾筆重疊(walk-forward換變體時)，直接相加。
    交易的進出場日必須在[start, end]內，否則ValueError。"""
    mult = CONTRACT_MULTIPLIER[contract]
    dates = arrs["dates"]
    lo, hi = _bounds(dates, start, end)
    if hi < lo:
        return pd.Series([], index=dates[:0], dtype=float)
    n = hi - lo + 1
    c = np.asarray(arrs["c"], float)[lo:hi + 1]
    realized = np.zeros(n)
    unreal = np.zeros(n)
    if trades:
        e_idx = dates.get_indexer(pd.DatetimeIndex([t["entry_date"] for t in trades])) - lo
        x_idx = dates.get_indexer(pd.DatetimeIndex([t["exit_date"] for t in trades])) - lo
        if (e_idx < 0).any() or (x_idx >= n).any() or (x_idx < e_idx).any():
            raise ValueError("交易日期超出逐日市值曲線的區間")
        for t, e, x in zip(trades, e_idx, x_idx):
            realized[x] += t["pnl_ntd"]
            if x > e:
                d = 1.0 if t["side"] == "long" else -1.0
                unreal[e:x] += (c[e:x] - t["entry_price"]) * d * mult
    return pd.Series(np.cumsum(realized) + unreal, index=dates[lo:hi + 1])


def mtm_equity_segments(segments, contract="mini") -> pd.Series:
    """segments = [(trades, arrs, start, end), ...](依時間順序)；各段曲線接起來，後一段從前一段終值開始。"""
    parts, carry = [], 0.0
    for trades, arrs, start, end in segments:
        s = mtm_equity_curve(trades, arrs, start, end, contract)
        if len(s) == 0:
            continue
        parts.append(s + carry)
        carry = float(parts[-1].iloc[-1])
    if not parts:
        return pd.Series([], dtype=float)
    return pd.concat(parts)


def max_drawdown(equity) -> float:
    """最大回撤(NT$，<=0)；起點0也算一個高點(跟平倉回撤的算法一致)。"""
    eq = np.concatenate([[0.0], np.asarray(equity, float)])
    return float((eq - np.maximum.accumulate(eq)).min())


def pnl_to_dd_ratio(pnl, mdd):
    """損益 / |最大回撤|；回撤=0時：賺 → ∞、否則0。"""
    if mdd < 0:
        return float(pnl) / abs(float(mdd))
    return float("inf") if pnl > 0 else 0.0


def profit_factor(pnl):
    pnl = np.asarray(pnl, float)
    gw, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    return gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)


def boot_pct(trades, n_boot):
    return summarize_bootstrap(bootstrap_resample_pnl(trades, n_resamples=n_boot,
                                                      seed=ctd.BOOTSTRAP_SEED))["pct_positive"]


def yearly_pnl_by_entry(trades) -> dict:
    out = {}
    for t in trades:
        y = pd.Timestamp(t["entry_date"]).year
        out[y] = out.get(y, 0.0) + t["pnl_ntd"]
    return out


def positive_year_share(trades, years):
    """(獲利年度數, 年度數, 比例%)：依進場日歸年；years裡沒有交易的年度算沒獲利。"""
    years = list(years)
    if not years:
        return 0, 0, float("nan")
    by = yearly_pnl_by_entry(trades)
    k = sum(1 for y in years if by.get(y, 0.0) > 0)
    return k, len(years), k / len(years) * 100


def period_years(dates, start, end):
    lo, hi = _bounds(dates, start, end)
    if hi < lo:
        return []
    return sorted(set(dates[lo:hi + 1].year))


# ---------------------------------------------------------------------------
# 跑全部變體
# ---------------------------------------------------------------------------
def period_metrics(trades, stress_trades, arrs, start, end, contract="mini", n_boot=ctd.N_BOOTSTRAP):
    dates = arrs["dates"]
    s = grid.quick_stats(trades, grid.period_n_days(dates, start, end))
    s["mtm_mdd_ntd"] = max_drawdown(mtm_equity_curve(trades, arrs, start, end, contract))
    s["bootstrap_pct_positive"] = boot_pct(trades, n_boot)
    s["stress_pf"] = profit_factor([t["pnl_ntd"] for t in stress_trades])
    s["stress_pnl_ntd"] = float(sum(t["pnl_ntd"] for t in stress_trades))
    k, n, pct = positive_year_share(trades, period_years(dates, start, end))
    s["pos_years"], s["n_years"], s["pos_year_pct"] = k, n, pct
    return s


def buy_hold_metrics(df, arrs, start, end, contract="mini", commission=DEFAULT_COMMISSION_PER_SIDE):
    bh = eng.run_buy_and_hold(df, start=start, end=end, contract=contract, commission_per_side=commission)
    pnl = float(sum(t["pnl_ntd"] for t in bh))
    mdd = max_drawdown(mtm_equity_curve(bh, arrs, start, end, contract))
    return {"trades": bh, "total_pnl_ntd": pnl, "mtm_mdd_ntd": mdd, "ratio": pnl_to_dd_ratio(pnl, mdd)}


def run_refine(ds, variants, contract="mini", commission=DEFAULT_COMMISSION_PER_SIDE,
               n_boot=ctd.N_BOOTSTRAP, log=None):
    """回傳 {"stats": {(name, period)}, "trades": {(name, period): [...]}, "bh": {period}, "frames", "timing"}。"""
    t0 = time.perf_counter()
    frames = refine_period_frames(ds)
    t_ind = time.perf_counter() - t0
    stats, trades, bh = {}, {}, {}
    t1 = time.perf_counter()
    n_runs = 0
    for period in PERIOD_KEYS:
        fr = frames[period]
        arrs = fr["arrs"]
        start, end = grid._period_bounds(period)
        tp = time.perf_counter()
        sig_cache = {}
        for v in variants:
            key = (v["sig_col"], v["direction"])
            if key not in sig_cache:
                sig_cache[key] = refine_entry_signals(arrs, v)
            es, rs = sig_cache[key]
            common = dict(start=start, end=end, contract=contract, commission_per_side=commission,
                          variant_name=v["name"])
            tr = eng.run_tx_daily_grid_backtest(arrs, es, rs, v["stop_atr"], v["exit_kind"], v["exit_mult"],
                                                **common)
            tr_s = eng.run_tx_daily_grid_backtest(arrs, es, rs, v["stop_atr"], v["exit_kind"], v["exit_mult"],
                                                  slippage_points=STRESS_SLIPPAGE_POINTS, **common)
            n_runs += 2
            trades[(v["name"], period)] = tr
            stats[(v["name"], period)] = period_metrics(tr, tr_s, arrs, start, end, contract, n_boot)
        bh[period] = buy_hold_metrics(fr["df"], arrs, start, end, contract, commission)
        if log:
            log(f"  精煉：{PERIOD_LABEL[period]}段 {len(variants)}個變體(含成本加壓重跑)完成，"
                f"{time.perf_counter() - tp:.1f}秒")
    timing = {"indicators_sec": t_ind, "backtests_sec": time.perf_counter() - t1,
              "total_sec": time.perf_counter() - t0, "n_runs": n_runs}
    return {"stats": stats, "trades": trades, "bh": bh, "frames": frames, "timing": timing}


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def combined_trades(variants, trades):
    """{name: [trade(多一個"period"欄位), ...]}，訓練/測試/近期三段依時間串起來。"""
    out = {}
    for v in variants:
        lst = []
        for p in PERIOD_KEYS:
            lst += [dict(t, period=p) for t in trades.get((v["name"], p)) or []]
        lst.sort(key=lambda t: pd.Timestamp(t["entry_date"]))
        out[v["name"]] = lst
    return out


def walk_forward(variants, trades_by_variant, years, min_hist=WF_MIN_HIST_TRADES):
    """每年Y：只用出場日 < Y年1/1的交易評分(歷史交易數>=min_hist才合格)，選歷史PF最高
    (平手 → 歷史損益 → 登記順序)，取它進場日在Y年的交易。回傳每年一個dict的list。"""
    pre = []
    for order, v in enumerate(variants):
        tr = trades_by_variant.get(v["name"]) or []
        exit_ts = np.array([pd.Timestamp(t["exit_date"]).value for t in tr], dtype=np.int64)
        pnl = np.array([t["pnl_ntd"] for t in tr], float)
        pre.append((order, v, tr, exit_ts, pnl))
    rows = []
    for y in years:
        cutoff = pd.Timestamp(year=int(y), month=1, day=1).value
        best, best_key, n_cand = None, None, 0
        for order, v, tr, exit_ts, pnl in pre:
            m = exit_ts < cutoff
            n = int(m.sum())
            if n < min_hist:
                continue
            n_cand += 1
            ph = pnl[m]
            key = (profit_factor(ph), float(ph.sum()), -order)
            if best_key is None or key > best_key:
                best, best_key = (v, n, ph), key
        if best is None:
            rows.append({"year": int(y), "variant": None, "n_candidates": 0, "hist_trades": 0,
                         "hist_pf": float("nan"), "hist_pnl": float("nan"), "trades": []})
            continue
        v, n, ph = best
        yt = [t for t in trades_by_variant[v["name"]] if pd.Timestamp(t["entry_date"]).year == int(y)]
        rows.append({"year": int(y), "variant": v, "n_candidates": n_cand, "hist_trades": n,
                     "hist_pf": profit_factor(ph), "hist_pnl": float(ph.sum()), "trades": yt})
    return rows


def wf_years(frames, start_year=WF_START_YEAR):
    last = max(fr["arrs"]["dates"].max() for fr in frames.values())
    return list(range(start_year, int(pd.Timestamp(last).year) + 1))


def _wf_segments(frames, trades, start_year=WF_START_YEAR):
    """把(帶period欄位的)交易分到「主資料段(start_year~2023)」與「近期指數段(2024~)」。"""
    _, main_end = grid._period_bounds("test")
    rec_start, rec_end = grid._period_bounds("recent")
    return [
        ([t for t in trades if t["period"] in MAIN_SEGMENT_PERIODS], frames["test"]["arrs"],
         f"{start_year}-01-01", main_end),
        ([t for t in trades if t["period"] in RECENT_SEGMENT_PERIODS], frames["recent"]["arrs"],
         rec_start, rec_end),
    ]


def series_stats(trades, frames, years, contract="mini", n_boot=ctd.N_BOOTSTRAP, start_year=WF_START_YEAR):
    """一串(帶period欄位的)交易在walk-forward年度上的統計。"""
    s = grid.quick_stats(trades)
    eq = mtm_equity_segments(_wf_segments(frames, trades, start_year), contract)
    s["mtm_mdd_ntd"] = max_drawdown(eq)
    s["ratio"] = pnl_to_dd_ratio(s["total_pnl_ntd"], s["mtm_mdd_ntd"])
    s["bootstrap_pct_positive"] = boot_pct(trades, n_boot)
    k, n, pct = positive_year_share(trades, years)
    s["pos_years"], s["n_years"], s["pos_year_pct"] = k, n, pct
    s["yearly"] = yearly_pnl_by_entry(trades)
    return s


def buy_hold_wf(frames, years, contract="mini", commission=DEFAULT_COMMISSION_PER_SIDE, start_year=WF_START_YEAR):
    """同一段年度的買進持有：主資料段(start_year~2023)一口、近期指數段(2024~)一口，逐日市值曲線接起來。"""
    segs = []
    for keys, fr_key in ((MAIN_SEGMENT_PERIODS, "test"), (RECENT_SEGMENT_PERIODS, "recent")):
        fr = frames[fr_key]
        if keys == MAIN_SEGMENT_PERIODS:
            start, end = f"{start_year}-01-01", grid._period_bounds("test")[1]
        else:
            start, end = grid._period_bounds("recent")
        bh = eng.run_buy_and_hold(fr["df"], start=start, end=end, contract=contract, commission_per_side=commission)
        segs.append((bh, fr["arrs"], start, end))
    eq = mtm_equity_segments(segs, contract)
    pnl = float(sum(t["pnl_ntd"] for tr, *_ in segs for t in tr))
    mdd = max_drawdown(eq)
    yearly = {}
    if len(eq):
        ye = eq.groupby(eq.index.year).last()
        prev = 0.0
        for y, val in ye.items():
            yearly[int(y)] = float(val - prev)
            prev = float(val)
    return {"total_pnl_ntd": pnl, "mtm_mdd_ntd": mdd, "ratio": pnl_to_dd_ratio(pnl, mdd),
            "n_lots_segments": sum(1 for tr, *_ in segs if tr), "yearly": yearly}


def wf_verdict(wf, bh):
    checks = [
        (f"walk-forward PF {ctd._fmt_pf(wf['profit_factor'])} > 1", wf["profit_factor"] > 1),
        (f"walk-forward bootstrap正報酬比例 {wf['bootstrap_pct_positive']:.1f}% > {WF_BOOTSTRAP_PASS_PCT:g}%",
         wf["bootstrap_pct_positive"] > WF_BOOTSTRAP_PASS_PCT),
        (f"walk-forward拿掉最賺3筆後損益 {wf['pnl_excl_top3_ntd']:,.0f} > 0", wf["pnl_excl_top3_ntd"] > 0),
        (f"walk-forward 損益/逐日市值最大回撤 {_f(wf['ratio'])} > 同期買進持有 {_f(bh['ratio'])}",
         bool(wf["ratio"] > bh["ratio"])),
        (f"獲利年度比例 {wf['pos_years']}/{wf['n_years']} = {_f(wf['pos_year_pct'], 1)}% >= {WF_POS_YEAR_PCT:g}%",
         bool(np.isfinite(wf["pos_year_pct"]) and wf["pos_year_pct"] >= WF_POS_YEAR_PCT)),
    ]
    return all(ok for _, ok in checks), checks


def wf_change_count(rows):
    names = [r["variant"]["name"] if r["variant"] else None for r in rows]
    return sum(1 for a, b in zip(names, names[1:]) if a != b)


def wf_distribution(rows):
    chosen = [r["variant"] for r in rows if r["variant"] is not None]
    return {
        "MACD參數": Counter(v["macd"] or "(無MACD)" for v in chosen),
        "方向": Counter(v["direction_label"] for v in chosen),
        "出場": Counter(v["exit_label"] for v in chosen),
        "進場規則": Counter(v["rule"] for v in chosen),
    }


# ---------------------------------------------------------------------------
# 診斷
# ---------------------------------------------------------------------------
def _eligible(stats, v, min_trades=ctd.MIN_TRAIN_TRADES):
    return stats[(v["name"], "train")]["trade_count"] >= min_trades


def _median(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def axis_rows(variants, stats, wf_rows=None, min_trades=ctd.MIN_TRAIN_TRADES):
    chosen = Counter(r["variant"]["name"] for r in (wf_rows or []) if r["variant"] is not None)
    axes = [
        ("MACD參數", [pk for pk, _ in MACD_PARAMS], lambda v: v["macd"]),
        ("方向", [dl for _, dl in DIRECTIONS], lambda v: v["direction_label"]),
        ("出場", [ex["label"] for ex in refine_exit_configs()], lambda v: v["exit_label"]),
        ("進場規則", [r for r, _ in REFINE_RULES], lambda v: v["rule"]),
    ]
    rows = []
    for axis, values, fn in axes:
        for val in values:
            vs = [v for v in variants if fn(v) == val]
            el = [v for v in vs if _eligible(stats, v, min_trades)]
            both = [v for v in el if grid._both_gt1(stats, v["name"])]
            rows.append({
                "維度": axis, "設定": val, "變體數": len(vs), "合格數(訓練>=40筆)": len(el),
                "兩段PF都>1數量": len(both),
                "兩段PF都>1比例(%)": round(len(both) / len(el) * 100, 1) if el else float("nan"),
                "測試PF中位數": round(_median([stats[(v["name"], "test")]["profit_factor"] for v in el]), 3),
                "近期PF中位數": round(_median([stats[(v["name"], "recent")]["profit_factor"] for v in el]), 3),
                "walk-forward被選中年數": sum(chosen[v["name"]] for v in vs),
            })
    return rows


def _pair_compare(pairs, stats):
    """pairs = [(挑戰者, 對照)]；回傳(配對數, 測試PF較高%, 近期PF較高%, 測試逐日市值回撤較小%)。"""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "test_pf_higher": float("nan"), "recent_pf_higher": float("nan"),
                "test_mdd_smaller": float("nan")}
    a = sum(stats[(x["name"], "test")]["profit_factor"] > stats[(y["name"], "test")]["profit_factor"]
            for x, y in pairs)
    b = sum(stats[(x["name"], "recent")]["profit_factor"] > stats[(y["name"], "recent")]["profit_factor"]
            for x, y in pairs)
    c = sum(abs(stats[(x["name"], "test")]["mtm_mdd_ntd"]) < abs(stats[(y["name"], "test")]["mtm_mdd_ntd"])
            for x, y in pairs)
    return {"n": n, "test_pf_higher": a / n * 100, "recent_pf_higher": b / n * 100, "test_mdd_smaller": c / n * 100}


def head_to_head(variants, stats):
    def key(v, **over):
        d = {"rule": v["rule"], "macd": v["macd"], "direction": v["direction"], "exit_id": v["exit_id"]}
        d.update(over)
        return (d["rule"], d["macd"], d["direction"], d["exit_id"])
    idx = {key(v): v for v in variants}
    out = {}
    pairs = [(v, idx[key(v, direction="both")]) for v in variants if v["direction"] == "long"]
    out["只做多 vs 多空"] = _pair_compare(pairs, stats)
    for pk in ("快", "慢"):
        pairs = [(v, idx[key(v, macd="標準")]) for v in variants if v["macd"] == pk]
        out[f"MACD{pk} vs 標準"] = _pair_compare(pairs, stats)
    return out


# ---------------------------------------------------------------------------
# 報告
# ---------------------------------------------------------------------------
_fmt_pf = ctd._fmt_pf
_f = grid._f


def refine_rules_lines(contract, commission, n_variants=None):
    mult = CONTRACT_MULTIPLIER[contract]
    name = "小台MTX" if contract == "mini" else "大台TX"
    n_variants = n_variants or len(refine_variant_list())
    exits = "、".join(f"停利{m:.1f}" if k == "target" else f"移動停損{m:.1f}(無停利)" for k, m in REFINE_EXIT_OPTIONS)
    return [
        "== 預先登記的規則(精煉模式；結果出來前就固定，不會依結果修改) ==",
        f"商品：{name}，每點NT${mult}，一次最多1口，不反手",
        f"變體數：4條含MACD的規則 × 3組MACD參數 × 2方向 × 12出場 + MA_5_20&DONCHIAN20 × 2方向 × 12出場"
        f" = 288 + 24 = {n_variants}個",
        "進場規則(5條)：MACD ; MA_5_20&MACD ; MACD&DONCHIAN20 ; MACD&RSI50 ; MA_5_20&DONCHIAN20",
        "  混搭定義跟網格完全相同：第t天方向d ⇔ 一個在第t天發出d、另一個在[t−2, t]內也發出過d",
        "MACD參數(只用在含MACD的規則；訊號 = MACD線上穿/下穿訊號線，前slow+signal根暖身不出訊號)："
        + "、".join(f"{pk}({f},{s},{g})" for pk, (f, s, g) in MACD_PARAMS),
        "方向：多空(多空訊號都進場) / 只做多(空訊號不進場；多單的反向訊號出場仍用原始空訊號)",
        f"出場(12種)：初始停損{{1.0, 1.5, 2.0}}×ATR14 × {{{exits}}}×ATR14；",
        "  停利 = 限價單(觸及以停利價成交、不扣滑價；開盤跳空越過以開盤價成交；同日碰到停損停利 → 當作先停損)",
        f"  反向原始訊號 → 隔天開盤出場；持有滿{MAX_HOLD_DAYS}天 → 隔天開盤出場；ATR取訊號日的值；無動能門檻",
        f"成本：手續費每邊NT${commission:g} + 期交稅每邊{FUTURES_TAX_RATE*100:.3f}%(未調整價格)；"
        f"開盤/停損成交滑價{SLIPPAGE_POINTS:g}點；換月時持倉 → 多收一次來回手續費+稅",
        f"成本加壓：滑價改{STRESS_SLIPPAGE_POINTS:g}點整個重跑，只報告PF",
        "期間：訓練2001~2012、測試2013~2023、近期2024~最新(指數代理)；每段獨立重跑，期末強制平倉",
        "逐日市值最大回撤：每天收盤的累積損益(已平倉損益 + 持倉以收盤價計的未實現損益，調整後點數×每點金額；"
        "成本在平倉日認列)的最大回撤",
        "獲利年度比例：依進場日歸年；有資料但沒交易的年度算沒獲利",
        f"Walk-forward(主要判定)：Y = {WF_START_YEAR} ~ 資料最後一年；每年1/1只用「出場日 < Y年1/1」(已平倉)的"
        "歷史交易(2001起累積；2001~2012取訓練段、2013~2023取測試段、2024~取近期段的獨立回測)評分；",
        f"  歷史交易數>={WF_MIN_HIST_TRADES}才合格；選歷史PF最高(平手→歷史損益→登記順序)；取該變體進場日在Y年的交易；逐年串接",
        "  (原始規格寫「進場日 < Y年1/1」；跨年交易的損益要用Y年的價格才知道，所以改成出場日，避免偷看未來)",
        f"  對照(同樣{WF_START_YEAR}~)：固定基準 = {fixed_baseline_name()}(永遠同一個)；買進持有(1口一路換月)",
        f"通過(全部成立)：WF PF>1；WF bootstrap正報酬比例>{WF_BOOTSTRAP_PASS_PCT:g}%；WF拿掉最賺3筆後損益>0；"
        f"WF 損益/逐日市值最大回撤 > 同期買進持有的同一比值；獲利年度比例>={WF_POS_YEAR_PCT:g}%",
        f"診斷(一定印出)：維度檢視(MACD參數/方向/出場/進場規則；合格 = 訓練交易數>={ctd.MIN_TRAIN_TRADES})、"
        "配對比較(只做多 vs 多空、快/慢 vs 標準)、多重比較與資料已看過警告",
        f"起始資金NT${STARTING_CAPITAL:,}",
    ]


def hindsight_warning_lines(n_variants):
    return [
        "== ⚠️ 多重比較 / 資料已經看過 警告(請先看這段) ==",
        "1. 這5條進場規則(MACD系列 + MA_5_20&DONCHIAN20)是看過--grid在2013~2023(測試)與2024~(近期)的結果之後"
        "才挑出來的。所以任何「固定變體」在測試期/近期的數字都是樂觀的(已經被挑過一次)，不能當成獨立驗證。",
        f"2. {n_variants}個變體 = 又一輪多重比較；就算MACD類完全沒有優勢，{n_variants}個裡面也一定有些測試/近期很好看。"
        "不要從tx_refine_all.csv裡挑測試期或近期最好的那一格來用。",
        "3. Walk-forward(每年只用當時已平倉的歷史挑變體)比較誠實：它檢驗的是「每年依歷史PF重新挑一次」這個流程，"
        "不是某一格。但候選清單本身(這312個)仍是事後挑的 → walk-forward也還是偏樂觀。",
        "4. 唯一真正乾淨的檢驗是往後的紙上交易(forward paper trading)：規則現在凍結，用之後才發生的行情來看。",
    ]


def _stat_row_line(label, s, is_bh=False):
    if is_bh:
        return (f"  {label:<10} {'-':>5} {'-':>7} {s['total_pnl_ntd']:>12,.0f} {'-':>11} {s['mtm_mdd_ntd']:>12,.0f} "
                f"{_f(s['ratio']):>7} {'-':>8} {'-':>12} {'-':>9}")
    return (f"  {label:<10} {s['trade_count']:>5} {_fmt_pf(s['profit_factor']):>7} {s['total_pnl_ntd']:>12,.0f} "
            f"{s['max_drawdown_ntd']:>11,.0f} {s['mtm_mdd_ntd']:>12,.0f} {_f(s['ratio']):>7} "
            f"{s['bootstrap_pct_positive']:>7.1f}% {s['pnl_excl_top3_ntd']:>12,.0f} "
            f"{str(s['pos_years']) + '/' + str(s['n_years']):>9}")


def wf_section_lines(wf_rows, wf, fixed, bh, years):
    lines = [f"== Walk-forward判定({years[0] if years else '-'}~{years[-1] if years else '-'}；主要結論) =="]
    passed, checks = wf_verdict(wf, bh)
    for desc, ok in checks:
        lines.append(f"  [{'✓' if ok else '✗'}] {desc}")
    lines.append(f"判定：{'通過' if passed else '未通過'}")
    lines.append("")
    lines.append(f"  {'':<10} {'筆數':>5} {'PF':>7} {'損益':>12} {'平倉回撤':>11} {'逐日市值回撤':>12} "
                 f"{'損益/回撤':>7} {'boot+':>8} {'拿掉前3筆':>12} {'獲利年':>9}")
    lines.append(_stat_row_line("walk-forward", wf))
    lines.append(_stat_row_line("固定基準", fixed))
    lines.append(_stat_row_line("買進持有", bh, is_bh=True))
    lines.append(f"  (固定基準 = {fixed_baseline_name()}；2024~用加權指數代理；損益/回撤 = 損益/|逐日市值最大回撤|)")
    lines.append("")
    lines.append("  逐年(依進場日歸年；選擇只用出場日 < 當年1/1 的歷史)：")
    lines.append(f"  {'年度':<6}{'被選變體':<58}{'候選':>5}{'歷史筆':>7}{'歷史PF':>8}  {'當年筆':>6}{'當年損益':>11}"
                 f"{'當年PF':>8}  {'固定基準':>10}{'買進持有':>11}")
    for r in wf_rows:
        y = r["year"]
        tr = r["trades"]
        pnl = [t["pnl_ntd"] for t in tr]
        name = r["variant"]["name"] if r["variant"] else "(沒有合格變體)"
        lines.append(f"  {y:<6}{name:<58}{r['n_candidates']:>5}{r['hist_trades']:>7}{_fmt_pf(r['hist_pf']):>8}  "
                     f"{len(tr):>6}{sum(pnl):>11,.0f}{_fmt_pf(profit_factor(pnl)) if pnl else '-':>8}  "
                     f"{fixed['yearly'].get(y, 0.0):>10,.0f}{bh['yearly'].get(y, 0.0):>11,.0f}")
    lines.append(f"  被選變體換了{wf_change_count(wf_rows)}次(相鄰兩年不同就算一次；共{len(wf_rows)}年)")
    for dim, cnt in wf_distribution(wf_rows).items():
        lines.append(f"  被選{dim}分布：" + ("、".join(f"{k} {n}年" for k, n in cnt.most_common()) or "-"))
    return lines, passed


def axis_lines(axes):
    lines = ["== 維度檢視(分母 = 訓練交易數>=40的合格變體；只用來看哪個方向一致，不是用來挑格子) =="]
    lines.append(f"  {'維度':<8}{'設定':<18}{'合格/全部':>10}{'兩段>1%':>9}{'測試PF中位':>11}{'近期PF中位':>11}{'WF選中年':>9}")
    for r in axes:
        lines.append(f"  {r['維度']:<8}{str(r['設定']):<18}{str(r['合格數(訓練>=40筆)']) + '/' + str(r['變體數']):>10}"
                     f"{_f(r['兩段PF都>1比例(%)'], 1):>9}{_f(r['測試PF中位數']):>11}{_f(r['近期PF中位數']):>11}"
                     f"{r['walk-forward被選中年數']:>9}")
    return lines


def h2h_lines(h2h):
    lines = ["== 配對比較(其他設定完全相同，只差一個維度；全部配對) =="]
    for k, r in h2h.items():
        lines.append(f"  {k}：{r['n']}對；測試PF較高 {_f(r['test_pf_higher'], 1)}%、近期PF較高 "
                     f"{_f(r['recent_pf_higher'], 1)}%、測試逐日市值回撤較小 {_f(r['test_mdd_smaller'], 1)}%")
    lines.append("  (50%左右 = 這個維度沒有一致的差別；而且這些都是已經看過的期間)")
    return lines


def period_overview_lines(variants, stats, bh):
    lines = ["== 各期間概況(全體312個變體；固定基準與買進持有逐段) =="]
    base = fixed_baseline_name()
    for p in PERIOD_KEYS:
        pfs = np.array([stats[(v["name"], p)]["profit_factor"] for v in variants], float)
        s = stats[(base, p)]
        b = bh[p]
        lines.append(f"  {PERIOD_LABEL[p]}：PF>1的變體 {int((pfs > 1).sum())}/{len(pfs)}、PF中位數 {_f(_median(pfs))}")
        lines.append(f"    固定基準：{s['trade_count']}筆 PF {_fmt_pf(s['profit_factor'])} 勝率{s['win_rate']:.1f}% "
                     f"損益{s['total_pnl_ntd']:,.0f} 平倉回撤{s['max_drawdown_ntd']:,.0f} "
                     f"逐日市值回撤{s['mtm_mdd_ntd']:,.0f} boot+{s['bootstrap_pct_positive']:.1f}% "
                     f"拿掉前3筆{s['pnl_excl_top3_ntd']:,.0f} 平均持有{s['avg_hold_days']:.1f}天 "
                     f"在場{_f(s['time_in_market_pct'], 1)}% 成本加壓PF {_fmt_pf(s['stress_pf'])} "
                     f"獲利年{s['pos_years']}/{s['n_years']}")
        lines.append(f"    買進持有：損益{b['total_pnl_ntd']:,.0f} 逐日市值回撤{b['mtm_mdd_ntd']:,.0f} "
                     f"損益/回撤 {_f(b['ratio'])}")
    return lines


def refine_caveat_lines(ds, contract, n_variants):
    out = []
    for line in ctd.caveat_lines(ds, contract):
        if line.startswith("6."):
            line = (f"6. {n_variants}個變體 = 多重比較，而且候選清單是看過測試/近期結果後才定的(見最上面的警告)；"
                    "固定變體的測試/近期數字一定偏樂觀。")
        out.append(line)
    out.append("9. Walk-forward換變體時，前一年進場、隔年才出場的部位照原本規則出場，可能跟新變體的部位短暫重疊"
               "(實務上要多一口保證金)；年初已持有的部位不會被新變體接手。")
    out.append("10. 2024~用加權指數代理，walk-forward的最後幾年也是用指數代理算的。")
    return out


def build_refine_report(ds, res, variants, wf_rows, wf, fixed, bh_wf, years, axes, h2h, contract, commission):
    lines = [ds.get("source_line", ""), "",
             f"台指期日線策略 —— 精煉模式(MACD等5種候選，{len(variants)}變體，walk-forward驗證)", ""]
    lines += ctd.data_source_lines(ds) + [""]
    lines += refine_rules_lines(contract, commission, len(variants)) + [""]
    lines += hindsight_warning_lines(len(variants)) + [""]
    wl, passed = wf_section_lines(wf_rows, wf, fixed, bh_wf, years)
    lines += wl + [""]
    lines += axis_lines(axes) + [""]
    lines += h2h_lines(h2h) + [""]
    lines += period_overview_lines(variants, res["stats"], res["bh"]) + [""]
    t = res["timing"]
    lines.append(f"執行時間：指標/訊號{t['indicators_sec']:.1f}秒、回測{t['backtests_sec']:.1f}秒"
                 f"({t['n_runs']}次 = {len(variants)}變體×3期間×2(基本+成本加壓))、合計{t['total_sec']:.1f}秒")
    lines.append("")
    lines += refine_caveat_lines(ds, contract, len(variants))
    return lines, passed


# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------
def _r(x, nd=3):
    return round(float(x), nd) if np.isfinite(x) else float(x)


def all_rows(variants, stats, wf_rows):
    chosen = Counter(r["variant"]["name"] for r in wf_rows if r["variant"] is not None)
    rows = []
    for order, v in enumerate(variants):
        row = {"登記順序": order + 1, "變體": v["name"], "進場規則": v["rule"],
               "MACD參數": _macd_tag(v["macd"]) if v["macd"] else "", "方向": v["direction_label"],
               "初始停損(×ATR)": v["stop_atr"],
               "出場方式": (f"停利{v['exit_mult']:.1f}×ATR" if v["exit_kind"] == "target"
                        else f"移動停損{v['exit_mult']:.1f}×ATR")}
        for p in PERIOD_KEYS:
            s = stats[(v["name"], p)]
            lab = PERIOD_LABEL[p].replace("(指數代理)", "")
            row[f"{lab}交易數"] = s["trade_count"]
            row[f"{lab}PF"] = _r(s["profit_factor"])
            row[f"{lab}勝率(%)"] = round(s["win_rate"], 1)
            row[f"{lab}損益(NT$)"] = round(s["total_pnl_ntd"], 0)
            row[f"{lab}平倉最大回撤(NT$)"] = round(s["max_drawdown_ntd"], 0)
            row[f"{lab}逐日市值最大回撤(NT$)"] = round(s["mtm_mdd_ntd"], 0)
            row[f"{lab}拿掉前3筆損益(NT$)"] = round(s["pnl_excl_top3_ntd"], 0)
            row[f"{lab}bootstrap正報酬(%)"] = round(s["bootstrap_pct_positive"], 1)
            row[f"{lab}平均持有天數"] = round(s["avg_hold_days"], 1)
            row[f"{lab}在場時間比例(%)"] = _r(s["time_in_market_pct"], 1)
            row[f"{lab}多單損益(NT$)"] = round(s["long_pnl_ntd"], 0)
            row[f"{lab}空單損益(NT$)"] = round(s["short_pnl_ntd"], 0)
            row[f"{lab}成本加壓PF(滑價{STRESS_SLIPPAGE_POINTS:g}點)"] = _r(s["stress_pf"])
            row[f"{lab}獲利年度比例(%)"] = _r(s["pos_year_pct"], 1)
        row["訓練合格(>=40筆)"] = "是" if _eligible(stats, v) else "否"
        row["兩段PF都>1"] = "是" if grid._both_gt1(stats, v["name"]) else "否"
        row["walk-forward被選中年數"] = chosen.get(v["name"], 0)
        rows.append(row)
    return rows


WF_COLS = ["年度", "被選變體", "合格候選數", "歷史交易數(選擇時)", "歷史PF(選擇時)", "歷史損益(選擇時,NT$)",
           "當年交易數", "當年損益(NT$)", "當年PF", "換變體", "固定基準當年交易數", "固定基準當年損益(NT$)",
           "買進持有當年損益(逐日市值,NT$)"]


def wf_table_rows(wf_rows, fixed_trades, bh_wf):
    out = []
    prev = None
    fixed_by = {}
    for t in fixed_trades:
        fixed_by.setdefault(pd.Timestamp(t["entry_date"]).year, []).append(t["pnl_ntd"])
    for k, r in enumerate(wf_rows):
        name = r["variant"]["name"] if r["variant"] else ""
        pnl = [t["pnl_ntd"] for t in r["trades"]]
        fx = fixed_by.get(r["year"], [])
        out.append({"年度": r["year"], "被選變體": name, "合格候選數": r["n_candidates"],
                    "歷史交易數(選擇時)": r["hist_trades"], "歷史PF(選擇時)": _r(r["hist_pf"]),
                    "歷史損益(選擇時,NT$)": _r(r["hist_pnl"], 0), "當年交易數": len(pnl),
                    "當年損益(NT$)": round(sum(pnl), 0), "當年PF": _r(profit_factor(pnl)) if pnl else "",
                    "換變體": ("是" if name != prev else "否") if k > 0 else "",
                    "固定基準當年交易數": len(fx), "固定基準當年損益(NT$)": round(sum(fx), 0),
                    "買進持有當年損益(逐日市值,NT$)": round(bh_wf["yearly"].get(r["year"], 0.0), 0)})
        prev = name
    return out


def write_refine_outputs(out_dir, lines, variants, stats, wf_rows, wf_trades, fixed_trades, bh_wf, axes):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary_refine.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    pd.DataFrame(all_rows(variants, stats, wf_rows)).to_csv(
        os.path.join(out_dir, "tx_refine_all.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(wf_table_rows(wf_rows, fixed_trades, bh_wf), columns=WF_COLS).to_csv(
        os.path.join(out_dir, "tx_refine_walkforward.csv"), index=False, encoding="utf-8-sig")
    by_p = {p: [t for t in wf_trades if t["period"] == p] for p in PERIOD_KEYS}
    pd.DataFrame(grid.trade_rows(by_p), columns=grid.TRADE_COLS).to_csv(
        os.path.join(out_dir, "tx_refine_wf_trades.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(axes).to_csv(os.path.join(out_dir, "tx_refine_axes.csv"), index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
def evaluate(ds, variants, contract="mini", commission=DEFAULT_COMMISSION_PER_SIDE,
             n_boot=ctd.N_BOOTSTRAP, log=None, res=None):
    if res is None:
        res = run_refine(ds, variants, contract=contract, commission=commission, n_boot=n_boot, log=log)
    t0 = time.perf_counter()
    frames = res["frames"]
    years = wf_years(frames)
    by_v = combined_trades(variants, res["trades"])
    wf_rows = walk_forward(variants, by_v, years)
    wf_trades = [t for r in wf_rows for t in r["trades"]]
    wf = series_stats(wf_trades, frames, years, contract, n_boot)
    fixed_trades = [t for t in by_v.get(fixed_baseline_name(), []) if pd.Timestamp(t["entry_date"]).year >= WF_START_YEAR]
    fixed = series_stats(fixed_trades, frames, years, contract, n_boot)
    bh_wf = buy_hold_wf(frames, years, contract, commission)
    axes = axis_rows(variants, res["stats"], wf_rows)
    h2h = head_to_head(variants, res["stats"])
    res["timing"]["walkforward_sec"] = time.perf_counter() - t0
    return {"res": res, "years": years, "wf_rows": wf_rows, "wf_trades": wf_trades, "wf": wf,
            "fixed_trades": fixed_trades, "fixed": fixed, "bh_wf": bh_wf, "axes": axes, "h2h": h2h}


def run_refine_mode(args, ds):
    variants = refine_variant_list()
    t0 = time.perf_counter()
    out = evaluate(ds, variants, contract=args.contract, commission=args.commission_per_side,
                   n_boot=args.n_bootstrap, log=lambda m: print(m, flush=True))
    lines, passed = build_refine_report(ds, out["res"], variants, out["wf_rows"], out["wf"], out["fixed"],
                                        out["bh_wf"], out["years"], out["axes"], out["h2h"],
                                        args.contract, args.commission_per_side)
    lines.append(f"精煉模式總執行時間(不含資料下載)：{time.perf_counter() - t0:.1f}秒"
                 f"(walk-forward與診斷{out['res']['timing']['walkforward_sec']:.1f}秒)")
    write_refine_outputs(args.results_dir, lines, variants, out["res"]["stats"], out["wf_rows"], out["wf_trades"],
                         out["fixed_trades"], out["bh_wf"], out["axes"])
    print("\n".join(lines), flush=True)
    return 0
