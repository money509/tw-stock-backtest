"""
tx_daily_grid.py
====================
台指期(預設小台MTX)日線策略 —— 「網格模式」(compare_tx_daily.py --grid)。

1872個變體 = (12個單一訊號 + 66組兩兩混搭) × {無門檻, 動能分數門檻} × 12種出場
(訊號/混搭/出場定義見tx_daily_engine.py模組docstring下半段)。資料載入、驗證、備援
跟8變體模式共用同一份dataset(compare_tx_daily.load_dataset，只載一次)。

事先登記的評估規則(結果出來前就固定，報告最前面會印出來)：
  期間：訓練2001~2012、測試2013~2023、近期2024~(加權指數代理，只報告)；每段獨立重跑。
  選擇：只看訓練期。訓練交易數 >= 40 的變體中訓練PF最高者(平手 → 訓練損益 → 登記順序)。
        「訓練前20名」用同一個排序取前20。
  通過(被選中的那一個，全部都要成立)：
    測試PF > 1；測試bootstrap正報酬比例 > 80%；測試期拿掉最賺3筆後損益 > 0；
    高原：同一條進場規則+同一個門檻設定下，出場網格(3停損 × 4出場方式)上「相鄰」格子
    (停損差一格及/或出場方式差一格，含斜角；不含自己；角落3格、邊5格、中間8格)的
    測試PF平均 > 1(平均時PF上限截在10，避免某格「只賺不賠」的∞把平均灌爆)。
    出場方式的順序就是登記順序：停利2.0 → 停利3.0 → 停利4.0 → 移動停損3.0(移動停損排在
    停利4.0旁邊)。
  近期PF只報告、不參與判定(指數代理資料)。
  過度擬合診斷(一定印出)：訓練前20名在測試期的表現 vs 全體；訓練PF與測試PF的Spearman
    等級相關(不用scipy：先排名(同分取平均名次)再算Pearson)；測試PF分布(明講不能從裡面挑)；
    家族檢視(每個單一訊號 → 含它的所有變體裡訓練、測試PF都>1的比例)；出場檢視(每種出場
    設定的同一個比例)。分母一律是「訓練交易數 >= 40」的合格變體。
tx_grid_all.csv / tx_grid_top20.csv 每段另有「平均持有天數」(每筆交易持有交易日數的平均)與
「在場時間比例(%)」(持有天數加總 / 該段交易日數 × 100)，只報告、不參與選擇或判定。
"""
import os
import time

import numpy as np
import pandas as pd

import tx_daily_engine as eng
from tx_daily_engine import (
    CONTRACT_MULTIPLIER, DEFAULT_COMMISSION_PER_SIDE, FUTURES_TAX_RATE, SLIPPAGE_POINTS, STARTING_CAPITAL,
    MAX_HOLD_DAYS, ADX_GATE, GRID_STOP_MULTS, GRID_EXIT_OPTIONS, GRID_SIGNAL_NAMES,
)
import compare_tx_daily as ctd

TOP_N = 20
PLATEAU_PF_CAP = 10.0
PERIOD_KEYS = [p["key"] for p in ctd.PERIODS]
PERIOD_LABEL = {p["key"]: p["label"] for p in ctd.PERIODS}


# ---------------------------------------------------------------------------
# 規則文字
# ---------------------------------------------------------------------------
def grid_rules_lines(contract, commission, n_variants=None):
    mult = CONTRACT_MULTIPLIER[contract]
    name = "小台MTX" if contract == "mini" else "大台TX"
    n_variants = n_variants or len(eng.grid_variant_list())
    exits = "、".join(
        [f"停利{m:.1f}×ATR" if k == "target" else f"移動停損{m:.1f}×ATR(無停利)" for k, m in GRID_EXIT_OPTIONS])
    return [
        "== 預先登記的規則(網格模式；結果出來前就固定，不會依結果修改) ==",
        f"商品：{name}，每點NT${mult}，一次最多1口，可多可空，不反手",
        f"變體數：(12個單一訊號 + 66組混搭) × 動能門檻有/無 × 12種出場 = {n_variants}個",
        "單一訊號(第t天收盤後算，只用<=t的資料，第t+1天開盤進場；「穿越」= 今天在上(下)、昨天不在)：",
        "  MA_5_20：MA5上穿MA20 → 多、下穿 → 空；MA_20_60：MA20上穿/下穿MA60(=既有MA_CROSS)",
        "  MACD：MACD線(EMA12−EMA26)上穿/下穿訊號線(EMA9)；MACD_ZERO：MACD線上穿/下穿0軸",
        "    (原登記的「柱狀圖穿越0」跟MACD線穿越訊號線是同一件事，改成0軸穿越，避免重複變體)",
        "  BODY：實體/全距>=0.6 且 全距>=1.0×ATR14；紅K且收在全距最上25% → 多；黑K且收在最下25% → 空",
        "  RSI50：RSI(14)上穿/下穿50；DONCHIAN20 / DONCHIAN55：收盤突破前20 / 55天最高(最低)",
        "  BB_BREAK：收盤上穿布林上軌(20,2) → 多；下穿下軌 → 空",
        "  KDJ_CROSS：KDJ(9,3,3)，K = 1/3平滑RSV、D = 1/3平滑K；K上穿D且K<30 → 多；K下穿D且K>70 → 空",
        "  SQZ_RELEASE：擠壓解除，方向=動能符號(既有)；SQZ_KDJ：擠壓+KDJ關鍵K棒(既有，只做多)",
        "混搭(A&B，66組)：第t天方向d ⇔ 一個在第t天發出d、另一個在[t−2, t]內也發出過d；含SQZ_KDJ的只會做多",
        f"動能分數門檻(+動能門檻)：ADX(14) >= {ADX_GATE:g} 且 均線排列同向；只過濾進場",
        "出場網格：初始停損 {1.0, 1.5, 2.0}×ATR14 × {" + exits + "}",
        "  停利 = 限價單：觸及以停利價成交(不扣滑價)，開盤跳空越過以開盤價成交；同一天同時碰到停損與停利 → 當作先停損",
        f"  共同：反向原始訊號(同一條進場規則、未過濾) → 隔天開盤出場；持有滿{MAX_HOLD_DAYS}天 → 隔天開盤出場；"
        "ATR取訊號日的值",
        f"成本：手續費每邊NT${commission:g} + 期交稅每邊{FUTURES_TAX_RATE*100:.3f}%(未調整價格)；"
        f"開盤/停損成交滑價{SLIPPAGE_POINTS:g}點；換月時持倉 → 多收一次來回手續費+稅",
        "期間：訓練2001~2012、測試2013~2023、近期2024~最新(指數代理，只報告)；每段獨立重跑，期末強制平倉",
        f"選擇：只看訓練期，交易數>={ctd.MIN_TRAIN_TRADES}的變體中PF最高(平手→訓練損益→登記順序)；"
        f"同一排序的前{TOP_N}名 = 「訓練前{TOP_N}名」",
        f"通過(全部成立)：測試PF>1；測試bootstrap正報酬比例>{ctd.BOOTSTRAP_PASS_PCT:g}%；測試拿掉最賺3筆後損益>0；"
        f"高原：同規則同門檻、出場網格相鄰格(含斜角、不含自己)的測試PF平均>1(PF上限截在{PLATEAU_PF_CAP:g})",
        "近期PF只報告、不參與判定",
        "過度擬合診斷(一定印出)：訓練前20名的測試表現、訓練PF vs 測試PF的Spearman等級相關、測試PF分布、"
        "家族檢視(每個訊號)、出場檢視(每種出場)；分母 = 訓練交易數>=40的合格變體",
        f"起始資金NT${STARTING_CAPITAL:,}",
    ]


# ---------------------------------------------------------------------------
# 跑網格
# ---------------------------------------------------------------------------
def period_n_days(dates, start, end):
    """[start, end]內的交易日數(跟回測引擎切區間的方式完全相同)。"""
    lo = 0 if start is None else int(dates.searchsorted(pd.Timestamp(start), side="left"))
    hi = len(dates) - 1 if end is None else int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    return max(0, hi - lo + 1)


def hold_stats(trades, n_days=None):
    """平均持有天數(每筆trade的hold_days平均) 與 在場時間比例(%) = 持有天數加總 / 期間交易日數 × 100。
    一次只有1口、出場當天不會再進場，所以持有天數不會重疊(<=100%)。沒給n_days → 比例為NaN。"""
    hold = np.array([t["hold_days"] for t in trades], float)
    avg = float(hold.mean()) if len(hold) else 0.0
    if not n_days:
        tim = float("nan")
    else:
        tim = float(hold.sum() / n_days * 100)
    return {"avg_hold_days": avg, "time_in_market_pct": tim}


def quick_stats(trades, n_days=None):
    """網格每個(變體, 期間)用的輕量統計(PF定義跟summarize_mr相同；沒有bootstrap)。
    n_days = 期間交易日數(算在場時間比例用)。"""
    n = len(trades)
    if n == 0:
        out = {"trade_count": 0, "profit_factor": 0.0, "total_pnl_ntd": 0.0, "win_rate": 0.0,
               "max_drawdown_ntd": 0.0, "long_count": 0, "short_count": 0, "long_pnl_ntd": 0.0,
               "short_pnl_ntd": 0.0, "pnl_excl_top3_ntd": 0.0}
        out.update(hold_stats(trades, n_days))
        return out
    pnl = np.array([t["pnl_ntd"] for t in trades], float)
    is_long = np.array([t["side"] == "long" for t in trades], bool)
    gw = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    pf = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
    eq = STARTING_CAPITAL + np.concatenate([[0.0], np.cumsum(pnl)])
    dd = float((eq - np.maximum.accumulate(eq)).min())
    out = {"trade_count": n, "profit_factor": pf, "total_pnl_ntd": float(pnl.sum()),
           "win_rate": float((pnl > 0).mean() * 100), "max_drawdown_ntd": dd,
           "long_count": int(is_long.sum()), "short_count": int((~is_long).sum()),
           "long_pnl_ntd": float(pnl[is_long].sum()), "short_pnl_ntd": float(pnl[~is_long].sum()),
           "pnl_excl_top3_ntd": float(np.sort(pnl)[::-1][3:].sum())}
    out.update(hold_stats(trades, n_days))
    return out


def grid_period_frames(ds):
    """{period_key: (arrs, label)}；訓練/測試共用ds["main"](指標只算一次)，近期一律加權指數。"""
    main = ds["main"]
    arrs_main = eng.prepare_grid_arrays(main, eng.compute_grid_indicators(main))
    if ds["source"] == "index":
        arrs_index = arrs_main
    else:
        arrs_index = eng.prepare_grid_arrays(ds["index"], eng.compute_grid_indicators(ds["index"]))
    main_label = "台指期(社群分鐘資料，換月調整)" if ds["source"] == "futures" else "加權指數(備援)"
    return {"train": (arrs_main, main_label), "test": (arrs_main, main_label),
            "recent": (arrs_index, "加權指數代理")}


def _period_bounds(key):
    p = [p for p in ctd.PERIODS if p["key"] == key][0]
    return p["start"], p["end"]


def run_variant(frames, v, period, contract="mini", commission=DEFAULT_COMMISSION_PER_SIDE):
    arrs, _ = frames[period]
    es, rs = eng.grid_entry_signals(arrs, v["rule"], v["gate"])
    start, end = _period_bounds(period)
    return eng.run_tx_daily_grid_backtest(arrs, es, rs, v["stop_atr"], v["exit_kind"], v["exit_mult"],
                                          start=start, end=end, contract=contract,
                                          commission_per_side=commission, variant_name=v["name"])


def run_grid(ds, variants, contract="mini", commission=DEFAULT_COMMISSION_PER_SIDE, log=None):
    """回傳 {"stats": {(name, period): quick_stats}, "frames", "timing": {...}}。
    進場訊號序列每個(規則, 門檻, frame)只算一次，12種出場共用。"""
    t0 = time.perf_counter()
    frames = grid_period_frames(ds)
    t_ind = time.perf_counter() - t0
    stats = {}
    by_rule_gate = {}
    for v in variants:
        by_rule_gate.setdefault((v["rule"], v["gate"]), []).append(v)
    t1 = time.perf_counter()
    n_runs = 0
    for period in PERIOD_KEYS:
        arrs, _ = frames[period]
        start, end = _period_bounds(period)
        n_days = period_n_days(arrs["dates"], start, end)
        tp = time.perf_counter()
        for (rule, gate), vs in by_rule_gate.items():
            es, rs = eng.grid_entry_signals(arrs, rule, gate)
            for v in vs:
                tr = eng.run_tx_daily_grid_backtest(arrs, es, rs, v["stop_atr"], v["exit_kind"], v["exit_mult"],
                                                    start=start, end=end, contract=contract,
                                                    commission_per_side=commission, variant_name=v["name"])
                stats[(v["name"], period)] = quick_stats(tr, n_days)
                n_runs += 1
        if log:
            log(f"  網格：{PERIOD_LABEL[period]}段 {len(variants)}個變體完成，{time.perf_counter() - tp:.1f}秒")
    timing = {"indicators_sec": t_ind, "backtests_sec": time.perf_counter() - t1,
              "total_sec": time.perf_counter() - t0, "n_runs": n_runs}
    return {"stats": stats, "frames": frames, "timing": timing}


# ---------------------------------------------------------------------------
# 選擇 / 高原 / 診斷
# ---------------------------------------------------------------------------
def eligible_variants(variants, stats, min_trades=ctd.MIN_TRAIN_TRADES):
    return [v for v in variants if stats[(v["name"], "train")]["trade_count"] >= min_trades]


def train_ranking(variants, stats, min_trades=ctd.MIN_TRAIN_TRADES):
    """只看訓練期：合格變體依(訓練PF, 訓練損益, −登記順序)由大到小排序。回傳變體list。"""
    keyed = []
    for order, v in enumerate(variants):
        s = stats[(v["name"], "train")]
        if s["trade_count"] < min_trades:
            continue
        keyed.append(((s["profit_factor"], s["total_pnl_ntd"], -order), v))
    keyed.sort(key=lambda kv: kv[0], reverse=True)
    return [v for _, v in keyed]


def select_grid_variant(variants, stats, min_trades=ctd.MIN_TRAIN_TRADES):
    """跟8變體模式同一個選擇函式(compare_tx_daily.select_variant)，只看訓練期。"""
    return ctd.select_variant(variants, stats, min_trades=min_trades)


def exit_neighbor_cells(stop_idx, exit_idx, n_stops=len(GRID_STOP_MULTS), n_exits=len(GRID_EXIT_OPTIONS)):
    """出場網格上的相鄰格(含斜角、不含自己)：[(stop_idx, exit_idx), ...]，順序固定。"""
    out = []
    for ds_ in (-1, 0, 1):
        for de in (-1, 0, 1):
            if ds_ == 0 and de == 0:
                continue
            si, ei = stop_idx + ds_, exit_idx + de
            if 0 <= si < n_stops and 0 <= ei < n_exits:
                out.append((si, ei))
    return out


def _cell_index(variants):
    return {(v["rule"], v["gate"], v["stop_idx"], v["exit_idx"]): v for v in variants}


def neighbor_names(v, cell_index):
    return [cell_index[(v["rule"], v["gate"], si, ei)]["name"]
            for si, ei in exit_neighbor_cells(v["stop_idx"], v["exit_idx"])
            if (v["rule"], v["gate"], si, ei) in cell_index]


def capped_pf(pf, cap=PLATEAU_PF_CAP):
    return min(float(pf), cap)


def neighbor_mean_pf(v, cell_index, stats, period="test"):
    names = neighbor_names(v, cell_index)
    if not names:
        return float("nan"), names
    return float(np.mean([capped_pf(stats[(n, period)]["profit_factor"]) for n in names])), names


def grid_verdict(test_full_stats, nb_mean_pf):
    """8變體的三項 + 高原。回傳(passed, checks)。"""
    passed, checks = ctd.verdict(test_full_stats)
    ok = bool(np.isfinite(nb_mean_pf) and nb_mean_pf > 1)
    checks = list(checks) + [(f"高原：出場網格相鄰格的測試PF平均 {nb_mean_pf:.2f} > 1", ok)]
    return all(c for _, c in checks), checks


def rankdata_avg(x):
    """同分取平均名次(1起算)，跟scipy.stats.rankdata(method='average')一樣，但不用scipy。"""
    x = np.asarray(x, float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (xs[j + 1] == xs[i]):
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman等級相關 = 兩邊各自排名(同分平均)後的Pearson相關；不用scipy。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata_avg(x), rankdata_avg(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _both_gt1(stats, name):
    return stats[(name, "train")]["profit_factor"] > 1 and stats[(name, "test")]["profit_factor"] > 1


def _share_row(vs_all, stats, min_trades):
    elig = [v for v in vs_all if stats[(v["name"], "train")]["trade_count"] >= min_trades]
    both = [v for v in elig if _both_gt1(stats, v["name"])]
    test_pfs = [stats[(v["name"], "test")]["profit_factor"] for v in elig]
    return {"變體數(全部)": len(vs_all), "合格變體數(訓練>=40筆)": len(elig),
            "兩段PF都>1的數量": len(both),
            "兩段PF都>1比例(%)": round(len(both) / len(elig) * 100, 1) if elig else float("nan"),
            "測試PF中位數": round(float(np.median(test_pfs)), 3) if test_pfs else float("nan")}


def family_rows(variants, stats, min_trades=ctd.MIN_TRAIN_TRADES):
    """每個單一訊號：含它的全部變體(單一+11組混搭、有無門檻、12出場 = 288個)。"""
    rows = []
    for s in GRID_SIGNAL_NAMES:
        vs = [v for v in variants if s in v["components"]]
        row = {"訊號": s}
        row.update(_share_row(vs, stats, min_trades))
        single = _share_row([v for v in vs if len(v["components"]) == 1], stats, min_trades)
        pair = _share_row([v for v in vs if len(v["components"]) == 2], stats, min_trades)
        row["單獨使用：兩段PF>1比例(%)"] = single["兩段PF都>1比例(%)"]
        row["單獨使用：合格數"] = single["合格變體數(訓練>=40筆)"]
        row["混搭：兩段PF>1比例(%)"] = pair["兩段PF都>1比例(%)"]
        row["混搭：合格數"] = pair["合格變體數(訓練>=40筆)"]
        rows.append(row)
    return rows


def exit_rows(variants, stats, min_trades=ctd.MIN_TRAIN_TRADES):
    rows = []
    for ex in eng.grid_exit_configs():
        vs = [v for v in variants if v["exit_id"] == ex["exit_id"]]
        row = {"出場設定": ex["label"], "代號": ex["exit_id"]}
        row.update(_share_row(vs, stats, min_trades))
        rows.append(row)
    return rows


def gate_rows(variants, stats, min_trades=ctd.MIN_TRAIN_TRADES):
    rows = []
    for g in (False, True):
        row = {"門檻": "動能門檻" if g else "無門檻"}
        row.update(_share_row([v for v in variants if v["gate"] == g], stats, min_trades))
        rows.append(row)
    return rows


def overfit_diagnostics(variants, stats, min_trades=ctd.MIN_TRAIN_TRADES, top_n=TOP_N):
    elig = eligible_variants(variants, stats, min_trades)
    ranking = train_ranking(variants, stats, min_trades)
    top = ranking[:top_n]
    tr_pf = np.array([stats[(v["name"], "train")]["profit_factor"] for v in elig], float)
    te_pf = np.array([stats[(v["name"], "test")]["profit_factor"] for v in elig], float)
    top_te = np.array([stats[(v["name"], "test")]["profit_factor"] for v in top], float)
    d = {"n_variants": len(variants), "n_eligible": len(elig), "n_top": len(top),
         "top_test_gt1": int((top_te > 1).sum()),
         "top_median_test_pf": float(np.median(top_te)) if len(top_te) else float("nan"),
         "all_median_test_pf": float(np.median(te_pf)) if len(te_pf) else float("nan"),
         "spearman": spearman(tr_pf, te_pf) if len(elig) >= 3 else float("nan"),
         "test_pct_gt1": float((te_pf > 1).mean() * 100) if len(te_pf) else float("nan"),
         "test_q25": float(np.percentile(te_pf[np.isfinite(te_pf)], 25)) if np.isfinite(te_pf).any() else float("nan"),
         "test_q75": float(np.percentile(te_pf[np.isfinite(te_pf)], 75)) if np.isfinite(te_pf).any() else float("nan"),
         "both_gt1_pct": (float(np.mean([(a > 1) and (b > 1) for a, b in zip(tr_pf, te_pf)]) * 100)
                          if len(elig) else float("nan")),
         "top": top, "ranking": ranking}
    return d


# ---------------------------------------------------------------------------
# 報告
# ---------------------------------------------------------------------------
_fmt_pf = ctd._fmt_pf


def _f(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else (
        "∞" if np.isinf(x) else f"{x:.{nd}f}")


def diagnostics_lines(diag, fam, exits, gates):
    n, ne = diag["n_variants"], diag["n_eligible"]
    lines = ["== 過度擬合診斷(請先看這段，再看被選中的變體) =="]
    lines.append(f"這次一共測了{n}個變體，其中{ne}個訓練期交易數>={ctd.MIN_TRAIN_TRADES}筆、可以參加選擇。")
    lines.append(f"從{ne}個裡面挑訓練期PF最高的那一個，它的「訓練期」數字一定被挑選動作灌水(偏高)——"
                 "就算全部變體都沒有真實優勢，最幸運的那個訓練PF也會很好看。訓練期數字不能當成預期績效，"
                 "只能看測試期。")
    if ne == 0:
        lines.append("沒有任何合格變體，以下診斷無法計算。")
        return lines
    lines.append("")
    lines.append(f"1. 訓練前{diag['n_top']}名到了測試期：{diag['top_test_gt1']}/{diag['n_top']}個測試PF>1；"
                 f"前{diag['n_top']}名測試PF中位數 {_f(diag['top_median_test_pf'])} vs "
                 f"全部合格變體測試PF中位數 {_f(diag['all_median_test_pf'])}")
    if diag["top_median_test_pf"] > diag["all_median_test_pf"]:
        lines.append("   → 訓練期排名前面的，測試期中位數比全體高：訓練排名有一點延續性(仍要看下面的相關係數)。")
    else:
        lines.append("   → 訓練期排名前面的，測試期並沒有比全體好：訓練期排名幾乎沒有預測力，"
                     "挑訓練冠軍跟隨便挑一個差不多。")
    r = diag["spearman"]
    lines.append(f"2. 訓練PF vs 測試PF 的Spearman等級相關 = {_f(r, 3)}({ne}個合格變體；不用scipy，排名後算Pearson)")
    if not np.isfinite(r):
        lines.append("   → 無法計算。")
    elif r < 0.1:
        lines.append("   → 接近0或負：訓練期表現好，不代表測試期也好(典型的過度擬合現象)。")
    elif r < 0.3:
        lines.append("   → 弱正相關：有一點點延續性，但很弱，單一變體的排名不可靠。")
    else:
        lines.append("   → 中等以上正相關：訓練期好的變體，測試期也傾向比較好(但仍不保證任何單一變體)。")
    lines.append(f"3. 全部合格變體的測試PF分布：PF>1佔{_f(diag['test_pct_gt1'], 1)}%、中位數{_f(diag['all_median_test_pf'])}、"
                 f"25%分位{_f(diag['test_q25'])}、75%分位{_f(diag['test_q75'])}")
    lines.append("   ⚠️ 這是事後看全體的分布，只用來了解「整體有沒有優勢」。不要從裡面挑測試期最好的變體來用——"
                 "那等於拿測試期當訓練期，測試期就不再是獨立的檢驗了。")
    lines.append(f"   兩段(訓練、測試)PF都>1的合格變體佔{_f(diag['both_gt1_pct'], 1)}%(下面家族/出場檢視的比較基準)")
    lines.append("")
    lines.append("4. 家族檢視(每個單一訊號：含它的全部變體 = 單獨使用 + 跟其他11個混搭、有無門檻、12種出場；"
                 "看「訓練、測試PF都>1」的比例 —— 比任何單一最佳格子更能看出哪個積木一直有用)：")
    lines.append(f"   {'訊號':<13}{'合格/全部':>10}{'兩段>1':>8}{'比例%':>8}{'測試PF中位':>10}"
                 f"{'單獨%':>8}{'混搭%':>8}")
    for r_ in fam:
        lines.append(f"   {r_['訊號']:<13}{str(r_['合格變體數(訓練>=40筆)']) + '/' + str(r_['變體數(全部)']):>10}"
                     f"{r_['兩段PF都>1的數量']:>8}{_f(r_['兩段PF都>1比例(%)'], 1):>8}{_f(r_['測試PF中位數']):>10}"
                     f"{_f(r_['單獨使用：兩段PF>1比例(%)'], 1):>8}{_f(r_['混搭：兩段PF>1比例(%)'], 1):>8}")
    lines.append(f"   (比較基準：全部合格變體{_f(diag['both_gt1_pct'], 1)}%；同一個變體會出現在兩個訊號的家族裡)")
    lines.append("5. 出場檢視(每種出場設定，橫跨所有訊號/門檻)：")
    lines.append(f"   {'出場設定':<16}{'合格/全部':>10}{'兩段>1':>8}{'比例%':>8}{'測試PF中位':>10}")
    for r_ in exits:
        lines.append(f"   {r_['出場設定']:<16}{str(r_['合格變體數(訓練>=40筆)']) + '/' + str(r_['變體數(全部)']):>10}"
                     f"{r_['兩段PF都>1的數量']:>8}{_f(r_['兩段PF都>1比例(%)'], 1):>8}{_f(r_['測試PF中位數']):>10}")
    lines.append("   門檻檢視：" + "；".join(
        f"{g['門檻']} 合格{g['合格變體數(訓練>=40筆)']}個、兩段>1 {_f(g['兩段PF都>1比例(%)'], 1)}%、"
        f"測試PF中位數{_f(g['測試PF中位數'])}" for g in gates))
    return lines


def exit_breakdown_lines(trades_by_period):
    lines = []
    for p in PERIOD_KEYS:
        tr = trades_by_period.get(p) or []
        if not tr:
            lines.append(f"  {PERIOD_LABEL[p]}：沒有交易")
            continue
        df = pd.DataFrame(tr)
        parts = []
        for reason, g in df.groupby("exit_reason", sort=False):
            parts.append(f"{reason}{len(g)}筆/損益{g['pnl_ntd'].sum():,.0f}/勝率{(g['pnl_ntd'] > 0).mean() * 100:.0f}%")
        lines.append(f"  {PERIOD_LABEL[p]}：" + "；".join(parts))
    return lines


def top_table_lines(top, stats):
    head = (f"{'#':>3} {'變體':<44}{'訓練PF':>8}{'筆':>5}{'損益':>11}  {'測試PF':>8}{'筆':>5}{'損益':>11}  "
            f"{'近期PF':>8}{'筆':>5}{'損益':>10}")
    lines = [head, "-" * len(head)]
    for k, v in enumerate(top, 1):
        tr, te, re_ = (stats[(v["name"], p)] for p in PERIOD_KEYS)
        lines.append(f"{k:>3} {v['name']:<44}{_fmt_pf(tr['profit_factor']):>8}{tr['trade_count']:>5}"
                     f"{tr['total_pnl_ntd']:>11,.0f}  {_fmt_pf(te['profit_factor']):>8}{te['trade_count']:>5}"
                     f"{te['total_pnl_ntd']:>11,.0f}  {_fmt_pf(re_['profit_factor']):>8}{re_['trade_count']:>5}"
                     f"{re_['total_pnl_ntd']:>10,.0f}")
    return lines


def grid_caveat_lines(ds, contract, n_variants):
    out = []
    for line in ctd.caveat_lines(ds, contract):
        if line.startswith("6."):
            line = (f"6. {n_variants}個變體 = 非常大量的多重比較：就算全部都沒有真實優勢，從裡面挑訓練期最好的，"
                    "測試期剛好過關的機率也不低；被選中變體的訓練期數字一定偏高。請以過度擬合診斷、家族/出場檢視"
                    "為主要判讀依據，不要從測試期結果裡挑變體。")
        out.append(line)
    out.append("9. 混搭訊號的三天視窗、各訊號參數(MACD 12/26/9、RSI14、KDJ 9/3/3、BODY門檻)都是事先登記的"
               "常見預設值，沒有針對台指期調過；調參數本身就是另一次多重比較。")
    out.append("10. 變體之間高度相關：例如DONCHIAN20&DONCHIAN55幾乎等於DONCHIAN55(收盤突破55日高通常也突破20日高)、"
               "同一訊號的12種出場共用同一批進場點；實際上獨立的檢驗次數比變體數少，但仍然非常多。")
    return out


def build_grid_report(ds, res, variants, selected, sel_info, diag, fam, exits, gates, contract, commission):
    stats = res["stats"]
    lines = [ds.get("source_line", ""), "",
             f"台指期日線策略 —— 網格模式預先登記測試({len(variants)}變體 × 訓練/測試/近期)", ""]
    lines += grid_rules_lines(contract, commission, len(variants)) + [""]
    lines += ctd.data_source_lines(ds) + [""]
    lines.append("== 選擇與判定 ==")
    lines.append(f"一共測試{len(variants)}個變體；被選中變體的訓練期數字是從{diag['n_eligible']}個合格變體裡挑出來的最大值，"
                 "一定偏高(選擇偏誤)，請只看測試期。")
    passed = False
    if selected is None:
        lines.append(f"沒有任何變體在訓練期達到{ctd.MIN_TRAIN_TRADES}筆交易 → 無法選擇，判定：未通過")
    else:
        tr = stats[(selected["name"], "train")]
        lines.append(f"訓練期選出：{selected['name']}(訓練PF {_fmt_pf(tr['profit_factor'])}，{tr['trade_count']}筆，"
                     f"損益{tr['total_pnl_ntd']:,.0f}；偏高)")
        passed, checks = grid_verdict(sel_info["full"]["test"], sel_info["nb_mean_pf"])
        for desc, ok in checks:
            lines.append(f"  [{'✓' if ok else '✗'}] {desc}")
        nb = "、".join(f"{n.split('｜')[1]} PF {_fmt_pf(stats[(n, 'test')]['profit_factor'])}"
                       for n in sel_info["neighbors"])
        lines.append(f"  (相鄰格測試PF：{nb})")
        lines.append(f"判定：{'通過' if passed else '未通過'}")
        rs = sel_info["full"]["recent"]
        lines.append(f"近期(指數代理，只參考)：{rs['trade_count']}筆 PF {_fmt_pf(rs['profit_factor'])} "
                     f"損益{rs['total_pnl_ntd']:,.0f}")
    lines.append("")
    lines += diagnostics_lines(diag, fam, exits, gates)
    if selected is not None:
        lines.append("")
        lines.append(f"== 被選中變體完整明細({selected['name']}) ==")
        for p in PERIOD_KEYS:
            s = sel_info["full"][p]
            lines.append(
                f"  {PERIOD_LABEL[p]}：{s['trade_count']}筆 PF {_fmt_pf(s['profit_factor'])} 勝率{s['win_rate']:.1f}% "
                f"均賺{s['avg_win_ntd']:,.0f}/均賠{s['avg_loss_ntd']:,.0f} 損益{s['total_pnl_ntd']:,.0f} "
                f"最大回撤{s['max_drawdown_ntd']:,.0f}({s['max_drawdown_pct']:.1f}%資金) 最長連虧{s['max_consecutive_losses']} "
                f"bootstrap+{s['bootstrap_pct_positive']:.1f}% 拿掉前3筆{s['pnl_excl_top3_ntd']:,.0f} "
                f"換月成本{s['total_roll_cost_ntd']:,.0f}")
            lines.append(
                f"    多單{s['long_count']}筆 PF {_fmt_pf(s['long_pf'])} 損益{s['long_pnl_ntd']:,.0f}；"
                f"空單{s['short_count']}筆 PF {_fmt_pf(s['short_pf'])} 損益{s['short_pnl_ntd']:,.0f}")
        lines.append("  出場原因：")
        lines += exit_breakdown_lines(sel_info["trades"])
        lines.append("  逐年：")
        yr = ctd.yearly_rows({(selected["name"], p): sel_info["trades"][p] for p in PERIOD_KEYS})
        for r in sorted(yr, key=lambda r: (r["年度"], PERIOD_KEYS.index(r["期間"]))):
            lines.append(f"    {r['年度']} [{r['期間']}] {r['交易數']}筆 損益{r['損益(NT$)']:,.0f} "
                         f"勝率{r['勝率(%)']}% PF {_fmt_pf(r['PF'])}")
    lines.append("")
    lines.append(f"== 訓練前{TOP_N}名(依訓練PF排序；測試/近期欄位只是讓你看「訓練名次會不會延續」，不是讓你重挑) ==")
    lines += top_table_lines(diag["top"], stats)
    lines.append("")
    t = res["timing"]
    lines.append(f"執行時間：指標/訊號{t['indicators_sec']:.1f}秒、回測{t['backtests_sec']:.1f}秒"
                 f"({t['n_runs']}次 = {len(variants)}變體×3期間)、網格合計{t['total_sec']:.1f}秒")
    lines.append("")
    lines += grid_caveat_lines(ds, contract, len(variants))
    return lines, passed


# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------
def _r(x, nd=3):
    return round(float(x), nd) if np.isfinite(x) else float(x)


def all_rows(variants, stats, cell_index, rank_of):
    rows = []
    for order, v in enumerate(variants):
        comps = v["components"]
        row = {"登記順序": order + 1, "變體": v["name"], "進場規則": v["rule"],
               "類型": "單一" if len(comps) == 1 else "混搭", "訊號A": comps[0],
               "訊號B": comps[1] if len(comps) == 2 else "", "動能門檻": "是" if v["gate"] else "否",
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
            row[f"{lab}最大回撤(NT$)"] = round(s["max_drawdown_ntd"], 0)
            row[f"{lab}多單損益(NT$)"] = round(s["long_pnl_ntd"], 0)
            row[f"{lab}空單損益(NT$)"] = round(s["short_pnl_ntd"], 0)
            row[f"{lab}拿掉前3筆損益(NT$)"] = round(s["pnl_excl_top3_ntd"], 0)
            row[f"{lab}平均持有天數"] = round(s.get("avg_hold_days", float("nan")), 1)
            row[f"{lab}在場時間比例(%)"] = round(s.get("time_in_market_pct", float("nan")), 1)
        row["訓練合格(>=40筆)"] = "是" if stats[(v["name"], "train")]["trade_count"] >= ctd.MIN_TRAIN_TRADES else "否"
        row["訓練排名"] = rank_of.get(v["name"], "")
        row["兩段PF都>1"] = "是" if _both_gt1(stats, v["name"]) else "否"
        nb, _ = neighbor_mean_pf(v, cell_index, stats, "test")
        row["相鄰出場格測試PF平均"] = _r(nb)
        rows.append(row)
    return rows


TRADE_COLS = ["變體", "期間", "方向", "訊號日", "進場日", "出場日", "進場價(調整後)", "出場價(調整後)",
              "進場價(未調整)", "出場價(未調整)", "初始停損", "停利價", "點數", "換月次數", "換月成本(NT$)",
              "總成本(NT$)", "損益(NT$)", "持有天數", "出場原因"]


def trade_rows(trades_by_period):
    out = []
    for p in PERIOD_KEYS:
        for t in trades_by_period.get(p) or []:
            out.append({
                "變體": t["variant"], "期間": p, "方向": "多" if t["side"] == "long" else "空",
                "訊號日": str(t["signal_date"])[:10], "進場日": str(t["entry_date"])[:10],
                "出場日": str(t["exit_date"])[:10], "進場價(調整後)": round(t["entry_price"], 2),
                "出場價(調整後)": round(t["exit_price"], 2), "進場價(未調整)": round(t["raw_entry_price"], 2),
                "出場價(未調整)": round(t["raw_exit_price"], 2), "初始停損": round(t["initial_stop"], 2),
                "停利價": round(t["target"], 2) if np.isfinite(t["target"]) else "",
                "點數": round(t["points"], 2), "換月次數": t["n_rolls"],
                "換月成本(NT$)": round(t["roll_cost_ntd"], 1), "總成本(NT$)": round(t["cost_ntd"], 1),
                "損益(NT$)": round(t["pnl_ntd"], 1), "持有天數": t["hold_days"], "出場原因": t["exit_reason"],
            })
    return out


def write_grid_outputs(out_dir, lines, variants, stats, diag, fam, exits, sel_info):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary_grid.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    cell_index = _cell_index(variants)
    rank_of = {v["name"]: k for k, v in enumerate(diag["ranking"], 1)}
    rows = all_rows(variants, stats, cell_index, rank_of)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "tx_grid_all.csv"), index=False, encoding="utf-8-sig")
    top_names = [v["name"] for v in diag["top"]]
    by_name = {r["變體"]: r for r in rows}
    top_df = pd.DataFrame([by_name[n] for n in top_names], columns=list(rows[0].keys()))
    top_df.insert(0, "訓練名次", range(1, len(top_df) + 1))
    top_df.to_csv(os.path.join(out_dir, "tx_grid_top20.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fam).to_csv(os.path.join(out_dir, "tx_grid_family.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(exits).to_csv(os.path.join(out_dir, "tx_grid_exits.csv"), index=False, encoding="utf-8-sig")
    tr = trade_rows(sel_info["trades"]) if sel_info else []
    pd.DataFrame(tr, columns=TRADE_COLS).to_csv(os.path.join(out_dir, "tx_grid_selected_trades.csv"),
                                                index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
def evaluate(ds, variants, contract="mini", commission=DEFAULT_COMMISSION_PER_SIDE,
             n_boot=ctd.N_BOOTSTRAP, log=None, res=None):
    """跑網格(或沿用res) → 選擇 → 被選中變體完整重跑 → 診斷。回傳dict。"""
    if res is None:
        res = run_grid(ds, variants, contract=contract, commission=commission, log=log)
    stats = res["stats"]
    name = select_grid_variant(variants, stats)
    selected = next((v for v in variants if v["name"] == name), None) if name else None
    sel_info = None
    if selected is not None:
        trades = {p: run_variant(res["frames"], selected, p, contract, commission) for p in PERIOD_KEYS}
        full = {p: ctd.compute_stats(trades[p], n_boot=n_boot) for p in PERIOD_KEYS}
        nb_mean, nb_names = neighbor_mean_pf(selected, _cell_index(variants), stats, "test")
        sel_info = {"trades": trades, "full": full, "nb_mean_pf": nb_mean, "neighbors": nb_names}
    diag = overfit_diagnostics(variants, stats)
    fam = family_rows(variants, stats)
    exits = exit_rows(variants, stats)
    gates = gate_rows(variants, stats)
    return {"res": res, "selected": selected, "sel_info": sel_info, "diag": diag,
            "family": fam, "exits": exits, "gates": gates}


def run_grid_mode(args, ds):
    variants = eng.grid_variant_list()
    t0 = time.perf_counter()
    out = evaluate(ds, variants, contract=args.contract, commission=args.commission_per_side,
                   n_boot=args.n_bootstrap, log=lambda m: print(m, flush=True))
    lines, passed = build_grid_report(ds, out["res"], variants, out["selected"], out["sel_info"], out["diag"],
                                      out["family"], out["exits"], out["gates"], args.contract,
                                      args.commission_per_side)
    lines.append(f"網格模式總執行時間(不含資料下載)：{time.perf_counter() - t0:.1f}秒")
    write_grid_outputs(args.results_dir, lines, variants, out["res"]["stats"], out["diag"], out["family"],
                       out["exits"], out["sel_info"])
    print("\n".join(lines), flush=True)
    return 0
