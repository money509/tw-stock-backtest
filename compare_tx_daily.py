"""
compare_tx_daily.py
====================
台指期(預設小台MTX)日線策略 —— 預先登記(pre-registered)的小型測試。

8個變體 = {DONCHIAN, MA_CROSS, SQZ_KDJ, SQZ_RELEASE} × {無門檻, 動能分數門檻}
(訊號/出場/成本定義見tx_daily_engine.py模組docstring，資料見tx_daily_data.py)。

期間(每段獨立重跑：期初空手、期末強制平倉；指標用該段之前的歷史暖身)：
  訓練 2001-01-01 ~ 2012-12-31
  測試 2013-01-01 ~ 2023-12-31
  近期 2024-01-01 ~ 最新(加權指數代理，只報告、不參與選擇或判定)
選擇(只看訓練期)：訓練期交易數 >= 40 的變體中，訓練PF最高者；平手 → 訓練損益高者 → 列表順序。
通過條件(只看被選中變體的測試期)：PF > 1 且 bootstrap正報酬比例 > 80% 且 拿掉最賺3筆後損益 > 0。

輸出(results_tx_daily/)：summary.txt、tx_daily_summary.csv、tx_daily_yearly.csv、
tx_daily_trades.csv，以及tx_daily_data.write_diagnostics()寫的診斷檔。

--grid：改跑網格模式(1872變體，見tx_daily_grid.py)，資料載入/驗證/備援跟這裡共用同一份；
輸出summary_grid.txt與tx_grid_*.csv(不會寫summary.txt；不加--grid時行為跟以前完全一樣)。
"""
import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

import tx_daily_data
from tx_daily_engine import (
    CONTRACT_MULTIPLIER, MARGIN_ESTIMATE, DEFAULT_COMMISSION_PER_SIDE, FUTURES_TAX_RATE,
    SLIPPAGE_POINTS, STARTING_CAPITAL, INITIAL_STOP_ATR, TRAILING_STOP_ATR, MAX_HOLD_DAYS,
    ADX_GATE, variant_list, compute_daily_indicators, run_tx_daily_backtest, run_buy_and_hold,
)
from mean_reversion_engine import summarize_mr
from robustness_analysis import bootstrap_resample_pnl, summarize_bootstrap, pnl_excluding_top_n_trades

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_tx_daily")

PERIODS = [
    {"key": "train", "label": "訓練", "start": "2001-01-01", "end": "2012-12-31"},
    {"key": "test", "label": "測試", "start": "2013-01-01", "end": "2023-12-31"},
    {"key": "recent", "label": "近期(指數代理)", "start": "2024-01-01", "end": None},
]
MIN_TRAIN_TRADES = 40
BOOTSTRAP_PASS_PCT = 80.0
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
BH_NAME = "買進持有(參考)"


# ---------------------------------------------------------------------------
def rules_lines(contract, commission):
    mult = CONTRACT_MULTIPLIER[contract]
    name = "小台MTX" if contract == "mini" else "大台TX"
    return [
        "== 預先登記的規則(結果出來前就固定，不會依結果修改) ==",
        f"商品：{name}，每點NT${mult}，一次最多1口，可多可空，不反手",
        "訊號(第t天收盤後算，只用<=t的資料，第t+1天開盤進場)：",
        "  DONCHIAN：收盤 > 前20天最高價 → 多；收盤 < 前20天最低價 → 空",
        "  MA_CROSS：MA20上穿MA60 → 多；下穿 → 空",
        "  SQZ_KDJ：擠壓+KDJ關鍵K棒(compute_squeeze_kdj_features的EntryFlag，跟個股策略同規則)，只做多",
        "  SQZ_RELEASE：昨天擠壓(BB在KC內)、今天解除；動能 = 收盤 − ((20日最高+20日最低)/2 + SMA20)/2，>0多、<0空",
        f"動能分數門檻(+動能門檻變體)：ADX(14) >= {ADX_GATE:g} 且 均線排列同向(多MA5>MA20>MA60/空MA5<MA20<MA60)；只過濾進場",
        f"出場：初始停損{INITIAL_STOP_ATR}×ATR(14)；移動停損 = 進場後最高(低)收盤 ∓ {TRAILING_STOP_ATR}×ATR(14)，只收緊；"
        f"無固定停利；反向原始訊號 → 隔天開盤出場；持有滿{MAX_HOLD_DAYS}個交易日 → 隔天開盤出場(安全閥)",
        f"成本：手續費每邊NT${commission:g} + 期交稅每邊{FUTURES_TAX_RATE*100:.3f}%×契約價值(未調整價格)；"
        f"開盤/停損成交滑價{SLIPPAGE_POINTS:g}點；跳空越過停損 → 開盤價成交；換月時持倉 → 多收一次來回手續費+稅",
        "期間：訓練2001~2012、測試2013~2023、近期2024~最新(指數代理，只報告)；每段獨立重跑，期末強制平倉",
        f"選擇：只看訓練期，交易數>={MIN_TRAIN_TRADES}的變體中PF最高(平手→訓練損益→列表順序)",
        f"通過：被選中變體的測試期 PF>1 且 bootstrap正報酬比例>{BOOTSTRAP_PASS_PCT:g}% 且 拿掉最賺3筆後損益>0",
        f"起始資金NT${STARTING_CAPITAL:,}(最大回撤%以此為分母；回撤用平倉損益計算，不含持倉中的浮動損益)",
    ]


def caveat_lines(ds, contract):
    margin = MARGIN_ESTIMATE[contract]
    lines = [
        "== 注意事項 ==",
        f"1. 資料來源信心：{ds.get('source_line')}。社群分鐘資料非官方，格式只在GitHub Actions上實際看過一次。",
        "2. 近期(2024~)用加權指數當代理：指數≠期貨，期貨有基差(7~8月除息旺季逆價差明顯)，只能當方向參考。",
        "3. 換月調整：檔案沒有合約欄位，換月日/價差是對加權指數基差跳動的啟發式估計(不顯著的月份價差當0)，"
        "訊號跟點數損益用Panama調整後序列，是近似值。",
        f"4. 成本/滑價是假設值：手續費依券商不同；滑價固定1點，真實行情急殺/急拉時可能更多。",
        f"5. 保證金：{'小台' if contract == 'mini' else '大台'}原始保證金估計約NT${margin:,}(估計值，以期交所公告為準)，"
        f"起始資金NT${STARTING_CAPITAL:,}；回測不會因保證金不足擋單，請自己對照最大回撤看撐不撐得住。",
        "6. 8個變體 = 多重比較：就算全部都沒有真實優勢，8個裡面挑訓練期最好的那個，測試期剛好過關的機率也不低；"
        "這是小規模、一次性的檢驗，不是證明。",
        "7. 擠壓/KDJ訊號原本來自加密貨幣1小時K/台股個股情境，直接套到台指期日線，參數沒有重新調過。",
        "8. bootstrap把每筆交易當獨立樣本，沒考慮連續虧損的時間相關性。",
    ]
    if margin > STARTING_CAPITAL:
        lines.append(f"⚠️ 估計保證金NT${margin:,}已超過起始資金NT${STARTING_CAPITAL:,}，實務上可能根本下不了單。")
    return lines


# ---------------------------------------------------------------------------
def period_frames(ds):
    """回傳 {period_key: (df, ind, source_label)}。訓練/測試用ds["main"]；近期一律用加權指數。"""
    main = ds["main"]
    ind_main = compute_daily_indicators(main)
    index_df = ds["index"]
    if ds["source"] == "index":
        ind_index = ind_main
    else:
        ind_index = compute_daily_indicators(index_df)
    main_label = "台指期(社群分鐘資料，換月調整)" if ds["source"] == "futures" else "加權指數(備援)"
    return {
        "train": (main, ind_main, main_label),
        "test": (main, ind_main, main_label),
        "recent": (index_df, ind_index, "加權指數代理"),
    }


def compute_stats(trades, capital=STARTING_CAPITAL, n_boot=N_BOOTSTRAP):
    s = summarize_mr(trades, capital)
    wins = [t["pnl_ntd"] for t in trades if t["pnl_ntd"] > 0]
    losses = [t["pnl_ntd"] for t in trades if t["pnl_ntd"] < 0]
    s["avg_win_ntd"] = float(np.mean(wins)) if wins else 0.0
    s["avg_loss_ntd"] = float(np.mean(losses)) if losses else 0.0
    s["max_drawdown_pct"] = float(s["max_drawdown_ntd"] / capital * 100)
    for side in ("long", "short"):
        st = [t for t in trades if t["side"] == side]
        s[f"{side}_pnl_ntd"] = float(sum(t["pnl_ntd"] for t in st))
        sw = sum(t["pnl_ntd"] for t in st if t["pnl_ntd"] > 0)
        sl = -sum(t["pnl_ntd"] for t in st if t["pnl_ntd"] < 0)
        s[f"{side}_pf"] = (sw / sl) if sl > 0 else (float("inf") if sw > 0 else 0.0)
    boot = summarize_bootstrap(bootstrap_resample_pnl(trades, n_resamples=n_boot, seed=BOOTSTRAP_SEED))
    s["bootstrap_pct_positive"] = boot["pct_positive"]
    s["bootstrap_p5"] = boot["p5"]
    s["bootstrap_p95"] = boot["p95"]
    s["pnl_excl_top3_ntd"] = pnl_excluding_top_n_trades(trades, 3)
    s["total_roll_cost_ntd"] = float(sum(t.get("roll_cost_ntd", 0.0) for t in trades))
    return s


def run_all(ds, variants, contract="mini", commission=DEFAULT_COMMISSION_PER_SIDE, n_boot=N_BOOTSTRAP):
    """回傳 {"trades": {(variant, period): [...]}, "stats": {(variant, period): {...}}, "frames": ...}
    買進持有參考以BH_NAME當variant名稱放在同一個dict裡。"""
    frames = period_frames(ds)
    trades, stats = {}, {}
    for p in PERIODS:
        df, ind, _label = frames[p["key"]]
        for v in variants:
            tr = run_tx_daily_backtest(df, ind, v["signal"], v["gate"], start=p["start"], end=p["end"],
                                       contract=contract, commission_per_side=commission,
                                       variant_name=v["name"])
            trades[(v["name"], p["key"])] = tr
            stats[(v["name"], p["key"])] = compute_stats(tr, n_boot=n_boot)
        bh = run_buy_and_hold(df, start=p["start"], end=p["end"], contract=contract,
                              commission_per_side=commission)
        trades[(BH_NAME, p["key"])] = bh
        stats[(BH_NAME, p["key"])] = compute_stats(bh, n_boot=n_boot)
    return {"trades": trades, "stats": stats, "frames": frames}


def select_variant(variants, stats, min_trades=MIN_TRAIN_TRADES):
    """只看訓練期stats選出變體名稱(或None)。"""
    best, best_key = None, None
    for order, v in enumerate(variants):
        s = stats[(v["name"], "train")]
        if s["trade_count"] < min_trades:
            continue
        key = (s["profit_factor"], s["total_pnl_ntd"], -order)
        if best_key is None or key > best_key:
            best, best_key = v["name"], key
    return best


def verdict(test_stats):
    """回傳(passed, checks)，checks = [(說明, bool)]。"""
    checks = [
        (f"測試期PF {_fmt_pf(test_stats['profit_factor'])} > 1", test_stats["profit_factor"] > 1),
        (f"測試期bootstrap正報酬比例 {test_stats['bootstrap_pct_positive']:.1f}% > {BOOTSTRAP_PASS_PCT:g}%",
         test_stats["bootstrap_pct_positive"] > BOOTSTRAP_PASS_PCT),
        (f"測試期拿掉最賺3筆後損益 {test_stats['pnl_excl_top3_ntd']:,.0f} > 0",
         test_stats["pnl_excl_top3_ntd"] > 0),
    ]
    return all(ok for _, ok in checks), checks


def yearly_rows(trades_by_key):
    rows = []
    for (vname, pkey), trades in trades_by_key.items():
        if not trades:
            continue
        df = pd.DataFrame(trades)
        df["year"] = pd.to_datetime(df["exit_date"]).dt.year
        for y, g in df.groupby("year"):
            w = g.loc[g["pnl_ntd"] > 0, "pnl_ntd"].sum()
            lo = -g.loc[g["pnl_ntd"] < 0, "pnl_ntd"].sum()
            pf = (w / lo) if lo > 0 else (float("inf") if w > 0 else 0.0)
            rows.append({"變體": vname, "期間": pkey, "年度": int(y), "交易數": int(len(g)),
                         "損益(NT$)": round(float(g["pnl_ntd"].sum()), 0),
                         "勝率(%)": round(float((g["pnl_ntd"] > 0).mean() * 100), 1),
                         "PF": round(pf, 3) if np.isfinite(pf) else pf})
    return rows


def _fmt_pf(pf):
    if pf is None or (isinstance(pf, float) and np.isnan(pf)):
        return "-"
    return "∞" if np.isinf(pf) else f"{pf:.2f}"


def summary_rows(names, stats):
    rows = []
    plabel = {p["key"]: p["label"] for p in PERIODS}
    for vname in names:
        for p in PERIODS:
            s = stats[(vname, p["key"])]
            rows.append({
                "變體": vname, "期間": plabel[p["key"]], "交易數": s["trade_count"],
                "PF": round(s["profit_factor"], 3) if np.isfinite(s["profit_factor"]) else s["profit_factor"],
                "勝率(%)": round(s["win_rate"], 1), "平均獲利(NT$)": round(s["avg_win_ntd"], 0),
                "平均虧損(NT$)": round(s["avg_loss_ntd"], 0), "總損益(NT$)": round(s["total_pnl_ntd"], 0),
                "最大回撤(NT$)": round(s["max_drawdown_ntd"], 0),
                "最大回撤(%資金)": round(s["max_drawdown_pct"], 1),
                "最長連虧(筆)": s["max_consecutive_losses"], "平均持有(天)": round(s["avg_hold_days"], 1),
                "多單筆數": s["long_count"], "多單損益(NT$)": round(s["long_pnl_ntd"], 0),
                "空單筆數": s["short_count"], "空單損益(NT$)": round(s["short_pnl_ntd"], 0),
                "bootstrap正報酬(%)": round(s["bootstrap_pct_positive"], 1),
                "拿掉前3筆損益(NT$)": round(s["pnl_excl_top3_ntd"], 0),
                "換月成本(NT$)": round(s["total_roll_cost_ntd"], 0),
            })
    return rows


def side_by_side_lines(names, stats):
    head = (f"{'變體':<22}{'訓練PF':>8}{'筆':>5}{'損益':>11}  {'測試PF':>8}{'筆':>5}{'損益':>11}  "
            f"{'近期PF':>8}{'筆':>5}{'損益':>10}  {'兩段都PF>1':>10}")
    lines = [head, "-" * len(head)]
    for n in names:
        tr, te, re_ = stats[(n, "train")], stats[(n, "test")], stats[(n, "recent")]
        both = "是" if (tr["profit_factor"] > 1 and te["profit_factor"] > 1) else "否"
        if n == BH_NAME:
            both = "(參考)"
        lines.append(
            f"{n:<22}{_fmt_pf(tr['profit_factor']):>8}{tr['trade_count']:>5}{tr['total_pnl_ntd']:>11,.0f}  "
            f"{_fmt_pf(te['profit_factor']):>8}{te['trade_count']:>5}{te['total_pnl_ntd']:>11,.0f}  "
            f"{_fmt_pf(re_['profit_factor']):>8}{re_['trade_count']:>5}{re_['total_pnl_ntd']:>10,.0f}  {both:>10}")
    return lines


def detail_lines(names, stats):
    lines = []
    for n in names:
        lines.append(f"[{n}]")
        for p in PERIODS:
            s = stats[(n, p["key"])]
            lines.append(
                f"  {p['label']}：{s['trade_count']}筆 PF {_fmt_pf(s['profit_factor'])} 勝率{s['win_rate']:.1f}% "
                f"均賺{s['avg_win_ntd']:,.0f}/均賠{s['avg_loss_ntd']:,.0f} 損益{s['total_pnl_ntd']:,.0f} "
                f"最大回撤{s['max_drawdown_ntd']:,.0f}({s['max_drawdown_pct']:.1f}%資金) 最長連虧{s['max_consecutive_losses']} "
                f"多{s['long_count']}筆/{s['long_pnl_ntd']:,.0f} 空{s['short_count']}筆/{s['short_pnl_ntd']:,.0f} "
                f"bootstrap+{s['bootstrap_pct_positive']:.1f}% 換月成本{s['total_roll_cost_ntd']:,.0f}")
    return lines


def data_source_lines(ds):
    lines = ["== 資料來源 =="]
    lines.append(ds.get("source_line", ""))
    if ds["source"] == "futures":
        fd = ds["main"]
        v = ds.get("validation", {})
        ri = ds.get("roll_info", {})
        lines.append(f"訓練/測試：台指期日盤日線 {fd.index.min().date()} ~ {fd.index.max().date()}，{len(fd)}天")
        lines.append(f"驗證：覆蓋率{v.get('coverage_pct', float('nan')):.2f}%、日報酬相關{v.get('return_corr', float('nan')):.4f}、"
                     f"|基差|/指數中位數{v.get('median_abs_basis_pct', float('nan')):.3f}%、"
                     f"OHLC自洽{v.get('ohlc_consistent_pct', float('nan')):.2f}%")
        lines.append(f"換月：{ri.get('n_rolls')}次，價差顯著{ri.get('n_significant')}次，"
                     f"雜訊σ≈{ri.get('noise_sigma', float('nan')):.1f}點(詳見tx_roll_adjustments.csv)")
        missing = (ds.get("parse") or {}).get("missing_periods")
        if missing:
            lines.append(f"⚠️ 有分段下載/解析失敗：{missing}(其餘分段仍通過驗證)")
    else:
        lines.append(f"⚠️ 整段(訓練/測試/近期)都用加權指數日線，不是期貨價格。原因：{ds.get('fallback_reason')}")
    idx = ds["index"]
    lines.append(f"近期(2024~)：加權指數代理，資料到 {idx.index.max().date()}(指數≠期貨，7~8月除息季基差明顯)")
    lines.append("診斷檔：data_parse_report.txt / data_validation.json / tx_daily_bars_sample.csv / tx_roll_adjustments.csv")
    return lines


def build_report(ds, res, variants, selected, contract, commission):
    stats = res["stats"]
    names = [v["name"] for v in variants] + [BH_NAME]
    lines = [ds.get("source_line", ""), "",
             "台指期日線策略 —— 預先登記小型測試(8變體 × 訓練/測試/近期)", ""]
    lines += rules_lines(contract, commission) + [""]
    lines += data_source_lines(ds) + [""]
    lines.append("== 選擇與判定 ==")
    if selected is None:
        lines.append(f"沒有任何變體在訓練期達到{MIN_TRAIN_TRADES}筆交易 → 無法選擇，判定：未通過")
        passed = False
    else:
        tr = stats[(selected, "train")]
        lines.append(f"訓練期選出：{selected}(訓練PF {_fmt_pf(tr['profit_factor'])}，{tr['trade_count']}筆，"
                     f"損益{tr['total_pnl_ntd']:,.0f})")
        passed, checks = verdict(stats[(selected, "test")])
        for desc, ok in checks:
            lines.append(f"  [{'✓' if ok else '✗'}] {desc}")
        lines.append(f"判定：{'通過' if passed else '未通過'}")
        rs = stats[(selected, "recent")]
        lines.append(f"近期(指數代理，只參考)：{rs['trade_count']}筆 PF {_fmt_pf(rs['profit_factor'])} "
                     f"損益{rs['total_pnl_ntd']:,.0f}")
    lines.append("")
    lines.append("== 並排比較(損益單位NT$) ==")
    lines += side_by_side_lines(names, stats)
    lines.append("")
    lines.append("== 各變體明細 ==")
    lines += detail_lines(names, stats)
    if selected is not None:
        lines.append("")
        lines.append(f"== 被選中變體逐年表({selected}) ==")
        yr = [r for r in yearly_rows({k: v for k, v in res["trades"].items() if k[0] == selected})]
        for r in sorted(yr, key=lambda r: r["年度"]):
            lines.append(f"  {r['年度']} [{r['期間']}] {r['交易數']}筆 損益{r['損益(NT$)']:,.0f} "
                         f"勝率{r['勝率(%)']}% PF {_fmt_pf(r['PF'])}")
    lines.append("")
    lines += caveat_lines(ds, contract)
    return lines, passed


def write_outputs(out_dir, lines, res, variants):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    names = [v["name"] for v in variants] + [BH_NAME]
    pd.DataFrame(summary_rows(names, res["stats"])).to_csv(
        os.path.join(out_dir, "tx_daily_summary.csv"), index=False, encoding="utf-8-sig")
    yr = yearly_rows(res["trades"])
    pd.DataFrame(yr, columns=["變體", "期間", "年度", "交易數", "損益(NT$)", "勝率(%)", "PF"]).to_csv(
        os.path.join(out_dir, "tx_daily_yearly.csv"), index=False, encoding="utf-8-sig")
    all_tr = []
    for (vname, pkey), trades in res["trades"].items():
        for t in trades:
            all_tr.append({
                "變體": vname, "期間": pkey, "方向": "多" if t["side"] == "long" else "空",
                "訊號日": str(t["signal_date"])[:10], "進場日": str(t["entry_date"])[:10],
                "出場日": str(t["exit_date"])[:10], "進場價(調整後)": round(t["entry_price"], 2),
                "出場價(調整後)": round(t["exit_price"], 2), "進場價(未調整)": round(t["raw_entry_price"], 2),
                "出場價(未調整)": round(t["raw_exit_price"], 2), "點數": round(t["points"], 2),
                "換月次數": t["n_rolls"], "換月成本(NT$)": round(t["roll_cost_ntd"], 1),
                "總成本(NT$)": round(t["cost_ntd"], 1), "損益(NT$)": round(t["pnl_ntd"], 1),
                "持有天數": t["hold_days"], "出場原因": t["exit_reason"],
            })
    cols = ["變體", "期間", "方向", "訊號日", "進場日", "出場日", "進場價(調整後)", "出場價(調整後)",
            "進場價(未調整)", "出場價(未調整)", "點數", "換月次數", "換月成本(NT$)", "總成本(NT$)",
            "損益(NT$)", "持有天數", "出場原因"]
    pd.DataFrame(all_tr, columns=cols).to_csv(
        os.path.join(out_dir, "tx_daily_trades.csv"), index=False, encoding="utf-8-sig")


def load_dataset(args):
    return tx_daily_data.build_tx_daily_dataset(refresh=args.refresh_data,
                                                index_cache_dir=args.index_cache_dir)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="台指期日線策略預先登記測試")
    ap.add_argument("--contract", choices=["mini", "big"], default="mini")
    ap.add_argument("--commission-per-side", type=float, default=DEFAULT_COMMISSION_PER_SIDE)
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="只跑部分變體(名稱，例如 DONCHIAN 'MA_CROSS+動能門檻')；預設全部8個")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--index-cache-dir", default=tx_daily_data.INDEX_CACHE_DIR_DEFAULT,
                    help="加權指數下載快取資料夾")
    ap.add_argument("--grid", action="store_true",
                    help="網格模式：(12單一訊號+66混搭)×動能門檻有/無×12出場 = 1872變體(見tx_daily_grid.py)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.grid and args.variants:
        print("--grid 模式不能搭配 --variants(網格的變體清單是事先登記、固定的)", file=sys.stderr)
        return 2
    variants = variant_list()
    if args.variants:
        wanted = set(args.variants)
        unknown = wanted - {v["name"] for v in variants}
        if unknown:
            print(f"未知的變體名稱：{sorted(unknown)}；可用：{[v['name'] for v in variants]}", file=sys.stderr)
            return 2
        variants = [v for v in variants if v["name"] in wanted]

    os.makedirs(args.results_dir, exist_ok=True)
    if args.grid:
        import tx_daily_grid
        rules = tx_daily_grid.grid_rules_lines(args.contract, args.commission_per_side)
    else:
        rules = rules_lines(args.contract, args.commission_per_side)
    for line in rules:
        print(line, flush=True)
    print("", flush=True)

    ds = load_dataset(args)
    tx_daily_data.write_diagnostics(ds, args.results_dir)
    if ds.get("main") is None:
        msg = [ds.get("source_line", "價格來源：無"), "", "資料完全取不到，回測沒有執行。",
               f"原因：{ds.get('fallback_reason')}"]
        summary_name = "summary_grid.txt" if args.grid else "summary.txt"
        with open(os.path.join(args.results_dir, summary_name), "w", encoding="utf-8") as f:
            f.write("\n".join(msg) + "\n")
        print("\n".join(msg), flush=True)
        return 1
    if args.grid:
        return tx_daily_grid.run_grid_mode(args, ds)

    res = run_all(ds, variants, contract=args.contract, commission=args.commission_per_side,
                  n_boot=args.n_bootstrap)
    selected = select_variant(variants, res["stats"])
    lines, passed = build_report(ds, res, variants, selected, args.contract, args.commission_per_side)
    write_outputs(args.results_dir, lines, res, variants)
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
