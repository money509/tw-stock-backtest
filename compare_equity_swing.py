"""
compare_equity_swing.py
==========================
現股波段策略(基本面/籌碼面為主，2~8週持有) - 主執行腳本。

這是v1，刻意比compare_breakout.py(累積了很多輪、大約10個階段)小很多——
不是偷工，是故意先做一個小而完整、能跑通端到端的版本，之後有需要再逐輪擴充，
跟compare_breakout.py本身當初也是從小骨架開始、逐輪加上ATR網格/跨週期驗證/
walk-forward的歷程一樣。v1包含5個階段：

  1. 資料下載/預計算：股價(data_loader) + 大盤氛圍(2330代理) + 三個新的基本面/
     籌碼面資料源(revenue_data_loader/valuation_data_loader，financial_statement_loader
     包一層try/except，失敗不影響其餘流程)。
  2. 單一訊號拆解(signal ablation)：跟compare_breakout.py同樣的IS/OOS切分慣例
     (split_is_oos，直接重用)。
  3. 結構門檻：v1只有一種門檻(站上MA60)，不像突破策略有多個門檻變體可以比較，
     先讓一個合理的門檻端到端跑通，門檻變體比較留給之後有需要再加。
  4. 最終IS/OOS + bootstrap：用「IS內PF>1」自動篩選的訊號組合(select_winning_signals，
     直接重用)去跑。
  5. 固定規則walk-forward：把最終選出的組合原封不動套到N個獨立不重疊的歷史區塊，
     檢查跨時間穩不穩定，做法完全比照compare_breakout.py.run_fixed_combo_walkforward()
     的精神重寫一份(這支策略的引擎函式簽章不同，沒辦法直接重用該函式本身，
     但驗證邏輯——切N段、同一組規則、不重選——完全一樣)。

v1明確**沒有**做的(留給之後有需要再加，不是忘記)：
  - 訊號組合比較(自動篩選 vs 手動指定的固定組合對照)
  - 多個結構門檻變體的比較表
  - ATR停損倍數敏感度網格搜尋(這裡用固定的--atr-stop-mult，不做網格)
  - 跨市場週期驗證(套到2022年修正段等不同市況)
  - 「每折重新選一次」的walk-forward(只做「固定規則」版本，理由跟
    compare_breakout.py一致：reselect版本更貴、雜訊更大，這個專案已經有
    足夠證據支持這個判斷)

⚠️ 誠實聲明：這整支策略是**全新、未經任何驗證的假設**，不是「已經找到現股波段
的優勢」。三個新的資料loader，只有valuation_data_loader.py/revenue_data_loader.py
有中等以上的信心，financial_statement_loader.py明確標記未經驗證，第一次
GitHub Actions真實執行才是這整套流程(從抓資料到跑出PF數字)的第一次真正測試。
不管這裡印出來的數字好不好看，在多做幾輪walk-forward/跨週期驗證之前，都不該
被當作「找到能用的策略」。
"""
import os
import argparse
import datetime
import pandas as pd

from data_loader import load_price_data, build_master_calendar
from mean_reversion_engine import precompute_regime_series, compute_regime, summarize_mr
from equity_swing_engine import (
    EQUITY_SIGNAL_NAMES,
    precompute_all_equity_indicators, run_equity_swing_backtest,
)
from robustness_analysis import bootstrap_resample_pnl, summarize_bootstrap, bootstrap_p_value
from taifex_universe import STOCK_FUTURES_UNIVERSE
from compare_breakout import split_is_oos, _fmt_pf, _prefer_nonzero_trades

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_equity_swing")

MIN_TRADES_FOR_RANKING = 15  # 現股波段一次最多同時持有幾個部位、且持有週數長，
                              # 交易頻率本來就比短線引擎低很多，門檻比照比例調低

SIGNAL_LABELS = {
    "score_revenue_yoy": "月營收年增率(YoY)",
    "score_revenue_mom": "月營收月增率(MoM)",
    "score_valuation_pe": "本益比(PE，越低越好)",
    "score_valuation_pb": "股價淨值比(PB，越低越好)",
    "score_dividend_yield": "殖利率",
    "score_eps_growth": "EPS季增率(QoQ)",
    "score_gross_margin": "毛利率",
    "score_roe": "股東權益報酬率(ROE)",
    "score_foreign_streak": "外資連續買超天數",
    "score_trust_streak": "投信連續買超天數",
}

DEFAULT_FIXED_COMBO_WALKFORWARD_FOLDS = 4


def select_winning_signals(ablation_df, min_trades=MIN_TRADES_FOR_RANKING):
    """跟compare_breakout.py select_winning_signals()同一套規則：挑「交易筆數夠多、
    且PF>1」的訊號；都不滿足時退回總損益前3名並標示reliable=False。"""
    non_baseline = _prefer_nonzero_trades(ablation_df[ablation_df["signal"] != "__baseline_equal_weight__"])
    reliable_pool = non_baseline[non_baseline["trade_count"] >= min_trades]
    picked = reliable_pool[reliable_pool["profit_factor"] > 1.0]
    if picked.empty:
        fallback = non_baseline.sort_values("total_pnl_ntd", ascending=False).head(3)
        return list(fallback["signal"]), False
    return list(picked["signal"]), True


def run_signal_ablation(price_data, indicators_by_code, regime_series, is_calendar, starting_capital,
                         common_kwargs, has_fundamentals: dict):
    """對EQUITY_SIGNAL_NAMES每個訊號單獨開啟(權重1.0，其餘0)，在IS跑一次回測。
    has_fundamentals：{signal_name: bool}，標記這次回測有沒有下載到對應的資料源——
    完全沒有資料源的訊號(例如沒開--with-financial-statements)乾脆不测，因為
    測了也只會全部是中性分數0.0，徒然浪費運算時間、結果沒有意義。"""
    rows = []
    signal_names = [s for s in EQUITY_SIGNAL_NAMES if has_fundamentals.get(s, True)]

    for i, signal_name in enumerate(signal_names, start=1):
        trades = run_equity_swing_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, starting_capital=starting_capital,
            signal_weights={signal_name: 1.0}, **common_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "signal": signal_name, "label": SIGNAL_LABELS[signal_name],
            "trade_count": stats["trade_count"], "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"], "total_pnl_ntd": stats["total_pnl_ntd"],
            "max_drawdown_ntd": stats["max_drawdown_ntd"],
        })
        print(f"  [{i}/{len(signal_names)}] {SIGNAL_LABELS[signal_name]} "
              f"-> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)

    baseline_trades = run_equity_swing_backtest(
        price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
        master_calendar=is_calendar, starting_capital=starting_capital,
        signal_weights={name: 1.0 for name in signal_names}, **common_kwargs,
    )
    baseline_stats = summarize_mr(baseline_trades, starting_capital)
    rows.append({
        "signal": "__baseline_equal_weight__", "label": f"基準({len(signal_names)}訊號等權重)",
        "trade_count": baseline_stats["trade_count"], "win_rate": baseline_stats["win_rate"],
        "profit_factor": baseline_stats["profit_factor"], "total_pnl_ntd": baseline_stats["total_pnl_ntd"],
        "max_drawdown_ntd": baseline_stats["max_drawdown_ntd"],
    })
    print(f"  [基準] {len(signal_names)}訊號等權重 -> {baseline_stats['trade_count']}筆, "
          f"PF={_fmt_pf(baseline_stats['profit_factor'])}, 損益={baseline_stats['total_pnl_ntd']:,.0f}", flush=True)

    return pd.DataFrame(rows), signal_names


def run_fixed_combo_equity_walkforward(price_data, indicators_by_code, regime_series, master_calendar,
                                        starting_capital, common_kwargs, signal_weights, n_folds) -> pd.DataFrame:
    """把master_calendar切成n_folds個獨立不重疊的區塊，同一組signal_weights(不重新
    挑選)原封不動套到每一塊各跑一次，檢查跨時間表現穩不穩定。跟compare_breakout.py
    的run_fixed_combo_walkforward()驗證精神完全一樣，見本檔案模組docstring第5點。"""
    n = len(master_calendar)
    bounds = [int(round(k * n / n_folds)) for k in range(n_folds + 1)]
    rows = []
    for i in range(n_folds):
        chunk = master_calendar[bounds[i]:bounds[i + 1]]
        if len(chunk) == 0:
            print(f"  [固定規則walk-forward 第{i + 1}折] 區塊天數為0，跳過", flush=True)
            continue
        trades = run_equity_swing_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=chunk, starting_capital=starting_capital,
            signal_weights=signal_weights, **common_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        if stats["trade_count"] < 5:
            print(f"  [固定規則walk-forward 第{i + 1}/{n_folds}折] "
                  f"{chunk[0].date()}~{chunk[-1].date()}：交易筆數({stats['trade_count']}筆)太少，跳過(不計入統計)",
                  flush=True)
            continue
        rows.append({
            "fold": i + 1, "period_start": chunk[0].date().isoformat(), "period_end": chunk[-1].date().isoformat(),
            "trade_count": stats["trade_count"], "profit_factor": stats["profit_factor"],
            "win_rate": stats["win_rate"], "total_pnl_ntd": stats["total_pnl_ntd"],
            "avg_hold_days": stats["avg_hold_days"],
        })
        print(f"  [固定規則walk-forward 第{i + 1}/{n_folds}折] {chunk[0].date()}~{chunk[-1].date()} "
              f"-> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="現股波段策略(基本面/籌碼面為主) - 訊號拆解 + 最終組合驗證")
    parser.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=1095)).isoformat())
    parser.add_argument("--end", default=datetime.date.today().isoformat())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--starting-capital", type=float, default=1_000_000)
    parser.add_argument("--max-stocks", type=int, default=0,
                         help="限制掃描股票數量(0=全部320檔，測試時可以設小一點加快速度)")
    parser.add_argument("--min-hold-weeks", type=int, default=2)
    parser.add_argument("--max-hold-weeks", type=int, default=8)
    parser.add_argument("--atr-stop-mult", type=float, default=2.0,
                         help="ATR停損倍數(週級別持有，預設比短線引擎的1.0/1.5寬，避免正常波動洗出場)")
    parser.add_argument("--max-concurrent-positions", type=int, default=5,
                         help="最多同時持有幾檔(資金平分成這麼多槽位，見equity_swing_engine.py說明)")
    parser.add_argument("--risk-pct-per-trade", type=float, default=None,
                         help="改用風險預算式股數計算：每筆風險=帳戶權益 x 這個比例。不指定時用槽位現金平分")
    parser.add_argument("--slippage-pct", type=float, default=0.0)
    parser.add_argument("--with-revenue", action="store_true", help="下載月營收資料(YoY/MoM訊號)")
    parser.add_argument("--with-valuation", action="store_true", help="下載每日PE/PB/殖利率資料")
    parser.add_argument("--with-chip", action="store_true", help="下載三大法人籌碼資料(外資/投信連續買超訊號)")
    parser.add_argument("--with-financial-statements", action="store_true",
                         help="下載季度財報資料(EPS成長率/毛利率/ROE訊號)。⚠️這支loader未經驗證，"
                              "失敗/逾時不會中斷流程，只會讓這三個訊號退化成中性分數，"
                              "且逐股查詢速度慢很多，全市場開啟會需要相當長時間")
    parser.add_argument("--fixed-combo-walkforward-folds", type=int, default=0,
                         help="固定規則walk-forward的區塊數，0代表不啟用(預設)。"
                              f"建議值{DEFAULT_FIXED_COMBO_WALKFORWARD_FOLDS}")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    universe = dict(STOCK_FUTURES_UNIVERSE)
    INDEX_PROXY_CODE = "2330"
    if args.max_stocks > 0:
        universe = dict(list(universe.items())[: args.max_stocks])
        if INDEX_PROXY_CODE not in universe:
            universe[INDEX_PROXY_CODE] = STOCK_FUTURES_UNIVERSE[INDEX_PROXY_CODE]

    # ---- 階段1：資料下載/預計算 ----
    print(f"下載/讀取歷史股價 ({args.start} ~ {args.end})，共 {len(universe)} 檔標的 ...", flush=True)
    price_data = load_price_data(universe, args.start, args.end, refresh=args.refresh)
    if INDEX_PROXY_CODE not in price_data:
        print("⚠️ 大盤代理指標(2330)下載失敗，無法建立回測日曆，中止")
        return
    master_calendar = build_master_calendar(price_data, reference_code=INDEX_PROXY_CODE)
    regime_series = precompute_regime_series(price_data[INDEX_PROXY_CODE])

    universe_codes = set(universe.keys())
    revenue_data = None
    valuation_data = None
    financial_data = None
    chip_data = None

    if args.with_revenue:
        print("下載月營收資料 ...", flush=True)
        from revenue_data_loader import load_revenue_data
        revenue_data = load_revenue_data(args.start, args.end, universe_codes=universe_codes, refresh=args.refresh)
    if args.with_valuation:
        print("下載每日PE/PB/殖利率資料 ...", flush=True)
        from valuation_data_loader import load_valuation_data
        valuation_data = load_valuation_data(args.start, args.end, universe_codes=universe_codes, refresh=args.refresh)
    if args.with_chip:
        print("下載三大法人籌碼資料 ...", flush=True)
        from chip_data_loader import load_chip_data
        chip_data = load_chip_data(args.start, args.end, universe_codes=universe_codes, refresh=args.refresh)
    if args.with_financial_statements:
        print("下載季度財報資料 (⚠️ 未經驗證，失敗不影響其餘流程) ...", flush=True)
        try:
            from financial_statement_loader import load_financial_statement_data
            financial_data = load_financial_statement_data(
                args.start, args.end, universe_codes=universe_codes, refresh=args.refresh)
        except Exception as e:
            print(f"⚠️ 季度財報資料下載失敗(financial_statement_loader未經驗證，這是預期中可能發生的情況): {e}\n"
                  f"   這三個訊號(EPS成長率/毛利率/ROE)這次回測會全部退化成中性分數，不影響其餘訊號/流程。",
                  flush=True)
            financial_data = None

    has_fundamentals = {name: True for name in EQUITY_SIGNAL_NAMES}
    if not args.with_revenue:
        has_fundamentals["score_revenue_yoy"] = has_fundamentals["score_revenue_mom"] = False
    if not args.with_valuation:
        for s in ("score_valuation_pe", "score_valuation_pb", "score_dividend_yield"):
            has_fundamentals[s] = False
    if not args.with_financial_statements or financial_data is None:
        for s in ("score_eps_growth", "score_gross_margin", "score_roe"):
            has_fundamentals[s] = False
    if not args.with_chip:
        has_fundamentals["score_foreign_streak"] = has_fundamentals["score_trust_streak"] = False

    print("預計算技術面/基本面/籌碼面指標 ...", flush=True)
    indicators_by_code = precompute_all_equity_indicators(
        price_data, universe, revenue_data=revenue_data, valuation_data=valuation_data,
        financial_data=financial_data, chip_data=chip_data,
    )

    is_calendar, oos_calendar = split_is_oos(master_calendar)
    print(f"IS: {is_calendar[0].date()} ~ {is_calendar[-1].date()} ({len(is_calendar)}天) / "
          f"OOS: {oos_calendar[0].date()} ~ {oos_calendar[-1].date()} ({len(oos_calendar)}天)\n", flush=True)

    common_kwargs = dict(
        min_hold_weeks=args.min_hold_weeks, max_hold_weeks=args.max_hold_weeks,
        atr_stop_mult=args.atr_stop_mult, max_concurrent_positions=args.max_concurrent_positions,
        risk_pct_per_trade=args.risk_pct_per_trade, slippage_pct=args.slippage_pct,
    )

    # ---- 階段2：單一訊號拆解(IS) ----
    print("=== 階段2：單一訊號拆解(IS，站上MA60門檻) ===", flush=True)
    ablation_df, tested_signals = run_signal_ablation(
        price_data, indicators_by_code, regime_series, is_calendar, args.starting_capital,
        common_kwargs, has_fundamentals,
    )
    ablation_df.to_csv(os.path.join(RESULTS_DIR, "signal_ablation.csv"), index=False)

    winning_signals, reliable = select_winning_signals(ablation_df)
    winning_weights = {name: 1.0 for name in winning_signals}
    print(f"\n{'✅' if reliable else '⚠️'} 選出的訊號組合："
          f"{[SIGNAL_LABELS.get(s, s) for s in winning_signals]}"
          f"{'' if reliable else '(探索性選擇，沒有訊號單獨PF>1，僅供參考)'}\n", flush=True)

    # ---- 階段3+4：最終IS/OOS + bootstrap ----
    print("=== 階段3/4：最終組合 IS vs OOS + bootstrap ===", flush=True)
    final_results = {}
    for split_name, calendar in [("IS", is_calendar), ("OOS", oos_calendar)]:
        trades = run_equity_swing_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=calendar, starting_capital=args.starting_capital,
            signal_weights=winning_weights, **common_kwargs,
        )
        stats = summarize_mr(trades, args.starting_capital)
        final_results[split_name] = stats
        if split_name == "OOS":
            oos_trades = trades
        print(f"  [{split_name}] {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}, "
              f"平均持有{stats['avg_hold_days']:.1f}天", flush=True)

    boot_results = bootstrap_resample_pnl(oos_trades, n_resamples=1000, seed=42)
    boot_stats = summarize_bootstrap(boot_results)
    boot_p = bootstrap_p_value(boot_results)
    print(f"  [OOS bootstrap] 正報酬比例={boot_stats['pct_positive']:.1f}%, p值={boot_p:.3f}, "
          f"90%信賴區間=[{boot_stats['p5']:,.0f}, {boot_stats['p95']:,.0f}]\n", flush=True)
    pd.DataFrame(oos_trades).to_csv(os.path.join(RESULTS_DIR, "oos_trades.csv"), index=False)

    # ---- 階段5：固定規則walk-forward ----
    fixed_combo_wf_df = None
    if args.fixed_combo_walkforward_folds > 0:
        print(f"=== 階段5：固定規則walk-forward(共{args.fixed_combo_walkforward_folds}折) ===", flush=True)
        fixed_combo_wf_df = run_fixed_combo_equity_walkforward(
            price_data, indicators_by_code, regime_series, master_calendar, args.starting_capital,
            common_kwargs, winning_weights, args.fixed_combo_walkforward_folds,
        )
        if not fixed_combo_wf_df.empty:
            fixed_combo_wf_df.to_csv(os.path.join(RESULTS_DIR, "fixed_combo_walkforward.csv"), index=False)

    # ---- 寫summary.txt ----
    summary_lines = [
        "現股波段策略(基本面/籌碼面為主，2-8週持有) - 回測摘要",
        f"期間: {args.start} ~ {args.end}  起始資金: {args.starting_capital:,.0f}  "
        f"持有週數: {args.min_hold_weeks}~{args.max_hold_weeks}週  最多同時持有: {args.max_concurrent_positions}檔",
        f"資料源: 月營收={'✅' if args.with_revenue else '未開啟'}  "
        f"評價(PE/PB/殖利率)={'✅' if args.with_valuation else '未開啟'}  "
        f"籌碼(三大法人)={'✅' if args.with_chip else '未開啟'}  "
        f"財報(EPS/毛利率/ROE)={'✅' if financial_data is not None else ('下載失敗' if args.with_financial_statements else '未開啟')}",
        "",
        "--- 單一訊號拆解(IS) ---",
    ]
    header = f"{'訊號':30s} {'交易數':>6s} {'PF':>8s} {'勝率%':>7s} {'總損益':>14s}"
    summary_lines.append(header)
    summary_lines.append("-" * len(header))
    for _, r in ablation_df.iterrows():
        summary_lines.append(
            f"{r['label']:30s} {r['trade_count']:>6d} {_fmt_pf(r['profit_factor']):>8s} "
            f"{r['win_rate']:>6.1f}% {r['total_pnl_ntd']:>14,.0f}")

    summary_lines.append(f"\n選出的訊號組合(reliable={reliable}): "
                          f"{[SIGNAL_LABELS.get(s, s) for s in winning_signals]}")

    summary_lines.append("\n--- 最終組合：IS vs OOS + bootstrap ---")
    for split_name in ("IS", "OOS"):
        s = final_results[split_name]
        summary_lines.append(
            f"[{split_name}] 交易數={s['trade_count']} PF={_fmt_pf(s['profit_factor'])} "
            f"勝率={s['win_rate']:.1f}% 總損益={s['total_pnl_ntd']:,.0f} "
            f"最大回撤={s['max_drawdown_ntd']:,.0f} 平均持有天數={s['avg_hold_days']:.1f}")
    summary_lines.append(
        f"[OOS bootstrap] 正報酬比例={boot_stats['pct_positive']:.1f}% p值={boot_p:.3f} "
        f"90%信賴區間=[{boot_stats['p5']:,.0f}, {boot_stats['p95']:,.0f}]")

    if fixed_combo_wf_df is not None:
        summary_lines.append(f"\n--- 固定規則walk-forward(共{args.fixed_combo_walkforward_folds}折) ---")
        if fixed_combo_wf_df.empty:
            summary_lines.append("  ⚠️ 所有折都因為交易筆數太少被跳過，沒有可用結果")
        else:
            header_wf = f"{'折':>4s} {'期間':22s} {'交易數':>6s} {'PF':>8s} {'勝率%':>7s} {'總損益':>14s}"
            summary_lines.append(header_wf)
            summary_lines.append("-" * len(header_wf))
            for _, r in fixed_combo_wf_df.iterrows():
                period = f"{r['period_start']}~{r['period_end']}"
                summary_lines.append(
                    f"{int(r['fold']):>4d} {period:22s} {int(r['trade_count']):>6d} "
                    f"{_fmt_pf(r['profit_factor']):>8s} {r['win_rate']:>6.1f}% {r['total_pnl_ntd']:>14,.0f}")
            mean_pf = fixed_combo_wf_df["profit_factor"].replace(float("inf"), float("nan")).mean()
            summary_lines.append(f"\n  平均PF(排除無限大)={mean_pf:.2f}，"
                                  f"{len(fixed_combo_wf_df)}/{args.fixed_combo_walkforward_folds}折有足夠交易可比較")

    summary_lines.append(
        "\n--- 怎麼判讀 ---\n"
        "這整支策略是全新、未經任何驗證的假設，跟其他既有引擎多輪測試後的結論一樣，\n"
        "PF>1不代表「找到了」，要同時看：(1)OOS的PF是不是也>1、(2)bootstrap正報酬比例\n"
        "是不是夠高(理想上>80%)、(3)固定規則walk-forward跨區塊是不是穩定(不是只有\n"
        "某一折特別好)。三個新資料源裡，financial_statement_loader.py明確未經驗證，\n"
        "如果這次回測沒有開啟--with-financial-statements，score_eps_growth/\n"
        "score_gross_margin/score_roe這三個訊號全部是中性分數，不代表它們沒用，\n"
        "只代表這次沒有測試到。第一次在GitHub Actions真實環境執行，才是三個新loader\n"
        "(尤其是revenue_data_loader.py的URL格式、financial_statement_loader.py整支)\n"
        "的第一次真正驗證，不是這裡的模擬資料/邏輯測試。"
    )

    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n完成，結果寫入 {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
