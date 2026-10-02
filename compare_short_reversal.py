"""
compare_short_reversal.py
===========================
短期反轉(Short-Term Reversal)策略 - 主執行腳本。

三階段流程(跟compare_breakout.py同樣骨架：變體拆解 → 變體比較 → 最終IS/OOS + bootstrap
驗證)：
1. 回看窗口比較(3/5/10日)：用固定5天持有(文獻基準)當基準出場天數，只換「用幾天的報酬率
   排名」，選總損益最高、且交易筆數夠多的窗口。
2. 固定持有天數比較(3/5/10日)：用階段1選出的回看窗口，換「持有幾天後強制出場」，選總損益
   最高的天數。
3. 最終IS/OOS切分 + OOS bootstrap穩健性檢查：把階段1/2選出的(回看窗口, 持有天數)組合，
   真正驗證一次，報告PF、勝率、bootstrap正報酬比例、p值、90%信賴區間——這個專案看這類報告
   的習慣是bootstrap正報酬比例「想要>80%」當一個非正式的及格線，這裡沿用同樣的講法，不發明
   新的指標名稱或報告格式。

⚠️ 誠實揭露：這個策略假設(短期反轉效應在大型/高流動性股票子集合裡扣成本後仍然存在)是
外國市場的學術文獻結果，還沒在台灣股票期貨上驗證過，見short_reversal_engine.py模組
docstring。這支腳本跑出來的結果才是誠實的答案，不是先驗認定一定會賺錢。

用法：
    python3 compare_short_reversal.py --start 2023-09-15 --end 2026-09-14 --max-stocks 50
    python3 compare_short_reversal.py --slippage-pct 0.002 --top-n 5

跟mean_reversion_engine.py/momentum_breakout_engine.py共用的已知限制(結算日近似、
跌停鎖死/注意股處置股未實作、倖存者偏差、保證金追繳/強制斷頭沒有完整模擬)在這裡一樣成立，
請見 README。
"""
import argparse
import datetime
import os

import pandas as pd

from data_loader import load_price_data
from taifex_universe import STOCK_FUTURES_UNIVERSE
from short_reversal_engine import (
    run_short_reversal_backtest, precompute_all_reversal_indicators,
    LOOKBACK_WINDOW_OPTIONS, HOLD_DAYS_OPTIONS, DEFAULT_TOP_N, DEFAULT_ATR_STOP_MULT,
)
from mean_reversion_engine import summarize_mr
from robustness_analysis import (
    bootstrap_resample_pnl, summarize_bootstrap, pnl_excluding_top_n_trades, bootstrap_p_value,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_short_reversal")

MIN_TRADES_FOR_RANKING = 20
DEFAULT_HOLD_DAYS_FOR_WINDOW_STAGE = 5  # 階段1(回看窗口比較)用文獻基準的5天持有當固定對照


def _fmt_pf(pf):
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def split_is_oos(master_calendar, is_ratio=0.7):
    n = len(master_calendar)
    split_point = int(n * is_ratio)
    return master_calendar[:split_point], master_calendar[split_point:]


def _prefer_nonzero_trades(df):
    """跟compare_breakout.py同名函式一樣的理由：0筆交易的組合不該因為總損益剛好是0.0，
    就被誤選成贏家，只有全部候選都是0筆時才保留。"""
    non_zero = df[df["trade_count"] > 0]
    return non_zero if not non_zero.empty else df


def run_lookback_window_comparison(price_data, indicators_by_code, is_calendar, starting_capital,
                                    execution_kwargs, extra_kwargs=None):
    """階段1：只換回看窗口(3/5/10日)，固定用5天持有當基準出場天數，看「用幾天的報酬率
    排名候選」比較好。選出來的窗口會固定下來，階段2(持有天數比較)跟最終驗證都套用。"""
    extra_kwargs = extra_kwargs or {}
    rows = []
    for window in LOOKBACK_WINDOW_OPTIONS:
        trades = run_short_reversal_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, master_calendar=is_calendar,
            max_hold_days=DEFAULT_HOLD_DAYS_FOR_WINDOW_STAGE, starting_capital=starting_capital,
            lookback_window=window, allow_short=True, **execution_kwargs, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "lookback_window": window, "trade_count": stats["trade_count"],
            "profit_factor": stats["profit_factor"], "win_rate": stats["win_rate"],
            "total_pnl_ntd": stats["total_pnl_ntd"],
        })
        print(f"  回看{window}日 -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)
    return pd.DataFrame(rows)


def select_winning_lookback_window(window_df, min_trades=MIN_TRADES_FOR_RANKING):
    pool = _prefer_nonzero_trades(window_df)
    reliable = pool[pool["trade_count"] >= min_trades]
    if reliable.empty:
        reliable = pool
    if reliable.empty:
        return LOOKBACK_WINDOW_OPTIONS[1]  # 找不到任何可用結果時，退回5日(文獻基準)
    best_row = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    return int(best_row["lookback_window"])


def run_hold_days_comparison(price_data, indicators_by_code, is_calendar, starting_capital,
                              lookback_window, execution_kwargs, extra_kwargs=None):
    """階段2：用階段1選出的回看窗口，換「持有幾天後強制出場」(3/5/10日)，看哪個最好。"""
    extra_kwargs = extra_kwargs or {}
    rows = []
    for hold_days in HOLD_DAYS_OPTIONS:
        trades = run_short_reversal_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, master_calendar=is_calendar,
            max_hold_days=hold_days, starting_capital=starting_capital,
            lookback_window=lookback_window, allow_short=True, **execution_kwargs, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "hold_days": hold_days, "trade_count": stats["trade_count"],
            "profit_factor": stats["profit_factor"], "win_rate": stats["win_rate"],
            "total_pnl_ntd": stats["total_pnl_ntd"], "avg_hold_days": stats["avg_hold_days"],
        })
        print(f"  持有{hold_days}日 -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
              f"損益={stats['total_pnl_ntd']:,.0f}", flush=True)
    return pd.DataFrame(rows)


def select_winning_hold_days(hold_days_df, min_trades=MIN_TRADES_FOR_RANKING):
    pool = _prefer_nonzero_trades(hold_days_df)
    reliable = pool[pool["trade_count"] >= min_trades]
    if reliable.empty:
        reliable = pool
    if reliable.empty:
        return HOLD_DAYS_OPTIONS[1]  # 找不到任何可用結果時，退回5日(文獻基準)
    best_row = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    return int(best_row["hold_days"])


def evaluate_final_combo(label, price_data, indicators_by_code, is_calendar, oos_calendar,
                          starting_capital, lookback_window, hold_days, execution_kwargs, extra_kwargs=None):
    """把一組(回看窗口, 持有天數)組合，完整跑一次IS/OOS + OOS的bootstrap穩健性檢查。"""
    extra_kwargs = extra_kwargs or {}
    results = {"label": label}
    oos_trades = None
    for split_name, calendar in [("IS", is_calendar), ("OOS", oos_calendar)]:
        trades = run_short_reversal_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, master_calendar=calendar,
            max_hold_days=hold_days, starting_capital=starting_capital,
            lookback_window=lookback_window, allow_short=True, **execution_kwargs, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        results[split_name] = stats
        if split_name == "OOS":
            oos_trades = trades

    bootstrap_results = bootstrap_resample_pnl(oos_trades or [], n_resamples=1000, seed=42)
    bootstrap_stats = summarize_bootstrap(bootstrap_results)
    results["bootstrap"] = {
        **bootstrap_stats,
        "p_value": bootstrap_p_value(bootstrap_results),
        "pnl_excluding_top3_ntd": pnl_excluding_top_n_trades(oos_trades or [], n=3),
    }
    results["oos_trades"] = oos_trades
    return results


def main():
    parser = argparse.ArgumentParser(description="短期反轉策略 - 回看窗口 + 持有天數 + 最終組合驗證")
    parser.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=1095)).isoformat())
    parser.add_argument("--end", default=datetime.date.today().isoformat())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--starting-capital", type=float, default=200_000)
    parser.add_argument("--max-stocks", type=int, default=0,
                         help="限制掃描股票數量(0=全部249檔，測試時可以設小一點加快速度)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                         help="多方/空方各自取前幾名候選(預設5，同時最多可能有2倍top_n個部位)")
    parser.add_argument("--atr-stop-mult", type=float, default=DEFAULT_ATR_STOP_MULT,
                         help="ATR保護性停損倍數(安全網，不是主要出場依據，主要出場依據是固定持有天數)")
    parser.add_argument("--slippage-pct", type=float, default=0.0,
                         help="進場/停損出場的滑價假設(例如0.002代表0.2%%)，預設0(不模擬滑價)")
    parser.add_argument("--max-concurrent-positions", type=int, default=None,
                         help="同時最多持有的部位數，預設2*top_n(多空雙向各top_n名都進場)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    universe = dict(STOCK_FUTURES_UNIVERSE)
    INDEX_PROXY_CODE = "2330"
    if args.max_stocks > 0:
        universe = dict(list(universe.items())[: args.max_stocks])
        if INDEX_PROXY_CODE not in universe:
            universe[INDEX_PROXY_CODE] = STOCK_FUTURES_UNIVERSE[INDEX_PROXY_CODE]

    print(f"下載/讀取歷史資料 ({args.start} ~ {args.end})，共 {len(universe)} 檔標的 ...")
    price_data = load_price_data(universe, args.start, args.end, refresh=args.refresh)
    print(f"成功取得 {len(price_data)} / {len(universe)} 檔股票的資料")

    if INDEX_PROXY_CODE not in price_data:
        raise RuntimeError(f"參考交易日曆用的標的 {INDEX_PROXY_CODE} 沒有成功下載，無法繼續。")
    index_df = price_data[INDEX_PROXY_CODE]
    master_calendar = index_df.index
    is_calendar, oos_calendar = split_is_oos(master_calendar)
    print(f"樣本內(IS)天數={len(is_calendar)}，樣本外(OOS)天數={len(oos_calendar)}\n")

    print("預先計算全市場短期反轉指標 ...")
    indicators_by_code = precompute_all_reversal_indicators(price_data, universe)
    print(f"指標預計算完成，涵蓋 {len(indicators_by_code)} 檔股票\n")

    max_concurrent_positions = args.max_concurrent_positions
    if max_concurrent_positions is None:
        max_concurrent_positions = 2 * args.top_n

    execution_kwargs = dict(
        lots=2, top_n=args.top_n, atr_stop_mult=args.atr_stop_mult,
        max_concurrent_positions=max_concurrent_positions,
    )
    extra_kwargs = dict(slippage_pct=args.slippage_pct)

    if args.slippage_pct > 0:
        print(f"⚠️ 滑價模擬：進場/停損出場套用 {args.slippage_pct:.2%} 滑價\n")

    print(f"[階段1] 回看窗口比較(3/5/10日，只在IS內，固定{DEFAULT_HOLD_DAYS_FOR_WINDOW_STAGE}天持有當基準出場天數) ...")
    window_df = run_lookback_window_comparison(
        price_data, indicators_by_code, is_calendar, args.starting_capital,
        execution_kwargs, extra_kwargs=extra_kwargs,
    )
    window_df.to_csv(os.path.join(RESULTS_DIR, "lookback_window_comparison.csv"),
                      index=False, encoding="utf-8-sig")
    winning_window = select_winning_lookback_window(window_df)
    print(f"  → 選出回看窗口：{winning_window}日\n")

    print(f"[階段2] 固定持有天數比較(3/5/10日，只在IS內，用回看{winning_window}日) ...")
    hold_days_df = run_hold_days_comparison(
        price_data, indicators_by_code, is_calendar, args.starting_capital,
        winning_window, execution_kwargs, extra_kwargs=extra_kwargs,
    )
    hold_days_df.to_csv(os.path.join(RESULTS_DIR, "hold_days_comparison.csv"),
                         index=False, encoding="utf-8-sig")
    winning_hold_days = select_winning_hold_days(hold_days_df)
    print(f"  → 選出持有天數：{winning_hold_days}日\n")

    print(f"[階段3] 最終組合(回看{winning_window}日 + 持有{winning_hold_days}日)：IS vs OOS + bootstrap穩健性檢查 ...")
    final_result = evaluate_final_combo(
        f"最終組合(回看{winning_window}日, 持有{winning_hold_days}日)",
        price_data, indicators_by_code, is_calendar, oos_calendar, args.starting_capital,
        winning_window, winning_hold_days, execution_kwargs, extra_kwargs=extra_kwargs,
    )
    for split_name in ["IS", "OOS"]:
        stats = final_result[split_name]
        print(f"  {split_name} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
              f"損益={stats['total_pnl_ntd']:,.0f}")
    b = final_result["bootstrap"]
    print(f"  bootstrap：正報酬比例={b['pct_positive']:.1f}%, p值={b['p_value']:.3f}, "
          f"拿掉最大3筆後損益={b['pnl_excluding_top3_ntd']:,.0f}\n")
    pd.DataFrame(final_result["oos_trades"]).to_csv(
        os.path.join(RESULTS_DIR, "trades_OOS_final_combo.csv"), index=False, encoding="utf-8-sig",
    )

    summary_lines = [
        "=" * 100,
        "短期反轉(Short-Term Reversal)策略 回看窗口 + 持有天數 + 最終組合驗證",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}　"
        f"top_n={args.top_n}　最大並行部位={max_concurrent_positions}　"
        f"ATR安全網停損倍數={args.atr_stop_mult}　滑價：{args.slippage_pct:.2%}",
        "=" * 100,
        "\n--- 階段1：回看窗口比較(IS，固定5天持有當基準) ---",
    ]
    header1 = f"{'回看窗口':<10}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}"
    summary_lines.append(header1)
    summary_lines.append("-" * len(header1))
    for _, r in window_df.sort_values("total_pnl_ntd", ascending=False).iterrows():
        summary_lines.append(
            f"{str(int(r['lookback_window'])) + '日':<10}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}"
            f"{r['win_rate']:>8.1f}{r['total_pnl_ntd']:>14,.0f}"
        )

    summary_lines.append(f"\n--- 階段2：固定持有天數比較(IS，用回看{winning_window}日) ---")
    header2 = f"{'持有天數':<10}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'平均持有天':>10}{'總損益NT$':>14}"
    summary_lines.append(header2)
    summary_lines.append("-" * len(header2))
    for _, r in hold_days_df.sort_values("total_pnl_ntd", ascending=False).iterrows():
        summary_lines.append(
            f"{str(int(r['hold_days'])) + '日':<10}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}"
            f"{r['win_rate']:>8.1f}{r['avg_hold_days']:>10.1f}{r['total_pnl_ntd']:>14,.0f}"
        )

    summary_lines.append(f"\n--- 階段3：最終組合 IS vs OOS + bootstrap ---")
    summary_lines.append(f"\n[{final_result['label']}]")
    for split_name in ["IS", "OOS"]:
        stats = final_result[split_name]
        split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
        summary_lines.append(
            f"  {split_full}: {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
            f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
            f"總損益NT${stats['total_pnl_ntd']:,.0f}, "
            f"最大回撤NT${stats['max_drawdown_ntd']:,.0f}, 最大連續虧損{stats['max_consecutive_losses']}筆"
        )
    summary_lines.append(
        f"  [穩健性] OOS bootstrap 1000次重抽樣：平均總損益NT${b['mean']:,.0f}，"
        f"90%信賴區間=[NT${b['p5']:,.0f}, NT${b['p95']:,.0f}]，"
        f"正報酬比例={b['pct_positive']:.1f}%，p值={b['p_value']:.3f}，"
        f"拿掉最大3筆交易後總損益NT${b['pnl_excluding_top3_ntd']:,.0f}"
    )

    summary_lines.append(
        "\n判讀方式：先看階段1/2的IS總損益，確認回看窗口/持有天數的選擇不是在挑0筆或極少筆交易"
        "的雜訊組合(交易筆數太少的組合會被自動排除排名，見_prefer_nonzero_trades()/"
        "MIN_TRADES_FOR_RANKING)。再看階段3的OOS數字，這是比IS更誠實的參考依據。bootstrap"
        "正報酬比例是這個專案看這類報告的習慣：想要>80%才算「優勢在不同交易順序下都站得住」，"
        "p值越接近0越好，超過0.2以上不該當作已驗證的優勢看待。這支引擎測試的是外國市場學術"
        "文獻裡的短期反轉假說，還沒在台灣股票期貨上驗證過——這份報告的OOS/bootstrap數字"
        "才是誠實的答案，不是先驗認定一定會賺錢(見short_reversal_engine.py模組docstring)。"
    )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


if __name__ == "__main__":
    main()
