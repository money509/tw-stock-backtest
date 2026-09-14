"""
均值回歸雙向策略 - 主執行腳本。

跑「持有天數 x RSI門檻 x 出場目標」共 3x2x2=12 種組合，並做樣本內/樣本外切分驗證，
避免參數是「調到歷史資料最好看」的巧合。多空雙向固定開啟(大盤氛圍濾網會自動視情況停用其中一邊)。

使用方式：
    python compare_mean_reversion.py
    python compare_mean_reversion.py --start 2022-01-01 --end 2025-01-01 --starting-capital 200000
"""
import argparse
import os
import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_price_data
from taifex_universe import STOCK_FUTURES_UNIVERSE
from mean_reversion_engine import (
    run_mean_reversion_backtest, summarize_mr,
    precompute_all_indicators, precompute_regime_series,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_mean_reversion")

HOLD_DAYS_OPTIONS = [("短線5天", 5), ("中期15天", 15), ("長版30天", 30)]
RSI_THRESHOLD_OPTIONS = [("RSI30", 30), ("RSI35", 35)]
TARGET_MODE_OPTIONS = [("回中軌", "mid_band"), ("回對側軌道", "opposite_band")]


def build_combos():
    combos = []
    for hold_label, hold_days in HOLD_DAYS_OPTIONS:
        for rsi_label, rsi_val in RSI_THRESHOLD_OPTIONS:
            for target_label, target_mode in TARGET_MODE_OPTIONS:
                key = f"{hold_label}_{rsi_label}_{target_label}"
                combos.append({
                    "key": key, "hold_days": hold_days,
                    "rsi_threshold": rsi_val, "target_mode": target_mode,
                })
    return combos


def build_equity_series(trades, starting_capital):
    if not trades:
        return pd.Series(dtype=float)
    trades_sorted = sorted(trades, key=lambda t: t["exit_date"])
    dates = [t["exit_date"] for t in trades_sorted]
    equity = [starting_capital]
    for t in trades_sorted:
        equity.append(equity[-1] + t["pnl_ntd"])
    return pd.Series(equity[1:], index=pd.DatetimeIndex(dates))


def split_is_oos(master_calendar, is_ratio=0.7):
    n = len(master_calendar)
    split_point = int(n * is_ratio)
    return master_calendar[:split_point], master_calendar[split_point:]


def main():
    parser = argparse.ArgumentParser(description="均值回歸雙向策略回測 - 持有天數x RSI門檻x出場目標 多維度比較")
    parser.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=1095)).isoformat(),
                         help="回測起始日期，預設抓最近3年")
    parser.add_argument("--end", default=datetime.date.today().isoformat())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--starting-capital", type=float, default=200_000)
    parser.add_argument("--max-stocks", type=int, default=0,
                         help="限制掃描股票數量(0=全部320檔，測試時可以設小一點加快速度)")
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
        raise RuntimeError(
            f"大盤氛圍代理指標 {INDEX_PROXY_CODE} 沒有成功下載，無法繼續。"
            f"請確認 --max-stocks 有涵蓋到這檔，或改用其他已確認能下載成功的高流動性股票代碼。"
        )
    index_df = price_data[INDEX_PROXY_CODE]

    master_calendar = index_df.index
    is_calendar, oos_calendar = split_is_oos(master_calendar)
    print(f"樣本內(IS)天數={len(is_calendar)}，樣本外(OOS)天數={len(oos_calendar)}\n")

    combos = build_combos()
    print(f"共 {len(combos)} 種組合 (持有天數x RSI門檻x出場目標)，開始逐一回測 ...\n")

    print("預先計算全市場技術指標 (RSI/布林通道/60日均線/ATR)，之後24次回測共用，不用重複計算 ...")
    indicators_by_code = precompute_all_indicators(price_data, universe)
    regime_series = precompute_regime_series(index_df)
    print(f"指標預計算完成，涵蓋 {len(indicators_by_code)} 檔股票\n")

    all_results = {}  # combo_key -> {"IS": (trades, stats), "OOS": (trades, stats)}
    for combo in combos:
        all_results[combo["key"]] = {}
        for split_name, calendar in [("IS", is_calendar), ("OOS", oos_calendar)]:
            trades = run_mean_reversion_backtest(
                price_data, indicators_by_code, regime_series, calendar,
                max_hold_days=combo["hold_days"], starting_capital=args.starting_capital,
                allow_short=True, lots=2,
                rsi_long_threshold=combo["rsi_threshold"], rsi_short_threshold=100 - combo["rsi_threshold"],
                target_mode=combo["target_mode"],
            )
            stats = summarize_mr(trades, args.starting_capital)
            all_results[combo["key"]][split_name] = (trades, stats)

            csv_path = os.path.join(RESULTS_DIR, f"trades_{split_name}_{combo['key']}.csv")
            pd.DataFrame(trades).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  完成: {combo['key']}", flush=True)

    # ---- 輸出比較摘要，依OOS總損益由高到低排序，最容易獲利的組合排在最上面 ----
    sorted_keys = sorted(all_results.keys(), key=lambda k: all_results[k]["OOS"][1]["total_pnl_ntd"], reverse=True)

    summary_lines = [
        "=" * 100,
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        f"共 {len(combos)} 種組合，依樣本外(OOS)總損益由高到低排序",
        "=" * 100,
    ]

    for split_name in ["IS", "OOS"]:
        split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
        summary_lines.append(f"\n--- {split_full} ---")
        header = f"{'組合':<32}{'交易數':>8}{'多/空':>10}{'勝率%':>8}{'總損益NT$':>14}{'最大回撤NT$':>14}{'期末資金NT$':>14}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        for key in sorted_keys:
            trades, stats = all_results[key][split_name]
            long_short = f"{stats['long_count']}/{stats['short_count']}"
            summary_lines.append(
                f"{key:<32}{stats['trade_count']:>8}{long_short:>10}{stats['win_rate']:>8.1f}"
                f"{stats['total_pnl_ntd']:>14,.0f}{stats['max_drawdown_ntd']:>14,.0f}{stats['ending_equity_ntd']:>14,.0f}"
            )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")

    # ---- 權益曲線只畫OOS表現最好的前4名，避免12條線疊在一起看不清楚 ----
    plt.figure(figsize=(11, 6))
    top4_keys = sorted_keys[:4]
    for key in top4_keys:
        trades, _ = all_results[key]["OOS"]
        eq = build_equity_series(trades, args.starting_capital)
        if len(eq) > 0:
            plt.plot(eq.index, eq.values, label=key)
    plt.axhline(args.starting_capital, color="gray", linestyle="--", linewidth=0.8)
    plt.title("樣本外(OOS)權益曲線 - 表現最好的前4種組合")
    plt.xlabel("出場日期")
    plt.ylabel("帳戶權益 (NT$)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "oos_equity_curve_top4.png")
    plt.savefig(chart_path, dpi=150)
    print(f"已輸出：{chart_path}")

    print("\n提醒：樣本外(OOS)的數字才是比較誠實的參考依據，"
          "樣本內(IS)數字容易受到參數剛好適合這段歷史資料的巧合影響，僅供對照。"
          "12種組合同時比較時，也要注意「剛好在這批組合裡表現最好」本身也可能有一定運氣成分，"
          "同一組合最好能在不同時間區間重複驗證過，才比較有信心。")


if __name__ == "__main__":
    main()
