"""
均值回歸雙向策略 - 主執行腳本。

跑4種組合：{短線版, 中期版} x {純多, 多空雙向}，並做樣本內/樣本外切分驗證，
避免RSI/布林通道這組參數是「調到歷史資料最好看」的巧合。

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
from mean_reversion_engine import run_mean_reversion_backtest, summarize_mr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_mean_reversion")

CONFIGS = [
    ("短線版_多空雙向", 5, True),
    ("短線版_純多", 5, False),
    ("中期版_多空雙向", 15, True),
    ("中期版_純多", 15, False),
]


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
    parser = argparse.ArgumentParser(description="均值回歸雙向策略回測")
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
    if args.max_stocks > 0:
        universe = dict(list(universe.items())[: args.max_stocks])

    print(f"下載/讀取歷史資料 ({args.start} ~ {args.end})，共 {len(universe)} 檔標的 ...")
    price_data = load_price_data(universe, args.start, args.end, refresh=args.refresh)
    print(f"成功取得 {len(price_data)} / {len(universe)} 檔股票的資料")

    print("下載大盤氛圍代理指標 (0050.TW) ...")
    index_data = load_price_data({"0050": {"otc": False}}, args.start, args.end, refresh=args.refresh)
    if "0050" not in index_data:
        raise RuntimeError("無法取得0050.TW資料，大盤氛圍濾網無法運作，請檢查網路連線")
    index_df = index_data["0050"]

    master_calendar = index_df.index
    is_calendar, oos_calendar = split_is_oos(master_calendar)
    print(f"樣本內(IS)天數={len(is_calendar)}，樣本外(OOS)天數={len(oos_calendar)}\n")

    summary_lines = [
        "=" * 90,
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        "=" * 90,
    ]

    for split_name, calendar in [("樣本內(IS)", is_calendar), ("樣本外(OOS)", oos_calendar)]:
        summary_lines.append(f"\n--- {split_name} ---")
        header = f"{'組合':<20}{'交易數':>8}{'多/空':>10}{'勝率%':>8}{'總損益NT$':>14}{'最大回撤NT$':>14}{'期末資金NT$':>14}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))

        for label, max_hold, allow_short in CONFIGS:
            trades = run_mean_reversion_backtest(
                price_data, index_df, universe, calendar,
                max_hold_days=max_hold, starting_capital=args.starting_capital,
                allow_short=allow_short, lots=2,
            )
            stats = summarize_mr(trades, args.starting_capital)
            long_short = f"{stats['long_count']}/{stats['short_count']}"
            summary_lines.append(
                f"{label:<20}{stats['trade_count']:>8}{long_short:>10}{stats['win_rate']:>8.1f}"
                f"{stats['total_pnl_ntd']:>14,.0f}{stats['max_drawdown_ntd']:>14,.0f}{stats['ending_equity_ntd']:>14,.0f}"
            )

            csv_path = os.path.join(RESULTS_DIR, f"trades_{split_name.split('(')[0]}_{label}.csv")
            pd.DataFrame(trades).to_csv(csv_path, index=False, encoding="utf-8-sig")

            if split_name.startswith("樣本外"):
                # 樣本外的權益曲線才是真正有意義的「沒偷看答案」驗證，畫圖只畫這組
                eq = build_equity_series(trades, args.starting_capital)
                if len(eq) > 0:
                    plt.plot(eq.index, eq.values, label=label)

    plt.axhline(args.starting_capital, color="gray", linestyle="--", linewidth=0.8)
    plt.title("樣本外(OOS)權益曲線比較 - 均值回歸雙向策略")
    plt.xlabel("出場日期")
    plt.ylabel("帳戶權益 (NT$)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "oos_equity_curve.png")
    plt.savefig(chart_path, dpi=150)

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")
    print(f"已輸出：{chart_path}")
    print("\n提醒：樣本外(OOS)的數字才是比較誠實的參考依據，"
          "樣本內(IS)數字容易受到參數剛好適合這段歷史資料的巧合影響，僅供對照。")


if __name__ == "__main__":
    main()
