"""
主執行腳本：下載歷史資料 -> 分別用「原版(buggy ATR)」與「新版(correct ATR)」跑回測 -> 輸出比較報告。

使用方式（在這個資料夾底下）：
    python compare.py
    python compare.py --start 2022-01-01 --end 2025-01-01
    python compare.py --refresh          # 強制重新下載資料，不用快取

跑完之後會產生：
    results/trades_buggy.csv     原版每一筆交易明細
    results/trades_correct.csv   新版每一筆交易明細
    results/summary.txt          兩版績效比較摘要
    results/equity_curve.png     權益曲線比較圖
"""
import argparse
import os
import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 沒有圖形介面的環境也能存檔
import matplotlib.pyplot as plt

from data_loader import load_price_data, build_master_calendar, STOCK_FUTURES_WHITELIST
from engine import run_backtest, summarize

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def build_equity_series(trades):
    """把交易紀錄轉成一條依照出場日期排序的權益曲線 (起始本金正規化為1.0)。"""
    if not trades:
        return pd.Series(dtype=float)
    trades_sorted = sorted(trades, key=lambda t: t["exit_date"])
    dates = [t["exit_date"] for t in trades_sorted]
    equity = [1.0]
    for t in trades_sorted:
        equity.append(equity[-1] * (1 + t["return_pct"]))
    equity = equity[1:]
    return pd.Series(equity, index=pd.DatetimeIndex(dates))


def main():
    parser = argparse.ArgumentParser(description="比較原版(buggy ATR)與新版(correct ATR)策略績效")
    parser.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=730)).isoformat(),
                         help="回測起始日期 YYYY-MM-DD，預設抓最近兩年")
    parser.add_argument("--end", default=datetime.date.today().isoformat(),
                         help="回測結束日期 YYYY-MM-DD，預設今天")
    parser.add_argument("--refresh", action="store_true", help="強制重新下載歷史資料，不使用快取")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"下載/讀取歷史資料 ({args.start} ~ {args.end}) ...")
    price_data = load_price_data(STOCK_FUTURES_WHITELIST, args.start, args.end, refresh=args.refresh)
    print(f"成功取得 {len(price_data)} / {len(STOCK_FUTURES_WHITELIST)} 檔股票的資料\n")

    if len(price_data) < 5:
        print("警告：成功下載的股票數量過少，回測結果可能沒有代表性。"
              "請檢查網路連線，或用 --refresh 重新下載。")

    master_calendar = build_master_calendar(price_data)

    results = {}
    for mode, label in [("buggy", "原版 (錯誤ATR)"), ("correct", "新版 (正確ATR)")]:
        print(f"開始回測：{label} ...")
        trades = run_backtest(price_data, STOCK_FUTURES_WHITELIST, master_calendar, atr_mode=mode)
        stats = summarize(trades)
        results[mode] = {"label": label, "trades": trades, "stats": stats}
        print(f"  完成，共 {stats['trade_count']} 筆交易\n")

    # ---- 輸出交易明細 CSV ----
    for mode in ["buggy", "correct"]:
        trades = results[mode]["trades"]
        df = pd.DataFrame(trades)
        csv_path = os.path.join(RESULTS_DIR, f"trades_{mode}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"已輸出：{csv_path}")

    # ---- 輸出比較摘要 ----
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(f"回測期間：{args.start} ~ {args.end}")
    summary_lines.append("=" * 60)
    header = f"{'指標':<16}{'原版(錯誤ATR)':>18}{'新版(正確ATR)':>18}"
    summary_lines.append(header)
    summary_lines.append("-" * len(header))

    rows = [
        ("交易次數", "trade_count", "{:d}"),
        ("勝率(%)", "win_rate", "{:.2f}"),
        ("平均每筆報酬(%)", "avg_return_pct", "{:.2f}"),
        ("累積報酬(%)", "cumulative_return_pct", "{:.2f}"),
        ("最大回撤(%)", "max_drawdown_pct", "{:.2f}"),
    ]
    for label, key, fmt in rows:
        v1 = results["buggy"]["stats"][key]
        v2 = results["correct"]["stats"][key]
        v1_str = fmt.format(int(v1)) if key == "trade_count" else fmt.format(v1)
        v2_str = fmt.format(int(v2)) if key == "trade_count" else fmt.format(v2)
        summary_lines.append(f"{label:<16}{v1_str:>18}{v2_str:>18}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")

    # ---- 畫權益曲線比較圖 ----
    plt.figure(figsize=(10, 6))
    for mode, color in [("buggy", "tab:red"), ("correct", "tab:blue")]:
        eq = build_equity_series(results[mode]["trades"])
        if len(eq) > 0:
            plt.plot(eq.index, eq.values, label=results[mode]["label"], color=color)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("策略權益曲線比較（起始本金正規化為 1.0）")
    plt.xlabel("出場日期")
    plt.ylabel("權益倍數")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "equity_curve.png")
    plt.savefig(chart_path, dpi=150)
    print(f"已輸出：{chart_path}")


if __name__ == "__main__":
    main()
