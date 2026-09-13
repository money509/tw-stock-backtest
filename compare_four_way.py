"""
四合一策略比較：動能分數公式(原版/量能加權) x 部位大小規則(固定2口/風險動態調整)。

ATR一律使用「正確版本」(main.py已修正過的邏輯)，因為那已經確認是對的，
這次要比較的是另外兩個維度，避免混在一起看不出誰影響了什麼。

使用方式：
    python compare_four_way.py
    python compare_four_way.py --start 2022-01-01 --end 2025-01-01 --risk-per-trade 30000
"""
import argparse
import os
import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_price_data, build_master_calendar, STOCK_FUTURES_WHITELIST
from engine import run_backtest, summarize
from sizing import apply_sizing

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_four_way")

SCORE_MODES = [("original", "原版動能分數"), ("volume_weighted", "量能加權動能分數")]
SIZING_MODES = [("fixed2", "固定2口"), ("risk_based", "風險動態調整口數")]


def build_equity_series_ntd(trades, starting_capital):
    if not trades:
        return pd.Series(dtype=float)
    trades_sorted = sorted(trades, key=lambda t: t["exit_date"])
    dates = [t["exit_date"] for t in trades_sorted]
    equity = [starting_capital]
    for t in trades_sorted:
        equity.append(equity[-1] + t["pnl_ntd"])
    equity = equity[1:]
    return pd.Series(equity, index=pd.DatetimeIndex(dates))


def main():
    parser = argparse.ArgumentParser(description="比較動能分數公式 x 部位大小規則的四種組合績效")
    parser.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=730)).isoformat())
    parser.add_argument("--end", default=datetime.date.today().isoformat())
    parser.add_argument("--refresh", action="store_true", help="強制重新下載歷史資料")
    parser.add_argument("--risk-per-trade", type=float, default=20000,
                         help="風險動態調整口數模式下，每筆交易願意承受的風險金額(新台幣)，預設20000")
    parser.add_argument("--starting-capital", type=float, default=1_000_000,
                         help="模擬帳戶起始資金(新台幣)，用來畫權益曲線，預設1,000,000")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"下載/讀取歷史資料 ({args.start} ~ {args.end}) ...")
    price_data = load_price_data(STOCK_FUTURES_WHITELIST, args.start, args.end, refresh=args.refresh)
    print(f"成功取得 {len(price_data)} / {len(STOCK_FUTURES_WHITELIST)} 檔股票的資料\n")

    if len(price_data) < 5:
        print("警告：成功下載的股票數量過少，回測結果可能沒有代表性。")

    master_calendar = build_master_calendar(price_data)

    # 先依 score_mode 各跑一次完整walk-forward，取得原始交易紀錄
    # (進出場的價格/時間只取決於score_mode，跟sizing_mode無關，所以只需要跑2次，不用跑4次)
    raw_trades_by_score = {}
    for score_mode, label in SCORE_MODES:
        print(f"回測選股邏輯：{label} ...")
        trades = run_backtest(price_data, STOCK_FUTURES_WHITELIST, master_calendar,
                               atr_mode="correct", score_mode=score_mode)
        raw_trades_by_score[score_mode] = trades
        print(f"  完成，共 {len(trades)} 筆交易\n")

    # 針對每個 score_mode 的交易紀錄，分別套用兩種 sizing 規則，組成4種結果
    combo_results = {}
    for score_mode, score_label in SCORE_MODES:
        raw_trades = raw_trades_by_score[score_mode]
        for sizing_mode, sizing_label in SIZING_MODES:
            sized_trades = apply_sizing(raw_trades, sizing_mode, risk_per_trade_ntd=args.risk_per_trade)
            stats = summarize(sized_trades, starting_capital=args.starting_capital)
            combo_key = f"{score_mode}__{sizing_mode}"
            combo_label = f"{score_label} + {sizing_label}"
            combo_results[combo_key] = {"label": combo_label, "trades": sized_trades, "stats": stats}

            csv_path = os.path.join(RESULTS_DIR, f"trades_{combo_key}.csv")
            pd.DataFrame(sized_trades).to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"已輸出：{csv_path}")

    # ---- 輸出比較摘要表 ----
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append(f"回測期間：{args.start} ~ {args.end}")
    summary_lines.append(f"風險動態調整口數：每筆風險金額 NT${args.risk_per_trade:,.0f}")
    summary_lines.append(f"模擬起始資金：NT${args.starting_capital:,.0f}")
    summary_lines.append("=" * 70)

    combo_keys = list(combo_results.keys())
    col_width = 16
    header = f"{'指標':<18}" + "".join(f"{k:>{col_width}}" for k in combo_keys)
    summary_lines.append(header)
    summary_lines.append("-" * len(header))

    rows = [
        ("交易次數", "trade_count", "{:.0f}"),
        ("勝率(%)", "win_rate", "{:.2f}"),
        ("平均每筆報酬(%)", "avg_return_pct", "{:.2f}"),
        ("累積報酬(%)", "cumulative_return_pct", "{:.2f}"),
        ("最大回撤(%)", "max_drawdown_pct", "{:.2f}"),
        ("總損益(NT$)", "total_pnl_ntd", "{:,.0f}"),
        ("最大回撤(NT$)", "max_drawdown_ntd", "{:,.0f}"),
        ("期末資金(NT$)", "ending_equity_ntd", "{:,.0f}"),
    ]
    for label, key, fmt in rows:
        line = f"{label:<18}"
        for combo_key in combo_keys:
            v = combo_results[combo_key]["stats"].get(key, float("nan"))
            line += f"{fmt.format(v):>{col_width}}"
        summary_lines.append(line)

    summary_lines.append("")
    summary_lines.append("組合對照：")
    for combo_key in combo_keys:
        summary_lines.append(f"  {combo_key:<30} = {combo_results[combo_key]['label']}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")

    # ---- 畫4條權益曲線 (以新台幣計，因為風險動態調整口數的意義在金額上才看得出來) ----
    plt.figure(figsize=(11, 6))
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:green"]
    for (combo_key, color) in zip(combo_keys, colors):
        eq = build_equity_series_ntd(combo_results[combo_key]["trades"], args.starting_capital)
        if len(eq) > 0:
            plt.plot(eq.index, eq.values, label=combo_results[combo_key]["label"], color=color)
    plt.axhline(args.starting_capital, color="gray", linestyle="--", linewidth=0.8)
    plt.title("四種組合權益曲線比較（新台幣）")
    plt.xlabel("出場日期")
    plt.ylabel("帳戶權益 (NT$)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "equity_curve_four_way.png")
    plt.savefig(chart_path, dpi=150)
    print(f"已輸出：{chart_path}")


if __name__ == "__main__":
    main()
