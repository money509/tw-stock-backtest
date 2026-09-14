"""
技術面 + 籌碼面 組合策略比較。

用上一輪找出的技術面參數(中期15天_RSI35_回對側軌道)當基礎，
比較「純技術面」vs「技術面+外資連續買賣超確認」，看加入籌碼面訊號是不是真的有幫助，
而不是想當然爾地假設「機構認同」一定會讓策略變好。

使用方式：
    python compare_technical_plus_chip.py
    python compare_technical_plus_chip.py --start 2021-01-01 --end 2023-01-01
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
from chip_data_loader import load_chip_data, precompute_chip_streak
from mean_reversion_engine import (
    run_mean_reversion_backtest, summarize_mr,
    precompute_all_indicators, precompute_regime_series,
)
from compare_mean_reversion import build_equity_series

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_chip")

# 沿用上一輪驗證中期表現最穩的技術面參數當基礎
BASE_COMBO = {"hold_days": 15, "rsi_threshold": 35, "target_mode": "opposite_band"}

CHIP_CONFIRM_OPTIONS = [
    ("純技術面(無籌碼確認)", 0),
    ("+外資連續買賣超2天確認", 2),
    ("+外資連續買賣超3天確認", 3),
    ("+外資連續買賣超5天確認", 5),
]


def main():
    parser = argparse.ArgumentParser(description="技術面+籌碼面組合策略比較")
    parser.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=1095)).isoformat())
    parser.add_argument("--end", default=datetime.date.today().isoformat())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--starting-capital", type=float, default=200_000)
    parser.add_argument("--max-stocks", type=int, default=0,
                         help="限制掃描股票數量(0=全部320檔，僅限技術面資料；籌碼面目前只涵蓋上市股)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    universe = dict(STOCK_FUTURES_UNIVERSE)
    INDEX_PROXY_CODE = "2330"
    if args.max_stocks > 0:
        universe = dict(list(universe.items())[: args.max_stocks])
        if INDEX_PROXY_CODE not in universe:
            universe[INDEX_PROXY_CODE] = STOCK_FUTURES_UNIVERSE[INDEX_PROXY_CODE]

    print(f"下載/讀取股價資料 ({args.start} ~ {args.end})，共 {len(universe)} 檔標的 ...")
    price_data = load_price_data(universe, args.start, args.end, refresh=args.refresh)
    print(f"成功取得 {len(price_data)} / {len(universe)} 檔股票的股價資料")

    if INDEX_PROXY_CODE not in price_data:
        raise RuntimeError(f"大盤氛圍代理指標 {INDEX_PROXY_CODE} 沒有成功下載，無法繼續。")
    index_df = price_data[INDEX_PROXY_CODE]
    master_calendar = index_df.index

    print("\n下載三大法人籌碼面資料 (這步驟是逐日抓取，可能需要一些時間) ...")
    chip_data = load_chip_data(args.start, args.end, universe_codes=set(universe.keys()), refresh=args.refresh)
    print(f"取得 {len(chip_data)} 檔股票的籌碼面資料\n")

    print("預先計算技術指標與籌碼面連續天數 ...")
    indicators_by_code = precompute_all_indicators(price_data, universe)
    regime_series = precompute_regime_series(index_df)
    chip_streak_by_code = {
        code: precompute_chip_streak(df, "foreign_net")
        for code, df in chip_data.items()
    }
    print(f"完成，{len(chip_streak_by_code)} 檔股票有籌碼面資料可用\n")

    results = {}
    for label, min_days in CHIP_CONFIRM_OPTIONS:
        trades = run_mean_reversion_backtest(
            price_data, indicators_by_code, regime_series, master_calendar,
            max_hold_days=BASE_COMBO["hold_days"], starting_capital=args.starting_capital,
            allow_short=True, lots=2,
            rsi_long_threshold=BASE_COMBO["rsi_threshold"], rsi_short_threshold=100 - BASE_COMBO["rsi_threshold"],
            target_mode=BASE_COMBO["target_mode"],
            chip_streak_by_code=chip_streak_by_code, min_chip_confirm_days=min_days,
        )
        stats = summarize_mr(trades, args.starting_capital)
        results[label] = (trades, stats)
        pd.DataFrame(trades).to_csv(os.path.join(RESULTS_DIR, f"trades_{label}.csv"), index=False, encoding="utf-8-sig")
        print(f"  完成: {label} 交易數={stats['trade_count']} 總損益=NT${stats['total_pnl_ntd']:,.0f}")

    summary_lines = [
        "=" * 90,
        f"技術面+籌碼面比較　基礎參數：{BASE_COMBO}",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        "=" * 90,
        "",
        f"{'組合':<28}{'交易數':>8}{'多/空':>10}{'勝率%':>8}{'總損益NT$':>14}{'最大回撤NT$':>14}{'期末資金NT$':>14}",
        "-" * 90,
    ]
    for label, min_days in CHIP_CONFIRM_OPTIONS:
        trades, stats = results[label]
        long_short = f"{stats['long_count']}/{stats['short_count']}"
        summary_lines.append(
            f"{label:<28}{stats['trade_count']:>8}{long_short:>10}{stats['win_rate']:>8.1f}"
            f"{stats['total_pnl_ntd']:>14,.0f}{stats['max_drawdown_ntd']:>14,.0f}{stats['ending_equity_ntd']:>14,.0f}"
        )

    summary_lines.append("")
    summary_lines.append("提醒：籌碼面資料目前只涵蓋上市(TWSE)股票，上櫃股票在開啟籌碼確認時會被自動排除，")
    summary_lines.append("這會讓「有籌碼確認」的版本候選池比「純技術面」版本小，也是造成交易次數減少的部分原因，")
    summary_lines.append("不完全是「篩選變嚴格」單一因素，解讀勝率/報酬率差異時要把這點納入考量。")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    plt.figure(figsize=(10, 6))
    for label, min_days in CHIP_CONFIRM_OPTIONS:
        trades, _ = results[label]
        eq = build_equity_series(trades, args.starting_capital)
        if len(eq) > 0:
            plt.plot(eq.index, eq.values, label=label)
    plt.axhline(args.starting_capital, color="gray", linestyle="--", linewidth=0.8)
    plt.title("純技術面 vs 技術面+籌碼確認 權益曲線比較")
    plt.xlabel("出場日期")
    plt.ylabel("帳戶權益 (NT$)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "chip_comparison_equity.png")
    plt.savefig(chart_path, dpi=150)
    print(f"\n已輸出：{os.path.join(RESULTS_DIR, 'summary.txt')}")
    print(f"已輸出：{chart_path}")


if __name__ == "__main__":
    main()
