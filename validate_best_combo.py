"""
獨立時間驗證：只測「中期15天_RSI35_回對側軌道」這一組在2023-09~2026-09這段期間
表現最好的參數，換到一段完全不同氛圍的歷史期間，看它是不是真的有效，
還是只是剛好搭到那段期間的多頭順風車。

同時會統計這段驗證期間裡，大盤氛圍濾網判斷「多頭/空頭/中性」的天數比例，
直接回答「這組參數的獲利，是不是主要來自搭順風車」這個問題。

使用方式：
    python validate_best_combo.py
    python validate_best_combo.py --start 2021-01-01 --end 2023-01-01
"""
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_price_data
from taifex_universe import STOCK_FUTURES_UNIVERSE
from mean_reversion_engine import (
    run_mean_reversion_backtest, summarize_mr,
    precompute_all_indicators, precompute_regime_series, compute_regime,
)
from compare_mean_reversion import build_equity_series

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_validation")

# 這是上一輪12組合比較裡，樣本外(OOS)表現最好的那一組參數
BEST_COMBO = {"hold_days": 15, "rsi_threshold": 35, "target_mode": "opposite_band"}
BEST_COMBO_LABEL = "中期15天_RSI35_回對側軌道"


def main():
    parser = argparse.ArgumentParser(description="獨立時間驗證最佳參數組合，確認不是搭順風車")
    parser.add_argument("--start", default="2021-01-01", help="驗證期間起始日期，預設2021-01-01")
    parser.add_argument("--end", default="2023-01-01", help="驗證期間結束日期，預設2023-01-01")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--starting-capital", type=float, default=200_000)
    parser.add_argument("--max-stocks", type=int, default=0,
                         help="限制掃描股票數量(0=全部320檔)")
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
        raise RuntimeError(f"大盤氛圍代理指標 {INDEX_PROXY_CODE} 沒有成功下載，無法繼續。")
    index_df = price_data[INDEX_PROXY_CODE]
    master_calendar = index_df.index

    print("預先計算全市場技術指標 ...")
    indicators_by_code = precompute_all_indicators(price_data, universe)
    regime_series = precompute_regime_series(index_df)

    # ---- 先統計這段期間的大盤氛圍分布，用來解讀等一下的績效數字 ----
    regime_counts = {"bull": 0, "bear": 0, "neutral": 0}
    for date in master_calendar:
        regime_counts[compute_regime(regime_series, date)] += 1
    total_days = sum(regime_counts.values())
    print(f"\n此驗證期間大盤氛圍分布："
          f"多頭 {regime_counts['bull']}天({regime_counts['bull']/total_days*100:.0f}%)　"
          f"空頭 {regime_counts['bear']}天({regime_counts['bear']/total_days*100:.0f}%)　"
          f"中性(資料不足) {regime_counts['neutral']}天({regime_counts['neutral']/total_days*100:.0f}%)\n")

    print(f"開始回測：{BEST_COMBO_LABEL} ...")
    trades = run_mean_reversion_backtest(
        price_data, indicators_by_code, regime_series, master_calendar,
        max_hold_days=BEST_COMBO["hold_days"], starting_capital=args.starting_capital,
        allow_short=True, lots=2,
        rsi_long_threshold=BEST_COMBO["rsi_threshold"], rsi_short_threshold=100 - BEST_COMBO["rsi_threshold"],
        target_mode=BEST_COMBO["target_mode"],
    )
    stats = summarize_mr(trades, args.starting_capital)

    summary_lines = [
        "=" * 80,
        f"獨立時間驗證：{BEST_COMBO_LABEL}",
        f"驗證期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        "=" * 80,
        "",
        f"大盤氛圍分布：多頭{regime_counts['bull']/total_days*100:.0f}% / "
        f"空頭{regime_counts['bear']/total_days*100:.0f}% / 中性{regime_counts['neutral']/total_days*100:.0f}%",
        "",
        f"交易次數：{stats['trade_count']} (多{stats['long_count']}/空{stats['short_count']})",
        f"勝率：{stats['win_rate']:.1f}%",
        f"平均每筆報酬：{stats['avg_return_pct']:.2f}%",
        f"總損益：NT${stats['total_pnl_ntd']:,.0f}",
        f"最大回撤：NT${stats['max_drawdown_ntd']:,.0f}",
        f"期末資金：NT${stats['ending_equity_ntd']:,.0f}",
        "",
    ]

    # ---- 解讀判斷：跟原本那段期間(2023-2026 OOS)對照 ----
    original_oos_pnl = 25086  # 上一輪驗證出來的原始OOS總損益，寫死在這裡方便對照
    if stats["total_pnl_ntd"] > 0:
        summary_lines.append(
            f"【判讀】這組參數換到完全不同的歷史期間，總損益仍然是正的(NT${stats['total_pnl_ntd']:,.0f})，"
            f"初步支持「這組參數本身有一定道理」，不是純粹搭上一輪(2023-2026)那段多頭順風車。"
            f"不過仍建議再用第三段不同期間交叉驗證，不要只憑兩次結果就完全定案。"
        )
    else:
        summary_lines.append(
            f"【判讀】這組參數換到不同歷史期間後轉為虧損(NT${stats['total_pnl_ntd']:,.0f})，"
            f"這是一個警訊：上一輪(2023-2026)那組+NT${original_oos_pnl:,.0f}的正報酬，"
            f"很可能主要來自搭上那段期間剛好的多頭順風車，而不是這組參數本身特別有效。"
            f"建議不要直接拿這組參數去用，應該回頭檢視進場條件本身。"
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    csv_path = os.path.join(RESULTS_DIR, "trades.csv")
    pd.DataFrame(trades).to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    eq = build_equity_series(trades, args.starting_capital)
    if len(eq) > 0:
        plt.plot(eq.index, eq.values, label=BEST_COMBO_LABEL, color="tab:blue")
    plt.axhline(args.starting_capital, color="gray", linestyle="--", linewidth=0.8)
    plt.title(f"獨立時間驗證權益曲線：{BEST_COMBO_LABEL}\n({args.start} ~ {args.end})")
    plt.xlabel("出場日期")
    plt.ylabel("帳戶權益 (NT$)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "validation_equity_curve.png")
    plt.savefig(chart_path, dpi=150)

    print(f"\n已輸出：{summary_path}")
    print(f"已輸出：{csv_path}")
    print(f"已輸出：{chart_path}")


if __name__ == "__main__":
    main()
