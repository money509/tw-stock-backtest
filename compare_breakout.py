"""
compare_breakout.py
====================
右側順勢突破策略 - 主執行腳本。

跟 compare_mean_reversion.py 一樣的整體架構(讀資料、算指標、IS/OOS切分、輸出summary)，
但這裡除了「持有天數」以外，主要目的是回答選股邏輯本身的問題：

    量比、相對大盤強弱、RSI由下往上穿越50、MACD轉正、外資連續買超確認——
    這5個可以搭配的訊號，哪些對「突破後會不會延續」真的有預測力？

做法分兩階段：
1. 單一訊號拆解(ablation)：5個訊號每次只開一個，其餘權重設0，看誰單獨表現最好
   (跟 sweep_signal_ablation.py 對隔日衝做的事完全同一個精神)。
2. 綜合比較：拿階段1排名前面的訊號組成幾種權重組合，跟基準(5訊號等權重)一起，
   分別跑「短線5天/中期15天」兩種持有天數，共同在IS/OOS上比較。

用法：
    python3 compare_breakout.py --start 2023-09-15 --end 2026-09-14 --max-stocks 50
    python3 compare_breakout.py --with-chip-confirm   # 額外抓籌碼資料，把外資買超確認也接上測試

⚠️ 誠實揭露：跟 mean_reversion_engine.py 共用的已知限制(結算日近似、大盤氛圍濾網用0050
代理、跌停鎖死/除權息/注意股處置股未實作、倖存者偏差)在這裡一樣成立，請見 README。
"""
import argparse
import datetime
import os

import pandas as pd

from data_loader import load_price_data
from taifex_universe import STOCK_FUTURES_UNIVERSE
from momentum_breakout_engine import (
    run_momentum_breakout_backtest, BREAKOUT_SIGNAL_NAMES,
    precompute_all_breakout_indicators,
)
from mean_reversion_engine import summarize_mr, precompute_regime_series

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_breakout")

MIN_TRADES_FOR_RANKING = 30

SIGNAL_LABELS = {
    "score_volume_ratio": "量比",
    "score_rel_strength": "相對大盤強弱",
    "score_rsi_cross": "RSI穿越50(動能轉強)",
    "score_macd": "MACD柱狀圖",
    "score_chip_confirm": "外資連續買超確認",
}

HOLD_DAYS_OPTIONS = [("短線5天", 5), ("中期15天", 15)]


def split_is_oos(master_calendar, is_ratio=0.7):
    n = len(master_calendar)
    split_point = int(n * is_ratio)
    return master_calendar[:split_point], master_calendar[split_point:]


def run_signal_ablation(price_data, universe, indicators_by_code, regime_series, is_calendar,
                         starting_capital, hold_days, chip_streak_by_code=None, has_chip=False):
    """對 SIGNAL_LABELS 裡每個訊號單獨開啟(權重1.0，其餘0)，在IS跑一次回測。"""
    rows = []
    signal_names = [s for s in BREAKOUT_SIGNAL_NAMES if has_chip or s != "score_chip_confirm"]

    common_kwargs = dict(
        price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
        master_calendar=is_calendar, max_hold_days=hold_days, starting_capital=starting_capital,
        allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0,
        chip_streak_by_code=chip_streak_by_code,
    )

    for i, signal_name in enumerate(signal_names, start=1):
        trades = run_momentum_breakout_backtest(**common_kwargs, signal_weights={signal_name: 1.0})
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "signal": signal_name, "label": SIGNAL_LABELS[signal_name],
            "trade_count": stats["trade_count"], "win_rate": stats["win_rate"],
            "total_pnl_ntd": stats["total_pnl_ntd"], "max_drawdown_ntd": stats["max_drawdown_ntd"],
        })
        print(f"  [{i}/{len(signal_names)}] {SIGNAL_LABELS[signal_name]} "
              f"-> {stats['trade_count']}筆, 勝率={stats['win_rate']:.1f}%, "
              f"損益={stats['total_pnl_ntd']:,.0f}", flush=True)

    baseline_trades = run_momentum_breakout_backtest(
        **common_kwargs, signal_weights={name: 1.0 for name in signal_names},
    )
    baseline_stats = summarize_mr(baseline_trades, starting_capital)
    rows.append({
        "signal": "__baseline_equal_weight__", "label": f"基準({len(signal_names)}訊號等權重)",
        "trade_count": baseline_stats["trade_count"], "win_rate": baseline_stats["win_rate"],
        "total_pnl_ntd": baseline_stats["total_pnl_ntd"], "max_drawdown_ntd": baseline_stats["max_drawdown_ntd"],
    })
    print(f"  [基準] {len(signal_names)}訊號等權重 -> {baseline_stats['trade_count']}筆, "
          f"勝率={baseline_stats['win_rate']:.1f}%, 損益={baseline_stats['total_pnl_ntd']:,.0f}", flush=True)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="右側順勢突破策略 - 訊號拆解 + 持有天數比較")
    parser.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=1095)).isoformat())
    parser.add_argument("--end", default=datetime.date.today().isoformat())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--starting-capital", type=float, default=200_000)
    parser.add_argument("--max-stocks", type=int, default=0,
                         help="限制掃描股票數量(0=全部320檔，測試時可以設小一點加快速度)")
    parser.add_argument("--with-chip-confirm", action="store_true",
                         help="額外下載三大法人籌碼資料，把外資連續買超確認這個訊號也接上測試"
                              "(不加這個flag的話，score_chip_confirm會被跳過，不影響其他4個訊號)")
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
        raise RuntimeError(f"大盤代理指標 {INDEX_PROXY_CODE} 沒有成功下載，無法繼續。")
    index_df = price_data[INDEX_PROXY_CODE]
    master_calendar = index_df.index
    is_calendar, oos_calendar = split_is_oos(master_calendar)
    print(f"樣本內(IS)天數={len(is_calendar)}，樣本外(OOS)天數={len(oos_calendar)}\n")

    print("預先計算全市場突破指標 (量比/相對大盤強弱/RSI/MACD/20日均線/前高前低/ATR) ...")
    indicators_by_code = precompute_all_breakout_indicators(price_data, universe, index_code=INDEX_PROXY_CODE)
    regime_series = precompute_regime_series(index_df)
    print(f"指標預計算完成，涵蓋 {len(indicators_by_code)} 檔股票\n")

    chip_streak_by_code = None
    if args.with_chip_confirm:
        import chip_data_loader
        print("下載三大法人籌碼資料，計算外資連續買超/賣超天數 ...")
        chip_data = chip_data_loader.load_chip_data(args.start, args.end, universe_codes=set(universe.keys()),
                                                       refresh=args.refresh)
        chip_streak_by_code = {
            code: chip_data_loader.precompute_chip_streak(df) for code, df in chip_data.items()
        }
        print(f"籌碼資料涵蓋 {len(chip_streak_by_code)} 檔股票\n")
    else:
        print("未加 --with-chip-confirm，本次不測試外資連續買超確認訊號\n")

    all_ablation = {}
    all_holdday_results = {}

    for hold_label, hold_days in HOLD_DAYS_OPTIONS:
        print(f"=== 持有天數: {hold_label} ===")
        print("[階段1] 單一訊號拆解 (只在IS內) ...")
        ablation_df = run_signal_ablation(
            price_data, universe, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, chip_streak_by_code=chip_streak_by_code,
            has_chip=args.with_chip_confirm,
        )
        all_ablation[hold_label] = ablation_df
        ablation_csv = os.path.join(RESULTS_DIR, f"ablation_{hold_label}.csv")
        ablation_df.to_csv(ablation_csv, index=False, encoding="utf-8-sig")

        print(f"\n[階段2] 綜合比較：基準(等權重) vs IS/OOS ...")
        signal_names = [s for s in BREAKOUT_SIGNAL_NAMES if args.with_chip_confirm or s != "score_chip_confirm"]
        combo_results = {}
        for split_name, calendar in [("IS", is_calendar), ("OOS", oos_calendar)]:
            trades = run_momentum_breakout_backtest(
                price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
                master_calendar=calendar, max_hold_days=hold_days, starting_capital=args.starting_capital,
                allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0,
                signal_weights={name: 1.0 for name in signal_names},
                chip_streak_by_code=chip_streak_by_code,
            )
            stats = summarize_mr(trades, args.starting_capital)
            combo_results[split_name] = stats
            csv_path = os.path.join(RESULTS_DIR, f"trades_{split_name}_{hold_label}_baseline.csv")
            pd.DataFrame(trades).to_csv(csv_path, index=False, encoding="utf-8-sig")
        all_holdday_results[hold_label] = combo_results
        print(f"  IS  -> {combo_results['IS']['trade_count']}筆, 勝率={combo_results['IS']['win_rate']:.1f}%, "
              f"損益={combo_results['IS']['total_pnl_ntd']:,.0f}")
        print(f"  OOS -> {combo_results['OOS']['trade_count']}筆, 勝率={combo_results['OOS']['win_rate']:.1f}%, "
              f"損益={combo_results['OOS']['total_pnl_ntd']:,.0f}\n")

    summary_lines = [
        "=" * 100,
        f"右側順勢突破策略 訊號拆解 + 持有天數比較",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        f"籌碼確認訊號：{'有測試' if args.with_chip_confirm else '未測試(需加 --with-chip-confirm)'}",
        "=" * 100,
    ]
    for hold_label, _ in HOLD_DAYS_OPTIONS:
        summary_lines.append(f"\n--- {hold_label} / 單一訊號拆解(IS) ---")
        ablation_df = all_ablation[hold_label]
        reliable = ablation_df[ablation_df["trade_count"] >= MIN_TRADES_FOR_RANKING]
        if reliable.empty:
            summary_lines.append(f"⚠️ 沒有任何訊號的交易筆數 >= {MIN_TRADES_FOR_RANKING}，統計上都不夠可靠，全部原樣列出：")
            reliable = ablation_df
        reliable = reliable.sort_values("total_pnl_ntd", ascending=False)
        header = f"{'訊號':<24}{'交易數':>8}{'勝率%':>8}{'總損益NT$':>14}{'最大回撤NT$':>14}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        for _, r in reliable.iterrows():
            summary_lines.append(
                f"{r['label']:<24}{r['trade_count']:>8}{r['win_rate']:>8.1f}"
                f"{r['total_pnl_ntd']:>14,.0f}{r['max_drawdown_ntd']:>14,.0f}"
            )

        summary_lines.append(f"\n--- {hold_label} / 基準(等權重)組合 IS vs OOS ---")
        combo_results = all_holdday_results[hold_label]
        for split_name in ["IS", "OOS"]:
            stats = combo_results[split_name]
            split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
            summary_lines.append(
                f"{split_full}: {stats['trade_count']}筆, 勝率={stats['win_rate']:.1f}%, "
                f"總損益NT${stats['total_pnl_ntd']:,.0f}, 最大回撤NT${stats['max_drawdown_ntd']:,.0f}"
            )

    summary_lines.append(
        "\n判讀方式：先看每個持有天數下的單一訊號拆解，總損益明顯>0且交易筆數夠多的訊號，"
        "代表真的有預測力，值得之後加重權重；如果全部訊號單獨測都賺不到錢，代表這套右側突破邏輯"
        "本身可能也需要重新設計，不是調權重能解決的。基準組合的IS/OOS落差則是看有沒有過度配適："
        "OOS數字才是比較誠實的參考依據。"
    )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


if __name__ == "__main__":
    main()
