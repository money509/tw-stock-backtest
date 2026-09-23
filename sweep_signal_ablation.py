"""
sweep_signal_ablation.py
===========================
單一訊號拆解測試：8個子訊號（技術面5個+籌碼面3個）每次只開一個，其餘全部關掉，
分別在樣本內(IS)跑一次回測，比較誰單獨拿來選股時表現最好。

目的不是要找「最終要用的參數」，是要回答一個更基本的問題：
    現在這8個訊號裡，到底哪幾個真的對「明天股價會不會延續」有預測力？
    還是全部都只是雜訊，掃了老半天權重組合也不會有結果？

用法：
    python3 sweep_signal_ablation.py --start 2023-09-15 --end 2026-09-14

會用跟 sweep_overnight_params.py 一樣的方式：資料只在一開始載入一次，之後每個
訊號的測試都是對同一份資料重跑 run_overnight_backtest()，不重新下載。

⚠️ 時間差修正：預設每個訊號都用「三大法人+美股濾網都錯開(真正能實測)」的版本
（use_prior_day_chip_data=True, use_prior_day_us_market_data=True），也就是
只用T-1日已知的資料，跟cross_period_validation.py裡唯一能拿去實盤判讀的版本
一致。這是為了避免重蹈覆轍：如果拆解單一訊號時還是用「完全T日當天(舊版，不可能
實測)」在挑訊號，挑出來的「有效訊號」本身就可能只是偷看了收盤後才公布的三大法人
資料或晚上才開盤的美股資料，跨期驗證時一樣會現形、白做工。
如果想比較兩種時間假設差多少（僅供除錯/對照，不建議拿legacy版本的結果去挑訊號），
可以加 --legacy-timing 改回舊版T日當天資料。
"""

import argparse
import datetime

import pandas as pd

import data_loader
import overnight_momentum_engine as ome
from compare_overnight import build_pipeline_inputs, split_is_oos, IS_RATIO

MIN_TRADES_FOR_RANKING = 30

SIGNAL_LABELS = {
    "score_close_position": "技術-收盤位置",
    "score_volume_ratio": "技術-量比",
    "score_rel_strength": "技術-相對大盤強弱",
    "score_gain_pct": "技術-漲幅%",
    "score_price_level": "技術-股價位階",
    "score_day_trading": "籌碼-當沖比例",
    "score_foreign": "籌碼-外資買超比重",
    "score_trust": "籌碼-投信買超比重",
}


def run_ablation(pipeline_inputs, is_ratio=IS_RATIO, top_n=5,
                  use_prior_day_chip_data=True, use_prior_day_us_market_data=True):
    """對 SIGNAL_LABELS 裡每一個訊號，單獨開啟後在IS跑一次回測，回傳結果 DataFrame。

    use_prior_day_chip_data / use_prior_day_us_market_data：預設都是True，也就是
    只用T-1日已知的三大法人/美股資料（跟cross_period_validation.py裡唯一能實測的
    版本一致）。設False的話會改用T日當天資料(舊版，不可能實測)，僅供除錯/對照用，
    不建議拿這個版本的結果去挑訊號——挑出來的「有效訊號」可能只是偷看了未來資料。
    """
    is_days, oos_days = split_is_oos(pipeline_inputs["trading_days"], is_ratio)

    common_args = dict(
        indicators_by_code=pipeline_inputs["indicators_by_code"],
        market_returns_df=pipeline_inputs["market_returns_df"],
        foreign_ratio_df=pipeline_inputs["foreign_ratio_df"],
        trust_ratio_df=pipeline_inputs["trust_ratio_df"],
        day_trading_ratio_df=pipeline_inputs["day_trading_ratio_df"],
        universe_codes=pipeline_inputs["universe_codes"],
        us_market_returns_df=pipeline_inputs.get("us_market_returns_df"),
        ex_dividend_dates_by_code=pipeline_inputs.get("ex_dividend_dates_by_code"),
        trading_days=is_days,
        top_n=top_n,
        use_prior_day_chip_data=use_prior_day_chip_data,
        use_prior_day_us_market_data=use_prior_day_us_market_data,
    )

    rows = []
    signal_names = list(SIGNAL_LABELS.keys())
    for i, signal_name in enumerate(signal_names, start=1):
        trades = ome.run_overnight_backtest(
            **common_args, signal_weights={signal_name: 1.0},
        )
        summary = ome.summarize_overnight(trades)
        rows.append({
            "signal": signal_name,
            "label": SIGNAL_LABELS[signal_name],
            "total_trades": summary["total_trades"],
            "win_rate": summary["win_rate"],
            "profit_factor": summary["profit_factor"],
            "total_pnl": summary["total_pnl"],
            "avg_pnl": summary["avg_pnl"],
        })
        print(f"[{i}/{len(signal_names)}] {SIGNAL_LABELS[signal_name]} "
              f"-> {summary['total_trades']}筆, PF={summary['profit_factor']:.2f}, "
              f"勝率={summary['win_rate']:.1f}%, 損益={summary['total_pnl']:,.0f}", flush=True)

    # 額外加一個「全部訊號等權重」的基準組，方便對照單一訊號有沒有比全部混在一起好
    baseline_trades = ome.run_overnight_backtest(
        **common_args, signal_weights={name: 1.0 for name in signal_names},
    )
    baseline_summary = ome.summarize_overnight(baseline_trades)
    rows.append({
        "signal": "__baseline_equal_weight__",
        "label": "基準(8訊號等權重)",
        "total_trades": baseline_summary["total_trades"],
        "win_rate": baseline_summary["win_rate"],
        "profit_factor": baseline_summary["profit_factor"],
        "total_pnl": baseline_summary["total_pnl"],
        "avg_pnl": baseline_summary["avg_pnl"],
    })
    print(f"[基準] 8訊號等權重 -> {baseline_summary['total_trades']}筆, "
          f"PF={baseline_summary['profit_factor']:.2f}, "
          f"勝率={baseline_summary['win_rate']:.1f}%, "
          f"損益={baseline_summary['total_pnl']:,.0f}", flush=True)

    return pd.DataFrame(rows)


def rank_results(result_df, min_trades=MIN_TRADES_FOR_RANKING):
    reliable = result_df[result_df["total_trades"] >= min_trades].copy()
    if reliable.empty:
        print(f"⚠️ 沒有任何訊號的交易筆數 >= {min_trades}，統計上都不夠可靠，全部原樣列出：")
        reliable = result_df.copy()
    return reliable.sort_values("profit_factor", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="隔日衝策略單一訊號拆解測試（只在IS內）")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--refresh", action="store_true", help="忽略快取，強制重新下載所有資料")
    parser.add_argument("--legacy-timing", action="store_true",
                         help="改用T日當天資料(舊版，不可能實測)，僅供除錯/對照用，"
                              "不建議拿這個版本的結果去挑訊號")
    args = parser.parse_args()

    start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()

    use_realistic_timing = not args.legacy_timing

    print(f"載入資料 ({args.start} ~ {args.end})，若快取已存在會直接使用，不重新下載 ...")
    pipeline_inputs = build_pipeline_inputs(
        data_loader.STOCK_FUTURES_WHITELIST, start_date, end_date, refresh=args.refresh,
    )

    if use_realistic_timing:
        print("\n時間差修正：三大法人+美股濾網都錯開(真正能實測) —— 只用T-1日已知的資料。")
    else:
        print("\n⚠️ 時間差修正：已停用(--legacy-timing)，用的是T日當天資料(舊版，不可能實測)，"
              "結果僅供對照，不代表能實盤。")

    print(f"\n開始單一訊號拆解測試（只在IS, 前70%）...\n")
    result_df = run_ablation(
        pipeline_inputs, top_n=args.top_n,
        use_prior_day_chip_data=use_realistic_timing,
        use_prior_day_us_market_data=use_realistic_timing,
    )

    ranked = rank_results(result_df)
    print(f"\n=== 依 profit_factor 排名 ===")
    print(ranked[["label", "total_trades", "win_rate", "profit_factor", "total_pnl"]]
          .to_string(index=False))

    print(f"\n判讀方式：如果某個訊號單獨測PF明顯 > 1 且交易筆數夠多，代表這個訊號" \
          f"本身有真正的預測力，值得在之後的權重掃描裡加重；" \
          f"如果全部訊號單獨測都 <= 1，代表現在這套評分邏輯可能都在製造雜訊，"
          f"要回頭重新設計訊號本身，而不是繼續調權重。")


if __name__ == "__main__":
    main()
