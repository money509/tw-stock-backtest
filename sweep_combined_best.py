"""
sweep_combined_best.py
=========================
把「單一訊號拆解測試」跟「ATR停利停損掃描」這兩條分開驗證過的改善方向疊在一起測，
確認合起來是不是真的比任何一個單獨的改善更好，而不是想當然爾地假設兩個好東西
加在一起一定更好（實務上常常因為互相干擾反而更差，一定要實際測過）。

背景：
- 單一訊號拆解（sweep_signal_ablation.py）顯示，真正有預測力的訊號集中在
  投信買超比重/量比/相對大盤強弱這3個，現在的8訊號等權重反而被外資買超比重、
  當沖比例這兩個反效果訊號拖累，比只用好訊號還差。
- ATR掃描（sweep_atr_params.py）顯示，停損拉緊到0.3~0.5倍、停利拉遠到3.0倍附近，
  比原本的0.8/1.2好很多；但停損緊到0.1倍已經是回測失真的範圍（現實下單會有
  滑價，這麼緊的停損不可能穩定成交在理論價位），不採用。

用法：
    python3 sweep_combined_best.py --start 2023-09-15 --end 2026-09-14

⚠️ 時間差修正：預設用「三大法人+美股濾網都錯開(真正能實測)」的版本，只用T-1日
已知的資料，跟cross_period_validation.py裡唯一能拿去實盤判讀的版本一致。加
--legacy-timing 可以切回舊版T日當天資料，僅供除錯/對照用。

⚠️ SIGNAL_WEIGHT_VARIANTS 目前仍是依照舊版(偷看T日當天資料)的訊號拆解結果挑的
候選（投信+量比+相對大盤強弱），還沒有依照修正後的誠實版拆解結果（量比+相對大盤
強弱+外資買超比重PF較高，投信只是打平）更新——重新掃這個之前，建議先確認要不要
把候選組合也換成新的訊號組合，不要只改時間差就直接沿用舊候選。
"""

import argparse
import datetime

import pandas as pd

import data_loader
import overnight_momentum_engine as ome
from compare_overnight import build_pipeline_inputs, split_is_oos, IS_RATIO

MIN_TRADES_FOR_RANKING = 30

# 候選訊號權重組合：baseline是現在的8訊號等權重，其餘是依訊號拆解結果挑出的候選
SIGNAL_WEIGHT_VARIANTS = {
    "baseline_8訊號等權重": {
        "score_close_position": 1.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0,
        "score_gain_pct": 1.0, "score_price_level": 1.0, "score_day_trading": 1.0,
        "score_foreign": 1.0, "score_trust": 1.0,
    },
    "只用最強3個(投信+量比+相對大盤強弱)": {
        "score_trust": 1.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0,
    },
    "最強3個+漲幅%": {
        "score_trust": 1.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0,
        "score_gain_pct": 1.0,
    },
    "最強3個，投信加重2倍": {
        "score_trust": 2.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0,
    },
}

# 候選ATR組合：保留舊預設值當對照，另外兩組是掃描裡表現好、且風控上還算合理
# （不是掃到失真邊界0.1倍那種）的組合
ATR_VARIANTS = {
    "舊預設(0.8/1.2)": {"atr_stop_mult": 0.8, "atr_target_mult": 1.2},
    "停損0.5倍/停利3.0倍": {"atr_stop_mult": 0.5, "atr_target_mult": 3.0},
    "停損0.3倍/停利3.0倍": {"atr_stop_mult": 0.3, "atr_target_mult": 3.0},
}


def run_combined_sweep(pipeline_inputs, is_ratio=IS_RATIO, top_n=5,
                        signal_variants=None, atr_variants=None,
                        use_prior_day_chip_data=True, use_prior_day_us_market_data=True):
    """訊號權重 x ATR參數 交叉測試，只在IS區間跑。

    use_prior_day_chip_data / use_prior_day_us_market_data：預設都是True，只用
    T-1日已知的三大法人/美股資料(真正能實測的版本)。設False會改用T日當天資料
    (舊版，不可能實測)，僅供除錯/對照用。
    """
    signal_variants = signal_variants or SIGNAL_WEIGHT_VARIANTS
    atr_variants = atr_variants or ATR_VARIANTS
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
    combo_num = 0
    total = len(signal_variants) * len(atr_variants)
    for signal_label, signal_weights in signal_variants.items():
        for atr_label, atr_params in atr_variants.items():
            combo_num += 1
            trades = ome.run_overnight_backtest(
                **common_args, signal_weights=signal_weights, **atr_params,
            )
            summary = ome.summarize_overnight(trades)
            rows.append({
                "signal_variant": signal_label,
                "atr_variant": atr_label,
                "atr_stop_mult": atr_params["atr_stop_mult"],
                "atr_target_mult": atr_params["atr_target_mult"],
                "total_trades": summary["total_trades"],
                "win_rate": summary["win_rate"],
                "profit_factor": summary["profit_factor"],
                "total_pnl": summary["total_pnl"],
                "avg_pnl": summary["avg_pnl"],
            })
            print(f"[{combo_num}/{total}] {signal_label} x {atr_label} "
                  f"-> {summary['total_trades']}筆, PF={summary['profit_factor']:.2f}, "
                  f"勝率={summary['win_rate']:.1f}%, 損益={summary['total_pnl']:,.0f}", flush=True)

    return pd.DataFrame(rows), is_days, oos_days


def rank_results(result_df, min_trades=MIN_TRADES_FOR_RANKING):
    reliable = result_df[result_df["total_trades"] >= min_trades].copy()
    if reliable.empty:
        print(f"⚠️ 沒有任何組合的交易筆數 >= {min_trades}，統計上都不夠可靠，全部組合原樣列出：")
        reliable = result_df.copy()
    return reliable.sort_values("profit_factor", ascending=False).reset_index(drop=True)


def validate_best_on_oos(pipeline_inputs, best_row, oos_days, top_n=5,
                          signal_weight_variants=None,
                          use_prior_day_chip_data=True, use_prior_day_us_market_data=True):
    """把排名第一的組合，拿到 OOS 跑一次，僅供參考、不能拿來重新挑參數。"""
    signal_weight_variants = signal_weight_variants or SIGNAL_WEIGHT_VARIANTS
    signal_label = best_row["signal_variant"]
    signal_weights = signal_weight_variants[signal_label]
    trades = ome.run_overnight_backtest(
        indicators_by_code=pipeline_inputs["indicators_by_code"],
        market_returns_df=pipeline_inputs["market_returns_df"],
        foreign_ratio_df=pipeline_inputs["foreign_ratio_df"],
        trust_ratio_df=pipeline_inputs["trust_ratio_df"],
        day_trading_ratio_df=pipeline_inputs["day_trading_ratio_df"],
        universe_codes=pipeline_inputs["universe_codes"],
        us_market_returns_df=pipeline_inputs.get("us_market_returns_df"),
        ex_dividend_dates_by_code=pipeline_inputs.get("ex_dividend_dates_by_code"),
        trading_days=oos_days,
        top_n=top_n,
        signal_weights=signal_weights,
        # 保險：避免 DataFrame.iloc[0].to_dict() 把數值型欄位轉成非預期的 dtype
        # （sweep_overnight_params.py 曾經因為這個炸過一次）。
        atr_stop_mult=float(best_row["atr_stop_mult"]),
        atr_target_mult=float(best_row["atr_target_mult"]),
        use_prior_day_chip_data=use_prior_day_chip_data,
        use_prior_day_us_market_data=use_prior_day_us_market_data,
    )
    return ome.summarize_overnight(trades)


def main():
    parser = argparse.ArgumentParser(description="訊號權重 x ATR參數 合併測試（僅在IS內選參數）")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--refresh", action="store_true", help="忽略快取，強制重新下載所有資料")
    parser.add_argument("--legacy-timing", action="store_true",
                         help="改用T日當天資料(舊版，不可能實測)，僅供除錯/對照用，"
                              "不建議拿這個版本的結果去挑參數")
    args = parser.parse_args()

    start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()

    use_realistic_timing = not args.legacy_timing

    print(f"載入資料 ({args.start} ~ {args.end})，若快取已存在會直接使用，不重新下載 ...")
    pipeline_inputs = build_pipeline_inputs(
        data_loader.STOCK_FUTURES_WHITELIST, start_date, end_date, refresh=args.refresh,
    )

    if use_realistic_timing:
        print("時間差修正：三大法人+美股濾網都錯開(真正能實測) —— 只用T-1日已知的資料。")
    else:
        print("⚠️ 時間差修正：已停用(--legacy-timing)，用的是T日當天資料(舊版，不可能實測)，"
              "結果僅供對照，不代表能實盤。")

    print(f"\n共 {len(SIGNAL_WEIGHT_VARIANTS)} 組訊號權重 x {len(ATR_VARIANTS)} 組ATR參數，"
          f"只在樣本內(IS, 前70%)跑回測 ...\n")
    result_df, is_days, oos_days = run_combined_sweep(
        pipeline_inputs, top_n=args.top_n,
        use_prior_day_chip_data=use_realistic_timing,
        use_prior_day_us_market_data=use_realistic_timing,
    )

    ranked = rank_results(result_df)
    print(f"\n=== 依 profit_factor 排名 ===")
    print(ranked[["signal_variant", "atr_variant", "total_trades", "win_rate",
                   "profit_factor", "total_pnl"]].to_string(index=False))

    if not ranked.empty:
        best = ranked.iloc[0].to_dict()
        print(f"\n=== 排名第一的組合在 OOS 驗證（僅供參考，不能用來重新選參數） ===")
        print(f"組合: {best['signal_variant']} x {best['atr_variant']}")
        oos_summary = validate_best_on_oos(
            pipeline_inputs, best, oos_days, top_n=args.top_n,
            use_prior_day_chip_data=use_realistic_timing,
            use_prior_day_us_market_data=use_realistic_timing,
        )
        print(f"OOS 交易次數: {oos_summary['total_trades']}")
        print(f"OOS 勝率: {oos_summary['win_rate']:.1f}%")
        print(f"OOS 盈虧比: {oos_summary['profit_factor']:.2f}")
        print(f"OOS 總損益: {oos_summary['total_pnl']:,.0f}")
        print(f"\n⚠️ 這只是同一份資料切出來的 OOS，不是獨立的第三段歷史期間，"
              f"跨期驗證仍要另外找一段完全獨立的歷史資料才算數。")


if __name__ == "__main__":
    main()
