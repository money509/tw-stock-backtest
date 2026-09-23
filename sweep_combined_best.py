"""
sweep_combined_best.py
=========================
把「單一訊號拆解測試」跟「ATR停利停損掃描」這兩條分開驗證過的改善方向疊在一起測，
確認合起來是不是真的比任何一個單獨的改善更好，而不是想當然爾地假設兩個好東西
加在一起一定更好（實務上常常因為互相干擾反而更差，一定要實際測過）。

背景（以下均為「三大法人+美股濾網都錯開(真正能實測)」+ slippage_pct=0.1 的誠實版結果）：
- 單一訊號拆解（sweep_signal_ablation.py，honest timing版）顯示，真正有預測力
  （PF明顯>1）的訊號是量比(PF1.12)、相對大盤強弱(PF1.11)、外資買超比重(PF1.07)
  這3個；投信買超比重(PF1.01)、漲幅%(PF1.00)只是打平；收盤位置、股價位階、
  當沖比例是負貢獻，8訊號等權重的基準組(PF0.93)反而比多數單一訊號還差，
  代表現在的作法是把好訊號跟壞訊號混在一起、互相稀釋。
- ATR掃描（sweep_atr_params.py，honest timing + slippage_pct=0.1版）顯示，
  一旦套上合理滑價，停損0.1倍以下的組合雖然PF還撐得住(>1)，但停損距離小到
  可能低於實際下單的最小跳動/價差，執行上不可信；停損0.1~1.5倍的「正常」
  範圍，全部PF<1(0.71~0.95)，找不到任何站得住腳的ATR組合能單靠出場距離
  把PF拉過1——問題核心在訊號本身太弱，不是出場設定。
- 第一次合併測試發現「最強2個(量比+相對大盤強弱，拿掉外資)」是唯一一組在
  多種ATR設定下都PF>=1.00的組合(IS最佳PF=1.12)，且因為這2個都是股票自己
  當天的價量訊號(不像籌碼資料要等收盤後才公布)，完全不受時間差修正影響。
  但拿去cross_period_validation.py在完全獨立的2020-2023區間驗證，PF掉到
  0.88~0.90，沒有通過——不過同一批候選裡，它仍然是所有候選中PF最高、
  跌幅最小的一組(基準8訊號組在同一段資料上只有0.66)。

這次調整：既然量比/相對大盤強弱這兩個純技術訊號本身不受時間差問題影響、
表現也相對最穩，SIGNAL_WEIGHT_VARIANTS這次改成專注在「技術面內部」找更好的
搭配——測試全部5個技術訊號等權重、只調量比/相對大盤強弱的權重比例、
加入漲幅%(打平訊號)當多樣化對照，看純技術面裡有沒有比目前的「最強2個」更好
的組合。ATR_VARIANTS維持前一版避開執行失真邊界(<=0.1倍)的合理區間。

用法：
    python3 sweep_combined_best.py --start 2023-09-15 --end 2026-09-14 --slippage-pct 0.1

⚠️ 時間差修正：預設用「三大法人+美股濾網都錯開(真正能實測)」的版本，只用T-1日
已知的資料，跟cross_period_validation.py裡唯一能拿去實盤判讀的版本一致。加
--legacy-timing 可以切回舊版T日當天資料，僅供除錯/對照用。

⚠️ 滑價：預設0.0(不模擬)，維持舊行為向後相容。但atr_sweep已經證實滑價對排名
影響很大(套了0.1%滑價後，原本PF>1的組合大量翻成PF<1)，這次合併測試強烈建議
加 --slippage-pct 0.1，不然排名結果可能只是沒套滑價時的假象，跟前面兩支腳本
的判讀基準不一致。
"""

import argparse
import datetime

import pandas as pd

import data_loader
import overnight_momentum_engine as ome
from compare_overnight import build_pipeline_inputs, split_is_oos, IS_RATIO

MIN_TRADES_FOR_RANKING = 30

# 候選訊號權重組合：baseline是現在的8訊號等權重(對照組)，其餘全部是純技術面
# 組合(不含任何籌碼訊號)——「最強2個」是上一版combined測試驗證過的最佳解，
# 保留當這一輪的對照基準；其餘4組是圍繞它做的技術面內部調整，看能不能找到
# 更好的搭配。
SIGNAL_WEIGHT_VARIANTS = {
    "baseline_8訊號等權重": {
        "score_close_position": 1.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0,
        "score_gain_pct": 1.0, "score_price_level": 1.0, "score_day_trading": 1.0,
        "score_foreign": 1.0, "score_trust": 1.0,
    },
    "技術5訊號等權重(不含籌碼)": {
        "score_close_position": 1.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0,
        "score_gain_pct": 1.0, "score_price_level": 1.0,
    },
    "最強2個(量比+相對大盤強弱)": {
        "score_volume_ratio": 1.0, "score_rel_strength": 1.0,
    },
    "最強2個，量比加重2倍": {
        "score_volume_ratio": 2.0, "score_rel_strength": 1.0,
    },
    "最強2個，相對大盤強弱加重2倍": {
        "score_volume_ratio": 1.0, "score_rel_strength": 2.0,
    },
    "最強2個+漲幅%(打平訊號，當多樣化對照)": {
        "score_volume_ratio": 1.0, "score_rel_strength": 1.0, "score_gain_pct": 1.0,
    },
}

# 候選ATR組合：保留舊預設值當對照，其餘避開執行上不可信的失真邊界
# （atr_sweep套滑價後顯示，停損<=0.1倍雖然PF還撐得住，但停損距離可能小於
# 實際下單的最小跳動/價差，執行上不可信；0.1~1.5倍這個「正常」範圍套滑價後
# 全部PF<1，這裡選相對表現較好、風控上也合理的0.3~0.8倍停損區間）。
ATR_VARIANTS = {
    "舊預設(0.8/1.2)": {"atr_stop_mult": 0.8, "atr_target_mult": 1.2},
    "停損0.3倍/停利2.0倍": {"atr_stop_mult": 0.3, "atr_target_mult": 2.0},
    "停損0.3倍/停利3.0倍": {"atr_stop_mult": 0.3, "atr_target_mult": 3.0},
    "停損0.5倍/停利2.0倍": {"atr_stop_mult": 0.5, "atr_target_mult": 2.0},
    "停損0.5倍/停利3.0倍": {"atr_stop_mult": 0.5, "atr_target_mult": 3.0},
    "停損0.8倍/停利2.0倍": {"atr_stop_mult": 0.8, "atr_target_mult": 2.0},
}


def run_combined_sweep(pipeline_inputs, is_ratio=IS_RATIO, top_n=5,
                        signal_variants=None, atr_variants=None,
                        use_prior_day_chip_data=True, use_prior_day_us_market_data=True,
                        slippage_pct=0.0):
    """訊號權重 x ATR參數 交叉測試，只在IS區間跑。

    use_prior_day_chip_data / use_prior_day_us_market_data：預設都是True，只用
    T-1日已知的三大法人/美股資料(真正能實測的版本)。設False會改用T日當天資料
    (舊版，不可能實測)，僅供除錯/對照用。

    slippage_pct：預設0.0(不模擬)，維持舊行為向後相容。⚠️atr_sweep已經證實
    滑價對排名影響很大，強烈建議呼叫端傳入0.1，跟前面兩支腳本的判讀基準一致，
    避免排名結果只是沒套滑價時的假象。
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
        slippage_pct=slippage_pct,
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
                          use_prior_day_chip_data=True, use_prior_day_us_market_data=True,
                          slippage_pct=0.0):
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
        slippage_pct=slippage_pct,
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
    parser.add_argument("--slippage-pct", type=float, default=0.0,
                         help="模擬滑價百分比(預設0=不模擬)，強烈建議填0.1，"
                              "跟sweep_atr_params.py的判讀基準一致，避免排名結果"
                              "只是沒套滑價時的假象")
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

    if args.slippage_pct:
        print(f"⚠️ 有套用滑價模擬：slippage_pct={args.slippage_pct}%（買進多付、賣出少拿），"
              f"排名結果反映的是滑價侵蝕後的真實表現，不是理論最佳值。")
    else:
        print("⚠️ 未套用滑價模擬(slippage_pct=0)，排名結果可能偏樂觀，"
              "建議加 --slippage-pct 0.1 才跟其他腳本的判讀基準一致。")

    print(f"\n共 {len(SIGNAL_WEIGHT_VARIANTS)} 組訊號權重 x {len(ATR_VARIANTS)} 組ATR參數，"
          f"只在樣本內(IS, 前70%)跑回測 ...\n")
    result_df, is_days, oos_days = run_combined_sweep(
        pipeline_inputs, top_n=args.top_n,
        use_prior_day_chip_data=use_realistic_timing,
        use_prior_day_us_market_data=use_realistic_timing,
        slippage_pct=args.slippage_pct,
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
            slippage_pct=args.slippage_pct,
        )
        print(f"OOS 交易次數: {oos_summary['total_trades']}")
        print(f"OOS 勝率: {oos_summary['win_rate']:.1f}%")
        print(f"OOS 盈虧比: {oos_summary['profit_factor']:.2f}")
        print(f"OOS 總損益: {oos_summary['total_pnl']:,.0f}")
        print(f"\n⚠️ 這只是同一份資料切出來的 OOS，不是獨立的第三段歷史期間，"
              f"跨期驗證仍要另外找一段完全獨立的歷史資料才算數。")


if __name__ == "__main__":
    main()
