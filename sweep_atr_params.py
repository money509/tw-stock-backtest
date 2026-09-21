"""
sweep_atr_params.py
======================
專門掃描 ATR 停利/停損倍數（atr_stop_mult / atr_target_mult）。

背景：sweep_overnight_params.py 的第一版掃描完全沒有動到這兩個參數，
一直固定用引擎預設值 atr_stop_mult=0.8 / atr_target_mult=1.2（風報比只有1.5）。
但實際回測結果顯示 61% 的交易都是被「強制平倉」在接近平盤的地方結束，
真正碰到停損/停利出場的交易反而互相抵銷，代表現在這組停損停利的「距離」
設定可能完全不合理——太緊會被雜訊掃到、太鬆又等於沒設。這支腳本就是要把
這兩個維度真正掃過一次，看看拉開或縮小這個距離會不會改變出場結構的分佈。

設計原則：跟 sweep_signal_ablation.py 一樣，只掃這兩個維度，其餘（top_n、
技術/籌碼權重、跳空停損門檻）固定在目前的預設值，這樣掃出來的結果才看得懂
是「ATR倍數」本身在起作用，而不是跟其他維度交叉污染，掃出一堆看不懂的組合。

用法：
    python3 sweep_atr_params.py --start 2023-09-15 --end 2026-09-14

資料一樣只載入一次（吃現有快取），之後每個 ATR 組合都是對同一份資料重跑
run_overnight_backtest()，只在記憶體內計算，不會再打網路請求。
"""

import argparse
import datetime

import pandas as pd

import data_loader
import overnight_momentum_engine as ome
from compare_overnight import build_pipeline_inputs, split_is_oos, IS_RATIO

MIN_TRADES_FOR_RANKING = 30

# 固定住其他維度，只讓 ATR 倍數變動
FIXED_TOP_N = 5
FIXED_TECH_WEIGHT = 0.35
FIXED_CHIP_WEIGHT = 0.65
FIXED_GAP_STOP_THRESHOLD = -0.015


def build_atr_grid():
    """
    停損倍數（atr_stop_mult）：距離進場價多少個ATR就停損，數字越小代表停損越緊。
    停利倍數（atr_target_mult）：距離進場價多少個ATR就停利，數字越小代表越容易觸發。
    只掃「停利倍數 >= 停損倍數」的組合（風報比至少1:1），風報比小於1的組合
    邏輯上不太合理（賺得比虧得少），先不浪費掃描次數。

    第一版(0.5~1.5停損 × 1.0~3.0停利)掃出來，排名前幾名清一色都是停利=3.0
    (這次掃描範圍裡最遠的一個)，代表PF在停利拉遠的方向上還沒到頂，所以第二版
    把停利延伸到6.0、停損下探到0.3——結果停利在3.0左右真的出現頂點了(拉遠到
    4.0/5.0/6.0後PF微幅下降)，但換成停損又卡在邊界(0.3倍ATR，這次掃描範圍裡
    最緊的一個，還是表現最好)。這一版繼續把停損往更緊的方向延伸到0.1，確認
    「停損越緊越好」這個方向會不會在更極端的地方反轉——如果一直緊到0.1都還在
    進步，那就要小心是不是已經緊到滑價和雜訊主導的範圍，不是真正可靠的訊號。
    """
    combos = []
    for atr_stop_mult in (0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5):
        for atr_target_mult in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
            if atr_target_mult < atr_stop_mult:
                continue
            combos.append({
                "atr_stop_mult": atr_stop_mult,
                "atr_target_mult": atr_target_mult,
            })
    return combos


def run_sweep(pipeline_inputs, is_ratio=IS_RATIO, param_grid=None,
              top_n=FIXED_TOP_N, tech_weight=FIXED_TECH_WEIGHT,
              chip_weight=FIXED_CHIP_WEIGHT,
              gap_stop_threshold=FIXED_GAP_STOP_THRESHOLD):
    """只在 IS 區間跑，回傳每個 ATR 組合的結果 DataFrame（不看 OOS，理由同 sweep_overnight_params.py）。"""
    param_grid = param_grid or build_atr_grid()
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
        tech_weight=tech_weight,
        chip_weight=chip_weight,
        gap_stop_threshold=gap_stop_threshold,
    )

    rows = []
    for i, params in enumerate(param_grid, start=1):
        trades = ome.run_overnight_backtest(**common_args, **params)
        summary = ome.summarize_overnight(trades)
        exit_breakdown = summarize_exit_reasons(trades)
        rows.append({**params, **{
            "total_trades": summary["total_trades"],
            "win_rate": summary["win_rate"],
            "profit_factor": summary["profit_factor"],
            "total_pnl": summary["total_pnl"],
            "avg_pnl": summary["avg_pnl"],
            **exit_breakdown,
        }})
        print(f"[{i}/{len(param_grid)}] atr_stop={params['atr_stop_mult']:.1f} "
              f"atr_target={params['atr_target_mult']:.1f} "
              f"-> {summary['total_trades']}筆, PF={summary['profit_factor']:.2f}, "
              f"損益={summary['total_pnl']:,.0f}", flush=True)

    result_df = pd.DataFrame(rows)
    return result_df, is_days, oos_days


def summarize_exit_reasons(trades):
    """算出各出場原因(target/forced_close/stop/gap_stop)佔全部交易的比例，
    方便觀察拉開/縮小 ATR 倍數會不會改變「61%卡在強制平倉」這個結構。
    trades 是 run_overnight_backtest() 回傳的 list[dict]（不是 DataFrame）。"""
    empty_result = {
        "pct_target": 0.0, "pct_forced_close": 0.0,
        "pct_stop": 0.0, "pct_gap_stop": 0.0,
    }
    if not trades:
        return empty_result
    trades_df = pd.DataFrame(trades)
    if "exit_reason" not in trades_df.columns:
        return empty_result
    counts = trades_df["exit_reason"].value_counts(normalize=True) * 100.0
    return {
        "pct_target": float(counts.get("target", 0.0)),
        "pct_forced_close": float(counts.get("forced_close", 0.0)),
        "pct_stop": float(counts.get("stop", 0.0)),
        "pct_gap_stop": float(counts.get("gap_stop", 0.0)),
    }


def rank_results(result_df, min_trades=MIN_TRADES_FOR_RANKING):
    reliable = result_df[result_df["total_trades"] >= min_trades].copy()
    if reliable.empty:
        print(f"⚠️ 沒有任何組合的交易筆數 >= {min_trades}，統計上都不夠可靠，全部組合原樣列出：")
        reliable = result_df.copy()
    return reliable.sort_values("profit_factor", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="隔日衝策略 ATR 停利停損倍數掃描（僅在IS內選參數）")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=10, help="列出前幾名組合")
    parser.add_argument("--refresh", action="store_true", help="忽略快取，強制重新下載所有資料")
    args = parser.parse_args()

    start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()

    print(f"載入資料 ({args.start} ~ {args.end})，若快取已存在會直接使用，不重新下載 ...")
    pipeline_inputs = build_pipeline_inputs(
        data_loader.STOCK_FUTURES_WHITELIST, start_date, end_date, refresh=args.refresh,
    )

    grid = build_atr_grid()
    print(f"\n共 {len(grid)} 組 ATR 倍數組合，其餘參數固定在 top_n={FIXED_TOP_N}, "
          f"tech/chip={FIXED_TECH_WEIGHT}/{FIXED_CHIP_WEIGHT}, "
          f"gap_stop={FIXED_GAP_STOP_THRESHOLD}，只在樣本內(IS, 前70%)跑回測 ...\n")
    result_df, is_days, oos_days = run_sweep(pipeline_inputs, param_grid=grid)

    ranked = rank_results(result_df)
    print(f"\n=== IS 排名前 {args.top} 組合（依 profit_factor） ===")
    cols = ["atr_stop_mult", "atr_target_mult", "total_trades", "win_rate",
            "profit_factor", "total_pnl", "pct_target", "pct_forced_close",
            "pct_stop", "pct_gap_stop"]
    print(ranked[cols].head(args.top).to_string(index=False))

    print(f"\n判讀方式："
          f"如果拉大 atr_target_mult（停利拉遠）能明顯降低 pct_forced_close 的比例，"
          f"代表現在的停利設太近，很多本來會續漲的交易提早被停利或卡到隔天強制平倉；"
          f"如果不管怎麼調 pct_forced_close 都還是佔大宗，代表問題不是出場距離，"
          f"是選出來的股票本身隔天大部分就是橫盤，那就要回頭看選股訊號本身"
          f"（呼應同時在測的單一訊號拆解結果）。")

    if not ranked.empty:
        best = ranked.iloc[0].to_dict()
        print(f"\n=== 排名第一的組合在 OOS 驗證（僅供參考，不能用來重新選參數） ===")
        oos_summary = validate_best_on_oos(pipeline_inputs, best, oos_days)
        print(f"組合: atr_stop={best['atr_stop_mult']}, atr_target={best['atr_target_mult']}")
        print(f"OOS 交易次數: {oos_summary['total_trades']}")
        print(f"OOS 勝率: {oos_summary['win_rate']:.1f}%")
        print(f"OOS 盈虧比: {oos_summary['profit_factor']:.2f}")
        print(f"OOS 總損益: {oos_summary['total_pnl']:,.0f}")
        print(f"\n⚠️ 這只是同一份資料切出來的 OOS，不是獨立的第三段歷史期間，"
              f"跨期驗證仍要另外找一段完全獨立的歷史資料才算數。")


def validate_best_on_oos(pipeline_inputs, best_params, oos_days,
                          top_n=FIXED_TOP_N, tech_weight=FIXED_TECH_WEIGHT,
                          chip_weight=FIXED_CHIP_WEIGHT,
                          gap_stop_threshold=FIXED_GAP_STOP_THRESHOLD):
    """把排名第一的 ATR 組合，拿到 OOS 跑一次，僅供參考、不能拿來重新挑參數。"""
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
        tech_weight=tech_weight,
        chip_weight=chip_weight,
        gap_stop_threshold=gap_stop_threshold,
        # 跟 sweep_overnight_params.py 一樣的保險：避免 DataFrame.iloc[0].to_dict()
        # 把數值型欄位轉成非預期的 dtype。
        atr_stop_mult=float(best_params["atr_stop_mult"]),
        atr_target_mult=float(best_params["atr_target_mult"]),
    )
    return ome.summarize_overnight(trades)


if __name__ == "__main__":
    main()
