"""
sweep_overnight_params.py
============================
在「樣本內(IS)」資料上掃描隔日衝策略的參數組合，找出表現較穩的組合，
再回頭在「樣本外(OOS)」驗證一次（僅供參考，不能拿 OOS 結果來挑參數，
否則 OOS 就失去它「策略沒看過」的意義了——這是 IS/OOS 切分存在的唯一理由）。

重要設計：資料只在一開始載入一次（會吃到 compare_overnight 第一次跑留下的
data_cache/chip_cache/day_trading_cache，不需要重新下載），之後每個參數組合
都是對同一份已經算好指標的資料重跑 run_overnight_backtest()，純粹是記憶體內
的計算，不會再打任何網路請求——真正花時間的是第一次下載，那個成本已經付過了。

用法：
    python3 sweep_overnight_params.py --start 2023-09-15 --end 2026-09-14

掃描的參數（第一版先掃這幾個，範圍不用太大——你自己說了「慢慢測，慢慢優化」，
先看這幾個維度有沒有明顯的方向，比一次掃幾百種組合更容易看懂結果）：
    top_n              : 每日選幾檔
    tech_weight/chip_weight : 技術面/籌碼面權重（兩者相加=1）
    gap_stop_threshold : 跳空停損門檻（這次回測顯示 gap_stop 是最大的虧損來源，
                          值得優先確認這個門檻設定合不合理）
"""

import argparse
import datetime

import pandas as pd

import data_loader
import overnight_momentum_engine as ome
from compare_overnight import build_pipeline_inputs, split_is_oos, IS_RATIO

MIN_TRADES_FOR_RANKING = 30  # 交易筆數太少的組合，統計上不可靠，排名時排除


def build_param_grid():
    """
    回傳要掃描的參數組合 list[dict]。
    先固定 ATR 停利/停損倍數用引擎預設值(0.8/1.2)，只掃 top_n、權重、跳空門檻，
    避免第一版就是幾百種組合的全排列，掃出來的結果反而看不懂哪個維度在起作用。
    """
    combos = []
    for top_n in (3, 5, 8):
        for tech_weight, chip_weight in ((0.5, 0.5), (0.35, 0.65), (0.2, 0.8)):
            for gap_stop_threshold in (-0.010, -0.015, -0.020):
                combos.append({
                    "top_n": top_n,
                    "tech_weight": tech_weight,
                    "chip_weight": chip_weight,
                    "gap_stop_threshold": gap_stop_threshold,
                })
    return combos


def run_sweep(pipeline_inputs, is_ratio=IS_RATIO, param_grid=None):
    """
    對每個參數組合，只在 IS 區間跑回測，回傳依 profit_factor 排序的結果 DataFrame。
    （刻意不去看 OOS，避免「挑到讓 OOS 也好看的參數」這種變相偷看答案的行為）
    """
    param_grid = param_grid or build_param_grid()
    is_days, oos_days = split_is_oos(pipeline_inputs["trading_days"], is_ratio)

    common_args = dict(
        indicators_by_code=pipeline_inputs["indicators_by_code"],
        market_returns_df=pipeline_inputs["market_returns_df"],
        foreign_ratio_df=pipeline_inputs["foreign_ratio_df"],
        trust_ratio_df=pipeline_inputs["trust_ratio_df"],
        day_trading_ratio_df=pipeline_inputs["day_trading_ratio_df"],
        universe_codes=pipeline_inputs["universe_codes"],
        us_market_returns_df=pipeline_inputs.get("us_market_returns_df"),
        trading_days=is_days,
    )

    rows = []
    for i, params in enumerate(param_grid, start=1):
        trades = ome.run_overnight_backtest(**common_args, **params)
        summary = ome.summarize_overnight(trades)
        rows.append({**params, **{
            "total_trades": summary["total_trades"],
            "win_rate": summary["win_rate"],
            "profit_factor": summary["profit_factor"],
            "total_pnl": summary["total_pnl"],
            "avg_pnl": summary["avg_pnl"],
        }})
        print(f"[{i}/{len(param_grid)}] top_n={params['top_n']} "
              f"tech/chip={params['tech_weight']:.2f}/{params['chip_weight']:.2f} "
              f"gap_stop={params['gap_stop_threshold']:.3f} "
              f"-> {summary['total_trades']}筆, PF={summary['profit_factor']:.2f}, "
              f"損益={summary['total_pnl']:,.0f}", flush=True)

    result_df = pd.DataFrame(rows)
    return result_df, is_days, oos_days


def rank_results(result_df, min_trades=MIN_TRADES_FOR_RANKING):
    """依 profit_factor 排序，先濾掉交易筆數太少（統計上不可靠）的組合。"""
    reliable = result_df[result_df["total_trades"] >= min_trades].copy()
    if reliable.empty:
        print(f"⚠️ 沒有任何組合的交易筆數 >= {min_trades}，統計上都不夠可靠，全部組合原樣列出：")
        reliable = result_df.copy()
    return reliable.sort_values("profit_factor", ascending=False).reset_index(drop=True)


def validate_best_on_oos(pipeline_inputs, best_params, oos_days):
    """把排名第一的組合，拿到 OOS 跑一次，僅供參考、不能拿來重新挑參數。"""
    trades = ome.run_overnight_backtest(
        indicators_by_code=pipeline_inputs["indicators_by_code"],
        market_returns_df=pipeline_inputs["market_returns_df"],
        foreign_ratio_df=pipeline_inputs["foreign_ratio_df"],
        trust_ratio_df=pipeline_inputs["trust_ratio_df"],
        day_trading_ratio_df=pipeline_inputs["day_trading_ratio_df"],
        universe_codes=pipeline_inputs["universe_codes"],
        us_market_returns_df=pipeline_inputs.get("us_market_returns_df"),
        trading_days=oos_days,
        # 強制轉 int：呼叫端如果是從 DataFrame.iloc[0].to_dict() 拿到 best_params，
        # top_n 常常會被連帶轉成 float（例如 8.0），這裡再保險一次，不依賴呼叫端記得處理。
        top_n=int(best_params["top_n"]),
        tech_weight=best_params["tech_weight"],
        chip_weight=best_params["chip_weight"],
        gap_stop_threshold=best_params["gap_stop_threshold"],
    )
    return ome.summarize_overnight(trades)


def main():
    parser = argparse.ArgumentParser(description="隔日衝策略參數掃描（僅在IS內選參數）")
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

    grid = build_param_grid()
    print(f"\n共 {len(grid)} 組參數，只在樣本內(IS, 前70%)跑回測 ...\n")
    result_df, is_days, oos_days = run_sweep(pipeline_inputs, param_grid=grid)

    ranked = rank_results(result_df)
    print(f"\n=== IS 排名前 {args.top} 組合（依 profit_factor） ===")
    print(ranked.head(args.top).to_string(index=False))

    if not ranked.empty:
        best = ranked.iloc[0].to_dict()
        # ranked 是 DataFrame，iloc[0].to_dict() 會把整列轉成同一種 dtype，
        # 導致 top_n 這種整數欄位被浮點化(例如 8 變成 8.0)。
        # scan_candidates_for_date() 裡的 cand.head(top_n) 需要真正的 int，
        # 浮點數會讓 pandas 的 iloc 切片直接丟例外，這裡強制轉回 int 修正。
        best["top_n"] = int(best["top_n"])
        print(f"\n=== 用排名第一的組合在 OOS 驗證（僅供參考，不能用來重新選參數） ===")
        print(f"組合: top_n={best['top_n']}, tech/chip={best['tech_weight']}/{best['chip_weight']}, "
              f"gap_stop={best['gap_stop_threshold']}")
        oos_summary = validate_best_on_oos(pipeline_inputs, best, oos_days)
        print(f"OOS 交易次數: {oos_summary['total_trades']}")
        print(f"OOS 勝率: {oos_summary['win_rate']:.1f}%")
        print(f"OOS 盈虧比: {oos_summary['profit_factor']:.2f}")
        print(f"OOS 總損益: {oos_summary['total_pnl']:,.0f}")
        print(f"\n⚠️ 這只是同一份資料切出來的 OOS，不是獨立的第三段歷史期間。"
              f"要確認這組參數不是只在這3年的市場環境裡剛好有效，"
              f"還是要照專案既有標準，另外拿完全獨立的一段歷史資料做跨期驗證。")


if __name__ == "__main__":
    main()
