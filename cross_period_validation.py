"""
cross_period_validation.py
==============================
真正的跨期驗證：不是同一份資料切出來的IS/OOS，而是完全獨立的一段歷史區間——
這段資料從頭到尾都沒有被用來挑過任何訊號權重或ATR參數。

背景：用現有資料(2023-09-15~2026-09-14)做的IS(前70%)/OOS(後30%)驗證，
挑出的最佳候選組合是：
    訊號權重：只用 投信買超比重/量比/相對大盤強弱 這3個(可選投信加重2倍)
    ATR參數：停損0.3~0.5倍 / 停利3.0倍
IS結果 PF約1.25~1.38，OOS(同一份資料切出來的)PF約1.6。但這整個挑選過程
（訊號拆解 -> ATR掃描 -> 合併測試）都是在同一份資料上做的多層挑選，OOS也只是
同一份連續資料的最後一段，不是真正獨立的驗證——這支腳本就是要補上這一塊。

做法：把這些候選參數鎖定下來(不再調整)，直接在一段全新的、完全沒用過的歷史
區間上跑一次完整回測(不切IS/OOS，因為這整段資料本身就是「樣本外」)，看PF
是不是還站得住腳。如果在完全沒看過的資料上PF還是>1，才算是真正可信的訊號；
如果掉到1以下甚至更差，代表前面的結果是被這3年的市場環境或多層選擇撐起來的。

用法：
    python3 cross_period_validation.py --start 2021-09-15 --end 2023-09-14

⚠️ start/end 務必填一段跟原本(2023-09-15~2026-09-14)完全不重疊的期間，
不然就失去「獨立驗證」的意義了。
"""

import argparse
import datetime

import pandas as pd

import data_loader
import overnight_momentum_engine as ome
from compare_overnight import build_pipeline_inputs

# 鎖定候選參數：來自訊號拆解 + ATR掃描 + 合併測試選出來的最佳組合，
# 這裡不再調整，只是原封不動拿去一段全新的資料上驗證。
CANDIDATES = [
    {
        "label": "最強3個等權重(投信/量比/相對大盤強弱)，停損0.5倍/停利3.0倍",
        "signal_weights": {"score_trust": 1.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0},
        "atr_stop_mult": 0.5,
        "atr_target_mult": 3.0,
    },
    {
        "label": "最強3個等權重，停損0.3倍/停利3.0倍",
        "signal_weights": {"score_trust": 1.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0},
        "atr_stop_mult": 0.3,
        "atr_target_mult": 3.0,
    },
    {
        "label": "最強3個，投信加重2倍，停損0.3倍/停利3.0倍",
        "signal_weights": {"score_trust": 2.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0},
        "atr_stop_mult": 0.3,
        "atr_target_mult": 3.0,
    },
    {
        "label": "對照組：基準8訊號等權重，舊預設ATR(0.8/1.2)",
        "signal_weights": {
            "score_close_position": 1.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0,
            "score_gain_pct": 1.0, "score_price_level": 1.0, "score_day_trading": 1.0,
            "score_foreign": 1.0, "score_trust": 1.0,
        },
        "atr_stop_mult": 0.8,
        "atr_target_mult": 1.2,
    },
]


def run_cross_period_validation(pipeline_inputs, candidates=None, top_n=5):
    """對每個鎖定的候選組合，在整段(未切IS/OOS)資料上跑一次回測。"""
    candidates = candidates or CANDIDATES
    trading_days = pipeline_inputs["trading_days"]

    common_args = dict(
        indicators_by_code=pipeline_inputs["indicators_by_code"],
        market_returns_df=pipeline_inputs["market_returns_df"],
        foreign_ratio_df=pipeline_inputs["foreign_ratio_df"],
        trust_ratio_df=pipeline_inputs["trust_ratio_df"],
        day_trading_ratio_df=pipeline_inputs["day_trading_ratio_df"],
        universe_codes=pipeline_inputs["universe_codes"],
        us_market_returns_df=pipeline_inputs.get("us_market_returns_df"),
        ex_dividend_dates_by_code=pipeline_inputs.get("ex_dividend_dates_by_code"),
        trading_days=trading_days,
        top_n=top_n,
    )

    rows = []
    for i, cand in enumerate(candidates, start=1):
        trades = ome.run_overnight_backtest(
            **common_args,
            signal_weights=cand["signal_weights"],
            atr_stop_mult=cand["atr_stop_mult"],
            atr_target_mult=cand["atr_target_mult"],
        )
        summary = ome.summarize_overnight(trades)
        rows.append({
            "label": cand["label"],
            "total_trades": summary["total_trades"],
            "win_rate": summary["win_rate"],
            "profit_factor": summary["profit_factor"],
            "total_pnl": summary["total_pnl"],
            "avg_pnl": summary["avg_pnl"],
        })
        print(f"[{i}/{len(candidates)}] {cand['label']} "
              f"-> {summary['total_trades']}筆, PF={summary['profit_factor']:.2f}, "
              f"勝率={summary['win_rate']:.1f}%, 損益={summary['total_pnl']:,.0f}", flush=True)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="用鎖定的候選參數，在一段全新的獨立歷史區間上驗證")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD（務必跟原本的區間不重疊）")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD（務必跟原本的區間不重疊）")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--refresh", action="store_true", help="忽略快取，強制重新下載所有資料")
    args = parser.parse_args()

    start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()

    print(f"載入獨立驗證區間資料 ({args.start} ~ {args.end}) ...")
    print("⚠️ 這段區間如果跟原本的2023-09-15~2026-09-14重疊，就不是真正獨立的驗證，"
          "請確認 --start/--end 填的是全新的期間。")
    pipeline_inputs = build_pipeline_inputs(
        data_loader.STOCK_FUTURES_WHITELIST, start_date, end_date, refresh=args.refresh,
    )

    print(f"\n在整段獨立資料上（不切IS/OOS，因為這整段本身就是樣本外）跑鎖定的候選組合 ...\n")
    result_df = run_cross_period_validation(pipeline_inputs, top_n=args.top_n)

    print(f"\n=== 跨期驗證結果 ===")
    print(result_df.to_string(index=False))

    print(f"\n判讀方式：")
    print(f"這幾組參數是在另一份資料(2023-09-15~2026-09-14)上，經過訊號拆解、ATR掃描、"
          f"合併測試三層挑選出來的，現在是它們第一次接觸這段全新的資料。")
    print(f"如果「最強3個」那幾組的PF在這裡還是明顯 > 1（尤其是要贏過對照組），"
          f"代表這個選股+風控邏輯有機會是真訊號，不是被原本那3年的市場環境或"
          f"多層選擇撐起來的巧合；如果PF掉到1以下或接近對照組，"
          f"代表前面的結果沒有真正的跨期穩健性，還需要回頭重新設計，"
          f"不能直接拿去實盤用。")


if __name__ == "__main__":
    main()
