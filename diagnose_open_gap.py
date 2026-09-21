"""
diagnose_open_gap.py
=======================
診斷用腳本（不改變任何現有策略邏輯，純分析）：驗證「隔日沒開紅（開盤價沒有
高於進場價）就提早在開盤出場」這個假設，值不值得做成正式規則。

背景：現在的出場邏輯要等到碰到停損/停利/跳空停損，或撐到收盤被強制平倉。
如果隔天一開盤動能就已經不對了（沒有開高），現在的邏輯還是會讓這筆交易照樣
跑完一整天。這支腳本要回答的問題是：

    那些「隔天沒開紅」的交易，照現在的邏輯跑完一整天，最後的損益到底是好是壞？
    如果本來就是負的/打平的，那「提早在開盤出場」就有道理；
    如果裡面其實藏著不少後來反而翻正的交易，提早出場反而是砍在阿呆谷。

做法：對既有的 run_overnight_backtest() 產生的每一筆交易，額外去查「隔天開盤價」，
分成「有開紅」vs「沒開紅」兩組，比較：
  (a) 實際損益（現有邏輯跑完一整天的結果）
  (b) 假設性損益（如果那天直接在開盤價出場，會是多少）
兩組的差異就是「提早出場」這個規則可能帶來的影響，不需要真的先實作規則
再回測——用現有的回測結果就能算出來。

用法：
    python3 diagnose_open_gap.py --start 2023-09-15 --end 2026-09-14
"""

import argparse
import datetime

import pandas as pd

import data_loader
import overnight_momentum_engine as ome
from compare_overnight import build_pipeline_inputs, split_is_oos, IS_RATIO


def run_backtest_with_open_diagnostics(indicators_by_code, market_returns_df,
                                        foreign_ratio_df, trust_ratio_df,
                                        day_trading_ratio_df, trading_days,
                                        universe_codes=None, top_n=ome.TOP_N,
                                        tech_weight=ome.TECH_WEIGHT,
                                        chip_weight=ome.CHIP_WEIGHT,
                                        fee_per_trade=200,
                                        gap_stop_threshold=ome.GAP_STOP_THRESHOLD,
                                        atr_stop_mult=ome.ATR_STOP_MULT,
                                        atr_target_mult=ome.ATR_TARGET_MULT,
                                        us_market_returns_df=None,
                                        us_market_drop_threshold=ome.US_MARKET_DROP_THRESHOLD_PCT,
                                        signal_weights=None,
                                        ex_dividend_dates_by_code=None):
    """
    跑一次正常的隔日衝回測，再對每一筆交易額外查出「隔天開盤價」，
    算出「有沒有開紅」(opened_up) 跟「假設在開盤價出場的損益」(hypothetical_open_exit_pnl)。
    回傳 DataFrame，每一列是一筆交易，欄位比 summarize_overnight() 用的 trades 多這兩個。
    """
    trades = ome.run_overnight_backtest(
        indicators_by_code=indicators_by_code,
        market_returns_df=market_returns_df,
        foreign_ratio_df=foreign_ratio_df,
        trust_ratio_df=trust_ratio_df,
        day_trading_ratio_df=day_trading_ratio_df,
        trading_days=trading_days,
        universe_codes=universe_codes,
        top_n=top_n,
        tech_weight=tech_weight,
        chip_weight=chip_weight,
        fee_per_trade=fee_per_trade,
        gap_stop_threshold=gap_stop_threshold,
        atr_stop_mult=atr_stop_mult,
        atr_target_mult=atr_target_mult,
        us_market_returns_df=us_market_returns_df,
        us_market_drop_threshold=us_market_drop_threshold,
        signal_weights=signal_weights,
        ex_dividend_dates_by_code=ex_dividend_dates_by_code,
    )

    enriched_rows = []
    for t in trades:
        code = t["code"]
        exit_date = t["exit_date"]
        df = indicators_by_code.get(code)
        if df is None:
            continue
        next_row = df[df["date"] == exit_date]
        if next_row.empty or pd.isna(next_row.iloc[0]["open"]):
            continue
        next_open = float(next_row.iloc[0]["open"])
        entry_price = t["entry_price"]

        mult = ome.get_contract_multiplier(entry_price)
        hypothetical_pnl = (next_open - entry_price) * mult - fee_per_trade

        enriched_rows.append({
            **t,
            "next_open": next_open,
            "opened_up": next_open > entry_price,
            "hypothetical_open_exit_pnl": hypothetical_pnl,
        })

    return pd.DataFrame(enriched_rows)


def summarize_by_open_gap(enriched_df):
    """依「隔天有没有開紅」分兩組，比較實際損益 vs 假設在開盤出場的損益。"""
    if enriched_df.empty:
        return pd.DataFrame(columns=[
            "opened_up", "trade_count", "avg_actual_pnl",
            "avg_hypothetical_open_exit_pnl", "total_actual_pnl",
        ])
    grouped = enriched_df.groupby("opened_up").agg(
        trade_count=("pnl", "count"),
        avg_actual_pnl=("pnl", "mean"),
        avg_hypothetical_open_exit_pnl=("hypothetical_open_exit_pnl", "mean"),
        total_actual_pnl=("pnl", "sum"),
    ).reset_index()
    return grouped


def exit_reason_breakdown_for_group(enriched_df, opened_up_value=False):
    """看「沒開紅」這組裡面，現有邏輯實際上是怎麼出場的（分佈在哪些 exit_reason）。"""
    subset = enriched_df[enriched_df["opened_up"] == opened_up_value]
    if subset.empty:
        return pd.Series(dtype=float)
    return (subset["exit_reason"].value_counts(normalize=True) * 100.0).round(1)


def summarize_by_open_gap_and_exit_reason(enriched_df):
    """
    跟 summarize_by_open_gap() 一樣，但多依 exit_reason 再拆一層。

    原因：summarize_by_open_gap() 算出來的「沒開紅」整組平均，其實混了兩種
    完全不同的情況——一種是本來就已經觸發停損/跳空停損提早出場的交易（現有邏輯
    本來就跟「提早出場」做的事差不多，不會有太大差異）；另一種是撐到收盤才被
    強制平倉的交易（這才是「提早出場」這個規則真正能改變結果的地方）。
    混在一起平均，會讓「強制平倉」那組真正的改善幅度被稀釋或放大，看不清楚。
    """
    if enriched_df.empty:
        return pd.DataFrame(columns=[
            "opened_up", "exit_reason", "trade_count",
            "avg_actual_pnl", "avg_hypothetical_open_exit_pnl",
        ])
    grouped = enriched_df.groupby(["opened_up", "exit_reason"]).agg(
        trade_count=("pnl", "count"),
        avg_actual_pnl=("pnl", "mean"),
        avg_hypothetical_open_exit_pnl=("hypothetical_open_exit_pnl", "mean"),
    ).reset_index()
    return grouped.sort_values(["opened_up", "exit_reason"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="診斷：隔日沒開紅就提早出場，值不值得做")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--refresh", action="store_true", help="忽略快取，強制重新下載所有資料")
    args = parser.parse_args()

    start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()

    print(f"載入資料 ({args.start} ~ {args.end})，若快取已存在會直接使用，不重新下載 ...")
    pipeline_inputs = build_pipeline_inputs(
        data_loader.STOCK_FUTURES_WHITELIST, start_date, end_date, refresh=args.refresh,
    )

    is_days, oos_days = split_is_oos(pipeline_inputs["trading_days"], IS_RATIO)

    print(f"\n用預設參數在樣本內(IS, 前70%)跑一次回測，並額外查每筆交易隔天的開盤價 ...\n")
    enriched = run_backtest_with_open_diagnostics(
        indicators_by_code=pipeline_inputs["indicators_by_code"],
        market_returns_df=pipeline_inputs["market_returns_df"],
        foreign_ratio_df=pipeline_inputs["foreign_ratio_df"],
        trust_ratio_df=pipeline_inputs["trust_ratio_df"],
        day_trading_ratio_df=pipeline_inputs["day_trading_ratio_df"],
        universe_codes=pipeline_inputs["universe_codes"],
        us_market_returns_df=pipeline_inputs.get("us_market_returns_df"),
        ex_dividend_dates_by_code=pipeline_inputs.get("ex_dividend_dates_by_code"),
        trading_days=is_days,
    )

    if enriched.empty:
        print("沒有交易資料，無法診斷。")
        return

    summary = summarize_by_open_gap(enriched)
    print("=== 依「隔天有没有開紅」分組比較 ===")
    print(summary.to_string(index=False))

    print("\n=== 「沒開紅」這組，現有邏輯實際上是怎麼出場的 ===")
    breakdown = exit_reason_breakdown_for_group(enriched, opened_up_value=False)
    if breakdown.empty:
        print("（沒有資料）")
    else:
        print(breakdown.to_string())

    print("\n=== 「沒開紅」這組，依現有出場原因再拆一層比較 ===")
    print("（重要：停損/跳空停損本來就已經提早出場了，真正該看的是「強制平倉」這一列——"
          "那才是現在撐到收盤、而「提早出場」規則真正能改變結果的地方）")
    breakdown_pnl = summarize_by_open_gap_and_exit_reason(enriched)
    not_opened_up_breakdown = breakdown_pnl[breakdown_pnl["opened_up"] == False]
    if not_opened_up_breakdown.empty:
        print("（沒有資料）")
    else:
        print(not_opened_up_breakdown[
            ["exit_reason", "trade_count", "avg_actual_pnl", "avg_hypothetical_open_exit_pnl"]
        ].to_string(index=False))

    not_opened_up = summary[summary["opened_up"] == False]
    if not not_opened_up.empty:
        row = not_opened_up.iloc[0]
        print(f"\n判讀方式（整體）：")
        print(f"「沒開紅」這組共 {int(row['trade_count'])} 筆，"
              f"現有邏輯（跑完一整天）的平均損益是 {row['avg_actual_pnl']:,.0f}，"
              f"如果直接在開盤價出場，平均損益會是 {row['avg_hypothetical_open_exit_pnl']:,.0f}。")
        print("但這個整體平均混了停損/跳空停損（本來就提早出場，改善幅度理應很小）"
              "跟強制平倉（現在撐到收盤，改善幅度才是重點）——請直接看上面「強制平倉」那一列：")

        forced_close_row = not_opened_up_breakdown[
            not_opened_up_breakdown["exit_reason"] == "forced_close"
        ]
        if not forced_close_row.empty:
            fc = forced_close_row.iloc[0]
            fc_diff = fc["avg_hypothetical_open_exit_pnl"] - fc["avg_actual_pnl"]
            print(f"強制平倉這 {int(fc['trade_count'])} 筆，"
                  f"現有邏輯平均損益 {fc['avg_actual_pnl']:,.0f}，"
                  f"假設開盤就出場平均損益 {fc['avg_hypothetical_open_exit_pnl']:,.0f}，"
                  f"差 {fc_diff:,.0f}。")
            if fc_diff > 0:
                print("→ 強制平倉這批單獨拆開來看，提早在開盤出場確實比較好，"
                      "代表「沒開紅就跑」這個規則的效果不是被停損/跳空停損稀釋出來的假象，"
                      "值得做成正式規則並在 OOS 驗證。")
            else:
                print("→ 強制平倉這批單獨拆開來看，提早在開盤出場反而比較差（或差不多），"
                      "代表整體平均的改善主要是被停損/跳空停損那兩類「本來就提早出場」的交易帶動的，"
                      "「沒開紅就跑」這個規則本身，對真正該改善的強制平倉這批，可能沒有實質幫助，"
                      "不建議只憑這個結果就正式做進去。")
        else:
            print("（強制平倉這組沒有足夠資料可以單獨判讀）")

        print("\n⚠️ 這只是同一份 IS 資料上的假設性計算（不是真的改邏輯重跑），"
              "如果結果顯示值得做，還是要把規則真的寫進 run_overnight_backtest()，"
              "在 IS 選完之後拿 OOS 驗證一次，才能真正確認不是巧合。")


if __name__ == "__main__":
    main()
