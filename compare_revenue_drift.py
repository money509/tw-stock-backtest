"""
compare_revenue_drift.py
===========================
月營收意外漂移(Monthly Revenue Surprise Drift)策略 - 主執行腳本。

跟compare_short_reversal.py同樣骨架，但只有兩階段(不是三階段)：這支引擎只有一個真正
要比較的參數(固定持有天數)，回看窗口的概念在這裡不存在(訊號是「這個月的意外」，不是
「過去N日報酬率」，沒有回看窗口可以比較)，見revenue_drift_engine.py模組docstring。

1. 固定持有天數比較(5 vs 10日，只在IS內)：選總損益最高、且交易筆數夠多的持有天數。
2. 最終IS/OOS切分 + OOS bootstrap穩健性檢查：把階段1選出的持有天數，真正驗證一次，
   報告PF、勝率、bootstrap正報酬比例、p值、90%信賴區間——跟compare_short_reversal.py
   一樣，這個專案看這類報告的習慣是bootstrap正報酬比例「想要>80%」當非正式及格線，
   這裡沿用同樣講法，不發明新的指標名稱或報告格式。

⚠️ 誠實揭露：這個策略假設(台灣月營收意外宣告後有延續效應)是台灣本地學術文獻的結果，
這支腳本跑出來的結果才是誠實的答案，不是先驗認定一定會賺錢，見revenue_drift_engine.py
模組docstring。

用法：
    python3 compare_revenue_drift.py --start 2023-09-15 --end 2026-09-14 --max-stocks 50
    python3 compare_revenue_drift.py --slippage-pct 0.002 --top-n 5

跟short_reversal_engine.py/mean_reversion_engine.py共用的已知限制(結算日近似、跌停
鎖死/注意股處置股未實作、倖存者偏差、保證金追繳/強制斷頭沒有完整模擬，加上revenue_
data_loader.py自己的已知限制：公告日是「次月10日」的近似值，不是每家公司真正公告的
那一天)在這裡一樣成立，請見 README/revenue_data_loader.py模組docstring。
"""
import argparse
import datetime
import os

import pandas as pd

from data_loader import load_price_data
from taifex_universe import STOCK_FUTURES_UNIVERSE
from revenue_data_loader import load_revenue_data
from revenue_drift_engine import (
    run_revenue_drift_backtest, precompute_revenue_drift_price_indicators,
    precompute_all_revenue_surprise, build_revenue_drift_events, build_entry_date_index,
    HOLD_DAYS_OPTIONS, DEFAULT_TOP_N, DEFAULT_ATR_STOP_MULT, TRAILING_MONTHS,
)
from mean_reversion_engine import summarize_mr
from robustness_analysis import (
    bootstrap_resample_pnl, summarize_bootstrap, pnl_excluding_top_n_trades, bootstrap_p_value,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_revenue_drift")

MIN_TRADES_FOR_RANKING = 20
DEFAULT_HOLD_DAYS_FOR_FALLBACK = HOLD_DAYS_OPTIONS[0]  # 階段1找不到任何可用結果時的退回值(5日)


def _fmt_pf(pf):
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def split_is_oos(master_calendar, is_ratio=0.7):
    n = len(master_calendar)
    split_point = int(n * is_ratio)
    return master_calendar[:split_point], master_calendar[split_point:]


def _prefer_nonzero_trades(df):
    """跟compare_short_reversal.py同名函式一樣的理由：0筆交易的組合不該因為總損益剛好
    是0.0，就被誤選成贏家，只有全部候選都是0筆時才保留。"""
    non_zero = df[df["trade_count"] > 0]
    return non_zero if not non_zero.empty else df


def run_hold_days_comparison(price_data, price_indicators_by_code, events_by_date, events_by_entry_date,
                              is_calendar, starting_capital, execution_kwargs, extra_kwargs=None):
    """階段1：只換「持有幾天後強制出場」(5/10日)，看哪個比較好。events_by_date/
    events_by_entry_date跟持有天數無關，外面只算一次、重複傳進來，不是每個變體重算。"""
    extra_kwargs = extra_kwargs or {}
    rows = []
    for hold_days in HOLD_DAYS_OPTIONS:
        trades = run_revenue_drift_backtest(
            price_data=price_data, price_indicators_by_code=price_indicators_by_code,
            events_by_date=events_by_date, events_by_entry_date=events_by_entry_date,
            master_calendar=is_calendar, max_hold_days=hold_days, starting_capital=starting_capital,
            allow_short=True, **execution_kwargs, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "hold_days": hold_days, "trade_count": stats["trade_count"],
            "profit_factor": stats["profit_factor"], "win_rate": stats["win_rate"],
            "total_pnl_ntd": stats["total_pnl_ntd"], "avg_hold_days": stats["avg_hold_days"],
        })
        print(f"  持有{hold_days}日 -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
              f"損益={stats['total_pnl_ntd']:,.0f}", flush=True)
    return pd.DataFrame(rows)


def select_winning_hold_days(hold_days_df, min_trades=MIN_TRADES_FOR_RANKING):
    pool = _prefer_nonzero_trades(hold_days_df)
    reliable = pool[pool["trade_count"] >= min_trades]
    if reliable.empty:
        reliable = pool
    if reliable.empty:
        return DEFAULT_HOLD_DAYS_FOR_FALLBACK  # 找不到任何可用結果時，退回5日
    best_row = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    return int(best_row["hold_days"])


def evaluate_final_combo(label, price_data, price_indicators_by_code, events_by_date, events_by_entry_date,
                          is_calendar, oos_calendar, starting_capital, hold_days, execution_kwargs,
                          extra_kwargs=None):
    """把階段1選出的持有天數，完整跑一次IS/OOS + OOS的bootstrap穩健性檢查。"""
    extra_kwargs = extra_kwargs or {}
    results = {"label": label}
    oos_trades = None
    for split_name, calendar in [("IS", is_calendar), ("OOS", oos_calendar)]:
        trades = run_revenue_drift_backtest(
            price_data=price_data, price_indicators_by_code=price_indicators_by_code,
            events_by_date=events_by_date, events_by_entry_date=events_by_entry_date,
            master_calendar=calendar, max_hold_days=hold_days, starting_capital=starting_capital,
            allow_short=True, **execution_kwargs, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        results[split_name] = stats
        if split_name == "OOS":
            oos_trades = trades

    bootstrap_results = bootstrap_resample_pnl(oos_trades or [], n_resamples=1000, seed=42)
    bootstrap_stats = summarize_bootstrap(bootstrap_results)
    results["bootstrap"] = {
        **bootstrap_stats,
        "p_value": bootstrap_p_value(bootstrap_results),
        "pnl_excluding_top3_ntd": pnl_excluding_top_n_trades(oos_trades or [], n=3),
    }
    results["oos_trades"] = oos_trades
    return results


def main():
    parser = argparse.ArgumentParser(description="月營收意外漂移策略 - 持有天數 + 最終組合驗證")
    parser.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=1095)).isoformat())
    parser.add_argument("--end", default=datetime.date.today().isoformat())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--starting-capital", type=float, default=200_000)
    parser.add_argument("--max-stocks", type=int, default=0,
                         help="限制掃描股票數量(0=全部249檔，測試時可以設小一點加快速度)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                         help="多方/空方各自取前幾名候選(預設5，同時最多可能有2倍top_n個部位)")
    parser.add_argument("--atr-stop-mult", type=float, default=DEFAULT_ATR_STOP_MULT,
                         help="ATR保護性停損倍數(安全網，不是主要出場依據，主要出場依據是固定持有天數)")
    parser.add_argument("--slippage-pct", type=float, default=0.0,
                         help="進場/停損出場的滑價假設(例如0.002代表0.2%%)，預設0(不模擬滑價)")
    parser.add_argument("--max-concurrent-positions", type=int, default=None,
                         help="同時最多持有的部位數，預設2*top_n(多空雙向各top_n名都進場)")
    parser.add_argument("--trailing-months", type=int, default=TRAILING_MONTHS,
                         help="算「營收意外」用的過去幾個月平均視窗(預設6個月)")
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
        raise RuntimeError(f"參考交易日曆用的標的 {INDEX_PROXY_CODE} 沒有成功下載，無法繼續。")
    index_df = price_data[INDEX_PROXY_CODE]
    master_calendar = index_df.index
    is_calendar, oos_calendar = split_is_oos(master_calendar)
    print(f"樣本內(IS)天數={len(is_calendar)}，樣本外(OOS)天數={len(oos_calendar)}\n")

    print("下載月營收資料 ...")
    revenue_data = load_revenue_data(
        args.start, args.end, universe_codes=set(universe.keys()), refresh=args.refresh,
    )
    print(f"成功取得 {len(revenue_data)} 檔股票的月營收資料\n")

    print("預先計算全市場價格面指標(ATR/成交金額) ...")
    price_indicators_by_code = precompute_revenue_drift_price_indicators(price_data, universe)
    print(f"預先計算每檔股票的月營收意外(surprise)序列(過去{args.trailing_months}個月平均為基準) ...")
    surprise_by_code = precompute_all_revenue_surprise(revenue_data, trailing_months=args.trailing_months)
    events_by_date = build_revenue_drift_events(surprise_by_code)
    events_by_entry_date = build_entry_date_index(events_by_date, master_calendar)
    print(f"指標預計算完成，涵蓋 {len(price_indicators_by_code)} 檔股票的價格指標、"
          f"{len(events_by_date)} 個公告可見日、{sum(len(v) for v in events_by_date.values())} 筆事件\n")

    max_concurrent_positions = args.max_concurrent_positions
    if max_concurrent_positions is None:
        max_concurrent_positions = 2 * args.top_n

    execution_kwargs = dict(
        lots=2, top_n=args.top_n, atr_stop_mult=args.atr_stop_mult,
        max_concurrent_positions=max_concurrent_positions,
    )
    extra_kwargs = dict(slippage_pct=args.slippage_pct)

    if args.slippage_pct > 0:
        print(f"⚠️ 滑價模擬：進場/停損出場套用 {args.slippage_pct:.2%} 滑價\n")

    print(f"[階段1] 固定持有天數比較({'/'.join(str(h) for h in HOLD_DAYS_OPTIONS)}日，只在IS內) ...")
    hold_days_df = run_hold_days_comparison(
        price_data, price_indicators_by_code, events_by_date, events_by_entry_date,
        is_calendar, args.starting_capital, execution_kwargs, extra_kwargs=extra_kwargs,
    )
    hold_days_df.to_csv(os.path.join(RESULTS_DIR, "hold_days_comparison.csv"),
                         index=False, encoding="utf-8-sig")
    winning_hold_days = select_winning_hold_days(hold_days_df)
    print(f"  → 選出持有天數：{winning_hold_days}日\n")

    print(f"[階段2] 最終組合(持有{winning_hold_days}日)：IS vs OOS + bootstrap穩健性檢查 ...")
    final_result = evaluate_final_combo(
        f"最終組合(持有{winning_hold_days}日)",
        price_data, price_indicators_by_code, events_by_date, events_by_entry_date,
        is_calendar, oos_calendar, args.starting_capital, winning_hold_days,
        execution_kwargs, extra_kwargs=extra_kwargs,
    )
    for split_name in ["IS", "OOS"]:
        stats = final_result[split_name]
        print(f"  {split_name} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
              f"損益={stats['total_pnl_ntd']:,.0f}")
    b = final_result["bootstrap"]
    print(f"  bootstrap：正報酬比例={b['pct_positive']:.1f}%, p值={b['p_value']:.3f}, "
          f"拿掉最大3筆後損益={b['pnl_excluding_top3_ntd']:,.0f}\n")
    pd.DataFrame(final_result["oos_trades"]).to_csv(
        os.path.join(RESULTS_DIR, "trades_OOS_final_combo.csv"), index=False, encoding="utf-8-sig",
    )

    summary_lines = [
        "=" * 100,
        "月營收意外漂移(Revenue Surprise Drift)策略 持有天數 + 最終組合驗證",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}　"
        f"top_n={args.top_n}　最大並行部位={max_concurrent_positions}　"
        f"ATR安全網停損倍數={args.atr_stop_mult}　過去{args.trailing_months}個月平均為基準　"
        f"滑價：{args.slippage_pct:.2%}",
        "=" * 100,
        "\n--- 階段1：固定持有天數比較(IS) ---",
    ]
    header1 = f"{'持有天數':<10}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'平均持有天':>10}{'總損益NT$':>14}"
    summary_lines.append(header1)
    summary_lines.append("-" * len(header1))
    for _, r in hold_days_df.sort_values("total_pnl_ntd", ascending=False).iterrows():
        summary_lines.append(
            f"{str(int(r['hold_days'])) + '日':<10}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}"
            f"{r['win_rate']:>8.1f}{r['avg_hold_days']:>10.1f}{r['total_pnl_ntd']:>14,.0f}"
        )

    summary_lines.append(f"\n--- 階段2：最終組合 IS vs OOS + bootstrap ---")
    summary_lines.append(f"\n[{final_result['label']}]")
    for split_name in ["IS", "OOS"]:
        stats = final_result[split_name]
        split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
        summary_lines.append(
            f"  {split_full}: {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
            f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
            f"總損益NT${stats['total_pnl_ntd']:,.0f}, "
            f"最大回撤NT${stats['max_drawdown_ntd']:,.0f}, 最大連續虧損{stats['max_consecutive_losses']}筆"
        )
    summary_lines.append(
        f"  [穩健性] OOS bootstrap 1000次重抽樣：平均總損益NT${b['mean']:,.0f}，"
        f"90%信賴區間=[NT${b['p5']:,.0f}, NT${b['p95']:,.0f}]，"
        f"正報酬比例={b['pct_positive']:.1f}%，p值={b['p_value']:.3f}，"
        f"拿掉最大3筆交易後總損益NT${b['pnl_excluding_top3_ntd']:,.0f}"
    )

    summary_lines.append(
        "\n判讀方式：先看階段1的IS總損益，確認持有天數的選擇不是在挑0筆或極少筆交易的雜訊"
        "組合(交易筆數太少的組合會被自動排除排名，見_prefer_nonzero_trades()/"
        "MIN_TRADES_FOR_RANKING)。再看階段2的OOS數字，這是比IS更誠實的參考依據。bootstrap"
        "正報酬比例是這個專案看這類報告的習慣：想要>80%才算「優勢在不同交易順序下都站得住」，"
        "p值越接近0越好，超過0.2以上不該當作已驗證的優勢看待。這支引擎測試的是台灣本地學術"
        "文獻裡的月營收意外漂移假說，用事件驅動的方式(只在公告可見日附近進出，不是整個月都"
        "掛著訊號)，是跟equity_swing_engine.py(把營收當成緩慢變化的月頻排名訊號持有數週)"
        "完全不同的用法——這份報告的OOS/bootstrap數字才是誠實的答案，不是先驗認定一定會"
        "賺錢(見revenue_drift_engine.py模組docstring)。"
    )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


if __name__ == "__main__":
    main()
