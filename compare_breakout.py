"""
compare_breakout.py
====================
右側順勢突破策略 - 主執行腳本。

三階段(單一訊號拆解 → 結構門檻變體比較 → 最終組合IS/OOS驗證)的骨架維持不變，
這一輪針對「回測方法論本身」跟「實戰執行細節」做了幾個修正：

1. **修正階段3沒有真正驗證「最佳組合」的問題**：舊版階段3不管前兩階段找出什麼，
   永遠固定跑「全部訊號等權重 + 基本門檻」。這次改成：先依階段1/2在IS的結果，
   自動挑出「PF>1且交易筆數夠多」的訊號、以及總損益最高的門檻，組成「最佳組合」，
   拿這組真正被驗證過的組合去跑IS/OOS/bootstrap，不是驗證一個跟選股結果無關的
   固定基準。舊的「全部等權重+基本門檻」還是會一起跑，當作對照組留著方便比較，
   但報告會明確標示哪一組才是「篩選後的答案」。
2. **ATR倍數敏感度測試(僅移動停利模式)**：在最佳訊號/門檻組合固定之後，額外
   在IS內測一個小網格的(初始停損倍數, 移動停利倍數)組合，選PF最高的一組帶進
   最終驗證，取代原本寫死的1.0倍。刻意只在IS內搜尋、只搜一個很小的網格，
   避免搜索空間太大又製造新的多重比較問題。
3. **跨市場週期驗證(--multi-period-test)**：目前的回測期間大部分落在多頭段，
   對順勢策略先天有利。這個選項會額外抓兩段歷史上市況明顯不同的期間(2022年
   有過一輪顯著修正、2021下半年到2022年初有一段偏盤整)，把最終驗證出的
   組合原封不動地套用過去跑一次，檢查PF/bootstrap正報酬比例是不是一到空頭
   /盤整就明顯轉差——如果是，代表這組優勢的本質比較接近「牛市beta」，不是
   在任何市況都成立的右側交易優勢。
4. **滑價模擬(--slippage-pct)**：進場、以及停損/跳空停損這類市價出場，
   額外套用一個滑價百分比，比原本假設「開盤價/理論停損價就是成交價」更保守寫實。
5. **風險預算式部位大小(--risk-pct-per-trade)**：取代固定口數，每筆交易的口數
   改用「帳戶權益 x 風險比例 ÷ 停損距離」反推，不同波動度的標的不會用同樣口數
   扛到不一樣的風險，帳戶權益也會隨已實現損益動態調整(複利)。只套用在最終驗證
   的組合，不套用在訊號/門檻篩選階段(篩選階段要的是「哪個訊號有預測力」，
   跟部位大小怎麼配是兩件事，混在一起篩選反而讓比較不是蘋果比蘋果)。
6. **多部位並行(--max-concurrent-positions)**：允許同時持有多檔不同標的的部位，
   不再是「一次只能一個部位、資金大部分時間閒置」。同樣只套用在最終驗證階段。

用法：
    python3 compare_breakout.py --start 2023-09-15 --end 2026-09-14 --max-stocks 50
    python3 compare_breakout.py --with-chip-confirm --clean-ex-dividend
    python3 compare_breakout.py --use-trailing-stop --atr-stop-mult 1.0
    python3 compare_breakout.py --use-trailing-stop --max-concurrent-positions 3 \\
        --risk-pct-per-trade 0.02 --slippage-pct 0.002 --multi-period-test

⚠️ 誠實揭露：跟 mean_reversion_engine.py 共用的已知限制(結算日近似、大盤氛圍濾網用0050
代理、跌停鎖死/注意股處置股未實作、倖存者偏差、保證金追繳/強制斷頭沒有完整模擬)在這裡
一樣成立，請見 README。
"""
import argparse
import datetime
import os

import pandas as pd

from data_loader import load_price_data
from taifex_universe import STOCK_FUTURES_UNIVERSE
from momentum_breakout_engine import (
    run_momentum_breakout_backtest, BREAKOUT_SIGNAL_NAMES, CHIP_DEPENDENT_SIGNALS,
    precompute_all_breakout_indicators,
)
from mean_reversion_engine import summarize_mr, precompute_regime_series
from robustness_analysis import (
    bootstrap_resample_pnl, summarize_bootstrap, pnl_excluding_top_n_trades, bootstrap_p_value,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_breakout")

MIN_TRADES_FOR_RANKING = 30
MIN_TRADES_FOR_GATE_RANKING = 10
TRAILING_STOP_MAX_HOLD_DAYS = 250  # 工程上的安全上限，不是真正的出場依據，見上方docstring

SIGNAL_LABELS = {
    "score_volume_ratio": "量比",
    "score_rel_strength": "相對大盤強弱",
    "score_rsi_cross": "RSI穿越50(動能轉強)",
    "score_macd": "MACD柱狀圖",
    "score_golden_cross": "均線黃金/死亡交叉",
    "score_price_volume_new_high": "價量同步創高/創低",
    "score_gap_breakout": "跳空突破",
    "score_candle_body": "K棒實體比例",
    "score_foreign_ratio": "外資買超比重",
    "score_trust_ratio": "投信買超比重",
}

HOLD_DAYS_OPTIONS = [("短線5天", 5), ("中期15天", 15)]
TRAILING_STOP_HOLD_DAYS_OPTIONS = [("移動停利(不設固定天數)", TRAILING_STOP_MAX_HOLD_DAYS)]

GATE_VARIANTS = [
    ("基本門檻(站上均線+創新高)", {}),
    ("+站上季線", {"require_above_ma60": True}),
    ("+均線多頭排列", {"require_ma_bullish_alignment": True}),
    ("+爆量(量比>=1.5)", {"min_volume_ratio": 1.5}),
    ("+雙法人同步買超", {"require_dual_institutional_buy": True}),
]
GATE_KWARGS_BY_LABEL = dict(GATE_VARIANTS)

# 移動停利模式下的ATR倍數敏感度網格：(初始停損倍數, 移動停利倍數)。
# 刻意只有6組、只在IS內搜尋，避免搜索空間太大又製造新的多重比較問題。
ATR_SENSITIVITY_GRID = [(0.8, 1.0), (1.0, 1.0), (1.0, 1.5), (1.5, 1.5), (1.5, 2.0), (0.8, 1.5)]

# 跨市場週期驗證用的額外歷史窗口：刻意挑跟近期多頭段落明顯不同的市況，
# 檢查最終驗證出的組合是不是只吃到牛市紅利。實際PF/bootstrap結果要看下載回來的
# 真實資料，這裡只是選定「市況應該明顯不同」的日期區間，不對回測結果預設立場。
EXTRA_REGIME_PERIODS = [
    ("2022年修正段", "2022-01-01", "2022-10-31"),
    ("2021下半年~2022年初盤整段", "2021-07-01", "2022-01-15"),
]


def split_is_oos(master_calendar, is_ratio=0.7):
    n = len(master_calendar)
    split_point = int(n * is_ratio)
    return master_calendar[:split_point], master_calendar[split_point:]


def _fmt_pf(pf):
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def _prefer_nonzero_trades(df):
    """
    篩選/排名之前一律先套用：0筆交易的組合(通常是門檻太嚴格、篩不出任何候選)
    完全沒有驗證意義，不該因為它的總損益剛好是0.0、比其他有交易但賠錢的組合
    「數字上更高」就被誤選成贏家。只有在所有候選都是0筆交易時(全部都篩不出
    候選)才會保留0筆的列，讓呼叫方至少拿得到一個可以印出來的結果並示警。
    """
    non_zero = df[df["trade_count"] > 0]
    return non_zero if not non_zero.empty else df


def select_winning_signals(ablation_df, min_trades=MIN_TRADES_FOR_RANKING):
    """
    從單一訊號拆解結果裡，挑出「交易筆數夠多、且PF>1」的訊號組成最終要用的訊號組合。
    如果沒有任何訊號同時滿足這兩個條件(常見於樣本還不夠大、或訊號真的都沒用)，
    退回選總損益前3名，並回傳reliable=False，提醒這組是探索性的、還沒被真正驗證過。
    """
    non_baseline = _prefer_nonzero_trades(ablation_df[ablation_df["signal"] != "__baseline_equal_weight__"])
    reliable_pool = non_baseline[non_baseline["trade_count"] >= min_trades]
    picked = reliable_pool[reliable_pool["profit_factor"] > 1.0]
    if picked.empty:
        fallback = non_baseline.sort_values("total_pnl_ntd", ascending=False).head(3)
        return list(fallback["signal"]), False
    return list(picked["signal"]), True


def select_winning_gate(gate_df, min_trades=MIN_TRADES_FOR_GATE_RANKING):
    """從門檻變體比較結果裡，挑總損益最高、且交易筆數不會太少的那組門檻。"""
    pool = _prefer_nonzero_trades(gate_df)
    reliable = pool[pool["trade_count"] >= min_trades]
    if reliable.empty:
        reliable = pool
    if reliable.empty:
        return GATE_VARIANTS[0]
    best_row = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    label = best_row["gate"]
    return label, GATE_KWARGS_BY_LABEL[label]


def run_signal_ablation(price_data, indicators_by_code, regime_series, is_calendar,
                         starting_capital, hold_days, has_chip=False, extra_kwargs=None):
    """對 SIGNAL_LABELS 裡每個訊號單獨開啟(權重1.0，其餘0)，用基本門檻在IS跑一次回測。
    這裡固定用lots=2、concurrency=1，不套用風險預算式部位大小/多部位並行——篩選階段
    要回答的是「哪個訊號有預測力」，部位大小怎麼配是另一個議題，混在一起篩選會讓
    「訊號A表現比訊號B好」這個比較摻雜了部位大小的干擾，不是乾淨的訊號對照。"""
    extra_kwargs = extra_kwargs or {}
    rows = []
    signal_names = [s for s in BREAKOUT_SIGNAL_NAMES if has_chip or s not in CHIP_DEPENDENT_SIGNALS]

    common_kwargs = dict(
        price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
        master_calendar=is_calendar, max_hold_days=hold_days, starting_capital=starting_capital,
        allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0,
        **extra_kwargs,
    )

    for i, signal_name in enumerate(signal_names, start=1):
        trades = run_momentum_breakout_backtest(**common_kwargs, signal_weights={signal_name: 1.0})
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "signal": signal_name, "label": SIGNAL_LABELS[signal_name],
            "trade_count": stats["trade_count"], "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "total_pnl_ntd": stats["total_pnl_ntd"], "max_drawdown_ntd": stats["max_drawdown_ntd"],
            "max_consecutive_losses": stats["max_consecutive_losses"],
        })
        print(f"  [{i}/{len(signal_names)}] {SIGNAL_LABELS[signal_name]} "
              f"-> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)

    baseline_trades = run_momentum_breakout_backtest(**common_kwargs, signal_weights={name: 1.0 for name in signal_names})
    baseline_stats = summarize_mr(baseline_trades, starting_capital)
    rows.append({
        "signal": "__baseline_equal_weight__", "label": f"基準({len(signal_names)}訊號等權重)",
        "trade_count": baseline_stats["trade_count"], "win_rate": baseline_stats["win_rate"],
        "profit_factor": baseline_stats["profit_factor"],
        "total_pnl_ntd": baseline_stats["total_pnl_ntd"], "max_drawdown_ntd": baseline_stats["max_drawdown_ntd"],
        "max_consecutive_losses": baseline_stats["max_consecutive_losses"],
    })
    print(f"  [基準] {len(signal_names)}訊號等權重 -> {baseline_stats['trade_count']}筆, "
          f"PF={_fmt_pf(baseline_stats['profit_factor'])}, 勝率={baseline_stats['win_rate']:.1f}%, "
          f"損益={baseline_stats['total_pnl_ntd']:,.0f}", flush=True)

    return pd.DataFrame(rows), signal_names


def run_gate_comparison(price_data, indicators_by_code, regime_series, is_calendar,
                         starting_capital, hold_days, signal_names, has_chip=False, extra_kwargs=None):
    """用等權重訊號當基準，把結構門檻換成不同變體，比較哪種過濾條件篩出的候選比較好。"""
    extra_kwargs = extra_kwargs or {}
    rows = []
    equal_weights = {name: 1.0 for name in signal_names}
    for label, gate_kwargs in GATE_VARIANTS:
        if gate_kwargs.get("require_dual_institutional_buy") and not has_chip:
            continue
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=hold_days, starting_capital=starting_capital,
            allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0,
            signal_weights=equal_weights, **gate_kwargs, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "gate": label, "trade_count": stats["trade_count"], "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "total_pnl_ntd": stats["total_pnl_ntd"], "max_drawdown_ntd": stats["max_drawdown_ntd"],
        })
        print(f"  {label} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)
    return pd.DataFrame(rows)


def run_atr_sensitivity_grid(price_data, indicators_by_code, regime_series, is_calendar,
                              starting_capital, signal_weights, gate_kwargs, extra_kwargs):
    """僅移動停利模式使用：在最佳訊號/門檻組合固定之後，測一個小網格的
    (初始停損倍數, 移動停利倍數)，只在IS內搜尋，回傳完整結果表 + PF最高的一組。"""
    rows = []
    for stop_mult, trail_mult in ATR_SENSITIVITY_GRID:
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=TRAILING_STOP_MAX_HOLD_DAYS,
            starting_capital=starting_capital, allow_short=True, lots=2,
            signal_weights=signal_weights, atr_stop_mult=stop_mult,
            use_trailing_stop=True, trailing_atr_mult=trail_mult,
            **gate_kwargs, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "atr_stop_mult": stop_mult, "trailing_atr_mult": trail_mult,
            "trade_count": stats["trade_count"], "profit_factor": stats["profit_factor"],
            "total_pnl_ntd": stats["total_pnl_ntd"],
        })
        print(f"  停損x{stop_mult} / 移動停利x{trail_mult} -> {stats['trade_count']}筆, "
              f"PF={_fmt_pf(stats['profit_factor'])}, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)

    df = pd.DataFrame(rows)
    pool = _prefer_nonzero_trades(df)
    reliable = pool[pool["trade_count"] >= MIN_TRADES_FOR_RANKING]
    if reliable.empty:
        reliable = pool
    best = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    if best["trade_count"] == 0:
        print("  ⚠️ 整個ATR網格都篩不出任何候選(交易筆數全部是0)，回傳的組合僅供參考，無法驗證")
    return df, (float(best["atr_stop_mult"]), float(best["trailing_atr_mult"]))


def evaluate_combo(label, price_data, indicators_by_code, regime_series, is_calendar, oos_calendar,
                    starting_capital, hold_days, signal_weights, gate_kwargs, atr_stop_mult,
                    trailing_atr_mult, use_trailing_stop, extra_kwargs, execution_kwargs):
    """把一組(訊號權重, 門檻, ATR倍數)組合，完整跑一次IS/OOS + OOS的bootstrap穩健性檢查，
    回傳一個彙整好的dict，用於階段3輸出跟跨週期驗證共用同一套邏輯。"""
    results = {"label": label}
    common = dict(
        price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
        starting_capital=starting_capital, allow_short=True, atr_stop_mult=atr_stop_mult,
        signal_weights=signal_weights, use_trailing_stop=use_trailing_stop,
        trailing_atr_mult=trailing_atr_mult if use_trailing_stop else None,
        **gate_kwargs, **extra_kwargs, **execution_kwargs,
    )
    if "lots" not in execution_kwargs:
        common["lots"] = 2
    if not use_trailing_stop:
        common["atr_target_mult"] = 2.0

    oos_trades = None
    for split_name, calendar in [("IS", is_calendar), ("OOS", oos_calendar)]:
        trades = run_momentum_breakout_backtest(
            master_calendar=calendar, max_hold_days=hold_days, **common,
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


def run_multi_period_validation(universe, index_proxy_code, winning_signal_weights, winning_gate_kwargs,
                                 atr_stop_mult, trailing_atr_mult, use_trailing_stop, hold_days,
                                 starting_capital, extra_kwargs, execution_kwargs, chip_data, refresh):
    """把最終驗證出的組合，原封不動套到幾段跟近期多頭明顯不同市況的歷史期間，
    檢查PF/bootstrap正報酬比例是不是一到空頭/盤整就明顯轉差。每段期間都是獨立
    下載、獨立跑完整回測(不分IS/OOS，因為這裡的目的是「換一段市況」而不是
    「換一段沒看過的時間」)，所以只看單一次的PF+bootstrap，不能跟主回測的
    IS/OOS驗證力度直接類比。"""
    rows = []
    for label, start, end in EXTRA_REGIME_PERIODS:
        print(f"  下載 {label} ({start} ~ {end}) 的資料 ...", flush=True)
        period_price_data = load_price_data(universe, start, end, refresh=refresh)
        if index_proxy_code not in period_price_data:
            print(f"  ⚠️ {label}：大盤代理指標下載失敗，跳過這段期間\n")
            continue
        period_index_df = period_price_data[index_proxy_code]
        if len(period_index_df) < 80:
            print(f"  ⚠️ {label}：資料天數不足({len(period_index_df)}天)，跳過這段期間\n")
            continue

        period_indicators = precompute_all_breakout_indicators(
            period_price_data, universe, index_code=index_proxy_code, chip_data=chip_data,
        )
        period_regime = precompute_regime_series(period_index_df)

        # 注意：lots/risk_pct_per_trade由execution_kwargs提供(main()保證兩者恰好給一個)，
        # 這裡不能再額外寫死lots=2，否則跟execution_kwargs unpack會撞成重複關鍵字參數。
        trades = run_momentum_breakout_backtest(
            price_data=period_price_data, indicators_by_code=period_indicators, regime_series=period_regime,
            master_calendar=period_index_df.index, max_hold_days=hold_days, starting_capital=starting_capital,
            allow_short=True, atr_stop_mult=atr_stop_mult, atr_target_mult=2.0,
            signal_weights=winning_signal_weights, use_trailing_stop=use_trailing_stop,
            trailing_atr_mult=trailing_atr_mult if use_trailing_stop else None,
            **winning_gate_kwargs, **extra_kwargs, **execution_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        boot_results = bootstrap_resample_pnl(trades, n_resamples=1000, seed=42)
        boot_stats = summarize_bootstrap(boot_results)
        rows.append({
            "label": label, "start": start, "end": end,
            "trade_count": stats["trade_count"], "profit_factor": stats["profit_factor"],
            "total_pnl_ntd": stats["total_pnl_ntd"], "pct_positive": boot_stats["pct_positive"],
        })
        print(f"  {label} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"損益={stats['total_pnl_ntd']:,.0f}, bootstrap正報酬比例={boot_stats['pct_positive']:.1f}%\n",
              flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="右側順勢突破策略 - 訊號拆解 + 結構門檻 + 最終組合驗證")
    parser.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=1095)).isoformat())
    parser.add_argument("--end", default=datetime.date.today().isoformat())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--starting-capital", type=float, default=200_000)
    parser.add_argument("--max-stocks", type=int, default=0,
                         help="限制掃描股票數量(0=全部320檔，測試時可以設小一點加快速度)")
    parser.add_argument("--with-chip-confirm", action="store_true",
                         help="額外下載三大法人籌碼資料，把外資/投信買超比重訊號、以及「雙法人同步買超」"
                              "門檻變體也接上測試")
    parser.add_argument("--clean-ex-dividend", action="store_true",
                         help="額外下載除權息日期資料，把除權息當天的股票排除在候選之外，避免機械性跳空"
                              "污染量比/相對大盤強弱/突破前高這些訊號")
    parser.add_argument("--use-trailing-stop", action="store_true",
                         help="出場改用移動停利，不設固定停利目標價、不設固定持有天數(讓獲利奔跑)，"
                              "取代原本短線5天/中期15天的固定天數比較")
    parser.add_argument("--atr-stop-mult", type=float, default=1.0,
                         help="初始停損的ATR倍數(移動停利模式下會被ATR敏感度網格搜出的結果覆蓋)")
    parser.add_argument("--trailing-atr-mult", type=float, default=None,
                         help="移動停利的ATR倍數，不指定時跟--atr-stop-mult用同一個值(只在--use-trailing-stop時"
                              "有意義；會被ATR敏感度網格搜出的結果覆蓋)")
    parser.add_argument("--slippage-pct", type=float, default=0.0,
                         help="進場/停損出場的滑價假設(例如0.002代表0.2%%)，預設0(不模擬滑價)")
    parser.add_argument("--max-concurrent-positions", type=int, default=1,
                         help="最終驗證組合允許同時持有的最大部位數，預設1(維持原本一次一個部位)。"
                              ">1時允許同時持有多檔不同標的")
    parser.add_argument("--risk-pct-per-trade", type=float, default=None,
                         help="最終驗證組合改用風險預算式部位大小：每筆交易風險 = 帳戶權益 x 這個比例"
                              "(例如0.02代表2%%)。不指定時維持固定2口")
    parser.add_argument("--multi-period-test", action="store_true",
                         help="額外把最終驗證出的組合套到2022年修正段、2021下半年~2022年初盤整段"
                              "這兩段跟近期多頭明顯不同市況的歷史期間，檢查是不是只吃到牛市紅利")
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

    chip_data = None
    if args.with_chip_confirm:
        import chip_data_loader
        print("下載三大法人籌碼資料 ...")
        chip_data = chip_data_loader.load_chip_data(args.start, args.end, universe_codes=set(universe.keys()),
                                                       refresh=args.refresh)
        print(f"籌碼資料涵蓋 {len(chip_data)} 檔股票\n")
    else:
        print("未加 --with-chip-confirm，本次不測試外資/投信買超比重訊號、也不測「雙法人同步買超」門檻\n")

    ex_dividend_dates_by_code = None
    if args.clean_ex_dividend:
        import dividend_data_loader
        print("下載除權息日期資料 ...")
        ex_dividend_dates_by_code = dividend_data_loader.load_dividend_events(
            universe, args.start, args.end, refresh=args.refresh,
        )
        print(f"除權息資料涵蓋 {len(ex_dividend_dates_by_code)} 檔股票\n")
    else:
        print("未加 --clean-ex-dividend，除權息當天的機械性跳空不會被排除\n")

    print("預先計算全市場突破指標 ...")
    indicators_by_code = precompute_all_breakout_indicators(
        price_data, universe, index_code=INDEX_PROXY_CODE, chip_data=chip_data,
    )
    regime_series = precompute_regime_series(index_df)
    print(f"指標預計算完成，涵蓋 {len(indicators_by_code)} 檔股票\n")

    # extra_kwargs：訊號/門檻篩選階段也要套用的執行細節(除權息清洗、滑價)，
    # 這兩個是「回測寫不寫實」的問題，不是「部位大小怎麼配」的問題，篩選階段也該套用。
    extra_kwargs = dict(ex_dividend_dates_by_code=ex_dividend_dates_by_code, slippage_pct=args.slippage_pct)

    # execution_kwargs：只套用在最終驗證組合 + 跨週期驗證，不套用在訊號/門檻篩選階段
    # (見run_signal_ablation的docstring說明)。
    execution_kwargs = dict(max_concurrent_positions=args.max_concurrent_positions)
    if args.risk_pct_per_trade is not None:
        execution_kwargs["risk_pct_per_trade"] = args.risk_pct_per_trade
    else:
        execution_kwargs["lots"] = 2

    if args.use_trailing_stop:
        hold_days_options = TRAILING_STOP_HOLD_DAYS_OPTIONS
        print(f"⚠️ 移動停利模式：出場不設固定天數(工程安全上限={TRAILING_STOP_MAX_HOLD_DAYS}個交易日)、"
              f"不設固定停利目標價，初始停損/移動停利倍數會由階段2.5的ATR敏感度網格搜出\n")
    else:
        hold_days_options = HOLD_DAYS_OPTIONS

    if args.max_concurrent_positions > 1:
        print(f"⚠️ 多部位並行模式：最多同時持有 {args.max_concurrent_positions} 個部位\n")
    if args.risk_pct_per_trade is not None:
        print(f"⚠️ 風險預算式部位大小：每筆交易風險預算 = 帳戶權益 x {args.risk_pct_per_trade:.2%}\n")
    if args.slippage_pct > 0:
        print(f"⚠️ 滑價模擬：進場/停損出場套用 {args.slippage_pct:.2%} 滑價\n")

    all_ablation = {}
    all_gate_comparison = {}
    all_atr_grid = {}
    all_combo_results = {}  # {hold_label: {"對照組": {...}, "最佳組合": {...}}}
    all_multi_period = {}
    signal_reliable_flags = {}

    for hold_label, hold_days in hold_days_options:
        print(f"=== {hold_label} ===")

        print("[階段1] 單一訊號拆解 (只在IS內，基本門檻) ...")
        ablation_df, signal_names_used = run_signal_ablation(
            price_data, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, has_chip=args.with_chip_confirm, extra_kwargs=extra_kwargs,
        )
        all_ablation[hold_label] = ablation_df
        ablation_df.to_csv(os.path.join(RESULTS_DIR, f"ablation_{hold_label}.csv"),
                            index=False, encoding="utf-8-sig")

        print(f"\n[階段2] 結構門檻變體比較 (只在IS內，等權重訊號) ...")
        gate_df = run_gate_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, signal_names_used, has_chip=args.with_chip_confirm,
            extra_kwargs=extra_kwargs,
        )
        all_gate_comparison[hold_label] = gate_df
        gate_df.to_csv(os.path.join(RESULTS_DIR, f"gate_comparison_{hold_label}.csv"),
                        index=False, encoding="utf-8-sig")

        winning_signals, is_reliable = select_winning_signals(ablation_df)
        winning_gate_label, winning_gate_kwargs = select_winning_gate(gate_df)
        signal_reliable_flags[hold_label] = is_reliable
        winning_signal_weights = {name: 1.0 for name in winning_signals}
        print(f"\n  → 依IS結果選出的最佳組合：訊號={[SIGNAL_LABELS[s] for s in winning_signals]}"
              f"{'' if is_reliable else '(⚠️沒有訊號單獨PF>1，退回選總損益前3名，屬探索性選擇)'}，"
              f"門檻={winning_gate_label}")

        atr_stop_mult, trailing_atr_mult = args.atr_stop_mult, args.trailing_atr_mult
        if args.use_trailing_stop:
            print(f"\n[階段2.5] ATR倍數敏感度網格(只在IS內，用上面選出的最佳訊號/門檻組合) ...")
            atr_grid_df, (atr_stop_mult, trailing_atr_mult) = run_atr_sensitivity_grid(
                price_data, indicators_by_code, regime_series, is_calendar,
                args.starting_capital, winning_signal_weights, winning_gate_kwargs, extra_kwargs,
            )
            all_atr_grid[hold_label] = atr_grid_df
            print(f"  → 選出：初始停損x{atr_stop_mult}, 移動停利x{trailing_atr_mult}")

        print(f"\n[階段3] 對照組 vs 最佳組合，IS vs OOS + bootstrap穩健性檢查 ...")
        naive_weights = {name: 1.0 for name in signal_names_used}
        naive_result = evaluate_combo(
            "對照組(全部訊號等權重+基本門檻)", price_data, indicators_by_code, regime_series,
            is_calendar, oos_calendar, args.starting_capital, hold_days, naive_weights, {},
            args.atr_stop_mult, args.trailing_atr_mult, args.use_trailing_stop, extra_kwargs,
            {"lots": 2, "max_concurrent_positions": 1},
        )
        winning_result = evaluate_combo(
            f"最佳組合(訊號={winning_gate_label})", price_data, indicators_by_code, regime_series,
            is_calendar, oos_calendar, args.starting_capital, hold_days, winning_signal_weights,
            winning_gate_kwargs, atr_stop_mult, trailing_atr_mult, args.use_trailing_stop,
            extra_kwargs, execution_kwargs,
        )
        all_combo_results[hold_label] = {"對照組": naive_result, "最佳組合": winning_result}

        for result in (naive_result, winning_result):
            print(f"  [{result['label']}]")
            print(f"    IS  -> {result['IS']['trade_count']}筆, PF={_fmt_pf(result['IS']['profit_factor'])}, "
                  f"損益={result['IS']['total_pnl_ntd']:,.0f}")
            print(f"    OOS -> {result['OOS']['trade_count']}筆, PF={_fmt_pf(result['OOS']['profit_factor'])}, "
                  f"損益={result['OOS']['total_pnl_ntd']:,.0f}")
            b = result["bootstrap"]
            print(f"    bootstrap：正報酬比例={b['pct_positive']:.1f}%, p值={b['p_value']:.3f}, "
                  f"拿掉最大3筆後損益={b['pnl_excluding_top3_ntd']:,.0f}")
            pd.DataFrame(result["oos_trades"]).to_csv(
                os.path.join(RESULTS_DIR, f"trades_OOS_{hold_label}_{result['label'][:10]}.csv"),
                index=False, encoding="utf-8-sig",
            )
        print()

        if args.multi_period_test:
            print(f"[跨市場週期驗證] 把最佳組合套到跟近期多頭明顯不同市況的歷史期間 ...")
            all_multi_period[hold_label] = run_multi_period_validation(
                universe, INDEX_PROXY_CODE, winning_signal_weights, winning_gate_kwargs,
                atr_stop_mult, trailing_atr_mult, args.use_trailing_stop, hold_days,
                args.starting_capital, extra_kwargs, execution_kwargs, chip_data, args.refresh,
            )

    summary_lines = [
        "=" * 100,
        f"右側順勢突破策略 訊號拆解 + 結構門檻 + 最終組合驗證",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        f"籌碼相關訊號/門檻：{'有測試' if args.with_chip_confirm else '未測試'}　"
        f"除權息清洗：{'有' if args.clean_ex_dividend else '沒有'}　"
        f"出場模式：{'移動停利' if args.use_trailing_stop else '固定天數+固定ATR停利'}　"
        f"滑價：{args.slippage_pct:.2%}　最大並行部位：{args.max_concurrent_positions}　"
        f"部位大小：{'風險預算式(' + f'{args.risk_pct_per_trade:.2%}' + ')' if args.risk_pct_per_trade else '固定2口'}",
        "=" * 100,
    ]
    for hold_label, _ in hold_days_options:
        summary_lines.append(f"\n--- {hold_label} / 單一訊號拆解(IS，基本門檻) ---")
        ablation_df = all_ablation[hold_label]
        reliable = ablation_df[ablation_df["trade_count"] >= MIN_TRADES_FOR_RANKING]
        if reliable.empty:
            summary_lines.append(f"⚠️ 沒有任何訊號的交易筆數 >= {MIN_TRADES_FOR_RANKING}，統計上都不夠可靠，全部原樣列出：")
            reliable = ablation_df
        reliable = reliable.sort_values("total_pnl_ntd", ascending=False)
        header = f"{'訊號':<24}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}{'最大連虧':>8}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        for _, r in reliable.iterrows():
            summary_lines.append(
                f"{r['label']:<24}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}{r['win_rate']:>8.1f}"
                f"{r['total_pnl_ntd']:>14,.0f}{r['max_consecutive_losses']:>8}"
            )
        if not signal_reliable_flags[hold_label]:
            summary_lines.append("⚠️ 沒有任何訊號單獨PF>1，下面的「最佳組合」是探索性選擇(總損益前3名)，不是已驗證過的訊號")

        summary_lines.append(f"\n--- {hold_label} / 結構門檻變體比較(IS，等權重訊號) ---")
        gate_df = all_gate_comparison[hold_label].sort_values("total_pnl_ntd", ascending=False)
        header2 = f"{'門檻':<24}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}"
        summary_lines.append(header2)
        summary_lines.append("-" * len(header2))
        for _, r in gate_df.iterrows():
            summary_lines.append(
                f"{r['gate']:<24}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}{r['win_rate']:>8.1f}"
                f"{r['total_pnl_ntd']:>14,.0f}"
            )

        if hold_label in all_atr_grid:
            summary_lines.append(f"\n--- {hold_label} / ATR倍數敏感度網格(IS，最佳訊號/門檻組合) ---")
            atr_df = all_atr_grid[hold_label].sort_values("total_pnl_ntd", ascending=False)
            header3 = f"{'初始停損x':>10}{'移動停利x':>10}{'交易數':>8}{'PF':>8}{'總損益NT$':>14}"
            summary_lines.append(header3)
            summary_lines.append("-" * len(header3))
            for _, r in atr_df.iterrows():
                summary_lines.append(
                    f"{r['atr_stop_mult']:>10.2f}{r['trailing_atr_mult']:>10.2f}{r['trade_count']:>8}"
                    f"{_fmt_pf(r['profit_factor']):>8}{r['total_pnl_ntd']:>14,.0f}"
                )

        summary_lines.append(f"\n--- {hold_label} / 對照組 vs 最佳組合：IS vs OOS + bootstrap ---")
        for combo_key, result in all_combo_results[hold_label].items():
            summary_lines.append(f"\n[{result['label']}]")
            for split_name in ["IS", "OOS"]:
                stats = result[split_name]
                split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
                summary_lines.append(
                    f"  {split_full}: {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
                    f"勝率={stats['win_rate']:.1f}%, 總損益NT${stats['total_pnl_ntd']:,.0f}, "
                    f"最大回撤NT${stats['max_drawdown_ntd']:,.0f}, 最大連續虧損{stats['max_consecutive_losses']}筆"
                )
            b = result["bootstrap"]
            summary_lines.append(
                f"  [穩健性] OOS bootstrap 1000次重抽樣：平均總損益NT${b['mean']:,.0f}，"
                f"5%~95%區間=[NT${b['p5']:,.0f}, NT${b['p95']:,.0f}]，"
                f"正報酬比例={b['pct_positive']:.1f}%，p值={b['p_value']:.3f}，"
                f"拿掉最大3筆交易後總損益NT${b['pnl_excluding_top3_ntd']:,.0f}"
            )

        if hold_label in all_multi_period:
            summary_lines.append(f"\n--- {hold_label} / 跨市場週期驗證(最佳組合套到不同市況的歷史期間) ---")
            for row in all_multi_period[hold_label]:
                summary_lines.append(
                    f"  {row['label']}({row['start']}~{row['end']}): {row['trade_count']}筆, "
                    f"PF={_fmt_pf(row['profit_factor'])}, 損益NT${row['total_pnl_ntd']:,.0f}, "
                    f"bootstrap正報酬比例={row['pct_positive']:.1f}%"
                )
            if not all_multi_period[hold_label]:
                summary_lines.append("  (所有額外期間都下載失敗或資料不足，無法驗證)")

    summary_lines.append(
        "\n判讀方式：先看單一訊號拆解，PF明顯>1且交易筆數夠多的訊號才代表真的有預測力；"
        "「最佳組合」是程式依這個結果自動選出來的，不是隨便挑的——如果標示「探索性選擇」，"
        "代表這次測試裡沒有任何訊號真正驗證過，結果僅供參考。再看OOS的PF、bootstrap正報酬比例、"
        "p值(越接近0代表優勢在不同交易順序下都站得住，超過0.2以上不該當作已驗證的優勢)。"
        "如果有跑跨市場週期驗證，最後看PF/bootstrap正報酬比例是不是一到2022年修正段/盤整段就"
        "明顯轉差——如果是，代表這組優勢比較接近「牛市beta」，不是在任何市況都成立的右側交易優勢。"
    )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


if __name__ == "__main__":
    main()
