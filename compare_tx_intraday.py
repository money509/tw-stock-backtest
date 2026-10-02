"""
compare_tx_intraday.py
====================
台指期(TX)15分鐘K棒 右側突破策略 —— 訊號拆解 + 自動組合選擇 + 簡易IS/OOS。

⚠️⚠️ 這支腳本存在的前提就是「樣本量極小」，不是意外或事後發現的限制：資料來源
taifex_intraday_loader.py只提供TAIFEX免費的「前30個交易日」逐筆成交資料，換算成
15分鐘K棒、只算日盤，整個可用樣本大約是20~22個交易日 x 20根/天 ≈ 400~450根K棒。
這裡沿用momentum_breakout_engine.py/compare_breakout.py同一套「單一訊號拆解→
自動篩選PF>1的訊號→組合成策略」方法論，套用在tx_intraday_engine.py(見該檔案模組
docstring的完整架構說明)，但**在這麼小的樣本上跑這整套方法論，本質上就是在對雜訊
做多重比較**——這不是這支腳本的bug，是使用者在明確理解這個風險之後，要求「照原計畫
跑全套單一訊號+自動組合，但報告裡強調樣本太小不能信」的刻意選擇。所以下面每一個
輸出區塊(console print跟summary.txt)都會重複標示這個警告，不是只在檔案開頭寫一次
就算了事——PF/勝率這類數字在這個樣本數下極不穩定，一兩筆交易就能整組翻轉，
**不能**跟compare_breakout.py/compare_equity_swing.py裡經過IS/OOS+多年資料驗證過的
「可靠組合」相提並論。

這支腳本刻意**不做**bootstrap、**不做**walk-forward——那兩種驗證方式的前提是
「有足夠多獨立的區塊/重抽樣才有意義」，在幾百根K棒、可能只有個位數到十幾筆交易的
樣本上跑，只是把假象包裝得更像一回事，不會產生任何額外的可信度，等於統計劇場，
所以這裡刻意不建。唯一保留的驗證形式是簡單的IS/OOS切分(見split_is_oos_bars())，
只是為了跟這個repo其餘引擎的方法論結構保持一致，OOS那一段的交易筆數可能只有
個位數，這裡明確標示為「僅供參考，完全不能當結論」，不是真正意義上的樣本外驗證。

三階段結構：
  階段1 單一訊號拆解(ablation)：TX_INTRADAY_SIGNAL_NAMES每個訊號單獨開啟(權重1.0，
    其餘0)，在IS區間跑一次，報告交易筆數/PF/勝率/總損益(同compare_breakout.py的
    run_signal_ablation()表格形狀)。
  階段2 自動組合：挑出「交易筆數 >= MIN_TRADES_FOR_TX_RANKING 且 PF>1」的訊號，
    等權重組成最終組合；沒有任何訊號同時滿足兩個條件時，退回選總損益前3名，
    並標記為「探索性選擇」(reliable=False)，同compare_breakout.py
    select_winning_signals()的邏輯，只是筆數門檻大幅降低(見MIN_TRADES_FOR_TX_RANKING
    常數說明，因為30筆的門檻在這個樣本數下永遠不會有任何訊號達標)。
  階段3 簡易IS/OOS：把階段2選出的組合，分別在IS/OOS各跑一次，兩者放在一起比較。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from taifex_intraday_loader import load_tx_bars, DAY_SESSION_START, DAY_SESSION_END
from tx_intraday_engine import (
    precompute_tx_intraday_indicators, run_tx_intraday_backtest, TX_INTRADAY_SIGNAL_NAMES,
    ROLLING_PCT_WINDOW_BARS_DEFAULT, BREAKOUT_WINDOW_BARS_DEFAULT, ATR_PERIOD_BARS_DEFAULT,
    MAX_HOLD_BARS_DEFAULT, SCORE_ENTRY_THRESHOLD_DEFAULT,
)
from mean_reversion_engine import summarize_mr
from squeeze_kdj_tx_hourly_preview import (
    CONTRACT_MULTIPLIER, DEFAULT_COMMISSION_PER_SIDE, DEFAULT_EXCHANGE_FEE_PER_SIDE,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_tx_intraday")

SIGNAL_LABELS = {
    "score_macd": "MACD柱狀圖",
    "score_candle_body": "K棒實體比例",
    "score_rsi_cross": "RSI偏離50",
    "score_volume_ratio": "量比(前一交易日均量)",
    "score_macd_divergence": "MACD背離(近似實作)",
    "score_squeeze_kdj": "布林+Keltner擠壓+KDJ(僅多方，見squeeze_kdj_signal.py)",
}

# 交易筆數門檻：compare_breakout.py日線版本用30筆(MIN_TRADES_FOR_RANKING)，但這裡
# 整個IS區間的總交易數可能都湊不到30筆，用日線的門檻會讓「自動篩選」永遠找不到任何
# 一個訊號，退化成每次都用探索性後備選項——那樣這個門檻形同虛設。這裡大幅降低到3筆，
# 誠實承認這已經不是「統計上有意義的最低門檻」，只是「至少不是單一極端交易撐出來的
# PF」這麼低的標準，見下面select_winning_tx_signals()的說明。
MIN_TRADES_FOR_TX_RANKING = 3

DISCLAIMER_HEADER = "⚠️⚠️ 台指期15分鐘K棒右側突破 — 小樣本訊號拆解+自動組合(不是驗證) ⚠️⚠️"


def _caveat_lines(n_bars: int, n_days_est: int) -> list:
    return [
        f"⚠️⚠️ 樣本極小警告：這批資料大約只有{n_days_est}個交易日、{n_bars}根15分鐘K棒",
        "(TAIFEX免費「前30個交易日」滾動窗口的先天限制，見taifex_intraday_loader.py模組docstring)。",
        "以下跑的是跟momentum_breakout_engine.py/compare_breakout.py同一套「單一訊號拆解→",
        "自動篩選PF>1的訊號→組合成策略」方法論，但樣本量遠低於能建立信心的最低門檻——",
        "這裡看到的PF/勝率/總損益數字，有很高機率只是這幾百根K棒剛好對到雜訊或運氣，",
        "不是真的有預測力的訊號，請不要拿來當作可以信賴的交易依據，最多只能當作",
        "「哪個訊號的方向看起來至少沒有反著來」的粗略觀察，不是任何形式的策略驗證。",
    ]


def _fmt_pf(pf: float) -> str:
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def _stats_row_text(label: str, stats: dict) -> str:
    if stats["trade_count"] == 0:
        return f"  {label} -> 0筆交易，無法計算PF"
    return (f"  {label} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
            f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}")


def split_is_oos_bars(bars: pd.DataFrame, indicators: pd.DataFrame, is_ratio: float = 0.7):
    """
    簡易IS/OOS切分：indicators在切分之前就用完整序列算好(rolling_percentile_score等
    都是只看過去、不看未來的因果式計算，見tx_intraday_engine.py模組docstring)，
    這裡只是把「哪些bar算IS、哪些算OOS」切開，OOS那段的指標值仍然正確地看得到
    切分點之前的真實歷史(這才貼近實際交易時的狀態：走到OOS的第一根K棒時，
    IS那幾百根K棒的歷史本來就已經存在，不是要OOS假裝自己沒有過去)。
    """
    n = len(bars)
    split_point = int(n * is_ratio)
    is_bars, oos_bars = bars.iloc[:split_point], bars.iloc[split_point:]
    is_ind, oos_ind = indicators.iloc[:split_point], indicators.iloc[split_point:]
    return is_bars, is_ind, oos_bars, oos_ind


def select_winning_tx_signals(ablation_df: pd.DataFrame, min_trades: int = MIN_TRADES_FOR_TX_RANKING):
    """
    跟compare_breakout.select_winning_signals()同樣的邏輯(挑「交易筆數夠多且PF>1」的
    訊號，找不到就退回總損益前3名、標記reliable=False)，唯一差異是筆數門檻(見
    MIN_TRADES_FOR_TX_RANKING常數說明)。回傳(訊號名稱list, reliable: bool)。
    """
    non_baseline = ablation_df[ablation_df["signal"] != "__baseline_equal_weight__"]
    non_zero = non_baseline[non_baseline["trade_count"] > 0]
    pool = non_zero if not non_zero.empty else non_baseline
    reliable_pool = pool[pool["trade_count"] >= min_trades]
    picked = reliable_pool[reliable_pool["profit_factor"] > 1.0]
    if picked.empty:
        fallback = pool.sort_values("total_pnl_ntd", ascending=False).head(3)
        return list(fallback["signal"]), False
    return list(picked["signal"]), True


def run_signal_ablation(is_bars, is_indicators, common_kwargs, starting_capital):
    rows = []
    print("階段1：單一訊號拆解(IS區間)", flush=True)
    for i, name in enumerate(TX_INTRADAY_SIGNAL_NAMES, start=1):
        trades = run_tx_intraday_backtest(is_bars, is_indicators, signal_weights={name: 1.0}, **common_kwargs)
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "signal": name, "label": SIGNAL_LABELS[name],
            "trade_count": stats["trade_count"], "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"], "total_pnl_ntd": stats["total_pnl_ntd"],
        })
        print(f"  [{i}/{len(TX_INTRADAY_SIGNAL_NAMES)}] " + _stats_row_text(SIGNAL_LABELS[name], stats).strip(), flush=True)

    baseline_weights = {name: 1.0 for name in TX_INTRADAY_SIGNAL_NAMES}
    baseline_trades = run_tx_intraday_backtest(is_bars, is_indicators, signal_weights=baseline_weights, **common_kwargs)
    baseline_stats = summarize_mr(baseline_trades, starting_capital)
    rows.append({
        "signal": "__baseline_equal_weight__", "label": f"基準({len(TX_INTRADAY_SIGNAL_NAMES)}訊號等權重)",
        "trade_count": baseline_stats["trade_count"], "win_rate": baseline_stats["win_rate"],
        "profit_factor": baseline_stats["profit_factor"], "total_pnl_ntd": baseline_stats["total_pnl_ntd"],
    })
    print("  [基準] " + _stats_row_text(f"{len(TX_INTRADAY_SIGNAL_NAMES)}訊號等權重", baseline_stats).strip(), flush=True)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=["big", "mini"], default="big",
                         help="大台(big，預設，每點NT$200)或小台(mini，每點NT$50)")
    parser.add_argument("--lots", type=int, default=1)
    parser.add_argument("--commission-per-side", type=float, default=DEFAULT_COMMISSION_PER_SIDE)
    parser.add_argument("--exchange-fee-per-side", type=float, default=DEFAULT_EXCHANGE_FEE_PER_SIDE)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--commodity-id", default="TXF")
    parser.add_argument("--starting-capital", type=float, default=200000.0)
    parser.add_argument("--bar-minutes", type=int, default=15,
                         help="K棒週期(分鐘)，預設15(這支腳本的設計目標)")
    parser.add_argument("--rolling-window-bars", type=int, default=ROLLING_PCT_WINDOW_BARS_DEFAULT,
                         help="rolling_percentile_score()用的自身歷史回看根數，見tx_intraday_engine.py說明")
    parser.add_argument("--breakout-window-bars", type=int, default=BREAKOUT_WINDOW_BARS_DEFAULT)
    parser.add_argument("--score-entry-threshold", type=float, default=SCORE_ENTRY_THRESHOLD_DEFAULT)
    parser.add_argument("--max-hold-bars", type=int, default=MAX_HOLD_BARS_DEFAULT)
    parser.add_argument("--atr-stop-mult", type=float, default=1.0)
    parser.add_argument("--atr-target-mult", type=float, default=2.0)
    parser.add_argument("--use-trailing-stop", action="store_true")
    parser.add_argument("--trailing-atr-mult", type=float, default=None)
    parser.add_argument("--no-short", action="store_true", help="關閉空方，只做多方")
    parser.add_argument("--is-ratio", type=float, default=0.7, help="IS佔整體樣本的比例，其餘當OOS")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    trades_path = os.path.join(RESULTS_DIR, "trades.csv")

    print(DISCLAIMER_HEADER, flush=True)
    bars, diag = load_tx_bars(commodity_id=args.commodity_id, refresh=args.refresh_cache, bar_minutes=args.bar_minutes)

    if bars is None:
        lines = [
            DISCLAIMER_HEADER,
            "資料下載失敗，看log診斷。",
            f"失敗原因(reason)：{diag.get('reason')}",
            f"是否使用快取：{diag.get('used_cache')}",
            "",
            "taifex_intraday_loader.py的下載機制是防禦性猜測，尚未經真實TAIFEX端點驗證",
            "(見該模組docstring)，失敗屬於預期中可能發生的情況，不代表程式設計有錯。",
        ]
        text = "\n".join(lines)
        print(text, flush=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("資料下載失敗，看log診斷。", file=sys.stderr, flush=True)
        sys.exit(0)  # 優雅結束，不是crash：資料抓不到是預期中可能發生的情況

    n_bars = len(bars)
    session_minutes = (13 * 60 + 45) - (8 * 60 + 45)  # DAY_SESSION_START~DAY_SESSION_END, 見taifex_intraday_loader.py
    bars_per_day = max(1, round(session_minutes / args.bar_minutes))
    n_days_est = max(1, round(n_bars / bars_per_day))

    print(f"成功取得日盤{args.bar_minutes}分鐘K棒：{n_bars}根(約{n_days_est}個交易日)，"
          f"近月合約={diag.get('front_month')}，過濾掉非日盤ticks={diag.get('n_night_session_dropped')}筆，"
          f"使用快取={diag.get('used_cache')}", flush=True)

    for line in _caveat_lines(n_bars, n_days_est):
        print(line, flush=True)

    indicators = precompute_tx_intraday_indicators(
        bars, rolling_window_bars=args.rolling_window_bars, breakout_window_bars=args.breakout_window_bars,
    )
    is_bars, is_ind, oos_bars, oos_ind = split_is_oos_bars(bars, indicators, is_ratio=args.is_ratio)
    print(f"IS/OOS切分：IS {len(is_bars)}根 / OOS {len(oos_bars)}根(is_ratio={args.is_ratio})", flush=True)

    multiplier = CONTRACT_MULTIPLIER[args.contract]
    common_kwargs = dict(
        max_hold_bars=args.max_hold_bars, allow_short=not args.no_short, lots=args.lots,
        atr_stop_mult=args.atr_stop_mult, atr_target_mult=args.atr_target_mult,
        use_trailing_stop=args.use_trailing_stop, trailing_atr_mult=args.trailing_atr_mult,
        score_entry_threshold=args.score_entry_threshold,
        contract_multiplier=multiplier, commission_per_side=args.commission_per_side,
        exchange_fee_per_side=args.exchange_fee_per_side,
    )

    summary_lines = [
        DISCLAIMER_HEADER,
        f"資料範圍：{bars.index.min()} ~ {bars.index.max()}(TX近月合約={diag.get('front_month')}，"
        f"僅日盤，共{n_bars}根{args.bar_minutes}分鐘K棒，約{n_days_est}個交易日)",
        f"契約：{'大台(TX)' if args.contract == 'big' else '小台(MTX)'}，每點NT${multiplier}，{args.lots}口，"
        f"成本假設：手續費NT${args.commission_per_side:.0f}/口/邊 + "
        f"期交所規費(含期貨交易稅粗略近似)NT${args.exchange_fee_per_side:.0f}/口/邊"
        f"(跟squeeze_kdj_tx_hourly_preview.py共用同一組假設，未逐一查證最新公告費率)",
        f"rolling_percentile_score窗口={args.rolling_window_bars}根，突破窗口={args.breakout_window_bars}根，"
        f"進場分數門檻={args.score_entry_threshold}，最長持有={args.max_hold_bars}根(見tx_intraday_engine.py說明)",
        f"IS/OOS切分：IS {len(is_bars)}根 / OOS {len(oos_bars)}根(is_ratio={args.is_ratio})",
        "",
    ] + _caveat_lines(n_bars, n_days_est) + [""]

    # ---- 階段1：單一訊號拆解(IS) ----
    ablation_df = run_signal_ablation(is_bars, is_ind, common_kwargs, args.starting_capital)
    summary_lines.append("[階段1] 單一訊號拆解(IS區間)：")
    for _, r in ablation_df.iterrows():
        fake_stats = {"trade_count": r["trade_count"], "profit_factor": r["profit_factor"],
                       "win_rate": r["win_rate"], "total_pnl_ntd": r["total_pnl_ntd"]}
        summary_lines.append(_stats_row_text(r["label"], fake_stats))
    summary_lines += ["", "↑ " + _caveat_lines(n_bars, n_days_est)[0], ""]

    # ---- 階段2：自動組合 ----
    winning_signals, reliable = select_winning_tx_signals(ablation_df)
    combo_weights = {name: 1.0 for name in (winning_signals or TX_INTRADAY_SIGNAL_NAMES)}
    combo_label = ("自動篩選(PF>1且交易筆數>=%d)：%s" % (MIN_TRADES_FOR_TX_RANKING,
                   "、".join(SIGNAL_LABELS[n] for n in winning_signals)) if reliable else
                   "探索性選擇(沒有任何訊號單獨PF>1，改選總損益前3名)：%s" %
                   "、".join(SIGNAL_LABELS.get(n, n) for n in winning_signals))
    print(f"階段2：{combo_label}", flush=True)

    is_combo_trades = run_tx_intraday_backtest(is_bars, is_ind, signal_weights=combo_weights, **common_kwargs)
    is_combo_stats = summarize_mr(is_combo_trades, args.starting_capital)
    print("  " + _stats_row_text("最終組合(IS)", is_combo_stats).strip(), flush=True)

    summary_lines += [
        f"[階段2] 自動組合選擇：{combo_label}",
        f"{'' if reliable else '⚠️ 沒有任何訊號單獨PF>1，以下是探索性選擇，不是已驗證過的訊號組合'}",
        _stats_row_text("最終組合(IS)", is_combo_stats),
        "",
        "↑ " + _caveat_lines(n_bars, n_days_est)[0],
        "",
    ]

    # ---- 階段3：簡易IS/OOS(illustrative only) ----
    oos_combo_trades = run_tx_intraday_backtest(oos_bars, oos_ind, signal_weights=combo_weights, **common_kwargs)
    oos_combo_stats = summarize_mr(oos_combo_trades, args.starting_capital)
    print(f"階段3：簡易IS/OOS(⚠️ OOS筆數可能個位數，純示意，不是可信的樣本外驗證)", flush=True)
    print("  " + _stats_row_text("最終組合(IS)", is_combo_stats).strip(), flush=True)
    print("  " + _stats_row_text("最終組合(OOS)", oos_combo_stats).strip(), flush=True)

    summary_lines += [
        "[階段3] 簡易IS/OOS切分(⚠️ 這不是嚴謹的樣本外驗證，OOS段交易筆數可能只有個位數，",
        "一兩筆交易就能讓PF/勝率整組翻轉，這裡只是為了跟repo其餘引擎的方法論結構一致才保留，",
        "結果僅供參考，不能當作任何形式的結論，更不能拿來決定要不要用真錢下單)：",
        _stats_row_text("最終組合(IS)", is_combo_stats),
        _stats_row_text("最終組合(OOS)", oos_combo_stats),
        "",
    ]

    summary_lines += _caveat_lines(n_bars, n_days_est)
    summary_lines += [
        "",
        "沒有跑bootstrap、沒有跑walk-forward：在這個樣本數下，那兩種驗證方式只會把假象",
        "包裝得更像一回事，不會產生任何額外的可信度，這裡刻意不做(見本檔案模組docstring)。",
    ]

    text = "\n".join(str(x) for x in summary_lines)
    print(text, flush=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    all_trades_rows = []
    for t in is_combo_trades:
        all_trades_rows.append({"phase": "IS", **t})
    for t in oos_combo_trades:
        all_trades_rows.append({"phase": "OOS", **t})
    trades_df = pd.DataFrame(all_trades_rows)
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    print(f"已寫入 {summary_path} 與 {trades_path}", flush=True)


if __name__ == "__main__":
    main()
