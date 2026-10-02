"""
squeeze_kdj_tx_hourly_preview.py
====================

台指期(TX/大台指期貨) 1小時K棒版「布林+Keltner擠壓+KDJ」訊號 —— **一次性預覽腳本，
不是驗證流程**。

為什麼要有這支腳本：squeeze_kdj_signal.py這個訊號的原始構想來自一支「加密貨幣1小時
K線」教學影片，目前既有的兩個引擎(compare_breakout.py/compare_equity_swing.py)都是把
它套用在台股**日線**資料上，跟原始描述的時間週期(1小時)有落差。這支腳本改用TX期貨
自己的1小時K棒，是對原始構想更貼近的還原，但代價是資料來源：TAIFEX免費的逐筆成交
資料只提供「最近30個交易日」的滾動窗口(taifex_intraday_loader.py)，換算成1小時K棒
大約只有 30天 x 5根/天 = 150根K棒上下(確切數字視實際交易日數而定，遠少於使用者原本
提到的「約450-500根」估計，因為這裡只算日盤、不含夜盤，見taifex_intraday_loader.py
模組docstring)，這個樣本數完全不足以做IS/OOS切分或bootstrap統計檢定，任何這類統計
包裝在這麼小的樣本上都是假象，這支腳本刻意**不做**那些，只印出訊號在這批真實TX
1小時K棒上實際觸發了幾次、長什麼樣子，讓使用者對訊號行為有個直覺，僅此而已。

⚠️⚠️ 「預覽/小樣本，不是驗證」：本腳本所有輸出(summary.txt、終端機print、trades.csv)
都會重複標示這句話，PF/勝率這類數字**不能**跟compare_breakout.py/
compare_equity_swing.py裡經過IS/OOS+bootstrap驗證過的「可靠組合」相提並論——那些
數字背後有數百筆交易、跨越數年不同市況；這裡可能只有個位數到十幾筆交易，一兩筆
極端結果就能把PF/勝率完全翻轉，只能當「訊號有沒有動起來、方向感覺對不對」的粗略
觀察，不是任何形式的策略驗證。

資料來源信心：taifex_intraday_loader.py是這個repo目前信心程度最低的資料來源(見該
模組docstring)，下載/解析都可能失敗——這支腳本把失敗視為**預期中可能發生的正常情況**，
資料抓不到就印出明確診斷訊息、乾淨結束(exit code 0)，不會丟出原始traceback。

契約規格假設(誠實列出，未逐一查證TAIFEX最新公告，見下方常數定義)：
  大台指期貨(TX)：每點新台幣200元(--contract big，預設)
  小型台指期貨(MTX/小台)：每點新台幣50元(--contract mini)
成本假設(誠實列出，屬於「合理但未逐一查證」的估計值，不是查證過的最新公告費率)：
  手續費：每口每邊新台幣約60元(零售常見範圍40~100元，這裡取中間值當預設，
    --commission-per-side覆寫)
  期交所交易稅/手續費：每口每邊新台幣約20元(--exchange-fee-per-side覆寫)
  期貨本身**沒有**證券交易稅(那是股票交易才有的稅目，期貨課的是「期貨交易稅」，
    這裡的exchange-fee-per-side概念上把期交所規費+期貨交易稅一併粗略估進去，
    不是精確拆分兩者，因為兩者合計金額相對整體損益影響很小，這個近似對這種
    「小樣本預覽」的精確度需求來說足夠)。
一律用同一批交易跑變體A(原規則)跟變體B(ATR框架，atr_period=14/atr_stop_mult=1.0/
atr_target_mult=2.0，取跟compare_breakout.py一致的既有預設值，不是重新調過的參數)。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from taifex_intraday_loader import load_tx_hourly_bars
from squeeze_kdj_signal import (
    compute_squeeze_kdj_features, simulate_variant_a_trades, simulate_variant_b_trades,
)
from mean_reversion_engine import summarize_mr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_tx_hourly_preview")

CONTRACT_MULTIPLIER = {"big": 200, "mini": 50}  # 新台幣/點，見模組docstring
DEFAULT_COMMISSION_PER_SIDE = 60.0   # 新台幣，零售常見範圍40~100元的中間值，未逐一查證
DEFAULT_EXCHANGE_FEE_PER_SIDE = 20.0  # 新台幣，粗略估計，未逐一查證最新公告費率

PREVIEW_DISCLAIMER = "⚠️ 預覽/小樣本，不是驗證(PREVIEW ONLY, NOT A VALIDATION) ⚠️"


def _cost_trades(raw_trades: list, multiplier: int, commission_per_side: float,
                  exchange_fee_per_side: float, lots: int = 1) -> list:
    """把simulate_variant_*_trades()算出來的原始交易，套上TX契約乘數+成本假設，
    轉成summarize_mr()看得懂的格式。跟compare_breakout.py的
    _cost_squeeze_kdj_trades()同樣精神，這裡的乘數/成本是TX期貨自己的假設
    (見模組docstring)，不是沿用股票期貨那組COMMISSION_PER_LOT_PER_LEG。"""
    out = []
    round_trip_cost = (commission_per_side + exchange_fee_per_side) * 2 * lots
    for t in raw_trades:
        e_price, exit_price = t["entry_price"], t["exit_price"]
        price_pnl = (exit_price - e_price) * multiplier * lots
        pnl_ntd = price_pnl - round_trip_cost
        notional = e_price * multiplier * lots
        return_pct = pnl_ntd / notional if notional > 0 else 0.0
        out.append({
            "entry_date": t["entry_date"], "exit_date": t["exit_date"],
            "e_price": e_price, "exit_price": exit_price, "exit_reason": t["exit_reason"],
            "lots": lots, "pnl_ntd": pnl_ntd, "return_pct": return_pct,
            # simulate_variant_*_trades()回傳的"hold_days"其實是「K棒根數」，這裡
            # 用的是1小時K棒，所以就是持有小時數，不是持有天數(見squeeze_kdj_signal.py
            # 模組docstring：這兩個函式對K棒週期是通用的，欄位名稱沿用原本日線
            # 語境下取的"hold_days"，語意上這裡等於"hold_hours")。
            "hold_days": t["hold_days"],
        })
    return out


def _print_and_write_summary(lines: list, path: str):
    text = "\n".join(lines)
    print(text, flush=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def _stats_block(label: str, trades: list, starting_capital: float) -> list:
    stats = summarize_mr(trades, starting_capital)
    pf = stats["profit_factor"]
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    lines = [
        f"[{label}]",
        f"  交易筆數：{stats['trade_count']}",
        f"  Profit Factor：{pf_str}",
        f"  勝率：{stats['win_rate']:.1f}%",
        f"  總損益：NT${stats['total_pnl_ntd']:,.0f}",
        f"  平均持有：{stats['avg_hold_days']:.1f} 小時(1小時K棒，見程式註解)",
        f"  ⚠️ 交易筆數{'過少，' if stats['trade_count'] < 20 else ''}以上數字僅供觀察訊號行為，"
        f"不是統計上可靠的驗證結果",
    ]
    return lines, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["a", "b", "both"], default="both",
                         help="要跑哪個出場變體，預設both(兩個都跑)")
    parser.add_argument("--contract", choices=["big", "mini"], default="big",
                         help="大台(big，預設，每點NT$200)或小台(mini，每點NT$50)")
    parser.add_argument("--lots", type=int, default=1, help="口數，預設1口")
    parser.add_argument("--commission-per-side", type=float, default=DEFAULT_COMMISSION_PER_SIDE)
    parser.add_argument("--exchange-fee-per-side", type=float, default=DEFAULT_EXCHANGE_FEE_PER_SIDE)
    parser.add_argument("--refresh-cache", action="store_true",
                         help="忽略今天已有的快取，重新向TAIFEX下載")
    parser.add_argument("--commodity-id", default="TXF",
                         help="TAIFEX商品代號猜測值，預設TXF(未經驗證，見taifex_intraday_loader.py)")
    parser.add_argument("--starting-capital", type=float, default=200000.0,
                         help="只用來算return_pct/回撤，不是這支腳本的重點統計量")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    trades_path = os.path.join(RESULTS_DIR, "trades.csv")

    print(PREVIEW_DISCLAIMER, flush=True)
    print("資料來源：TAIFEX「前30個交易日期貨每筆成交資料」，免費、僅滾動30個交易日窗口，"
          "見taifex_intraday_loader.py模組docstring。", flush=True)

    hourly, diag = load_tx_hourly_bars(commodity_id=args.commodity_id, refresh=args.refresh_cache)

    if hourly is None:
        lines = [
            PREVIEW_DISCLAIMER,
            "資料下載失敗，看log診斷。",
            f"失敗原因(reason)：{diag.get('reason')}",
            f"是否使用快取：{diag.get('used_cache')}",
            "",
            "這支loader(taifex_intraday_loader.py)的下載機制是防禦性猜測，尚未經真實",
            "TAIFEX端點驗證(見該模組docstring)，失敗屬於預期中可能發生的情況，不代表",
            "程式設計有錯——需要人工檢查TAIFEX頁面實際的表單/下載機制，回頭修正",
            "taifex_intraday_loader.py裡的猜測部分。",
        ]
        _print_and_write_summary(lines, summary_path)
        print("資料下載失敗，看log診斷。", file=sys.stderr, flush=True)
        sys.exit(0)  # 優雅結束，不是crash：資料抓不到是預期中可能發生的情況

    print(f"成功取得日盤1小時K棒：{len(hourly)}根，近月合約={diag.get('front_month')}，"
          f"過濾掉非日盤ticks={diag.get('n_night_session_dropped')}筆，"
          f"使用快取={diag.get('used_cache')}", flush=True)

    if len(hourly) < 30:
        print(f"⚠️ 樣本只有{len(hourly)}根K棒，遠低於統計上有意義的最低門檻，"
              f"以下結果純屬觀察，不代表任何結論。", flush=True)

    features = compute_squeeze_kdj_features(hourly)
    n_entries = int(features["EntryFlag"].sum())
    print(f"訊號在這批K棒裡總共觸發{n_entries}次(EntryFlag=True的根數)。", flush=True)

    multiplier = CONTRACT_MULTIPLIER[args.contract]

    all_trades_rows = []
    summary_lines = [
        PREVIEW_DISCLAIMER,
        f"資料範圍：{hourly.index.min()} ~ {hourly.index.max()}(TX近月合約={diag.get('front_month')}，"
        f"僅日盤，共{len(hourly)}根1小時K棒)",
        f"契約：{'大台(TX)' if args.contract == 'big' else '小台(MTX)'}，"
        f"每點NT${multiplier}，{args.lots}口，"
        f"成本假設：手續費NT${args.commission_per_side:.0f}/口/邊 + "
        f"期交所規費(含期貨交易稅粗略近似)NT${args.exchange_fee_per_side:.0f}/口/邊"
        f"(未逐一查證最新公告費率，見程式註解)",
        f"訊號觸發次數(EntryFlag)：{n_entries}",
        "",
    ]

    if args.variant in ("a", "both"):
        raw_a = simulate_variant_a_trades(hourly, features)
        trades_a = _cost_trades(raw_a, multiplier, args.commission_per_side,
                                 args.exchange_fee_per_side, lots=args.lots)
        lines_a, _ = _stats_block("變體A(原規則：前K棒低點停損+K跌破80停利)", trades_a,
                                   args.starting_capital)
        summary_lines += lines_a + [""]
        for t in trades_a:
            all_trades_rows.append({"variant": "A", **t})

    if args.variant in ("b", "both"):
        raw_b = simulate_variant_b_trades(hourly, features, atr_period=14,
                                           atr_stop_mult=1.0, atr_target_mult=2.0)
        trades_b = _cost_trades(raw_b, multiplier, args.commission_per_side,
                                 args.exchange_fee_per_side, lots=args.lots)
        lines_b, _ = _stats_block("變體B(沿用ATR框架：停損1.0x ATR/停利2.0x ATR)", trades_b,
                                   args.starting_capital)
        summary_lines += lines_b + [""]
        for t in trades_b:
            all_trades_rows.append({"variant": "B", **t})

    summary_lines += [
        "再次強調：以上是30個交易日(約150根左右1小時K棒，僅日盤)的一次性預覽，",
        "交易筆數通常個位數到十幾筆，PF/勝率這類統計量在這種樣本數下極不穩定，",
        "跟compare_breakout.py/compare_equity_swing.py裡經過IS/OOS切分+bootstrap",
        "驗證過的「可靠組合」完全不是同一個信心層級，不能相提並論。",
    ]
    _print_and_write_summary(summary_lines, summary_path)

    trades_df = pd.DataFrame(all_trades_rows)
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    print(f"已寫入 {summary_path} 與 {trades_path}", flush=True)


if __name__ == "__main__":
    main()
