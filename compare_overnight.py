"""
compare_overnight.py
======================
隔日衝策略的完整驅動腳本：把 data_loader.py（股價）、chip_data_loader.py（三大法人）、
day_trading_loader.py（當沖比例）、overnight_chip_adapter.py（籌碼格式轉接）
串起來，餵給 overnight_momentum_engine.py 跑回測，並依本專案既有標準做 IS/OOS 切分。

用法（在 GitHub Actions 或任何能連上 TWSE/Yahoo 的環境執行，這個 sandbox 連不出去）：
    python3 compare_overnight.py --start 2023-09-15 --end 2026-09-14

IS/OOS 切分沿用本專案既有標準：70% 時間在前當 IS（樣本內），30% 在後當 OOS（樣本外），
避免像之前技術面組合那樣，樣本內看起來賺錢、實際上只是騎到多頭順風車。
"""

import argparse
import datetime

import pandas as pd

import data_loader
import chip_data_loader
import day_trading_loader
import us_market_loader
import dividend_data_loader
import overnight_chip_adapter as adapter
import overnight_momentum_engine as ome
import taifex_universe

IS_RATIO = 0.7

# --universe 59：目前已驗證過(訊號拆解/ATR掃描/合併測試/跨期驗證都在這份上做的)的
# 舊清單，59檔，只涵蓋部分熱門股期貨標的。
# --universe full：taifex_universe.py 的完整台股期貨可交易清單(249檔)，還沒有
# 用這份清單重新跑過任何選股策略挑選——只是把「能選的股票池」擴大，訊號權重/
# ATR參數仍然沿用59檔清單挑出來的那組，属於全新、未驗證的組合，建議先當成
# 獨立實驗跑一次IS/OOS或跨期驗證，不要直接假設它跟59檔清單的結果具有可比性。
UNIVERSE_CHOICES = {
    "59": data_loader.STOCK_FUTURES_WHITELIST,
    "full": taifex_universe.STOCK_FUTURES_UNIVERSE,
}


# ---------------------------------------------------------------------------
# 資料轉接：把 data_loader.py 的股價格式，轉成 overnight_momentum_engine.py 要的格式
# ---------------------------------------------------------------------------

def convert_price_data_to_indicators(price_data):
    """
    price_data: dict[code] -> DataFrame(index=DatetimeIndex, columns=Open/High/Low/Close/Volume)
                （data_loader.load_price_data() 的輸出格式）
    回傳: dict[code] -> overnight_momentum_engine.precompute_overnight_indicators() 算好的 DataFrame
    """
    indicators_by_code = {}
    for code, df in price_data.items():
        if df is None or df.empty:
            continue
        d = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }).copy()
        d = d.reset_index()
        # 不管原本 index 的欄位名稱叫 Date 或其他，強制把第一欄當作日期欄
        d.columns = ["date"] + list(d.columns[1:])
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y%m%d")
        d = d[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if d.empty:
            continue
        indicators_by_code[code] = ome.precompute_overnight_indicators(d)
    return indicators_by_code


def build_volume_df(price_data):
    """把 price_data 轉成 day_trading_loader.compute_day_trading_ratio() 要的
    (date, code, volume) 長格式，用來算當沖比例。"""
    frames = []
    for code, df in price_data.items():
        if df is None or df.empty or "Volume" not in df.columns:
            continue
        d = df.reset_index()
        d.columns = ["date"] + list(d.columns[1:])
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y%m%d")
        d["code"] = code
        frames.append(d[["date", "code", "Volume"]].rename(columns={"Volume": "volume"}))
    if not frames:
        return pd.DataFrame(columns=["date", "code", "volume"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 完整資料管線
# ---------------------------------------------------------------------------

def build_pipeline_inputs(whitelist, start_date, end_date, market_reference_code="2330",
                           refresh=False, price_loader=None, chip_loader=None,
                           day_trading_loader_fn=None, us_market_loader_fn=None,
                           dividend_loader_fn=None):
    """
    組出跑 overnight_momentum_engine.run_overnight_backtest() 所需的全部輸入。
    price_loader / chip_loader / day_trading_loader_fn / us_market_loader_fn /
    dividend_loader_fn 可用於測試時注入假資料源，預設分別是 data_loader.load_price_data /
    chip_data_loader.load_chip_data / day_trading_loader.load_day_trading_data /
    us_market_loader.load_us_market_returns / dividend_data_loader.load_dividend_events。

    start_date / end_date: datetime.date
    """
    price_loader = price_loader or data_loader.load_price_data
    chip_loader = chip_loader or chip_data_loader.load_chip_data
    day_trading_loader_fn = day_trading_loader_fn or day_trading_loader.load_day_trading_data
    us_market_loader_fn = us_market_loader_fn or us_market_loader.load_us_market_returns
    dividend_loader_fn = dividend_loader_fn or dividend_data_loader.load_dividend_events

    start_dash = start_date.strftime("%Y-%m-%d")
    end_dash = end_date.strftime("%Y-%m-%d")

    price_data = price_loader(whitelist, start_dash, end_dash, refresh=refresh)
    if market_reference_code not in price_data:
        raise RuntimeError(
            f"參考股 {market_reference_code} 沒有價格資料，無法計算相對大盤強弱，"
            f"請確認 whitelist 裡有這檔、且下載成功。"
        )

    indicators_by_code = convert_price_data_to_indicators(price_data)

    market_df = indicators_by_code[market_reference_code][["date", "close"]]
    market_returns_df = ome.precompute_market_returns(market_df)

    universe_codes = set(whitelist.keys())

    chip_data = chip_loader(start_dash, end_dash, universe_codes=universe_codes, refresh=refresh)
    foreign_ratio_df, trust_ratio_df = adapter.build_institutional_ratio_dfs(chip_data, indicators_by_code)

    volume_df = build_volume_df(price_data)
    day_trading_raw = day_trading_loader_fn(start_date, end_date, universe_codes=universe_codes, refresh=refresh)
    day_trading_ratio_df = day_trading_loader.compute_day_trading_ratio(day_trading_raw, volume_df)

    trading_days = sorted(indicators_by_code[market_reference_code]["date"].unique().tolist())

    us_market_returns_df = us_market_loader_fn(start_dash, end_dash, refresh=refresh)

    ex_dividend_dates_by_code = dividend_loader_fn(whitelist, start_dash, end_dash, refresh=refresh)

    return {
        "indicators_by_code": indicators_by_code,
        "market_returns_df": market_returns_df,
        "foreign_ratio_df": foreign_ratio_df,
        "trust_ratio_df": trust_ratio_df,
        "day_trading_ratio_df": day_trading_ratio_df,
        "trading_days": trading_days,
        "universe_codes": universe_codes,
        "us_market_returns_df": us_market_returns_df,
        "ex_dividend_dates_by_code": ex_dividend_dates_by_code,
    }


def split_is_oos(trading_days, is_ratio=IS_RATIO):
    split_idx = int(len(trading_days) * is_ratio)
    is_days = trading_days[:split_idx]
    oos_days = trading_days[split_idx:]
    return is_days, oos_days


def run_is_oos_backtest(pipeline_inputs, is_ratio=IS_RATIO, **backtest_kwargs):
    """對同一份資料分別跑 IS / OOS 回測，回傳 (is_trades, oos_trades, is_summary, oos_summary)。"""
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
    )
    common_args.update(backtest_kwargs)

    is_trades = ome.run_overnight_backtest(trading_days=is_days, **common_args)
    oos_trades = ome.run_overnight_backtest(trading_days=oos_days, **common_args)

    is_summary = ome.summarize_overnight(is_trades)
    oos_summary = ome.summarize_overnight(oos_trades)

    return is_trades, oos_trades, is_summary, oos_summary


def print_summary(label, summary):
    print(f"\n--- {label} ---")
    print(f"  交易次數: {summary['total_trades']}")
    print(f"  勝率: {summary['win_rate']:.1f}%")
    print(f"  盈虧比(PF): {summary['profit_factor']:.2f}")
    print(f"  總損益: {summary['total_pnl']:,.0f}")
    print(f"  平均每筆損益: {summary['avg_pnl']:,.0f}")
    if summary.get("by_exit_reason"):
        print("  出場原因分布:")
        for reason, stats in summary["by_exit_reason"].items():
            print(f"    {reason}: {stats['count']}筆, 合計{stats['sum']:,.0f}, 平均{stats['mean']:,.0f}")


# ---------------------------------------------------------------------------
# CLI 進入點
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="隔日衝策略 IS/OOS 回測")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--refresh", action="store_true", help="忽略快取，強制重新下載所有資料")
    parser.add_argument("--top-n", type=int, default=ome.TOP_N)
    parser.add_argument("--tech-weight", type=float, default=ome.TECH_WEIGHT)
    parser.add_argument("--chip-weight", type=float, default=ome.CHIP_WEIGHT)
    parser.add_argument("--no-us-filter", action="store_true",
                         help="停用隔夜美股大跌濾網，方便跟啟用時的結果對照比較")
    parser.add_argument("--us-drop-threshold", type=float, default=ome.US_MARKET_DROP_THRESHOLD_PCT,
                         help="美股隔夜跌幅超過這個%%(負數)就不進場，預設-1.5")
    parser.add_argument("--universe", choices=list(UNIVERSE_CHOICES.keys()), default="59",
                         help="選股池：59=目前已驗證過的59檔舊清單(預設)；"
                              "full=taifex_universe.py的249檔完整可交易清單"
                              "(全新、未驗證的股票池，訊號/ATR參數仍沿用59檔挑出來的那組，"
                              "結果不能直接跟59檔版本比較，建議當成獨立實驗看待)")
    parser.add_argument("--slippage-pct", type=float, default=0.0,
                         help="模擬滑價百分比(預設0=不模擬)，例如0.1代表買進多付0.1%%、賣出少拿0.1%%")
    args = parser.parse_args()

    start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()

    whitelist = UNIVERSE_CHOICES[args.universe]
    if args.universe == "full":
        print(f"⚠️ 使用 --universe full：{len(whitelist)}檔完整清單，"
              f"這是全新、還沒驗證過的股票池，訊號權重/ATR參數仍是用59檔清單挑出來的那組，"
              f"結果請當成獨立實驗看待，不要直接拿來跟59檔版本的PF比較。")

    print(f"下載/整理資料 ({args.start} ~ {args.end}) ...")
    pipeline_inputs = build_pipeline_inputs(
        whitelist, start_date, end_date, refresh=args.refresh,
    )
    if args.no_us_filter:
        pipeline_inputs["us_market_returns_df"] = None
        print("（已停用隔夜美股濾網）")

    print(f"共 {len(pipeline_inputs['trading_days'])} 個交易日，開始跑 IS/OOS 回測 ...")
    is_trades, oos_trades, is_summary, oos_summary = run_is_oos_backtest(
        pipeline_inputs, top_n=args.top_n,
        tech_weight=args.tech_weight, chip_weight=args.chip_weight,
        us_market_drop_threshold=args.us_drop_threshold,
        slippage_pct=args.slippage_pct,
    )

    print_summary("樣本內 IS (前70%)", is_summary)
    print_summary("樣本外 OOS (後30%)", oos_summary)


if __name__ == "__main__":
    main()
