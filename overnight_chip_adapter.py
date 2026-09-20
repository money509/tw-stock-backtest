"""
overnight_chip_adapter.py
============================
把 chip_data_loader.py 的原始輸出（買賣超「股數」，dict[code] -> DataFrame(index=Timestamp)）
轉換成 overnight_momentum_engine.py 期待的介面（買超「金額佔成交金額比重」，
DataFrame[date(str YYYYMMDD), code, ratio]）。

為什麼需要這一層轉接，不直接讓引擎吃 chip_data_loader.py 的原始格式：
  1. chip_data_loader.load_chip_data() 給的是「股數」(foreign_net/trust_net)，
     不是金額。要換算成「買超金額 ÷ 當天成交金額」的比重，需要當天收盤價
     （成交金額本身也要用收盤價 close * 成交量 volume 估算，這是
     overnight_momentum_engine.precompute_overnight_indicators() 已經算好的
     turnover_value 欄位）。
  2. chip_data_loader 的 index 是 pandas Timestamp，overnight_momentum_engine
     用的是 'date' 字串欄位（YYYYMMDD），格式要對齊才能 merge。

換算公式（用收盤價估算買超金額，非交易所公布的真實成交均價，是近似值）：
    買超金額 ≈ 買賣超股數 × 當天收盤價
    ratio = 買超金額 / 當天成交金額(turnover_value)
"""

import pandas as pd


def build_institutional_ratio_dfs(chip_data, indicators_by_code):
    """
    參數：
        chip_data          : chip_data_loader.load_chip_data() 的回傳值，
                              dict[code] -> DataFrame(index=Timestamp,
                              columns=[foreign_net, trust_net, dealer_net, total_net])
        indicators_by_code  : dict[code] -> overnight_momentum_engine
                              .precompute_overnight_indicators() 的回傳值
                              （需要有 'date'(str YYYYMMDD), 'close', 'turnover_value' 欄位）

    回傳：
        (foreign_ratio_df, trust_ratio_df)，每個都是
        DataFrame[date(str), code(str), ratio(float)]，可以直接傳給
        overnight_momentum_engine.scan_candidates_for_date() /
        run_overnight_backtest() 的 foreign_ratio_df / trust_ratio_df 參數。

    只有「當天有股價資料、且成交金額 > 0」的日期才會被保留；
    沒有股價資料可以對照的籌碼資料會被安靜跳過（例如籌碼資料涵蓋的日期範圍
    比股價資料還長的情況）。
    """
    foreign_frames = []
    trust_frames = []

    for code, chip_df in chip_data.items():
        price_df = indicators_by_code.get(code)
        if price_df is None or price_df.empty or chip_df.empty:
            continue

        cdf = chip_df.reset_index()
        # chip_data_loader 用 .set_index("date") 存 Timestamp，reset_index 後欄位叫 'date'
        date_col = "date" if "date" in cdf.columns else cdf.columns[0]
        cdf = cdf.rename(columns={date_col: "date"})
        cdf["date"] = pd.to_datetime(cdf["date"]).dt.strftime("%Y%m%d")

        price_slim = price_df[["date", "close", "turnover_value"]]
        merged = cdf.merge(price_slim, on="date", how="inner")
        merged = merged[merged["turnover_value"] > 0].copy()
        if merged.empty:
            continue

        merged["code"] = code
        merged["foreign_amount"] = merged["foreign_net"] * merged["close"]
        merged["trust_amount"] = merged["trust_net"] * merged["close"]
        merged["foreign_ratio_val"] = merged["foreign_amount"] / merged["turnover_value"]
        merged["trust_ratio_val"] = merged["trust_amount"] / merged["turnover_value"]

        foreign_frames.append(
            merged[["date", "code", "foreign_ratio_val"]].rename(columns={"foreign_ratio_val": "ratio"})
        )
        trust_frames.append(
            merged[["date", "code", "trust_ratio_val"]].rename(columns={"trust_ratio_val": "ratio"})
        )

    foreign_ratio_df = (
        pd.concat(foreign_frames, ignore_index=True) if foreign_frames
        else pd.DataFrame(columns=["date", "code", "ratio"])
    )
    trust_ratio_df = (
        pd.concat(trust_frames, ignore_index=True) if trust_frames
        else pd.DataFrame(columns=["date", "code", "ratio"])
    )
    return foreign_ratio_df, trust_ratio_df
