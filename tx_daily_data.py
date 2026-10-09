"""
tx_daily_data.py
====================
台指期(TX)「日線」資料建構 —— 給tx_daily_engine.py / compare_tx_daily.py用。

資料來源(依序)：
  1. 主要：taifex_history_loader.py的社群1分鐘K棒(Google Drive，crazyindicator.pixnet.net)，
     2001-01-01 ~ 2023-12-31(只用2001_2010/2011_2020/2021_2023三段，1998_2000不用)。
     只取「日盤」08:45~13:45的分鐘K棒聚合成日線。
  2. 驗證：^TWII加權指數日線(yfinance)，在重疊日期上做OHLC自洽/覆蓋率/報酬相關/基差檢查。
  3. 備援：期貨資料下載/解析/驗證任何一步失敗 → **整段(1998~最新)改用^TWII日線**，
     報告最上方明確標示「價格來源：加權指數(期貨資料驗證失敗：原因…)」，不會
     把一半期貨、一半「看起來怪怪的」資料悄悄混在一起。
  4. 2024-01-01 ~ 最新：Drive資料只到2023 → 這段一律用^TWII當「代理」(指數≠期貨，
     7~8月除息旺季期貨對指數的逆價差明顯)，只做參考報告。

已知/未知的Drive檔案格式(誠實列出)：
  - 已知(第一次真實執行的raw_preview診斷，見git commit ba5a484)：7z壓縮，內含CSV，
    欄位 Date,Time,Open,High,Low,Close,Volume，Date像"2011/01/03"、Time像"08:46:00"。
    → **沒有合約月份/到期日欄位**，是一條「單一連續」的分鐘序列(作者說是「原始逐口
    合約版」，推測是每天的近月合約，換月時價格直接跳到下一個合約，沒有做價差調整；
    使用者指定先不用的另一份「調整/連續合約價差版」才有調整)。
  - 未知：(a) K棒時間是「結束時間」(08:46=08:45~08:46)還是「開始時間」——兩種都落在
    08:45~13:45的過濾範圍內，不影響日線；(b) 有沒有夜盤(2017-05-15起才有夜盤)——
    用時間過濾，見下面「日盤/夜盤」；(c) 換月是在結算日當天盤中、結算日收盤後、
    還是更早(依成交量)——用下面的啟發式偵測，誤差在報告裡講清楚；(d) 2001~2010那段
    的格式是否跟2011以後一樣——沒看過，sniff失敗就整個退回加權指數。

日盤/夜盤(day vs night session)：
  每根分鐘K棒取「時間(HH:MM)」，只保留 08:45 <= HH:MM <= 13:45 的列(日盤，含08:45開盤那
  根跟13:45收盤那根，不管時間戳是起點或終點標記都涵蓋)。15:00~隔天05:00的夜盤K棒、
  以及其他時間的列都丟掉，並在診斷檔裡統計「日盤/夜盤/其他」各有幾列、有幾天出現夜盤
  列，讓使用者確認。夜盤跨午夜的K棒日期是「隔天」，但因為是用時間過濾，不會被誤算進
  隔天的日盤。日線的日期 = 日盤K棒所在的日曆日。

換月(rollover)與調整(只在期貨模式)：
  因為檔案沒有合約欄位，沒辦法用taifex_intraday_loader.select_front_month()那種
  「依合約月份挑近月」的邏輯，改用**跟^TWII比較的啟發式偵測**(調整是近似值，報告會說)：
    基差 b_t = 期貨13:30收盤 − 指數收盤(用13:30那根期貨收盤，跟指數13:30收盤對齊時間，
    避免13:30~13:45這15分鐘的行情把雜訊灌進基差)。
    每個月的理論結算日 = 第三個星期三(settlement_calendar.third_wednesday)，遇假日順延
    = 期貨資料裡第一個 >= 第三個星期三的交易日 s。換月視窗 = s的前2個交易日 ~ s的後1個
    交易日；視窗內 |Δb| 最大的那天當作換月日 d，Δb_d 當作價差(新合約 − 舊合約)。
    雜訊門檻：非視窗日Δb的穩健標準差 σ(1.4826×MAD)；|Δb_d| >= max(ROLL_SIGMA_MULT×σ,
    ROLL_MIN_POINTS) 才算「顯著」；不顯著的月份仍記一次換月(預設日 = s的下一個交易日，
    換月成本照收)，但價差當作0(在雜訊裡分辨不出來，硬調只是在加雜訊)。
  調整方式：Panama(差額)倒推調整 —— 每個換月日d之前的所有K棒都加上該次價差，最後一段
  (最新合約)不動。未調整(Raw*)跟調整後(Open/High/Low/Close)兩份都保留，Offset = 調整後 − 未調整。
  訊號跟點數損益都用調整後序列(連續期貨回測的標準做法)；期交稅用未調整價格算。
  ⚠️ Panama調整累積20多年的逆價差，早年調整後價格可能變得很低甚至是負數——這裡用到的
  指標(均線/ATR/ADX/唐奇安/KDJ/布林/Keltner/動能)全部只跟「價格差」有關、跟整條序列加
  一個常數無關(test_tx_daily_data.py有測)，所以不影響訊號；百分比類的東西(稅)用未調整價格。

驗證(期貨模式，對^TWII重疊日期；任一項沒過就整段退回加權指數)：
  - OHLC自洽：High >= max(Open,Close) 且 Low <= min(Open,Close) 的天數 >= 99%
  - 覆蓋率：期貨資料起訖範圍內的^TWII交易日，有 >= 95% 也出現在期貨日線裡
  - 日報酬相關(調整後期貨13:30收盤 vs 指數收盤) >= 0.95
    (調整後報酬 = Δ調整後價格 / 前一天未調整價格，避免Panama早年負價格讓分母出問題)
  - 未調整期貨13:30收盤 vs 指數收盤 的 |差|/指數 中位數 < 3%
"""
import datetime
import io
import json
import os
import time

import numpy as np
import pandas as pd

from settlement_calendar import third_wednesday

DAY_SESSION_START_MIN = 8 * 60 + 45    # 08:45
DAY_SESSION_END_MIN = 13 * 60 + 45     # 13:45
INDEX_CLOSE_MIN = 13 * 60 + 30         # 13:30(加權指數收盤時間)
NIGHT_SESSION_START_MIN = 15 * 60      # 15:00
NIGHT_SESSION_END_MIN = 5 * 60         # 05:00(隔天)

FUTURES_PERIOD_KEYS = ["2001_2010", "2011_2020", "2021_2023"]
FUTURES_START = "2001-01-01"
FUTURES_END = "2023-12-31"
INDEX_SYMBOL = "^TWII"
INDEX_FALLBACK_START = "1998-01-01"

ROLL_WINDOW_BEFORE = 2
ROLL_WINDOW_AFTER = 1
ROLL_SIGMA_MULT = 4.0
ROLL_MIN_POINTS = 10.0

MIN_OHLC_CONSISTENT_PCT = 99.0
MIN_COVERAGE_PCT = 95.0
MIN_RETURN_CORR = 0.95
MAX_MEDIAN_BASIS_PCT = 3.0
MIN_FUTURES_DAYS = 500

INDEX_CACHE_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")


# ---------------------------------------------------------------------------
# 1分鐘 → 日線
# ---------------------------------------------------------------------------
def classify_sessions(index: pd.DatetimeIndex) -> pd.Series:
    """每根分鐘K棒標記 "day"(08:45~13:45) / "night"(15:00~23:59或00:00~05:00) / "other"。"""
    minutes = np.asarray(index.hour * 60 + index.minute)
    day = (minutes >= DAY_SESSION_START_MIN) & (minutes <= DAY_SESSION_END_MIN)
    night = (minutes >= NIGHT_SESSION_START_MIN) | (minutes <= NIGHT_SESSION_END_MIN)
    out = np.where(day, "day", np.where(night, "night", "other"))
    return pd.Series(out, index=index)


def minute_bars_to_daily(bars: pd.DataFrame):
    """分鐘K棒(index=時間戳，欄位Open/High/Low/Close/[Volume]) → 日盤日線。
    回傳(daily_df, session_stats)。daily_df欄位：Open/High/Low/Close/Close1330/Volume/NBars，
    index=日期(Timestamp，00:00)。Close1330=當天<=13:30最後一根的收盤(對齊加權指數收盤
    時間，只用在基差/驗證，不用在交易)。"""
    if bars is None or bars.empty:
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Close1330", "Volume", "NBars"])
        return empty, {"rows_total": 0, "rows_day": 0, "rows_night": 0, "rows_other": 0,
                       "days_with_night_rows": 0}
    bars = bars.sort_index()
    sessions = classify_sessions(bars.index)
    stats = {
        "rows_total": int(len(bars)),
        "rows_day": int((sessions == "day").sum()),
        "rows_night": int((sessions == "night").sum()),
        "rows_other": int((sessions == "other").sum()),
    }
    night_dates = pd.Index(bars.index[(sessions == "night").to_numpy()].normalize()).unique()
    stats["days_with_night_rows"] = int(len(night_dates))
    stats["first_night_row"] = (str(bars.index[(sessions == "night").to_numpy()][0])
                                if stats["rows_night"] else None)

    day = bars[(sessions == "day").to_numpy()].copy()
    if day.empty:
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Close1330", "Volume", "NBars"])
        return empty, stats
    if "Volume" not in day.columns:
        day["Volume"] = 0.0
    day["_date"] = day.index.normalize()
    g = day.groupby("_date")
    daily = pd.DataFrame({
        "Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
        "Close": g["Close"].last(), "Volume": g["Volume"].sum(), "NBars": g["Close"].size(),
    })
    minutes = day.index.hour * 60 + day.index.minute
    pre = day[np.asarray(minutes <= INDEX_CLOSE_MIN)]
    c1330 = pre.groupby("_date")["Close"].last()
    daily["Close1330"] = c1330.reindex(daily.index)
    daily["Close1330"] = daily["Close1330"].fillna(daily["Close"])
    daily.index = pd.DatetimeIndex(daily.index)
    daily.index.name = "Date"
    stats["n_days"] = int(len(daily))
    stats["median_bars_per_day"] = float(daily["NBars"].median())
    stats["days_lt_100_bars"] = int((daily["NBars"] < 100).sum())
    return daily[["Open", "High", "Low", "Close", "Close1330", "Volume", "NBars"]], stats


# ---------------------------------------------------------------------------
# 加權指數OHLC(yfinance)
# ---------------------------------------------------------------------------
def _yf_download(symbol, start, end):
    """實際打yfinance(測試會monkeypatch這個函式，保證不連網)。"""
    import yfinance as yf
    return yf.download(tickers=symbol, start=start, end=end, interval="1d",
                       progress=False, auto_adjust=False, threads=False)


def _clean_index_ohlc(raw: pd.DataFrame, symbol: str):
    """yfinance原始結果 → Open/High/Low/Close(float，日期index，去tz/去重/排序)。
    回傳(df, n_repaired)：Open缺值或<=0用前一天收盤補；High/Low不自洽的列用
    max/min(O,H,L,C)修正並計數(診斷檔會列出修了幾列，不是悄悄修)。"""
    cols = {}
    for c in ("Open", "High", "Low", "Close"):
        if c not in raw.columns.get_level_values(0):
            return pd.DataFrame(columns=["Open", "High", "Low", "Close"]), 0
        s = raw[c]
        if isinstance(s, pd.DataFrame):
            s = s[symbol] if symbol in s.columns else s.iloc[:, 0]
        cols[c] = pd.to_numeric(s, errors="coerce")
    df = pd.DataFrame(cols)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Close"] > 0].dropna(subset=["Close"])
    bad_open = df["Open"].isna() | (df["Open"] <= 0)
    df.loc[bad_open, "Open"] = df["Close"].shift(1)[bad_open]
    df["Open"] = df["Open"].fillna(df["Close"])
    for c in ("High", "Low"):
        bad = df[c].isna() | (df[c] <= 0)
        df.loc[bad, c] = df.loc[bad, "Close"]
    hi = df[["Open", "High", "Low", "Close"]].max(axis=1)
    lo = df[["Open", "High", "Low", "Close"]].min(axis=1)
    n_repaired = int(((df["High"] < hi) | (df["Low"] > lo)).sum())
    df["High"] = hi
    df["Low"] = lo
    df.index.name = "Date"
    return df.astype(float), n_repaired


def load_index_ohlc(symbol=INDEX_SYMBOL, start=INDEX_FALLBACK_START, end=None, refresh=False,
                    cache_dir=INDEX_CACHE_DIR_DEFAULT):
    """^TWII日線OHLC。end跟yfinance一樣「不含當天」(預設=今天+1，包含今天)。有本機
    CSV快取(檔名含start/end)。失敗回傳(空df, reason)，不丟例外。
    為什麼不沿用data_loader.load_index_series：那支只回傳收盤價，這裡需要OHLC(交易要用
    開盤價成交、ATR要用高低價)；也刻意不用0050.TW當備援——0050的價格尺度跟指數點數
    完全不同，不能拿來算台指期的點數損益。回傳(df, info_dict)。"""
    if end is None:
        end = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    os.makedirs(cache_dir, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in symbol)
    cache_path = os.path.join(cache_dir, f"INDEXOHLC_{safe}_{start}_{end}.csv")
    if not refresh and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not cached.empty and {"Open", "High", "Low", "Close"} <= set(cached.columns):
                return cached[["Open", "High", "Low", "Close"]].astype(float), {
                    "reason": "cached", "n_repaired": None}
        except Exception:
            pass
    raw = None
    last_err = None
    for attempt in range(2):
        try:
            raw = _yf_download(symbol, start, end)
            break
        except Exception as e:  # 網路錯誤：重試一次
            last_err = e
            raw = None
            time.sleep(1)
    if raw is None or getattr(raw, "empty", True):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"]), {
            "reason": f"download_failed:{last_err}" if last_err else "download_empty", "n_repaired": None}
    df, n_repaired = _clean_index_ohlc(raw, symbol)
    if df.empty:
        return df, {"reason": "parse_failed", "n_repaired": None}
    try:
        df.to_csv(cache_path)
    except Exception:
        pass
    return df, {"reason": "ok", "n_repaired": n_repaired}


# ---------------------------------------------------------------------------
# 換月偵測 + Panama調整
# ---------------------------------------------------------------------------
def settlement_trading_days(dates: pd.DatetimeIndex):
    """每個月理論結算日(第三個星期三)順延到dates裡第一個>=它的交易日。回傳
    [(theoretical_date, position_in_dates)]，只列出落在dates範圍內的月份。"""
    dates = pd.DatetimeIndex(dates).sort_values()
    if len(dates) == 0:
        return []
    out = []
    y, m = dates[0].year, dates[0].month
    last = (dates[-1].year, dates[-1].month)
    while (y, m) <= last:
        tw = pd.Timestamp(third_wednesday(y, m))
        pos = int(dates.searchsorted(tw, side="left"))
        if pos < len(dates) and tw >= dates[0] and dates[pos].month == m:
            out.append((tw, pos))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def calendar_roll_days(dates: pd.DatetimeIndex) -> pd.Series:
    """沒有實際換月資訊時(加權指數代理)用的換月日：結算交易日的下一個交易日。
    回傳bool Series(index=dates)。"""
    dates = pd.DatetimeIndex(dates).sort_values()
    flag = pd.Series(False, index=dates)
    for _tw, pos in settlement_trading_days(dates):
        if pos + 1 < len(dates):
            flag.iloc[pos + 1] = True
    return flag


def detect_rolls_vs_index(fut_daily: pd.DataFrame, index_close: pd.Series,
                          sigma_mult=ROLL_SIGMA_MULT, min_points=ROLL_MIN_POINTS,
                          window_before=ROLL_WINDOW_BEFORE, window_after=ROLL_WINDOW_AFTER):
    """啟發式換月偵測(見模組docstring)。fut_daily需要Close1330(沒有就用Close)。
    回傳(rolls, info)：rolls = list of dict {roll_date, settle_date, gap, significant,
    delta_basis}，每個月一筆；info = {"noise_sigma":..., "n_significant":...}。"""
    dates = pd.DatetimeIndex(fut_daily.index).sort_values()
    fclose = fut_daily["Close1330"] if "Close1330" in fut_daily.columns else fut_daily["Close"]
    idx_close = index_close.reindex(dates)
    basis = fclose.reindex(dates) - idx_close
    dbasis = basis.diff()  # 前一個期貨交易日 → 今天；指數缺值的日子是NaN

    settles = settlement_trading_days(dates)
    in_window = np.zeros(len(dates), dtype=bool)
    for _tw, pos in settles:
        lo, hi = max(0, pos - window_before), min(len(dates) - 1, pos + window_after)
        in_window[lo:hi + 1] = True
    quiet = dbasis[~in_window].dropna()
    if len(quiet) >= 10:
        mad = float((quiet - quiet.median()).abs().median())
        sigma = 1.4826 * mad
    else:
        sigma = float("nan")
    threshold = max(sigma_mult * sigma, min_points) if np.isfinite(sigma) else float("inf")

    rolls = []
    for tw, pos in settles:
        lo, hi = max(1, pos - window_before), min(len(dates) - 1, pos + window_after)
        win = dbasis.iloc[lo:hi + 1].dropna()
        default_pos = pos + 1
        if default_pos >= len(dates):
            continue  # 資料最後一個月：結算後已經沒有資料，無從換月
        if len(win) and np.isfinite(threshold) and win.abs().max() >= threshold:
            d = win.abs().idxmax()
            rolls.append({"roll_date": pd.Timestamp(d), "settle_date": dates[pos],
                          "gap": float(win.loc[d]), "significant": True,
                          "delta_basis": float(win.loc[d])})
        else:
            d = dates[default_pos]
            db = dbasis.iloc[default_pos]
            rolls.append({"roll_date": pd.Timestamp(d), "settle_date": dates[pos],
                          "gap": 0.0, "significant": False,
                          "delta_basis": float(db) if pd.notna(db) else float("nan")})
    info = {"noise_sigma": sigma, "threshold": threshold,
            "n_rolls": len(rolls), "n_significant": int(sum(r["significant"] for r in rolls))}
    return rolls, info


def apply_back_adjustment(daily: pd.DataFrame, rolls: list) -> pd.DataFrame:
    """Panama倒推調整：每個換月日d(新合約從d開始)，d之前所有K棒的OHLC(與Close1330)
    都加上gap(=新合約−舊合約)。保留Raw*欄位(未調整)，Offset=調整後−未調整，
    RollDay=該天是換月日(不論價差是否顯著，換月成本都照收)。"""
    out = daily.copy()
    price_cols = [c for c in ("Open", "High", "Low", "Close", "Close1330") if c in out.columns]
    for c in price_cols:
        out["Raw" + c] = out[c].astype(float)
    offset = pd.Series(0.0, index=out.index)
    roll_flag = pd.Series(False, index=out.index)
    for r in rolls:
        d = pd.Timestamp(r["roll_date"])
        if d in roll_flag.index:
            roll_flag.loc[d] = True
        offset[out.index < d] += float(r["gap"])
    for c in price_cols:
        out[c] = out["Raw" + c] + offset
    out["Offset"] = offset
    out["RollDay"] = roll_flag
    return out


# ---------------------------------------------------------------------------
# 驗證
# ---------------------------------------------------------------------------
def validate_against_index(adj_daily: pd.DataFrame, index_df: pd.DataFrame,
                           min_ohlc_pct=MIN_OHLC_CONSISTENT_PCT, min_coverage_pct=MIN_COVERAGE_PCT,
                           min_corr=MIN_RETURN_CORR, max_median_basis_pct=MAX_MEDIAN_BASIS_PCT,
                           min_days=MIN_FUTURES_DAYS):
    """回傳(passed: bool, stats: dict, fail_reasons: list[str])。見模組docstring的門檻。"""
    stats, fails = {}, []
    n = len(adj_daily)
    stats["n_futures_days"] = int(n)
    if n < min_days:
        fails.append(f"期貨日線只有{n}天(<{min_days})")
    if n == 0 or index_df is None or index_df.empty:
        if index_df is None or index_df.empty:
            fails.append("沒有加權指數資料可以比對")
        return False, stats, fails

    o, h, l, c = (adj_daily["Raw" + k] if "Raw" + k in adj_daily.columns else adj_daily[k]
                  for k in ("Open", "High", "Low", "Close"))
    consistent = (h >= np.maximum(o, c)) & (l <= np.minimum(o, c))
    stats["ohlc_consistent_pct"] = float(consistent.mean() * 100)
    if stats["ohlc_consistent_pct"] < min_ohlc_pct:
        fails.append(f"OHLC自洽天數只有{stats['ohlc_consistent_pct']:.2f}%(<{min_ohlc_pct}%)")

    first, last = adj_daily.index.min(), adj_daily.index.max()
    idx_in = index_df.loc[(index_df.index >= first) & (index_df.index <= last)]
    covered = idx_in.index.isin(adj_daily.index)
    stats["index_days_in_range"] = int(len(idx_in))
    stats["coverage_pct"] = float(covered.mean() * 100) if len(idx_in) else 0.0
    if stats["coverage_pct"] < min_coverage_pct:
        fails.append(f"覆蓋率{stats['coverage_pct']:.2f}%(<{min_coverage_pct}%)")

    raw1330 = adj_daily["RawClose1330"] if "RawClose1330" in adj_daily.columns else (
        adj_daily["RawClose"] if "RawClose" in adj_daily.columns else adj_daily["Close"])
    adj1330 = adj_daily["Close1330"] if "Close1330" in adj_daily.columns else adj_daily["Close"]
    common = adj_daily.index.intersection(index_df.index)
    stats["n_common_days"] = int(len(common))
    if len(common) < 30:
        fails.append(f"跟加權指數重疊的日期只有{len(common)}天")
        return False, stats, fails
    fr = (adj1330.loc[common].diff() / raw1330.loc[common].shift(1)).iloc[1:]
    ir = index_df["Close"].loc[common].pct_change().iloc[1:]
    ok = fr.notna() & ir.notna() & np.isfinite(fr)
    corr = float(np.corrcoef(fr[ok], ir[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    stats["return_corr"] = corr
    raw_close = adj_daily["RawClose"] if "RawClose" in adj_daily.columns else adj_daily["Close"]
    adj_close = adj_daily["Close"]
    fr2 = (adj_close.loc[common].diff() / raw_close.loc[common].shift(1)).iloc[1:]
    ok2 = fr2.notna() & ir.notna() & np.isfinite(fr2)
    stats["return_corr_close1345"] = float(np.corrcoef(fr2[ok2], ir[ok2])[0, 1]) if ok2.sum() > 2 else float("nan")
    if not np.isfinite(corr) or corr < min_corr:
        fails.append(f"日報酬相關{corr:.4f}(<{min_corr})")

    basis_pct = ((raw1330.loc[common] - index_df["Close"].loc[common]).abs()
                 / index_df["Close"].loc[common] * 100)
    stats["median_abs_basis_pct"] = float(basis_pct.median())
    stats["p95_abs_basis_pct"] = float(basis_pct.quantile(0.95))
    if stats["median_abs_basis_pct"] >= max_median_basis_pct:
        fails.append(f"未調整期貨vs指數 |差|/指數 中位數{stats['median_abs_basis_pct']:.3f}%(>={max_median_basis_pct}%)")
    return len(fails) == 0, stats, fails


# ---------------------------------------------------------------------------
# 原始檔案格式描述(診斷用)
# ---------------------------------------------------------------------------
def describe_raw_format(local_path, n_lines=200):
    """讀已下載的原始檔(可能是7z/zip)前n_lines行，回報sniff出來的欄位、偵測結果、
    前幾行原文。只用在診斷檔，失敗回傳{"error": ...}。"""
    try:
        from taifex_history_loader import _is_archive, _extract_first_member, _sniff_table, _detect_columns
        with open(local_path, "rb") as f:
            raw = f.read()
        if _is_archive(raw):
            raw, _names = _extract_first_member(raw)
            if raw is None:
                return {"error": "archive_extract_failed"}
        head_lines = raw.splitlines()[:n_lines]
        head = b"\n".join(head_lines)
        df = _sniff_table(head)
        if df is None:
            return {"error": "sniff_failed", "first_lines": [repr(x) for x in head_lines[:3]]}
        det = _detect_columns(df)
        header_text = head_lines[0].decode("utf-8", errors="replace") if head_lines else ""
        contract_like = [c for c in df.columns
                         if any(k in str(c).lower() for k in ("contract", "month", "expir", "月份", "合約", "symbol"))]
        return {"columns": list(df.columns), "detected": det, "first_line": header_text,
                "sample_lines": [x.decode("utf-8", errors="replace") for x in head_lines[1:4]],
                "contract_like_columns": contract_like}
    except Exception as e:
        return {"error": f"describe_failed:{e}"}


# ---------------------------------------------------------------------------
# 整合
# ---------------------------------------------------------------------------
def _default_load_minute(refresh, cache_dir):
    from taifex_history_loader import load_tx_history_bars, TAIFEX_HISTORY_CACHE_DIR_DEFAULT
    cache_dir = cache_dir or TAIFEX_HISTORY_CACHE_DIR_DEFAULT
    bars, diag = load_tx_history_bars(FUTURES_PERIOD_KEYS, bar_minutes=1, refresh=refresh,
                                      cache_dir=cache_dir)
    formats = {}
    for key in FUTURES_PERIOD_KEYS:
        p = os.path.join(cache_dir, f"{key}.raw")
        if os.path.exists(p):
            formats[key] = describe_raw_format(p)
    diag["raw_formats"] = formats
    return bars, diag


def _index_frame(index_df: pd.DataFrame) -> pd.DataFrame:
    """加權指數 → 引擎吃的格式(Offset=0、RollDay=曆法換月日)。"""
    out = index_df[["Open", "High", "Low", "Close"]].astype(float).copy()
    for c in ("Open", "High", "Low", "Close"):
        out["Raw" + c] = out[c]
    out["Offset"] = 0.0
    out["RollDay"] = calendar_roll_days(out.index).reindex(out.index).fillna(False).astype(bool)
    return out


def build_tx_daily_dataset(refresh=False, futures_cache_dir=None, index_cache_dir=INDEX_CACHE_DIR_DEFAULT,
                           index_end=None, load_minute_fn=None, load_index_fn=None):
    """回傳dataset dict：
      source: "futures" / "index"
      source_line: 報告最上方那一行
      fallback_reason: None或原因字串
      main: 引擎用的日線(期貨模式=期貨2001~2023；備援=加權指數1998~最新)
      index: 加權指數日線(引擎格式，1998~最新，recent段與備援用)
      futures_raw_daily / rolls / roll_info / validation / parse ...：診斷用
    index下載也失敗時 source=None、main=None(呼叫端應印出原因並結束)。"""
    load_minute_fn = load_minute_fn or (lambda: _default_load_minute(refresh, futures_cache_dir))
    load_index_fn = load_index_fn or (lambda: load_index_ohlc(
        INDEX_SYMBOL, INDEX_FALLBACK_START, index_end, refresh=refresh, cache_dir=index_cache_dir))

    ds = {"source": None, "fallback_reason": None, "main": None, "index": None,
          "parse": {}, "session_stats": {}, "rolls": [], "roll_info": {}, "validation": {},
          "validation_fails": [], "futures_daily": None, "index_info": {}}

    try:
        index_df, index_info = load_index_fn()
    except Exception as e:
        index_df, index_info = pd.DataFrame(), {"reason": f"exception:{e}"}
    ds["index_info"] = index_info
    if index_df is None or index_df.empty:
        ds["fallback_reason"] = f"加權指數下載失敗({index_info.get('reason')})"
        ds["source_line"] = f"價格來源：無(加權指數也下載失敗：{index_info.get('reason')})，無法執行回測"
        return ds
    ds["index"] = _index_frame(index_df)

    fail = None
    try:
        bars, parse_diag = load_minute_fn()
    except Exception as e:
        bars, parse_diag = None, {"exception": str(e)}
    ds["parse"] = parse_diag
    if bars is None or bars.empty:
        reasons = {k: v.get("reason") for k, v in (parse_diag.get("periods") or {}).items()}
        fail = f"期貨1分鐘資料下載/解析失敗({reasons or parse_diag})"
    else:
        missing = [k for k in FUTURES_PERIOD_KEYS
                   if (parse_diag.get("periods") or {}).get(k, {}).get("reason", "ok") != "ok"]
        if missing:
            ds["parse"]["missing_periods"] = missing
        bars = bars[(bars.index >= pd.Timestamp(FUTURES_START))
                    & (bars.index < pd.Timestamp(FUTURES_END) + pd.Timedelta(days=1))]
        daily, sess = minute_bars_to_daily(bars)
        ds["session_stats"] = sess
        ds["parse"]["minute_rows_in_range"] = int(len(bars))
        ds["parse"]["minute_first"] = str(bars.index.min()) if len(bars) else None
        ds["parse"]["minute_last"] = str(bars.index.max()) if len(bars) else None
        if daily.empty:
            fail = "期貨資料沒有任何日盤(08:45~13:45)K棒"
        else:
            rolls, rinfo = detect_rolls_vs_index(daily, index_df["Close"])
            adj = apply_back_adjustment(daily, rolls)
            ds["rolls"], ds["roll_info"] = rolls, rinfo
            ds["futures_daily"] = adj
            passed, vstats, vfails = validate_against_index(adj, index_df)
            ds["validation"], ds["validation_fails"] = vstats, vfails
            if not passed:
                fail = "；".join(vfails)
            else:
                ds["main"] = adj
                ds["source"] = "futures"
                ds["source_line"] = (
                    "價格來源：台指期1分鐘歷史資料(社群/Google Drive)→日盤日線，2001~2023，"
                    f"已通過加權指數比對驗證；換月以基差跳動啟發式偵測、Panama調整為近似"
                    f"({rinfo['n_significant']}/{rinfo['n_rolls']}次換月價差顯著)")
                return ds

    ds["fallback_reason"] = fail
    ds["source"] = "index"
    ds["main"] = ds["index"]
    ds["source_line"] = f"價格來源：加權指數(期貨資料驗證失敗：原因：{fail})"
    return ds


def _json_default(o):
    if isinstance(o, (pd.Timestamp, datetime.date)):
        return str(o)[:10]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_diagnostics(ds: dict, out_dir: str):
    """寫出診斷檔：data_parse_report.txt、data_validation.json、tx_daily_bars_sample.csv、
    tx_roll_adjustments.csv(期貨模式)。回傳寫出的檔名清單。"""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    lines = [ds.get("source_line", ""), ""]
    lines.append(f"資料模式：{ds.get('source')}")
    if ds.get("fallback_reason"):
        lines.append(f"退回加權指數的原因：{ds['fallback_reason']}")
    lines.append(f"加權指數下載：{ds.get('index_info')}")
    idx = ds.get("index")
    if idx is not None and not idx.empty:
        lines.append(f"加權指數日線：{len(idx)}天，{idx.index.min().date()} ~ {idx.index.max().date()}")
    lines.append("")
    lines.append("== 期貨1分鐘資料解析 ==")
    parse = ds.get("parse") or {}
    for k, v in (parse.get("periods") or {}).items():
        vv = {kk: vv2 for kk, vv2 in v.items() if kk != "raw_preview"}
        lines.append(f"  {k}: {vv}")
        if v.get("raw_preview"):
            lines.append(f"    原始內容預覽：{v['raw_preview']}")
    for k, v in (parse.get("raw_formats") or {}).items():
        lines.append(f"  {k} 欄位偵測：{v}")
    for k in ("missing_periods", "minute_rows_in_range", "minute_first", "minute_last", "exception"):
        if k in parse:
            lines.append(f"  {k}: {parse[k]}")
    lines.append(f"  日盤/夜盤統計(日盤=08:45~13:45，夜盤=15:00~05:00，只用日盤)：{ds.get('session_stats')}")
    fd = ds.get("futures_daily")
    if fd is not None and not fd.empty:
        lines.append(f"  期貨日線：{len(fd)}天，{fd.index.min().date()} ~ {fd.index.max().date()}")
    lines.append("")
    lines.append("== 換月偵測(啟發式，對加權指數基差跳動) ==")
    ri = ds.get("roll_info") or {}
    lines.append(f"  {ri}")
    rolls = ds.get("rolls") or []
    if rolls:
        sig = [r for r in rolls if r["significant"]]
        total_adj = sum(r["gap"] for r in rolls)
        lines.append(f"  換月{len(rolls)}次，顯著{len(sig)}次，累積調整{total_adj:.1f}點"
                     f"(最早K棒被加上的總調整量)")
        if sig:
            gaps = pd.Series([r["gap"] for r in sig])
            lines.append(f"  顯著價差：中位數{gaps.median():.1f}點、最小{gaps.min():.1f}、最大{gaps.max():.1f}")
        rdf = pd.DataFrame([{
            "換月日": str(r["roll_date"])[:10], "結算交易日": str(r["settle_date"])[:10],
            "調整點數(新-舊)": round(r["gap"], 2), "是否顯著": r["significant"],
            "基差變化": round(r["delta_basis"], 2) if np.isfinite(r["delta_basis"]) else None,
        } for r in rolls])
        p = os.path.join(out_dir, "tx_roll_adjustments.csv")
        rdf.to_csv(p, index=False, encoding="utf-8-sig")
        written.append(p)
    lines.append("")
    lines.append("== 驗證(期貨 vs 加權指數) ==")
    lines.append(f"  門檻：OHLC自洽>={MIN_OHLC_CONSISTENT_PCT}%、覆蓋率>={MIN_COVERAGE_PCT}%、"
                 f"日報酬相關>={MIN_RETURN_CORR}、|未調整期貨−指數|/指數 中位數<{MAX_MEDIAN_BASIS_PCT}%")
    lines.append(f"  結果：{ds.get('validation')}")
    lines.append(f"  未通過項目：{ds.get('validation_fails') or '無'}")
    p = os.path.join(out_dir, "data_parse_report.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    written.append(p)

    p = os.path.join(out_dir, "data_validation.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"source": ds.get("source"), "fallback_reason": ds.get("fallback_reason"),
                   "validation": ds.get("validation"), "validation_fails": ds.get("validation_fails"),
                   "roll_info": ds.get("roll_info"), "session_stats": ds.get("session_stats"),
                   "index_info": ds.get("index_info")},
                  f, ensure_ascii=False, indent=2, default=_json_default)
    written.append(p)

    sample_src = fd if fd is not None and not fd.empty else ds.get("main")
    if sample_src is not None and not sample_src.empty:
        rename = {"Open": "開(調整後)", "High": "高(調整後)", "Low": "低(調整後)", "Close": "收(調整後)",
                  "RawOpen": "開(未調整)", "RawHigh": "高(未調整)", "RawLow": "低(未調整)",
                  "RawClose": "收(未調整)", "Close1330": "13:30收(調整後)", "RawClose1330": "13:30收(未調整)",
                  "Offset": "調整量", "RollDay": "換月日", "Volume": "量", "NBars": "分鐘K棒數"}
        n = len(sample_src)
        pick = sorted(set(list(range(min(20, n))) + list(range(max(0, n // 2 - 10), min(n, n // 2 + 10)))
                          + list(range(max(0, n - 20), n))))
        sample = sample_src.iloc[pick].rename(columns=rename)
        if ds.get("index") is not None:
            sample["加權指數收盤"] = ds["index"]["Close"].reindex(sample.index)
        sample.index.name = "日期"
        p = os.path.join(out_dir, "tx_daily_bars_sample.csv")
        sample.to_csv(p, encoding="utf-8-sig")
        written.append(p)
    return written
