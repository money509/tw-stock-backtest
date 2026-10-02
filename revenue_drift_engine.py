"""
月營收意外漂移(Monthly Revenue Surprise Drift)事件驅動策略回測引擎。

策略假設(誠實揭露：跟short_reversal_engine.py同樣的態度——這是台灣特有的學術文獻支持
的假說，寫這支引擎的目的就是跑一次這個專案自己的IS/OOS + bootstrap驗證，誠實看結果，
不是先驗認定會賺錢)：

這支引擎要測的是跟short_reversal_engine.py(純價格/技術面訊號)完全不同性質的資料源。
short_reversal已經真的在GitHub Actions跑過：9組(回看窗口x持有天數)全部IS PF<1
(0.65~0.75)，唯一看起來正的OOS結果拿掉前3大交易就轉負，bootstrap正報酬比例77.5%
沒過這個專案自己訂的>80%及格線、p值0.225也不顯著——誠實的結論是「再挑一個價格面訊號
賭賭看」不是好的下一步，應該換一個台灣特有、還沒在這個repo驗證過的資料源。

台灣股市的「每月營收公告」比其他市場常見的「季報」頻率高很多(每月10日前要公告上個月
營收)，有實際的台灣本地學術文獻support「未預期月營收(unexpected revenue)」宣告後
股價有延續效應(例如「台灣股市的未預期月營收：綜合分析」、「上市櫃公司每月營收宣告對
股價之影響」這類研究)，重點在「未預期」——不是「營收成長本身」，這是這支引擎跟
equity_swing_engine.py用法最關鍵的差異，見下方「跟equity_swing_engine.py的關鍵
設計差異」。

資料來源：重用 revenue_data_loader.py (已在equity_swing的GitHub Actions真實環境
驗證過會成功載入revenue_yoy_pct/revenue_mom_pct，index是「近似公告可見日」——
所屬月份的次月10日，刻意設計成這樣來避免look-ahead bias，見該模組docstring)，
不重新寫一次月營收爬蟲。

跟equity_swing_engine.py的關鍵設計差異(這是這支引擎存在的理由，不是重複造輪子)：
equity_swing_engine.py把revenue_yoy_pct/revenue_mom_pct當成「緩慢變化的月頻排名
訊號」，用reindex(method="ffill")把它forward-fill成每個交易日都看得到的分數，
持有週數是以「週」為單位(min_hold_weeks/max_hold_weeks)，這正是之前在這個對話裡
已經得到「基本面不適合短波段」結論的原因——forward-fill後的訊號在宣告後好幾週都長
一樣，拿來做3-10個交易日的短波段進出沒有道理(訊號本身的「新鮮度」只有宣告當下最高)。

這支引擎反過來：把每一次月營收公告當成一個「離散事件」，只在事件發生的那個時間點附近
做一次固定天數的短線進出，不是整個月都掛著這個訊號找進場點。多數交易日完全不會有任何
候選(因為多數交易日沒有任何公司在那天公告月營收)，只有公告日叢集的那幾天(每月10日
前後)才會真正產生候選名單——這跟short_reversal_engine.py「每天都重新掃描全市場」的
設計完全不同，見run_revenue_drift_backtest()的事件驅動迴圈結構。

訊號定義(這是這支引擎真正要花心思的地方，不是直接把revenue_yoy_pct原始值拿來排名)：
「意外(surprise)」= 這個月的revenue_yoy_pct 減去「這檔股票自己過去6個月(不含當月)
的revenue_yoy_pct平均」——不是看YoY成長率本身的絕對高低(那是equity_swing在做的事)，
而是看「這個月相對於自己最近的成長慣性，是超出預期還是低於預期」，這比較貼近文獻討論
的「未預期(unexpected)」成分，而不是「長期成長水準」本身(見precompute_revenue_surprise()
的docstring)。計算只用「嚴格早於當月(含)」的歷史，要求至少有6筆之前的公告紀錄才有資格
(不足6筆的股票在這個時間點直接跳過，不勉強拿不夠的資料算平均)。

排名方式：在「同一個公告可見日」這個集合裡(台灣月營收公告很密集地卡在每月10日附近，
同一天通常有幾十家公司一起公告)，用percentile_score()相對排名(跟short_reversal_
engine.py/overnight_momentum_engine.percentile_score()一致的「相對排名比絕對門檻
穩定」理由)——意外最高的前top_n名當多方候選(營收超預期)，意外最低(相對自己趨勢是
營收不如預期/miss)的前top_n名當空方候選。

標的池：沿用taifex_universe.STOCK_FUTURES_UNIVERSE，理由跟short_reversal_engine.py
一樣——放空一個「營收不如預期」的股票，沒有股票期貨(不用融券，不怕券源/軋空)幾乎不可能
真的做到，而且這個標的池本身已經是期交所篩過的流動性子集合。

進場時機：嚴格晚於公告可見日——用entry_date = master_calendar裡「嚴格晚於公告可見日」
的第一個交易日，絕不會跟公告可見日同一天進場(公告可見日本身已經是「約略已公開」的近似
值，在它之後的下一個交易日開盤才進場，是這個repo一貫對gap/時間風控的保守處理方式)。

出場機制：固定持有天數(5或10個交易日，這是比較對象，見HOLD_DAYS_OPTIONS——這個策略
本質上是較低頻率的事件，不像short_reversal還測3日，5/10日的比較才有意義)當主要出場
依據，底下疊一層ATR保護性停損當安全網，跟short_reversal_engine.py完全同一套雙重機制
(重用mean_reversion_engine._process_mr_day()/短反轉的try_enter_short_reversal())。

這支引擎重用的既有機制(不重新發明)：
- mean_reversion_engine.py 的 compute_atr_correct/is_near_settlement/_lookup_prior_row/
  _process_mr_day/summarize_mr/DEFAULT_MARGIN_CAP_RATIO
- overnight_momentum_engine.py 的 percentile_score()/MIN_TURNOVER_VALUE
- short_reversal_engine.py 的 try_enter_short_reversal()(候選進場的跳空風控/保證金
  上限/滑價邏輯完全一樣，候選dict的欄位形狀{"code","side","score","c_prev","atr"}
  剛好跟這支引擎產生的候選一致，直接重用，不重寫一份幾乎一樣的程式碼)、
  DEFAULT_ATR_STOP_MULT、多部位day-by-day走訪的迴圈結構(run_short_reversal_backtest)

⚠️ 已知簡化(誠實列出，跟這個repo其他引擎同樣的態度，不是刻意隱瞞)：
1. 沒有大盤氛圍濾網——理由跟short_reversal_engine.py一樣，這是橫斷面相對排名策略，
   不是順勢/逆勢策略，這一輪先不加。
2. 公告可見日本身是revenue_data_loader.py的「次月10日」近似值，不是每家公司真正
   公告的那一天(有些公司提早、有些壓到月底前最後幾天)，這個近似值本身的誤差會直接
   傳導到這支引擎的entry_date——如果近似值偏晚，代表部分個股其實更早就能進場而少賺
   漂移的前段，如果偏早，極端情況下可能在真正公告前就進場(look-ahead)；revenue_data_
   loader.py模組docstring已經誠實說明這是推斷出來的格式，還沒有机会在真實環境裡逐家
   驗證公告日準確度。
3. 「過去6個月平均」用的是這檔股票「過去6筆有資料的公告紀錄」，不是嚴格的「過去6個
   日曆月」——如果某個月剛好沒有公告(停業/下市/當月漏公告，revenue_data_loader.py
   docstring第5點已經說明這種情況會讓該月整個不出現在彙總表)，這裡會往更早的紀錄補，
   不會因為中間有缺口就讓訊號整個失效，但也因此「6個月」在有缺口的股票身上實際跨的
   日曆天數會比沒缺口的股票長，這裡沒有特別處理這個差異。
4. 同一個公告可見日如果剛好只有1、2家公司公告(例如罕見離群的公告日)，percentile_score()
   在樣本數很小時排名意義有限(甚至只有1檔時排名恆為滿分)，這裡沒有額外設下「同一天至少
   N家公司才排名」的門檻，統計上的穩健性要靠後面bootstrap驗證去抓，不在這一層額外過濾。
"""
import pandas as pd
import numpy as np

from mean_reversion_engine import (
    compute_atr_correct, is_near_settlement, _lookup_prior_row, _process_mr_day,
    DEFAULT_MARGIN_CAP_RATIO,
)
from overnight_momentum_engine import percentile_score, MIN_TURNOVER_VALUE
from short_reversal_engine import try_enter_short_reversal, DEFAULT_ATR_STOP_MULT

# 固定持有天數的比較對象：事件頻率比short_reversal低(一檔股票一個月通常只公告一次)，
# 不像short_reversal還測3日(太短，雜訊會蓋過事件本身的訊號)，5日(約一週)/10日(約兩週)
# 才是對這個事件頻率有意義的比較，見compare_revenue_drift.py階段1。
HOLD_DAYS_OPTIONS = (5, 10)

DEFAULT_TOP_N = 5
TRAILING_MONTHS = 6  # 「過去幾個月平均」的視窗長度，用來算「意外」相對自己趨勢的偏離
MIN_PRICE_WARMUP_DAYS = 20  # 價格面指標(ATR等)最少需要的暖身交易日數，跟其他引擎同精神


def precompute_revenue_drift_price_indicators(price_data: dict, universe: dict) -> dict:
    """
    針對universe裡每檔股票，只算一次ATR(安全網停損用)、成交金額(流動性門檻用)。
    跟short_reversal_engine.precompute_reversal_indicators()同樣的「day-by-day迴圈
    之前先查表算好」精神，但這支引擎不需要Return_N這些回看窗口報酬率欄位(訊號來自
    月營收意外，不是價格動能)，所以這裡只留真正會用到的欄位，不是整份重用short_
    reversal的函式而硬是多算一堆用不到的東西。
    """
    result = {}
    for code in universe:
        df = price_data.get(code)
        if df is None:
            continue
        close = df["Close"]
        volume = df["Volume"]
        atr = compute_atr_correct(df)
        turnover_value = close * volume
        result[code] = pd.DataFrame(
            {"Close": close, "ATR": atr, "TurnoverValue": turnover_value}, index=df.index
        )
    return result


def precompute_revenue_surprise(revenue_df: pd.DataFrame, trailing_months: int = TRAILING_MONTHS) -> pd.DataFrame:
    """
    單一股票的「月營收意外(surprise)」序列。

    surprise[t] = revenue_yoy_pct[t] - mean(revenue_yoy_pct[t-trailing_months : t])
    (後面這個平均嚴格只用「t之前」的trailing_months筆紀錄，不含t當月本身)——用
    .shift(1).rolling(trailing_months, min_periods=trailing_months).mean()實作：
    shift(1)先把當月本身排除在視窗外，min_periods=trailing_months強制要求視窗內
    真的有trailing_months筆「非NaN」的歷史紀錄才算得出平均，不足的月份(包含這檔
    股票剛開始有資料的前幾個月)平均值是NaN，算出來的surprise自然也是NaN——這就是
    「冷啟動安全機制」：呼叫端(build_revenue_drift_events)只收surprise非NaN的紀錄，
    不需要另外寫一道「至少6筆歷史」的判斷，NaN本身就已經代表「資格不足」。

    這個寫法天生不會偷看未來：shift(1)把當月排除，rolling只往回看，附加在revenue_df
    後面的任何未來紀錄都不會改變這一列算出來的值(見test_revenue_drift_engine.py對
    這點的直接驗證)。
    """
    yoy = revenue_df["revenue_yoy_pct"]
    trailing_avg = yoy.shift(1).rolling(trailing_months, min_periods=trailing_months).mean()
    surprise = yoy - trailing_avg
    return pd.DataFrame(
        {"revenue_yoy_pct": yoy, "trailing_avg_yoy_pct": trailing_avg, "surprise": surprise},
        index=revenue_df.index,
    )


def precompute_all_revenue_surprise(revenue_data: dict, trailing_months: int = TRAILING_MONTHS) -> dict:
    """對revenue_data裡每檔股票各算一次surprise序列，回傳{code: surprise_df}。"""
    result = {}
    for code, revenue_df in revenue_data.items():
        if revenue_df is None or revenue_df.empty:
            continue
        result[code] = precompute_revenue_surprise(revenue_df, trailing_months=trailing_months)
    return result


def build_revenue_drift_events(surprise_by_code: dict) -> dict:
    """
    把「每檔股票各自的surprise序列」轉成「以公告可見日為key」的事件表：
    {announcement_date: [{"code": ..., "surprise": ...}, ...]}。

    只收surprise非NaN的紀錄(冷啟動不足6個月歷史的紀錄，NaN，天然被排除，見
    precompute_revenue_surprise()說明)。revenue_data_loader.py對同一個(年,月)算出
    來的announcement_date是同一個pd.Timestamp物件值(「次月10日」不因公司而異)，
    所以不同股票只要是同一個會計月份，這裡的key天生就會對齊成同一天——這正是
    「台灣月營收公告叢集在每月10日附近」這件事在資料結構上的體現，不需要額外的
    日期模糊比對。
    """
    events_by_date = {}
    for code, surprise_df in surprise_by_code.items():
        valid = surprise_df[surprise_df["surprise"].notna()]
        for ann_date, row in valid.iterrows():
            events_by_date.setdefault(ann_date, []).append(
                {"code": code, "surprise": float(row["surprise"])}
            )
    return events_by_date


def build_entry_date_index(events_by_date: dict, master_calendar: pd.DatetimeIndex) -> dict:
    """
    把每個「公告可見日」轉成「嚴格晚於它的下一個交易日」(這支引擎的entry_date)，
    回傳{entry_date: [announcement_date, ...]}(方向反過來，方便day-by-day迴圈用
    「今天是不是某個事件的進場日」直接查表，不用每天重新算一次)。

    用master_calendar.searchsorted(announcement_date, side="right")：不管
    announcement_date本身是不是交易日(10日常常剛好是週末)，都會找到「嚴格大於」它
    的第一個交易日位置，天生滿足「進場嚴格晚於公告可見日，絕不同一天」的要求。
    如果公告可見日已經晚於master_calendar涵蓋範圍的最後一天(資料還沒更新到那麼晚)，
    這個事件就沒有對應的進場日，直接捨棄(不是錯誤，只是這個事件目前還進不了場)。
    """
    entry_date_by_announcement = {}
    for ann_date in events_by_date:
        pos = master_calendar.searchsorted(ann_date, side="right")
        if pos >= len(master_calendar):
            continue
        entry_date_by_announcement[ann_date] = master_calendar[pos]

    events_by_entry_date = {}
    for ann_date, entry_date in entry_date_by_announcement.items():
        events_by_entry_date.setdefault(entry_date, []).append(ann_date)
    return events_by_entry_date


def scan_revenue_drift_candidates(events_by_date: dict, announcement_date, price_indicators_by_code: dict,
                                   entry_date, excluded_codes: set, top_n: int = DEFAULT_TOP_N,
                                   min_turnover_value: float = MIN_TURNOVER_VALUE,
                                   allow_short: bool = True) -> list:
    """
    針對「單一個公告可見日」的事件集合，掃出這一批候選裡的多方/空方名單。

    c_prev(進場跳空風控用的參考價)/atr(安全網停損用)/流動性門檻，都用
    _lookup_prior_row(price_indicators_by_code[code], entry_date)查「嚴格早於
    entry_date」的最後一列收盤價/ATR/成交金額——因為entry_date已經保證晚於
    announcement_date(見build_entry_date_index)，這裡查到的就是「進場前最後一個
    已知交易日」的資料，跟short_reversal_engine.scan_short_reversal_candidates()
    用_lookup_prior_row()防止偷看未來資料的寫法一致。

    排名方式：percentile_score()相對排名，意外最高(score_long)的前top_n名當多方
    候選，意外最低/miss最嚴重(score_short)的前top_n名當空方候選——跟short_reversal_
    engine.py的score_long/score_short寫法同樣精神，只是這裡排名依據是「月營收意外」
    不是「N日報酬率」。
    """
    events = events_by_date.get(announcement_date, [])
    if not events:
        return []

    rows = []
    for ev in events:
        code = ev["code"]
        if code in excluded_codes:
            continue
        ind_df = price_indicators_by_code.get(code)
        if ind_df is None:
            continue
        row, pos = _lookup_prior_row(ind_df, entry_date)
        if row is None or pos < MIN_PRICE_WARMUP_DAYS:
            continue

        atr = row["ATR"]
        turnover = row["TurnoverValue"]
        close = row["Close"]
        if pd.isna(atr) or atr <= 0 or pd.isna(turnover) or pd.isna(close):
            continue
        if turnover < min_turnover_value:
            continue

        rows.append({"code": code, "close": close, "atr": atr, "surprise": ev["surprise"]})

    if not rows:
        return []

    df = pd.DataFrame(rows)
    # 多方：意外越高(營收超預期)分數越高；空方：意外越低(miss最嚴重)分數越高。
    df["score_long"] = percentile_score(df["surprise"], higher_is_better=True)
    df["score_short"] = percentile_score(df["surprise"], higher_is_better=False)

    # 去重複指派：同一支股票不能同時當多方又當空方候選。當一個公告可見日只有少數幾家
    # 公司公告(事件天數很小)時，呼叫端的top_n=max(top_n, slots_available)可能把
    # top_n撐大到超過當天候選總數，長/短名單(head(top_n))會整批重疊——若不去重，
    # 最後實際進場方向會變成由候選排列順序(多方先、空方後，見後面的候選組裝順序)決定，
    # 不是由訊號強弱決定，這是真正的正確性問題，不只是理論上的edge case(revenue_drift
    # 每個事件日的候選池本來就比short_reversal_engine每天全市場掃描小很多，撞到的機率
    # 高很多)。
    # 修法：每檔股票只能依「意外是偏高還是偏低」二選一歸類(score_long跟score_short是
    # 同一個surprise值的對稱百分位排名，score_long >= score_short代表這檔股票的意外
    # 落在當天這批的中位數以上，歸多方；反之歸空方)，分類後的top_n名次只在「已經二選一
    # 歸類好」的子集合裡排，不會再有同一支股票同時出現在兩份名單的狀況。
    df["side"] = np.where(df["score_long"] >= df["score_short"], "long", "short")

    candidates = []
    long_df = df[df["side"] == "long"].sort_values("score_long", ascending=False).head(top_n)
    for _, r in long_df.iterrows():
        candidates.append({
            "code": r["code"], "side": "long", "score": float(r["score_long"]),
            "c_prev": float(r["close"]), "atr": float(r["atr"]),
        })

    if allow_short:
        short_df = df[df["side"] == "short"].sort_values("score_short", ascending=False).head(top_n)
        for _, r in short_df.iterrows():
            candidates.append({
                "code": r["code"], "side": "short", "score": float(r["score_short"]),
                "c_prev": float(r["close"]), "atr": float(r["atr"]),
            })

    return candidates


def run_revenue_drift_backtest(price_data: dict, price_indicators_by_code: dict, events_by_date: dict,
                                events_by_entry_date: dict, master_calendar: pd.DatetimeIndex,
                                max_hold_days: int, starting_capital: float, allow_short: bool = True,
                                lots: int = 2, atr_stop_mult: float = DEFAULT_ATR_STOP_MULT,
                                top_n: int = DEFAULT_TOP_N, min_turnover_value: float = MIN_TURNOVER_VALUE,
                                slippage_pct: float = 0.0, max_concurrent_positions: int = 2 * DEFAULT_TOP_N,
                                total_margin_cap_ratio: float = None, max_gap_pct: float = None):
    """
    完整day-by-day walk-forward模擬，但跟short_reversal_engine.run_short_reversal_
    backtest()最大的不同是「候選產生」這一步是事件驅動的：只有當天剛好是某個公告可見日
    的entry_date(見build_entry_date_index)，才會呼叫scan_revenue_drift_candidates()
    找候選；絕大多數交易日(events_by_entry_date裡沒有這一天)完全不會產生新候選，直接
    跳到「處理既有部位出場」這一步——不是每天都重新掃描全市場，這是月營收公告本身的
    發生頻率決定的(一檔股票一個月通常只公告一次)，不是工程上的偷懶。

    出場判定/強制平倉/停損冷卻期/多部位並行，整段重用mean_reversion_engine.
    _process_mr_day()跟short_reversal_engine.try_enter_short_reversal()，跟
    run_short_reversal_backtest()同樣的理由：這是橫斷面策略(同時買一籃子意外最高的、
    空一籃子miss最嚴重的)，需要多部位並行的迴圈結構，不是單一部位版本。
    """
    trades = []
    cooldown_until = {}
    open_positions = []

    effective_total_margin_cap_ratio = total_margin_cap_ratio
    if max_concurrent_positions > 1 and effective_total_margin_cap_ratio is None:
        effective_total_margin_cap_ratio = min(DEFAULT_MARGIN_CAP_RATIO * max_concurrent_positions, 0.9)

    for date in master_calendar:
        # 1) 先處理所有既有部位的出場判定/強制平倉(可能更新cooldown_until)
        still_open = []
        for position in open_positions:
            df = price_data.get(position["code"])
            if df is None or date not in df.index:
                still_open.append(position)
                continue
            if date != position["entry_date"]:
                position["hold_days"] += 1
            row = df.loc[date]
            updated = _process_mr_day(position, row, date, trades, max_hold_days, cooldown_until,
                                       slippage_pct=slippage_pct)
            if updated is not None:
                still_open.append(updated)
        open_positions = still_open

        # 2) 事件驅動：只有今天剛好是某個公告可見日的entry_date，才掃候選、補進新部位
        announcement_dates = events_by_entry_date.get(date)
        held_codes = {p["code"] for p in open_positions}
        excluded_codes = {c for c, until in cooldown_until.items() if date < until} | held_codes

        slots_available = max_concurrent_positions - len(open_positions)
        if announcement_dates and slots_available > 0 and not is_near_settlement(date, days_before=2):
            used_margin = sum(p["margin_used"] for p in open_positions)

            candidates = []
            for ann_date in announcement_dates:
                candidates.extend(scan_revenue_drift_candidates(
                    events_by_date, ann_date, price_indicators_by_code, date, excluded_codes,
                    top_n=max(top_n, slots_available), min_turnover_value=min_turnover_value,
                    allow_short=allow_short,
                ))

            while slots_available > 0 and candidates:
                new_position = try_enter_short_reversal(
                    price_data, candidates, date, starting_capital, lots,
                    atr_stop_mult=atr_stop_mult, slippage_pct=slippage_pct,
                    used_margin=used_margin, total_margin_cap_ratio=effective_total_margin_cap_ratio,
                    max_gap_pct=max_gap_pct,
                )
                if new_position is None:
                    break

                used_margin += new_position["margin_used"]
                slots_available -= 1
                candidates = [c for c in candidates if c["code"] != new_position["code"]]

                # 進場當天立刻檢查一次出場(跟short_reversal_engine一致：跳空穿越停損
                # 可能當天就出場)
                row = price_data[new_position["code"]].loc[date]
                updated = _process_mr_day(new_position, row, date, trades, max_hold_days, cooldown_until,
                                           slippage_pct=slippage_pct)
                if updated is not None:
                    open_positions.append(updated)
                else:
                    used_margin -= new_position["margin_used"]

    return trades
