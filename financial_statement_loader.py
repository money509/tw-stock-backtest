"""
基本面資料：季度財報(EPS、毛利率、ROE)。

⚠️⚠️ 這支模組是這一輪新增的三個資料來源裡信心程度**最低**的一個，明確標記為
**未經驗證(UNVERIFIED)**：這個sandbox環境對外連線到mops.twse.com.tw被proxy
allowlist擋掉，完全沒有機會對真實回應實測過下面這個端點的HTML結構/表單參數
是否正確，也不知道MOPS有沒有改版。**第一次在GitHub Actions真實環境執行
compare_equity_swing.py，才是這支loader真正的第一次整合測試**——如果第一次
真實執行就失敗(端點打不通、表單參數錯誤、HTML表格結構解析不出來)，不代表
設計有問題，代表這裡對端點的假設需要根據真實回應重新調整，這是預期中、
本來就會發生的事，不是意外。

資料來源：公開資訊觀測站(MOPS)「採用IFRSs後」個別財報查詢(t163sb04)，這是一個
HTML查詢表單，不是乾淨的公開JSON API——公開資訊觀測站歷史上一直沒有提供
穩定、有官方文件的季報JSON端點(不像月營收/本益比那樣有對應的開放資料端點)，
這裡採用「模擬表單提交(POST) + pandas.read_html解析回傳HTML表格」的做法，
是MOPS查詢類報表最常見、但同時也最脆弱的存取方式：MOPS改版HTML結構、
調整表單欄位名稱、加驗證碼/防爬蟲機制，都會讓這支程式失效，且不會有明確的
「版本不對」錯誤訊息，很可能只是解析不出預期的表格。

因為信心程度最低，呼叫端(equity_swing_engine.py / compare_equity_swing.py)
把這支loader的每一次呼叫都包在try/except裡：拋出例外、逾時、或回傳空結果，
都不會讓整個回測中斷，只會讓EPS成長率/毛利率/ROE這三個訊號在缺資料的股票/
期間自動退化成中性分數0.0(不計分，不影響其他訊號的排名)，這跟
momentum_breakout_engine.py裡「沒有籌碼資料時score_foreign_ratio/
score_trust_ratio自動變成0分」是完全一樣的優雅降級(graceful degradation)精神。

已知限制：
1. 表單URL/參數(TYPEK/co_id/year/season等鍵名)是根據對MOPS查詢介面的既有認識
   推斷，不是這次實測確認過的。
2. 就算端點打得通，季報通常在季度結束後1.5~2個月才公告(法定申報期限)，用來
   比對時間點一樣要用「公告日」不是「所屬季度」，這裡用「季度結束日+60天」
   粗略近似公告日，實際公告日可能更早或更晚幾天，回測用途的粗略近似，不是
   精確的公告日曆。
3. 欄位名稱(每股盈餘/毛利率/權益報酬率)一律用「欄位名稱包含特定子字串」動態
   比對，不依賴固定欄位位置/表格順序。
4. 只做個股逐檔查詢(MOPS這個查詢介面本身就是逐股設計，不是像T86/BWIBBU那樣
   一次回傳全市場)，所以是「逐股」迴圈，不是「逐日/逐月」迴圈——這代表全市場
   下載這份資料，請求數量會是「股票數 x 季度數」量級，比另外兩支loader慢很多，
   這是MOPS這個查詢介面本身的限制，不是這裡的設計選擇。
"""
import os
import time
import datetime
import io
import requests
import pandas as pd

from network_utils import run_with_hard_timeout, HardTimeout

FINANCIAL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "financial_cache")
HARD_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2  # 逐股查詢成本較高，重試次數比另外兩支loader保守一點
RETRY_BACKOFF_SECONDS = 3
REQUEST_DELAY_SECONDS = 2.0
THREAD_HARD_TIMEOUT_SECONDS = HARD_TIMEOUT_SECONDS + 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/x-www-form-urlencoded",
}

EPS_COL_CANDS = ("每股盈餘", "基本每股盈餘")
GROSS_MARGIN_COL_CANDS = ("毛利率",)
ROE_COL_CANDS = ("權益報酬率", "股東權益報酬率")


class FetchFailed(Exception):
    """代表這次請求本身失敗(逾時/連線錯誤/完全解析不出表格)，由呼叫端決定要不要重試。
    ⚠️ 呼叫端(equity_swing_engine.py)實務上永遠不會讓這個例外(或任何例外)中斷整個
    回測，見模組docstring最上方的優雅降級說明。"""
    pass


def _find_col(columns, candidates):
    for col in columns:
        col_str = str(col).replace(" ", "")
        for cand in candidates:
            if cand in col_str:
                return col
    return None


def _parse_float(s):
    if s is None:
        return float("nan")
    s = str(s).strip().replace(",", "")
    if s in ("", "--", "-", "N/A"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def fetch_quarterly_financials(code: str, year: int, season: int):
    """
    抓取單一股票、單一西元年+季度(1~4)的EPS/毛利率/ROE。
    回傳 dict {"eps": float, "gross_margin_pct": float, "roe_pct": float} 代表有資料；
    回傳 None 代表「確認這一期查無資料」(該公司這一期還沒申報，或代碼不存在)；
    request本身失敗或完全解析不出表格時丟出 FetchFailed。

    ⚠️ 表單參數是根據對MOPS查詢介面的既有認識推斷，未經這個sandbox環境實測，
    見模組docstring最上方的說明。
    """
    roc_year = year - 1911
    url = "https://mops.twse.com.tw/mops/web/ajax_t163sb04"
    payload = {
        "encodeURIComponent": "1", "step": "1", "firstin": "1", "off": "1",
        "queryName": "co_id", "inpuType": "co_id", "TYPEK": "all",
        "isnew": "false", "co_id": code, "year": str(roc_year), "season": f"{season:02d}",
    }
    try:
        resp = requests.post(url, data=payload, headers=HEADERS, timeout=HARD_TIMEOUT_SECONDS)
        resp.encoding = resp.encoding or "utf-8"
        html_text = resp.text
    except Exception as e:
        raise FetchFailed(str(e))

    try:
        tables = pd.read_html(io.StringIO(html_text))
    except Exception:
        return None  # 查無表格，視為這一期確認無資料(公告尚未出爐等常見情況)

    target_df = None
    for tbl in tables:
        cols = list(tbl.columns)
        if _find_col(cols, EPS_COL_CANDS) is not None or _find_col(cols, GROSS_MARGIN_COL_CANDS) is not None:
            target_df = tbl
            break

    if target_df is None or target_df.empty:
        return None

    row = target_df.iloc[0]
    eps_col = _find_col(target_df.columns, EPS_COL_CANDS)
    gm_col = _find_col(target_df.columns, GROSS_MARGIN_COL_CANDS)
    roe_col = _find_col(target_df.columns, ROE_COL_CANDS)

    return {
        "eps": _parse_float(row[eps_col]) if eps_col is not None else float("nan"),
        "gross_margin_pct": _parse_float(row[gm_col]) if gm_col is not None else float("nan"),
        "roe_pct": _parse_float(row[roe_col]) if roe_col is not None else float("nan"),
    }


def _fetch_one_with_retry(code: str, year: int, season: int):
    for attempt in range(MAX_RETRIES + 1):
        try:
            data = run_with_hard_timeout(
                fetch_quarterly_financials, args=(code, year, season), timeout=THREAD_HARD_TIMEOUT_SECONDS)
            return data, True
        except (FetchFailed, HardTimeout):
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            return None, False
    return None, False


def _quarters_between(start: str, end: str) -> list:
    start_d = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    quarters = []
    y, q = start_d.year, (start_d.month - 1) // 3 + 1
    end_y, end_q = end_d.year, (end_d.month - 1) // 3 + 1
    while (y, q) <= (end_y, end_q):
        quarters.append((y, q))
        q += 1
        if q > 4:
            q = 1
            y += 1
    return quarters


def _approx_announcement_date(year: int, season: int) -> pd.Timestamp:
    """季度結束日 + 60天，粗略近似公告日(見模組docstring已知限制第2點)。"""
    quarter_end_month = season * 3
    quarter_end_day = 31 if quarter_end_month in (3, 12) else 30
    if quarter_end_month == 6:
        quarter_end_day = 30
    quarter_end = datetime.date(year, quarter_end_month, quarter_end_day)
    return pd.Timestamp(quarter_end + datetime.timedelta(days=60))


def load_financial_statement_data(start: str, end: str, universe_codes: set, refresh: bool = False) -> dict:
    """
    回傳 {code: DataFrame(index=announcement_date, columns=[eps, gross_margin_pct, roe_pct])}。

    ⚠️ 未經驗證(見模組docstring)。呼叫端務必用try/except包住這個函式的呼叫，
    任何例外都不該讓整個回測中斷——equity_swing_engine.py / compare_equity_swing.py
    已經照這個原則實作，這裡只是再次提醒：這支函式本身**不會**吞掉例外(逾時/
    連線失敗重試用盡後，仍然可能讓某些股票的某些季度直接缺資料，不會拋例外中斷
    整批下載，但呼叫這個函式本身的那一行，呼叫端還是需要包一層try/except，
    因為request.post()這類底層例外理論上仍有極小機率不在run_with_hard_timeout
    的重試範圍內就往外冒出，例如codecs編碼錯誤等)。
    """
    if universe_codes is None:
        raise ValueError("financial_statement_loader 是逐股查詢，universe_codes 不能是 None")

    os.makedirs(FINANCIAL_CACHE_DIR, exist_ok=True)
    quarters = _quarters_between(start, end)
    codes = sorted(universe_codes)

    print(f"季度財報資料(EPS/毛利率/ROE，⚠️未經驗證)：{len(codes)} 檔股票 x {len(quarters)} 季，"
          f"逐股查詢中(每次間隔{REQUEST_DELAY_SECONDS}秒，量大會需要不少時間) ...", flush=True)

    fetched_count = no_data_count = failed_count = 0
    total = len(codes) * len(quarters)
    done_n = 0
    per_stock_records = {}

    for code in codes:
        for (y, season) in quarters:
            done_n += 1
            cache_path = os.path.join(FINANCIAL_CACHE_DIR, f"fs_{code}_{y}Q{season}.csv")
            if not refresh and os.path.exists(cache_path):
                try:
                    cached = pd.read_csv(cache_path)
                    if not cached.empty:
                        row = cached.iloc[0]
                        per_stock_records.setdefault(code, []).append({
                            "date": _approx_announcement_date(y, season),
                            "eps": row["eps"], "gross_margin_pct": row["gross_margin_pct"],
                            "roe_pct": row["roe_pct"],
                        })
                    continue
                except Exception:
                    pass  # 快取檔壞掉，當成沒快取重抓

            data, success = _fetch_one_with_retry(code, y, season)
            if not success:
                failed_count += 1
                continue  # 不寫快取，下次重跑補抓
            if data is None:
                no_data_count += 1
                pd.DataFrame(columns=["eps", "gross_margin_pct", "roe_pct"]).to_csv(cache_path, index=False)
                continue

            fetched_count += 1
            pd.DataFrame([data]).to_csv(cache_path, index=False)
            per_stock_records.setdefault(code, []).append({
                "date": _approx_announcement_date(y, season),
                "eps": data["eps"], "gross_margin_pct": data["gross_margin_pct"], "roe_pct": data["roe_pct"],
            })

            if done_n % 20 == 0 or done_n == total:
                print(f"  進度 {done_n}/{total} "
                      f"(成功{fetched_count} 無資料{no_data_count} 失敗待補{failed_count}) ...", flush=True)
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"季度財報資料下載完成：本次新抓{fetched_count}筆，確認無資料{no_data_count}筆，"
          f"逾時/失敗待補{failed_count}筆{'（下次重跑會自動補抓）' if failed_count else ''}", flush=True)

    result = {}
    for code, records in per_stock_records.items():
        df = pd.DataFrame(records).set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        result[code] = df
    return result
