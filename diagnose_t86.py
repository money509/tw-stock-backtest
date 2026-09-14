"""
診斷用小腳本：只測5天資料，序列化(不平行)請求，印出完整的錯誤細節，
用來搞清楚 T86 端點失敗的真正原因(逾時？連線被拒？回傳格式不對？)，
而不是繼續在「執行緒數字」這個可能是錯誤方向的地方瞎猜。
"""
import requests
import time

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

test_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]

for date_str in test_dates:
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    params = {"date": date_str, "selectType": "ALL", "response": "json"}
    print(f"\n=== 測試日期 {date_str} ===", flush=True)
    t0 = time.time()
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        elapsed = time.time() - t0
        print(f"HTTP狀態碼: {resp.status_code}　耗時: {elapsed:.2f}秒", flush=True)
        print(f"回應標頭 Content-Type: {resp.headers.get('Content-Type')}", flush=True)
        print(f"回應內容前200字: {resp.text[:200]}", flush=True)
        try:
            payload = resp.json()
            print(f"JSON解析成功，stat={payload.get('stat')}，資料筆數={len(payload.get('data', []))}", flush=True)
        except Exception as e:
            print(f"JSON解析失敗: {type(e).__name__}: {e}", flush=True)
    except requests.exceptions.Timeout as e:
        elapsed = time.time() - t0
        print(f"逾時 (等了{elapsed:.2f}秒): {e}", flush=True)
    except requests.exceptions.ConnectionError as e:
        elapsed = time.time() - t0
        print(f"連線錯誤 (等了{elapsed:.2f}秒): {e}", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"其他錯誤 (等了{elapsed:.2f}秒): {type(e).__name__}: {e}", flush=True)

    time.sleep(2)

print("\n診斷完成。")
