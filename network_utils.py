"""
network_utils.py
====================
背景：day_trading_loader.py 實測遇到過 requests.get(timeout=15) 卡住超過10分鐘
(後來甚至到快2小時)沒有返回、也沒有拋出逾時例外的情況——這是requests/urllib3
已知的邊緣案例：連線建立了，但資料傳輸中途卡住不動(伺服器端或中間網路設備的
異常)，某些狀況下底層socket不會正確觸發read timeout，讓requests自己宣稱的
timeout參數形同虛設，整個序列下載流程因此卡死。

這支模組提供 run_with_hard_timeout()：不管底層函式(例如requests.get)自己的
timeout參數有沒有真的生效，都用一個「背景thread + 固定時間硬性放棄」的方式，
確保呼叫端最多等待指定秒數，逾時就強制當作失敗處理、繼續往下跑，不會卡死
整個流程。chip_data_loader.py / day_trading_loader.py 這類序列打TWSE報表
API的模組都用這支模組包一層。

⚠️ 注意：Python沒有安全中止一個正在執行中的thread的方法，這裡的「放棄」
指的是呼叫端不再等待那個thread、視為逾時失敗並往下繼續，那個thread本身
可能還在背景卡著(例如真的卡在底層socket read)，直到它自己完成或程式
整個結束——用daemon=True確保它不會阻止程式正常結束。如果同一個ip/連線
反覆卡住，代表問題出在對方伺服器或網路路徑本身，不是這支模組能解決的，
需要考慮換一個資料來源或請求方式。
"""

import threading


class HardTimeout(Exception):
    """代表run_with_hard_timeout()等待逾時，底層函式沒有在時間內返回。"""


def run_with_hard_timeout(func, args=(), kwargs=None, timeout=30):
    """
    在背景thread執行 func(*args, **kwargs)，最多等待timeout秒。

    正常返回：回傳func的返回值。
    func執行時拋出例外：原封不動重新拋出同一個例外(呼叫端可以正常用
        except SomeSpecificError 接住，跟直接呼叫func()的行為一致)。
    逾時(thread在timeout秒內沒有結束)：拋出HardTimeout，不等待、不阻塞，
        讓呼叫端(通常是重試邏輯)決定要不要重試或放棄這次嘗試。
    """
    kwargs = kwargs or {}
    result_box = {}

    def _target():
        try:
            result_box["value"] = func(*args, **kwargs)
        except BaseException as e:
            result_box["error"] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise HardTimeout(
            f"硬性逾時({timeout}s)：底層函式沒有在時間內返回"
            f"(可能是requests的timeout參數在這次連線沒有真的生效)，強制放棄這次嘗試"
        )

    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("value")
