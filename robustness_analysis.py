"""
robustness_analysis.py
========================
穩健性/敏感度分析工具，拿完整回測跑出來的交易紀錄(trades list)做進一步檢查，
不是回測引擎本身的一部分——目的是回答一個summarize_mr()的單一數字沒辦法回答的問題：

    現在看到的總損益，是這個策略真的有穩定優勢，還是只是「歷史剛好照這個順序發生」
    的運氣，換一種可能的交易順序排列，還會不會一樣賺錢？

做法：Bootstrap重新抽樣——把每筆交易的損益(pnl_ntd)當成一個可以重複抽取的樣本，
隨機抽出跟原本一樣筆數的交易、加總成一個總損益，重複很多次，看這些「平行世界」
的總損益分布長什麼樣子。如果分布大部分都是正的，代表這個策略的優勢比較穩固；
如果分布有相當比例落在負的，代表現在看到的正報酬有不小成分是運氣。

這個做法沒有考慮交易之間的時間相關性(例如連續虧損可能不是隨機獨立發生的，
可能整段時間市場環境就是不利)，是簡化版的穩健性檢查，不是嚴謹的統計推論，
但比只看單一條歷史路徑的數字更誠實。
"""
import numpy as np


def bootstrap_resample_pnl(trades: list, n_resamples: int = 1000, seed: int = 42) -> list:
    """
    對trades的pnl_ntd做放回抽樣(bootstrap)，每次抽出跟原本一樣筆數，加總成一個總損益，
    回傳n_resamples次的結果列表。trades為空時回傳空列表。
    """
    if not trades:
        return []
    pnl_arr = np.array([t["pnl_ntd"] for t in trades], dtype=float)
    n = len(pnl_arr)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_resamples):
        sample = rng.choice(pnl_arr, size=n, replace=True)
        results.append(float(sample.sum()))
    return results


def summarize_bootstrap(results: list) -> dict:
    """
    把bootstrap_resample_pnl()的結果整理成幾個好讀的統計量：
    mean/p5/p95(5%~95%信賴區間)、pct_positive(有多少比例的「平行世界」總損益是正的)。
    pct_positive越接近100%，代表策略的優勢在不同交易順序下都還撐得住；
    如果只有五六成，代表現在看到的正報酬有相當成分是運氣使然。
    """
    if not results:
        return {"mean": 0.0, "p5": 0.0, "p95": 0.0, "pct_positive": 0.0}
    arr = np.array(results, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "pct_positive": float((arr > 0).mean() * 100),
    }


def bootstrap_p_value(results: list) -> float:
    """
    單尾bootstrap p值：n_resamples次「平行世界」重抽樣裡，總損益<=0的比例，等同於
    1 - pct_positive/100。獨立拉出來當一個有明確統計意義的名字，方便在報告裡直接
    標示「這組結果有多少機率其實只是雜訊」。p值越接近0，代表這組策略的優勢在不同
    交易順序下都還站得住；p值如果超過0.2甚至更高，代表現在看到的正報酬有相當機率
    只是運氣，不該當作已經驗證過的優勢看待。
    """
    if not results:
        return 1.0
    arr = np.array(results, dtype=float)
    return float((arr <= 0).mean())


def pnl_excluding_top_n_trades(trades: list, n: int = 3) -> float:
    """
    拿掉獲利最大的n筆交易之後，剩下的總損益。用來檢查策略是不是過度依賴少數幾筆
    極端交易——如果拿掉表現最好的3筆就從賺錢變賠錢，代表現在的正報酬不是穩定、
    可重複的優勢，是少數幾筆運氣好的交易撐出來的。
    """
    if not trades:
        return 0.0
    sorted_pnls = sorted((t["pnl_ntd"] for t in trades), reverse=True)
    return float(sum(sorted_pnls[n:]))
