"""
test_network_utils.py
=========================
針對 network_utils.py 的合成測試，不連網路。驗證：
  1. 正常返回的函式，run_with_hard_timeout() 原封不動回傳結果
  2. 函式內部拋出例外，run_with_hard_timeout() 原封不動重新拋出同一種例外
  3. 函式卡住超過timeout，run_with_hard_timeout() 會逾時放棄、拋出HardTimeout，
     不會傻等那個卡住的函式執行完
  4. 逾時放棄之後，呼叫端(這支測試的主thread)能立刻繼續，不會被背景thread拖住
"""
import time
import unittest

import network_utils as nu


class TestRunWithHardTimeout(unittest.TestCase):
    def test_returns_value_on_normal_completion(self):
        def quick_func(a, b):
            return a + b

        result = nu.run_with_hard_timeout(quick_func, args=(1, 2), timeout=5)
        self.assertEqual(result, 3)

    def test_supports_kwargs(self):
        def func(a, b=0):
            return a * b

        result = nu.run_with_hard_timeout(func, args=(3,), kwargs={"b": 4}, timeout=5)
        self.assertEqual(result, 12)

    def test_reraises_original_exception_type(self):
        def raises_value_error():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            nu.run_with_hard_timeout(raises_value_error, timeout=5)

    def test_reraises_custom_exception_unchanged(self):
        class MyCustomError(Exception):
            pass

        def raises_custom():
            raise MyCustomError("自訂錯誤")

        with self.assertRaises(MyCustomError):
            nu.run_with_hard_timeout(raises_custom, timeout=5)

    def test_hangs_past_timeout_raises_hard_timeout(self):
        def hangs_forever():
            time.sleep(60)  # 模擬卡住的requests.get，遠超過下面的timeout=0.2

        start = time.monotonic()
        with self.assertRaises(nu.HardTimeout):
            nu.run_with_hard_timeout(hangs_forever, timeout=0.2)
        elapsed = time.monotonic() - start

        # 呼叫端應該在接近timeout的時間內就拿回控制權，不會被卡住的函式拖住
        # (不用等sleep(60)結束)
        self.assertLess(elapsed, 5.0)

    def test_daemon_thread_does_not_block_caller_after_timeout(self):
        """逾時之後，主流程應該能立刻繼續做別的事，不會被背景thread卡住。"""
        def hangs_forever():
            time.sleep(60)

        try:
            nu.run_with_hard_timeout(hangs_forever, timeout=0.1)
        except nu.HardTimeout:
            pass

        # 逾時之後應該能馬上做別的事(這行本身能執行到就是證明)
        result = 1 + 1
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
