"""
test_compare_tx_daily.py —— compare_tx_daily.py離線測試：選擇只看訓練期、判定分支、
買進持有參考、輸出檔、端對端(合成期貨/備援指數)、scipy被擋也能跑、不連網。
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import compare_tx_daily as ctd
import tx_daily_data as tdd
from tx_daily_engine import variant_list
from test_tx_daily_data import make_index, make_futures_daily_truth, daily_to_minutes, make_yf_frame

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _st(pf, n, pnl, boot=90.0, excl=1.0):
    return {"profit_factor": pf, "trade_count": n, "total_pnl_ntd": pnl,
            "bootstrap_pct_positive": boot, "pnl_excl_top3_ntd": excl}


class TestSelection:
    def _stats(self, train):
        variants = variant_list()
        stats = {}
        for v in variants:
            pf, n, pnl = train.get(v["name"], (1.0, 50, 0.0))
            stats[(v["name"], "train")] = _st(pf, n, pnl)
            stats[(v["name"], "test")] = _st(0.5, 50, -1.0)
        return variants, stats

    def test_train_only(self):
        variants, stats = self._stats({"MA_CROSS": (1.6, 45, 1000.0)})
        assert ctd.select_variant(variants, stats) == "MA_CROSS"
        # 把測試期數字改成極度偏好另一個變體 → 選擇不變
        stats[("SQZ_RELEASE", "test")] = _st(9.9, 500, 1e9)
        stats[("MA_CROSS", "test")] = _st(0.1, 50, -1e9)
        assert ctd.select_variant(variants, stats) == "MA_CROSS"

    def test_min_trades(self):
        variants, stats = self._stats({"MA_CROSS": (5.0, 39, 1e6), "DONCHIAN": (1.2, 40, 10.0)})
        assert ctd.select_variant(variants, stats) == "DONCHIAN"

    def test_ties(self):
        variants, stats = self._stats({"MA_CROSS": (2.0, 50, 100.0), "SQZ_KDJ": (2.0, 50, 200.0)})
        assert ctd.select_variant(variants, stats) == "SQZ_KDJ"
        variants, stats = self._stats({"MA_CROSS": (2.0, 50, 100.0), "SQZ_KDJ": (2.0, 50, 100.0)})
        assert ctd.select_variant(variants, stats) == "MA_CROSS"  # 列表順序

    def test_none_qualifies(self):
        variants, stats = self._stats({v["name"]: (3.0, 10, 1.0) for v in variant_list()})
        assert ctd.select_variant(variants, stats) is None


class TestVerdict:
    @pytest.mark.parametrize("pf,boot,excl,expected", [
        (1.3, 85.0, 100.0, True),
        (0.99, 85.0, 100.0, False),
        (1.3, 80.0, 100.0, False),
        (1.3, 95.0, 0.0, False),
        (1.3, 95.0, -5.0, False),
    ])
    def test_branches(self, pf, boot, excl, expected):
        passed, checks = ctd.verdict(_st(pf, 50, 1.0, boot, excl))
        assert passed is expected
        assert len(checks) == 3


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth_market():
    index_df = make_index()
    fut, _ = make_futures_daily_truth(index_df)
    bars = daily_to_minutes(fut)
    return index_df, bars


def _patch_sources(monkeypatch, index_df, bars):
    monkeypatch.setattr(tdd, "_yf_download", lambda symbol, start, end: make_yf_frame(index_df))
    monkeypatch.setattr(tdd, "_default_load_minute",
                        lambda refresh, cache_dir: (bars, {"periods": {k: {"reason": "ok", "n_rows": 1}
                                                                       for k in tdd.FUTURES_PERIOD_KEYS}}))


class TestEndToEnd:
    def test_futures_mode(self, synth_market, tmp_path, monkeypatch, capsys):
        index_df, bars = synth_market
        _patch_sources(monkeypatch, index_df, bars)
        out = tmp_path / "res"
        rc = ctd.main(["--results-dir", str(out), "--n-bootstrap", "100", "--index-cache-dir", str(tmp_path / "ic")])
        assert rc == 0
        for f in ("summary.txt", "tx_daily_summary.csv", "tx_daily_yearly.csv", "tx_daily_trades.csv",
                  "data_parse_report.txt", "data_validation.json", "tx_daily_bars_sample.csv",
                  "tx_roll_adjustments.csv"):
            assert (out / f).exists(), f
        summary = (out / "summary.txt").read_text(encoding="utf-8")
        assert summary.splitlines()[0].startswith("價格來源：台指期")
        assert summary.index("預先登記的規則") < summary.index("並排比較")
        assert "判定：" in summary and "兩段都PF>1" in summary and "買進持有(參考)" in summary
        printed = capsys.readouterr().out
        assert printed.index("預先登記的規則") < printed.index("並排比較")
        sm = pd.read_csv(out / "tx_daily_summary.csv", encoding="utf-8-sig")
        assert len(sm) == 9 * 3 and "最大回撤(%資金)" in sm.columns
        tr = pd.read_csv(out / "tx_daily_trades.csv", encoding="utf-8-sig")
        assert {"變體", "期間", "出場原因", "損益(NT$)", "換月成本(NT$)"} <= set(tr.columns)
        # 訓練/測試期的交易落在各自區間；近期段來自加權指數(Offset=0 → 調整後=未調整)
        rec = tr[tr["期間"] == "recent"]
        assert (rec["進場價(調整後)"] == rec["進場價(未調整)"]).all()
        assert (pd.to_datetime(tr[tr["期間"] == "train"]["出場日"]) <= "2012-12-31").all()
        assert (pd.to_datetime(tr[tr["期間"] == "test"]["進場日"]) >= "2013-01-01").all()
        # 買進持有在期貨段有換月成本
        bh = tr[(tr["變體"] == "買進持有(參考)") & (tr["期間"] == "train")]
        assert len(bh) == 1 and bh["換月次數"].iloc[0] > 100 and bh["換月成本(NT$)"].iloc[0] > 0
        yr = pd.read_csv(out / "tx_daily_yearly.csv", encoding="utf-8-sig")
        assert {"變體", "年度", "PF"} <= set(yr.columns)

    def test_fallback_mode(self, synth_market, tmp_path, monkeypatch):
        index_df, bars = synth_market
        bad = bars.copy()
        bad[["Open", "High", "Low", "Close"]] *= 1.5
        _patch_sources(monkeypatch, index_df, bad)
        out = tmp_path / "res"
        rc = ctd.main(["--results-dir", str(out), "--n-bootstrap", "50", "--index-cache-dir", str(tmp_path / "ic"),
                       "--variants", "DONCHIAN", "MA_CROSS+動能門檻", "--contract", "big"])
        assert rc == 0
        summary = (out / "summary.txt").read_text(encoding="utf-8")
        assert summary.splitlines()[0].startswith("價格來源：加權指數(期貨資料驗證失敗：原因")
        assert "整段(訓練/測試/近期)都用加權指數" in summary
        assert "大台TX，每點NT$200" in summary
        sm = pd.read_csv(out / "tx_daily_summary.csv", encoding="utf-8-sig")
        assert set(sm["變體"]) == {"DONCHIAN", "MA_CROSS+動能門檻", "買進持有(參考)"}
        rep = (out / "data_parse_report.txt").read_text(encoding="utf-8")
        assert "退回加權指數的原因" in rep

    def test_no_data_at_all(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tdd, "_yf_download", lambda *a: pd.DataFrame())
        monkeypatch.setattr(tdd.time, "sleep", lambda s: None)
        monkeypatch.setattr(tdd, "_default_load_minute", lambda refresh, cache_dir: (None, {}))
        out = tmp_path / "res"
        rc = ctd.main(["--results-dir", str(out), "--index-cache-dir", str(tmp_path / "ic")])
        assert rc == 1
        assert "回測沒有執行" in (out / "summary.txt").read_text(encoding="utf-8")

    def test_unknown_variant(self, tmp_path):
        assert ctd.main(["--results-dir", str(tmp_path), "--variants", "NOPE"]) == 2


class TestWithoutScipyAndNetwork:
    def test_e2e_scipy_blocked_no_network(self, tmp_path):
        code = textwrap.dedent(f"""
            import sys
            class _Block:
                def find_spec(self, name, path=None, target=None):
                    if name == "scipy" or name.startswith("scipy."):
                        raise ImportError("scipy blocked for test")
                    return None
            sys.meta_path.insert(0, _Block())
            sys.path.insert(0, {REPO_DIR!r})
            import socket
            def _no_net(*a, **k):
                raise RuntimeError("network blocked in test")
            socket.socket.connect = _no_net
            import tx_daily_data as tdd
            import compare_tx_daily as ctd
            from test_tx_daily_data import make_index, make_futures_daily_truth, daily_to_minutes, make_yf_frame
            idx = make_index()
            fut, _ = make_futures_daily_truth(idx)
            bars = daily_to_minutes(fut)
            tdd._yf_download = lambda symbol, start, end: make_yf_frame(idx)
            tdd._default_load_minute = lambda refresh, cache_dir: (bars, {{"periods": {{}}}})
            rc = ctd.main(["--results-dir", {str(tmp_path / 'res')!r}, "--n-bootstrap", "100",
                           "--index-cache-dir", {str(tmp_path / 'ic')!r}])
            assert rc == 0
            assert "scipy" not in sys.modules or sys.modules["scipy"] is None
            print("E2E_OK")
        """)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_DIR)
        assert proc.returncode == 0, proc.stderr[-3000:]
        assert "E2E_OK" in proc.stdout
        assert (tmp_path / "res" / "summary.txt").exists()


class TestWorkflow:
    def test_workflow_file(self):
        from taifex_history_loader import GDRIVE_FILES
        path = os.path.join(REPO_DIR, ".github", "workflows", "tx_daily_backtest.yml")
        txt = open(path, encoding="utf-8").read()
        assert "workflow_dispatch" in txt
        for inp in ("contract:", "commission_per_side:", "refresh_data:"):
            assert inp in txt
        assert "default: mini" in txt and "default: '50'" in txt and "default: false" in txt
        assert 'python-version: "3.10"' in txt
        assert "pytest -q test_tx_daily_engine.py test_tx_daily_data.py test_compare_tx_daily.py" in txt
        assert "python compare_tx_daily.py" in txt
        assert "if: always()" in txt and "path: results_tx_daily/" in txt
        # 快取key包含三段Drive檔案ID(檔案換掉才會重抓)
        key_line = [l for l in txt.splitlines() if l.strip().startswith("key:")][0]
        for k in tdd.FUTURES_PERIOD_KEYS:
            assert GDRIVE_FILES[k]["id"] in key_line, k
        assert "path: taifex_history_cache" in txt
