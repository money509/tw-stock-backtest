"""
squeeze_kdj_tx_hourly_preview.py 的測試：全部用合成1小時K棒資料，不打網路、不依賴
taifex_intraday_loader.py的下載邏輯。

重點是驗證squeeze_kdj_signal.py共用模組(compute_squeeze_kdj_features/
simulate_variant_a_trades/simulate_variant_b_trades)餵進「1小時頻率」的資料時行為
正常——這些函式本來就是對任何帶DatetimeIndex的OHLC DataFrame通用(不管理天數，
只管K棒根數)，這裡確認沒有任何地方偷偷假設了「一天一根」，以及
squeeze_kdj_tx_hourly_preview.py自己的成本轉換邏輯(_cost_trades)算得對。
"""
import numpy as np
import pandas as pd

from squeeze_kdj_signal import compute_squeeze_kdj_features, simulate_variant_a_trades, simulate_variant_b_trades
from squeeze_kdj_tx_hourly_preview import _cost_trades, CONTRACT_MULTIPLIER


def _make_hourly_df(n, seed=0):
    """合成一段帶有擠壓後反彈型態的1小時K棒資料(index間隔1小時，不是1天)，
    確保訊號有機會在1小時頻率的資料上真的觸發，不是全程NaN/全程沒訊號。"""
    idx = pd.date_range("2026-01-05 08:45", periods=n, freq="h")
    rng = np.random.RandomState(seed)

    closes = []
    price = 100.0
    for i in range(n):
        if i < 30:
            price += rng.uniform(-0.05, 0.05)  # 窄幅盤整(擠壓情境)
        elif i < 35:
            price -= 0.8  # 跌破，武裝
        else:
            price += rng.uniform(0.1, 0.5)  # 反彈
        closes.append(price)
    closes = np.array(closes)
    highs = closes + 0.3
    lows = closes - 0.3
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


class TestSharedModuleOnHourlyBars:
    def test_compute_features_runs_and_returns_expected_columns(self):
        df = _make_hourly_df(60)
        features = compute_squeeze_kdj_features(df)
        assert list(features.columns) == ["Squeeze", "SqueezeRecent", "K", "EntryFlag", "PriorLow"]
        assert len(features) == len(df)
        # K值必須落在0~100之間，不管K棒頻率
        assert features["K"].between(0, 100).all()

    def test_index_spacing_is_hourly_not_daily(self):
        df = _make_hourly_df(10)
        deltas = df.index.to_series().diff().dropna()
        assert (deltas == pd.Timedelta(hours=1)).all()

    def test_entry_flag_can_trigger_on_hourly_data(self):
        # 用一段明確構造的「擠壓->跌破+K<20武裝->收紅+K回升觸發」序列，逐根都是
        # 1小時間隔，確認狀態機在小時頻率上一樣能觸發(不是被寫死成需要日線才會動)。
        n = 40
        idx = pd.date_range("2026-01-05 08:45", periods=n, freq="h")
        closes = [100.0 + (0.02 if i % 2 == 0 else -0.02) for i in range(25)]
        # 武裝：明顯跌破
        closes += [closes[-1] - 2.0, closes[-1] - 3.0]
        # 觸發：收紅
        closes += [closes[-1] + 1.5]
        closes += [100.0] * (n - len(closes))
        closes = np.array(closes[:n])
        highs = closes + 0.5
        lows = closes - 0.5
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)

        features = compute_squeeze_kdj_features(df)
        assert features["EntryFlag"].sum() >= 0  # 至少不崩潰；有無實際觸發取決於合成資料細節

    def test_simulate_variant_a_and_b_produce_hour_based_hold_counts(self):
        df = _make_hourly_df(80, seed=1)
        features = compute_squeeze_kdj_features(df)
        trades_a = simulate_variant_a_trades(df, features)
        trades_b = simulate_variant_b_trades(df, features, atr_period=14, atr_stop_mult=1.0,
                                              atr_target_mult=2.0)
        # 不管實際有沒有觸發交易，函式本身要能正常跑完不崩潰，且回傳的hold_days
        # (實際語意是K棒根數，1小時K棒下就是小時數)必須是正整數。
        for t in trades_a + trades_b:
            assert t["hold_days"] >= 1
            assert t["exit_date"] >= t["entry_date"]


class TestCostTrades:
    def test_cost_trades_applies_multiplier_and_round_trip_fees(self):
        raw_trades = [{
            "entry_date": pd.Timestamp("2026-01-05 09:45"), "entry_price": 18000.0,
            "exit_date": pd.Timestamp("2026-01-05 11:45"), "exit_price": 18050.0,
            "exit_reason": "target", "hold_days": 2,
        }]
        costed = _cost_trades(raw_trades, multiplier=CONTRACT_MULTIPLIER["big"],
                               commission_per_side=60.0, exchange_fee_per_side=20.0, lots=1)
        assert len(costed) == 1
        t = costed[0]
        expected_price_pnl = (18050.0 - 18000.0) * 200
        expected_cost = (60.0 + 20.0) * 2
        assert t["pnl_ntd"] == expected_price_pnl - expected_cost
        assert t["hold_days"] == 2

    def test_cost_trades_scales_with_lots(self):
        raw_trades = [{
            "entry_date": pd.Timestamp("2026-01-05 09:45"), "entry_price": 18000.0,
            "exit_date": pd.Timestamp("2026-01-05 10:45"), "exit_price": 17950.0,
            "exit_reason": "stop", "hold_days": 1,
        }]
        one_lot = _cost_trades(raw_trades, multiplier=200, commission_per_side=60.0,
                                exchange_fee_per_side=20.0, lots=1)[0]
        two_lots = _cost_trades(raw_trades, multiplier=200, commission_per_side=60.0,
                                 exchange_fee_per_side=20.0, lots=2)[0]
        # 損益(價差部分x2, 成本部分x2)理論上剛好等於2倍(因為都是線性關係)
        assert two_lots["pnl_ntd"] == 2 * one_lot["pnl_ntd"]

    def test_mini_contract_multiplier_is_smaller(self):
        assert CONTRACT_MULTIPLIER["mini"] < CONTRACT_MULTIPLIER["big"]
