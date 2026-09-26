"""
compare_breakout.py
====================
右側順勢突破策略 - 主執行腳本。

三階段(單一訊號拆解 → 結構門檻變體比較 → 最終組合IS/OOS驗證)的骨架維持不變，
這一輪針對「回測方法論本身」跟「實戰執行細節」做了幾個修正：

1. **修正階段3沒有真正驗證「最佳組合」的問題**：舊版階段3不管前兩階段找出什麼，
   永遠固定跑「全部訊號等權重 + 基本門檻」。這次改成：先依階段1/2在IS的結果，
   自動挑出「PF>1且交易筆數夠多」的訊號、以及總損益最高的門檻，組成「最佳組合」，
   拿這組真正被驗證過的組合去跑IS/OOS/bootstrap，不是驗證一個跟選股結果無關的
   固定基準。舊的「全部等權重+基本門檻」還是會一起跑，當作對照組留著方便比較，
   但報告會明確標示哪一組才是「篩選後的答案」。
2. **ATR倍數敏感度測試(僅移動停利模式)**：在最佳訊號/門檻組合固定之後，額外
   在IS內測一個小網格的(初始停損倍數, 移動停利倍數)組合，選PF最高的一組帶進
   最終驗證，取代原本寫死的1.0倍。刻意只在IS內搜尋、只搜一個很小的網格，
   避免搜索空間太大又製造新的多重比較問題。
3. **跨市場週期驗證(--multi-period-test)**：目前的回測期間大部分落在多頭段，
   對順勢策略先天有利。這個選項會額外抓兩段歷史上市況明顯不同的期間(2022年
   有過一輪顯著修正、2021下半年到2022年初有一段偏盤整)，把最終驗證出的
   組合原封不動地套用過去跑一次，檢查PF/bootstrap正報酬比例是不是一到空頭
   /盤整就明顯轉差——如果是，代表這組優勢的本質比較接近「牛市beta」，不是
   在任何市況都成立的右側交易優勢。
4. **滑價模擬(--slippage-pct)**：進場、以及停損/跳空停損這類市價出場，
   額外套用一個滑價百分比，比原本假設「開盤價/理論停損價就是成交價」更保守寫實。
5. **風險預算式部位大小(--risk-pct-per-trade)**：取代固定口數，每筆交易的口數
   改用「帳戶權益 x 風險比例 ÷ 停損距離」反推，不同波動度的標的不會用同樣口數
   扛到不一樣的風險，帳戶權益也會隨已實現損益動態調整(複利)。只套用在最終驗證
   的組合，不套用在訊號/門檻篩選階段(篩選階段要的是「哪個訊號有預測力」，
   跟部位大小怎麼配是兩件事，混在一起篩選反而讓比較不是蘋果比蘋果)。
6. **多部位並行(--max-concurrent-positions)**：允許同時持有多檔不同標的的部位，
   不再是「一次只能一個部位、資金大部分時間閒置」。同樣只套用在最終驗證階段。
7. **進場時機本身的診斷/優化(這一輪新增，回應「出場已經測過、規模穩健性也測過，
   問題可能出在進場」的判斷)**：
   a. **突破窗口比較(新增的「階段0」)**：原本固定用「20日創新高」當結構門檻，
      這次額外比較5日(提早進場)、60日(季線級別)、250日(年線級別，約52週新高)
      三種替代窗口，一樣只在IS內、用等權重訊號+基本門檻比較，自動選總損益最高
      的窗口，後面所有階段(訊號拆解、門檻比較、ATR網格、最終IS/OOS驗證、跨週期
      驗證)都改用選出的窗口，不再寫死20日。
   b. **趨勢強度門檻(ADX)**：新增「趨勢強度(ADX>=25)」門檻變體，只在ADX(採
      Wilder平滑的近似算法，非精確遞迴公式)達到閾值、代表確實在趨勢中時才放行，
      過濾掉盤整雜訊造成的假突破。
   c. **波動收縮後突破門檻**：新增「波動收縮後突破」門檻變體，只在近期波動度
      (10日平均ATR%)相對前期(20日平均ATR%)明顯收縮(比值<=0.85)時才放行，
      嘗試抓「盤整蓄積後才噴出」比隨機一天創新高更可信的型態。這兩個門檻會
      自動併入既有的門檻比較(階段2)一起排名，不需要額外的CLI參數。
   本輪明確**沒有**動的(留給下一輪，因為需要把預先計算好的指標值一路傳進
   共用的逐日出場處理函式，目前那些函式只看得到原始OHLCV)：均線動態停利、
   Chandelier停利錨點(用高/低點而非收盤價)、拉回測試型進場(狀態機式，跟現在
   單日掃描+進場的架構不同)、類股輪動/市場廣度濾網(需要新的資料來源)、
   分批加碼/減碼。
8. **「只吃魚身」出場配置比較(僅移動停利模式，新增的「階段2.6」)**：回應
   「不追求抱到魚頭魚尾，只要主升段那幾天、不要拖太久」的交易哲學，在ATR倍數
   固定之後，額外測試(a)持有天數上限(`max_hold_days_override`)、(b)移動停利
   延遲啟動(`trailing_activation_days`/`trailing_activation_profit_atr`，進場後
   前幾天或還沒累積一定倍數ATR獲利之前，只用進場當下的固定初始停損，不提前啟動
   移動停利，避免健康拉回被貼太緊的移動停利提早洗出場)、(c)開盤跳空上限
   (`max_gap_pct`，避免追在隔日沖已經拉高、獲利空間被追價盤吃乾抹淨的位置，跟
   原本就有的跳空下限方向相反)這幾種配置的排列組合，自動選總損益最高的一組，
   套用在最終驗證的「最佳組合」(對照組不受影響，維持原本的固定天數+立即啟動)。
   `summarize_mr()`同時新增`avg_hold_days`(平均持有天數)這個統計量，用來檢查
   選出的配置有沒有真的抓到預期的短波段長度，不是憑感覺猜參數。
   實測結果：這6種「只吃魚身」配置全部沒有比「現行(不限天數/立即啟動)」好，最好的一組
   PF反而是原本設定，因為現有的ATR移動停利本身平均就只抱4~5天，額外加限制只會篩掉
   原本就會自然停利的健康交易。這是一個誠實的負向結果，不是「這輪沒做完」。
9. **雙重確認門檻(新增，回應「單一濾網篩出來的候選裡可能還混了邊緣訊號」的假設)**：
   把兩個單獨測試表現最好的門檻(趨勢強度ADX>=25、均線多頭排列/站上季線)同時疊加成
   `+趨勢強度+均線排列雙重確認`、`+趨勢強度+站上季線雙重確認`兩個新的門檻變體，
   自動併入既有的門檻比較(階段2)一起排名，不需要額外的CLI參數，也不需要動引擎——
   單純是GATE_VARIANTS新增兩組kwargs組合。這是用交易筆數換訊號純度的假設，跟出場端
   一樣，是需要驗證、不能先驗認定答案的假設。

用法：
    python3 compare_breakout.py --start 2023-09-15 --end 2026-09-14 --max-stocks 50
    python3 compare_breakout.py --with-chip-confirm --clean-ex-dividend
    python3 compare_breakout.py --use-trailing-stop --atr-stop-mult 1.0
    python3 compare_breakout.py --use-trailing-stop --max-concurrent-positions 3 \\
        --risk-pct-per-trade 0.02 --slippage-pct 0.002 --multi-period-test

⚠️ 誠實揭露：跟 mean_reversion_engine.py 共用的已知限制(結算日近似、大盤氛圍濾網用0050
代理、跌停鎖死/注意股處置股未實作、倖存者偏差、保證金追繳/強制斷頭沒有完整模擬)在這裡
一樣成立，請見 README。
"""
import argparse
import datetime
import os

import pandas as pd

from data_loader import load_price_data
from taifex_universe import STOCK_FUTURES_UNIVERSE
from momentum_breakout_engine import (
    run_momentum_breakout_backtest, BREAKOUT_SIGNAL_NAMES, CHIP_DEPENDENT_SIGNALS,
    precompute_all_breakout_indicators,
)
from mean_reversion_engine import summarize_mr, precompute_regime_series
from robustness_analysis import (
    bootstrap_resample_pnl, summarize_bootstrap, pnl_excluding_top_n_trades, bootstrap_p_value,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_breakout")

MIN_TRADES_FOR_RANKING = 30
MIN_TRADES_FOR_GATE_RANKING = 10
TRAILING_STOP_MAX_HOLD_DAYS = 250  # 工程上的安全上限，不是真正的出場依據，見上方docstring

SIGNAL_LABELS = {
    "score_volume_ratio": "量比",
    "score_rel_strength": "相對大盤強弱",
    "score_rsi_cross": "RSI穿越50(動能轉強)",
    "score_macd": "MACD柱狀圖",
    "score_golden_cross": "均線黃金/死亡交叉",
    "score_price_volume_new_high": "價量同步創高/創低",
    "score_gap_breakout": "跳空突破",
    "score_candle_body": "K棒實體比例",
    "score_foreign_ratio": "外資買超比重",
    "score_trust_ratio": "投信買超比重",
}

HOLD_DAYS_OPTIONS = [("短線5天", 5), ("中期15天", 15)]
TRAILING_STOP_HOLD_DAYS_OPTIONS = [("移動停利(不設固定天數)", TRAILING_STOP_MAX_HOLD_DAYS)]

GATE_VARIANTS = [
    ("基本門檻(站上均線+創新高)", {}),
    ("+站上季線", {"require_above_ma60": True}),
    ("+均線多頭排列", {"require_ma_bullish_alignment": True}),
    ("+爆量(量比>=1.5)", {"min_volume_ratio": 1.5}),
    ("+雙法人同步買超", {"require_dual_institutional_buy": True}),
    ("+趨勢強度(ADX>=25)", {"min_adx": 25.0}),
    ("+波動收縮後突破", {"require_vol_contraction": True}),
    # 雙重確認門檻：把兩個單獨測試表現最好的濾網同時疊加，假設是單一濾網篩出來的
    # 候選裡還混了一些「有趨勢強度但均線排列不乾淨」或「均線排列漂亮但趨勢不夠強」
    # 的邊緣訊號，兩個同時成立才留下，用意是用交易筆數換訊號純度，不是先驗地認為
    # 疊加一定更好——跟這輪出場端「疊加限制反而更差」是同一種需要驗證、不能假設
    # 答案的關係。
    ("+趨勢強度+均線排列雙重確認", {"min_adx": 25.0, "require_ma_bullish_alignment": True}),
    ("+趨勢強度+站上季線雙重確認", {"min_adx": 25.0, "require_above_ma60": True}),
]
GATE_KWARGS_BY_LABEL = dict(GATE_VARIANTS)

# 突破窗口比較：算「有沒有突破」用哪個窗口的前高/前低。20日是原本的設定，
# 其他窗口用來回答「20日創新高這個進場時點是不是太晚/太早」這個問題——
# 5日窗口進場更早、60日/250日(約季線/年線)窗口進場更晚但濾掉更多雜訊。
BREAKOUT_WINDOW_VARIANTS = [
    ("5日創新高(提早進場)", 5),
    ("20日創新高(原本設定)", 20),
    ("60日創新高(季線級別)", 60),
    ("52週創新高(年線級別)", 250),
]

# 移動停利模式下的ATR倍數敏感度網格：(初始停損倍數, 移動停利倍數)。
# 刻意只有6組、只在IS內搜尋，避免搜索空間太大又製造新的多重比較問題。
ATR_SENSITIVITY_GRID = [(0.8, 1.0), (1.0, 1.0), (1.0, 1.5), (1.5, 1.5), (1.5, 2.0), (0.8, 1.5)]

MIN_TRADES_FOR_EXIT_STYLE_RANKING = 10

# 「只吃魚身」出場配置比較(僅移動停利模式使用)：在ATR倍數固定之後，額外測試
# 持有天數上限、移動停利延遲啟動、開盤跳空上限這幾個「進出場配置」的組合，
# 目的是縮短平均持有天數、避免抱到魚尾、也避免進場後健康拉回被貼太緊的移動停利
# 提早洗出場。每個變體的kwargs鍵：
#   max_hold_days_override：不給代表沿用TRAILING_STOP_MAX_HOLD_DAYS(不限天數)
#   trailing_activation_days/trailing_activation_profit_atr：移動停利延遲啟動的
#     天數/獲利ATR倍數門檻，兩者任一達成就啟動(見mean_reversion_engine的判斷)
#   max_gap_pct：開盤跳空幅度上限，超過就放棄進場(避免追在隔日沖已經拉高的位置)
EXIT_STYLE_VARIANTS = [
    ("現行(不限天數/移動停利立即啟動)", {}),
    ("10天強制出場", {"max_hold_days_override": 10}),
    ("移動停利延遲啟動(2天或獲利1.5ATR)", {"trailing_activation_days": 2, "trailing_activation_profit_atr": 1.5}),
    ("10天出場+延遲啟動", {
        "max_hold_days_override": 10, "trailing_activation_days": 2, "trailing_activation_profit_atr": 1.5,
    }),
    ("加開盤跳空上限+2.5%", {"max_gap_pct": 0.025}),
    ("魚身整合版(10天+延遲啟動+跳空上限)", {
        "max_hold_days_override": 10, "trailing_activation_days": 2,
        "trailing_activation_profit_atr": 1.5, "max_gap_pct": 0.025,
    }),
]

# 跨市場週期驗證用的額外歷史窗口：刻意挑跟近期多頭段落明顯不同的市況，
# 檢查最終驗證出的組合是不是只吃到牛市紅利。實際PF/bootstrap結果要看下載回來的
# 真實資料，這裡只是選定「市況應該明顯不同」的日期區間，不對回測結果預設立場。
EXTRA_REGIME_PERIODS = [
    ("2022年修正段", "2022-01-01", "2022-10-31"),
    ("2021下半年~2022年初盤整段", "2021-07-01", "2022-01-15"),
]


def split_is_oos(master_calendar, is_ratio=0.7):
    n = len(master_calendar)
    split_point = int(n * is_ratio)
    return master_calendar[:split_point], master_calendar[split_point:]


def _fmt_pf(pf):
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def _prefer_nonzero_trades(df):
    """
    篩選/排名之前一律先套用：0筆交易的組合(通常是門檻太嚴格、篩不出任何候選)
    完全沒有驗證意義，不該因為它的總損益剛好是0.0、比其他有交易但賠錢的組合
    「數字上更高」就被誤選成贏家。只有在所有候選都是0筆交易時(全部都篩不出
    候選)才會保留0筆的列，讓呼叫方至少拿得到一個可以印出來的結果並示警。
    """
    non_zero = df[df["trade_count"] > 0]
    return non_zero if not non_zero.empty else df


def select_winning_signals(ablation_df, min_trades=MIN_TRADES_FOR_RANKING):
    """
    從單一訊號拆解結果裡，挑出「交易筆數夠多、且PF>1」的訊號組成最終要用的訊號組合。
    如果沒有任何訊號同時滿足這兩個條件(常見於樣本還不夠大、或訊號真的都沒用)，
    退回選總損益前3名，並回傳reliable=False，提醒這組是探索性的、還沒被真正驗證過。
    """
    non_baseline = _prefer_nonzero_trades(ablation_df[ablation_df["signal"] != "__baseline_equal_weight__"])
    reliable_pool = non_baseline[non_baseline["trade_count"] >= min_trades]
    picked = reliable_pool[reliable_pool["profit_factor"] > 1.0]
    if picked.empty:
        fallback = non_baseline.sort_values("total_pnl_ntd", ascending=False).head(3)
        return list(fallback["signal"]), False
    return list(picked["signal"]), True


def select_winning_gate(gate_df, min_trades=MIN_TRADES_FOR_GATE_RANKING):
    """從門檻變體比較結果裡，挑總損益最高、且交易筆數不會太少的那組門檻。"""
    pool = _prefer_nonzero_trades(gate_df)
    reliable = pool[pool["trade_count"] >= min_trades]
    if reliable.empty:
        reliable = pool
    if reliable.empty:
        return GATE_VARIANTS[0]
    best_row = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    label = best_row["gate"]
    return label, GATE_KWARGS_BY_LABEL[label]


def select_winning_breakout_window(window_df, min_trades=MIN_TRADES_FOR_GATE_RANKING):
    """從突破窗口比較結果裡，挑總損益最高、且交易筆數不會太少的那個窗口。"""
    pool = _prefer_nonzero_trades(window_df)
    reliable = pool[pool["trade_count"] >= min_trades]
    if reliable.empty:
        reliable = pool
    if reliable.empty:
        return BREAKOUT_WINDOW_VARIANTS[1]  # 找不到任何可用結果時，退回原本的20日設定
    best_row = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    return best_row["window_label"], int(best_row["breakout_window"])


def run_breakout_window_comparison(price_data, indicators_by_code, regime_series, is_calendar,
                                    starting_capital, hold_days, all_signal_names, extra_kwargs=None):
    """用全部訊號等權重+基本門檻當基準，只換突破窗口(5/20/60/250日)，測哪個進場時點
    比較好。刻意用最單純的訊號/門檻設定，先回答「進場時點本身」這一個變數的問題，
    不跟訊號/門檻篩選的結果交互影響——選出來的窗口會固定下來，之後的訊號拆解、
    門檻比較、ATR網格、最終驗證都套用這個窗口，不會針對每個窗口重跑一次完整流程
    (那樣運算量會爆炸，也會製造更嚴重的多重比較問題)。這是一個簡化假設：不同窗口
    下最有效的訊號/門檻組合理論上可能不同，這裡沒有各自重新篩選，見README的誠實揭露。"""
    extra_kwargs = extra_kwargs or {}
    rows = []
    equal_weights = {name: 1.0 for name in all_signal_names}
    for label, window in BREAKOUT_WINDOW_VARIANTS:
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=hold_days, starting_capital=starting_capital,
            allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0,
            signal_weights=equal_weights, breakout_window=window, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "window_label": label, "breakout_window": window,
            "trade_count": stats["trade_count"], "profit_factor": stats["profit_factor"],
            "total_pnl_ntd": stats["total_pnl_ntd"], "win_rate": stats["win_rate"],
        })
        print(f"  {label} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)
    return pd.DataFrame(rows)


def run_signal_ablation(price_data, indicators_by_code, regime_series, is_calendar,
                         starting_capital, hold_days, has_chip=False, extra_kwargs=None):
    """對 SIGNAL_LABELS 裡每個訊號單獨開啟(權重1.0，其餘0)，用基本門檻在IS跑一次回測。
    這裡固定用lots=2、concurrency=1，不套用風險預算式部位大小/多部位並行——篩選階段
    要回答的是「哪個訊號有預測力」，部位大小怎麼配是另一個議題，混在一起篩選會讓
    「訊號A表現比訊號B好」這個比較摻雜了部位大小的干擾，不是乾淨的訊號對照。"""
    extra_kwargs = extra_kwargs or {}
    rows = []
    signal_names = [s for s in BREAKOUT_SIGNAL_NAMES if has_chip or s not in CHIP_DEPENDENT_SIGNALS]

    common_kwargs = dict(
        price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
        master_calendar=is_calendar, max_hold_days=hold_days, starting_capital=starting_capital,
        allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0,
        **extra_kwargs,
    )

    for i, signal_name in enumerate(signal_names, start=1):
        trades = run_momentum_breakout_backtest(**common_kwargs, signal_weights={signal_name: 1.0})
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "signal": signal_name, "label": SIGNAL_LABELS[signal_name],
            "trade_count": stats["trade_count"], "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "total_pnl_ntd": stats["total_pnl_ntd"], "max_drawdown_ntd": stats["max_drawdown_ntd"],
            "max_consecutive_losses": stats["max_consecutive_losses"],
        })
        print(f"  [{i}/{len(signal_names)}] {SIGNAL_LABELS[signal_name]} "
              f"-> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)

    baseline_trades = run_momentum_breakout_backtest(**common_kwargs, signal_weights={name: 1.0 for name in signal_names})
    baseline_stats = summarize_mr(baseline_trades, starting_capital)
    rows.append({
        "signal": "__baseline_equal_weight__", "label": f"基準({len(signal_names)}訊號等權重)",
        "trade_count": baseline_stats["trade_count"], "win_rate": baseline_stats["win_rate"],
        "profit_factor": baseline_stats["profit_factor"],
        "total_pnl_ntd": baseline_stats["total_pnl_ntd"], "max_drawdown_ntd": baseline_stats["max_drawdown_ntd"],
        "max_consecutive_losses": baseline_stats["max_consecutive_losses"],
    })
    print(f"  [基準] {len(signal_names)}訊號等權重 -> {baseline_stats['trade_count']}筆, "
          f"PF={_fmt_pf(baseline_stats['profit_factor'])}, 勝率={baseline_stats['win_rate']:.1f}%, "
          f"損益={baseline_stats['total_pnl_ntd']:,.0f}", flush=True)

    return pd.DataFrame(rows), signal_names


def run_gate_comparison(price_data, indicators_by_code, regime_series, is_calendar,
                         starting_capital, hold_days, signal_names, has_chip=False, extra_kwargs=None):
    """用等權重訊號當基準，把結構門檻換成不同變體，比較哪種過濾條件篩出的候選比較好。"""
    extra_kwargs = extra_kwargs or {}
    rows = []
    equal_weights = {name: 1.0 for name in signal_names}
    for label, gate_kwargs in GATE_VARIANTS:
        if gate_kwargs.get("require_dual_institutional_buy") and not has_chip:
            continue
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=hold_days, starting_capital=starting_capital,
            allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0,
            signal_weights=equal_weights, **gate_kwargs, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "gate": label, "trade_count": stats["trade_count"], "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "total_pnl_ntd": stats["total_pnl_ntd"], "max_drawdown_ntd": stats["max_drawdown_ntd"],
        })
        print(f"  {label} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)
    return pd.DataFrame(rows)


def run_atr_sensitivity_grid(price_data, indicators_by_code, regime_series, is_calendar,
                              starting_capital, signal_weights, gate_kwargs, extra_kwargs):
    """僅移動停利模式使用：在最佳訊號/門檻組合固定之後，測一個小網格的
    (初始停損倍數, 移動停利倍數)，只在IS內搜尋，回傳完整結果表 + PF最高的一組。"""
    rows = []
    for stop_mult, trail_mult in ATR_SENSITIVITY_GRID:
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=TRAILING_STOP_MAX_HOLD_DAYS,
            starting_capital=starting_capital, allow_short=True, lots=2,
            signal_weights=signal_weights, atr_stop_mult=stop_mult,
            use_trailing_stop=True, trailing_atr_mult=trail_mult,
            **gate_kwargs, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "atr_stop_mult": stop_mult, "trailing_atr_mult": trail_mult,
            "trade_count": stats["trade_count"], "profit_factor": stats["profit_factor"],
            "total_pnl_ntd": stats["total_pnl_ntd"],
        })
        print(f"  停損x{stop_mult} / 移動停利x{trail_mult} -> {stats['trade_count']}筆, "
              f"PF={_fmt_pf(stats['profit_factor'])}, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)

    df = pd.DataFrame(rows)
    pool = _prefer_nonzero_trades(df)
    reliable = pool[pool["trade_count"] >= MIN_TRADES_FOR_RANKING]
    if reliable.empty:
        reliable = pool
    best = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    if best["trade_count"] == 0:
        print("  ⚠️ 整個ATR網格都篩不出任何候選(交易筆數全部是0)，回傳的組合僅供參考，無法驗證")
    return df, (float(best["atr_stop_mult"]), float(best["trailing_atr_mult"]))


def run_exit_style_comparison(price_data, indicators_by_code, regime_series, is_calendar,
                               starting_capital, signal_weights, gate_kwargs, atr_stop_mult, trailing_atr_mult,
                               extra_kwargs):
    """「只吃魚身」出場配置比較(僅移動停利模式使用)：ATR倍數固定之後，另外測試持有天數
    上限/移動停利延遲啟動/開盤跳空上限這幾個進出場配置組合，只在IS內搜尋。額外印出
    平均持有天數，用來檢查有沒有真的抓到預期的短波段長度，不是憑感覺猜參數。"""
    rows = []
    for label, style_kwargs in EXIT_STYLE_VARIANTS:
        style_kwargs = dict(style_kwargs)
        max_hold_days = style_kwargs.pop("max_hold_days_override", TRAILING_STOP_MAX_HOLD_DAYS)
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=max_hold_days,
            starting_capital=starting_capital, allow_short=True, lots=2,
            signal_weights=signal_weights, atr_stop_mult=atr_stop_mult,
            use_trailing_stop=True, trailing_atr_mult=trailing_atr_mult,
            **gate_kwargs, **extra_kwargs, **style_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "exit_style": label, "max_hold_days": max_hold_days,
            "trailing_activation_days": style_kwargs.get("trailing_activation_days", 0),
            "trailing_activation_profit_atr": style_kwargs.get("trailing_activation_profit_atr", 0.0),
            "max_gap_pct": style_kwargs.get("max_gap_pct"),
            "trade_count": stats["trade_count"], "profit_factor": stats["profit_factor"],
            "win_rate": stats["win_rate"], "total_pnl_ntd": stats["total_pnl_ntd"],
            "avg_hold_days": stats["avg_hold_days"],
        })
        print(f"  {label} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
              f"損益={stats['total_pnl_ntd']:,.0f}", flush=True)
    return pd.DataFrame(rows)


def select_winning_exit_style(exit_style_df, min_trades=MIN_TRADES_FOR_EXIT_STYLE_RANKING):
    """從出場配置比較結果裡，挑總損益最高、且交易筆數不會太少的那組配置，回傳
    (label, max_hold_days, extra_kwargs)，extra_kwargs只包含trailing_activation_days/
    trailing_activation_profit_atr/max_gap_pct這三個要餵進run_momentum_breakout_backtest
    的鍵(None的max_gap_pct會被拿掉，避免傳一個沒意義的None覆蓋掉預設值以外的行為——
    其實傳None結果一樣，這裡拿掉純粹讓kwargs乾淨)。"""
    pool = _prefer_nonzero_trades(exit_style_df)
    reliable = pool[pool["trade_count"] >= min_trades]
    if reliable.empty:
        reliable = pool
    if reliable.empty:
        best_row = None
    else:
        best_row = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    if best_row is None:
        label, max_hold_days = EXIT_STYLE_VARIANTS[0][0], TRAILING_STOP_MAX_HOLD_DAYS
        return label, max_hold_days, {}
    extra = {
        "trailing_activation_days": int(best_row["trailing_activation_days"]),
        "trailing_activation_profit_atr": float(best_row["trailing_activation_profit_atr"]),
    }
    if pd.notna(best_row["max_gap_pct"]):
        extra["max_gap_pct"] = float(best_row["max_gap_pct"])
    return best_row["exit_style"], int(best_row["max_hold_days"]), extra


def evaluate_combo(label, price_data, indicators_by_code, regime_series, is_calendar, oos_calendar,
                    starting_capital, hold_days, signal_weights, gate_kwargs, atr_stop_mult,
                    trailing_atr_mult, use_trailing_stop, extra_kwargs, execution_kwargs):
    """把一組(訊號權重, 門檻, ATR倍數)組合，完整跑一次IS/OOS + OOS的bootstrap穩健性檢查，
    回傳一個彙整好的dict，用於階段3輸出跟跨週期驗證共用同一套邏輯。"""
    results = {"label": label}
    common = dict(
        price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
        starting_capital=starting_capital, allow_short=True, atr_stop_mult=atr_stop_mult,
        signal_weights=signal_weights, use_trailing_stop=use_trailing_stop,
        trailing_atr_mult=trailing_atr_mult if use_trailing_stop else None,
        **gate_kwargs, **extra_kwargs, **execution_kwargs,
    )
    if "lots" not in execution_kwargs:
        common["lots"] = 2
    if not use_trailing_stop:
        common["atr_target_mult"] = 2.0

    oos_trades = None
    for split_name, calendar in [("IS", is_calendar), ("OOS", oos_calendar)]:
        trades = run_momentum_breakout_backtest(
            master_calendar=calendar, max_hold_days=hold_days, **common,
        )
        stats = summarize_mr(trades, starting_capital)
        results[split_name] = stats
        if split_name == "OOS":
            oos_trades = trades

    bootstrap_results = bootstrap_resample_pnl(oos_trades or [], n_resamples=1000, seed=42)
    bootstrap_stats = summarize_bootstrap(bootstrap_results)
    results["bootstrap"] = {
        **bootstrap_stats,
        "p_value": bootstrap_p_value(bootstrap_results),
        "pnl_excluding_top3_ntd": pnl_excluding_top_n_trades(oos_trades or [], n=3),
    }
    results["oos_trades"] = oos_trades
    return results


def run_multi_period_validation(universe, index_proxy_code, winning_signal_weights, winning_gate_kwargs,
                                 atr_stop_mult, trailing_atr_mult, use_trailing_stop, hold_days,
                                 starting_capital, extra_kwargs, execution_kwargs, chip_data, refresh):
    """把最終驗證出的組合，原封不動套到幾段跟近期多頭明顯不同市況的歷史期間，
    檢查PF/bootstrap正報酬比例是不是一到空頭/盤整就明顯轉差。每段期間都是獨立
    下載、獨立跑完整回測(不分IS/OOS，因為這裡的目的是「換一段市況」而不是
    「換一段沒看過的時間」)，所以只看單一次的PF+bootstrap，不能跟主回測的
    IS/OOS驗證力度直接類比。"""
    rows = []
    for label, start, end in EXTRA_REGIME_PERIODS:
        print(f"  下載 {label} ({start} ~ {end}) 的資料 ...", flush=True)
        period_price_data = load_price_data(universe, start, end, refresh=refresh)
        if index_proxy_code not in period_price_data:
            print(f"  ⚠️ {label}：大盤代理指標下載失敗，跳過這段期間\n")
            continue
        period_index_df = period_price_data[index_proxy_code]
        if len(period_index_df) < 80:
            print(f"  ⚠️ {label}：資料天數不足({len(period_index_df)}天)，跳過這段期間\n")
            continue

        period_indicators = precompute_all_breakout_indicators(
            period_price_data, universe, index_code=index_proxy_code, chip_data=chip_data,
        )
        period_regime = precompute_regime_series(period_index_df)

        # 注意：lots/risk_pct_per_trade由execution_kwargs提供(main()保證兩者恰好給一個)，
        # 這裡不能再額外寫死lots=2，否則跟execution_kwargs unpack會撞成重複關鍵字參數。
        trades = run_momentum_breakout_backtest(
            price_data=period_price_data, indicators_by_code=period_indicators, regime_series=period_regime,
            master_calendar=period_index_df.index, max_hold_days=hold_days, starting_capital=starting_capital,
            allow_short=True, atr_stop_mult=atr_stop_mult, atr_target_mult=2.0,
            signal_weights=winning_signal_weights, use_trailing_stop=use_trailing_stop,
            trailing_atr_mult=trailing_atr_mult if use_trailing_stop else None,
            **winning_gate_kwargs, **extra_kwargs, **execution_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        boot_results = bootstrap_resample_pnl(trades, n_resamples=1000, seed=42)
        boot_stats = summarize_bootstrap(boot_results)
        rows.append({
            "label": label, "start": start, "end": end,
            "trade_count": stats["trade_count"], "profit_factor": stats["profit_factor"],
            "total_pnl_ntd": stats["total_pnl_ntd"], "pct_positive": boot_stats["pct_positive"],
        })
        print(f"  {label} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"損益={stats['total_pnl_ntd']:,.0f}, bootstrap正報酬比例={boot_stats['pct_positive']:.1f}%\n",
              flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="右側順勢突破策略 - 訊號拆解 + 結構門檻 + 最終組合驗證")
    parser.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=1095)).isoformat())
    parser.add_argument("--end", default=datetime.date.today().isoformat())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--starting-capital", type=float, default=200_000)
    parser.add_argument("--max-stocks", type=int, default=0,
                         help="限制掃描股票數量(0=全部320檔，測試時可以設小一點加快速度)")
    parser.add_argument("--with-chip-confirm", action="store_true",
                         help="額外下載三大法人籌碼資料，把外資/投信買超比重訊號、以及「雙法人同步買超」"
                              "門檻變體也接上測試")
    parser.add_argument("--clean-ex-dividend", action="store_true",
                         help="額外下載除權息日期資料，把除權息當天的股票排除在候選之外，避免機械性跳空"
                              "污染量比/相對大盤強弱/突破前高這些訊號")
    parser.add_argument("--use-trailing-stop", action="store_true",
                         help="出場改用移動停利，不設固定停利目標價、不設固定持有天數(讓獲利奔跑)，"
                              "取代原本短線5天/中期15天的固定天數比較")
    parser.add_argument("--atr-stop-mult", type=float, default=1.0,
                         help="初始停損的ATR倍數(移動停利模式下會被ATR敏感度網格搜出的結果覆蓋)")
    parser.add_argument("--trailing-atr-mult", type=float, default=None,
                         help="移動停利的ATR倍數，不指定時跟--atr-stop-mult用同一個值(只在--use-trailing-stop時"
                              "有意義；會被ATR敏感度網格搜出的結果覆蓋)")
    parser.add_argument("--slippage-pct", type=float, default=0.0,
                         help="進場/停損出場的滑價假設(例如0.002代表0.2%%)，預設0(不模擬滑價)")
    parser.add_argument("--max-concurrent-positions", type=int, default=1,
                         help="最終驗證組合允許同時持有的最大部位數，預設1(維持原本一次一個部位)。"
                              ">1時允許同時持有多檔不同標的")
    parser.add_argument("--risk-pct-per-trade", type=float, default=None,
                         help="最終驗證組合改用風險預算式部位大小：每筆交易風險 = 帳戶權益 x 這個比例"
                              "(例如0.02代表2%%)。不指定時維持固定2口")
    parser.add_argument("--multi-period-test", action="store_true",
                         help="額外把最終驗證出的組合套到2022年修正段、2021下半年~2022年初盤整段"
                              "這兩段跟近期多頭明顯不同市況的歷史期間，檢查是不是只吃到牛市紅利")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    universe = dict(STOCK_FUTURES_UNIVERSE)
    INDEX_PROXY_CODE = "2330"
    if args.max_stocks > 0:
        universe = dict(list(universe.items())[: args.max_stocks])
        if INDEX_PROXY_CODE not in universe:
            universe[INDEX_PROXY_CODE] = STOCK_FUTURES_UNIVERSE[INDEX_PROXY_CODE]

    print(f"下載/讀取歷史資料 ({args.start} ~ {args.end})，共 {len(universe)} 檔標的 ...")
    price_data = load_price_data(universe, args.start, args.end, refresh=args.refresh)
    print(f"成功取得 {len(price_data)} / {len(universe)} 檔股票的資料")

    if INDEX_PROXY_CODE not in price_data:
        raise RuntimeError(f"大盤代理指標 {INDEX_PROXY_CODE} 沒有成功下載，無法繼續。")
    index_df = price_data[INDEX_PROXY_CODE]
    master_calendar = index_df.index
    is_calendar, oos_calendar = split_is_oos(master_calendar)
    print(f"樣本內(IS)天數={len(is_calendar)}，樣本外(OOS)天數={len(oos_calendar)}\n")

    chip_data = None
    if args.with_chip_confirm:
        import chip_data_loader
        print("下載三大法人籌碼資料 ...")
        chip_data = chip_data_loader.load_chip_data(args.start, args.end, universe_codes=set(universe.keys()),
                                                       refresh=args.refresh)
        print(f"籌碼資料涵蓋 {len(chip_data)} 檔股票\n")
    else:
        print("未加 --with-chip-confirm，本次不測試外資/投信買超比重訊號、也不測「雙法人同步買超」門檻\n")

    ex_dividend_dates_by_code = None
    if args.clean_ex_dividend:
        import dividend_data_loader
        print("下載除權息日期資料 ...")
        ex_dividend_dates_by_code = dividend_data_loader.load_dividend_events(
            universe, args.start, args.end, refresh=args.refresh,
        )
        print(f"除權息資料涵蓋 {len(ex_dividend_dates_by_code)} 檔股票\n")
    else:
        print("未加 --clean-ex-dividend，除權息當天的機械性跳空不會被排除\n")

    print("預先計算全市場突破指標 ...")
    indicators_by_code = precompute_all_breakout_indicators(
        price_data, universe, index_code=INDEX_PROXY_CODE, chip_data=chip_data,
    )
    regime_series = precompute_regime_series(index_df)
    print(f"指標預計算完成，涵蓋 {len(indicators_by_code)} 檔股票\n")

    # extra_kwargs：訊號/門檻篩選階段也要套用的執行細節(除權息清洗、滑價)，
    # 這兩個是「回測寫不寫實」的問題，不是「部位大小怎麼配」的問題，篩選階段也該套用。
    extra_kwargs = dict(ex_dividend_dates_by_code=ex_dividend_dates_by_code, slippage_pct=args.slippage_pct)

    # execution_kwargs：只套用在最終驗證組合 + 跨週期驗證，不套用在訊號/門檻篩選階段
    # (見run_signal_ablation的docstring說明)。
    execution_kwargs = dict(max_concurrent_positions=args.max_concurrent_positions)
    if args.risk_pct_per_trade is not None:
        execution_kwargs["risk_pct_per_trade"] = args.risk_pct_per_trade
    else:
        execution_kwargs["lots"] = 2

    if args.use_trailing_stop:
        hold_days_options = TRAILING_STOP_HOLD_DAYS_OPTIONS
        print(f"⚠️ 移動停利模式：出場不設固定天數(工程安全上限={TRAILING_STOP_MAX_HOLD_DAYS}個交易日)、"
              f"不設固定停利目標價，初始停損/移動停利倍數會由階段2.5的ATR敏感度網格搜出\n")
    else:
        hold_days_options = HOLD_DAYS_OPTIONS

    if args.max_concurrent_positions > 1:
        print(f"⚠️ 多部位並行模式：最多同時持有 {args.max_concurrent_positions} 個部位\n")
    if args.risk_pct_per_trade is not None:
        print(f"⚠️ 風險預算式部位大小：每筆交易風險預算 = 帳戶權益 x {args.risk_pct_per_trade:.2%}\n")
    if args.slippage_pct > 0:
        print(f"⚠️ 滑價模擬：進場/停損出場套用 {args.slippage_pct:.2%} 滑價\n")

    all_ablation = {}
    all_gate_comparison = {}
    all_atr_grid = {}
    all_exit_style = {}
    all_combo_results = {}  # {hold_label: {"對照組": {...}, "最佳組合": {...}}}
    all_multi_period = {}
    signal_reliable_flags = {}

    all_breakout_window = {}
    signal_names_for_window = [s for s in BREAKOUT_SIGNAL_NAMES
                                if args.with_chip_confirm or s not in CHIP_DEPENDENT_SIGNALS]

    for hold_label, hold_days in hold_days_options:
        print(f"=== {hold_label} ===")

        print("[階段0] 突破窗口比較 (5/20/60/250日創新高，只在IS內，等權重訊號+基本門檻) ...")
        window_df = run_breakout_window_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, signal_names_for_window, extra_kwargs=extra_kwargs,
        )
        all_breakout_window[hold_label] = window_df
        window_df.to_csv(os.path.join(RESULTS_DIR, f"breakout_window_{hold_label}.csv"),
                          index=False, encoding="utf-8-sig")
        window_label, winning_window = select_winning_breakout_window(window_df)
        print(f"  → 選出突破窗口：{window_label}\n")
        window_extra_kwargs = dict(extra_kwargs, breakout_window=winning_window)

        print("[階段1] 單一訊號拆解 (只在IS內，基本門檻) ...")
        ablation_df, signal_names_used = run_signal_ablation(
            price_data, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, has_chip=args.with_chip_confirm, extra_kwargs=window_extra_kwargs,
        )
        all_ablation[hold_label] = ablation_df
        ablation_df.to_csv(os.path.join(RESULTS_DIR, f"ablation_{hold_label}.csv"),
                            index=False, encoding="utf-8-sig")

        print(f"\n[階段2] 結構門檻變體比較 (只在IS內，等權重訊號) ...")
        gate_df = run_gate_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, signal_names_used, has_chip=args.with_chip_confirm,
            extra_kwargs=window_extra_kwargs,
        )
        all_gate_comparison[hold_label] = gate_df
        gate_df.to_csv(os.path.join(RESULTS_DIR, f"gate_comparison_{hold_label}.csv"),
                        index=False, encoding="utf-8-sig")

        winning_signals, is_reliable = select_winning_signals(ablation_df)
        winning_gate_label, winning_gate_kwargs = select_winning_gate(gate_df)
        signal_reliable_flags[hold_label] = is_reliable
        winning_signal_weights = {name: 1.0 for name in winning_signals}
        print(f"\n  → 依IS結果選出的最佳組合：訊號={[SIGNAL_LABELS[s] for s in winning_signals]}"
              f"{'' if is_reliable else '(⚠️沒有訊號單獨PF>1，退回選總損益前3名，屬探索性選擇)'}，"
              f"門檻={winning_gate_label}")

        atr_stop_mult, trailing_atr_mult = args.atr_stop_mult, args.trailing_atr_mult
        winning_hold_days = hold_days
        winning_exit_extra = {}
        if args.use_trailing_stop:
            print(f"\n[階段2.5] ATR倍數敏感度網格(只在IS內，用上面選出的最佳訊號/門檻組合) ...")
            atr_grid_df, (atr_stop_mult, trailing_atr_mult) = run_atr_sensitivity_grid(
                price_data, indicators_by_code, regime_series, is_calendar,
                args.starting_capital, winning_signal_weights, winning_gate_kwargs, window_extra_kwargs,
            )
            all_atr_grid[hold_label] = atr_grid_df
            print(f"  → 選出：初始停損x{atr_stop_mult}, 移動停利x{trailing_atr_mult}")

            print(f"\n[階段2.6] 出場配置比較(只吃魚身：持有天數上限/延遲啟動/跳空上限，只在IS內) ...")
            exit_style_df = run_exit_style_comparison(
                price_data, indicators_by_code, regime_series, is_calendar,
                args.starting_capital, winning_signal_weights, winning_gate_kwargs,
                atr_stop_mult, trailing_atr_mult, window_extra_kwargs,
            )
            all_exit_style[hold_label] = exit_style_df
            exit_style_df.to_csv(os.path.join(RESULTS_DIR, f"exit_style_{hold_label}.csv"),
                                  index=False, encoding="utf-8-sig")
            exit_style_label, winning_hold_days, winning_exit_extra = select_winning_exit_style(exit_style_df)
            print(f"  → 選出：{exit_style_label}(持有天數上限={winning_hold_days})")

        print(f"\n[階段3] 對照組 vs 最佳組合，IS vs OOS + bootstrap穩健性檢查 ...")
        naive_weights = {name: 1.0 for name in signal_names_used}
        naive_result = evaluate_combo(
            "對照組(全部訊號等權重+基本門檻)", price_data, indicators_by_code, regime_series,
            is_calendar, oos_calendar, args.starting_capital, hold_days, naive_weights, {},
            args.atr_stop_mult, args.trailing_atr_mult, args.use_trailing_stop, window_extra_kwargs,
            {"lots": 2, "max_concurrent_positions": 1},
        )
        winning_result = evaluate_combo(
            f"最佳組合(訊號={winning_gate_label}, 突破窗口={window_label}, 出場={exit_style_label if args.use_trailing_stop else '固定天數'})",
            price_data, indicators_by_code, regime_series, is_calendar, oos_calendar, args.starting_capital,
            winning_hold_days, winning_signal_weights, winning_gate_kwargs, atr_stop_mult, trailing_atr_mult,
            args.use_trailing_stop, dict(window_extra_kwargs, **winning_exit_extra), execution_kwargs,
        )
        all_combo_results[hold_label] = {"對照組": naive_result, "最佳組合": winning_result}

        for result in (naive_result, winning_result):
            print(f"  [{result['label']}]")
            print(f"    IS  -> {result['IS']['trade_count']}筆, PF={_fmt_pf(result['IS']['profit_factor'])}, "
                  f"平均持有{result['IS']['avg_hold_days']:.1f}天, 損益={result['IS']['total_pnl_ntd']:,.0f}")
            print(f"    OOS -> {result['OOS']['trade_count']}筆, PF={_fmt_pf(result['OOS']['profit_factor'])}, "
                  f"平均持有{result['OOS']['avg_hold_days']:.1f}天, 損益={result['OOS']['total_pnl_ntd']:,.0f}")
            b = result["bootstrap"]
            print(f"    bootstrap：正報酬比例={b['pct_positive']:.1f}%, p值={b['p_value']:.3f}, "
                  f"拿掉最大3筆後損益={b['pnl_excluding_top3_ntd']:,.0f}")
            pd.DataFrame(result["oos_trades"]).to_csv(
                os.path.join(RESULTS_DIR, f"trades_OOS_{hold_label}_{result['label'][:10]}.csv"),
                index=False, encoding="utf-8-sig",
            )
        print()

        if args.multi_period_test:
            print(f"[跨市場週期驗證] 把最佳組合套到跟近期多頭明顯不同市況的歷史期間 ...")
            all_multi_period[hold_label] = run_multi_period_validation(
                universe, INDEX_PROXY_CODE, winning_signal_weights, winning_gate_kwargs,
                atr_stop_mult, trailing_atr_mult, args.use_trailing_stop, winning_hold_days,
                args.starting_capital, dict(window_extra_kwargs, **winning_exit_extra),
                execution_kwargs, chip_data, args.refresh,
            )

    summary_lines = [
        "=" * 100,
        f"右側順勢突破策略 訊號拆解 + 結構門檻 + 最終組合驗證",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        f"籌碼相關訊號/門檻：{'有測試' if args.with_chip_confirm else '未測試'}　"
        f"除權息清洗：{'有' if args.clean_ex_dividend else '沒有'}　"
        f"出場模式：{'移動停利' if args.use_trailing_stop else '固定天數+固定ATR停利'}　"
        f"滑價：{args.slippage_pct:.2%}　最大並行部位：{args.max_concurrent_positions}　"
        f"部位大小：{'風險預算式(' + f'{args.risk_pct_per_trade:.2%}' + ')' if args.risk_pct_per_trade else '固定2口'}",
        "=" * 100,
    ]
    for hold_label, _ in hold_days_options:
        summary_lines.append(f"\n--- {hold_label} / 突破窗口比較(IS，等權重訊號+基本門檻) ---")
        window_df = all_breakout_window[hold_label].sort_values("total_pnl_ntd", ascending=False)
        header0 = f"{'突破窗口':<24}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}"
        summary_lines.append(header0)
        summary_lines.append("-" * len(header0))
        for _, r in window_df.iterrows():
            summary_lines.append(
                f"{r['window_label']:<24}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}"
                f"{r['win_rate']:>8.1f}{r['total_pnl_ntd']:>14,.0f}"
            )

        summary_lines.append(f"\n--- {hold_label} / 單一訊號拆解(IS，基本門檻) ---")
        ablation_df = all_ablation[hold_label]
        reliable = ablation_df[ablation_df["trade_count"] >= MIN_TRADES_FOR_RANKING]
        if reliable.empty:
            summary_lines.append(f"⚠️ 沒有任何訊號的交易筆數 >= {MIN_TRADES_FOR_RANKING}，統計上都不夠可靠，全部原樣列出：")
            reliable = ablation_df
        reliable = reliable.sort_values("total_pnl_ntd", ascending=False)
        header = f"{'訊號':<24}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}{'最大連虧':>8}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        for _, r in reliable.iterrows():
            summary_lines.append(
                f"{r['label']:<24}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}{r['win_rate']:>8.1f}"
                f"{r['total_pnl_ntd']:>14,.0f}{r['max_consecutive_losses']:>8}"
            )
        if not signal_reliable_flags[hold_label]:
            summary_lines.append("⚠️ 沒有任何訊號單獨PF>1，下面的「最佳組合」是探索性選擇(總損益前3名)，不是已驗證過的訊號")

        summary_lines.append(f"\n--- {hold_label} / 結構門檻變體比較(IS，等權重訊號) ---")
        gate_df = all_gate_comparison[hold_label].sort_values("total_pnl_ntd", ascending=False)
        header2 = f"{'門檻':<24}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}"
        summary_lines.append(header2)
        summary_lines.append("-" * len(header2))
        for _, r in gate_df.iterrows():
            summary_lines.append(
                f"{r['gate']:<24}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}{r['win_rate']:>8.1f}"
                f"{r['total_pnl_ntd']:>14,.0f}"
            )

        if hold_label in all_atr_grid:
            summary_lines.append(f"\n--- {hold_label} / ATR倍數敏感度網格(IS，最佳訊號/門檻組合) ---")
            atr_df = all_atr_grid[hold_label].sort_values("total_pnl_ntd", ascending=False)
            header3 = f"{'初始停損x':>10}{'移動停利x':>10}{'交易數':>8}{'PF':>8}{'總損益NT$':>14}"
            summary_lines.append(header3)
            summary_lines.append("-" * len(header3))
            for _, r in atr_df.iterrows():
                summary_lines.append(
                    f"{r['atr_stop_mult']:>10.2f}{r['trailing_atr_mult']:>10.2f}{r['trade_count']:>8}"
                    f"{_fmt_pf(r['profit_factor']):>8}{r['total_pnl_ntd']:>14,.0f}"
                )

        if hold_label in all_exit_style:
            summary_lines.append(f"\n--- {hold_label} / 出場配置比較(IS，只吃魚身：天數上限/延遲啟動/跳空上限) ---")
            exit_df = all_exit_style[hold_label].sort_values("total_pnl_ntd", ascending=False)
            header4 = f"{'出場配置':<32}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'平均持有天':>10}{'總損益NT$':>14}"
            summary_lines.append(header4)
            summary_lines.append("-" * len(header4))
            for _, r in exit_df.iterrows():
                summary_lines.append(
                    f"{r['exit_style']:<32}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}"
                    f"{r['win_rate']:>8.1f}{r['avg_hold_days']:>10.1f}{r['total_pnl_ntd']:>14,.0f}"
                )

        summary_lines.append(f"\n--- {hold_label} / 對照組 vs 最佳組合：IS vs OOS + bootstrap ---")
        for combo_key, result in all_combo_results[hold_label].items():
            summary_lines.append(f"\n[{result['label']}]")
            for split_name in ["IS", "OOS"]:
                stats = result[split_name]
                split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
                summary_lines.append(
                    f"  {split_full}: {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
                    f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
                    f"總損益NT${stats['total_pnl_ntd']:,.0f}, "
                    f"最大回撤NT${stats['max_drawdown_ntd']:,.0f}, 最大連續虧損{stats['max_consecutive_losses']}筆"
                )
            b = result["bootstrap"]
            summary_lines.append(
                f"  [穩健性] OOS bootstrap 1000次重抽樣：平均總損益NT${b['mean']:,.0f}，"
                f"5%~95%區間=[NT${b['p5']:,.0f}, NT${b['p95']:,.0f}]，"
                f"正報酬比例={b['pct_positive']:.1f}%，p值={b['p_value']:.3f}，"
                f"拿掉最大3筆交易後總損益NT${b['pnl_excluding_top3_ntd']:,.0f}"
            )

        if hold_label in all_multi_period:
            summary_lines.append(f"\n--- {hold_label} / 跨市場週期驗證(最佳組合套到不同市況的歷史期間) ---")
            for row in all_multi_period[hold_label]:
                summary_lines.append(
                    f"  {row['label']}({row['start']}~{row['end']}): {row['trade_count']}筆, "
                    f"PF={_fmt_pf(row['profit_factor'])}, 損益NT${row['total_pnl_ntd']:,.0f}, "
                    f"bootstrap正報酬比例={row['pct_positive']:.1f}%"
                )
            if not all_multi_period[hold_label]:
                summary_lines.append("  (所有額外期間都下載失敗或資料不足，無法驗證)")

    summary_lines.append(
        "\n判讀方式：先看單一訊號拆解，PF明顯>1且交易筆數夠多的訊號才代表真的有預測力；"
        "「最佳組合」是程式依這個結果自動選出來的，不是隨便挑的——如果標示「探索性選擇」，"
        "代表這次測試裡沒有任何訊號真正驗證過，結果僅供參考。再看OOS的PF、bootstrap正報酬比例、"
        "p值(越接近0代表優勢在不同交易順序下都站得住，超過0.2以上不該當作已驗證的優勢)。"
        "如果有跑跨市場週期驗證，最後看PF/bootstrap正報酬比例是不是一到2022年修正段/盤整段就"
        "明顯轉差——如果是，代表這組優勢比較接近「牛市beta」，不是在任何市況都成立的右側交易優勢。"
        "\n\n新增的「突破窗口比較」是在測試進場時機本身：原本固定用20日創新高當進場門檻，"
        "可能買在短線衝高、快要拉回的位置，而不是真正的趨勢起漲點；改比較5/20/60/250日窗口，"
        "看換一個進場時機是不是能改善績效。門檻變體裡新增的「趨勢強度(ADX)」「波動收縮後突破」"
        "則是分別測試「只在真的有趨勢時才進場」跟「只在盤整蓄積後才進場」這兩種篩選能不能提高"
        "進場品質，篩掉單純雜訊造成的假突破。"
        "\n\n新增的「出場配置比較」是針對「只吃魚身、不拖時間」這個需求：不追求抱到趨勢走完"
        "(那樣容易連末升段的洗盤都吃到)，而是設持有天數上限、以及移動停利延遲啟動(給進場後的"
        "健康拉回一點呼吸空間，不要一進場就被貼太緊的移動停利提早洗出場)、開盤跳空上限(避免"
        "追在隔日沖已經拉高的位置)。這段結果裡的「平均持有天數」是用來檢查有沒有真的抓到預期"
        "的短波段長度——如果選出來的配置平均持有天數還是偏長，代表天數上限或延遲啟動的參數"
        "要再調緊；如果偏短(例如不到2天就出場)，代表移動停利貼得太緊，健康拉回也被洗出場。"
    )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


if __name__ == "__main__":
    main()
