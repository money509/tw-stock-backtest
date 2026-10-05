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
10. **「專業投資經理」設計：相對排名取代絕對值硬門檻(新增的「階段0.7」)**：回應
    「不追求「準」，追求「賠得少、賺得多、選得穩」」這個核心哲學。前幾輪測試發現：
    既有的score_*訊號全部都是相對排名(percentile_score)，在標的池從50~100檔擴大到
    全市場320檔時相對穩定；但「收盤價>突破窗口高/低點」這個結構性硬門檻是絕對值判定，
    同樣的擴大測試中表現大幅震盪(甚至整套邏輯站不住)。這一輪新增：
    a. **score_breakout_strength**：突破強度改成連續分數——「離突破窗口高/低點還有幾倍
       ATR距離」，再用既有的percentile_score()相對排名，取代非黑即白的「有沒有突破」。
    b. **require_hard_breakout參數**(momentum_breakout_engine.py)：預設True維持舊版
       行為(一定要收盤價>前高/前低才算候選)；False時改用「軟性排名」模式，只留最基本的
       「站上/跌破20日均線」當結構性資格，突破強度完全交給上面的相對排名去決定「這批
       候選裡誰的突破比較像樣」，不再用一個固定的絕對值切斷候選。
    c. **突破風格比較(階段0.7)**：在突破窗口選定之後、訊號拆解之前，新增一個小比較：
       硬門檻 vs 軟性排名，兩種都用等權重訊號+基本門檻，選總損益較高的一個，固定下來
       套用到後面所有階段(訊號拆解/門檻比較/ATR網格/最終驗證/跨週期驗證)。
    d. **execution_kwargs一致性修正**：上一輪發現的方法論問題——ATR敏感度網格
       (階段2.5)跟出場配置比較(階段2.6)過去寫死lots=2，沒有套用--risk-pct-per-trade/
       --max-concurrent-positions，導致「篩選階段選出的最佳組合」跟「最終驗證實際
       套用的部位大小」不是同一套規則，篩選結果可能失真。這一輪把這兩個函式改成
       接收真正的execution_kwargs，確保篩選跟最終驗證用同一套部位大小規則。
11. **使用者盤感提出的新假設(這輪新增)**：
    a. **訊號組合比較(新增的「階段1.5」)**：使用者提出「MACD柱狀圖轉正+RSI穿越50+
       量比放大」代表「主力進場、散戶跟進、短進短出、最後誰接盤」的組合效應假設，
       這跟既有的「單一訊號拆解」自動篩選機制邏輯不同(自動篩選只驗證單一訊號的個別
       預測力，量比/RSI單獨測都不到1，永遠不會被自動選中)，所以新增這個階段，把
       自動篩選出的組合跟這組手動指定的固定組合放在一起比較，選總損益較高的那組，
       固定套用到後面所有階段(門檻比較改用選定的組合當基準，不再是全部訊號等權重)。
    b. **MA5基礎+均線型態門檻(新增的GATE_VARIANTS一項)**：短波段基本方向確認改用
       MA5(5日均線)取代MA20，額外要求「均線糾結(MA5/MA20/MA60三線貼在一起，
       (三線最大值-最小值)/ATR<=0.5)」或「多頭並列且三線同步向上(不只排列順序對，
       三條線各自都要比5天前的自己高)」兩者擇一，見momentum_breakout_engine.py新增的
       MASpreadOverATR/MA5Slope5/MA20Slope5/MA60Slope5指標欄位跟use_ma5_base/
       require_ma_pattern參數。
    c. **保本停損(新增的EXIT_STYLE_VARIANTS一項)**：回應「只要有漲，停利一定是成本價，
       不要漲了後面下跌還賠錢」這個原則，新增breakeven_after_profit機制——浮動獲利
       轉正後把停損移到成本價，不管有沒有開移動停利都會生效，是獨立於移動停利之外的
       一道「保本地板」，實作在mean_reversion_engine._process_mr_day()。搭配新的
       「3天出場+保本停損」出場配置(強制出場天數從10天縮短到3天)。
    這幾個都是使用者盤感提出、還沒被驗證過的新假設，程式會照實測出來的PF/總損益排名，
    不會因為背後的邏輯聽起來合理就預設會贏。
12. **新增7個訊號 + 明確跳過融資融券比/家數差 + 走勢前進(Walk-Forward)分折驗證(這輪新增)**：
    a. 7個新訊號(見momentum_breakout_engine.py模組docstring/SIGNAL_LABELS)：外資/投信
       連續買超天數(score_foreign_streak/score_trust_streak)、主力(外資+投信合計)買賣超
       強度1/5/10日窗口(score_institutional_net_1d/5d/10d，這就是使用者說的「主力買賣超」)、
       MACD背離近似訊號(score_macd_divergence，⚠️近似實作，不是嚴謹的擺盪高低點背離)、
       短窗(5日)成交量急增(score_volume_spike，跟既有20日窗口的量比訊號刻意區隔開)。
    b. **明確跳過融資融券比/家數差**：不是忘記做，是資料源缺口(需要另一個TWSE報表/
       付費第三方資料)，見chip_data_loader.py模組docstring最後一段的完整說明。
    c. **走勢前進(Walk-Forward)分折驗證**(新增`--walkforward-folds`，見
       run_walkforward_validation())：回應「單一次OOS切分可能只是運氣好」的疑慮，把歷史
       切成N+1個等長區塊，每折用擴張視窗重新跑一次完整選股流程，只在自己的測試窗口驗證，
       用來檢查選出的訊號/門檻/出場配置跨時間是否穩定。0(預設)代表不啟用，不影響原本
       單一次IS/OOS驗證的行為。詳見README「走勢前進(Walk-Forward)分折驗證」一節。

用法：
    python3 compare_breakout.py --start 2023-09-15 --end 2026-09-14 --max-stocks 50
    python3 compare_breakout.py --with-chip-confirm --clean-ex-dividend
    python3 compare_breakout.py --use-trailing-stop --atr-stop-mult 1.0
    python3 compare_breakout.py --use-trailing-stop --max-concurrent-positions 3 \\
        --risk-pct-per-trade 0.02 --slippage-pct 0.002 --multi-period-test
    python3 compare_breakout.py --use-trailing-stop --walkforward-folds 3
    python3 compare_breakout.py --squeeze-kdj-grid --max-stocks 50   # squeeze+KDJ混搭網格(1440組，只用IS選贏家)

⚠️ 誠實揭露：跟 mean_reversion_engine.py 共用的已知限制(結算日近似、大盤氛圍濾網用0050
代理、跌停鎖死/注意股處置股未實作、倖存者偏差、保證金追繳/強制斷頭沒有完整模擬)在這裡
一樣成立，請見 README。
"""
import argparse
import datetime
import os

import numpy as np
import pandas as pd

from data_loader import load_price_data
from taifex_universe import STOCK_FUTURES_UNIVERSE
from momentum_breakout_engine import (
    run_momentum_breakout_backtest, BREAKOUT_SIGNAL_NAMES, CHIP_DEPENDENT_SIGNALS,
    precompute_all_breakout_indicators,
)
from mean_reversion_engine import summarize_mr, precompute_regime_series, COMMISSION_PER_LOT_PER_LEG
from robustness_analysis import (
    bootstrap_resample_pnl, summarize_bootstrap, pnl_excluding_top_n_trades, bootstrap_p_value,
)
from taifex_universe import get_contract_multiplier
from squeeze_kdj_signal import (
    compute_squeeze_kdj_features, simulate_variant_a_trades, simulate_variant_b_trades,
    precompute_squeeze_kdj_features_by_code, run_squeeze_kdj_capital_constrained_backtest,
    precompute_squeeze_kdj_backtest_arrays, CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS,
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
    "score_breakout_strength": "突破強度(ATR倍數相對排名)",
    "score_foreign_streak": "外資連續買超天數",
    "score_trust_streak": "投信連續買超天數",
    "score_institutional_net_1d": "主力買賣超(1日)",
    "score_institutional_net_5d": "主力買賣超(5日)",
    "score_institutional_net_10d": "主力買賣超(10日)",
    "score_macd_divergence": "MACD背離(近似)",
    "score_volume_spike": "成交量急增(5日窗口)",
    "score_squeeze_kdj": "布林+Keltner擠壓+KDJ(僅多方)",
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
    # 使用者盤感提出的假設：短波段基本方向改看MA5(比MA20更即時)，額外要求「均線糾結
    # (像要爆發)」或「多頭並列且三線同步向上」兩者擇一——不是隨便均線排列對就算，
    # 而是三線真的都在漲，或是三線貼在一起代表盤整蓄積、隨時要噴出。這是還沒被驗證過
    # 的新假設，不是先驗認定一定比MA20+創新高好。
    ("MA5基礎+均線糾結或多頭同步向上", {"use_ma5_base": True, "require_ma_pattern": True}),
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

# 突破風格比較(新增的「階段0.7」)：硬門檻(舊版，一定要收盤價>突破窗口高/低點才算候選)
# vs 軟性排名(只留最基本的「站上/跌破20日均線」當結構性資格，突破強度改交給
# score_breakout_strength做相對排名，不再用絕對值判定「有沒有突破」)。回應「專業投資經理」
# 設計裡的核心觀察：既有的score_*訊號(全部都是相對排名)在標的池大小變動時比二元的硬門檻
# 穩定，這裡直接測試「把突破確認本身也改成相對排名」是不是能提高穩定性。
BREAKOUT_STYLE_VARIANTS = [
    ("硬門檻(收盤價需站上突破窗口高/低點)", True),
    ("軟性排名(僅站上/跌破20日均線+相對排名)", False),
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
#   breakeven_after_profit：保本停損，浮動獲利轉正後把停損移到成本價，不管有沒有
#     開移動停利都會生效(使用者盤感提出：「只要有漲，停利一定是成本價，不要漲了
#     後面下跌還賠錢」)
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
    ("3天出場+保本停損", {"max_hold_days_override": 3, "breakeven_after_profit": True}),
]

# --simple-combo模式(刻意跟上面6階段自動搜尋分開)：固定用這個使用者已經確認過的訊號組合
# (MACD柱狀圖+K棒實體比例，跟FIXED_WALKFORWARD_COMBO是同一組)，只比較「加不加這兩個門檻」
# 這一個問題本身，不讓訊號/突破窗口/突破風格/出場配置也變成「挑出來的」——見main()裡
# --simple-combo分支的docstring說明，這是為了回應一次真實GitHub Actions執行結果：
# 完整6階段流程選出的「最佳組合」IS PF=1.41、OOS PF=0.48、bootstrap正報酬比例只有5.2%，
# 嚴重的IS/OOS表現落差，幾乎可以確定是連續6輪都在同一份IS資料上挑贏家、疊加起來的
# multiple-comparisons(多重比較)問題，不是這個訊號組合本身沒用。
SIMPLE_COMBO_SIGNAL_WEIGHTS = {"score_macd": 1.0, "score_candle_body": 1.0}

# 三個「累加」門檻變體：每一個都只比前一個多加一道濾網，方便看出「這道濾網到底有沒有用」，
# 不是互相獨立的平行選項。use_regime_gate是momentum_breakout_engine.py這次新增的開關——
# 這支引擎的大盤氛圍regime濾網(只在大盤偏多時放行多方候選、只在大盤偏空時放行空方候選，
# 順勢方向)過去是寫死套用、沒有開關，--simple-combo模式需要「不開regime」當基準
# (變體1/2)才能跟「加regime」的變體3比較，所以在引擎那邊補上這個開關，預設True維持
# 舊版(完整6階段流程/main()其餘所有呼叫)的行為不變，見momentum_breakout_engine.
# run_momentum_breakout_backtest()的use_regime_gate docstring。
SIMPLE_COMBO_VARIANTS = [
    ("變體1(基本門檻，無ADX無大盤氛圍regime)", {"use_regime_gate": False}),
    ("變體2(+趨勢強度ADX>=25)", {"min_adx": 25.0, "use_regime_gate": False}),
    ("變體3(+趨勢強度ADX>=25+大盤氛圍regime濾網)", {"min_adx": 25.0, "use_regime_gate": True}),
]


def run_simple_combo_comparison(price_data, indicators_by_code, regime_series, is_calendar,
                                 starting_capital, hold_days, atr_stop_mult, trailing_atr_mult,
                                 extra_kwargs=None, signal_weights=None):
    """--simple-combo模式的核心比較：固定訊號組合(預設SIMPLE_COMBO_SIGNAL_WEIGHTS，可由
    signal_weights參數覆蓋——見--combo-search模式：把這裡換成資料驅動選出的「單一訊號
    混搭」組合，重用同一套門檻比較邏輯，不重新實作)，突破窗口/突破風格維持引擎預設
    (20日、硬門檻，使用者沒有要求重新測這兩個維度，重測會重新引入同一種多重比較風險)，
    只在IS內跑SIMPLE_COMBO_VARIANTS這三組累加門檻，印出IS PF/勝率/交易數/總損益讓使用者
    直接看「加這道濾網有沒有讓結果變好」，不自動挑贏家、不往下一階段傳遞「選出來的」設定
    ——三個變體都跑完就結束，呼叫端(main())自己決定要把哪一個(預設變體3，使用者要求的
    「最終要測的那組」)拿去跑IS/OOS+bootstrap。"""
    extra_kwargs = extra_kwargs or {}
    signal_weights = signal_weights if signal_weights is not None else SIMPLE_COMBO_SIGNAL_WEIGHTS
    rows = []
    for label, gate_kwargs in SIMPLE_COMBO_VARIANTS:
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=hold_days, starting_capital=starting_capital,
            allow_short=True, lots=2, atr_stop_mult=atr_stop_mult, trailing_atr_mult=trailing_atr_mult,
            use_trailing_stop=True, signal_weights=signal_weights,
            **gate_kwargs, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "variant": label, "trade_count": stats["trade_count"], "profit_factor": stats["profit_factor"],
            "win_rate": stats["win_rate"], "total_pnl_ntd": stats["total_pnl_ntd"],
        })
        print(f"  {label} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)
    return pd.DataFrame(rows)


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


def select_winning_breakout_style(style_df, min_trades=MIN_TRADES_FOR_GATE_RANKING):
    """從突破風格比較(硬門檻 vs 軟性排名)結果裡，挑總損益最高、且交易筆數不會太少的那個風格。"""
    pool = _prefer_nonzero_trades(style_df)
    reliable = pool[pool["trade_count"] >= min_trades]
    if reliable.empty:
        reliable = pool
    if reliable.empty:
        return BREAKOUT_STYLE_VARIANTS[0]  # 找不到任何可用結果時，退回硬門檻(舊版行為)
    best_row = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    return best_row["style_label"], bool(best_row["require_hard_breakout"])


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


def run_breakout_style_comparison(price_data, indicators_by_code, regime_series, is_calendar,
                                   starting_capital, hold_days, all_signal_names, extra_kwargs=None):
    """突破窗口選定之後，測「硬門檻(絕對值判定有沒有突破)」vs「軟性排名(只留MA20結構性
    資格，突破強度改用相對排名)」哪個比較好。一樣用全部訊號等權重+基本門檻，只換
    require_hard_breakout這一個變數，先回答「突破確認本身該不該用絕對值判定」這個問題，
    選出來的風格會固定下來，之後的訊號拆解/門檻比較/ATR網格/最終驗證都套用。"""
    extra_kwargs = extra_kwargs or {}
    rows = []
    equal_weights = {name: 1.0 for name in all_signal_names}
    for label, require_hard in BREAKOUT_STYLE_VARIANTS:
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=hold_days, starting_capital=starting_capital,
            allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0,
            signal_weights=equal_weights, require_hard_breakout=require_hard, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "style_label": label, "require_hard_breakout": require_hard,
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


# 使用者盤感提出的固定訊號組合：MACD柱狀圖轉正 + RSI穿越50 + 量比放大，代表「主力進場、
# 散戶也跟著進來追、短進短出、最後誰接盤」的組合效應假設——這跟上面的自動篩選機制邏輯
# 不同：自動篩選只驗證「單一訊號」的個別預測力(PF>1才選)，量比/RSI單獨測都不到1，永遠
# 不會被自動選中，但這個假設認為「這三個訊號組合在一起才有效」，是單一訊號拆解測不出來
# 的組合效應，所以另外測試這組固定組合，跟自動篩選的結果放在一起比較，不預設誰比較好。
MANUAL_SIGNAL_COMBO_LABEL = "手動指定(MACD轉正+RSI>50+量比放大)"
MANUAL_SIGNAL_COMBO = {"score_macd": 1.0, "score_rsi_cross": 1.0, "score_volume_ratio": 1.0}


# 這組具體規則(訊號+門檻+突破窗口/風格+ATR倍數+出場配置)連續兩輪(本輪+上一輪)完整跑過
# 整套選擇流程(訊號拆解→訊號組合比較→門檻比較→ATR網格→出場配置比較)後，都被自動選中，
# 不是隨便挑的候選——但兩輪的歷史區間幾乎完全重疊(只差1天的起訖日)，嚴格來說不算獨立
# 驗證過兩次，只能算「同一份資料重複跑，程式邏輯本身沒有隨機性導致結果一致」，這裡才
# 用真正互相獨立、不重疊的歷史切段去測試這組固定規則，才是這個常數存在的意義。
FIXED_WALKFORWARD_COMBO = {
    "label": "固定候選(MACD柱狀圖+K棒實體比例, +站上季線, 52週窗口+軟性排名, ATR1.5/1.5, 魚身整合版出場)",
    "signal_weights": {"score_macd": 1.0, "score_candle_body": 1.0},
    "gate_kwargs": {"require_above_ma60": True},
    "breakout_window": 250,
    "require_hard_breakout": False,
    "atr_stop_mult": 1.5,
    "trailing_atr_mult": 1.5,
    "exit_kwargs": {
        "max_hold_days_override": 10, "trailing_activation_days": 2,
        "trailing_activation_profit_atr": 1.5, "max_gap_pct": 0.025,
    },
}

# 固定候選規則的變體比較：FIXED_WALKFORWARD_COMBO在6個獨立區塊上測出「穩定但平均PF<1
# (0.95)」——這代表規則本身不是在追雜訊(範圍窄)，但也還沒有正期望值。這裡列出幾個
# 「只改一個維度、其餘完全不變」的變體，同樣各自套到獨立區塊上跑固定規則walk-forward，
# 想知道換哪個維度可以把平均PF推過1、同時不犧牲掉「範圍窄=穩定」這個已經驗證到的優點。
# 每個變體都只改自FIXED_WALKFORWARD_COMBO(基準)裡的「一個」維度，方便歸因是哪個改動
# 造成差異，不是同時改好幾個變數搞不清楚是誰的功勞。
FIXED_WALKFORWARD_COMBO_VARIANTS = [
    FIXED_WALKFORWARD_COMBO,
    {
        **FIXED_WALKFORWARD_COMBO,
        "label": "固定候選B(訊號加入相對大盤強弱，其餘不變)",
        "signal_weights": {"score_macd": 1.0, "score_candle_body": 1.0, "score_rel_strength": 1.0},
    },
    {
        **FIXED_WALKFORWARD_COMBO,
        "label": "固定候選C(門檻改為趨勢強度+站上季線雙重確認，其餘不變)",
        "gate_kwargs": {"min_adx": 25.0, "require_above_ma60": True},
    },
    {
        **FIXED_WALKFORWARD_COMBO,
        "label": "固定候選D(ATR倍數改為1.0/1.0，交易數較多，其餘不變)",
        "atr_stop_mult": 1.0,
        "trailing_atr_mult": 1.0,
    },
    {
        **FIXED_WALKFORWARD_COMBO,
        "label": "固定候選E(出場改為單純10天強制出場，不含延遲啟動/跳空上限，其餘不變)",
        "exit_kwargs": {"max_hold_days_override": 10},
    },
    {
        **FIXED_WALKFORWARD_COMBO,
        "label": "固定候選G(訊號只用K棒實體比例，不跟MACD等權重平均，其餘不變)",
        "signal_weights": {"score_candle_body": 1.0},
    },
    {
        **FIXED_WALKFORWARD_COMBO,
        # 拆解魚身整合版(天數上限+延遲啟動+跳空上限)：候選E已經測過「全拿掉只留天數上限」，
        # 這裡改成「只拿掉跳空上限，保留延遲啟動」，用來隔離是延遲啟動還是跳空上限在起作用。
        "label": "固定候選H(出場只留延遲啟動，拿掉跳空上限，其餘不變)",
        "exit_kwargs": {
            "max_hold_days_override": 10, "trailing_activation_days": 2,
            "trailing_activation_profit_atr": 1.5,
        },
    },
]


def run_signal_combo_comparison(price_data, indicators_by_code, regime_series, is_calendar,
                                 starting_capital, hold_days, auto_label, auto_weights, extra_kwargs=None):
    """比較「自動篩選出的訊號組合」vs 使用者手動指定的固定組合(MANUAL_SIGNAL_COMBO)。
    只用基本門檻(門檻比較階段還沒開始)，只在IS內比較，選出來的訊號組合會固定下來，
    後面的門檻比較/ATR網格/出場配置/最終驗證都套用。"""
    extra_kwargs = extra_kwargs or {}
    rows = []
    for label, weights in [(auto_label, auto_weights), (MANUAL_SIGNAL_COMBO_LABEL, MANUAL_SIGNAL_COMBO)]:
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=hold_days, starting_capital=starting_capital,
            allow_short=True, lots=2, atr_stop_mult=1.0, atr_target_mult=2.0,
            signal_weights=weights, **extra_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "combo_label": label, "trade_count": stats["trade_count"], "profit_factor": stats["profit_factor"],
            "win_rate": stats["win_rate"], "total_pnl_ntd": stats["total_pnl_ntd"],
        })
        print(f"  {label} -> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)
    return pd.DataFrame(rows)


def select_winning_signal_combo(combo_df, auto_label, auto_weights, min_trades=MIN_TRADES_FOR_GATE_RANKING):
    """從訊號組合比較(自動篩選 vs 手動指定)裡，挑總損益較高、且交易筆數不會太少的那組，
    回傳(label, weights)。"""
    pool = _prefer_nonzero_trades(combo_df)
    reliable = pool[pool["trade_count"] >= min_trades]
    if reliable.empty:
        reliable = pool
    if reliable.empty:
        return auto_label, auto_weights
    best_row = reliable.sort_values("total_pnl_ntd", ascending=False).iloc[0]
    label = best_row["combo_label"]
    if label == MANUAL_SIGNAL_COMBO_LABEL:
        return label, dict(MANUAL_SIGNAL_COMBO)
    return auto_label, auto_weights


def run_gate_comparison(price_data, indicators_by_code, regime_series, is_calendar,
                         starting_capital, hold_days, signal_names, has_chip=False, extra_kwargs=None):
    """用階段1.5選出的訊號組合(等權重)當基準，把結構門檻換成不同變體，比較哪種過濾條件
    篩出的候選比較好——這輪改成「用已經選定的訊號組合」而不是「全部訊號等權重」當基準，
    因為階段1.5可能選出手動指定的小組合(例如MACD+RSI+量比)，門檻比較應該回答「在這個
    已經選定的訊號組合下，哪個門檻最好」，不是拿一個跟訊號選擇無關的基準去測門檻。"""
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
                              starting_capital, signal_weights, gate_kwargs, extra_kwargs,
                              execution_kwargs=None):
    """僅移動停利模式使用：在最佳訊號/門檻組合固定之後，測一個小網格的
    (初始停損倍數, 移動停利倍數)，只在IS內搜尋，回傳完整結果表 + PF最高的一組。

    execution_kwargs：最終驗證會用到的真實部位大小設定(risk_pct_per_trade或lots、
    max_concurrent_positions)。不傳時退回舊版寫死的lots=2/單一部位，但這樣選出來的
    「最佳ATR倍數」是在跟最終驗證不同的部位大小規則下選的，可能不是真的最佳——
    見這一輪修正的execution_kwargs一致性問題，這裡改成預設吃真正的執行參數，
    確保篩選階段跟最終驗證用同一套部位大小規則。"""
    execution_kwargs = dict(execution_kwargs) if execution_kwargs else {"lots": 2}
    rows = []
    for stop_mult, trail_mult in ATR_SENSITIVITY_GRID:
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=TRAILING_STOP_MAX_HOLD_DAYS,
            starting_capital=starting_capital, allow_short=True,
            signal_weights=signal_weights, atr_stop_mult=stop_mult,
            use_trailing_stop=True, trailing_atr_mult=trail_mult,
            **gate_kwargs, **extra_kwargs, **execution_kwargs,
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
                               extra_kwargs, execution_kwargs=None):
    """「只吃魚身」出場配置比較(僅移動停利模式使用)：ATR倍數固定之後，另外測試持有天數
    上限/移動停利延遲啟動/開盤跳空上限這幾個進出場配置組合，只在IS內搜尋。額外印出
    平均持有天數，用來檢查有沒有真的抓到預期的短波段長度，不是憑感覺猜參數。

    execution_kwargs：同run_atr_sensitivity_grid()，預設吃真正的部位大小設定，
    確保這裡選出的「最佳出場配置」是在跟最終驗證相同的部位大小規則下選的。"""
    execution_kwargs = dict(execution_kwargs) if execution_kwargs else {"lots": 2}
    rows = []
    for label, style_kwargs in EXIT_STYLE_VARIANTS:
        style_kwargs = dict(style_kwargs)
        max_hold_days = style_kwargs.pop("max_hold_days_override", TRAILING_STOP_MAX_HOLD_DAYS)
        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=is_calendar, max_hold_days=max_hold_days,
            starting_capital=starting_capital, allow_short=True,
            signal_weights=signal_weights, atr_stop_mult=atr_stop_mult,
            use_trailing_stop=True, trailing_atr_mult=trailing_atr_mult,
            **gate_kwargs, **extra_kwargs, **style_kwargs, **execution_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        rows.append({
            "exit_style": label, "max_hold_days": max_hold_days,
            "trailing_activation_days": style_kwargs.get("trailing_activation_days", 0),
            "trailing_activation_profit_atr": style_kwargs.get("trailing_activation_profit_atr", 0.0),
            "max_gap_pct": style_kwargs.get("max_gap_pct"),
            "breakeven_after_profit": style_kwargs.get("breakeven_after_profit", False),
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
    (label, max_hold_days, extra_kwargs)，extra_kwargs包含trailing_activation_days/
    trailing_activation_profit_atr/max_gap_pct/breakeven_after_profit這幾個要餵進
    run_momentum_breakout_backtest的鍵(None的max_gap_pct、False的breakeven_after_profit
    會被拿掉，避免傳一個沒意義的值覆蓋掉預設值以外的行為——其實傳了結果一樣，這裡拿掉
    純粹讓kwargs乾淨)。"""
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
    if bool(best_row.get("breakeven_after_profit", False)):
        extra["breakeven_after_profit"] = True
    return best_row["exit_style"], int(best_row["max_hold_days"]), extra


def _cost_squeeze_kdj_trades(raw_trades: list, code: str, lots: int = 2) -> list:
    """把squeeze_kdj_signal.py算出來的原始交易(entry_price/exit_price/hold_days)，
    套上股票期貨的合約乘數+手續費，轉成summarize_mr()看得懂的trades格式(只做多方，
    這裡沒有side欄位判斷，固定當long算)。"""
    out = []
    for t in raw_trades:
        e_price, exit_price = t["entry_price"], t["exit_price"]
        mult = get_contract_multiplier(code, e_price)
        price_pnl = (exit_price - e_price) * mult * lots
        commission = COMMISSION_PER_LOT_PER_LEG * lots * 2
        pnl_ntd = price_pnl - commission
        return_pct = pnl_ntd / (e_price * mult * lots) if e_price > 0 else 0.0
        out.append({
            "code": code, "side": "long", "entry_date": t["entry_date"], "exit_date": t["exit_date"],
            "e_price": e_price, "exit_price": exit_price, "exit_reason": t["exit_reason"],
            "lots": lots, "pnl_ntd": pnl_ntd, "return_pct": return_pct, "hold_days": t["hold_days"],
        })
    return out


def run_squeeze_kdj_exit_style_comparison(price_data: dict, universe: dict, starting_capital: float,
                                           lots: int = 2) -> pd.DataFrame:
    """
    「布林+Keltner擠壓+KDJ」訊號隔離比較：同一套進場規則，變體A(原規則：前一根K棒低點
    停損 + K衝高後跌破80停利) vs 變體B(沿用這支引擎既有的ATR停損/停利框架，
    atr_stop_mult=1.0/atr_target_mult=2.0，跟main()非移動停利模式的預設值一致)。

    這裡刻意不透過scan_momentum_breakout_candidates()/run_momentum_breakout_backtest()
    的完整候選排名+資金管理流程，而是對universe裡每一檔股票各自獨立模擬(見
    squeeze_kdj_signal.simulate_variant_a_trades/simulate_variant_b_trades)，因為
    這裡要回答的問題單純是「同一個進場訊號，兩種出場方式誰比較好」，不需要跟其他
    訊號比排名、也不需要模擬「資金只夠買前top_n名」的排擠效應——那些跟「這個出場方式
    好不好」是兩個不同的問題，混在一起反而讓比較結果變得不乾淨。
    """
    variant_a_trades, variant_b_trades = [], []
    for code in universe:
        df = price_data.get(code)
        if df is None or len(df) < 60:
            continue
        features = compute_squeeze_kdj_features(df)
        raw_a = simulate_variant_a_trades(df, features)
        # atr_period=14跟momentum_breakout_engine.precompute_breakout_indicators()裡
        # ATR欄位用的期數一致(compute_atr_correct()預設period=14)，才算真的「沿用」
        # 這支引擎既有的ATR框架，不是另外發明一個不同期數的ATR。
        raw_b = simulate_variant_b_trades(df, features, atr_period=14, atr_stop_mult=1.0, atr_target_mult=2.0)
        variant_a_trades.extend(_cost_squeeze_kdj_trades(raw_a, code, lots=lots))
        variant_b_trades.extend(_cost_squeeze_kdj_trades(raw_b, code, lots=lots))

    stats_a = summarize_mr(variant_a_trades, starting_capital)
    stats_b = summarize_mr(variant_b_trades, starting_capital)
    rows = [
        {"variant": "變體A(原規則：前K棒低點停損+K跌破80停利)", **stats_a},
        {"variant": "變體B(沿用ATR框架：停損1.0x ATR/停利2.0x ATR)", **stats_b},
    ]
    return pd.DataFrame(rows)


SQUEEZE_KDJ_VARIANT_LABELS = {
    "A": "變體A(原規則：前K棒低點停損+K跌破80停利)",
    "B": "變體B(沿用ATR框架：停損1.0x ATR/停利2.0x ATR)",
}


def run_squeeze_kdj_exit_style_comparison_is_oos(price_data: dict, universe: dict, starting_capital: float,
                                                  is_calendar, oos_calendar, lots: int = 2) -> dict:
    """
    run_squeeze_kdj_exit_style_comparison()的IS/OOS + bootstrap版本——這份比較是這個檔案
    裡最後一個還沒套用專案自己整套驗證方法論(70/30 IS/OOS切分 + OOS的1000次bootstrap重抽樣，
    見evaluate_combo())的地方，之前只跑過全樣本、沒切IS/OOS，曾經看到變體之一PF=1.84、
    總損益+NT$518萬這種好看的數字，但完全沒驗證過是不是過擬合/運氣——這個repo別的地方已經
    示範過好看的全樣本/IS數字換到OOS可能完全不是那回事(例如momentum_breakout 6階段流程
    IS PF=1.41但OOS只有0.48)，所以在對這個數字做任何進一步調整之前，第一件事是先老實驗證，
    不是直接拿來优化。

    做法：跟run_squeeze_kdj_exit_style_comparison()一樣，對universe每檔股票各自獨立模擬出
    變體A/B的完整交易列表(不重新模擬兩次)，但接下來不是直接summarize_mr()全樣本，而是
    把每一筆交易依entry_date是否早於OOS切分點(is_calendar/oos_calendar的分界，即
    oos_calendar[0])分成IS/OOS兩組——因為這個函式本身不經過
    run_momentum_breakout_backtest()/master_calendar逐日迴圈那條路徑(直接呼叫
    squeeze_kdj_signal.simulate_variant_a_trades/simulate_variant_b_trades，一次性算出
    整段期間的交易列表)，沒有「只給IS那段calendar去跑」這個選項，退而求其次用交易的
    entry_date直接切，等價於「如果IS calendar到哪天為止，只保留在那之前進場的交易」，
    不會有用到未來資料的問題(訊號計算本身只看當天以前的OHLCV，分組動作是在交易模擬完成
    之後才做，不影響訊號本身)。

    IS/OOS統計都用summarize_mr()，OOS再額外跑bootstrap_resample_pnl()
    (n_resamples=1000, seed=42，跟evaluate_combo()同一套慣例)+summarize_bootstrap()+
    bootstrap_p_value()+pnl_excluding_top_n_trades()。回傳格式比照evaluate_combo()，
    方便呼叫端用同一套印法：{"A": {"label":..., "IS":..., "OOS":..., "bootstrap":...,
    "oos_trades":...}, "B": {...}}。

    誠實caveat(沿用自run_squeeze_kdj_exit_style_comparison()，這裡的IS/OOS/bootstrap
    驗證並沒有改變這一點)：這個比較刻意不經過資金/部位管理(沒有top_n排名、沒有同時
    持倉上限)，universe裡每一檔股票的每一個訊號都視為獨立成交，這是為了單純比較「這個
    進場/出場邏輯本身好不好」，但代表這裡的PF/損益數字比真實帳戶(資金有限、不可能
    同時吃下所有訊號)能拿到的數字樂觀——IS/OOS+bootstrap驗證回答的是「這套進場/出場邏輯
    方向上是否穩健」，不是「我的帳戶實際能拿到的PF」，不要把這裡的OOS數字直接當成
    實盤可以期待的報酬。
    """
    is_cutoff = oos_calendar[0] if len(oos_calendar) > 0 else None
    is_trades = {"A": [], "B": []}
    oos_trades = {"A": [], "B": []}

    for code in universe:
        df = price_data.get(code)
        if df is None or len(df) < 60:
            continue
        features = compute_squeeze_kdj_features(df)
        raw_a = simulate_variant_a_trades(df, features)
        raw_b = simulate_variant_b_trades(df, features, atr_period=14, atr_stop_mult=1.0, atr_target_mult=2.0)
        costed_a = _cost_squeeze_kdj_trades(raw_a, code, lots=lots)
        costed_b = _cost_squeeze_kdj_trades(raw_b, code, lots=lots)
        for key, costed in (("A", costed_a), ("B", costed_b)):
            for t in costed:
                bucket = oos_trades[key] if (is_cutoff is not None and t["entry_date"] >= is_cutoff) else is_trades[key]
                bucket.append(t)

    results = {}
    for key, label in SQUEEZE_KDJ_VARIANT_LABELS.items():
        is_stats = summarize_mr(is_trades[key], starting_capital)
        oos_stats = summarize_mr(oos_trades[key], starting_capital)
        boot_results = bootstrap_resample_pnl(oos_trades[key], n_resamples=1000, seed=42)
        boot_stats = summarize_bootstrap(boot_results)
        results[key] = {
            "label": label, "IS": is_stats, "OOS": oos_stats,
            "bootstrap": {
                **boot_stats,
                "p_value": bootstrap_p_value(boot_results),
                "pnl_excluding_top3_ntd": pnl_excluding_top_n_trades(oos_trades[key], n=3),
            },
            "oos_trades": oos_trades[key],
        }
    return results


def _squeeze_kdj_is_oos_results_to_df(results: dict) -> pd.DataFrame:
    """把run_squeeze_kdj_exit_style_comparison_is_oos()的回傳dict整理成一張長格式的
    DataFrame，方便寫成單一個CSV：每個變體各兩列(IS/OOS)的基本統計，另外加一列
    OOS bootstrap的彙整統計(mean/p5/p95/pct_positive/p_value/拿掉最大3筆後損益)。"""
    rows = []
    for r in results.values():
        for split_name in ("IS", "OOS"):
            stats = r[split_name]
            rows.append({"variant": r["label"], "split": split_name, **stats})
        b = r["bootstrap"]
        rows.append({
            "variant": r["label"], "split": "OOS_bootstrap(1000次重抽樣)",
            "trade_count": len(r["oos_trades"]),
            "bootstrap_mean_ntd": b["mean"], "bootstrap_p5_ntd": b["p5"], "bootstrap_p95_ntd": b["p95"],
            "bootstrap_pct_positive": b["pct_positive"], "bootstrap_p_value": b["p_value"],
            "pnl_excluding_top3_ntd": b["pnl_excluding_top3_ntd"],
        })
    return pd.DataFrame(rows)


def run_squeeze_kdj_only_mode(args, price_data, universe, is_calendar, oos_calendar):
    """--squeeze-kdj-only模式：只跑squeeze+KDJ訊號的IS/OOS+bootstrap驗證(見
    run_squeeze_kdj_exit_style_comparison_is_oos() docstring)，完全跳過突破窗口比較→
    突破風格比較→訊號自動搜尋→門檻網格→ATR敏感度網格→出場配置網格這整套6階段流程——
    squeeze+KDJ驗證本來就不依賴這6個階段選出的任何東西(它是獨立對universe逐檔股票
    模擬，不經過scan_momentum_breakout_candidates()/run_momentum_breakout_backtest()
    那條路徑)，之前要看這個驗證結果，得先等整套6階段流程跑完才會印出來，這裡讓使用者
    可以只測這一段、不用等前面不相關的流程，跟--simple-combo/--combo-search的「只測
    使用者真正想看的那一段」精神一致。

    加這個旗標時，--atr-stop-mult/--trailing-atr-mult/--start/--end/--starting-capital/
    --max-stocks仍然有效(影響universe大小跟bootstrap用的起始資金)，但squeeze+KDJ訊號
    本身的進場/出場規則是寫死在squeeze_kdj_signal.py裡的(不吃訊號權重/門檻這些參數)，
    所以跟訊號/門檻/突破窗口/出場配置自動搜尋相關的旗標在這裡本來就不適用，不是「被忽略」，
    是這個模式測的東西跟那些旗標控制的維度完全無關。
    """
    print("=" * 100)
    print("--squeeze-kdj-only模式：只跑布林+Keltner擠壓+KDJ訊號的IS/OOS+bootstrap驗證，"
          "跳過突破窗口/突破風格/訊號自動搜尋/門檻網格/ATR網格/出場配置網格這6個階段")
    print("(這個驗證本來就不依賴那6個階段選出的任何東西，這裡只是讓使用者不用等前面"
          "不相關的流程跑完)")
    print("=" * 100)
    print(f"\n布林+Keltner擠壓+KDJ訊號：變體A(原規則) vs 變體B(沿用ATR框架) ...")
    print("  ⚠️ 之前只跑過全樣本、沒有分IS/OOS，之前一次好看的全樣本數字(PF=1.84、"
          "損益+NT$518萬)完全沒驗證過是不是過擬合——先驗證再談優化，見"
          "run_squeeze_kdj_exit_style_comparison_is_oos() docstring")
    squeeze_kdj_results = run_squeeze_kdj_exit_style_comparison_is_oos(
        price_data, universe, args.starting_capital, is_calendar, oos_calendar,
    )
    squeeze_kdj_exit_df = _squeeze_kdj_is_oos_results_to_df(squeeze_kdj_results)
    squeeze_kdj_exit_df.to_csv(os.path.join(RESULTS_DIR, "squeeze_kdj_is_oos.csv"),
                                index=False, encoding="utf-8-sig")

    summary_lines = [
        "=" * 100,
        "布林+Keltner擠壓+KDJ訊號 --squeeze-kdj-only模式(只驗證這一個訊號，跳過其餘6階段流程)",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        "=" * 100,
        "\n(這個比較跟main()其餘階段獨立，只做多方，見squeeze_kdj_signal.py模組docstring；"
        "之前只跑過全樣本、沒有驗證過是不是過擬合，這裡補上IS/OOS切分+OOS bootstrap"
        "穩健性檢查)",
    ]
    for r in squeeze_kdj_results.values():
        print(f"  [{r['label']}]")
        summary_lines.append(f"\n  [{r['label']}]")
        for split_name in ["IS", "OOS"]:
            stats = r[split_name]
            split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
            line = (f"    {split_full}: {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
                    f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
                    f"總損益NT${stats['total_pnl_ntd']:,.0f}, "
                    f"最大回撤NT${stats['max_drawdown_ntd']:,.0f}, 最大連續虧損{stats['max_consecutive_losses']}筆")
            print("  " + line.strip())
            summary_lines.append(line)
        b = r["bootstrap"]
        boot_line = (f"    [穩健性] OOS bootstrap 1000次重抽樣：平均總損益NT${b['mean']:,.0f}，"
                     f"5%~95%區間=[NT${b['p5']:,.0f}, NT${b['p95']:,.0f}]，"
                     f"正報酬比例={b['pct_positive']:.1f}%，p值={b['p_value']:.3f}，"
                     f"拿掉最大3筆交易後總損益NT${b['pnl_excluding_top3_ntd']:,.0f}")
        print("  bootstrap：正報酬比例={:.1f}%, p值={:.3f}, 拿掉最大3筆後損益={:,.0f}".format(
            b["pct_positive"], b["p_value"], b["pnl_excluding_top3_ntd"]))
        summary_lines.append(boot_line)

    caveat = ("\n  ⚠️ 以上全程不經過資金/部位管理(每個訊號都視為獨立成交，沒有top_n排名、"
              "沒有同時持倉上限)，PF/損益數字比真實帳戶能拿到的樂觀——這組驗證回答的是"
              "「進場/出場邏輯方向上是否穩健」，不是「我的帳戶實際能拿到的PF」")
    print(caveat.strip())
    summary_lines.append(caveat)
    summary_lines.append(
        "\n判讀方式：先看OOS PF是不是也>1(不是只看IS/全樣本)，再看bootstrap正報酬比例"
        "(想要>80%才算站得住)跟p值(越接近0越好，超過0.2以上不該當作已驗證)。即使OOS/"
        "bootstrap數字好看，別忘了上面的資金/部位管理但書——這裡驗證的是訊號邏輯方向，"
        "不是實際帳戶報酬。"
    )

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


# 之前(--squeeze-kdj-only)驗證過的「資金無限」OOS基準數字，寫死在這裡純粹是為了讓
# run_squeeze_kdj_capital_constrained_mode()的summary能做「誠實對照」，不是這裡重新跑出來的——
# 如果之後--squeeze-kdj-only的驗證結果改變(例如universe/期間設定不同)，這裡的基準數字
# 應該跟著更新，不然對照會失真。
SQUEEZE_KDJ_UNCONSTRAINED_OOS_REFERENCE = {
    "A": {"profit_factor": 5.01, "avg_hold_days": 16.2},
    "B": {"profit_factor": 3.44, "avg_hold_days": 6.1},
}


def evaluate_squeeze_kdj_capital_constrained(price_data: dict, universe: dict, starting_capital: float,
                                              is_calendar, oos_calendar, variant: str = "B",
                                              top_n: int = 3, max_concurrent_positions: int = 3,
                                              lots: int = 2) -> dict:
    """
    把squeeze_kdj_signal.run_squeeze_kdj_capital_constrained_backtest()接上這個檔案
    既有的IS/OOS + bootstrap驗證方法論，結構完全比照evaluate_combo()：IS/OOS calendar
    各自完整跑一次day-by-day回測(不是先跑一次全期間再事後切分)，因為資金/部位管理
    本身有跨日的狀態(open_positions/cooldown_until)，IS跑到最後剩下的未平倉部位、冷卻期
    不該帶進OOS那次回測——每次呼叫都是全新的trades=[]/open_positions=[]/cooldown_until={}，
    OOS那次回測對「IS最後幾天發生了什麼」完全不知情，這是跟compare_breakout.py其餘所有
    IS/OOS組合(evaluate_combo()/run_walkforward_validation()等)一致的做法，也是
    run_squeeze_kdj_exit_style_comparison_is_oos()故意指出「這裡不適用」的那個模式——
    那個函式是先一次性模擬完整段落再用entry_date切分，原因是它完全不經過day-by-day
    master_calendar迴圈(直接呼叫simulate_variant_a_trades/simulate_variant_b_trades)，
    沒有calendar可以切；這裡改用run_squeeze_kdj_capital_constrained_backtest()，本來
    就是day-by-day walk-forward，有calendar可以切，就該跟專案裡其他有calendar的回測
    一樣老老實實跑兩次。

    features_by_code(BB/KC/KDJ狀態機計算)只算一次、IS/OOS共用——這部分不受
    max_concurrent_positions/variant等資金管理參數影響，跟其他引擎的precompute_*
    共用慣例一致。

    回傳格式跟evaluate_combo()一樣：{"label", "IS", "OOS", "bootstrap", "oos_trades"}，
    方便呼叫端用同一套輸出/印法(例如_squeeze_kdj_is_oos_results_to_df()可以直接重用，
    只要把variant="A"/"B"兩次呼叫的結果組成{"A": ..., "B": ...}這種dict)。
    """
    features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)
    label = f"資金受限版 {SQUEEZE_KDJ_VARIANT_LABELS[variant]}(max_concurrent_positions={max_concurrent_positions}, top_n={top_n})"

    results = {"label": label}
    oos_trades = None
    for split_name, calendar in [("IS", is_calendar), ("OOS", oos_calendar)]:
        trades = run_squeeze_kdj_capital_constrained_backtest(
            price_data=price_data, universe=universe, master_calendar=calendar,
            starting_capital=starting_capital, variant=variant, lots=lots,
            top_n=top_n, max_concurrent_positions=max_concurrent_positions,
            features_by_code=features_by_code,
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


def run_squeeze_kdj_capital_constrained_mode(args, price_data, universe, is_calendar, oos_calendar):
    """--squeeze-kdj-capital-constrained模式：把已經驗證過(--squeeze-kdj-only)的
    squeeze+KDJ進場訊號，接上這個專案「真正會拿去模擬實戰」的資金/部位管理框架
    (top_n排名+max_concurrent_positions持倉上限+保證金查表，見
    squeeze_kdj_signal.run_squeeze_kdj_capital_constrained_backtest())，回答
    --squeeze-kdj-only刻意沒有回答的問題：「拿掉『資金無限、每個訊號都能同時成交』
    這個樂觀假設之後，實際帳戶能拿到的PF還剩多少」。

    只跑變體A跟變體B各一組(max_concurrent_positions=3, top_n=3)，不另外做
    max_concurrent_positions=1 vs 3這種維度的網格——這兩個是本來就已經驗證過方向正確
    的訊號，這裡要回答的是「資金受限後退化多少」這一個問題，不是重新做一次參數搜尋
    (參數搜尋的多重比較風險，這個repo已經在--simple-combo/--combo-search的docstring
    裡用IS PF=1.41/OOS PF=0.48的真實教訓講得很清楚了，這裡沒有必要再冒一次)；
    max_concurrent_positions=3/top_n=3是矩陣裡一個中庸、不算激進的設定，足以看出
    「有資金限制 vs 沒有」這個最核心的對比，GitHub Actions本來就已經跑得很慢，
    --squeeze-kdj-only存在的理由就是給一個快速模式，這裡沒有必要犧牲這個優點。
    """
    print("=" * 100)
    print("--squeeze-kdj-capital-constrained模式：把squeeze+KDJ訊號接上資金/部位管理框架"
          "(top_n排名+max_concurrent_positions持倉上限+保證金查表)，看真實帳戶(資金有限、"
          "不可能同時吃下所有訊號)實際能拿到的PF")
    print("=" * 100)

    results = {}
    for variant in ("A", "B"):
        print(f"\n[{SQUEEZE_KDJ_VARIANT_LABELS[variant]}] 資金受限版 "
              f"(max_concurrent_positions=3, top_n=3) ...")
        results[variant] = evaluate_squeeze_kdj_capital_constrained(
            price_data, universe, args.starting_capital, is_calendar, oos_calendar,
            variant=variant, top_n=3, max_concurrent_positions=3, lots=2,
        )

    constrained_df = _squeeze_kdj_is_oos_results_to_df(results)
    constrained_df.to_csv(os.path.join(RESULTS_DIR, "squeeze_kdj_capital_constrained_is_oos.csv"),
                           index=False, encoding="utf-8-sig")

    summary_lines = [
        "=" * 100,
        "布林+Keltner擠壓+KDJ訊號 --squeeze-kdj-capital-constrained模式"
        "(資金/部位受限版，接上top_n排名+max_concurrent_positions持倉上限+保證金查表)",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        "=" * 100,
        "\n(這個比較跟--squeeze-kdj-only共用同一個訊號定義，差別只在這裡加上了真實帳戶的"
        "資金/部位管理限制；只做多方，見squeeze_kdj_signal.py模組docstring)",
    ]
    for variant, r in results.items():
        ref = SQUEEZE_KDJ_UNCONSTRAINED_OOS_REFERENCE[variant]
        print(f"  [{r['label']}]")
        summary_lines.append(f"\n  [{r['label']}]")
        for split_name in ["IS", "OOS"]:
            stats = r[split_name]
            split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
            line = (f"    {split_full}: {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
                    f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
                    f"總損益NT${stats['total_pnl_ntd']:,.0f}, "
                    f"最大回撤NT${stats['max_drawdown_ntd']:,.0f}, 最大連續虧損{stats['max_consecutive_losses']}筆")
            print("  " + line.strip())
            summary_lines.append(line)
        b = r["bootstrap"]
        boot_line = (f"    [穩健性] OOS bootstrap 1000次重抽樣：平均總損益NT${b['mean']:,.0f}，"
                     f"5%~95%區間=[NT${b['p5']:,.0f}, NT${b['p95']:,.0f}]，"
                     f"正報酬比例={b['pct_positive']:.1f}%，p值={b['p_value']:.3f}，"
                     f"拿掉最大3筆交易後總損益NT${b['pnl_excluding_top3_ntd']:,.0f}")
        print("  bootstrap：正報酬比例={:.1f}%, p值={:.3f}, 拿掉最大3筆後損益={:,.0f}".format(
            b["pct_positive"], b["p_value"], b["pnl_excluding_top3_ntd"]))
        summary_lines.append(boot_line)

        # 這是整個練習最核心的輸出：跟之前--squeeze-kdj-only算出的「資金無限」OOS PF
        # 直接對照，誠實講清楚資金受限後是「撐住」「打折」還是「崩潰」。
        oos_pf = r["OOS"]["profit_factor"]
        unconstrained_pf = ref["profit_factor"]
        if oos_pf == float("inf") or unconstrained_pf == 0:
            verdict = "(無法直接算比例，見原始數字自行判斷)"
        elif oos_pf >= unconstrained_pf * 0.8:
            verdict = "PF大致撐住(>=資金無限版的80%)，資金限制對這個訊號的影響不大"
        elif oos_pf >= unconstrained_pf * 0.4:
            verdict = "PF明顯打折(介於資金無限版的40%~80%之間)，資金限制確實有實質影響，" \
                      "不能直接拿資金無限版的數字當實盤期待"
        elif oos_pf > 1.0:
            verdict = "PF大幅崩潰(<資金無限版的40%)，但仍>1，資金受限後優勢大幅萎縮"
        else:
            verdict = "PF崩潰到<=1，資金受限後這個訊號在真實帳戶規模下可能已經不具優勢"
        compare_line = (
            f"    ⚖️ 對照資金無限版(--squeeze-kdj-only)OOS PF={_fmt_pf(unconstrained_pf)}"
            f"(平均持有{ref['avg_hold_days']:.1f}天) vs 資金受限版OOS PF={_fmt_pf(oos_pf)}"
            f"(平均持有{r['OOS']['avg_hold_days']:.1f}天)：{verdict}"
        )
        print(compare_line.strip())
        summary_lines.append(compare_line)

    ranking_caveat = (
        "\n  ⚠️ 排名judgment call：同一天如果有超過max_concurrent_positions個空位的候選同時"
        "觸發訊號，這裡用「觸發K棒當天本身的漲幅(close_t/close_t-1 - 1)」排名、漲幅越大越"
        "優先進場——這是為了資金受限情境才新增的判斷，不是原始--squeeze-kdj-only驗證過的"
        "進場規則的一部分(原始驗證裡每個訊號都視為獨立可成交，不存在『選誰』這個問題)。"
        "這個排名規則本身完全沒有被驗證過是不是真的有效，只是在『資金有限、必須選一個』"
        "的前提下，一個跟進場規則本身相關、沒有引入新假設的務實選擇，不該被誤認成已經"
        "驗證過的訊號品質排序。"
    )
    print(ranking_caveat.strip())
    summary_lines.append(ranking_caveat)
    summary_lines.append(
        "\n判讀方式：先看OOS PF是不是也>1，再看跟資金無限版的對照(上面每個變體後面的"
        "⚖️那一行)是撐住/打折/崩潰，最後看bootstrap正報酬比例(>80%才算站得住)跟p值"
        "(越接近0越好)。這整組數字才是比較貼近『我的帳戶實際能拿到的PF』的誠實答案，"
        "--squeeze-kdj-only算出的PF=5.01/3.44本來就刻意不考慮資金限制，不該直接拿來"
        "當實盤期待(見run_squeeze_kdj_exit_style_comparison_is_oos() docstring的caveat)。"
    )

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


# ============================================================================
# --squeeze-kdj-grid：squeeze+KDJ資金受限版「風控 x 交易管理 x 進場方式」混搭網格搜尋
#
# 方法論(使用者/維護者事先定好、不可以事後改的規則，這是整個模式存在的意義)：
#   1. 網格只在樣本內(IS) calendar上跑、只用IS的數字挑贏家：選拔指標=IS獲利因子(PF)，
#      只有IS交易筆數>=SQUEEZE_KDJ_GRID_MIN_IS_TRADES(30)筆的組合有資格被選(筆數不夠的
#      組合照樣列在報告裡，只是不能當贏家)；同PF時比IS總損益，再同分時比組合編號(純粹
#      為了結果可重現，不代表任何偏好)。
#   2. 選出的「唯一一個」IS贏家，才拿去OOS跑一次+bootstrap(1000次、seed=42，跟
#      evaluate_combo()同一套欄位/慣例)，這是報告裡唯一能當「誠實頭條數字」的結果。
#   3. 另外列出IS前10名在OOS的PF，並算「全部有資格組合」IS PF跟OOS PF的Spearman排名相關，
#      回答「IS排名到底有沒有轉移到OOS」。
#   4. 每一組都在OOS跑一次，但只報告分布(幾組、OOS PF>1的比例、中位數/25%/75%分位數)，
#      當作參數穩健性檢查——刻意不印「OOS最好的組合」，因為從OOS挑最好的等於拿OOS做選擇，
#      OOS就不再是樣本外了。
#   5. 醒目印出總共測了幾組+白話的多重比較警告。
#   6. 註明OOS之前已經被看過一次(上一輪預設設定的--squeeze-kdj-capital-constrained結果)，
#      不是完全沒碰過的處女資料。
# 實作上刻意「先把全部組合的IS跑完、選出贏家，才開始跑OOS」，讓「選拔只看IS」這件事
# 在程式結構上就成立，不是靠自律。
# ============================================================================

SQUEEZE_KDJ_GRID_MIN_IS_TRADES = 30
SQUEEZE_KDJ_GRID_TOP_N = 3          # 不是網格維度，固定沿用資金受限版的預設值(見build_squeeze_kdj_grid_combos)
SQUEEZE_KDJ_GRID_ATR_PERIOD = 14    # 同上，固定沿用compare_breakout.py呼叫變體B時一貫的14天ATR
SQUEEZE_KDJ_GRID_TOP_K_REPORT = 10
SQUEEZE_KDJ_BOOTSTRAP_PASS_PCT_POSITIVE = 80.0  # 專案門檻：bootstrap正報酬比例 > 80%
SQUEEZE_KDJ_BOOTSTRAP_PASS_P_VALUE = 0.2        # 專案門檻：p值 < 0.2

# 出場配置(10組)：變體A 1組 + 變體B固定ATR停損/停利 3x2=6組 + 變體B移動停利 3組
SQUEEZE_KDJ_GRID_EXIT_CONFIGS = (
    [{"variant": "A", "atr_stop_mult": None, "atr_target_mult": None, "trailing_atr_mult": None}]
    + [{"variant": "B", "atr_stop_mult": s, "atr_target_mult": t, "trailing_atr_mult": None}
       for s in (1.0, 1.5, 2.0) for t in (2.0, 3.0)]
    + [{"variant": "B_trail", "atr_stop_mult": m, "atr_target_mult": None, "trailing_atr_mult": m}
       for m in (1.5, 2.0, 3.0)]
)
SQUEEZE_KDJ_GRID_MAX_CONCURRENT_POSITIONS = (1, 3, 5)
# 部位大小：(固定口數, 每筆風險比例)，兩者擇一
SQUEEZE_KDJ_GRID_SIZING = ((1, None), (2, None), (None, 0.01), (None, 0.02))
SQUEEZE_KDJ_GRID_MAX_HOLD_DAYS = (10, 20, 60)
SQUEEZE_KDJ_GRID_RANKING_RULES = ("trigger_return", "volume_ratio")
SQUEEZE_KDJ_GRID_ENTRY_FILTERS = (None, "above_ma60")

SQUEEZE_KDJ_RANKING_RULE_LABELS = {
    "trigger_return": "觸發K棒當天漲幅",
    "volume_ratio": "觸發K棒當天量比(當日量/20日均量)",
}
SQUEEZE_KDJ_ENTRY_FILTER_LABELS = {None: "無", "above_ma60": "觸發K棒收盤站上60日均線"}

# 上一輪真實GitHub Actions執行--squeeze-kdj-capital-constrained(預設設定：
# max_concurrent_positions=3、固定2口、最長60天、觸發K棒漲幅排名、無濾網、起始資金NT$200k、
# max_stocks=50)的結果，寫死在這裡純粹是為了讓網格模式的summary能跟「這次網格之前就已經
# 看過的OOS數字」做誠實對照(也是「OOS已經被看過一次」那句註記的依據)。跟
# SQUEEZE_KDJ_UNCONSTRAINED_OOS_REFERENCE一樣，條件改變時這裡應該跟著更新。
SQUEEZE_KDJ_CAPITAL_CONSTRAINED_DEFAULT_REFERENCE = {
    "A": {"is_pf": 0.93, "is_trades": 59, "oos_pf": 0.75, "oos_trades": 25,
          "pct_positive": 29.1, "p_value": 0.709},
    "B": {"is_pf": 0.84, "is_trades": 69, "oos_pf": 1.49, "oos_trades": 34,
          "pct_positive": 78.7, "p_value": 0.213},
}

# summarize_mr()裡要帶進網格表的統計量
SQUEEZE_KDJ_GRID_STAT_KEYS = (
    "trade_count", "profit_factor", "win_rate", "total_pnl_ntd", "max_drawdown_ntd",
    "avg_hold_days", "max_consecutive_losses", "pnl_excluding_top3_ntd",
)


def build_squeeze_kdj_grid_combos() -> list:
    """
    產生--squeeze-kdj-grid要測的全部參數組合：出場配置10組 x 最大持倉數3種 x 部位大小4種
    x 最長持有天數3種 x 同日排名規則2種 x 進場濾網2種 = 1440組，每組一個dict，
    combo_id從1開始連號(只是編號，不代表任何順序上的偏好)。

    刻意「不是」網格維度的東西(見各常數說明)：
      - starting_capital：使用者真實帳戶規模(--starting-capital)，不是拿來調到回測好看的參數。
      - top_n：固定3(資金受限版預設)，實際排名截斷是max(top_n, 空出的名額)，
        max_concurrent_positions=5時自動放寬到5，不需要另外搜尋。
      - atr_period：固定14天。
    變體B_trail的初始停損倍數 = 移動停利倍數(同一個值)，見SQUEEZE_KDJ_GRID_EXIT_CONFIGS。
    """
    combos = []
    for exit_cfg in SQUEEZE_KDJ_GRID_EXIT_CONFIGS:
        for mcp in SQUEEZE_KDJ_GRID_MAX_CONCURRENT_POSITIONS:
            for lots, risk_pct in SQUEEZE_KDJ_GRID_SIZING:
                for hold in SQUEEZE_KDJ_GRID_MAX_HOLD_DAYS:
                    for ranking in SQUEEZE_KDJ_GRID_RANKING_RULES:
                        for entry_filter in SQUEEZE_KDJ_GRID_ENTRY_FILTERS:
                            combos.append({
                                "combo_id": len(combos) + 1,
                                **exit_cfg,
                                "max_concurrent_positions": mcp,
                                "lots": lots, "risk_pct_per_trade": risk_pct,
                                "max_hold_days": hold,
                                "ranking_rule": ranking, "entry_filter": entry_filter,
                            })
    return combos


def squeeze_kdj_grid_combo_to_backtest_kwargs(combo: dict) -> dict:
    """把一組網格參數轉成run_squeeze_kdj_capital_constrained_backtest()的關鍵字參數。
    變體A不傳ATR倍數(用不到)；變體B傳停損/停利倍數；變體B_trail傳停損/移動停利倍數；
    風險預算部位時不傳lots(函式內部會改用風險預算反推口數，lots被忽略)。"""
    kwargs = {
        "variant": combo["variant"], "max_concurrent_positions": combo["max_concurrent_positions"],
        "max_hold_days": combo["max_hold_days"], "ranking_rule": combo["ranking_rule"],
        "entry_filter": combo["entry_filter"], "top_n": SQUEEZE_KDJ_GRID_TOP_N,
        "atr_period": SQUEEZE_KDJ_GRID_ATR_PERIOD,
    }
    if combo["variant"] == "B":
        kwargs["atr_stop_mult"] = combo["atr_stop_mult"]
        kwargs["atr_target_mult"] = combo["atr_target_mult"]
    elif combo["variant"] == "B_trail":
        kwargs["atr_stop_mult"] = combo["atr_stop_mult"]
        kwargs["trailing_atr_mult"] = combo["trailing_atr_mult"]
    if combo["risk_pct_per_trade"] is not None:
        kwargs["risk_pct_per_trade"] = combo["risk_pct_per_trade"]
    else:
        kwargs["lots"] = combo["lots"]
    return kwargs


def _describe_squeeze_kdj_exit(combo: dict) -> str:
    if combo["variant"] == "A":
        return "變體A(停損=觸發K棒前一根低點；停利=K衝上80後再跌破80)"
    if combo["variant"] == "B":
        return f"變體B固定ATR(停損{combo['atr_stop_mult']:.1f}倍ATR／停利{combo['atr_target_mult']:.1f}倍ATR)"
    return f"變體B移動停利(初始停損與移動停利都是{combo['trailing_atr_mult']:.1f}倍ATR，不設固定停利)"


def _describe_squeeze_kdj_sizing(combo: dict) -> str:
    # combo可能來自DataFrame的一列，None會變成NaN，所以用pd.notna判斷
    if pd.notna(combo["risk_pct_per_trade"]):
        return f"風險預算：每筆最多賠帳戶權益的{combo['risk_pct_per_trade'] * 100:.0f}%(口數依停損距離反推)"
    return f"固定{int(combo['lots'])}口"


def describe_squeeze_kdj_grid_combo(combo: dict) -> str:
    """一組網格參數的白話中文說明(summary.txt/CSV共用)。"""
    return (f"出場：{_describe_squeeze_kdj_exit(combo)}｜最多同時持有{combo['max_concurrent_positions']}檔"
            f"｜部位：{_describe_squeeze_kdj_sizing(combo)}｜最長持有{combo['max_hold_days']}天"
            f"｜同日多檔排名：{SQUEEZE_KDJ_RANKING_RULE_LABELS[combo['ranking_rule']]}"
            f"｜進場濾網：{SQUEEZE_KDJ_ENTRY_FILTER_LABELS[_entry_filter_key(combo['entry_filter'])]}")


def _entry_filter_key(value):
    """DataFrame裡的None可能變成NaN，統一還原成None才能查SQUEEZE_KDJ_ENTRY_FILTER_LABELS。"""
    return value if isinstance(value, str) else None


def rank_squeeze_kdj_grid_on_is(df: pd.DataFrame, min_is_trades: int = SQUEEZE_KDJ_GRID_MIN_IS_TRADES) -> pd.DataFrame:
    """
    只看IS欄位，幫每一組標上eligible(IS筆數>=min_is_trades)跟is_rank(只有eligible的組合
    才有名次，1=最好)。排序：IS PF由高到低 → IS總損益由高到低 → combo_id由小到大(最後這個
    只是讓同分時結果可重現)。這個函式刻意只讀is_開頭的欄位，就算df裡已經有oos_欄位也
    不會影響排名(測試會驗證這一點)。
    """
    df = df.copy()
    df["eligible"] = df["is_trade_count"] >= min_is_trades
    eligible = df[df["eligible"]].sort_values(
        ["is_profit_factor", "is_total_pnl_ntd", "combo_id"], ascending=[False, False, True],
    )
    df["is_rank"] = np.nan
    df.loc[eligible.index, "is_rank"] = np.arange(1, len(eligible) + 1)
    return df


def select_squeeze_kdj_grid_winner(df: pd.DataFrame):
    """回傳is_rank==1那一組的combo_id；沒有任何組合符合IS筆數門檻時回傳None。"""
    winners = df[df["is_rank"] == 1]
    if winners.empty:
        return None
    return int(winners.iloc[0]["combo_id"])


def _spearman_is_vs_oos_pf(df: pd.DataFrame) -> dict:
    """全部「有資格」組合的IS PF vs OOS PF Spearman排名相關，用「排名後Pearson相關」
    自己算，完全不依賴scipy(GitHub Actions沒裝scipy，見函式內註解)。PF=∞
    (只賺不賠)在排名上就是最大值，排名相關不受影響；少於3組或其中一邊完全沒有變異時
    回傳NaN。"""
    sub = df[df["eligible"]]
    n = len(sub)
    if n < 3:
        return {"rho": float("nan"), "n": n, "method": "n<3，無法計算"}
    is_pf = sub["is_profit_factor"].astype(float)
    oos_pf = sub["oos_profit_factor"].astype(float)
    if is_pf.nunique() < 2 or oos_pf.nunique() < 2:
        return {"rho": float("nan"), "n": n, "method": "PF沒有變異，無法計算"}
    # 不依賴scipy：GitHub Actions的requirements.txt沒有scipy，而pandas的
    # .corr(method="spearman")內部其實也會import scipy(pandas/core/nanops.py)，
    # 舊版「scipy沒裝就退回pandas」的寫法在GitHub上兩條路都會ImportError，
    # 讓--squeeze-kdj-grid跑完1440組之後在這裡崩潰(exit code 1)。
    # Spearman的定義就是「排名之後的Pearson相關」，pandas的rank()(平均名次處理同分，
    # 跟scipy.stats.spearmanr一樣)跟Pearson相關都不需要scipy，結果跟spearmanr完全相同。
    # PF=∞先換成很大的有限數，排名時一樣是最大值，但避免∞進到Pearson計算變成NaN。
    is_rank = is_pf.replace([np.inf], 1e18).rank(method="average")
    oos_rank = oos_pf.replace([np.inf], 1e18).rank(method="average")
    rho = float(np.corrcoef(is_rank.to_numpy(), oos_rank.to_numpy())[0, 1])
    return {"rho": rho, "n": n, "method": "排名後Pearson相關(等同Spearman，不需scipy)"}


def _interpret_spearman(rho: float) -> str:
    if rho is None or np.isnan(rho):
        return "無法計算(有資格的組合太少，或PF完全沒有變異)，沒辦法判斷IS排名有沒有轉移到OOS"
    if rho <= 0.1:
        return ("≈0或負相關：IS排名幾乎沒有轉移到OOS——IS上表現好的參數，到OOS並沒有比較好，"
                "IS選出來的「最佳」設定很可能只是雜訊")
    if rho < 0.3:
        return "弱正相關：IS排名只有很微弱地轉移到OOS，IS選出來的「最佳」設定仍有很大成分是雜訊"
    return ("中度以上正相關：IS表現好的參數區域在OOS也傾向比較好，參數之間的差異有一部分可能是真的；"
            "但這不保證IS贏家本身在OOS一定好，仍以上面IS贏家的OOS+bootstrap為準")


def _oos_pf_distribution(df: pd.DataFrame) -> dict:
    """全部組合OOS PF的分布(參數穩健性檢查用，不是拿來挑組合)。PF=∞先換成一個很大的
    有限數再算分位數(避免∞-∞變成NaN)，算完如果分位數落在那個值就還原成∞。"""
    pf = df["oos_profit_factor"].astype(float).to_numpy()
    n = len(pf)
    if n == 0:
        return {"count": 0, "pct_pf_gt_1": 0.0, "median": float("nan"), "p25": float("nan"),
                "p75": float("nan"), "zero_trade_count": 0}
    sentinel = 1e9
    finite = np.where(np.isinf(pf), sentinel, pf)

    def _q(q):
        v = float(np.percentile(finite, q))
        return float("inf") if v >= sentinel else v

    return {
        "count": n,
        "pct_pf_gt_1": float((pf > 1.0).mean() * 100),
        "median": _q(50), "p25": _q(25), "p75": _q(75),
        "zero_trade_count": int((df["oos_trade_count"] == 0).sum()),
    }


def run_squeeze_kdj_grid_search(price_data: dict, universe: dict, starting_capital: float,
                                 is_calendar, oos_calendar, combos: list = None,
                                 min_is_trades: int = SQUEEZE_KDJ_GRID_MIN_IS_TRADES,
                                 progress_every: int = 0) -> dict:
    """
    --squeeze-kdj-grid的核心：照上方區塊註解的方法論，跑完整個網格，回傳一個dict：
      "combos_df"：每組一列(參數、IS/OOS統計、IS/OOS診斷計數器、eligible、is_rank)，
                   欄位是英文key(寫CSV時才翻成中文，見SQUEEZE_KDJ_GRID_CSV_COLUMNS_ZH)
      "n_combos"：總共測了幾組
      "winner_id"/"winner"：IS贏家的combo_id，以及{"combo", "label", "IS", "OOS",
                   "bootstrap", "oos_trades", "is_diagnostics", "oos_diagnostics"}
                   (沒有任何組合符合IS筆數門檻時兩者都是None)
      "spearman"：全部有資格組合IS PF vs OOS PF的Spearman排名相關
      "oos_distribution"：全部組合OOS PF的分布
    執行順序刻意是「全部組合先跑IS → 只用IS選出贏家 → 才開始跑OOS」，讓「OOS數字不可能
    影響選拔」這件事在程式結構上就成立。

    效能：BB/KC/KDJ狀態機(precompute_squeeze_kdj_features_by_code)跟回測要查的陣列/事件表
    (precompute_squeeze_kdj_backtest_arrays)都只算一次，2880次回測(1440組 x IS/OOS)共用。

    combos：預設build_squeeze_kdj_grid_combos()的完整1440組；測試時可以傳較小的清單。
    """
    combos = combos if combos is not None else build_squeeze_kdj_grid_combos()
    features_by_code = precompute_squeeze_kdj_features_by_code(price_data, universe)
    precomputed = precompute_squeeze_kdj_backtest_arrays(
        price_data, features_by_code, atr_period=SQUEEZE_KDJ_GRID_ATR_PERIOD,
    )

    def _run(combo, calendar):
        return run_squeeze_kdj_capital_constrained_backtest(
            price_data=price_data, universe=universe, master_calendar=calendar,
            starting_capital=starting_capital, features_by_code=features_by_code,
            precomputed=precomputed, return_diagnostics=True,
            **squeeze_kdj_grid_combo_to_backtest_kwargs(combo),
        )

    # ---- 第一階段：全部組合只跑IS ----
    rows = []
    is_stats_by_id = {}
    is_diag_by_id = {}
    for n_done, combo in enumerate(combos, start=1):
        trades, diag = _run(combo, is_calendar)
        stats = summarize_mr(trades, starting_capital)
        is_stats_by_id[combo["combo_id"]] = stats
        is_diag_by_id[combo["combo_id"]] = diag
        rows.append({
            **combo,
            **{f"is_{k}": stats[k] for k in SQUEEZE_KDJ_GRID_STAT_KEYS},
            **{f"is_diag_{k}": diag[k] for k in CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS},
        })
        if progress_every and n_done % progress_every == 0:
            print(f"  [IS] 已完成 {n_done}/{len(combos)} 組", flush=True)

    df = rank_squeeze_kdj_grid_on_is(pd.DataFrame(rows), min_is_trades=min_is_trades)
    winner_id = select_squeeze_kdj_grid_winner(df)  # ← 選拔到這裡就定案，下面才開始碰OOS

    # ---- 第二階段：全部組合跑OOS(只拿來做分布/排名轉移檢查，不參與選拔) ----
    oos_cols = {f"oos_{k}": [] for k in SQUEEZE_KDJ_GRID_STAT_KEYS}
    oos_cols.update({f"oos_diag_{k}": [] for k in CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS})
    winner_oos = None
    combo_by_id = {c["combo_id"]: c for c in combos}
    for n_done, combo_id in enumerate(df["combo_id"], start=1):
        combo = combo_by_id[int(combo_id)]
        trades, diag = _run(combo, oos_calendar)
        stats = summarize_mr(trades, starting_capital)
        for k in SQUEEZE_KDJ_GRID_STAT_KEYS:
            oos_cols[f"oos_{k}"].append(stats[k])
        for k in CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS:
            oos_cols[f"oos_diag_{k}"].append(diag[k])
        if winner_id is not None and int(combo_id) == winner_id:
            winner_oos = (trades, stats, diag)
        if progress_every and n_done % progress_every == 0:
            print(f"  [OOS] 已完成 {n_done}/{len(combos)} 組", flush=True)
    for col, values in oos_cols.items():
        df[col] = values

    winner = None
    if winner_id is not None:
        oos_trades, oos_stats, oos_diag = winner_oos
        bootstrap_results = bootstrap_resample_pnl(oos_trades, n_resamples=1000, seed=42)
        winner = {
            "combo": combo_by_id[winner_id],
            "label": describe_squeeze_kdj_grid_combo(combo_by_id[winner_id]),
            "IS": is_stats_by_id[winner_id], "OOS": oos_stats,
            "bootstrap": {
                **summarize_bootstrap(bootstrap_results),
                "p_value": bootstrap_p_value(bootstrap_results),
                "pnl_excluding_top3_ntd": pnl_excluding_top_n_trades(oos_trades, n=3),
            },
            "oos_trades": oos_trades,
            "is_diagnostics": is_diag_by_id[winner_id], "oos_diagnostics": oos_diag,
        }

    return {
        "combos_df": df, "n_combos": len(combos), "winner_id": winner_id, "winner": winner,
        "spearman": _spearman_is_vs_oos_pf(df), "oos_distribution": _oos_pf_distribution(df),
        "min_is_trades": min_is_trades,
    }


SQUEEZE_KDJ_GRID_DIAG_LABELS_ZH = {
    "candidates_total": "候選總數",
    "skipped_entry_filter": "進場濾網擋掉",
    "skipped_no_slot": "名額已滿沒輪到",
    "skipped_invalid_stop": "停損價無效略過",
    "skipped_risk_lots_lt1": "風險口數不足1口略過",
    "skipped_single_margin_cap": "單筆保證金上限略過",
    "skipped_total_margin_cap": "總保證金上限略過",
}
SQUEEZE_KDJ_GRID_STAT_LABELS_ZH = {
    "trade_count": "交易筆數", "profit_factor": "獲利因子PF", "win_rate": "勝率(%)",
    "total_pnl_ntd": "總損益(NT$)", "max_drawdown_ntd": "最大回撤(NT$)",
    "avg_hold_days": "平均持有天數", "max_consecutive_losses": "最大連續虧損筆數",
    "pnl_excluding_top3_ntd": "拿掉最大3筆後損益(NT$)",
}

# 網格CSV的中文欄名(依輸出順序)。這個檔案既有的CSV都還是直接用英文key當欄名，
# 這裡是第一個照使用者「欄位翻成中文」要求輸出中文欄名的CSV；IS/OOS/PF/NT$/ATR這些
# 縮寫刻意保留，跟summary.txt一貫的寫法(例如「樣本內(IS)」「PF=」)一致。
SQUEEZE_KDJ_GRID_CSV_COLUMNS_ZH = {
    "is_rank": "IS排名(只有符合資格的組合有名次)",
    "eligible": f"符合選拔資格(IS筆數>={SQUEEZE_KDJ_GRID_MIN_IS_TRADES})",
    "combo_id": "組合編號",
    "description": "組合說明(白話)",
    "variant": "出場變體(A/B/B_trail)",
    "atr_stop_mult": "停損ATR倍數",
    "atr_target_mult": "停利ATR倍數",
    "trailing_atr_mult": "移動停利ATR倍數",
    "max_concurrent_positions": "最多同時持倉數",
    "lots": "固定口數",
    "risk_pct_per_trade": "每筆風險比例",
    "max_hold_days": "最長持有天數",
    "ranking_rule": "同日多檔排名規則",
    "entry_filter": "進場濾網",
    **{f"is_{k}": f"IS{v}" for k, v in SQUEEZE_KDJ_GRID_STAT_LABELS_ZH.items()},
    **{f"oos_{k}": f"OOS{v}" for k, v in SQUEEZE_KDJ_GRID_STAT_LABELS_ZH.items()},
    **{f"is_diag_{k}": f"IS診斷_{v}" for k, v in SQUEEZE_KDJ_GRID_DIAG_LABELS_ZH.items()},
    **{f"oos_diag_{k}": f"OOS診斷_{v}" for k, v in SQUEEZE_KDJ_GRID_DIAG_LABELS_ZH.items()},
}


def squeeze_kdj_grid_df_to_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """把run_squeeze_kdj_grid_search()的combos_df整理成要寫進CSV的樣子：加上白話說明、
    排名/濾網/排名規則翻成中文、依IS排名排序(有資格的在前、沒資格的在後依組合編號)、
    欄名翻成中文。"""
    out = df.copy()
    out["description"] = [describe_squeeze_kdj_grid_combo(r) for r in out.to_dict("records")]
    out["ranking_rule"] = out["ranking_rule"].map(SQUEEZE_KDJ_RANKING_RULE_LABELS)
    out["entry_filter"] = out["entry_filter"].map(lambda v: SQUEEZE_KDJ_ENTRY_FILTER_LABELS[_entry_filter_key(v)])
    out = out.sort_values(["is_rank", "combo_id"], na_position="last")
    cols = [c for c in SQUEEZE_KDJ_GRID_CSV_COLUMNS_ZH if c in out.columns]
    return out[cols].rename(columns=SQUEEZE_KDJ_GRID_CSV_COLUMNS_ZH)


def _squeeze_kdj_bootstrap_verdict(b: dict) -> str:
    passes = (b["pct_positive"] > SQUEEZE_KDJ_BOOTSTRAP_PASS_PCT_POSITIVE
              and b["p_value"] < SQUEEZE_KDJ_BOOTSTRAP_PASS_P_VALUE)
    if passes:
        verdict = (f"✅ 通過專案門檻(bootstrap正報酬比例>{SQUEEZE_KDJ_BOOTSTRAP_PASS_PCT_POSITIVE:.0f}% "
                   f"且 p值<{SQUEEZE_KDJ_BOOTSTRAP_PASS_P_VALUE})")
    else:
        verdict = (f"❌ 沒有通過專案門檻(需要bootstrap正報酬比例>{SQUEEZE_KDJ_BOOTSTRAP_PASS_PCT_POSITIVE:.0f}% "
                   f"且 p值<{SQUEEZE_KDJ_BOOTSTRAP_PASS_P_VALUE}；實際{b['pct_positive']:.1f}% / "
                   f"p={b['p_value']:.3f})——這組IS贏家在OOS不能算已驗證的優勢")
    if b["pnl_excluding_top3_ntd"] < 0:
        verdict += "；另外拿掉OOS最大3筆交易後總損益轉負，獲利高度依賴少數幾筆交易"
    return verdict


def _fmt_stats_line(stats: dict) -> str:
    return (f"{stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, 勝率={stats['win_rate']:.1f}%, "
            f"平均持有{stats['avg_hold_days']:.1f}天, 總損益NT${stats['total_pnl_ntd']:,.0f}, "
            f"最大回撤NT${stats['max_drawdown_ntd']:,.0f}, 最大連續虧損{stats['max_consecutive_losses']}筆")


def _fmt_diag_line(diag: dict) -> str:
    return "，".join(f"{SQUEEZE_KDJ_GRID_DIAG_LABELS_ZH[k]}={diag[k]}" for k in CAPITAL_CONSTRAINED_DIAGNOSTIC_KEYS)


def _find_default_capital_constrained_combo(df: pd.DataFrame, variant: str):
    """在網格裡找出「上一輪--squeeze-kdj-capital-constrained預設設定」那一組(它本來就是
    網格裡的其中一組)：max_concurrent_positions=3、固定2口、最長60天、觸發K棒漲幅排名、
    無濾網；變體B另外要停損1.0/停利2.0倍ATR。找不到時回傳None。"""
    mask = (
        (df["variant"] == variant) & (df["max_concurrent_positions"] == 3) & (df["lots"] == 2)
        & df["risk_pct_per_trade"].isna() & (df["max_hold_days"] == 60)
        & (df["ranking_rule"] == "trigger_return") & df["entry_filter"].isna()
    )
    if variant == "B":
        mask &= (df["atr_stop_mult"] == 1.0) & (df["atr_target_mult"] == 2.0)
    hit = df[mask]
    return None if hit.empty else hit.iloc[0]


def run_squeeze_kdj_grid_mode(args, price_data, universe, is_calendar, oos_calendar):
    """--squeeze-kdj-grid模式：squeeze+KDJ資金受限版的「風控 x 交易管理 x 進場方式」混搭
    網格搜尋(1440組，見build_squeeze_kdj_grid_combos())，方法論見上方區塊註解跟
    run_squeeze_kdj_grid_search() docstring——重點是：只用IS選出一個贏家，拿那一個贏家
    的OOS+bootstrap當誠實的頭條數字；其餘OOS資訊(IS前10名的OOS PF、Spearman排名相關、
    全部組合OOS PF分布)都只用來判斷「IS排名有沒有意義」，不拿來挑組合。

    跟--squeeze-kdj-only/--squeeze-kdj-capital-constrained一樣跳過完整6階段流程。
    --start/--end/--starting-capital/--max-stocks仍然有效；starting_capital刻意不是網格
    維度(它是使用者真實帳戶規模)。輸出squeeze_kdj_grid_all_combos.csv(中文欄名)+summary.txt。
    """
    print("=" * 100)
    print("--squeeze-kdj-grid模式：squeeze+KDJ資金受限版 風控x交易管理x進場方式 混搭網格搜尋"
          "(只在IS選贏家，贏家才跑OOS+bootstrap)")
    print("=" * 100)

    n_expected = len(build_squeeze_kdj_grid_combos())
    print(f"\n⚠️ 本次一共要測 {n_expected} 組參數組合(IS、OOS各跑一次)...", flush=True)
    result = run_squeeze_kdj_grid_search(
        price_data, universe, args.starting_capital, is_calendar, oos_calendar, progress_every=200,
    )
    df = result["combos_df"]
    n_combos = result["n_combos"]
    min_is_trades = result["min_is_trades"]
    n_eligible = int(df["eligible"].sum())

    csv_df = squeeze_kdj_grid_df_to_csv_frame(df)
    csv_df.to_csv(os.path.join(RESULTS_DIR, "squeeze_kdj_grid_all_combos.csv"), index=False, encoding="utf-8-sig")

    lines = [
        "=" * 100,
        "布林+Keltner擠壓+KDJ訊號 --squeeze-kdj-grid模式(資金受限版：風控 x 交易管理 x 進場方式 混搭網格搜尋)",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        "=" * 100,
        "",
        "#" * 100,
        f"### 本次一共測試了 {n_combos} 組參數組合(出場配置10 x 最大持倉數3 x 部位大小4 x 最長持有天數3 "
        f"x 同日排名規則2 x 進場濾網2) ###",
        "#" * 100,
        f"⚠️ 多重比較警告(白話)：同時測了{n_combos}組，就算每一組其實都沒有真正的優勢，純靠運氣也幾乎"
        "一定會有幾組在樣本內(IS)看起來很漂亮。IS贏家的IS數字是從上千組裡「挑出來」的，必然偏樂觀，"
        "不能拿來當實盤期待；比較誠實的只有「IS贏家拿到OOS跑一次」的結果+bootstrap。",
        "⚠️ OOS不是完全沒碰過的資料：這段OOS期間在上一輪--squeeze-kdj-capital-constrained(預設設定)"
        "就已經看過一次結果，這次網格的維度設計多少受到那次結果影響，OOS結論的可信度要再打一點折扣。",
        "ℹ️ 起始資金不是網格維度：它是你真實帳戶的規模(--starting-capital)，不是拿來調到回測好看的參數。",
        f"選拔規則(事先定好)：只看IS；IS交易筆數>={min_is_trades}筆才有資格；IS PF最高者勝出，同PF比IS總損益。"
        f"符合資格的組合：{n_eligible}/{n_combos}組。",
    ]

    winner = result["winner"]
    lines.append("\n【IS贏家 → OOS一次 + bootstrap(誠實頭條數字)】")
    if winner is None:
        lines.append(f"  沒有任何組合的IS交易筆數達到{min_is_trades}筆，無法選出贏家(這本身就是結果："
                     "資金受限後訊號太稀疏，沒辦法在IS累積足夠樣本)。")
    else:
        lines.append(f"  組合編號#{winner['combo']['combo_id']}：{winner['label']}")
        lines.append(f"    樣本內(IS): {_fmt_stats_line(winner['IS'])}")
        lines.append(f"    樣本外(OOS) ← 較誠實的參考依據: {_fmt_stats_line(winner['OOS'])}")
        b = winner["bootstrap"]
        lines.append(f"    [穩健性] OOS bootstrap 1000次重抽樣：平均總損益NT${b['mean']:,.0f}，"
                     f"5%~95%區間=[NT${b['p5']:,.0f}, NT${b['p95']:,.0f}]，正報酬比例={b['pct_positive']:.1f}%，"
                     f"p值={b['p_value']:.3f}，拿掉最大3筆交易後總損益NT${b['pnl_excluding_top3_ntd']:,.0f}")
        lines.append(f"    判定：{_squeeze_kdj_bootstrap_verdict(b)}")
        lines.append(f"    [診斷] IS：{_fmt_diag_line(winner['is_diagnostics'])}")
        lines.append(f"    [診斷] OOS：{_fmt_diag_line(winner['oos_diagnostics'])}")

    lines.append(f"\n【IS前{SQUEEZE_KDJ_GRID_TOP_K_REPORT}名在OOS的表現(看IS排名有沒有轉移到OOS，不是拿來挑組合)】")
    top = df[df["is_rank"].notna()].sort_values("is_rank").head(SQUEEZE_KDJ_GRID_TOP_K_REPORT)
    if top.empty:
        lines.append("  (沒有符合資格的組合)")
    for r in top.to_dict("records"):
        lines.append(
            f"  IS第{int(r['is_rank'])}名 #{r['combo_id']}：IS {r['is_trade_count']}筆 PF={_fmt_pf(r['is_profit_factor'])} "
            f"損益NT${r['is_total_pnl_ntd']:,.0f} → OOS {r['oos_trade_count']}筆 PF={_fmt_pf(r['oos_profit_factor'])} "
            f"損益NT${r['oos_total_pnl_ntd']:,.0f}｜{describe_squeeze_kdj_grid_combo(r)}"
        )

    sp = result["spearman"]
    rho_txt = "NaN" if np.isnan(sp["rho"]) else f"{sp['rho']:.3f}"
    lines.append(f"\n【IS排名能不能轉移到OOS】全部{sp['n']}組有資格組合的IS PF vs OOS PF Spearman排名相關"
                 f" = {rho_txt}({sp['method']})")
    lines.append(f"  解讀：{_interpret_spearman(sp['rho'])}")

    dist = result["oos_distribution"]
    lines.append(f"\n【全部{dist['count']}組的OOS PF分布(參數穩健性檢查)】")
    lines.append(f"  OOS PF>1的比例={dist['pct_pf_gt_1']:.1f}%，中位數={_fmt_pf(dist['median'])}，"
                 f"25%分位數={_fmt_pf(dist['p25'])}，75%分位數={_fmt_pf(dist['p75'])}"
                 f"(其中{dist['zero_trade_count']}組OOS完全沒有交易，PF依summarize_mr()慣例記為0)")
    lines.append("  ⚠️ 這張分布表只用來看「整片參數空間在OOS大致賺不賺」，絕對不要從這裡挑OOS表現最好的"
                 "組合來用——那等於拿OOS做選擇，OOS就不再是樣本外，挑出來的數字跟IS贏家的IS數字一樣會偏樂觀。")

    lines.append("\n【對照上一輪預設設定(--squeeze-kdj-capital-constrained：最多3檔、固定2口、最長60天、"
                 "觸發K棒漲幅排名、無濾網)】")
    for variant in ("A", "B"):
        ref = SQUEEZE_KDJ_CAPITAL_CONSTRAINED_DEFAULT_REFERENCE[variant]
        row = _find_default_capital_constrained_combo(df, variant)
        ref_txt = (f"上一輪GitHub Actions實際結果：IS PF={ref['is_pf']:.2f}({ref['is_trades']}筆)、"
                   f"OOS PF={ref['oos_pf']:.2f}({ref['oos_trades']}筆)、bootstrap正報酬{ref['pct_positive']:.1f}%、"
                   f"p={ref['p_value']:.3f}")
        if row is None:
            lines.append(f"  變體{variant}預設設定：{ref_txt}；(這次網格裡找不到對應組合)")
            continue
        rank_txt = f"IS第{int(row['is_rank'])}名" if pd.notna(row["is_rank"]) else "IS筆數不足、不參與排名"
        lines.append(f"  變體{variant}預設設定(網格#{int(row['combo_id'])}，{rank_txt})：這次重跑 IS PF="
                     f"{_fmt_pf(row['is_profit_factor'])}({int(row['is_trade_count'])}筆)、OOS PF="
                     f"{_fmt_pf(row['oos_profit_factor'])}({int(row['oos_trade_count'])}筆)；{ref_txt}")
    lines.append("  (IS贏家的IS數字比預設設定好是預期中的事——它本來就是在IS上挑出來的；"
                 "真正要比的是IS贏家的OOS+bootstrap有沒有比預設設定好、有沒有過專案門檻。)")
    lines.append(f"\n完整{n_combos}組明細(含IS/OOS統計與診斷計數器)見squeeze_kdj_grid_all_combos.csv。")

    summary_text = "\n".join(lines)
    print(summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")
    return result


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


DEFAULT_WALKFORWARD_FOLDS = 3  # 見run_walkforward_validation()docstring：時間預算考量下的保守預設值
WALKFORWARD_MIN_TRAIN_DAYS = 60  # 對應引擎裡「pos < 60」的最小暖身天數安全檢查(見下方docstring)


def run_walkforward_validation(price_data, indicators_by_code, regime_series, master_calendar,
                                starting_capital, extra_kwargs, execution_kwargs, n_folds, has_chip):
    """
    走勢前進(Walk-Forward)分折驗證：把master_calendar切成n_folds+1個等長區塊
    (chunks[0], chunks[1], ..., chunks[n_folds])，第i折(i=0..n_folds-1)的訓練窗口
    = chunks[0..i]接起來(擴張視窗，從最開始一路累積到第i折開始前)，測試窗口 =
    chunks[i+1](緊接在訓練窗口之後、訓練時完全沒看過的一段連續日期)。這是刻意選的
    最簡單切法(不是唯一合理的切法)，白話比喻：訓練窗口像是「模擬考題庫」，測試窗口
    是「正式大考」，每一折的大考範圍都是前面模擬考完全沒出現過的一段，而且題庫
    只會越滾越大(擴張視窗)，不會用到考試當下還沒發生的未來資料。

    每一折都用訓練窗口重新跑一次完整的選股流程(訊號拆解 → 訊號組合比較 → 門檻比較 →
    ATR敏感度網格 → 出場配置比較)，選出這一折自己的「最佳組合」，再原封不動套到這一折
    的測試窗口驗證一次。這樣可以看出「同一組訊號/門檻/出場配置是不是每一折都被選中」
    (穩定=可信，每折選到不同組合=不穩定，比較像是在追過去這段歷史的雜訊，不是抓到
    真正跨時間都成立的優勢)。

    ⚠️ 兩個明確的簡化/取捨(時間預算考量，見compare_breakout.py docstring)：
    1. **不重跑突破窗口比較(階段0)跟突破風格比較(階段0.7)**：這兩階段選出的
       breakout_window/require_hard_breakout，直接沿用外層(用全部歷史一次選出)的結果，
       透過extra_kwargs傳進來，每一折不會重新選一次。這是一個需要驗證、目前沒有驗證的
       簡化假設：這裡驗證的是「訊號/門檻/出場配置」的穩定性，不是「突破窗口/突破風格」
       的穩定性——理由是這兩者理論上對「市場regime」的敏感度比訊號權重/門檻組合低
       (窗口/風格改變的是「用哪個尺度定義突破」，不是「哪個訊號有效」)，但這是個判斷，
       不是實測結果。
    2. **每一折都固定跑移動停利模式**(ATR敏感度網格+出場配置比較這兩個階段本身就是
       僅移動停利模式使用的階段)，不管外層main()有沒有加`--use-trailing-stop`。這是因為
       走勢前進驗證的核心是「訊號/門檻/出場配置的穩定性」，而移動停利模式底下的出場配置
       選項(天數上限/延遲啟動/跳空上限/保本停損)本身就是這次要驗證穩定性的一部分，固定
       天數模式沒有這些可比較的維度。

    price_data/indicators_by_code/regime_series：main()已經對完整歷史下載/預先算好一次的
    結果，這裡直接重複使用(不重新下載、不重新precompute)，每一折只是換一段calendar去
    呼叫已經內建no-I/O的比較函式，這是符合GitHub Actions時間預算(~2.5小時)的關鍵設計—
    資料下載跟指標precompute只做一次，跟fold數量無關。

    extra_kwargs：預期已經含有breakout_window/require_hard_breakout(外層選好的突破窗口/
    風格)，以及ex_dividend_dates_by_code/slippage_pct這些「回測寫不寫實」的執行細節。
    execution_kwargs：部位大小規則(lots或risk_pct_per_trade、max_concurrent_positions)，
    篩選階段(訊號拆解/門檻比較)跟main()一樣不套用，只有ATR網格/出場配置比較/最終測試
    套用，維持跟main()一致的「execution_kwargs一致性」設計。

    回傳DataFrame，一列一折，找不到任何可用折(全部因為訓練窗口太短被跳過)時回傳空
    DataFrame(呼叫端要自己檢查empty，不會拋例外)。
    """
    n = len(master_calendar)
    total_chunks = n_folds + 1
    bounds = [int(round(k * n / total_chunks)) for k in range(total_chunks + 1)]

    signal_names = [s for s in BREAKOUT_SIGNAL_NAMES if has_chip or s not in CHIP_DEPENDENT_SIGNALS]
    hold_days = TRAILING_STOP_MAX_HOLD_DAYS

    rows = []
    for i in range(n_folds):
        train_calendar = master_calendar[bounds[0]:bounds[i + 1]]
        test_calendar = master_calendar[bounds[i + 1]:bounds[i + 2]]

        if len(train_calendar) < WALKFORWARD_MIN_TRAIN_DAYS or len(test_calendar) == 0:
            print(f"  [walk-forward 第{i + 1}折] 訓練窗口({len(train_calendar)}天)或測試窗口"
                  f"({len(test_calendar)}天)太短，跳過這一折", flush=True)
            continue

        print(f"  [walk-forward 第{i + 1}/{n_folds}折] 訓練={train_calendar[0].date()}~"
              f"{train_calendar[-1].date()}({len(train_calendar)}天)，"
              f"測試={test_calendar[0].date()}~{test_calendar[-1].date()}({len(test_calendar)}天) ...",
              flush=True)

        ablation_df, signal_names_used = run_signal_ablation(
            price_data, indicators_by_code, regime_series, train_calendar,
            starting_capital, hold_days, has_chip=has_chip, extra_kwargs=extra_kwargs,
        )
        winning_signals, _ = select_winning_signals(ablation_df)
        auto_signal_weights = {name: 1.0 for name in winning_signals}
        auto_signal_label = f"自動篩選({','.join(SIGNAL_LABELS[s] for s in winning_signals)})"

        combo_df = run_signal_combo_comparison(
            price_data, indicators_by_code, regime_series, train_calendar,
            starting_capital, hold_days, auto_signal_label, auto_signal_weights,
            extra_kwargs=extra_kwargs,
        )
        combo_label, winning_signal_weights = select_winning_signal_combo(
            combo_df, auto_signal_label, auto_signal_weights,
        )

        gate_df = run_gate_comparison(
            price_data, indicators_by_code, regime_series, train_calendar,
            starting_capital, hold_days, list(winning_signal_weights.keys()), has_chip=has_chip,
            extra_kwargs=extra_kwargs,
        )
        winning_gate_label, winning_gate_kwargs = select_winning_gate(gate_df)

        atr_grid_df, (atr_stop_mult, trailing_atr_mult) = run_atr_sensitivity_grid(
            price_data, indicators_by_code, regime_series, train_calendar,
            starting_capital, winning_signal_weights, winning_gate_kwargs, extra_kwargs,
            execution_kwargs=execution_kwargs,
        )

        exit_style_df = run_exit_style_comparison(
            price_data, indicators_by_code, regime_series, train_calendar,
            starting_capital, winning_signal_weights, winning_gate_kwargs,
            atr_stop_mult, trailing_atr_mult, extra_kwargs,
            execution_kwargs=execution_kwargs,
        )
        exit_style_label, winning_hold_days, winning_exit_extra = select_winning_exit_style(exit_style_df)

        test_trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=test_calendar, max_hold_days=winning_hold_days, starting_capital=starting_capital,
            allow_short=True, signal_weights=winning_signal_weights, atr_stop_mult=atr_stop_mult,
            use_trailing_stop=True, trailing_atr_mult=trailing_atr_mult,
            **winning_gate_kwargs, **extra_kwargs, **winning_exit_extra, **execution_kwargs,
        )
        test_stats = summarize_mr(test_trades, starting_capital)

        rows.append({
            "fold": i + 1,
            "train_start": train_calendar[0].date().isoformat(), "train_end": train_calendar[-1].date().isoformat(),
            "test_start": test_calendar[0].date().isoformat(), "test_end": test_calendar[-1].date().isoformat(),
            "signal_combo": combo_label, "gate": winning_gate_label, "exit_style": exit_style_label,
            "atr_stop_mult": atr_stop_mult, "trailing_atr_mult": trailing_atr_mult,
            "trade_count": test_stats["trade_count"], "profit_factor": test_stats["profit_factor"],
            "win_rate": test_stats["win_rate"], "total_pnl_ntd": test_stats["total_pnl_ntd"],
            "avg_hold_days": test_stats["avg_hold_days"],
        })
        print(f"    → 訊號組合={combo_label}, 門檻={winning_gate_label}, 出場={exit_style_label} "
              f"-> 測試期{test_stats['trade_count']}筆, PF={_fmt_pf(test_stats['profit_factor'])}, "
              f"損益={test_stats['total_pnl_ntd']:,.0f}", flush=True)

    return pd.DataFrame(rows)


DEFAULT_FIXED_COMBO_WALKFORWARD_FOLDS = 6  # 見run_fixed_combo_walkforward()docstring：規則固定、
# 不用每折重跑整套選擇流程，成本低很多，所以建議值比run_walkforward_validation()的3折高


def run_fixed_combo_walkforward(price_data, indicators_by_code, regime_series, master_calendar,
                                 starting_capital, extra_kwargs, execution_kwargs, n_folds,
                                 fixed_combo=FIXED_WALKFORWARD_COMBO) -> pd.DataFrame:
    """
    把master_calendar切成n_folds個等長、完全不重疊、彼此獨立的連續區塊，每一塊都直接套用
    同一組「固定死」的規則(fixed_combo，不重新挑選任何訊號/門檻/ATR倍數/出場配置)，各自
    回測一次。跟run_walkforward_validation()(每折重新選一次組合)是互補、不是取代關係：
    這裡驗證的是「一組已經定案的具體規則，本身跨時間穩不穩定」，因為規則是固定的、
    不是每折現選的，這裡沒有run_walkforward_validation()那種「每折都在挑當下表現最好的
    答案」的多重比較風險，所以可以放心切更多折(不用重跑整套選擇流程，每折只是單純
    backtest一次，成本低很多)。

    ⚠️ 這不是完全乾淨的盲測：fixed_combo這組規則本身，是从過去兩輪用全部歷史跑過的
    完整選擇流程裡「觀察到」被選中的候選，這些折涵蓋的日期，有極大部分正是當初挑出
    這組規則所使用的同一份歷史——不是先固定規則、才第一次看到這些資料。這裡驗證的是
    「這組已知規則，在切成多段獨立區間分別看時是否維持一致的表現」，不是嚴格意義上
    「這組規則在全新、從未用來挑選過任何東西的資料上」的驗證(後者是cross_period_validation.py
    那種完全跳到不重疊歷史年份的做法在做的事)。

    每個區塊只需要滿足這套引擎既有的60日暖身門檻(scan_momentum_breakout_candidates()裡
    pos<60會跳過)，不需要額外的訓練窗口——因為規則不是從這個區塊的資料裡選出來的，這個
    區塊前面的歷史(precompute階段已經算好的完整序列)已經提供了指標需要的暖身資料。

    回傳DataFrame，一列一個區塊：區塊編號、起訖日期、交易數/PF/勝率/總損益/平均持有天數。
    """
    n = len(master_calendar)
    bounds = [int(round(k * n / n_folds)) for k in range(n_folds + 1)]

    exit_kwargs = dict(fixed_combo["exit_kwargs"])
    max_hold_days = exit_kwargs.pop("max_hold_days_override", TRAILING_STOP_MAX_HOLD_DAYS)

    rows = []
    for i in range(n_folds):
        chunk = master_calendar[bounds[i]:bounds[i + 1]]
        if len(chunk) == 0:
            print(f"  [固定規則walk-forward 第{i + 1}折] 區塊天數為0，跳過這一折", flush=True)
            continue

        trades = run_momentum_breakout_backtest(
            price_data=price_data, indicators_by_code=indicators_by_code, regime_series=regime_series,
            master_calendar=chunk, max_hold_days=max_hold_days, starting_capital=starting_capital,
            allow_short=True, signal_weights=fixed_combo["signal_weights"],
            atr_stop_mult=fixed_combo["atr_stop_mult"], use_trailing_stop=True,
            trailing_atr_mult=fixed_combo["trailing_atr_mult"],
            breakout_window=fixed_combo["breakout_window"],
            require_hard_breakout=fixed_combo["require_hard_breakout"],
            **fixed_combo["gate_kwargs"], **extra_kwargs, **exit_kwargs, **execution_kwargs,
        )
        stats = summarize_mr(trades, starting_capital)
        if stats["trade_count"] < MIN_TRADES_FOR_GATE_RANKING:
            print(f"  [固定規則walk-forward 第{i + 1}/{n_folds}折] "
                  f"{chunk[0].date()}~{chunk[-1].date()}：交易筆數({stats['trade_count']}筆)"
                  f"不足{MIN_TRADES_FOR_GATE_RANKING}筆，跳過這一折(不計入統計)", flush=True)
            continue

        rows.append({
            "fold": i + 1,
            "period_start": chunk[0].date().isoformat(), "period_end": chunk[-1].date().isoformat(),
            "trade_count": stats["trade_count"], "profit_factor": stats["profit_factor"],
            "win_rate": stats["win_rate"], "total_pnl_ntd": stats["total_pnl_ntd"],
            "avg_hold_days": stats["avg_hold_days"],
        })
        print(f"  [固定規則walk-forward 第{i + 1}/{n_folds}折] {chunk[0].date()}~{chunk[-1].date()} "
              f"-> {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
              f"勝率={stats['win_rate']:.1f}%, 損益={stats['total_pnl_ntd']:,.0f}", flush=True)

    return pd.DataFrame(rows)


def run_simple_combo_mode(args, price_data, indicators_by_code, regime_series,
                           is_calendar, oos_calendar, extra_kwargs, execution_kwargs):
    """--simple-combo模式的完整流程，跟main()的6階段自動搜尋流程(window→style→訊號自動搜尋→
    門檻網格→ATR網格→出場配置網格)完全分開、互相獨立：

    1. 固定SIMPLE_COMBO_SIGNAL_WEIGHTS，只在IS內跑SIMPLE_COMBO_VARIANTS三組累加門檻比較
       (run_simple_combo_comparison)，印出IS PF/勝率/交易數/總損益讓使用者自己判斷「加這道
       濾網有沒有用」——不自動選贏家，三組都完整列出。
    2. 直接取SIMPLE_COMBO_VARIANTS的最後一組(變體3，門檻疊到最滿的那組，對應使用者要求
       「最終要測的那個設定」)，用main()既有的evaluate_combo()重用IS/OOS切分+bootstrap
       穩健性檢查的邏輯(不重新實作)，產出跟main()最終驗證階段同樣格式的IS/OOS結果。

    突破窗口/突破風格維持引擎預設(20日、硬門檻breakout)，ATR停損/移動停利倍數直接用
    --atr-stop-mult/--trailing-atr-mult(不給--trailing-atr-mult時，照引擎既有的慣例退回
    跟--atr-stop-mult同一個值，見momentum_breakout_engine.try_enter_breakout()docstring)，
    刻意不另外跑ATR敏感度網格——用网格搜出來的數字，等於是又在IS資料上做了一輪挑選，
    會重新引入這個模式原本要避開的多重比較問題，所以直接用引擎/CLI本來就有的預設值。
    """
    print("=" * 100)
    print("--simple-combo模式：跳過完整6階段IS自動搜尋，只測固定訊號組合 + 3組累加門檻變體")
    print("(存在理由：完整流程連續6輪在同一份IS資料上挑贏家，疊加起來的多重比較問題曾經讓"
          "選出的「最佳組合」IS PF=1.41但OOS PF=0.48、bootstrap正報酬比例只有5.2%，"
          "這裡用固定不挑的組合避開同一個風險)")
    print("=" * 100)

    hold_days = TRAILING_STOP_MAX_HOLD_DAYS
    atr_stop_mult = args.atr_stop_mult
    trailing_atr_mult = args.trailing_atr_mult if args.trailing_atr_mult is not None else atr_stop_mult
    print(f"固定訊號組合：{SIMPLE_COMBO_SIGNAL_WEIGHTS}　"
          f"出場：移動停利(初始停損x{atr_stop_mult}, 移動停利x{trailing_atr_mult})　"
          f"突破窗口/風格：維持引擎預設(20日創新高+硬門檻)\n")

    print("[門檻變體比較] 固定訊號組合 + 三組累加門檻(只在IS內) ...")
    variant_df = run_simple_combo_comparison(
        price_data, indicators_by_code, regime_series, is_calendar,
        args.starting_capital, hold_days, atr_stop_mult, trailing_atr_mult, extra_kwargs=extra_kwargs,
    )
    variant_df.to_csv(os.path.join(RESULTS_DIR, "simple_combo_variants.csv"),
                       index=False, encoding="utf-8-sig")

    final_label, final_gate_kwargs = SIMPLE_COMBO_VARIANTS[-1]
    print(f"\n[最終驗證] 取{final_label}，跑IS vs OOS + bootstrap穩健性檢查 ...")
    final_result = evaluate_combo(
        f"簡化模式最終組合({final_label})", price_data, indicators_by_code, regime_series,
        is_calendar, oos_calendar, args.starting_capital, hold_days, SIMPLE_COMBO_SIGNAL_WEIGHTS,
        final_gate_kwargs, atr_stop_mult, trailing_atr_mult, True, extra_kwargs, execution_kwargs,
    )
    print(f"  [{final_result['label']}]")
    print(f"    IS  -> {final_result['IS']['trade_count']}筆, PF={_fmt_pf(final_result['IS']['profit_factor'])}, "
          f"平均持有{final_result['IS']['avg_hold_days']:.1f}天, 損益={final_result['IS']['total_pnl_ntd']:,.0f}")
    print(f"    OOS -> {final_result['OOS']['trade_count']}筆, PF={_fmt_pf(final_result['OOS']['profit_factor'])}, "
          f"平均持有{final_result['OOS']['avg_hold_days']:.1f}天, 損益={final_result['OOS']['total_pnl_ntd']:,.0f}")
    b = final_result["bootstrap"]
    print(f"    bootstrap：正報酬比例={b['pct_positive']:.1f}%, p值={b['p_value']:.3f}, "
          f"拿掉最大3筆後損益={b['pnl_excluding_top3_ntd']:,.0f}")
    pd.DataFrame(final_result["oos_trades"]).to_csv(
        os.path.join(RESULTS_DIR, "trades_OOS_simple_combo.csv"), index=False, encoding="utf-8-sig",
    )

    summary_lines = [
        "=" * 100,
        "右側順勢突破策略 --simple-combo模式(固定組合 + 3組累加門檻，跳過完整6階段IS自動搜尋)",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        f"訊號組合：{SIMPLE_COMBO_SIGNAL_WEIGHTS}(固定，不自動搜尋)　"
        f"ATR停損x{atr_stop_mult}/移動停利x{trailing_atr_mult}(沿用--atr-stop-mult/--trailing-atr-mult，"
        f"不跑ATR敏感度網格)",
        "=" * 100,
        "\n存在理由：完整6階段流程(突破窗口→突破風格→訊號自動搜尋→門檻網格→ATR網格→出場配置網格)"
        "連續在同一份IS資料上挑贏家，一次真實執行結果顯示選出的「最佳組合」IS PF=1.41但"
        "OOS PF=0.48、bootstrap正報酬比例只有5.2%——嚴重的IS/OOS表現落差，幾乎可以確定是"
        "疊加6輪IS挑選造成的多重比較(multiple comparisons)問題，不是這個訊號組合本身沒用。"
        "這個模式刻意固定訊號組合、不自動搜尋突破窗口/突破風格/出場配置，只比較「加不加"
        "ADX/大盤氛圍regime這兩道門檻」，把需要驗證的自由度降到最低。",
        "\n--- 門檻變體比較(IS，固定訊號組合，三組累加門檻) ---",
    ]
    header_v = f"{'門檻變體':<40}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}"
    summary_lines.append(header_v)
    summary_lines.append("-" * len(header_v))
    for _, r in variant_df.iterrows():
        summary_lines.append(
            f"{r['variant']:<40}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}"
            f"{r['win_rate']:>8.1f}{r['total_pnl_ntd']:>14,.0f}"
        )

    summary_lines.append(f"\n--- 最終驗證：{final_label}，IS vs OOS + bootstrap ---")
    for split_name in ["IS", "OOS"]:
        stats = final_result[split_name]
        split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
        summary_lines.append(
            f"  {split_full}: {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
            f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
            f"總損益NT${stats['total_pnl_ntd']:,.0f}, "
            f"最大回撤NT${stats['max_drawdown_ntd']:,.0f}, 最大連續虧損{stats['max_consecutive_losses']}筆"
        )
    summary_lines.append(
        f"  [穩健性] OOS bootstrap 1000次重抽樣：平均總損益NT${b['mean']:,.0f}，"
        f"5%~95%區間=[NT${b['p5']:,.0f}, NT${b['p95']:,.0f}]，"
        f"正報酬比例={b['pct_positive']:.1f}%，p值={b['p_value']:.3f}，"
        f"拿掉最大3筆交易後總損益NT${b['pnl_excluding_top3_ntd']:,.0f}"
    )
    summary_lines.append(
        "\n判讀方式：先看門檻變體比較裡，PF是不是隨著累加濾網(無→+ADX→+ADX+regime)單調"
        "變好——如果是，代表這兩道濾網真的有篩掉雜訊；如果中間有一組反而變差，代表那道"
        "濾網在這份資料裡沒有幫助，不該因為邏輯聽起來合理就預設有效。再看最終驗證的OOS PF"
        "跟bootstrap正報酬比例/p值，這才是比IS數字誠實的參考依據——IS PF再漂亮，OOS大幅"
        "滑落都代表這組設定在樣本外站不住腳。"
    )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


def run_combo_search_mode(args, price_data, indicators_by_code, regime_series,
                           is_calendar, oos_calendar, extra_kwargs, execution_kwargs):
    """--combo-search模式：回應使用者「單一訊號混搭、找出最佳搭配，再加上固定ADX趨勢強度
    濾網+大盤氛圍regime濾網+移動停利」的要求，刻意只保留main()完整6階段流程裡「哪些單一
    訊號該混搭成組合」這一個資料驅動的選擇步驟，其餘全部固定：

    1. 突破窗口/突破風格維持引擎預設(20日創新高、硬門檻)，不重跑run_breakout_window_comparison/
       run_breakout_style_comparison——這兩階段是獨立的overfitting風險來源，使用者沒有
       要求重新搜尋。
    2. 只跑一次run_signal_ablation()(基本門檻，在IS內逐一測每個單一訊號，有--with-chip-confirm
       時連籌碼相關訊號也一起測)，印出每個訊號的個別表現。

    3. 這一輪新增：同時並排測試兩個候選訊號組合，而不是只挑一個──使用者想直接比較
       「篩選過的訊號」跟「全部訊號硬湊在一起」誰比較好，不想只看篩選後的結果：
         [候選1：篩選後混搭] 用select_winning_signals()挑出「PF>1且交易筆數夠多」的訊號、
         等權重組成訊號組合——這是原本就有的邏輯，完全不變。如果沒有任何訊號單獨PF>1，
         select_winning_signals()會退回總損益前3名並標記is_reliable=False，這裡照實印出
         ⚠️探索性選擇警告，不隱藏這個事實。
         [候選2：全部混搭不篩選] 把run_signal_ablation()測過的「所有」單一訊號(signal_names_used，
         已經依has_chip決定要不要包含籌碼訊號)不經PF篩選、全部等權重混在一起──這就是
         run_signal_ablation()本來就會算、但原本只在消融表裡出現一行的"__baseline_equal_weight__"
         基準列，這裡把它正式升格成第二個候選，一樣往下跑完整的門檻比較+IS/OOS/bootstrap
         驗證，而不是只停在消融表的那一行。
       兩個候選都「固定」下來後，不再比較「自動篩選 vs 手動指定」(跳過run_signal_combo_comparison，
       這裡已經有候選1/候選2兩個可比的對象，比較本身的角色改由下面的side-by-side表格取代)、
       不跑門檻網格(run_gate_comparison)、不跑ATR敏感度網格(run_atr_sensitivity_grid，直接用
       --atr-stop-mult/--trailing-atr-mult)、不跑出場配置比較(run_exit_style_comparison，移動
       停利維持現行「不限天數、立即啟動」設定)。
    4. 兩個候選各自重用run_simple_combo_comparison()(signal_weights參數換成對應的組合)跑
       SIMPLE_COMBO_VARIANTS三組累加門檻(無→+ADX→+ADX+大盤氛圍regime)比較，各自取門檻疊到
       最滿的那組，重用evaluate_combo()做IS/OOS+bootstrap驗證——跟--simple-combo模式共用
       同一套比較/驗證邏輯，不重新實作，只是現在呼叫兩次、每個候選各一次。

    跟--simple-combo模式的差異：--simple-combo的訊號組合是使用者已經確認過、完全寫死的
    MACD柱狀圖+K棒實體比例，不做任何資料驅動選擇；這裡多做了「挑選/混搭單一訊號組合」這
    一步，所以這個模式的OOS/bootstrap結果比--simple-combo多一點選擇偏誤(selection bias)
    風險，但只做了這一步選擇，比連續6輪都在同一份IS資料上挑贏家的完整流程安全得多。
    summary.txt會明確提醒這個權重判讀方式，而且不管候選1還是候選2哪個OOS數字比較好看，
    都只代表「兩種不同的湊法」互相比較的結果，不代表任何一個候選裡的訊號真的有個別預測力
    ──如果單一訊號拆解表完全沒有任何訊號PF>1，兩個候選本質上都是在拿噪音混搭，只是混搭
    方式不同，這一點summary.txt結尾會誠實點出來，不因為其中一個候選數字比較好看就暗示
    它比較可信。
    """
    print("=" * 100)
    print("--combo-search模式：並排搜尋/比較兩個「單一訊號混搭」候選(篩選後 vs 全部不篩選)，"
          "其餘(突破窗口/風格/門檻網格/ATR網格/出場配置)全部固定不再搜尋")
    print("(這是本模式唯一的一次資料驅動選擇步驟；兩個候選選定後都直接套用ADX趨勢強度濾網+"
          "大盤氛圍regime濾網+移動停利，不再做任何進一步搜尋——比完整6階段流程少5個選擇步驟，"
          "但比--simple-combo多做了這一步訊號挑選，OOS/bootstrap數字的選擇偏誤風險介於兩者之間)")
    print("=" * 100)

    hold_days = TRAILING_STOP_MAX_HOLD_DAYS
    atr_stop_mult = args.atr_stop_mult
    trailing_atr_mult = args.trailing_atr_mult if args.trailing_atr_mult is not None else atr_stop_mult

    print(f"\n[單一訊號拆解] 在IS內用基本門檻逐一測試每個單一訊號(本模式唯一的資料驅動選擇"
          f"{'，含籌碼相關訊號' if args.with_chip_confirm else '，未含籌碼相關訊號(未加--with-chip-confirm)'}) ...")
    ablation_df, signal_names_used = run_signal_ablation(
        price_data, indicators_by_code, regime_series, is_calendar,
        args.starting_capital, hold_days, has_chip=args.with_chip_confirm, extra_kwargs=extra_kwargs,
    )
    ablation_df.to_csv(os.path.join(RESULTS_DIR, "combo_search_ablation.csv"),
                        index=False, encoding="utf-8-sig")

    # ---------- 候選1：篩選後混搭(select_winning_signals()，邏輯跟上一輪完全不變) ----------
    winning_signals, is_reliable = select_winning_signals(ablation_df)
    combo_weights = {name: 1.0 for name in winning_signals}
    combo_label = f"混搭組合({','.join(SIGNAL_LABELS[s] for s in winning_signals)})" \
        f"{'' if is_reliable else '⚠️探索性選擇'}"
    print(f"\n  → 候選1(篩選後混搭)：{combo_label}")
    if not is_reliable:
        print("  ⚠️ 沒有任何單一訊號PF>1且交易筆數足夠，上面是探索性選擇(總損益前3名)，"
              "不是已經被驗證過的訊號，後面的OOS/bootstrap結果可信度要再打更多折扣")

    # ---------- 候選2：全部混搭、不篩選(把ablation裡的__baseline_equal_weight__基準列
    # 升格成第二個正式候選，往下跑完整的門檻比較+IS/OOS/bootstrap驗證) ----------
    all_signals_weights = {name: 1.0 for name in signal_names_used}
    all_signals_label = f"全部混搭({len(signal_names_used)}訊號等權重，不篩選)"
    print(f"  → 候選2(全部混搭不篩選)：{all_signals_label}")

    print(f"\n固定ATR設定：初始停損x{atr_stop_mult}, 移動停利x{trailing_atr_mult}　"
          f"突破窗口/風格：維持引擎預設(20日創新高+硬門檻，不重新搜尋)\n")

    candidates = [
        ("候選1：篩選後混搭", combo_weights, combo_label),
        ("候選2：全部混搭不篩選", all_signals_weights, all_signals_label),
    ]
    final_label, final_gate_kwargs = SIMPLE_COMBO_VARIANTS[-1]
    variant_csv_names = {
        "候選1：篩選後混搭": "combo_search_variants.csv",
        "候選2：全部混搭不篩選": "combo_search_variants_allsignals.csv",
    }
    oos_csv_names = {
        "候選1：篩選後混搭": "trades_OOS_combo_search_filtered.csv",
        "候選2：全部混搭不篩選": "trades_OOS_combo_search_allsignals.csv",
    }
    candidate_results = {}

    for section_key, weights, label in candidates:
        print(f"[門檻變體比較：{section_key}] 固定訊號組合 + 三組累加門檻"
              f"(ADX趨勢強度+大盤氛圍regime，只在IS內) ...")
        variant_df = run_simple_combo_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, atr_stop_mult, trailing_atr_mult,
            extra_kwargs=extra_kwargs, signal_weights=weights,
        )
        variant_df.to_csv(os.path.join(RESULTS_DIR, variant_csv_names[section_key]),
                           index=False, encoding="utf-8-sig")

        print(f"\n[最終驗證：{section_key}] 取{final_label}，跑IS vs OOS + bootstrap穩健性檢查 ...")
        final_result = evaluate_combo(
            f"combo-search{section_key}({label}, {final_label})", price_data, indicators_by_code,
            regime_series, is_calendar, oos_calendar, args.starting_capital, hold_days, weights,
            final_gate_kwargs, atr_stop_mult, trailing_atr_mult, True, extra_kwargs, execution_kwargs,
        )
        print(f"  [{final_result['label']}]")
        print(f"    IS  -> {final_result['IS']['trade_count']}筆, PF={_fmt_pf(final_result['IS']['profit_factor'])}, "
              f"平均持有{final_result['IS']['avg_hold_days']:.1f}天, 損益={final_result['IS']['total_pnl_ntd']:,.0f}")
        print(f"    OOS -> {final_result['OOS']['trade_count']}筆, PF={_fmt_pf(final_result['OOS']['profit_factor'])}, "
              f"平均持有{final_result['OOS']['avg_hold_days']:.1f}天, 損益={final_result['OOS']['total_pnl_ntd']:,.0f}")
        b = final_result["bootstrap"]
        print(f"    bootstrap：正報酬比例={b['pct_positive']:.1f}%, p值={b['p_value']:.3f}, "
              f"拿掉最大3筆後損益={b['pnl_excluding_top3_ntd']:,.0f}")
        pd.DataFrame(final_result["oos_trades"]).to_csv(
            os.path.join(RESULTS_DIR, oos_csv_names[section_key]), index=False, encoding="utf-8-sig",
        )

        candidate_results[section_key] = {
            "label": label, "weights": weights, "variant_df": variant_df, "final_result": final_result,
        }

    summary_lines = [
        "=" * 100,
        "右側順勢突破策略 --combo-search模式(單一訊號混搭搜尋：候選1篩選後 vs 候選2全部不篩選，"
        "並排比較 + 固定ADX/regime/移動停利)",
        f"回測期間：{args.start} ~ {args.end}　起始資金：NT${args.starting_capital:,.0f}",
        f"候選1(篩選後)：{combo_label}　候選2(全部混搭)：{all_signals_label}",
        f"ATR停損x{atr_stop_mult}/移動停利x{trailing_atr_mult}(沿用--atr-stop-mult/--trailing-atr-mult，"
        f"不跑ATR敏感度網格)",
        "=" * 100,
        "\n存在理由：使用者要求「這次把所有技術面+籌碼面的單一訊號都放上去，全部混搭，看看能不能"
        "找到好的搭配」，所以這一輪把run_signal_ablation()本來就會算的「全部訊號等權重」基準，"
        "跟select_winning_signals()篩選過的組合，兩個候選並排跑完整的門檻比較+IS/OOS/bootstrap"
        "驗證，讓使用者直接看「只混搭篩選過的贏家」跟「全部訊號硬湊在一起」誰的OOS表現比較好——"
        "比--simple-combo(完全固定，不做任何搜尋)多讓訊號組合本身是資料驅動的，但比完整6階段"
        "流程(突破窗口→突破風格→訊號自動搜尋→門檻網格→ATR網格→出場配置網格，連續6輪都在同一份"
        "IS資料上挑贏家)少了5個選擇步驟。",
        "\n--- 單一訊號拆解(IS，基本門檻，逐一測試每個訊號) ---",
    ]
    reliable_ablation = ablation_df[ablation_df["trade_count"] >= MIN_TRADES_FOR_RANKING]
    any_signal_pf_above_1 = bool((reliable_ablation["profit_factor"] > 1.0).any()) if not reliable_ablation.empty else False
    if reliable_ablation.empty:
        summary_lines.append(f"⚠️ 沒有任何訊號的交易筆數 >= {MIN_TRADES_FOR_RANKING}，統計上都不夠可靠，全部原樣列出：")
        reliable_ablation = ablation_df
    reliable_ablation = reliable_ablation.sort_values("total_pnl_ntd", ascending=False)
    header_ab = f"{'訊號':<24}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}{'最大連虧':>8}"
    summary_lines.append(header_ab)
    summary_lines.append("-" * len(header_ab))
    for _, r in reliable_ablation.iterrows():
        summary_lines.append(
            f"{r['label']:<24}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}{r['win_rate']:>8.1f}"
            f"{r['total_pnl_ntd']:>14,.0f}{r['max_consecutive_losses']:>8}"
        )
    if not is_reliable:
        summary_lines.append("⚠️ 沒有任何訊號單獨PF>1，候選1選出的組合是探索性選擇(總損益前3名)，不是已驗證過的訊號")
    if not any_signal_pf_above_1:
        summary_lines.append("⚠️ 沒有任何訊號單獨PF>1：候選2(全部混搭)跟候選1一樣，本質上都是拿沒有個別預測力的"
                              "訊號在混搭，差別只在混搭的方式，不代表候選2「找到」了篩選漏掉的好訊號")

    for section_key, _, _ in candidates:
        res = candidate_results[section_key]
        label = res["label"]
        variant_df = res["variant_df"]
        final_result = res["final_result"]
        b = final_result["bootstrap"]

        summary_lines.append(f"\n[{section_key}] 訊號組合：{label}")
        summary_lines.append(f"--- 門檻變體比較(IS，固定訊號組合，三組累加門檻) ---")
        header_v = f"{'門檻變體':<40}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}"
        summary_lines.append(header_v)
        summary_lines.append("-" * len(header_v))
        for _, r in variant_df.iterrows():
            summary_lines.append(
                f"{r['variant']:<40}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}"
                f"{r['win_rate']:>8.1f}{r['total_pnl_ntd']:>14,.0f}"
            )

        summary_lines.append(f"--- 最終驗證：{final_label}，IS vs OOS + bootstrap ---")
        for split_name in ["IS", "OOS"]:
            stats = final_result[split_name]
            split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
            summary_lines.append(
                f"  {split_full}: {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
                f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
                f"總損益NT${stats['total_pnl_ntd']:,.0f}, "
                f"最大回撤NT${stats['max_drawdown_ntd']:,.0f}, 最大連續虧損{stats['max_consecutive_losses']}筆"
            )
        summary_lines.append(
            f"  [穩健性] OOS bootstrap 1000次重抽樣：平均總損益NT${b['mean']:,.0f}，"
            f"5%~95%區間=[NT${b['p5']:,.0f}, NT${b['p95']:,.0f}]，"
            f"正報酬比例={b['pct_positive']:.1f}%，p值={b['p_value']:.3f}，"
            f"拿掉最大3筆交易後總損益NT${b['pnl_excluding_top3_ntd']:,.0f}"
        )

    summary_lines.append("\n--- 候選1 vs 候選2 並排比較(OOS，較誠實的參考依據) ---")
    header_cmp = f"{'候選':<28}{'OOS交易數':>10}{'OOS PF':>10}{'OOS損益NT$':>14}{'bootstrap正報酬%':>16}{'p值':>8}"
    summary_lines.append(header_cmp)
    summary_lines.append("-" * len(header_cmp))
    for section_key, _, _ in candidates:
        res = candidate_results[section_key]
        oos = res["final_result"]["OOS"]
        b = res["final_result"]["bootstrap"]
        summary_lines.append(
            f"{section_key:<28}{oos['trade_count']:>10}{_fmt_pf(oos['profit_factor']):>10}"
            f"{oos['total_pnl_ntd']:>14,.0f}{b['pct_positive']:>16.1f}{b['p_value']:>8.3f}"
        )

    summary_lines.append(
        "\n判讀方式：這個模式做了「單一訊號混搭挑組合」這一步資料驅動選擇(而且一次測了兩種挑法)，"
        "比--simple-combo(完全固定、零選擇)多一點選擇偏誤風險、比完整6階段流程(連續6輪挑贏家)"
        "少很多——看OOS PF/bootstrap正報酬比例/p值時，權重大致介於這兩個模式之間：不能像看"
        "--simple-combo的結果一樣完全沒有選擇偏誤疑慮，但也不該用完整流程那種高度懷疑的眼光"
        "全盤否定。如果候選1標示「探索性選擇」，代表連候選1唯一的這一步篩選都沒有真正驗證過"
        "的訊號可選。\n"
        "更根本的一點：不管上面候選1跟候選2哪一個OOS/bootstrap數字比較好看，都不能證明「混搭」"
        "本身讓策略變好——混搭只是把本來就存在的訊號組合在一起，不會無中生有出預測力。"
        + ("上面單一訊號拆解表已經顯示至少有訊號單獨PF>1，兩個候選的OOS結果值得認真看待。"
           if any_signal_pf_above_1 else
           "上面單一訊號拆解表顯示沒有任何訊號單獨PF>1，代表候選1、候選2本質上都是拿沒有個別預測力的"
           "噪音訊號在混搭，只是混搭方式不同──就算其中一個候選的OOS PF或bootstrap正報酬比例看起來"
           "比較好，也應該當成「這一種湊法剛好比較不糟」，不是「找到了真正有效的訊號組合」，"
           "結果僅供參考，不該直接拿去實盤。")
    )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


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
    parser.add_argument("--walkforward-folds", type=int, default=0,
                         help="走勢前進(Walk-Forward)分折驗證的折數，0代表不啟用(預設)。"
                              f"設>0時，額外把歷史切成N+1個等長區塊，每一折用擴張視窗重新跑一次"
                              f"完整選股流程(訊號拆解→訊號組合→門檻→ATR網格→出場配置)，"
                              f"驗證選出的組合在多折之間是不是穩定，不是只有一次OOS切分。"
                              f"預設建議值{DEFAULT_WALKFORWARD_FOLDS}折(GitHub Actions有~2.5小時的"
                              f"時間預算，這是全新階段、實際耗時還沒有校準過，先保守設小一點)")
    parser.add_argument("--simple-combo", action="store_true",
                         help="跳過整套6階段IS自動搜尋(突破窗口比較→突破風格比較→訊號組合自動搜尋→"
                              "門檻網格→ATR敏感度網格→出場配置網格)，改成只測使用者已經確認過的固定"
                              "訊號組合(MACD柱狀圖+K棒實體比例) + 三個累加門檻變體(無門檻→+趨勢強度ADX→"
                              "+趨勢強度+大盤氛圍regime濾網)，出場一律用移動停利(use_trailing_stop=True)。"
                              "這個模式存在的理由：一次真實GitHub Actions執行結果顯示，完整6階段流程"
                              "連續在同一份IS資料上挑贏家，疊加起來的多重比較(multiple comparisons)問題"
                              "嚴重到讓選出的「最佳組合」IS PF=1.41、OOS PF=0.48、bootstrap正報酬比例"
                              "只有5.2%——用這個固定不挑、只比較三個門檻變體的簡化版本，避開同一種"
                              "overfitting風險。加這個旗標時，--atr-stop-mult/--trailing-atr-mult/"
                              "--start/--end/--starting-capital仍然有效，其餘跟訊號/門檻/突破窗口/"
                              "出場配置自動搜尋相關的旗標會被忽略(因為對應的階段整個不會執行)")
    parser.add_argument("--combo-search", action="store_true",
                         help="跳過突破窗口比較→突破風格比較→訊號組合自動vs手動比較→門檻網格→"
                              "ATR敏感度網格→出場配置網格這幾個階段，只保留「單一訊號拆解」這一個"
                              "資料驅動的選擇步驟(run_signal_ablation，加--with-chip-confirm時連籌碼"
                              "相關訊號也一起測)，並排測試兩個混搭候選：候選1用select_winning_signals()"
                              "挑出「PF>1且交易數夠多」的訊號等權重組成(沒有任何訊號PF>1時退回總損益前3"
                              "名並標記⚠️探索性選擇)；候選2把這次消融測過的全部單一訊號不篩選、直接等"
                              "權重混在一起(就是run_signal_ablation()本來就會算的__baseline_equal_weight__"
                              "基準列，這裡正式升格成第二個候選)。兩個候選各自固定下來後，比照--simple-combo"
                              "模式跑三個累加門檻變體(無→+趨勢強度ADX→+趨勢強度+大盤氛圍regime濾網)"
                              "+移動停利，各自取門檻疊到最滿的那組做IS/OOS+bootstrap驗證，結果輸出到"
                              "combo_search_ablation.csv(共用的單一訊號拆解表)、"
                              "combo_search_variants.csv/trades_OOS_combo_search_filtered.csv(候選1："
                              "篩選後混搭)、combo_search_variants_allsignals.csv/"
                              "trades_OOS_combo_search_allsignals.csv(候選2：全部混搭不篩選)，summary.txt"
                              "用並排表格比較兩者的OOS表現，並誠實提醒：如果單一訊號拆解表沒有任何訊號"
                              "PF>1，兩個候選本質上都是拿噪音混搭，只是混搭方式不同。跟--simple-combo的"
                              "差異：--simple-combo測的是使用者已經確認過的固定訊號組合(MACD柱狀圖+"
                              "K棒實體比例)，完全不做資料驅動選擇；這裡仍然會搜尋/比較「單一訊號該怎麼"
                              "混搭」，只是把突破窗口/突破風格/門檻/ATR倍數/出場配置都固定下來，不跟著"
                              "訊號組合一起被搜尋，藉此把完整6階段流程的多重比較風險降到只剩1階段。加這個"
                              "旗標時，--atr-stop-mult/--trailing-atr-mult/--start/--end/--starting-capital/"
                              "--with-chip-confirm仍然有效(--with-chip-confirm開啟時，候選2的「全部混搭」"
                              "也會把籌碼訊號一起混進去)，其餘跟突破窗口/突破風格/門檻/ATR/出場配置"
                              "自動搜尋相關的旗標會被忽略(因為對應的階段整個不會執行)")
    parser.add_argument("--squeeze-kdj-only", action="store_true",
                         help="只跑布林+Keltner擠壓+KDJ訊號的IS/OOS+bootstrap驗證"
                              "(run_squeeze_kdj_exit_style_comparison_is_oos)，跳過突破窗口比較→"
                              "突破風格比較→訊號自動搜尋→門檻網格→ATR敏感度網格→出場配置網格這整套"
                              "6階段流程——squeeze+KDJ驗證本來就不依賴那6個階段選出的任何東西(它是"
                              "獨立對universe逐檔股票模擬，不經過scan_momentum_breakout_candidates()/"
                              "run_momentum_breakout_backtest()那條路徑，訊號本身的進場/出場規則寫死"
                              "在squeeze_kdj_signal.py裡，不吃訊號權重/門檻這些參數)，開這個旗標可以"
                              "不用等前面不相關的流程跑完，直接看這個訊號的IS/OOS/bootstrap結果。"
                              "加這個旗標時，--atr-stop-mult/--trailing-atr-mult/--start/--end/"
                              "--starting-capital/--max-stocks仍然有效，其餘跟訊號/門檻/突破窗口/"
                              "出場配置自動搜尋相關的旗標在這裡不適用(squeeze+KDJ的規則是寫死的，"
                              "不是被忽略，是這些旗標控制的維度跟這個模式測的東西無關)")
    parser.add_argument("--squeeze-kdj-capital-constrained", action="store_true",
                         help="資金/部位受限版squeeze+KDJ驗證(--squeeze-kdj-capital-constrained)："
                              "把--squeeze-kdj-only已經驗證過的進場訊號，接上這個專案『真正會拿去"
                              "模擬實戰』的資金/部位管理框架(top_n排名+max_concurrent_positions"
                              "持倉上限+保證金查表，見squeeze_kdj_signal.run_squeeze_kdj_"
                              "capital_constrained_backtest())，跑一次day-by-day walk-forward"
                              "+IS/OOS/bootstrap驗證，輸出變體A/B各自的資金受限版PF，並明確對照"
                              "--squeeze-kdj-only算出的資金無限版OOS PF(變體A=5.01/變體B=3.44)，"
                              "誠實講清楚『拿掉每個訊號都能無限制同時成交這個樂觀假設之後，實際"
                              "帳戶能拿到的PF撐住、打折、還是崩潰』——這是--squeeze-kdj-only刻意"
                              "沒有回答的問題(--squeeze-kdj-only的訊號驗證完全不經過資金/部位管理，"
                              "見該模式docstring的誠實caveat)。跟--squeeze-kdj-only一樣跳過突破窗口"
                              "比較→突破風格比較→訊號自動搜尋→門檻網格→ATR敏感度網格→出場配置網格"
                              "這整套6階段流程(squeeze+KDJ的訊號規則本身跟那6個階段選出的東西無關)。"
                              "加這個旗標時，--start/--end/--starting-capital/--max-stocks仍然有效，"
                              "其餘跟訊號/門檻/突破窗口/出場配置自動搜尋相關的旗標在這裡不適用")
    parser.add_argument("--squeeze-kdj-grid", action="store_true",
                         help="squeeze+KDJ資金受限版混搭網格搜尋(--squeeze-kdj-grid)：把出場方式(變體A/"
                              "變體B固定ATR停損停利6組/變體B移動停利3組)x最大持倉數(1/3/5)x部位大小(固定1口/"
                              "固定2口/風險1%%/風險2%%)x最長持有天數(10/20/60)x同日多檔排名規則(觸發K棒漲幅/"
                              "量比)x進場濾網(無/站上60日均線)共1440組，全部只在樣本內(IS)跑、用IS PF"
                              "(IS筆數>=30才有資格)選出唯一一個贏家，贏家才拿去樣本外(OOS)跑一次+bootstrap"
                              "當誠實的頭條數字；另外報告IS前10名的OOS PF、IS PF vs OOS PF的Spearman排名"
                              "相關、全部組合OOS PF的分布(只看分布、不挑OOS最好的組合)，並醒目標示總共測了"
                              "幾組跟多重比較警告。輸出squeeze_kdj_grid_all_combos.csv(中文欄名)+summary.txt。"
                              "跟--squeeze-kdj-only一樣跳過完整6階段流程；--start/--end/--starting-capital/"
                              "--max-stocks仍然有效(起始資金是真實帳戶規模，不是網格維度)")
    parser.add_argument("--fixed-combo-walkforward-folds", type=int, default=0,
                         help="測試幾組「固定死不重新挑選」的候選規則(FIXED_WALKFORWARD_COMBO_VARIANTS，"
                              "基準+只改一個維度的變體)跨N個獨立、不重疊歷史區塊的表現，0代表不啟用(預設)。"
                              "跟--walkforward-folds不同的是這裡不是每折重新選一次組合，而是同一組規則"
                              "原封不動套到每一個區塊，所以沒有--walkforward-folds那種每折都在挑當下"
                              f"最好答案的多重比較風險，可以放心切更多折(成本低很多，不用重跑整套"
                              f"訊號拆解→訊號組合→門檻→ATR網格→出場配置的選擇流程，每折只是單純"
                              f"backtest一次)。預設建議值{DEFAULT_FIXED_COMBO_WALKFORWARD_FOLDS}折")
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

    if args.simple_combo:
        run_simple_combo_mode(args, price_data, indicators_by_code, regime_series,
                               is_calendar, oos_calendar, extra_kwargs, execution_kwargs)
        return

    if args.combo_search:
        run_combo_search_mode(args, price_data, indicators_by_code, regime_series,
                               is_calendar, oos_calendar, extra_kwargs, execution_kwargs)
        return

    if args.squeeze_kdj_only:
        run_squeeze_kdj_only_mode(args, price_data, universe, is_calendar, oos_calendar)
        return

    if args.squeeze_kdj_capital_constrained:
        run_squeeze_kdj_capital_constrained_mode(args, price_data, universe, is_calendar, oos_calendar)
        return

    if args.squeeze_kdj_grid:
        run_squeeze_kdj_grid_mode(args, price_data, universe, is_calendar, oos_calendar)
        return

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
    all_signal_combo = {}
    all_gate_comparison = {}
    all_atr_grid = {}
    all_exit_style = {}
    all_breakout_style = {}
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

        print("[階段0.7] 突破風格比較 (硬門檻 vs 軟性排名，只在IS內，等權重訊號+基本門檻) ...")
        style_df = run_breakout_style_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, signal_names_for_window, extra_kwargs=window_extra_kwargs,
        )
        all_breakout_style[hold_label] = style_df
        style_df.to_csv(os.path.join(RESULTS_DIR, f"breakout_style_{hold_label}.csv"),
                         index=False, encoding="utf-8-sig")
        style_label, winning_require_hard_breakout = select_winning_breakout_style(style_df)
        print(f"  → 選出突破風格：{style_label}\n")
        window_extra_kwargs = dict(window_extra_kwargs, require_hard_breakout=winning_require_hard_breakout)

        print("[階段1] 單一訊號拆解 (只在IS內，基本門檻) ...")
        ablation_df, signal_names_used = run_signal_ablation(
            price_data, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, has_chip=args.with_chip_confirm, extra_kwargs=window_extra_kwargs,
        )
        all_ablation[hold_label] = ablation_df
        ablation_df.to_csv(os.path.join(RESULTS_DIR, f"ablation_{hold_label}.csv"),
                            index=False, encoding="utf-8-sig")

        winning_signals, is_reliable = select_winning_signals(ablation_df)
        signal_reliable_flags[hold_label] = is_reliable
        auto_signal_weights = {name: 1.0 for name in winning_signals}
        auto_signal_label = f"自動篩選({','.join(SIGNAL_LABELS[s] for s in winning_signals)})" \
            f"{'' if is_reliable else '⚠️探索性選擇'}"

        print(f"\n[階段1.5] 訊號組合比較(自動篩選 vs 手動指定：MACD轉正+RSI>50+量比放大，只在IS內) ...")
        combo_df = run_signal_combo_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, auto_signal_label, auto_signal_weights,
            extra_kwargs=window_extra_kwargs,
        )
        all_signal_combo[hold_label] = combo_df
        combo_df.to_csv(os.path.join(RESULTS_DIR, f"signal_combo_{hold_label}.csv"),
                         index=False, encoding="utf-8-sig")
        combo_label, winning_signal_weights = select_winning_signal_combo(
            combo_df, auto_signal_label, auto_signal_weights,
        )
        print(f"  → 選出訊號組合：{combo_label}\n")

        print(f"[階段2] 結構門檻變體比較 (只在IS內，用上面選出的訊號組合) ...")
        gate_df = run_gate_comparison(
            price_data, indicators_by_code, regime_series, is_calendar,
            args.starting_capital, hold_days, list(winning_signal_weights.keys()), has_chip=args.with_chip_confirm,
            extra_kwargs=window_extra_kwargs,
        )
        all_gate_comparison[hold_label] = gate_df
        gate_df.to_csv(os.path.join(RESULTS_DIR, f"gate_comparison_{hold_label}.csv"),
                        index=False, encoding="utf-8-sig")

        winning_gate_label, winning_gate_kwargs = select_winning_gate(gate_df)
        print(f"\n  → 依IS結果選出的最佳組合：訊號組合={combo_label}"
              f"，門檻={winning_gate_label}")

        atr_stop_mult, trailing_atr_mult = args.atr_stop_mult, args.trailing_atr_mult
        winning_hold_days = hold_days
        winning_exit_extra = {}
        if args.use_trailing_stop:
            print(f"\n[階段2.5] ATR倍數敏感度網格(只在IS內，用上面選出的最佳訊號/門檻組合) ...")
            atr_grid_df, (atr_stop_mult, trailing_atr_mult) = run_atr_sensitivity_grid(
                price_data, indicators_by_code, regime_series, is_calendar,
                args.starting_capital, winning_signal_weights, winning_gate_kwargs, window_extra_kwargs,
                execution_kwargs=execution_kwargs,
            )
            all_atr_grid[hold_label] = atr_grid_df
            print(f"  → 選出：初始停損x{atr_stop_mult}, 移動停利x{trailing_atr_mult}")

            print(f"\n[階段2.6] 出場配置比較(只吃魚身：持有天數上限/延遲啟動/跳空上限，只在IS內) ...")
            exit_style_df = run_exit_style_comparison(
                price_data, indicators_by_code, regime_series, is_calendar,
                args.starting_capital, winning_signal_weights, winning_gate_kwargs,
                atr_stop_mult, trailing_atr_mult, window_extra_kwargs,
                execution_kwargs=execution_kwargs,
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
            f"最佳組合(訊號組合={combo_label}, 門檻={winning_gate_label}, 突破窗口={window_label}, "
            f"突破風格={style_label}, 出場={exit_style_label if args.use_trailing_stop else '固定天數'})",
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

    print(f"\n[新增階段] 布林+Keltner擠壓+KDJ訊號：變體A(原規則) vs 變體B(沿用ATR框架) ...")
    print("  ⚠️ 這是這個檔案最後一個補上IS/OOS+bootstrap驗證的比較，之前只跑過全樣本、"
          "沒有分IS/OOS，之前一次好看的全樣本數字(PF=1.84、損益+NT$518萬)完全沒驗證過"
          "是不是過擬合——先驗證再談優化，見run_squeeze_kdj_exit_style_comparison_is_oos() "
          "docstring")
    squeeze_kdj_results = run_squeeze_kdj_exit_style_comparison_is_oos(
        price_data, universe, args.starting_capital, is_calendar, oos_calendar,
    )
    squeeze_kdj_exit_df = _squeeze_kdj_is_oos_results_to_df(squeeze_kdj_results)
    squeeze_kdj_exit_df.to_csv(os.path.join(RESULTS_DIR, "squeeze_kdj_is_oos.csv"),
                                index=False, encoding="utf-8-sig")
    for r in squeeze_kdj_results.values():
        print(f"  [{r['label']}]")
        print(f"    IS  -> {r['IS']['trade_count']}筆, PF={_fmt_pf(r['IS']['profit_factor'])}, "
              f"平均持有{r['IS']['avg_hold_days']:.1f}天, 損益={r['IS']['total_pnl_ntd']:,.0f}")
        print(f"    OOS -> {r['OOS']['trade_count']}筆, PF={_fmt_pf(r['OOS']['profit_factor'])}, "
              f"平均持有{r['OOS']['avg_hold_days']:.1f}天, 損益={r['OOS']['total_pnl_ntd']:,.0f}")
        b = r["bootstrap"]
        print(f"    bootstrap：正報酬比例={b['pct_positive']:.1f}%, p值={b['p_value']:.3f}, "
              f"拿掉最大3筆後損益={b['pnl_excluding_top3_ntd']:,.0f}")
    print("  ⚠️ 以上全程不經過資金/部位管理(每個訊號都視為獨立成交，沒有top_n排名、沒有"
          "同時持倉上限)，PF/損益數字比真實帳戶能拿到的樂觀——這組驗證回答的是「進場/出場"
          "邏輯方向上是否穩健」，不是「我的帳戶實際能拿到的PF」")

    walkforward_df = None
    if args.walkforward_folds > 0:
        print(f"\n[走勢前進(Walk-Forward)分折驗證] 切成{args.walkforward_folds + 1}個等長區塊，"
              f"共{args.walkforward_folds}折，每折用擴張視窗重新跑一次完整選股流程 ...")
        print("⚠️ 這個階段固定用移動停利模式跑(不管有沒有加--use-trailing-stop)、"
              "且沿用上面已經選定的突破窗口/突破風格，不會每折重新選一次，見"
              "run_walkforward_validation()docstring的簡化說明\n")
        walkforward_df = run_walkforward_validation(
            price_data, indicators_by_code, regime_series, master_calendar,
            args.starting_capital, window_extra_kwargs, execution_kwargs,
            args.walkforward_folds, args.with_chip_confirm,
        )
        if not walkforward_df.empty:
            walkforward_df.to_csv(os.path.join(RESULTS_DIR, "walkforward_folds.csv"),
                                   index=False, encoding="utf-8-sig")

    fixed_combo_walkforward_results = []  # [(combo_dict, df), ...]，每個候選規則各一份walk-forward結果
    if args.fixed_combo_walkforward_folds > 0:
        print(f"\n[固定規則walk-forward比較] 把{len(FIXED_WALKFORWARD_COMBO_VARIANTS)}組候選規則"
              f"(基準+只改一個維度的變體)分別原封不動套到{args.fixed_combo_walkforward_folds}個"
              f"獨立、不重疊的歷史區塊各跑一次，比較哪一組規則的PF範圍最窄(穩定)、平均PF最高 ...")
        print("⚠️ 這裡不重新挑選任何訊號/門檻/ATR倍數/出場配置，套的extra_kwargs是除權息清洗/"
              "滑價這些執行細節，不是外層選出的突破窗口/突破風格(每組候選規則自己指定了"
              "breakout_window/require_hard_breakout)，見run_fixed_combo_walkforward()docstring\n")
        for combo in FIXED_WALKFORWARD_COMBO_VARIANTS:
            print(f"  ...跑 {combo['label']}")
            df = run_fixed_combo_walkforward(
                price_data, indicators_by_code, regime_series, master_calendar,
                args.starting_capital, extra_kwargs, execution_kwargs,
                args.fixed_combo_walkforward_folds, fixed_combo=combo,
            )
            fixed_combo_walkforward_results.append((combo, df))
            if not df.empty:
                safe_label = combo["label"].split("(")[0]
                df.to_csv(os.path.join(RESULTS_DIR, f"fixed_combo_walkforward_{safe_label}.csv"),
                          index=False, encoding="utf-8-sig")

    # 用baseline(FIXED_WALKFORWARD_COMBO本身)的結果維持跟舊版summary.txt區塊的相容性
    fixed_combo_walkforward_df = fixed_combo_walkforward_results[0][1] if fixed_combo_walkforward_results else None

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

        summary_lines.append(f"\n--- {hold_label} / 突破風格比較(IS，硬門檻 vs 軟性排名) ---")
        style_df = all_breakout_style[hold_label].sort_values("total_pnl_ntd", ascending=False)
        header0b = f"{'突破風格':<40}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}"
        summary_lines.append(header0b)
        summary_lines.append("-" * len(header0b))
        for _, r in style_df.iterrows():
            summary_lines.append(
                f"{r['style_label']:<40}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}"
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

        summary_lines.append(f"\n--- {hold_label} / 訊號組合比較(IS，自動篩選 vs 手動指定MACD+RSI+量比) ---")
        combo_df = all_signal_combo[hold_label].sort_values("total_pnl_ntd", ascending=False)
        header1b = f"{'訊號組合':<50}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}"
        summary_lines.append(header1b)
        summary_lines.append("-" * len(header1b))
        for _, r in combo_df.iterrows():
            summary_lines.append(
                f"{r['combo_label']:<50}{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}"
                f"{r['win_rate']:>8.1f}{r['total_pnl_ntd']:>14,.0f}"
            )

        summary_lines.append(f"\n--- {hold_label} / 結構門檻變體比較(IS，用選定的訊號組合) ---")
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

    if walkforward_df is not None:
        summary_lines.append(f"\n--- 走勢前進(Walk-Forward)分折驗證(共{args.walkforward_folds}折，"
                              f"每折用擴張視窗重新選一次訊號/門檻/出場配置，只在各折自己的測試窗口驗證) ---")
        if walkforward_df.empty:
            summary_lines.append("  ⚠️ 所有折都因為訓練窗口太短被跳過，沒有任何可用結果")
        else:
            header_wf = (f"{'折':>4}{'訓練期間':<24}{'測試期間':<24}{'訊號組合':<18}{'門檻':<14}"
                          f"{'出場配置':<16}{'交易數':>8}{'PF':>8}{'勝率%':>8}{'總損益NT$':>14}{'平均持有天':>10}")
            summary_lines.append(header_wf)
            summary_lines.append("-" * len(header_wf))
            for _, r in walkforward_df.iterrows():
                summary_lines.append(
                    f"{int(r['fold']):>4}"
                    f"{r['train_start'] + '~' + r['train_end']:<24}"
                    f"{r['test_start'] + '~' + r['test_end']:<24}"
                    f"{r['signal_combo'][:16]:<18}{r['gate'][:12]:<14}{r['exit_style'][:14]:<16}"
                    f"{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}{r['win_rate']:>8.1f}"
                    f"{r['total_pnl_ntd']:>14,.0f}{r['avg_hold_days']:>10.1f}"
                )
            n_distinct_combos = walkforward_df["signal_combo"].nunique()
            finite_pf = [pf for pf in walkforward_df["profit_factor"] if pf != float("inf")]
            has_inf_pf = any(pf == float("inf") for pf in walkforward_df["profit_factor"])
            if finite_pf:
                pf_range_str = (f"PF範圍(排除∞)：min={min(finite_pf):.2f}, max={max(finite_pf):.2f}, "
                                 f"mean={sum(finite_pf) / len(finite_pf):.2f}"
                                 f"{'(另有折PF=∞，通常代表交易筆數太少，不是真的沒有風險)' if has_inf_pf else ''}")
            else:
                pf_range_str = "PF範圍：全部折都是∞或無交易，無法計算平均"
            summary_lines.append(
                f"\n  [穩定性] {len(walkforward_df)}折中出現{n_distinct_combos}種不同的訊號組合被選中"
                f"({'同一組合每折都被選中，穩定性較高' if n_distinct_combos == 1 else '每折選到不同組合，穩定性較低，較可能是在追這段歷史的雜訊'})"
                f"　{pf_range_str}"
            )

    if fixed_combo_walkforward_results:
        summary_lines.append(f"\n--- 固定規則walk-forward比較(共{len(fixed_combo_walkforward_results)}組候選規則，"
                              f"各自套到{args.fixed_combo_walkforward_folds}個獨立、不重疊區塊，不重新挑選) ---")
        summary_lines.append("  候選規則：基準是連續兩輪被自動選中的原始組合，其餘每個變體都只改一個維度"
                              "(訊號/門檻/ATR倍數/出場配置擇一)，方便歸因是哪個改動造成差異")

        compare_rows = []
        for combo, df in fixed_combo_walkforward_results:
            if df.empty:
                compare_rows.append((combo["label"], 0, None, None, None, 0))
                continue
            finite_pf = [pf for pf in df["profit_factor"] if pf != float("inf")]
            n_positive = sum(1 for pf in df["profit_factor"] if pf > 1)
            mean_pf = sum(finite_pf) / len(finite_pf) if finite_pf else None
            pf_range = (min(finite_pf), max(finite_pf)) if finite_pf else None
            compare_rows.append((combo["label"], len(df), pf_range, mean_pf, n_positive, int(df["trade_count"].sum())))

        header_cmp = f"{'候選規則':<58}{'可用折數':>8}{'PF範圍':>16}{'平均PF':>8}{'PF>1折數':>10}{'總交易數':>8}"
        summary_lines.append(header_cmp)
        summary_lines.append("-" * len(header_cmp))
        for label, n_used, pf_range, mean_pf, n_positive, total_trades in compare_rows:
            pf_range_str = f"{pf_range[0]:.2f}~{pf_range[1]:.2f}" if pf_range else "—"
            mean_pf_str = f"{mean_pf:.2f}" if mean_pf is not None else "—"
            summary_lines.append(
                f"{label:<58}{n_used:>8}{pf_range_str:>16}{mean_pf_str:>8}{f'{n_positive}/{n_used}':>10}"
                f"{total_trades:>8}"
            )

        # 挑一個「值得繼續看」的候選：至少要有2個可用折才敢談穩不穩定(1折沒有範圍可言)，
        # 在這些裡面選平均PF最高的；沒有任何候選滿足這個門檻時老實標成「無法判斷」，
        # 不會為了選出一個而降低門檻。
        eligible = [(label, mean_pf) for label, n_used, _, mean_pf, _, _ in compare_rows
                    if n_used >= 2 and mean_pf is not None]
        if eligible:
            best_label, best_mean_pf = max(eligible, key=lambda x: x[1])
            summary_lines.append(f"\n  平均PF最高且至少有2折可比較的候選：{best_label}(平均PF={best_mean_pf:.2f})"
                                  f"——這只是這批候選裡相對最好的，不代表平均PF就已經驗證超過1、"
                                  f"更不代表可以直接拿去實盤，仍要看PF範圍是否夠窄、總交易數是否夠多")
        else:
            summary_lines.append("\n  ⚠️ 沒有任何候選規則有至少2個可用折可以比較穩定性，這批候選在目前"
                                  "區塊切法下樣本都太少，結果僅供參考")

        # 詳細列出基準規則(FIXED_WALKFORWARD_COMBO)逐折結果，維持跟上一輪summary.txt一致的細節層級；
        # 其他變體的逐折明細各自存成獨立csv(fixed_combo_walkforward_<label>.csv)，不全部印在這裡避免過長
        baseline_df = fixed_combo_walkforward_results[0][1]
        summary_lines.append(f"\n  基準規則逐折明細：{FIXED_WALKFORWARD_COMBO['label']}")
        if baseline_df.empty:
            summary_lines.append("  ⚠️ 所有區塊都因為交易筆數太少被跳過，沒有任何可用結果")
        else:
            header_fcwf = (f"{'區塊':>4}{'期間':<24}{'交易數':>8}{'PF':>8}{'勝率%':>8}"
                            f"{'總損益NT$':>14}{'平均持有天':>10}")
            summary_lines.append(header_fcwf)
            summary_lines.append("-" * len(header_fcwf))
            for _, r in baseline_df.iterrows():
                summary_lines.append(
                    f"{int(r['fold']):>4}"
                    f"{r['period_start'] + '~' + r['period_end']:<24}"
                    f"{r['trade_count']:>8}{_fmt_pf(r['profit_factor']):>8}{r['win_rate']:>8.1f}"
                    f"{r['total_pnl_ntd']:>14,.0f}{r['avg_hold_days']:>10.1f}"
                )

    summary_lines.append(f"\n--- 布林+Keltner擠壓+KDJ訊號：變體A(原規則) vs 變體B(沿用ATR框架) ---")
    summary_lines.append("  (這個比較跟上面各階段獨立，只做多方，見squeeze_kdj_signal.py模組docstring；"
                          "本輪新增IS/OOS切分+OOS bootstrap穩健性檢查，取代之前只跑全樣本、"
                          "沒驗證過是不是過擬合的版本)")
    for r in squeeze_kdj_results.values():
        summary_lines.append(f"\n  [{r['label']}]")
        for split_name in ["IS", "OOS"]:
            stats = r[split_name]
            split_full = "樣本內(IS)" if split_name == "IS" else "樣本外(OOS) ← 較誠實的參考依據"
            summary_lines.append(
                f"    {split_full}: {stats['trade_count']}筆, PF={_fmt_pf(stats['profit_factor'])}, "
                f"勝率={stats['win_rate']:.1f}%, 平均持有{stats['avg_hold_days']:.1f}天, "
                f"總損益NT${stats['total_pnl_ntd']:,.0f}, "
                f"最大回撤NT${stats['max_drawdown_ntd']:,.0f}, 最大連續虧損{stats['max_consecutive_losses']}筆"
            )
        b = r["bootstrap"]
        summary_lines.append(
            f"    [穩健性] OOS bootstrap 1000次重抽樣：平均總損益NT${b['mean']:,.0f}，"
            f"5%~95%區間=[NT${b['p5']:,.0f}, NT${b['p95']:,.0f}]，"
            f"正報酬比例={b['pct_positive']:.1f}%，p值={b['p_value']:.3f}，"
            f"拿掉最大3筆交易後總損益NT${b['pnl_excluding_top3_ntd']:,.0f}"
        )
    summary_lines.append(
        "  ⚠️ 以上全程不經過資金/部位管理(每個訊號都視為獨立成交，沒有top_n排名、沒有同時"
        "持倉上限)，PF/損益數字比真實帳戶能拿到的樂觀——這組驗證回答的是「進場/出場邏輯"
        "方向上是否穩健」，不是「我的帳戶實際能拿到的PF」"
    )

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
        "\n\n新增的「突破風格比較」是這一輪「不追求「準」，追求「賠得少、賺得多、選得穩」」"
        "這個核心哲學的直接實作：測試把「收盤價一定要站上突破窗口高/低點」這個絕對值硬門檻，"
        "換成只留最基本的均線結構資格、突破強度改用相對排名(score_breakout_strength)去決定"
        "候選順序，是不是比絕對值判定更穩定——因為相對排名不管標的池大小或期間怎麼變，"
        "永遠是拿候選互相比較，不是拿候選跟一個固定的magic number比較。這裡選出的風格"
        "會固定套用到後面所有階段。"
        "\n\n新增的「訊號組合比較」是回應使用者盤感提出的假設：MACD柱狀圖轉正+RSI穿越50+"
        "量比放大，代表「主力進場、散戶也跟著追、短進短出、最後誰接盤」的組合效應，這跟"
        "「單一訊號拆解」的自動篩選機制邏輯不同——自動篩選只驗證單一訊號的個別預測力"
        "(PF>1才選)，量比/RSI單獨測都不到1，永遠不會被自動選中，但組合在一起可能有單一"
        "訊號測不出來的效應，所以另外測試這組固定組合，跟自動篩選的結果放在一起比較，"
        "不預設誰比較好。門檻變體裡新增的「MA5基礎+均線糾結或多頭同步向上」則是把基本方向"
        "確認從MA20改成MA5(短波段反應更即時)，額外要求「均線糾結(像要爆發)」或「多頭並列"
        "且三線同步向上」兩者擇一——不是排列剛好對就算，而是三線真的都在漲，或三線貼在一起"
        "代表盤整蓄積。出場配置新增的「3天出場+保本停損」則是回應「只要有漲，停利一定是"
        "成本價，不要漲了後面下跌還賠錢」這個原則：一旦浮動獲利轉正，停損就移到成本價，"
        "不管有沒有開移動停利都會生效。這幾個都是使用者盤感提出、還沒被驗證過的新假設，"
        "程式會照實測出來的PF/總損益排名，不會因為背後的邏輯聽起來合理就預設會贏。"
        "\n\n新增的「布林+Keltner擠壓+KDJ」訊號是把一支加密貨幣1小時K線教學影片描述的"
        "TTM squeeze概念翻譯成台股日線：布林通道縮進Keltner通道內代表盤整蓄積(擠壓)，"
        "擠壓附近跌破布林下軌且KDJ的K<20視為「武裝」，之後收紅且K回升穿越20視為關鍵K棒，"
        "訊號觸發。這個訊號本身已經以「變體B(沿用ATR框架)」的形式併入上面的單一訊號拆解，"
        "跟其他既有訊號放在同一套消融測試裡公平比較；額外新增的這張表則是把這個訊號單獨"
        "抽出來，比較它自己的兩種出場方式哪個更好——「變體A(原規則)」用前一根K棒的低點"
        "當固定停損、K衝高到80以上後跌破80才停利，是原始描述裡完整的規則；「變體B」則是"
        "拿掉原本的停損/停利邏輯，改用這支引擎既有的ATR停損/停利框架，方便跟其他訊號的"
        "出場方式一致比較。由於這是把1小時圖的概念套到日線，擠壓回看窗口(10天)、武裝"
        "過期天數(15天)這些參數都是合理但未經實測調整的預設值，不是先驗保證有效，這張"
        "表本身的交易筆數如果偏少，代表這個設定在目前的股票池/期間裡本來就不常觸發，"
        "PF數字的可信度要打折扣看待。"
    )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


if __name__ == "__main__":
    main()
