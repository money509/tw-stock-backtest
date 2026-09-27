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
    "score_breakout_strength": "突破強度(ATR倍數相對排名)",
    "score_foreign_streak": "外資連續買超天數",
    "score_trust_streak": "投信連續買超天數",
    "score_institutional_net_1d": "主力買賣超(1日)",
    "score_institutional_net_5d": "主力買賣超(5日)",
    "score_institutional_net_10d": "主力買賣超(10日)",
    "score_macd_divergence": "MACD背離(近似)",
    "score_volume_spike": "成交量急增(5日窗口)",
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
    )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n已輸出：{summary_path}")


if __name__ == "__main__":
    main()
