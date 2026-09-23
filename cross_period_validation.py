"""
cross_period_validation.py
==============================
真正的跨期驗證：不是同一份資料切出來的IS/OOS，而是完全獨立的一段歷史區間——
這段資料從頭到尾都沒有被用來挑過任何訊號權重或ATR參數。

背景：CANDIDATES這份候選清單，是經過一次完整的「誠實版」重新挑選才得到的
（先前用「T日當天」偷看版本選出的候選，拿到這裡驗證時PF全部掉到1以下，詳見
git歷史；以下是修正後的完整鏈路）：
  1. sweep_signal_ablation.py(honest timing版)：8訊號單獨測，只有量比(PF1.12)、
     相對大盤強弱(PF1.11)、外資買超比重(PF1.07)明顯>1，投信只是打平(PF1.01)。
  2. sweep_atr_params.py(honest timing + slippage_pct=0.1版)：確認停損<=0.1倍
     雖然PF還撐得住，但距離小到執行上不可信；0.1~1.5倍的合理範圍裡，固定用
     舊8訊號等權重時全部PF<1，代表光調ATR救不回訊號本身太弱的問題。
  3. sweep_combined_best.py(honest timing + slippage_pct=0.1版)：把①驗證過的
     訊號分別組合、疊上②篩過的合理ATR區間交叉測試，發現「只用量比+相對大盤
     強弱這2個、拿掉外資」表現最好且最穩——6組ATR全部PF>=1.00，其中停損0.3倍
     /停利2.0倍最佳(IS PF=1.12)。外資單獨測雖然PF1.07不差，但混進組合裡反而
     拖累整體，不採用。

現在CANDIDATES換成這組「最強2個(量比+相對大盤強弱)」搭配幾組合理ATR，加上
基準8訊號組跟舊版正式鎖定版本當對照。但這整個挑選過程（①②③）都是在同一份
資料(2023-09-15~2026-09-14)上做的多層挑選，就算③的OOS(同一份資料切出來的
後30%)也有PF=1.17，仍不是真正獨立的驗證——這支腳本就是要補上這一塊。

做法：把這些候選參數鎖定下來(不再調整)，直接在一段全新的、完全沒用過的歷史
區間上跑一次完整回測(不切IS/OOS，因為這整段資料本身就是「樣本外」)，看PF
是不是還站得住腳。如果在完全沒看過的資料上PF還是>1，才算是真正可信的訊號；
如果掉到1以下甚至更差，代表前面的結果是被這3年的市場環境或多層選擇撐起來的。

用法：
    python3 cross_period_validation.py --start 2021-09-15 --end 2023-09-14

⚠️ start/end 務必填一段跟原本(2023-09-15~2026-09-14)完全不重疊的期間，
不然就失去「獨立驗證」的意義了。
"""

import argparse
import datetime

import pandas as pd

import data_loader
import overnight_momentum_engine as ome
import taifex_universe
from compare_overnight import build_pipeline_inputs, UNIVERSE_CHOICES

# 鎖定候選參數：來自「誠實版」訊號拆解 + ATR掃描 + 合併測試(均為honest timing +
# slippage_pct=0.1)選出來的最佳組合，這裡不再調整，只是原封不動拿去一段全新的
# 資料上驗證。
CANDIDATES = [
    {
        "label": "最強2個(量比+相對大盤強弱)，停損0.3倍/停利2.0倍"
                  "——combined測試裡的第一名(IS honest+slippage版 PF=1.12)",
        "signal_weights": {"score_volume_ratio": 1.0, "score_rel_strength": 1.0},
        "atr_stop_mult": 0.3,
        "atr_target_mult": 2.0,
    },
    {
        "label": "最強2個(量比+相對大盤強弱)，停損0.5倍/停利2.0倍"
                  "——停損距離比第一名寬鬆，風控上更保守(IS PF=1.09)",
        "signal_weights": {"score_volume_ratio": 1.0, "score_rel_strength": 1.0},
        "atr_stop_mult": 0.5,
        "atr_target_mult": 2.0,
    },
    {
        "label": "最強2個(量比+相對大盤強弱)，停損0.8倍/停利2.0倍"
                  "——停損維持在舊預設倍數，只換訊號組合(IS PF=1.07)",
        "signal_weights": {"score_volume_ratio": 1.0, "score_rel_strength": 1.0},
        "atr_stop_mult": 0.8,
        "atr_target_mult": 2.0,
    },
    {
        "label": "對照組：基準8訊號等權重，舊預設ATR(0.8/1.2)",
        "signal_weights": {
            "score_close_position": 1.0, "score_volume_ratio": 1.0, "score_rel_strength": 1.0,
            "score_gain_pct": 1.0, "score_price_level": 1.0, "score_day_trading": 1.0,
            "score_foreign": 1.0, "score_trust": 1.0,
        },
        "atr_stop_mult": 0.8,
        "atr_target_mult": 1.2,
    },
    {
        "label": "對照組：舊版正式鎖定版本(tech/chip預設權重0.35/0.65 + 停損0.8倍/"
                  "停利3.0倍)——已知在honest timing下PF<1(見#42跨期驗證)，"
                  "保留純粹當歷史對照，不是候選",
        # 不寫signal_weights鍵，代表沿用run_overnight_backtest的
        # tech_weight/chip_weight邏輯(引擎預設0.35/0.65)，不是8訊號加權平均。
        "atr_stop_mult": 0.8,
        "atr_target_mult": 3.0,
    },
]


def run_cross_period_validation(pipeline_inputs, candidates=None, top_n=5,
                                 compare_chip_timing=True, slippage_pct=0.0,
                                 min_candidates=None, min_trust_ratio=None):
    """
    對每個鎖定的候選組合，在整段(未切IS/OOS)資料上跑一次回測。

    compare_chip_timing：預設True，每個候選組合都會跑三個版本，一次看清楚
    兩個「偷看未來」的問題(三大法人資料、美股濾網)分別修正之後的效果：
      1. 「完全T日當天(舊版，不可能實測)」——三大法人跟美股濾網都查T日自己
         當天的資料，這在實務上不可能做到(見overnight_momentum_engine.py的
         說明：T86收盤後才公布、美股當晚才開盤)。保留只是方便跟之前
         (2020-2023 / 2015-2019)已經拿到的結果直接對照。
      2. 「只錯開三大法人(美股濾網仍用T日)」——只修三大法人那部分，美股濾網
         還沒修，用來單獨看三大法人這個修正本身的影響有多大。
      3. 「三大法人+美股濾網都錯開(真正能實測)」——兩個問題都改用T-1日已知
         的資料，這才是真的能在收盤前執行的版本。如果要拿去實盤，判讀應該
         看這個版本的PF，不是前面兩個。
    設False的話，只跑第1個版本(跟最早的行為一樣)，通常不建議，僅供除錯用。

    slippage_pct：預設0.0，維持舊行為(不模擬滑價)，往下傳給每一次
    run_overnight_backtest()呼叫，均勻套用在所有候選組合、所有時間點變體上
    (不另外拆成第4個比較軸，避免跑的次數再翻倍——如果想單獨看滑價的影響，
    對同一批候選組合分別用slippage_pct=0.0跟slippage_pct>0各跑一次這支函式，
    自行比較兩次結果即可)。

    min_candidates：可選，往下傳給每次run_overnight_backtest()呼叫，見
    overnight_momentum_engine.py的說明。不傳(維持None)就完全不啟用，向後相容。
    ⚠️ 這個參數如果是新調的，應該先在原本的IS/OOS資料(2023-09-15~2026-09-14)
    上調好、鎖定下來，再拿來這裡的獨立驗證區間做最終確認——不要直接在這裡
    的資料上調參數，那樣就等於用樣本外資料選參數，失去「跨期驗證」的意義。

    min_trust_ratio：可選，往下傳給每次run_overnight_backtest()呼叫，見
    overnight_momentum_engine.py的說明。不傳(維持None)就完全不啟用，向後相容。
    ⚠️ 這個參數如果是新調的，應該先在原本的IS/OOS資料(2023-09-15~2026-09-14)
    上調好、鎖定下來，再拿來這裡的獨立驗證區間做最終確認，理由跟min_candidates
    一模一樣。
    """
    candidates = candidates or CANDIDATES
    trading_days = pipeline_inputs["trading_days"]

    common_args = dict(
        indicators_by_code=pipeline_inputs["indicators_by_code"],
        market_returns_df=pipeline_inputs["market_returns_df"],
        foreign_ratio_df=pipeline_inputs["foreign_ratio_df"],
        trust_ratio_df=pipeline_inputs["trust_ratio_df"],
        day_trading_ratio_df=pipeline_inputs["day_trading_ratio_df"],
        universe_codes=pipeline_inputs["universe_codes"],
        us_market_returns_df=pipeline_inputs.get("us_market_returns_df"),
        ex_dividend_dates_by_code=pipeline_inputs.get("ex_dividend_dates_by_code"),
        trading_days=trading_days,
        top_n=top_n,
        slippage_pct=slippage_pct,
        min_candidates=min_candidates,
        min_trust_ratio=min_trust_ratio,
    )

    # (use_prior_day_chip_data, use_prior_day_us_market_data, 顯示用標籤)
    timing_variants = [(False, False, "完全T日當天(舊版，不可能實測)")]
    if compare_chip_timing:
        timing_variants.append((True, False, "只錯開三大法人(美股濾網仍用T日)"))
        timing_variants.append((True, True, "三大法人+美股濾網都錯開(真正能實測)"))

    rows = []
    total_runs = len(candidates) * len(timing_variants)
    done = 0
    for cand in candidates:
        for use_chip_lag, use_us_lag, timing_label in timing_variants:
            done += 1
            trades = ome.run_overnight_backtest(
                **common_args,
                signal_weights=cand.get("signal_weights"),
                tech_weight=cand.get("tech_weight", ome.TECH_WEIGHT),
                chip_weight=cand.get("chip_weight", ome.CHIP_WEIGHT),
                atr_stop_mult=cand["atr_stop_mult"],
                atr_target_mult=cand["atr_target_mult"],
                use_prior_day_chip_data=use_chip_lag,
                use_prior_day_us_market_data=use_us_lag,
            )
            summary = ome.summarize_overnight(trades)
            rows.append({
                "label": cand["label"],
                "chip_timing": timing_label,
                "use_prior_day_chip_data": use_chip_lag,
                "use_prior_day_us_market_data": use_us_lag,
                "total_trades": summary["total_trades"],
                "win_rate": summary["win_rate"],
                "profit_factor": summary["profit_factor"],
                "total_pnl": summary["total_pnl"],
                "avg_pnl": summary["avg_pnl"],
            })
            print(f"[{done}/{total_runs}] {cand['label']} ({timing_label}) "
                  f"-> {summary['total_trades']}筆, PF={summary['profit_factor']:.2f}, "
                  f"勝率={summary['win_rate']:.1f}%, 損益={summary['total_pnl']:,.0f}", flush=True)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="用鎖定的候選參數，在一段全新的獨立歷史區間上驗證")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD（務必跟原本的區間不重疊）")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD（務必跟原本的區間不重疊）")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--refresh", action="store_true", help="忽略快取，強制重新下載所有資料")
    parser.add_argument("--slippage-pct", type=float, default=0.0,
                         help="模擬滑價百分比(預設0=不模擬)，例如0.1代表買進多付0.1%%、賣出少拿0.1%%")
    parser.add_argument("--universe", choices=list(UNIVERSE_CHOICES.keys()), default="59",
                         help="選股池：59=目前已驗證過的59檔舊清單(預設)；"
                              "full=taifex_universe.py的249檔完整可交易清單"
                              "(全新、未驗證的股票池，鎖定的候選參數是用59檔清單挑出來的那組，"
                              "結果不能直接跟59檔版本比較，建議當成獨立實驗看待)")
    parser.add_argument("--min-candidates", type=int, default=None,
                         help="候選股數量門檻(預設不啟用)：當天通過硬門檻的候選股數量"
                              "低於這個數字就整天不交易。⚠️這個值應該先在原本的IS/OOS"
                              "資料上調好、鎖定下來，不要直接在這裡的獨立驗證區間上試調")
    parser.add_argument("--min-trust-ratio", type=float, default=None,
                         help="投信買超比重(trust_ratio，原始數值%%，非分數)絕對門檻"
                              "(預設不啟用)：缺值或低於這個門檻的候選股直接剔除，"
                              "不進入排名評分。⚠️這個值應該先在原本的IS/OOS資料上"
                              "調好、鎖定下來，不要直接在這裡的獨立驗證區間上試調")
    args = parser.parse_args()

    start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()

    whitelist = UNIVERSE_CHOICES[args.universe]
    if args.universe == "full":
        print(f"⚠️ 使用 --universe full：{len(whitelist)}檔完整清單，"
              f"這是全新、還沒驗證過的股票池，CANDIDATES裡鎖定的訊號權重/ATR參數"
              f"仍是用59檔清單挑出來的那組，結果請當成獨立實驗看待。")

    print(f"載入獨立驗證區間資料 ({args.start} ~ {args.end}) ...")
    print("⚠️ 這段區間如果跟原本的2023-09-15~2026-09-14重疊，就不是真正獨立的驗證，"
          "請確認 --start/--end 填的是全新的期間。")
    pipeline_inputs = build_pipeline_inputs(
        whitelist, start_date, end_date, refresh=args.refresh,
    )

    if args.slippage_pct:
        print(f"⚠️ 有套用滑價模擬：slippage_pct={args.slippage_pct}%（買進多付、賣出少拿），"
              f"這會讓每筆交易的pnl比理論值更保守。")

    print(f"\n在整段獨立資料上（不切IS/OOS，因為這整段本身就是樣本外）跑鎖定的候選組合 ...\n")
    result_df = run_cross_period_validation(
        pipeline_inputs, top_n=args.top_n, slippage_pct=args.slippage_pct,
        min_candidates=args.min_candidates, min_trust_ratio=args.min_trust_ratio)

    print(f"\n=== 跨期驗證結果 ===")
    print(result_df.to_string(index=False))

    print(f"\n判讀方式：")
    print(f"這幾組參數是在另一份資料(2023-09-15~2026-09-14)上，經過訊號拆解、ATR掃描、"
          f"合併測試三層挑選出來的，現在是它們第一次接觸這段全新的資料。")
    print(f"如果「最強3個」那幾組的PF在這裡還是明顯 > 1（尤其是要贏過對照組），"
          f"代表這個選股+風控邏輯有機會是真訊號，不是被原本那3年的市場環境或"
          f"多層選擇撐起來的巧合；如果PF掉到1以下或接近對照組，"
          f"代表前面的結果沒有真正的跨期穩健性，還需要回頭重新設計，"
          f"不能直接拿去實盤用。")

    print(f"\n⚠️ 關於這三個版本：")
    print(f"「完全T日當天」用的是T日自己當天的三大法人買賣超資料+美股報酬率選股，"
          f"這兩件事在實務上都不可能做到——三大法人買賣超日報(T86)是T日收盤後才"
          f"公布；美股要等到台北時間T日晚上9點半後才開盤。保留這個版本只是方便跟"
          f"之前已經拿到的結果對照差多少。")
    print(f"「只錯開三大法人」讓你單獨看三大法人這個修正本身的影響有多大"
          f"(美股濾網還沒修)。")
    print(f"「三大法人+美股濾網都錯開」才是真的能在收盤前執行的版本(兩個都改用"
          f"T-1日已知的資料)。如果要拿去實盤，判讀請看這個版本的PF，不是前面兩個"
          f"——即使數字很接近，實際能用的也只有這一個。")


if __name__ == "__main__":
    main()
