# 台股策略回測比較工具

比較「原版（ATR計算有bug）」跟「新版（ATR計算已修正）」這兩套邏輯，
用同一份歷史股價回測，看修正 ATR 公式對策略績效實際上造成多大差異。

兩個版本的**進場條件完全一樣**（均線多頭排列、跳空風控），差別只在停損/停利價位的計算方式，
所以績效差異可以完全歸因於 ATR 公式的修正，不會被其他因素干擾。

---

## 一、檔案說明

| 檔案 | 用途 |
|---|---|
| `engine.py` | 回測核心邏輯（跟 `main.py` 的策略邏輯保持一致） |
| `data_loader.py` | 從 Yahoo Finance 下載歷史股價，並存成本機快取 |
| `compare.py` | 主程式，跑兩種版本並輸出比較報告 |
| `requirements.txt` | 需要安裝的 Python 套件清單 |

---

## 二、怎麼在本機執行（最簡單、不用 GitHub）

如果你只是想先在自己電腦上跑跑看，**不需要建立 GitHub 專案**，照下面步驟做就好：

### 步驟 1：安裝 Python

如果電腦還沒有 Python，先到 https://www.python.org/downloads/ 下載安裝（3.10 版以上）。
安裝時記得勾選 **"Add Python to PATH"** 這個選項。

### 步驟 2：把這幾個檔案放進同一個資料夾

例如放到桌面的 `stock-backtest` 資料夾裡，確認 `engine.py`、`data_loader.py`、`compare.py`、
`requirements.txt` 都在同一層。

### 步驟 3：安裝套件

打開終端機（Windows 叫「命令提示字元」或 PowerShell，Mac 叫「終端機 Terminal」），
切換到那個資料夾，輸入：

```bash
cd 桌面/stock-backtest
pip install -r requirements.txt
```

### 步驟 4：執行回測

```bash
python compare.py
```

預設會抓「最近兩年」的資料。想指定期間可以加參數，例如：

```bash
python compare.py --start 2022-01-01 --end 2025-01-01
```

第一次執行會需要一點時間下載約 60 檔股票的歷史資料，之後重複執行會用本機快取，速度快很多。
如果想強制重新下載最新資料：

```bash
python compare.py --refresh
```

### 步驟 5：看結果

執行完會在同一個資料夾下產生一個 `results/` 資料夾，裡面有：

- `summary.txt`：兩版績效比較表格（交易次數、勝率、平均報酬、累積報酬、最大回撤）
- `trades_buggy.csv`：原版每一筆交易明細，可以用 Excel 打開
- `trades_correct.csv`：新版每一筆交易明細
- `equity_curve.png`：兩版權益曲線比較圖，用看圖檔軟體打開就能看到走勢對比

---

## 三、如果你想放到 GitHub 上（英文介面逐步教學）

以下是完全從零開始建立一個新的 GitHub 專案的步驟，因為介面是英文，我把每個英文按鈕對應的中文意思都寫出來。

### 步驟 1：註冊/登入 GitHub

前往 https://github.com ，如果還沒有帳號，點右上角 **"Sign up"**（註冊）；已經有帳號就點
**"Sign in"**（登入）。

### 步驟 2：建立新的 repository（專案倉庫）

1. 登入後，點畫面右上角的 **"+"** 圖示，選 **"New repository"**（新增倉庫）
2. **"Repository name"**（倉庫名稱）：填一個英文名字，例如 `tw-stock-backtest`（不要有空格，可以用減號 `-`）
3. **"Description"**（描述）：可以簡單寫「台股策略回測比較」，非必填
4. 選擇 **"Public"**（公開，任何人都看得到）或 **"Private"**（私人，只有你能看到）
   - 你之前提到要「公共的」，選 **Public** 就對了
5. 下面 **"Add a README file"** 這個核取方塊可以打勾，會自動幫你建立一個空的說明檔
6. 最下面綠色按鈕 **"Create repository"**（建立倉庫），點下去

### 步驟 3：把檔案上傳上去

建立完成後會進到這個新專案的頁面，有兩種方式上傳檔案：

**方式A：網頁直接拖拉上傳（最簡單，不用裝任何東西）**

1. 點頁面上 **"Add file"**（新增檔案）→ **"Upload files"**（上傳檔案）
2. 把 `engine.py`、`data_loader.py`、`compare.py`、`requirements.txt`、這份 `README.md`
   直接拖拉到網頁上那個虛線框框裡
3. 下面 **"Commit changes"**（提交變更）區塊，直接點綠色按鈕 **"Commit changes"** 就完成了

**方式B：用 Git 指令上傳（之後要常常更新程式碼建議學這個）**

在終端機輸入（`你的帳號` 跟 `tw-stock-backtest` 換成你自己的）：

```bash
cd 桌面/stock-backtest
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/你的帳號/tw-stock-backtest.git
git push -u origin main
```

第一次 push 可能會跳出視窗要你登入 GitHub 帳號密碼或做身份驗證，照畫面指示完成即可。

### 步驟 4：以後要更新程式碼

如果用方式A（網頁上傳），之後改了檔案，一樣進到專案頁面點 **"Add file" → "Upload files"** 重新上傳，
GitHub 會自動偵測是同名檔案並覆蓋更新。

如果用方式B（Git 指令），之後只要在同一個資料夾重複這三行：

```bash
git add .
git commit -m "說明這次改了什麼"
git push
```

---

## 四、如何解讀報告裡的數字

| 指標 | 意義 |
|---|---|
| 交易次數 | 這段期間總共進出場幾次 |
| 勝率 | 賺錢的交易佔全部交易的比例 |
| 平均每筆報酬 | 每一筆交易平均賺/賠多少百分比 |
| 累積報酬 | 假設本金固定投入、連續做這些交易，總共累積賺/賠多少（不含資金運用效率） |
| 最大回撤 | 從最高點到最低點，資金曾經縮水過的最大幅度，數字越負代表風險越大 |

---

## 五、重要限制（誠實告知，不要照單全收）

1. **停損/停利同一天觸發時的判定是保守假設**：如果同一天內盤中股價曾經跌破停損、又曾經漲過停利，
   回測程式會假設「先跌破停損」，因為只有日K的高低點資料，沒有辦法知道分時走勢的真實先後順序。
   這個假設對原版跟新版是一致的，所以不影響兩版之間的相對比較，但代表回測結果本身可能比實盤更保守一點。
2. **用「當日開盤價」模擬即時報價**：正式程式是抓盤中即時報價來判斷跳空風控，
   回測沒有那麼細的資料，改用當日開盤價當作近似值。
3. **回測不等於未來會賺錢**：這個工具只能告訴你「如果過去用這套邏輯操作會發生什麼事」，
   不代表策略本身是穩賺不賠的，過去績效不代表未來表現。
