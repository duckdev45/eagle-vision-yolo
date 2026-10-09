# eagle-vision — Agent 守則

PMS 日報照片 → 工種分類／缺失偵測的資料與模型專案。Python，`uv` 管理。
權威文件：`README.md`（架構）、`WORKFLOW.md`（訓練流程契約）、`docs/ROADMAP.md`（路線）、`docs/PMS-CLASSIFICATION.md`（現行主線）。
文件與程式碼不符時，先改文件或先改碼，不要讓兩邊各說各話（`WORKFLOW.md` 開頭的原話）。

## 紅線：這是 public repo，公司資料不入 git

- `reference/`、`reports/`、`models/`、`data/`、`.env*` 都在 `.gitignore`，**不得 `git add -f`**（`reports/JOURNAL.md` 純文字統計是唯一例外，見 `.gitignore` 註解）。
- 不得把以下內容寫進任何會被 commit 的地方（程式、文件、測試 fixture、commit 訊息）：
  公司品質標準或合約的**原文**（條文句子、查驗項文字、工序總表）、案場名稱、工地照片、人員姓名、內網位址、金鑰。
  **QS 文件編號（`QS0403` 這種條號）是路由鍵，labels.yaml 與 `core/qs_data.py` 必須用，不在禁列**——
  禁的是把標準內容抄出來。2026-10-09 已依此把 `phase-report.html`、`qs-phases.html` 移出版控。
- **對外交付（判準包）的邊界另有一份可執行的白名單：`docs/DATA-BOUNDARY.md`。**
  `src/export_label_pack.py` 解析它、只匯出它列到的欄位、型別不符就整份拒匯
  （`tests/test_export_label_pack.py` 釘死）。要加欄位＝先改那份文件。
- 測試要用到 `reference/` 時必須能在它缺席時跳過（`make test` 已是這個行為），不要讓 CI 依賴公司資料。

## 常用指令（一律走 Makefile）

```bash
make test        # pytest + QS 資料層自檢（reference/ 不在則跳過後者）
make lint        # ruff check + format --check（只查不改）
make fmt         # ruff 自動修正，改動自行 review
make pms-app     # PMS 工種操作台（現行主線）
make pms-status  # 照片分類與待複核盤點（唯讀）
make daily       # 每日編排：分流→重訓→公平考卷過關自動切換（hermes cron 每天跑）
make route       # 只重算收件匣分流
make help        # 其餘 target 說明
```

改完程式先 `make lint && make test`。`ruff` line-length 110。

## 架構鐵則（摘自 README）

1. 職責分離：數據、規則、計算、預覽各司其職。
2. `src/app.py` 與 `src/ui/*` 只做 UI，**任何業務邏輯不得寫在這裡**。
3. PMS 訓練鏈的步驟定義只有一份：`pipeline/pms_workflow.py`（CLI `src/pms.py` 與操作台 `src/ui/pipeline.py` 都消費它）。
4. 規範／合約一律經 `core/qs_data.py`、`core/contractdata.py`；PMS 照片與裁決經 `core/pms_*.py`。腳本不自己解析 `reference/` 原始檔。
5. 程式碼與註解只用英文與中文。

> 2026-10-09：V2.0 示範層（根 `app.py`、`pipeline/run_full_qc_workflow.py`、`core/data_loader.py`、
> `core/models.py`、`core/inference_utils.py`、`inference/model_runner.py`）已刪除——零程式消費，
> 但舊鐵則 2/3/4 指著它們，等於要求 agent 往死路走。

## 訓練流程鐵則（摘自 WORKFLOW.md，皆有 `tests/test_core.py` 擋著）

- `raw/` 不可變，`derived/` 隨時可重建。
- 前處理必須與標籤完全無關。
- 畫面近乎相同的照片不可跨 train/test（日報按 `constrId × reportDate` 整天切）。
- 每個實驗都要有對照組；判斷看分組交叉驗證，不看單一數字。
- YOLO 訓練集**只收人審框（`HUMAN_REFINED`）**，純 `AI_GUESS` 一律排除（`docs/ROADMAP.md`）。
- 每日分流（`core/routing.py`）的**自動確認不得寫入 `review.csv`**；模型訊號對 split 內照片必須用分組 out-of-fold，
  不能用背過它的上線模型（`tests/test_routing.py`）。自動切換只能經 `core/promotion.py` 的公平考卷。

## 改 `labels.yaml` 的規則

- **由上而下第一個命中者勝，順序就是優先權。**
- 版本註記裡每一條「必須排在 XX 前面」都是踩過坑換來的，`tests/test_labels_yaml.py` 把它們釘死。**重排或新增規則後必須跑這支測試**。
- 路由測試用真實標題，不要編造最小案例。
- 中文沒有詞界，每條 pattern 都是子字串比對：新規則要先算會搶走既有類別幾張。優先在既有規則加負向前後瞻，其次才插新規則（插新規則會讓該 label 的「第一次出現位置」前移，可能破壞既有順序契約）。
- 改完把版本註記寫在檔頭（現有格式：`# vNN（日期）：…`）。
