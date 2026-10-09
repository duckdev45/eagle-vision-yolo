# 🦅 Eagle Vision Project: System Architecture Design Guide (V2.0)

> **2026-09-19 現行主線：PMS 施作項目（WORK_ITEM）照片工種分類＋新工種候選。**
> 入口 `make pms-app`，盤點 `make pms-status`；設定後可用 `make pms-ai` 產生圖像＋標題分類建議。資料契約、審閱包及訓練方式見
> [PMS 工種工作台](docs/PMS-CLASSIFICATION.md)。下方 V2.0 是較早的整體架構说明；
> PMS 主線以新文件與 `pipeline/pms_workflow.py` 為準，操作台入口是 `src/app.py`。
> 照片上傳後即時判斷可用 [獨立 FastAPI 服務](service/README.md)；PMS NestJS 負責驗證、取原圖、保存判斷與回饋。

**Version:** 2.0
**Date:** 2026-09-01（修訂：指令表對齊現行 Makefile，知識庫補落地數據）
**Status:** **🚀 Architecture Complete (Logical)**
**Purpose:** This document defines the canonical microservice architecture and functional separation of concerns for all QC/Audit operations.

> **公司資料不入 git**：`reference/`（QS 標準、合約工作約定）與 `reports/`（評估快照，
> 含工地照片）皆已列 `.gitignore` 並從 git 歷史移除。這兩處是公司營運資料，repo 是 public。
> 本機備份：`~/eagle-vision-reference-20260901.tar`、`~/eagle-vision-20260901-full.bundle`。

---

## 💡 核心設計原則 (Core Principles)

1.  **職責分離 (Separation of Concerns):** 每個模組必須擁有單一、可驗證的職責。數據、規則、計算、預覽，各司其職。
2.  **單點流程控制 (Single Entry Point):** PMS 訓練鏈的步驟定義只有一份——`pipeline/pms_workflow.py`；CLI（`src/pms.py`）與操作台（`src/ui/pipeline.py`）都消費它，不各寫一套。
3.  **資料層先行 (Data Layer First):** 規範與合約的查詢一律經 `core/qs_data.py`、`core/contractdata.py`；PMS 側的照片、建議與裁決經 `core/pms_*.py`。腳本不直接解析 `reference/` 原始檔。
4.  **Language Rule:** 所有程式碼和註解必須僅使用 **English and Chinese**。

> 2026-10-09：V2.0 草稿期留下的示範層（根 `app.py`、`pipeline/run_full_qc_workflow.py`、
> `core/data_loader.py`、`core/models.py`、`core/inference_utils.py`、`inference/model_runner.py`）
> 已移除——六個檔案互相 import、零程式消費，但文件把它們寫成鐵則，造成「文件說一套、程式跑另一套」。
> 本節現在描述的是**實際跑得起來的路徑**；要翻舊示範層看 git 歷史。

---

## 🧱 模組化架構圖 (Modular Architecture Map)

| Module Path | Responsibility | Key Function/Purpose |
| :--- | :--- | :--- |
| `src/app.py` | **Presentation Shell** | **UI Layer Only.** 最上層選系統（PMS 日報 / QMS 稽核 / 規範庫），分頁實作在 `src/ui/*`。任何業務邏輯均不在此計算。 |
| `pipeline/pms_workflow.py` | **Workflow Orchestrator** | PMS 訓練鏈的步驟定義與前置檢查（版本名、切分、特徵齊備、catalog 版本）。CLI 與操作台共用。 |
| `core/qs_data.py` | **QS Knowledge Service** | 載入與查詢公司 ISO（QS）品質標準；A~E 工具分派、請款靶、合約相依項。 |
| `core/contractdata.py` | **Contract Service** | 逐案合約工作約定：付款節點、罰則、驗收數值、QS 交叉引用。 |
| `core/labeler.py` | **Rule Engine** | `labels.yaml` 的規則匹配（順序即優先權），把日報標題歸到工程分類樹節點。 |
| `core/review_utils.py` | **Calculation Service** | 複核佇列分層（tier）與孤兒鄰居參考，CLI `src/review.py` 與操作台共用同一份。 |
| `core/evaluation_metrics.py` | **Scoring Service** | 工項融合（同日報同標題的兄弟照一起看）與信心門檻；`service/` 有平行實作，由 `tests/test_service_fusion_parity.py` 守住不漂移。 |
| `core/pms_source.py` / `pms_store.py` / `pms_exchange.py` / `pms_review.py` / `pms_vision.py` | **PMS Data Services** | 照片來源、本機事件 SQLite、審閱包匯出匯入、裁決與候選、VLM 看圖建議。 |
| `core/defects.py` | **Defect Box Service** | 缺失框資料層（HUMAN 層框才進表，AI_GUESS 不寫）。 |
| `src/*.py` | **Pipeline Scripts** | 一步一支、可單跑：`sync` → `prepare` → `features` → `split` → `train` → `evaluate` → `explain` → `journal`。 |
| `src/export_label_pack.py` | **Data Boundary Enforcer** | 去識別化判準包匯出（`make label-pack`）。白名單在 `docs/DATA-BOUNDARY.md`，欄位型別不符就整份拒匯——紅線由程式擋，不靠人眼。 |
| `service/vision_api/` | **Standalone Inference API** | 照片上傳後即時判斷的獨立 FastAPI（見 `service/README.md`），不與訓練端共用程序。 |

---

## ⚙️ 關鍵功能流轉描述 (Feature Flow Description)

### 1. 數據輸入流程 (Data Ingestion & Sync)
*   **目標:** 確保所有歷史/即時照片都能被系統識別。
*   **路徑:** `src/app.py` ①同步 $\to$ `src/sync.py`（或 `make sync`）。
*   **重點:** 此步驟負責更新 `derived/` 屬性文件（如 `manifest.csv`, `report_index.csv`），是所有後續計算的入場憑證。

### 2. 核心流程 (The Full QC Loop)
*   **目標:** 從原始照片流到最終的仲裁分數，必須嚴格按序執行。
*   **流程:** `make retrain` = `data`（sync → prepare → features）$\rightarrow$ `model`（split → train → evaluate → explain → journal）。PMS 專用版同一條鏈走 `pipeline/pms_workflow.py`（`make pms-retrain`）。
*   **核心挑戰:** 每一階段的輸出，都必須作為下一階段的**唯一輸入**；切換線上模型是獨立動作（`make use`），不是 `make model` 的副作用。

### 3. 關鍵機制說明 (Critical Mechanisms)

*   **🏷️ 標籤系統 (`core/labeler.py`):** 標籤命名必須遵循 `{工程類別}-{施作內容}` 的格式，且核心邏輯必須將舊的、模糊的標籤，轉換為屬於**「工程分類樹」**中的標準類別節點。
*   **🧠 模型推理 (`src/predict.py` / `src/explain.py`):** 模型僅能運算「它看到了什麼」（Top1/Margin Score），它無法判斷「為什麼這張照片會這樣拍攝」（Why / Context）。推論前處理必須與訓練的 `eval_tf` 逐步一致。
*   **📚 知識庫 (Knowledge):**
    *   **合同資料:** 優先權最高 (Contract > QS)。
    *   **QS 標準:** 適用於所有缺乏合約規定的通用行業標準。
    *   **實戰規則:** 當合約和 QS 發生矛盾（如：泛水高度），應以**合約約定**為最終權威。

---
## 🛠️ 操作指令表 (CLI Reference)

為維持工作流程的穩定性，所有的跨模組操作應使用 `make` 腳本，而非分散的 `uv run` 命令。

```bash
# Basic Workflow Commands
make sync                         # 運行日報同步和照片下載（只讀，不觸發重訓）
make retrain SPLIT=vN             # data（sync→prepare→features）→ model（split→train→evaluate→explain→journal）
make use SPLIT=vN                 # 看過分數後，才把操作台切到這一版

# Specialized Tasks
make newclass                     # 發現新的工種 / 規則空缺 (用於擴展規則庫)
make qs / make contract           # QS 標準與合約工作約定統計（需 reference/）
make help                         # 其餘 target 一覽
```

*Self-Correction Note: The most common mistake is treating model output as ground truth. Always treat model prediction as a **Signal** that requires human arbitration/validation against the `data/review.csv` record.*
