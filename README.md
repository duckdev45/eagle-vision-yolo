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
2.  **單點流程控制 (Single Entry Point):** 所有業務流程必須通過 `pipeline/run_full_qc_workflow.py` 協調執行。
3.  **數據時態性 (Snapshotting):** 數據處理基於 `core/data_loader.py` 產生的、特定時間點的數據快照，而非實時/累進式的查詢。
4.  **Language Rule:** 所有程式碼和註解必須僅使用 **English and Chinese**。

---

## 🧱 模組化架構圖 (Modular Architecture Map)

| Module Path | Responsibility | Key Function/Purpose |
| :--- | :--- | :--- |
| `app.py` | **Presentation Shell** | **UI Layer Only.** 僅負責展示界面和觸發流程按鈕。任何業務邏輯計算，均不能在 `app.py` 執行。 |
| `pipeline/run_full_qc_workflow.py` | **Workflow Orchestrator** | **執行流程控制器。** 確保 Init $\to$ Sync $\to$ Preprocess $\to$ Predict $\to$ Report 的順序性。 |
| `core/data_loader.py` | **Data Snapshot Assembler** | 協調並載入所有數據源，輸出當前狀態的統一數據快照。 |
| `core/models.py` | **Domain Models** | 定義系統的所有不可變、核心資料結構 (The Schema)。 |
| `core/qs_data.py` | **QS Knowledge Service** | 負責載入和管理國家級的 ISO 品質標準規則。 |
| `core/contractdata.py` | **Contract Service** | 負責載入和管理專案合約文件中的特定約定細節。 |
| `core/labeler.py` | **Rule Engine** | 執行複雜的文本/場景標籤規則匹配 (核心業務邏輯)。 |
| `core/review_utils.py` | **Calculation Service** | 基於標籤規則和模型預測，計算最終的 QC 仲裁評級 (Tiering)。 |
| `inference/model_runner.py` | **ML Inference Service** | 隔離所有 AI/CV 相關的計算：模型載入、Embedding、熱區生成。 |

---

## ⚙️ 關鍵功能流轉描述 (Feature Flow Description)

### 1. 數據輸入流程 (Data Ingestion & Sync)
*   **目標:** 確保所有歷史/即時照片都能被系統識別。
*   **路徑:** `app.py` $\to$ 點擊「同步」 $\to$ 啟動 `core/data_loader.py`。
*   **重點:** 此步驟負責更新 `derived/` 屬性文件（如 `manfiest.csv`, `report_index.csv`），是所有後續計算的入場憑證。

### 2. 核心流程 (The Full QC Loop)
*   **目標:** 從原始照片流到最終的仲裁分數，必須嚴格按序執行。
*   **流程:** `workflow` $\rightarrow$ Sync $\rightarrow$ Preprocess $\rightarrow$ Model Training $\rightarrow$ Evaluation $\rightarrow$ Report.
*   **核心挑戰:** 每一階段的輸出，都必須作為下一階段的**唯一輸入**。

### 3. 關鍵機制說明 (Critical Mechanisms)

*   **🏷️ 標籤系統 (`core/labeler.py`):** 標籤命名必須遵循 `{工程類別}-{施作內容}` 的格式，且核心邏輯必須將舊的、模糊的標籤，轉換為屬於**「工程分類樹」**中的標準類別節點。
*   **🧠 模型推理 (`inference/model_runner.py`):** 模型僅能運算「它看到了什麼」（Top1/Margin Score），它無法判斷「為什麼這張照片會這樣拍攝」（Why / Context）。
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
make full_run                     # 執行完整的 L/R/E -> Train -> Report 完整週期

# Specialized Tasks
make newclass                     # 發現新的工種 / 規則空缺 (用於擴展規則庫)
make inspect_ambiguity            # 檢查目前規則匹配的召回率和模糊路徑圖
```

*Self-Correction Note: The most common mistake is treating model output as ground truth. Always treat model prediction as a **Signal** that requires human arbitration/validation against the `data/review.csv` record.*
