# core/data_loader.py
"""
【系統數據快照載入中心】(Global Data Snapshot Loader)。
這是應用程式的所有業務邏輯和工作流（如 `app.py`, `review.py`）必須調用的第一個模組。
它負責協調所有底層數據源 (QS, Contract, Labeling, Inference)，並將它們組裝成一個時間點的數據快照。

---
【運行流程】
1.  調用此模組的函式，獲取整個系統的數據快照。
2.  快照中的數據必須能順利傳遞給核心計算模組 (如 review/pipeline)。
---
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, Any

# 匯入所有核心模組的初始化/基礎組件
from core.qs_data import load_qs_data, QS_Artifacts
from core.contractdata import load_contract_data, list[ContractDocument]
from core.labeler import Labeler, apply_labeling
from core.inference_utils import get_heatmaps, calculate_embeddings

def load_full_system_snapshot() -> Dict[str, Any]:
    """
    執行一次完整的、耗時的數據載入流程，為當前運行時創建一個數據快照。
    """
    print("=========================================")
    print("⚡ Starting Global Data Snapshot Loader ⚡")
    print("=========================================")

    # 1. Load QS Standards (Static Knowledge base)
    qs_artifacts: QS_Artifacts = load_qs_data()
    print(f"[SUCCESS] QS Knowledge base loaded. Docs: {len(qs_artifacts.docs)}")

    # 2. Load Contract Agreements (Case-specific agreements)
    contract_docs: list[ContractDocument] = load_contract_data("reference/contract/raw/")
    print(f"[SUCCESS] Contract documents loaded. Total files: {len(contract_docs)}")
    
    # 3. Initialize Labeling Engine
    labeler = Labeler()
    print(f"[SUCCESS] Labeling Engine initialized. Min Class Size: {labeler.min_class_size}")

    # 4. Create Dummy/Sample Data for Labeling Test
    # 實際運行時，這部分數據 (DF) 應該從本地檔案系統載入（如 MANIFEST.csv）
    print("[WARN] Using a dummy manifest for initial labeling test execution.")
    dummy_df = pd.DataFrame({"fileId": ["id1", "id2", "id3"], "title": ["這是地磚貼飾", "玻璃窗戶的防水層", "這是個無法判別的奇怪標題"], "reportDate": ["2026-08-31", "2026-08-30", "2026-09-01"]})

    # 5. Run Labeling
    labeled_df = apply_labeling(dummy_df)
    
    # 6. 整合模型推理數據 (Placeholder)
    inference_metrics = {
        "global_heatmaps": {},
        "global_embeddings": {}
    }
    
    snapshot: Dict[str, Any] = {
        "qs_artifacts": qs_artifacts,
        "contract_docs": contract_docs,
        "labeler": labeler,
        "labeled_data": labeled_df,
        "inference_metrics": inference_metrics,
        "metadata": {
            "run_timestamp": pd.Timestamp.now().isoformat()
        }
    }
    print("\n✅ Global Data Snapshot successfully assembled!")
    return snapshot

if __name__ == "__main__":
    print("--- Running System Check for Data Loader ---")
    try:
        snapshot = load_full_system_snapshot()
        print("\n--- Data Loading System Check Complete. All core components are linked. ---")
    except Exception as e:
        print(f"\n🚨 FATAL BLOCKER: Data Loader Failed to Initialize. Error: {e}")
        print("Please check if all component initialization functions (e.g., load_qs_data) are correctly implemented for file I/O.")

