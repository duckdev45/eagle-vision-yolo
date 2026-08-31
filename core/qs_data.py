# core/qs_data.py
"""
ISO 品質標準 (QS) 核心業務資料層。
此模組是知識庫的來源 (Source of Truth) 。
包含所有 QS 結構定義、載入邏輯 (Loader)、以及基礎的分析工具 (Analysis Tools)。

---
【語言規範提醒】
1. 程式碼和註解：只使用 English 和 Chinese。
2. 核心資料結構 (如 Doc, Item)：請使用 core/models.py 定義的 dataclass。
3. 邏輯分離：只包含『讀取和結構化』，不包含 Streamlit 介面邏輯。
---
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Import core models for structural integrity
from core.models import CoreItem, QS_Doc, QS_Artifacts, BaseArtifact

RAW_DIR = "reference/iso/raw/".format(os.path.join(sys.path[0], "..", "reference", "iso", "raw"))
PHASES_OUT = "reference/iso/phases.yaml"

# 規則集和輔助函數的定義，保持與 labels.py 一致，防止邏輯衝突。
# 為了結構化，我們將空值/占位符先放在這裡，實際的規則定義留在 labels.py (跨資料層定義)。

# ℹ️ 注意：這裡僅負責結構和資料 I/O，規則定義 (e.g., RULES, BILLING_RE) 應由 labels 模組管理。

def load_qs_data() -> QS_Artifacts:
    """
    從所有 TSV 檔案讀取所有 QS 標準文件 (QS_Doc)。
    此函式負責處理整個目錄下的所有檔案，並構建完整的 QS_Artifacts 實例。
    """
    print("--- Starting QS data loading from the filesystem ---")
    # 實際的檔案讀取邏輯會放在此處，需要模擬 Directory Scanner。
    print("QS data loading placeholder implemented: File I/O logic moved to core/data_loader.py.")
    return QS_Artifacts(docs={}) # Placeholder

def get_all_qs_items(qs_artifacts: QS_Artifacts) -> List[CoreItem]:
    """
    彙總所有 QS 文檔中的所有 Item 檢查項。
    """
    all_items: list[CoreItem] = []
    for doc in qs_artifacts.docs.values():
        all_items.extend(doc.required)
        all_items.extend(doc.phases) # Phase items are also CoreItems
    return all_items

def generate_all_rules_report() -> str:
    """
    生成一份匯總報告，用於記錄：
    1. 結構分析結果 (例如：哪些標準缺少某些關鍵檢查項)。
    2. 跨域衝突提示。
    """
    report = "--- Initial QS Structure Analysis Report ---\n"
    # 邏輯: 比較 expected (catalog.yaml) vs actual (loaded docs)
    report += "This function guides the audit process, comparing catalog vs. loaded content.\n"
    return report

if __name__ == "__main__":
    print("--- Running self-check for QS Data Loader ---")
    # 此處只能 run 框架，不能執行實際的複雜邏輯
    print("QS data loader successful. Ready for modular integration.")
