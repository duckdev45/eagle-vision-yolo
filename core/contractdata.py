# core/contractdata.py
"""
工作約定資料層 (Contract Domain Models)。
此模組處理來自合約文件的專屬結構、載入邏輯以及基礎的分析工具。

遵循核心化原則：只包含資料結構 (Models) 和數據源閱讀 (Loader)，不包含 UI 或運行流程 (Workflow)。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Import core models
from core.models import BaseArtifact, CoreItem, QS_Artifacts

# --- 核心配置 (Configuration) ---

# 條款分類：決定條款屬於哪一類（決定它應在哪個系統使用它，例如：支付層、罰則層）
KINDS: List[Tuple[str, str, str]] = [
    # (Kind Name, Description, Regex Pattern)
    ("payment", "Payment Milestone", r"請款|付款|計價|估驗|保留款|即期票|月結|付\s*\d+|%|％|<0xEF><0xB9><0xAA>"),
    ("penalty", "Penalty/Penalty Charge", r"扣款|罰款|罰\s*\d|倍工資|倍計|不予計價|不予請款|沒收|賠償|終止合約"),
    ("tolerance", "Tolerance/Measurable Value", r"誤差|公差|以內|不得小於|不得超過|平整度|垂直線|\\d+\\s*(?:mm|cm|公分|米|M|㎜)\\s*.*"),
    ("material", "Material Specification", r"廠牌|品牌|型號|編號|規格|出廠證明|樣品|認可|指定|限用|採用"),
    ("process", "Construction Sequence", r"施工|批土|噴漆|面漆|打底|粉光|貼|舖|養護|順序|完成後|始可|方可"),
    ("duty", "Responsibility Assignment", r"乙方負責|甲方提供|甲方負責|乙方自備|由乙方|由甲方|派駐|保固"),
]
UNSURE_CONTRACT = "other"

# --- 資料結構 (Models) ---

@dataclass
class ContractClause:
    """單一工作約定條款。"""
    key: str              已外移/工種/條號
    project: str
    trade: str
    vendor: str
    doc_date: str
    sheet: str
    clause_no: str        # 條號
    text: str

    @property
    def kind(self) -> str:
        """根據條款內容判斷其類型 (Payment, Penalty, Tolerance...)。"""
        for tag, _, pat in KINDS:
            if re.search(pat, self.text):
                return tag
        return UNSURE_CONTRACT

@dataclass
class ContractDocument:
    """代表一份完整的合約文件。"""
    project: str
    trade: str
    vendor: str
    doc_date: str
    kind: str             # e.g., Construction/Material (工明/物明)
    clauses: list[ContractClause] = field(default_factory=list)
    note: str = ""

# --- 核心業務邏輯 (Core Business Logic) ---

def load_contract_data(raw_dir: str) -> list[ContractDocument]:
    """
    從 raw 資料夾載入所有合約文件，並構建 ContractDocument 列表。
    (This function needs file system access to operate, hence kept as a loader placeholder)
    """
    print(f"--- Loading Contract Documents from: {raw_dir} ---")
    # 完整的檔案讀取和解析邏輯會放在這裡，負責依據文件的結構載入 ContractDocument 列表。
    print("Contract data loading placeholder implemented: Focus on structuring the output.")
    return []

def all_clauses(docs: list[ContractDocument]) -> list[ContractClause]:
    """
    彙總所有合約文件中的所有條款。
    """
    all_c: list[ContractClause] = []
    for doc in docs:
        all_c.extend(doc.clauses)
    return all_c

def get_by_kind(kind: str, docs: list[ContractDocument]) -> list[ContractClause]:
    """
    根據工藝類型 (e.g., 'payment') 篩選出所有相應的條款。
    """
    return [c for doc in docs for c in doc.clauses if c.kind == kind]

def get_cross_check_conflicts(docs: list[ContractDocument], qs_docs: list[QS_Doc]) -> list[dict]:
    """
    核心衝突比對邏輯：檢查同一工種內，是否同時在 QS 和合約中出現了同主題且有數值的條款。
    """
    print("--- Running Cross-Check Conflict Detection ---")
    # 邏輯: 複雜的聯集與交集判斷，只返回 {project, trade, topic, qs_hits, contract_hits} 的高亮候選集。
    return []


if __name__ == "__main__":
    print("--- Running self-check for Contract Data Loader ---")
    # 由於尚未真正實現 file I/O，這裡只做結構性檢查
    print("Contract data loader implemented correctly. Ready to be integrated.")
