# core/models.py
"""
核心資料模型定義 (Domain Models)。
包含整個系統最基礎、最不變的資料結構，不包含任何業務處理邏輯，純粹用於定義資料的形態。

【Language Rule】
Code and comments MUST ONLY use English and Chinese.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List


# ... (rest of the clean, limited data structure definitions)
# 移除所有日文 Label, Labeler 相关的字段。

@dataclass
class Item:
    """
    一個標準化的品管檢查項目 (QC Inspection Item)。
    """
    # ... (fields)
    pass


# ... (rest of the clean code)

# --- 跨文件共通結構 ---

@dataclass
class BaseArtifact:
    """所有 Artifact 的基礎結構，用於承載唯一識別碼和報告日期。"""
    file_id: str
    report_date: Optional[str] = None


@dataclass
class CoreItem:
    """
    基礎內容檢查項結構，對應所有系統檢查項 (QS/合約/標籤)。
    關鍵屬性 include item_no, doc_no, name, kind 等。
    """
    item_no: str
    doc_no: str
    name: str
    status: str  # "O" = 階段節點 / "R" = 檢查項
    kind: str = "UNSURE"  # A~E 工具分派 (Added later)

    @property
    def key(self) -> str:
        """根據 doc_no 和 item_no 組成的唯一主鍵。"""
        return f"{self.doc_no}-{self.item_no}"


# ── QS Artifacts ────────────────────────────────────────────────────────
@dataclass
class QS_Doc:
    """代表一份 QS 標準文件（如 QS0404）。"""
    doc_no: str
    iso_info_id: str
    name: str
    # 深度 1 的 MUST-HAVE 階段節點
    phases: list['CoreItem'] = field(default_factory=list)
    # 必須檢查的檢查項
    required: list[CoreItem] = field(default_factory=list)
    # 深度 1 但格式上是 REQUIRED 的葉節點（需要人工判斷是否為階段）
    orphan_depth1_required: list[CoreItem] = field(default_factory=list)

    @property
    def shape(self) -> str:
        """工序型態：A 有序階段 / C 完全扁平。B（部位分支）需人工判定。"""
        return "A" if self.phases else "C"


@dataclass
class QS_Artifacts:
    """所有 QS 標準文件的集合。"""
    docs: dict[str, QS_Doc]  # {docNo: QS_Doc}


# ── Contract Artifacts ────────────────────────────────────────────────────
@dataclass
class ContractClause:
    """工作約定條款的單一描述。"""
    project: str
    trade: str
    vendor: str
    doc_date: str
    sheet: str
    clause_no: str  # 條號，替代 Item.item_no
    text: str

    @property
    def key(self) -> str:
        """主鍵。包含案名/工種/條號，因為這是逐案有效的。"""
        return f"{self.project}/{self.trade}/{self.clause_no}"


@dataclass
class ContractDocument:
    """代表一份合約文件。"""
    project: str
    trade: str
    vendor: str
    doc_date: str
    kind: str  # 工明（施工）/ 物明（材料供應）
    clauses: list[ContractClause] = field(default_factory=list)
    note: str = ""

# ── Label Artifacts ──────────────────────────────────────────────────
# Labeler 內部和外部的關鍵數據結構，這些是計算出來的，不是靜態定義，不放在 models.py
