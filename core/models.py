# core/models.py
"""Domain Models 的單一來源（re-export hub）。

V2.0 草稿期這裡放過一批「示範用」的資料結構，但從沒被任何程式消費——
真實實作一直長在服務層（core/qs_data.py 的 Item/Doc、core/contractdata.py 的
Clause/ContractDoc、core/labeler.py 的 Labeler）。

2026-09-01 服務層接線後，本檔改為**別名模組**：所有領域結構向服務層看齊，
避免同一個概念存在兩份形狀不同的 dataclass（舊示範版 ContractClause 用
clause_no，真實版用 no——兩份並存必然漂移）。
"""
from __future__ import annotations

from core.contractdata import Clause as ContractClause  # noqa: F401
from core.contractdata import ContractDoc as ContractDocument  # noqa: F401
from core.labeler import Labeler  # noqa: F401
from core.qs_data import Doc as QS_Doc  # noqa: F401
from core.qs_data import Item as CoreItem  # noqa: F401
from core.qs_data import Item as QS_Item  # noqa: F401

__all__ = ["CoreItem", "QS_Item", "QS_Doc", "ContractClause", "ContractDocument", "Labeler"]
