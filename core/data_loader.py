# core/data_loader.py
"""系統數據快照載入中心（Global Data Snapshot Loader）。

V2.0 草稿期這裡回傳過 dummy manifest + 空 QS docs；2026-09-01 起接上真實服務層：
QS 76 份、合約 6 份 195 條、labels.yaml 規則引擎、以及 data/ 的即時標籤快照。

所有業務流程（pipeline、app.py）要拿「現在系統長什麼樣」，一律經過這裡——
同一份快照給到底，不會各模組各載各的然後對不起來。
"""
from __future__ import annotations

import sys as _sys
import os as _os

for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from typing import Any, Dict  # noqa: E402

import pandas as pd  # noqa: E402

import paths  # noqa: E402


def load_full_system_snapshot(labeled: pd.DataFrame | None = None) -> Dict[str, Any]:
    """組出當前狀態的統一數據快照。

    labeled 可由呼叫端注入（操作台已有快取時直接用）；None 就現場組一份。
    data/ 不在（全新 checkout、CI）時 labeled 給空 DataFrame，其餘服務層照常——
    reference/ 在本機磁碟（不入 git），載得到就有。
    """
    from core import contractdata, qs_data
    from core.labeler import Labeler

    print("⚡ Global Data Snapshot Loader")
    qs_artifacts = qs_data.load()
    print(f"[OK] QS 標準 {len(qs_artifacts)} 份 / "
          f"REQUIRED {sum(len(d.required) for d in qs_artifacts.values())} 項")

    contract_docs = contractdata.load()
    print(f"[OK] 合約 {len(contract_docs)} 份 / "
          f"{sum(len(c.clauses) for c in contract_docs)} 條")

    labeler = Labeler.load()
    print(f"[OK] Labeler v{labeler.version or '?'} · {len(labeler.rules)} 條規則 · "
          f"min_class_size={labeler.min_class_size}")

    if labeled is None:
        if paths.MANIFEST.exists():
            from labels import labeled_manifest
            labeled = labeled_manifest()
            print(f"[OK] manifest 標籤快照 {len(labeled)} 張 / {labeled.cls.nunique()} 類")
        else:
            labeled = pd.DataFrame()
            print("[WARN] data/raw/manifest.csv 不存在——照片側快照為空（新環境屬正常）")

    return {
        "qs_artifacts": qs_artifacts,
        "contract_docs": contract_docs,
        "labeler": labeler,
        "labeled_data": labeled,
        "metadata": {"run_timestamp": pd.Timestamp.now().isoformat()},
    }


if __name__ == "__main__":
    snap = load_full_system_snapshot()
    print("\n--- Snapshot self-check: all real services loaded ---")
    print(f"QS docs: {len(snap['qs_artifacts'])} · contracts: {len(snap['contract_docs'])} · "
          f"labeled: {len(snap['labeled_data'])}")
