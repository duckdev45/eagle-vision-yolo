# pipeline/run_full_qc_workflow.py
"""高階工作流程協調器（Workflow Orchestrator）——系統執行任務的單點入口。

V2.0 草稿期這裡全是 `lambda: True` 假步驟；2026-09-01 起接上真實服務層：
- `snapshot`  → core/data_loader（QS 76 份 + 合約 195 條 + 標籤快照）
- `queue`     → core/review_utils（複核佇列，真實分層）
- `model`     → 轉發 src/ 的 split/train/evaluate 鏈（唯一真實訓練路徑）

刻意**不**在這裡重新實作訓練：src/split.py 等腳本已是被 33+ 個測試守著的真實
流程，orchestrator 的職責是「調度與回報」，不是把邏輯抄第二份。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from core.data_loader import load_full_system_snapshot

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_pipeline_step(step_name: str, step_func: Callable, *args, **kwargs) -> tuple[bool, Any, str]:
    """統一執行單一步驟。回傳 (Success, Result, Message)。"""
    print(f"\n🚀 [PIPELINE] {step_name} ...")
    try:
        start = time.time()
        result = step_func(*args, **kwargs)
        return True, result, f"✅ {step_name}（{time.time() - start:.1f}s）"
    except Exception as e:
        return False, None, f"❌ {step_name} 失敗：{type(e).__name__}: {e}"


def _sub(cmd: list[str]) -> None:
    """以 root 為 cwd 跑 src/ 的真實腳本；非零退出碼直接拋錯（fail loud）。"""
    r = subprocess.run(cmd, cwd=_ROOT, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} 退出碼 {r.returncode}")


def snapshot() -> dict:
    """載入全系統快照（QS + 合約 + 標籤規則 + manifest 標籤）。"""
    return load_full_system_snapshot()


def queue(split_name: str = "") -> Any:
    """複核佇列（真實分層）：待人工裁決的照片，照 tier 排序。"""
    import pandas as pd

    from core.review_utils import build, scores
    from labels import Labeler, human_refs, load_reviews

    labeler = Labeler.load()
    import paths as _p

    if not _p.MANIFEST.exists():
        return pd.DataFrame()
    from sync import _truthy

    df = pd.read_csv(_p.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
    from core.pms_source import work_items

    df = work_items(df)
    if "active" in df:
        df = df[_truthy(df.active)]
    df = labeler.apply(df)
    df = df.join(human_refs(df, labeler))
    import split as split_mod

    sc, test_ids = scores("siglip", split_name or split_mod.current())
    q = build(df, labeler, sc, test_ids)
    done = set(load_reviews())
    return q[~q.fileId.isin(done)].sort_values(["tier", "syncedAt"], ascending=False, kind="stable")


# 向 app.py 與舊 callers 相容的名字
def get_workflow_status(stage: str) -> Any:
    if stage == "queue_preview":
        return queue().head(20)
    if stage == "report_status":
        import paths as _p

        rows = []
        for j in sorted(_p.REPORTS_OUT.glob("*/config.json")):
            import json

            cfg = json.loads(j.read_text())
            rows.append(
                {
                    "run": j.parent.name,
                    "split": cfg.get("split"),
                    "model": cfg.get("model"),
                    "labels": cfg.get("labelsVersion"),
                }
            )
        return rows
    return {}


def run_full_qc_workflow(stage: str, split_name: str = "v1", with_data: bool = False) -> dict:
    """主協調器。stage ∈ {snapshot, sync, model, full_run}。

    - snapshot / sync：輕量，讀檔案與抓新日報。
    - model / full_run：透過 src/ 的既有鏈重訓（subprocess，fail loud）。
      這兩個是**會動到 models/ 與 reports/ 的重量級操作**，刻意不在 import 時觸發。
    """
    results: dict[str, Any] = {"status": "PENDING", "details": []}

    if stage in ("snapshot", "sync"):
        if stage == "sync":
            ok, _, msg = run_pipeline_step("Data Sync（src/sync.py）", _sub, [sys.executable, "src/sync.py"])
            results["details"].append(msg)
            if not ok:
                results["status"] = "FAILED_SYNC"
                return results
        ok, snap, msg = run_pipeline_step("System Snapshot", load_full_system_snapshot)
        results["details"].append(msg)
        results["status"] = "SYNC_COMPLETE" if stage == "sync" else "SNAPSHOT_READY"
        if ok:
            results["qs_docs"] = len(snap["qs_artifacts"])
            results["contracts"] = len(snap["contract_docs"])
            results["labeled"] = len(snap["labeled_data"])

    elif stage in ("model", "full_run"):
        if with_data or stage == "full_run":
            ok, _, msg = run_pipeline_step(
                "S1-S3 sync→prepare→features", _sub, [sys.executable, "src/sync.py"]
            )
            results["details"].append(msg)
            if not ok:
                results["status"] = "FAILED_DATA"
                return results
        ok, _, msg = run_pipeline_step(
            f"S4-7 model 鏈（split={split_name}）", _sub, ["make", "model", f"SPLIT={split_name}"]
        )
        results["details"].append(msg)
        results["status"] = "WORKFLOW_COMPLETED_SUCCESS" if ok else "FAILED_MODEL"

    else:
        raise ValueError(f"Unknown stage '{stage}' requested.")

    return results


if __name__ == "__main__":
    print("--- Workflow Orchestrator Self-Check ---")
    result = run_full_qc_workflow(stage="snapshot")
    print(f"Final Status: {result['status']}")
    for d in result["details"]:
        print(f"  {d}")
    if result["status"] != "SNAPSHOT_READY":
        sys.exit(1)
