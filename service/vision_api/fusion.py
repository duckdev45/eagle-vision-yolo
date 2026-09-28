"""工項融合與打底／粉光分層判斷；純 numpy，不讀 manifest。

邏輯與訓練端 core.evaluation_metrics.fuse_work_items、core.pms_review.resolve_stage 相同
（tests/test_service_fusion_parity.py 逐一比對）。服務是獨立套件、Docker 只帶 vision_api，
所以在這裡保留一份，門檻由模型包 metadata 帶入，這裡只是預設值。
"""

from __future__ import annotations

import re

import numpy as np

REVIEW_CONFIDENCE = 0.520
STAGE_CLASSES = ("泥作-打底", "泥作-粉光")
STAGE_GROUP = 0.6
STAGE_SHARE = 0.8


def fuse(proba: np.ndarray, sibling_weight: float = 1.0) -> np.ndarray:
    """同一工項的 n 張：每張 = 自己與其餘兄弟平均的 log 機率加權平均，再正規化。"""
    lp = np.log(np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0))
    out = lp.copy()
    if len(lp) > 1:
        for i in range(len(lp)):
            sib = np.delete(lp, i, axis=0).mean(0)
            out[i] = (lp[i] + sibling_weight * sib) / (1 + sibling_weight)
    out = np.exp(out - out.max(1, keepdims=True))
    return out / out.sum(1, keepdims=True)


def title_stage(title: str | None) -> str:
    t = title or ""
    base, finish = "打底" in t, bool(re.search("粉光|粉刷", t))
    return STAGE_CLASSES[0] if base and not finish else STAGE_CLASSES[1] if finish and not base else ""


def resolve_stage(
    proba, classes: list[str], title: str | None, group: float = STAGE_GROUP, share: float = STAGE_SHARE
) -> tuple[str, str]:
    """(建議類別, model|title|manual)。圖像只負責判泥作打底/粉光群，階段不確定時看標題。"""
    top = classes[int(np.argmax(proba))]
    if not all(c in classes for c in STAGE_CLASSES):
        return top, "model"
    pb, pf = (float(proba[classes.index(c)]) for c in STAGE_CLASSES)
    total = pb + pf
    if total < group or max(pb, pf) / total >= share:
        return top, "model"
    stage = title_stage(title)
    return (stage, "title") if stage else (top, "manual")
