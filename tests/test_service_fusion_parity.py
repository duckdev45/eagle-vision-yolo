"""服務端 vision_api/fusion.py 與訓練端（core）的融合／階段規則必須一致。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from core import pms_review
from core.evaluation_metrics import REVIEW_CONFIDENCE, fuse_work_items

_spec = importlib.util.spec_from_file_location(
    "service_fusion", Path(__file__).resolve().parents[1] / "service" / "vision_api" / "fusion.py"
)
fusion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fusion)


def test_constants_match():
    assert fusion.REVIEW_CONFIDENCE == REVIEW_CONFIDENCE
    assert fusion.STAGE_CLASSES == pms_review.STAGE_CLASSES
    assert (fusion.STAGE_GROUP, fusion.STAGE_SHARE) == (pms_review.STAGE_GROUP, pms_review.STAGE_SHARE)


def test_fusion_and_stage_match_core():
    rng = np.random.default_rng(0)
    classes = ["油漆-批土塗裝", "泥作-打底", "泥作-粉光", "木作-天花封板"]
    titles = ["5F牆面粉光", "4F打底", "外牆", "打底粉光", None]
    for n in (1, 2, 3):
        p = rng.dirichlet(np.ones(len(classes)), size=n)
        assert np.allclose(fusion.fuse(p), fuse_work_items(p, ["k"] * n))
        for row in p:
            for title in titles:
                assert fusion.resolve_stage(row, classes, title) == pms_review.resolve_stage(
                    row, classes, title
                )
    stage_zone = np.array([0.02, 0.50, 0.46, 0.02])
    for title in titles:
        assert fusion.resolve_stage(stage_zone, classes, title) == pms_review.resolve_stage(
            stage_zone, classes, title
        )
