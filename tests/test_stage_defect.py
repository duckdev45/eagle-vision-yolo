"""打底／粉光分層判斷與缺失旗標（v17）。"""

from __future__ import annotations

import numpy as np
import pytest

from core import pms_review as review

CLASSES = ["油漆-批土塗裝", "泥作-打底", "泥作-粉光"]


@pytest.mark.parametrize(
    "title,expected",
    [
        ("14F內部牆面打底", "泥作-打底"),
        ("13F內部牆面粉光", "泥作-粉光"),
        ("外牆粉刷", "泥作-粉光"),
        ("1F公設中庭砌磚/打底施作", "泥作-打底"),
        ("打底及粉光", ""),  # 兩個階段都寫 → 不算明確
        ("13F~7F外牆磁磚貼飾", ""),
        (None, ""),
    ],
)
def test_title_stage(title, expected):
    assert review.title_stage(title) == expected


def test_resolve_stage_uses_title_only_inside_uncertain_group():
    uncertain = np.array([0.05, 0.50, 0.45])  # 群內 0.95，較大者佔 0.53 < STAGE_SHARE
    assert review.resolve_stage(uncertain, CLASSES, "5F牆面粉光") == ("泥作-粉光", "title")
    assert review.resolve_stage(uncertain, CLASSES, "5F牆面") == ("泥作-打底", "manual")
    sure = np.array([0.05, 0.90, 0.05])  # 階段夠確定 → 不看標題
    assert review.resolve_stage(sure, CLASSES, "5F牆面粉光") == ("泥作-打底", "model")
    other = np.array([0.70, 0.16, 0.14])  # 不是泥作群 → 不看標題
    assert review.resolve_stage(other, CLASSES, "5F牆面粉光") == ("油漆-批土塗裝", "model")
    assert review.resolve_stage(np.array([0.6, 0.4]), ["油漆-批土塗裝", "泥作-打底"], "粉光") == (
        "油漆-批土塗裝",
        "model",
    )


def test_defect_flag_from_title_without_model(pms_env):
    import pandas as pd

    from core import paths

    man = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    man.loc[man.fileId == "b", "title"] = "油漆缺失改善"
    man.to_csv(paths.MANIFEST, index=False)
    df, _ = review.snapshot()
    rows = df.set_index("fileId")
    assert bool(rows.loc["b", "defectFlag"]) and rows.loc["b", "defectSource"] == "title"
    assert not bool(rows.loc["a", "defectFlag"])
    review.set_defect("b", False, reviewer="tester")  # 人工蓋掉標題旗標
    df, _ = review.snapshot()
    assert not bool(df.set_index("fileId").loc["b", "defectFlag"])
