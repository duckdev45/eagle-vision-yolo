"""Synthetic coverage regression tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.evaluation_metrics import coverage_at_precision


@pytest.mark.parametrize("correct", [[True, False], [False, True]])
def test_ties_cannot_be_partially_accepted(correct):
    assert coverage_at_precision(correct, [0.9, 0.9]) == 0.0


def test_abstentions_are_not_coverage():
    assert coverage_at_precision([True] * 9 + [False], [1.0] * 10, abstained=[False] * 9 + [True]) == 0.9


def test_work_item_fusion_only_mixes_siblings():
    from core.evaluation_metrics import fuse_work_items

    proba = [[0.4, 0.6], [0.9, 0.1], [0.45, 0.55], [0.3, 0.7]]
    fused = fuse_work_items(proba, ["a", "a", "", "b"])
    assert fused[0].argmax() == 0 and fused[1].argmax() == 0  # 兄弟照把不確定的那張拉回來
    assert fused[2].tolist() == pytest.approx([0.45, 0.55])  # 無 key 不融合
    assert fused[3].tolist() == pytest.approx([0.3, 0.7])  # 單張工項不變
    assert fused.sum(1).tolist() == pytest.approx([1.0] * 4)
