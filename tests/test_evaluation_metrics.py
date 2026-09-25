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
