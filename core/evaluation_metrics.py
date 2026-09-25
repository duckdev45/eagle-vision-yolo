"""離線評估共用指標，不讀資料、不載入模型。"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

PRECISION_BAR = 0.90


def coverage_at_precision(
    correct: Sequence[bool],
    conf: Sequence[float],
    *,
    precision_bar: float = PRECISION_BAR,
    abstained: Sequence[bool] | None = None,
) -> float:
    """取可用信心門檻達成指定 precision 的最大 coverage。

    同信心的有效作答必須整組接受或排除；不能在同分群內挑選答對者。
    棄權永不接受，但保留在 coverage 的總樣本分母。空輸入回傳 0。
    precision 不一定單調，必須檢查所有同分群的邊界；回傳值不四捨五入。
    """
    if not isfinite(precision_bar) or not 0 < precision_bar <= 1:
        raise ValueError("precision_bar must be finite and in (0, 1]")
    total = len(correct)
    if len(conf) != total or (abstained is not None and len(abstained) != total):
        raise ValueError("correct, conf and abstained must have the same length")

    groups: dict[float, list[int]] = {}
    for i, (ok, value) in enumerate(zip(correct, conf, strict=True)):
        value = float(value)
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        if abstained is not None and abstained[i]:
            continue
        group = groups.setdefault(value, [0, 0])
        group[0] += int(bool(ok))
        group[1] += 1

    accepted = successes = covered = 0
    for value in sorted(groups, reverse=True):
        good, size = groups[value]
        successes += good
        accepted += size
        if successes / accepted >= precision_bar:
            covered = accepted
    return covered / total if total else 0.0
