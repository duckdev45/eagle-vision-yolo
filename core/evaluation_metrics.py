"""離線評估共用指標，不讀資料、不載入模型。"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

PRECISION_BAR = 0.90

# 工項融合後 top1 信心低於此 → 轉人工（操作台複核理由、影子 API 的 lowConfidence）。
# 來源：全 PMS 母體 5 折分組 OOF（融合後）上 precision ≥ 0.95 的最低門檻，3 seed 中位數。
# so400m/v42 為 0.520（v42 考卷套用：自動 97.0%、準 0.957）；siglip/v40 為 0.558
# （自動 92.9%、準 0.949）。只是分流提示，不是自動採用門檻——未經獨立 G2 驗證
# （docs/PMS-FEEDBACK-API.md）。換模型要重算。
REVIEW_CONFIDENCE = 0.520


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


def fuse_work_items(proba, keys: Sequence[str], sibling_weight: float = 1.0):
    """同一工項（同日報 × 同標題）的兄弟照互相參考：自己與兄弟平均的 log 機率加權平均。

    PMS 一個施作項目幾乎都上傳 2 張（v40 母體 748/754 工項兩張同類），單張看不出
    打底/粉光、油漆/批土時，另一張常看得出來。只融合同一次上傳的兄弟照，不跨工項，
    所以不引入 train/test 洩漏；key 為空字串的列不融合。回傳每列重新正規化的機率。
    """
    import numpy as np

    lp = np.log(np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0))
    if len(keys) != len(lp):
        raise ValueError("proba and keys must have the same length")
    members: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        if key:
            members.setdefault(key, []).append(i)
    out = lp.copy()
    for idx in members.values():
        for i in idx if len(idx) > 1 else ():
            sib = lp[[j for j in idx if j != i]].mean(0)
            out[i] = (lp[i] + sibling_weight * sib) / (1 + sibling_weight)
    out = np.exp(out - out.max(1, keepdims=True))
    return out / out.sum(1, keepdims=True)
