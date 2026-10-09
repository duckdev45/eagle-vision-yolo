"""G2 只能用兩版模型都未見過的 PMS 人工答案。"""

import json

import pandas as pd
import pytest


def test_g2_rejects_same_day_leakage_and_groups_all_gold(pms_env):
    import g2_gate
    from core import paths

    rows = [
        {
            "fileId": f"g{i}",
            "source": "WORK_ITEM",
            "active": "True",
            "constrId": "site",
            "reportDate": f"2026-08-{i // 50 + 1:02d}",
        }
        for i in range(300)
    ]
    rows.append(
        {
            "fileId": "other-on-gold-day",
            "source": "WORK_ITEM",
            "active": "True",
            "constrId": "site",
            "reportDate": "2026-08-01",
        }
    )
    pd.DataFrame(rows).to_csv(paths.MANIFEST, index=False)
    gold = pd.DataFrame({"cls": ["泥作-打底"] * 300}, index=[f"g{i}" for i in range(300)])
    for name in ("candidate", "baseline"):
        (paths.SPLITS / f"{name}.json").write_text(json.dumps({"train": [], "test": []}))

    groups = g2_gate.validate_holdout(gold, "candidate", "baseline")
    folds = g2_gate.group_folds(list(gold.index), groups, 5)
    assert sorted(f for fold in folds for f in fold) == sorted(gold.index)
    fold_of = {f: i for i, fold in enumerate(folds) for f in fold}
    assert all(
        len({fold_of[f] for f in gold.index if groups[f] == group}) == 1 for group in set(groups.values())
    )

    (paths.SPLITS / "candidate.json").write_text(
        json.dumps({"train": [], "test": [], "datasets": {"old": "legacy"}})
    )
    with pytest.raises(ValueError, match="非 PMS 訓練來源"):
        g2_gate.validate_holdout(gold, "candidate", "baseline")

    (paths.SPLITS / "candidate.json").write_text(json.dumps({"train": ["other-on-gold-day"], "test": []}))
    with pytest.raises(ValueError, match="案場日期重疊"):
        g2_gate.validate_holdout(gold, "candidate", "baseline")

    rows[0]["source"] = "WORKFORCE"
    pd.DataFrame(rows).to_csv(paths.MANIFEST, index=False)
    with pytest.raises(ValueError, match="非有效 PMS 施作項目"):
        g2_gate.validate_holdout(gold, "baseline", "baseline")


def test_g2_rejects_too_small_gold(pms_env):
    import g2_gate

    with pytest.raises(ValueError, match="至少需要 300"):
        g2_gate.validate_holdout(pd.DataFrame({"cls": ["a"]}, index=["legacy"]), "v40", "v39")
