"""隔離 PMS 工作台測試，所有寫入指向 pytest 暫存目錄。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from core import paths


@pytest.fixture()
def pms_env(tmp_path, monkeypatch):
    root = paths.ROOT
    for name, value in list(vars(paths).items()):
        if isinstance(value, Path) and value.is_relative_to(root):
            monkeypatch.setattr(paths, name, tmp_path / value.relative_to(root))
    paths.ensure_dirs()
    cfg = {
        "version": 1,
        "junk": ["行政"],
        "fallback": "其他",
        "drop_fallback": True,
        "min_class_size": 2,
        "small_class": "drop",
        "rules": [
            {"pattern": "打底", "label": "泥作-打底"},
            {"pattern": "油漆", "label": "油漆-塗裝"},
        ],
    }
    paths.LABELS_YAML.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    rows = []
    for fid, title, day, site in [
        ("a", "打底施作", "2026-08-01", "s1"),
        ("b", "油漆施作", "2026-08-02", "s1"),
        ("u1", "消音板施作", "2026-08-03", "s1"),
        ("u2", "消音板施作", "2026-08-04", "s2"),
        ("c", "", "2026-08-04", "s1"),
        ("d", "行政", "2026-08-04", "s1"),
        ("inactive", "打底", "2026-08-01", "s1"),
        ("future", "打底", "2099-01-01", "s1"),
        ("foreign", "打底", "2026-08-01", "s1"),
    ]:
        rows.append(
            {
                "fileId": fid,
                "title": title,
                "reportDate": day,
                "constrId": site,
                "constrName": site,
                "dataset": "legacy" if fid == "foreign" else "pms",
                "source": "WORK_ITEM",
                "active": fid != "inactive",
                "status": "SUBMITTED",
                "chipsOn": "",
                "specKey": "",
                "predWorkItem": "",
                "syncedAt": "2026-09-01T00:00:00Z",
            }
        )
        Image.new("RGB", (60, 40), (120, 130, 140)).save(paths.PHOTOS / f"{fid}.jpg")
    pd.DataFrame(rows).to_csv(paths.MANIFEST, index=False)
    yield {"root": tmp_path, "cfg": cfg, "rows": rows}


@pytest.fixture()
def route_queue(pms_env):
    """寫一份「最近一次分流」：指定照片在人工佇列。收件匣、進階複核、工作台、make queue 都讀它。"""
    import json

    from core import routing

    def write(file_ids: list[str], reason: str = "模型信心不足") -> None:
        paths.ROUTE.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "fileId": f,
                "bucket": "queue",
                "reason": reason,
                "priority": routing.REASONS.index(reason),
                "ruleClass": "",
                "modelClass": "",
                "modelConfidence": "",
                "signal": "live",
                "stageSource": "",
                "reportDate": "2026-08-01",
            }
            for f in file_ids
        ]
        pd.DataFrame(rows, columns=routing.COLUMNS).to_csv(paths.ROUTE / "latest.csv", index=False)
        (paths.ROUTE / "latest.json").write_text(
            json.dumps({"routedAt": "2026-08-02T00:00:00+00:00", "model": "v9", "counts": {}}),
            encoding="utf-8",
        )

    return write
