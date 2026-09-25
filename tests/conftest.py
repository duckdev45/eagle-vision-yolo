"""隔離 PMS 工作台測試，所有寫入指向 pytest 暫存目錄。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import paths  # noqa: E402


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
