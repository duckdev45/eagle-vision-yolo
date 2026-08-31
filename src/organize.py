"""照片的結構化視圖。

raw/photos/ 刻意是平的一坨 {fileId}.{ext}（SPEC §2.1：raw 不可變、不分類）。
人要看照片時需要結構，所以在 derived/tree/ 建 symlink 樹——零複製、
規則改了 rm -rf 重跑即可，不必重新下載。

    uv run src/organize.py            # 建三種視圖
    uv run src/organize.py --copy     # 要拷貝實體檔（給非本機的人）
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
import paths  # noqa: E402
from labels import labeled_manifest  # noqa: E402

BAD = re.compile(r'[/\\:*?"<>|\n\r\t]+')


def safe(s, fallback="_") -> str:
    s = BAD.sub("_", str(s or "")).strip().strip(".")
    return (s[:60] or fallback)


def _src(file_id: str):
    hits = list(paths.PHOTOS.glob(f"{file_id}.*"))
    return hits[0] if hits else None


def build(copy: bool = False, log=print) -> dict:
    paths.ensure_dirs()
    if paths.TREE.exists():
        shutil.rmtree(paths.TREE)
    df = labeled_manifest()

    linked = missing = 0
    for r in df.itertuples():
        src = _src(r.fileId)
        if src is None:
            missing += 1
            continue
        page = f"{int(r.pageSort or 0):02d}-{safe(r.title, 'untitled')}"
        serial = f"{int(r.serial or 0):02d}"
        targets = [
            # 每天各工地的日報 → 頁 → 照片
            paths.TREE / "by-date" / safe(r.reportDate) / safe(r.constrName or r.constrId)
            / page / f"{serial}-{r.fileId}{src.suffix}",
            # 同一工地的時間軸
            paths.TREE / "by-site" / safe(r.constrName or r.constrId) / safe(r.reportDate)
            / page / f"{serial}-{r.fileId}{src.suffix}",
            # 標籤視圖：核對 labels.yaml 分得對不對，一眼掃完一類
            paths.TREE / "by-class" / safe(r.cls) / f"{r.fileId}{src.suffix}",
        ]
        for t in targets:
            t.parent.mkdir(parents=True, exist_ok=True)
            if t.exists() or t.is_symlink():
                continue
            if copy:
                shutil.copy2(src, t)
            else:
                t.symlink_to(os.path.relpath(src, t.parent))
        linked += 1

    stat = {"photos": linked, "missingFile": missing, "root": str(paths.TREE)}
    log(f"照片樹完成：{stat}")
    return stat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="拷貝實體檔而非 symlink")
    build(copy=ap.parse_args().copy)
