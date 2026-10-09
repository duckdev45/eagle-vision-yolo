"""PMS 施作照母體與分類表：哪些照片算數、類別有哪些、照片檔在哪。

只讀、不寫。分流（routing）、考卷（promotion）、推論、工作台、審閱包都從這裡取母體，
「哪些照片算 PMS 施作照」才只有一個定義（`core/pms_source.work_items` 之上再排除 inactive 與 excluded）。
2026-10-09 自 pms_review.py 拆出。
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from core import paths
from core import pms_store as store
from core.labeler import Labeler
from core.pms_source import work_items

PHOTO_COLUMNS = ["fileId", "title", "constrId", "constrName", "reportDate", "chipsOn", "specKey", "syncedAt"]


def catalog(lab: Labeler | None = None) -> dict[str, dict]:
    lab = lab or Labeler.load()
    result: dict[str, dict] = {}
    for pattern, label in lab.rules:
        result.setdefault(label, {"label": label, "origin": "labels.yaml", "patterns": []})[
            "patterns"
        ].append(pattern.pattern)
    for label, row in store.approved_classes().items():
        result.setdefault(label, {k: v for k, v in row.items() if not k.startswith("_")})
    return result


def catalog_version() -> str:
    return store.digest([paths.LABELS_YAML.read_text(), catalog()])


def load_pool() -> pd.DataFrame:
    """只讀 PMS 施作項目；小類、fallback、空標題保留，出工照不進分類。"""
    if not paths.MANIFEST.exists():
        return pd.DataFrame(columns=PHOTO_COLUMNS)
    df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    if "fileId" not in df:
        raise ValueError("PMS manifest 缺 fileId 欄位。")
    df = work_items(df)
    for col in PHOTO_COLUMNS:
        if col not in df:
            df[col] = ""
    if "active" in df:
        df = df[df.active.str.lower().isin(["true", "1"])]
    df = Labeler.load().drop_excluded(df)
    return df[df.fileId != ""].drop_duplicates("fileId", keep="last").reset_index(drop=True)


def require_photos(file_ids: list[str]) -> pd.DataFrame:
    """驗證 fileId 並取回母體列；任何一張不在有效施作照母體就拒絕（寫入前的守門）。"""
    if not file_ids:
        raise ValueError("請至少選一張 PMS 照片。")
    for fid in file_ids:
        store.validate_file_id(fid)
    pool = load_pool().set_index("fileId", drop=False)
    missing = set(file_ids) - set(pool.index)
    if missing:
        raise ValueError(f"照片不在有效 PMS 施作項目母體（WORK_ITEM）：{sorted(missing)[:5]}")
    return pool.loc[list(dict.fromkeys(file_ids))]


def photo_path(file_id: str) -> Path | None:
    store.validate_file_id(file_id)
    for p in sorted(paths.PHOTOS.glob(f"{file_id}.*")):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            return p
    prepared = paths.IMAGES / f"{file_id}.jpg"
    return prepared if prepared.exists() else None


def photo_bytes(file_id: str) -> bytes:
    """審阅圖統一方向與 JPEG 格式，保留到 1600px；不附加文字或模型框。"""
    path = photo_path(file_id)
    if path is None:
        raise ValueError(f"找不到照片：{file_id}")
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def work_item_keys(df: pd.DataFrame) -> dict[str, str]:
    """fileId → 工項 key（同日報 × 同標題）。缺日報 id 的照片不融合（key 空字串）。"""
    if "dailyReportInfoId" not in df:
        return {}
    rid = df.dailyReportInfoId.fillna("").astype(str)
    title = df.title.fillna("").astype(str) if "title" in df else ""
    keys = (rid + "|" + title).where(rid != "", "")
    return dict(zip(df.fileId.astype(str), keys))
