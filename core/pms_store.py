"""PMS 工種工作台的本機事件紀錄；review.csv 仍是照片分類真值來源。

讀取不建檔。SQLite 交易保存建議、候選與分類定義，避免多個操作台寫入互相覆蓋。
資料庫與 review.csv 同目錄，不進 derived，也不進版本控制。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from core import paths


def database_path() -> Path:
    return paths.REVIEW.with_name("pms_review.sqlite3")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def events(kind: str) -> list[dict]:
    path = database_path()
    if not path.exists():
        return []
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT seq, entity, payload, created_at FROM events WHERE kind=? ORDER BY seq", (kind,)
        ).fetchall()
    return [dict(json.loads(p), _seq=s, _entity=e, _at=t) for s, e, p, t in rows]


def latest(kind: str) -> dict[str, dict]:
    return {r["_entity"]: r for r in events(kind)}


def append(records: list[tuple[str, str, dict]], *, expected: tuple[str, str, int] | None = None) -> None:
    """一批事件原子寫入；expected 防止舊畫面蓋掉另一個視窗的新裁決。"""
    if not records:
        return
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=10) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
            "kind TEXT NOT NULL, entity TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        con.execute("CREATE INDEX IF NOT EXISTS event_key ON events(kind, entity, seq)")
        con.execute("BEGIN IMMEDIATE")
        if expected is not None:
            kind, entity, seq = expected
            current = con.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE kind=? AND entity=?", (kind, entity)
            ).fetchone()[0]
            if current != seq:
                raise ValueError("紀錄已被更新，請重新載入後確認。")
        now = datetime.now(UTC).isoformat(timespec="microseconds")
        for kind, entity, payload in records:
            clean = {k: v for k, v in payload.items() if not k.startswith("_")}
            con.execute(
                "INSERT INTO events(kind, entity, payload, created_at) VALUES (?,?,?,?)",
                (kind, entity, json.dumps(clean, ensure_ascii=False, allow_nan=False), now),
            )


def approved_classes() -> dict[str, dict]:
    return latest("class")


def validate_label(label: str) -> str:
    if not isinstance(label, str):
        raise ValueError("類名必須是字串。")
    label = label.strip()
    if len(label) > 80 or label.count("-") != 1 or not all(p.strip() for p in label.split("-")):
        raise ValueError("類名須為「工種-施作內容」，且只含一個連字號。")
    if any(ord(c) < 32 for c in label):
        raise ValueError("類名不可包含控制字元。")
    if any(c in label for c in '/\\:*?"<>|'):
        raise ValueError("類名不可包含檔名保留字元。")
    return label


def validate_file_id(file_id: str) -> str:
    if not isinstance(file_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", file_id):
        raise ValueError("無效的 PMS 照片識別碼。")
    return file_id


def review_signatures() -> dict[str, str]:
    """含列序號，連同秒、同內容的重新裁決也能使先前的暫緩失效。"""
    if not paths.REVIEW.exists():
        return {}
    rows = pd.read_csv(paths.REVIEW, dtype=str, keep_default_na=False)
    if "fileId" not in rows:
        return {}
    return {r["fileId"]: digest([i, r]) for i, r in enumerate(rows.to_dict("records"))}


def active_decisions() -> dict[str, dict]:
    signatures = review_signatures()
    return {
        fid: row
        for fid, row in latest("decision").items()
        if row.get("reviewSignature", "") == signatures.get(fid, "")
    }


def blocked_ids() -> set[str]:
    """人工列為候選／不確定／排除的照片暫停訓練，直到再次明確裁決。"""
    return {f for f, r in active_decisions().items() if r["action"] in {"candidate", "uncertain", "excluded"}}
