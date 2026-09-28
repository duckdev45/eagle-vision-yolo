"""模型端回饋收件匣；單實例試行用 SQLite 持久化，eventId 冪等。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class FeedbackConflict(ValueError):
    pass


class FeedbackStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS class_catalog (
                    catalog_version TEXT PRIMARY KEY,
                    classes_json TEXT NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS prediction (
                    prediction_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    catalog_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS feedback_event (
                    event_id TEXT PRIMARY KEY,
                    payload_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    review_state TEXT NOT NULL DEFAULT 'pending'
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def record_prediction(self, prediction: dict, classes: list[str]) -> None:
        identity = (
            prediction["predictionId"],
            prediction["fileId"],
            prediction["imageSha256"],
            prediction["modelVersion"],
            prediction["modelSha256"],
            prediction["catalogVersion"],
        )
        with self._connect() as db:
            classes_json = json.dumps(classes, ensure_ascii=False, separators=(",", ":"))
            db.execute(
                "INSERT OR IGNORE INTO class_catalog(catalog_version,classes_json) VALUES (?,?)",
                (prediction["catalogVersion"], classes_json),
            )
            saved_catalog = db.execute(
                "SELECT classes_json FROM class_catalog WHERE catalog_version = ?",
                (prediction["catalogVersion"],),
            ).fetchone()
            if saved_catalog[0] != classes_json:
                raise FeedbackConflict("同一 catalogVersion 的類別表不同。")
            inserted = db.execute(
                """INSERT OR IGNORE INTO prediction
                   (prediction_id,file_id,image_sha256,model_version,model_sha256,catalog_version,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (*identity, datetime.now(UTC).isoformat()),
            ).rowcount
            if not inserted:
                existing = db.execute(
                    """SELECT prediction_id,file_id,image_sha256,model_version,model_sha256,catalog_version
                       FROM prediction WHERE prediction_id = ?""",
                    (prediction["predictionId"],),
                ).fetchone()
                if existing != identity:
                    raise FeedbackConflict("同一 predictionId 的預測身分不同。")

    def receive(self, payload: dict) -> bool:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode()).hexdigest()
        event_id = payload["eventId"]
        with self._connect() as db:
            prediction = db.execute(
                """SELECT file_id,image_sha256,model_version,model_sha256,catalog_version
                   FROM prediction WHERE prediction_id = ?""",
                (payload["predictionId"],),
            ).fetchone()
            expected = tuple(
                payload[key]
                for key in ("fileId", "imageSha256", "modelVersion", "modelSha256", "catalogVersion")
            )
            if prediction != expected:
                raise FeedbackConflict("找不到對應的預測，或回饋的照片與模型版本不符。")
            if payload.get("selectedClass"):
                catalog = db.execute(
                    "SELECT classes_json FROM class_catalog WHERE catalog_version = ?",
                    (payload["catalogVersion"],),
                ).fetchone()
                if not catalog or payload["selectedClass"] not in json.loads(catalog[0]):
                    raise FeedbackConflict("selectedClass 不在預測時的分類表。")
            inserted = db.execute(
                "INSERT OR IGNORE INTO feedback_event(event_id,payload_sha256,payload_json,received_at) VALUES (?,?,?,?)",
                (event_id, digest, raw, datetime.now(UTC).isoformat()),
            ).rowcount
            if inserted:
                return True
            existing = db.execute(
                "SELECT payload_sha256 FROM feedback_event WHERE event_id = ?", (event_id,)
            ).fetchone()
            if not existing or existing[0] != digest:
                raise FeedbackConflict("同一 eventId 的回饋內容不同。")
            return False
