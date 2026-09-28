import sqlite3

import pytest
from vision_api.feedback import FeedbackConflict, FeedbackStore


def test_feedback_is_durable_and_idempotent(tmp_path):
    path = tmp_path / "feedback.sqlite"
    prediction = {
        "predictionId": "prediction-1",
        "fileId": "file-1",
        "imageSha256": "a" * 64,
        "modelVersion": "v40",
        "modelSha256": "b" * 64,
        "catalogVersion": "catalog-1",
    }
    event = {"eventId": "event-1", "action": "accept", **prediction}

    FeedbackStore(path).record_prediction(prediction, ["水電", "木作"])
    FeedbackStore(path).record_prediction(prediction, ["水電", "木作"])
    assert FeedbackStore(path).receive(event) is True
    assert FeedbackStore(path).receive(event) is False
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*), review_state FROM feedback_event").fetchone() == (1, "pending")

    with pytest.raises(FeedbackConflict):
        FeedbackStore(path).receive({**event, "action": "correct"})
    with pytest.raises(FeedbackConflict):
        FeedbackStore(path).receive({**event, "eventId": "event-2", "fileId": "wrong"})
    with pytest.raises(FeedbackConflict):
        FeedbackStore(path).receive(
            {**event, "eventId": "event-3", "action": "correct", "selectedClass": "未知"}
        )
