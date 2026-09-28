from fastapi.testclient import TestClient
from vision_api.app import create_app
from vision_api.feedback import FeedbackStore
from vision_api.model import ModelInputError

TOKEN = "t" * 32
HEADERS = {
    "X-Service-Token": TOKEN,
    "X-File-Id": "file-1",
    "X-Source-Policy": "clean-raw-v1",
    "Content-Type": "image/jpeg",
}


class FakeModel:
    def __init__(self):
        self.meta = {
            "modelVersion": "v40",
            "trainingArtifactSha256": "b" * 64,
            "catalogVersion": "catalog-1",
        }
        self.classes = ["水電", "木作"]

    def predict(self, file_id, blob):
        if blob == b"bad":
            raise ModelInputError("invalid_photo", "無法解碼照片。")
        return {
            "predictionId": "prediction-1",
            "fileId": file_id,
            "imageSha256": "a" * 64,
            "modelVersion": "v40",
            "modelSha256": "b" * 64,
            "catalogVersion": "catalog-1",
            "size": len(blob),
            "status": "review_required",
        }


def test_predict_auth_policy_and_size(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_SERVICE_TOKEN", TOKEN)
    app = create_app(FakeModel(), FeedbackStore(tmp_path / "feedback.sqlite"))
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ready", "modelVersion": "v40"}
        assert (
            client.post(
                "/v1/predict", headers={**HEADERS, "X-Service-Token": "wrong"}, content=b"ok"
            ).status_code
            == 401
        )
        response = client.post("/v1/predict", headers=HEADERS, content=b"ok")
        assert response.status_code == 200
        assert response.json()["fileId"] == "file-1"
        assert response.json()["size"] == 2
        assert client.post("/v1/predict", headers=HEADERS, content=b"bad").status_code == 422
        assert (
            client.post(
                "/v1/predict", headers={**HEADERS, "X-Source-Policy": "masked"}, content=b"ok"
            ).status_code
            == 422
        )
        assert (
            client.post("/v1/predict", headers=HEADERS, content=b"x" * (5 * 1024 * 1024 + 1)).status_code
            == 413
        )


def test_feedback_contract_and_idempotency(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_SERVICE_TOKEN", TOKEN)
    app = create_app(FakeModel(), FeedbackStore(tmp_path / "feedback.sqlite"))
    event = {
        "eventId": "event-1",
        "predictionId": "prediction-1",
        "fileId": "file-1",
        "imageSha256": "a" * 64,
        "modelVersion": "v40",
        "modelSha256": "b" * 64,
        "catalogVersion": "catalog-1",
        "action": "accept",
    }
    with TestClient(app) as client:
        client.post("/v1/predict", headers=HEADERS, content=b"ok")
        assert client.post("/v1/feedback", headers={"X-Service-Token": TOKEN}, json=event).json() == {
            "eventId": "event-1",
            "status": "queued_for_review",
            "created": True,
        }
        assert (
            client.post("/v1/feedback", headers={"X-Service-Token": TOKEN}, json=event).json()["created"]
            is False
        )
        assert (
            client.post(
                "/v1/feedback",
                headers={"X-Service-Token": TOKEN},
                json={**event, "action": "correct", "selectedClass": "水電"},
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/v1/feedback",
                headers={"X-Service-Token": TOKEN},
                json={**event, "eventId": "event-3", "modelVersion": "v39"},
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/v1/feedback",
                headers={"X-Service-Token": TOKEN},
                json={**event, "eventId": "event-4", "action": "correct", "selectedClass": "未知"},
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/v1/feedback",
                headers={"X-Service-Token": TOKEN},
                json={**event, "eventId": "event-5", "predictionId": "missing"},
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/v1/feedback",
                headers={"X-Service-Token": TOKEN},
                json={**event, "eventId": "event-2", "action": "correct"},
            ).status_code
            == 422
        )


class FakeBatchModel(FakeModel):
    def predict_work_item(self, photos, title=None):
        if any(blob == b"bad" for _, blob in photos):
            raise ModelInputError("invalid_photo", "bad: 無法解碼照片。")
        preds = [{**self.predict(file_id, blob), "predictionId": f"p-{file_id}"} for file_id, blob in photos]
        return {
            "status": "review_required",
            "workItem": {
                "suggestedClass": "木作",
                "lowConfidence": False,
                "stageDecision": "model",
                "title": title,
            },
            "photos": [{**p, "fused": {"suggestedClass": "木作"}} for p in preds],
            "modelVersion": "v40",
            "encoderVersion": "so400m/x/y",
        }


def _batch(*photos, **extra):
    import base64

    return {
        "workItemId": "item-1",
        "sourcePolicy": "clean-raw-v1",
        "photos": [
            {"fileId": fid, "contentType": "image/jpeg", "dataBase64": base64.b64encode(blob).decode()}
            for fid, blob in photos
        ],
        **extra,
    }


def test_predict_batch_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_SERVICE_TOKEN", TOKEN)
    store = FeedbackStore(tmp_path / "feedback.sqlite")
    app = create_app(FakeBatchModel(), store)
    auth = {"X-Service-Token": TOKEN}
    with TestClient(app) as client:
        assert client.post("/v1/predict-batch", json=_batch(("a", b"ok"))).status_code == 401
        body = client.post(
            "/v1/predict-batch", headers=auth, json=_batch(("a", b"ok"), ("b", b"ok2"), title="5F牆面粉光")
        ).json()
        assert body["workItemId"] == "item-1" and body["status"] == "review_required"
        assert [p["fileId"] for p in body["photos"]] == ["a", "b"]
        assert body["photos"][0]["fused"]["suggestedClass"] == "木作"
        assert body["workItem"]["title"] == "5F牆面粉光"
        bad_b64 = _batch(("a", b"ok"))
        bad_b64["photos"][0]["dataBase64"] = "not base64!"
        assert client.post("/v1/predict-batch", headers=auth, json=bad_b64).status_code == 422
        assert client.post("/v1/predict-batch", headers=auth, json=_batch(("a", b"bad"))).status_code == 422
        too_many = _batch(*[(f"f{i}", b"ok") for i in range(9)])
        assert client.post("/v1/predict-batch", headers=auth, json=too_many).status_code == 422
        masked = _batch(("a", b"ok"))
        masked["sourcePolicy"] = "masked"
        assert client.post("/v1/predict-batch", headers=auth, json=masked).status_code == 422
        big = _batch(("a", b"x" * (5 * 1024 * 1024 + 1)))
        assert client.post("/v1/predict-batch", headers=auth, json=big).status_code == 413
