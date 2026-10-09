"""影子推論的來源限制、批次前處理一致性與唯讀回應。"""

from __future__ import annotations

import json
import pickle
import threading
from http.server import HTTPServer
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

import prepare
from core import paths, pms_photos
from pms_inference import InferenceError, PmsShadowPredictor, prepared_photo
from pms_shadow_api import handler_for


class TwoClassProbe:
    classes_ = np.array(["泥作-打底", "油漆-塗裝"])

    def predict_proba(self, embeddings):
        assert embeddings.shape == (1, 2)
        return np.array([[0.8, 0.2]])


def _large_photo(path):
    pixels = np.random.default_rng(7).integers(0, 256, size=(300, 600, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, quality=95)


def _test_artifact():
    (paths.SPLITS / "vtest.json").write_text(
        json.dumps(
            {
                "source": "report",
                "trainLegacy": [],
                "classes": ["泥作-打底", "油漆-塗裝"],
                "pmsCatalogVersion": pms_photos.catalog_version(),
            },
            ensure_ascii=False,
        )
    )
    (paths.MODELS / "probe-siglip-vtest.pkl").write_bytes(
        pickle.dumps({"clf": TwoClassProbe(), "split": "vtest", "encoder": "siglip"})
    )


def test_online_preparation_matches_batch_and_returns_versioned_suggestion(pms_env, monkeypatch, tmp_path):
    raw = paths.PHOTOS / "a.jpg"
    _large_photo(raw)
    expected = tmp_path / "batch.jpg"
    prepare.process(raw, expected, mask=True)
    assert prepared_photo("a") == expected.read_bytes()

    _test_artifact()
    predictor = PmsShadowPredictor("vtest")
    monkeypatch.setattr(predictor, "_encode", lambda _: np.array([[1.0, 0.0]]))
    response = predictor.predict("a")
    assert response["status"] == "review_required"
    assert response["suggestedClass"] == "泥作-打底"
    assert response["modelScore"] == 0.8
    assert response["modelVersion"] == "vtest"
    assert response["catalogVersion"] == pms_photos.catalog_version()
    assert response["modelSha256"]
    assert response["encoderVersion"].startswith("siglip/")


def test_rejects_bad_or_non_work_item_photo(pms_env):
    _large_photo(paths.PHOTOS / "a.jpg")
    (paths.PHOTOS / "a.jpg").write_bytes(b"test-jpeg-bytes")
    with pytest.raises(InferenceError, match="too_small"):
        prepared_photo("a")
    with pytest.raises(InferenceError, match="施作項目"):
        prepared_photo("foreign")
    with pytest.raises(InferenceError, match="施作項目"):
        prepared_photo("inactive")


def test_only_pms_model_and_same_catalog_can_serve(pms_env):
    _test_artifact()
    path = paths.SPLITS / "vtest.json"
    split = json.loads(path.read_text())
    split["source"] = "report+legacy"
    path.write_text(json.dumps(split))
    with pytest.raises(InferenceError, match="PMS-only"):
        PmsShadowPredictor("vtest")
    split["source"] = "report"
    split["pmsCatalogVersion"] = "old-catalog"
    path.write_text(json.dumps(split))
    with pytest.raises(InferenceError, match="分類表"):
        PmsShadowPredictor("vtest")


def test_http_endpoint_accepts_only_file_id_and_never_marks_accepted():
    class Stub:
        version = "vtest"

        def predict(self, file_id):
            return {"fileId": file_id, "status": "review_required", "modelVersion": self.version}

    server = HTTPServer(("127.0.0.1", 0), handler_for(Stub()))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/pms/photo-predictions"
        request = Request(
            url,
            data=json.dumps({"fileId": "a"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=2) as response:
            payload = json.load(response)
        assert payload == {"fileId": "a", "status": "review_required", "modelVersion": "vtest"}
    finally:
        server.shutdown()
        worker.join(timeout=2)
        server.server_close()


class SwitchProbe:
    """第一張照片偏打底、兄弟照強烈指向油漆。"""

    classes_ = np.array(["泥作-打底", "油漆-塗裝"])

    def predict_proba(self, embeddings):
        return np.array([[0.55, 0.45]]) if embeddings[0, 0] > 0 else np.array([[0.1, 0.9]])


def test_prediction_fuses_work_item_siblings_and_reads_encoder_from_split(pms_env, monkeypatch):
    import pandas as pd

    from core import model_registry as registry

    man = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    man["dailyReportInfoId"] = ["r1", "r1", "", "", "", "", "", "", ""]
    man.loc[man.fileId == "b", "title"] = "打底施作"  # a、b 同日報同標題＝同工項
    man.to_csv(paths.MANIFEST, index=False)
    for fid in ("a", "b"):
        _large_photo(paths.PHOTOS / f"{fid}.jpg")
    split = {
        "source": "report",
        "trainLegacy": [],
        "classes": ["泥作-打底", "油漆-塗裝"],
        "pmsCatalogVersion": pms_photos.catalog_version(),
        "encoder": registry.DEFAULT_ENCODER,
    }
    (paths.SPLITS / "vfuse.json").write_text(json.dumps(split, ensure_ascii=False))
    (paths.MODELS / f"probe-{registry.DEFAULT_ENCODER}-vfuse.pkl").write_bytes(
        pickle.dumps({"clf": SwitchProbe(), "split": "vfuse", "encoder": registry.DEFAULT_ENCODER})
    )
    predictor = PmsShadowPredictor("vfuse")
    calls = iter([np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0]])])
    monkeypatch.setattr(predictor, "_encode", lambda _: next(calls))
    response = predictor.predict("a")
    assert response["workItemPhotos"] == 2
    assert response["suggestedClass"] == "油漆-塗裝"  # 單張會答打底，兄弟照把它拉回來
    assert response["status"] == "review_required"
    assert response["encoderVersion"].startswith(f"{registry.DEFAULT_ENCODER}/")
    from core.evaluation_metrics import REVIEW_CONFIDENCE

    assert response["lowConfidence"] is (response["modelScore"] < REVIEW_CONFIDENCE)


class StageProbe:
    classes_ = np.array(["泥作-打底", "泥作-粉光", "油漆-塗裝"])

    def predict_proba(self, embeddings):
        return np.array([[0.48, 0.47, 0.05]])


def test_prediction_uses_title_stage_when_mortar_stage_is_uncertain(pms_env, monkeypatch):
    import pandas as pd

    man = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    man.loc[man.fileId == "a", "title"] = "5F牆面粉光缺失改善"
    man.to_csv(paths.MANIFEST, index=False)
    _large_photo(paths.PHOTOS / "a.jpg")
    _test_artifact()
    split = json.loads((paths.SPLITS / "vtest.json").read_text())
    split["classes"] = list(StageProbe.classes_)
    split["pmsCatalogVersion"] = pms_photos.catalog_version()
    (paths.SPLITS / "vtest.json").write_text(json.dumps(split, ensure_ascii=False))
    (paths.MODELS / "probe-siglip-vtest.pkl").write_bytes(
        pickle.dumps({"clf": StageProbe(), "split": "vtest", "encoder": "siglip"})
    )
    predictor = PmsShadowPredictor("vtest")
    monkeypatch.setattr(predictor, "_encode", lambda _: np.array([[1.0, 0.0]]))
    response = predictor.predict("a")
    assert response["suggestedClass"] == "泥作-粉光" and response["stageDecision"] == "title"
    assert response["defectTitle"] is True
    assert response["status"] == "review_required"
