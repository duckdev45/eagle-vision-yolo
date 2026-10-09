import io

import pytest
from PIL import Image
from vision_api.model import ModelInputError, prepare_clean_jpeg


def test_clean_preprocessing_matches_training_path():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]  # 服務是獨立 venv，訓練端不在它的 sys.path 上
    sys.path[:0] = [str(root), str(root / "src")]
    from prepare import prepare_jpeg

    photo = Image.new("RGB", (813, 609), (37, 91, 145))
    blob = io.BytesIO()
    photo.save(blob, "PNG")

    assert prepare_clean_jpeg(blob.getvalue()) == prepare_jpeg(io.BytesIO(blob.getvalue()), mask=False)


def test_invalid_photo_is_rejected():
    with pytest.raises(ModelInputError, match="過小"):
        prepare_clean_jpeg(b"bad")
    with pytest.raises(ModelInputError, match="無法解碼"):
        prepare_clean_jpeg(b"x" * 1024)


def _stub_model(table):
    import numpy as np
    from vision_api.model import VisionModel

    model = object.__new__(VisionModel)
    model.meta = {
        "modelVersion": "vX",
        "trainingArtifactSha256": "b" * 64,
        "catalogVersion": "c1",
        "encoderKey": "so400m",
        "encoderName": "ViT-SO400M-14-SigLIP-384",
        "encoderPretrained": "webli",
        "reviewConfidence": 0.52,
    }
    model.classes = ["泥作-打底", "泥作-粉光", "油漆-批土塗裝"]
    model.scores = lambda file_id, blob: ("a" * 64, np.array(table[file_id]))
    return model


def test_work_item_fusion_and_stage_rule():
    model = _stub_model({"x": [0.40, 0.35, 0.25], "y": [0.05, 0.15, 0.80], "z": [0.46, 0.44, 0.10]})
    out = model.predict_work_item([("x", b""), ("y", b"")])
    assert out["status"] == "review_required"
    assert out["photos"][0]["suggestedClass"] == "泥作-打底"  # 單張
    assert out["photos"][0]["fused"]["suggestedClass"] == "油漆-批土塗裝"  # 兄弟照拉回
    assert out["encoderVersion"] == "so400m/ViT-SO400M-14-SigLIP-384/webli"
    stage = model.predict_work_item([("z", b"")], title="5F牆面粉光")
    assert (
        stage["workItem"]["suggestedClass"] == "泥作-粉光" and stage["workItem"]["stageDecision"] == "title"
    )
    manual = model.predict_work_item([("z", b"")], title="5F牆面")
    assert manual["workItem"]["stageDecision"] == "manual" and manual["workItem"]["lowConfidence"] is True
    with pytest.raises(ModelInputError, match="重複"):
        model.predict_work_item([("x", b""), ("x", b"")])


def test_legacy_bundle_encoder_version_defaults_to_siglip():
    model = _stub_model({})
    del model.meta["encoderKey"]
    assert model.encoder_version.startswith("siglip/")
