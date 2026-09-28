"""將受信任的 PMS 探針匯出為獨立 API 可載入的推論包。

    uv run --extra train python src/export_service_bundle.py --version v43

輸出位於 gitignored models/service/<version>；不含照片、manifest 或 split 成員名單。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import features
import paths
from core.evaluation_metrics import REVIEW_CONFIDENCE
from core.pms_review import STAGE_CLASSES, STAGE_GROUP, STAGE_SHARE, catalog_version


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_bundle(version: str, destination: Path, *, encoder=None) -> Path:
    """將 sklearn 頭匯成數值矩陣，split 記載的編碼器（SigLIP 家族）視覺塔匯成 safetensors。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", version):
        raise ValueError("模型版本格式不正確。")
    split = json.loads((paths.SPLITS / f"{version}.json").read_text(encoding="utf-8"))
    if split.get("source") != "report" or split.get("trainLegacy"):
        raise ValueError("API 只封裝 PMS-only 模型。")
    if split.get("pmsCatalogVersion") != catalog_version():
        raise ValueError("目前分類表與模型訓練時不符。")
    encoder_key = split.get("encoder") or features.LEGACY_ENCODER  # 舊 split 缺欄＝siglip
    model_path = paths.MODELS / f"probe-{encoder_key}-{version}.pkl"
    with model_path.open("rb") as stream:
        artifact = pickle.load(stream)  # 只讀本機受信任的訓練檔
    if artifact.get("split") != version or artifact.get("encoder") != encoder_key:
        raise ValueError("探針版本或編碼器不符。")
    clf = artifact["clf"]
    classes = [str(label) for label in clf.classes_]
    if set(classes) != set(split["classes"]) or len(classes) < 3:
        raise ValueError("探針類別與 PMS split 不符。")
    if clf.coef_.shape != (len(classes), clf.n_features_in_):
        raise ValueError("分類器不是預期的多類 LogisticRegression。")
    if destination.exists():
        raise FileExistsError(f"推論包已存在，避免覆蓋：{destination}")

    import open_clip
    from safetensors.torch import save_file

    temp = destination.with_name(destination.name + ".building")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        head_path = temp / "classifier.npz"
        np.savez_compressed(
            head_path,
            coef=np.asarray(clf.coef_, dtype=np.float64),
            intercept=np.asarray(clf.intercept_, dtype=np.float64),
            classes=np.asarray(classes),
        )
        encoder_name, pretrained = features.MODELS[encoder_key]
        pretrained_config = open_clip.get_pretrained_cfg(encoder_name, pretrained)
        if encoder is None:
            encoder, _, _ = open_clip.create_model_and_transforms(encoder_name, pretrained=pretrained)
        visual = encoder.visual
        encoder_path = temp / "encoder.safetensors"
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in visual.state_dict().items()},
            encoder_path,
        )
        # safetensors 以 0600 建檔；容器內以非 root 的 vision 使用者執行會讀不到（v40 包同樣中招）
        encoder_path.chmod(0o644)
        metadata = {
            "schemaVersion": 1,
            "modelVersion": version,
            "modelSource": "pms-work-item",
            "catalogVersion": split["pmsCatalogVersion"],
            "encoderKey": encoder_key,
            "encoderName": encoder_name,
            "encoderPretrained": pretrained,
            "encoderTransform": {
                "mean": list(pretrained_config["mean"]),
                "std": list(pretrained_config["std"]),
                "interpolation": pretrained_config["interpolation"],
                "resizeMode": pretrained_config["resize_mode"],
            },
            "openClipVersion": open_clip.__version__,
            "classes": classes,
            # 服務端 vision_api/fusion.py 的分流門檻；與 core 同值（tests/test_service_fusion_parity.py）
            "reviewConfidence": REVIEW_CONFIDENCE,
            "stageRule": {"classes": list(STAGE_CLASSES), "group": STAGE_GROUP, "share": STAGE_SHARE},
            "preprocess": {"longEdge": 512, "jpegQuality": 90, "policy": "clean-raw-v1"},
            "trainingArtifactSha256": _sha256(model_path),
            "classifierSha256": _sha256(head_path),
            "encoderSha256": _sha256(encoder_path),
        }
        (temp / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or paths.MODELS / "service" / args.version
    print(export_bundle(args.version, out))
