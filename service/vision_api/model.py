"""固定模型推論：不讀 PMS manifest，不寫日報，也不下載權重。"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from safetensors.torch import load_file

from . import fusion

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 與訓練前處理的截斷 JPEG 政策相同

MIN_PHOTO_BYTES = 1024
MAX_PIXELS = 30_000_000


class ModelInputError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_clean_jpeg(blob: bytes) -> bytes:
    """與 prepare.prepare_jpeg(mask=False) 同步；僅服務 2026-08-14 後的乾淨原圖。"""
    if len(blob) < MIN_PHOTO_BYTES:
        raise ModelInputError("invalid_photo", "照片過小或內容不完整。")
    try:
        with Image.open(io.BytesIO(blob)) as original:
            if original.width * original.height > MAX_PIXELS:
                raise ModelInputError("invalid_photo", "照片像素超過上限。")
            image = original.convert("RGB")
        if max(image.size) > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, "JPEG", quality=90)
        return out.getvalue()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        if isinstance(exc, ModelInputError):
            raise
        raise ModelInputError("invalid_photo", "無法解碼照片。") from exc


class VisionModel:
    def __init__(self, bundle_dir: Path, device: str = "auto"):
        metadata_path = bundle_dir / "metadata.json"
        head_path = bundle_dir / "classifier.npz"
        encoder_path = bundle_dir / "encoder.safetensors"
        self.meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.meta.get("schemaVersion") != 1 or self.meta.get("modelSource") != "pms-work-item":
            raise ValueError("不支援的模型包格式。")
        if self.meta.get("openClipVersion") != open_clip.__version__:
            raise ValueError("OpenCLIP 版本與模型包不一致。")
        if self.meta.get("preprocess") != {"longEdge": 512, "jpegQuality": 90, "policy": "clean-raw-v1"}:
            raise ValueError("前處理設定與服務程式不一致。")
        if _sha256(head_path) != self.meta["classifierSha256"]:
            raise ValueError("分類器 SHA256 驗證失敗。")
        if _sha256(encoder_path) != self.meta["encoderSha256"]:
            raise ValueError("編碼器 SHA256 驗證失敗。")
        with np.load(head_path, allow_pickle=False) as head:
            self.coef = head["coef"]
            self.intercept = head["intercept"]
            self.classes = [str(label) for label in head["classes"]]
        if self.classes != self.meta["classes"] or self.coef.shape[0] != len(self.classes):
            raise ValueError("分類器與類別表不相符。")
        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        transform = self.meta["encoderTransform"]
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.meta["encoderName"],
            pretrained=None,
            load_weights=False,
            image_mean=tuple(transform["mean"]),
            image_std=tuple(transform["std"]),
            image_interpolation=transform["interpolation"],
            image_resize_mode=transform["resizeMode"],
        )
        visual = model.visual
        visual.load_state_dict(load_file(encoder_path), strict=True)
        self.visual = visual.to(device).eval()
        with torch.inference_mode():
            sample = self.preprocess(Image.new("RGB", (512, 512))).unsqueeze(0).to(device)
            embedding_dim = self.visual(sample).shape[-1]
        if self.coef.shape[1] != embedding_dim:
            raise ValueError("分類器與編碼器維度不相符。")

    @property
    def encoder_version(self) -> str:
        # 舊包沒有 encoderKey（當時只有 siglip）
        return f"{self.meta.get('encoderKey', 'siglip')}/{self.meta['encoderName']}/{self.meta['encoderPretrained']}"

    def scores(self, file_id: str, blob: bytes) -> tuple[str, np.ndarray]:
        """(原圖 SHA256, 單張類別機率)。"""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", file_id):
            raise ModelInputError("invalid_file_id", "fileId 格式不正確。")
        raw_sha = hashlib.sha256(blob).hexdigest()
        jpeg = prepare_clean_jpeg(blob)
        with Image.open(io.BytesIO(jpeg)) as image:
            tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            vector = self.visual(tensor)
            vector = vector / vector.norm(dim=-1, keepdim=True)
        embedding = vector.cpu().numpy().astype(np.float32)[0]
        logits = self.coef @ embedding.astype(np.float64) + self.intercept
        exp = np.exp(logits - np.max(logits))
        return raw_sha, exp / exp.sum()

    def _summary(self, scores: np.ndarray, title: str | None = None) -> dict:
        """建議類別（含打底／粉光分層）、前三候選與低信心旗標。"""
        stage_rule = self.meta.get("stageRule", {})
        suggested, stage = fusion.resolve_stage(
            scores,
            self.classes,
            title,
            stage_rule.get("group", fusion.STAGE_GROUP),
            stage_rule.get("share", fusion.STAGE_SHARE),
        )
        top = np.argsort(scores)[::-1][:3]
        threshold = self.meta.get("reviewConfidence", fusion.REVIEW_CONFIDENCE)
        return {
            "suggestedClass": suggested,
            "modelScore": round(float(scores[top[0]]), 6),
            "alternatives": [
                {"class": self.classes[int(i)], "modelScore": round(float(scores[i]), 6)} for i in top
            ],
            "lowConfidence": bool(scores[top[0]] < threshold or stage == "manual"),
            "stageDecision": stage,
        }

    def predict(self, file_id: str, blob: bytes) -> dict:
        raw_sha, scores = self.scores(file_id, blob)
        return self._prediction(file_id, raw_sha, scores)

    def predict_work_item(self, photos: list[tuple[str, bytes]], title: str | None = None) -> dict:
        """同一工項多張一起判：每張回單張結果＋工項融合結果；工項整體取所有照片平均。"""
        if len({file_id for file_id, _ in photos}) != len(photos):
            raise ModelInputError("duplicate_file_id", "同一批照片的 fileId 不可重複。")
        results = []
        for file_id, blob in photos:
            try:
                results.append((file_id, *self.scores(file_id, blob)))
            except ModelInputError as exc:  # 整批退回，但要說是哪一張
                raise ModelInputError(exc.code, f"{file_id}: {exc}") from exc
        single = np.array([s for _, _, s in results])
        fused = fusion.fuse(single)
        item = np.exp(np.log(np.clip(single, 1e-9, 1)).mean(0))  # 工項整體：所有照片幾何平均
        item /= item.sum()
        return {
            "status": "review_required",
            "workItem": self._summary(item, title),
            "photos": [
                {**self._prediction(file_id, raw_sha, scores), "fused": self._summary(fused[i], title)}
                for i, (file_id, raw_sha, scores) in enumerate(results)
            ],
            "modelVersion": self.meta["modelVersion"],
            "encoderVersion": self.encoder_version,
        }

    def _prediction(self, file_id: str, raw_sha: str, scores: np.ndarray) -> dict:
        top = np.argsort(scores)[::-1][:3]
        model_sha = self.meta["trainingArtifactSha256"]
        prediction_id = hashlib.sha256(f"{file_id}:{raw_sha}:{model_sha}".encode()).hexdigest()
        return {
            "predictionId": prediction_id,
            "fileId": file_id,
            "imageSha256": raw_sha,
            "status": "review_required",
            "suggestedClass": self.classes[int(top[0])],
            "modelScore": round(float(scores[top[0]]), 6),
            "alternatives": [
                {"class": self.classes[int(i)], "modelScore": round(float(scores[i]), 6)} for i in top
            ],
            "modelVersion": self.meta["modelVersion"],
            "modelSha256": model_sha,
            "catalogVersion": self.meta["catalogVersion"],
            "encoderVersion": self.encoder_version,
            "preprocessVersion": "clean-raw-v1",
        }
