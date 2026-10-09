"""唯讀 PMS 照片推論；影子測試固定使用明確指定的 PMS-only 模型。

對新同步的原圖沿用批次前處理，再以 split 檔記載的編碼器（舊 split 缺欄＝siglip）及探針預測；
同工項（同日報×同標題）的兄弟照一起編碼，機率以 fuse_work_items 融合。
這裡不修改 CURRENT、manifest、人工答案或模型檔。
"""

from __future__ import annotations

import hashlib
import io
import pickle
import re
from pathlib import Path

import numpy as np
from PIL import Image

import features
import prepare
from core import model_registry as registry
from core import paths, pms_review, pms_store
from core.evaluation_metrics import REVIEW_CONFIDENCE, fuse_work_items
from photo_quality import bytes_problem

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class InferenceError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


def _artifact(version: str) -> tuple[dict, object, str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", version):
        raise InferenceError("invalid_model_version", "模型版本格式不正確。")
    if not registry.split_path(version).is_file():
        raise InferenceError("model_unavailable", f"找不到完整的 {version} 模型與切分檔。")
    split = registry.load_split(version)
    encoder = registry.encoder_of(split)
    model_path = registry.probe_path(version, encoder)
    if encoder not in features.MODELS or not model_path.is_file():
        raise InferenceError("model_unavailable", f"找不到完整的 {version} 模型與切分檔。")
    if split.get("source") != "report" or split.get("trainLegacy"):
        raise InferenceError("not_pms_only", "影子測試只接受 PMS-only 模型。")
    model_bytes = model_path.read_bytes()
    # 僅載入本機受信任的訓練產物；不可把客戶端提交的 pickle 傳入此處。
    model = pickle.loads(model_bytes)
    if model.get("split") != version or model.get("encoder") != encoder:
        raise InferenceError("artifact_mismatch", "模型檔與切分版本或編碼器不相符。")
    if split.get("pmsCatalogVersion") != pms_review.catalog_version():
        raise InferenceError("catalog_changed", "分類表與模型訓練時不同，請先確認版本。")
    clf = model["clf"]
    if set(clf.classes_) != set(split["classes"]):
        raise InferenceError("artifact_mismatch", "模型類別與切分檔不相符。")
    return split, clf, hashlib.sha256(model_bytes).hexdigest()


def _raw_photo(file_id: str) -> Path:
    pms_store.validate_file_id(file_id)
    photos = [
        path
        for path in paths.PHOTOS.glob(f"{file_id}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(photos) != 1:
        raise InferenceError("photo_unavailable", "找不到唯一的 PMS 原圖，請先同步照片。")
    return photos[0]


def _manifest_row(file_id: str) -> dict:
    pool = pms_review.load_pool()
    rows = pool.loc[pool.fileId == file_id]
    if len(rows) != 1:
        raise InferenceError("not_work_item", "照片不在有效的 PMS 施作項目母體。")
    return rows.iloc[0].to_dict()


def prepared_photo(file_id: str) -> bytes:
    """以 manifest 的日期規則處理同步原圖；不寫入衍生層。"""
    row = _manifest_row(file_id)
    blob = _raw_photo(file_id).read_bytes()
    problem = bytes_problem(blob)
    if problem:
        raise InferenceError("invalid_photo", f"PMS 原圖無法使用：{problem}")
    clean = prepare.is_clean_report(row["reportDate"], row["syncedAt"])
    return prepare.prepare_jpeg(io.BytesIO(blob), "report", mask=not clean)


class PmsShadowPredictor:
    def __init__(self, version: str = "v40", device: str = "auto"):
        self.version = version
        self.split, self.clf, self.model_sha256 = _artifact(version)
        self.catalog_version = self.split["pmsCatalogVersion"]
        self.encoder = registry.encoder_of(self.split)
        self.encoder_version = "/".join((self.encoder, *features.MODELS[self.encoder]))
        self.device = device
        self._model = None
        self._preprocess = None

    def warmup(self) -> None:
        """啟動時載好編碼器；服務開始接請求前確認權重可用。"""
        import open_clip
        import torch

        if self._model is None:
            self.device = (
                ("mps" if torch.backends.mps.is_available() else "cpu")
                if self.device == "auto"
                else self.device
            )
            name, pretrained = features.MODELS[self.encoder]
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
            self._model = model.to(self.device).eval()
            self._preprocess = preprocess

    def _encode(self, jpeg: bytes) -> np.ndarray:
        import torch

        self.warmup()
        with Image.open(io.BytesIO(jpeg)) as image:
            tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self._model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype(np.float32)

    def _siblings(self, file_id: str) -> list[str]:
        """同日報×同標題的其他施作照；缺日報 id 就不融合。"""
        pool = pms_review.load_pool()
        keys = pms_review.work_item_keys(pool)
        key = keys.get(file_id, "")
        return [f for f, k in keys.items() if key and k == key and f != file_id]

    def predict(self, file_id: str) -> dict:
        if pms_review.catalog_version() != self.catalog_version:
            raise InferenceError("catalog_changed", "分類表與模型訓練時不同，請先確認版本。")
        jpeg = prepared_photo(file_id)
        rows = [self.clf.predict_proba(self._encode(jpeg))[0]]
        for sibling in self._siblings(file_id):
            try:  # 兄弟照壞圖或還沒同步：照樣回本張的預測，只是少一票
                rows.append(self.clf.predict_proba(self._encode(prepared_photo(sibling)))[0])
            except (OSError, ValueError):  # InferenceError 是 ValueError
                continue
        scores = fuse_work_items(np.array(rows), ["item"] * len(rows))[0]
        if not np.isfinite(scores).all() or len(scores) != len(self.clf.classes_):
            raise InferenceError("invalid_scores", "模型輸出不完整。")
        top = np.argsort(scores)[::-1][:3]
        title = _manifest_row(file_id).get("title", "")
        # 打底／粉光：圖像只確定泥作群、階段不確定時看標題；標題沒寫階段 → manual（轉人工）
        suggested, stage = pms_review.resolve_stage(scores, self.clf.classes_, title)
        return {
            "fileId": file_id,
            "status": "review_required",
            "suggestedClass": suggested,
            "modelScore": round(float(scores[top[0]]), 6),
            "alternatives": [
                {"class": str(self.clf.classes_[i]), "modelScore": round(float(scores[i]), 6)} for i in top
            ],
            "modelVersion": self.version,
            "modelSha256": self.model_sha256,
            "catalogVersion": self.catalog_version,
            "encoderVersion": self.encoder_version,
            "workItemPhotos": len(rows),
            # 分流提示（OOF 校準，未經 G2）：低於門檻建議優先人工看；status 一律 review_required
            "lowConfidence": bool(scores[top[0]] < REVIEW_CONFIDENCE or stage == "manual"),
            "stageDecision": stage,
            # 缺失旗標與工種正交（v17）；這裡只回標題弱標籤，圖像分數與人工修改在工作台
            "defectTitle": pms_review.defect_title(title),
        }
