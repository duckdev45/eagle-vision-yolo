"""私有 Eagle Vision API：由 PMS NestJS 後端呼叫，瀏覽器不直連。"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .feedback import FeedbackConflict, FeedbackStore
from .model import ModelInputError, VisionModel

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_BATCH_PHOTOS = 8  # 一個工項實測幾乎都 2 張，最多 6 張
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


class BatchPhoto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fileId: str = Field(min_length=1, max_length=128)
    contentType: Literal["image/jpeg", "image/png", "image/webp"]
    dataBase64: str = Field(min_length=1)


class WorkItemBatch(BaseModel):
    """同一施作項目（同日報 × 同標題）的照片，一次送進來做工項融合。"""

    model_config = ConfigDict(extra="forbid")

    workItemId: str = Field(min_length=1, max_length=128)
    sourcePolicy: Literal["clean-raw-v1"]
    title: str | None = Field(default=None, max_length=500)
    photos: list[BatchPhoto] = Field(min_length=1, max_length=MAX_BATCH_PHOTOS)


class FeedbackEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eventId: str = Field(min_length=1, max_length=128)
    predictionId: str = Field(min_length=1, max_length=128)
    fileId: str = Field(min_length=1, max_length=128)
    imageSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    modelVersion: str = Field(min_length=1, max_length=64)
    modelSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalogVersion: str = Field(min_length=1, max_length=128)
    action: Literal["accept", "correct", "new_candidate", "uncertain", "out_of_scope"]
    selectedClass: str | None = Field(default=None, max_length=128)
    candidateLabel: str | None = Field(default=None, max_length=128)
    note: str | None = Field(default=None, max_length=2000)
    reportId: str | None = None
    reportVersion: int | None = None


def create_app(model: VisionModel | None = None, feedback: FeedbackStore | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        token = os.getenv("VISION_SERVICE_TOKEN", "")
        if len(token) < 32:
            raise RuntimeError("VISION_SERVICE_TOKEN 至少需 32 字元。")
        app.state.service_token = token
        app.state.model = model or VisionModel(Path(os.environ["VISION_BUNDLE_DIR"]))
        app.state.feedback = feedback or FeedbackStore(
            Path(os.getenv("VISION_FEEDBACK_DB", "/data/feedback.sqlite"))
        )
        yield

    app = FastAPI(title="Eagle Vision PMS API", version="1.0", lifespan=lifespan)

    def require_token(request: Request, x_service_token: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_service_token, request.app.state.service_token):
            raise HTTPException(401, "Unauthorized")

    @app.get("/healthz")
    def health(request: Request) -> dict:
        return {"status": "ready", "modelVersion": request.app.state.model.meta["modelVersion"]}

    @app.post("/v1/predict", dependencies=[Depends(require_token)])
    async def predict(
        request: Request,
        x_file_id: str = Header(),
        x_source_policy: str = Header(),
    ) -> dict:
        if request.headers.get("content-type", "").split(";")[0] not in IMAGE_TYPES:
            raise HTTPException(415, "Unsupported image type")
        if x_source_policy != "clean-raw-v1":
            raise HTTPException(422, "Only clean-raw-v1 is supported")
        chunks = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise HTTPException(413, "Image exceeds 5 MB")
            chunks.append(chunk)
        try:
            prediction = request.app.state.model.predict(x_file_id, b"".join(chunks))
            request.app.state.feedback.record_prediction(prediction, request.app.state.model.classes)
            return prediction
        except ModelInputError as exc:
            raise HTTPException(422, {"code": exc.code, "detail": str(exc)}) from exc
        except FeedbackConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/v1/predict-batch", dependencies=[Depends(require_token)])
    def predict_batch(batch: WorkItemBatch, request: Request) -> dict:
        photos = []
        for photo in batch.photos:
            try:
                blob = base64.b64decode(photo.dataBase64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(422, {"code": "invalid_base64", "fileId": photo.fileId}) from exc
            if len(blob) > MAX_IMAGE_BYTES:
                raise HTTPException(413, f"Image {photo.fileId} exceeds 5 MB")
            photos.append((photo.fileId, blob))
        model = request.app.state.model
        try:
            result = model.predict_work_item(photos, batch.title)
            for prediction in result["photos"]:
                request.app.state.feedback.record_prediction(prediction, model.classes)
        except ModelInputError as exc:
            raise HTTPException(422, {"code": exc.code, "detail": str(exc)}) from exc
        except FeedbackConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"workItemId": batch.workItemId, **result}

    @app.post("/v1/feedback", dependencies=[Depends(require_token)])
    def receive_feedback(event: FeedbackEvent, request: Request) -> dict:
        if event.action == "correct" and not event.selectedClass:
            raise HTTPException(422, "correct requires selectedClass")
        if event.action != "correct" and event.selectedClass:
            raise HTTPException(422, "selectedClass is only valid for correct")
        if event.action == "new_candidate" and not event.candidateLabel:
            raise HTTPException(422, "new_candidate requires candidateLabel")
        if event.action != "new_candidate" and event.candidateLabel:
            raise HTTPException(422, "candidateLabel is only valid for new_candidate")
        try:
            inserted = request.app.state.feedback.receive(event.model_dump())
        except FeedbackConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"eventId": event.eventId, "status": "queued_for_review", "created": inserted}

    return app


app = create_app()
