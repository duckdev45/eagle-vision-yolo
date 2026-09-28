# Eagle Vision 獨立 API

PMS Next.js 前端只呼叫 PMS NestJS。NestJS 驗證使用者、從 S3 取已上傳原圖，將原圖 bytes 傳給本服務；本服務不接觸 PMS 登入憑證、S3 憑證或正式日報資料庫。

```text
現場使用者 → PMS Next.js → PMS NestJS → Eagle Vision FastAPI
                    │               ├─ /v1/predict：照片原圖 → 單張建議
                    │               ├─ /v1/predict-batch：同工項多張 → 每張＋工項融合建議
                    └─ 回饋/稽核/outbox └─ /v1/feedback：冪等收件 → 人工複核 → 後續訓練
```

## 固定模型包

在 Eagle Vision 訓練環境匯出目前已完成的 PMS-only 版本（2026-09-28 起 v43，so400m 編碼器）：

```bash
uv run --extra train python src/export_service_bundle.py --version v43
```

輸出 `models/service/v43/{metadata.json,classifier.npz,encoder.safetensors}`（約 1.6 GB；metadata 另含 `encoderKey`、`reviewConfidence`、`stageRule`）。裡面只有固定視覺編碼器、分類器係數、類別與版本；不含照片、manifest 或 train/test fileId。`models/` 已被 git 忽略，**公開 GitHub repo 不存此包**。部署時從受控的私有位置把同一包放入 build context，Docker build 將其烘進 image；用 image digest 與 `metadata.json` 的 SHA256 對版本。模型服務啟動時不從網路下載權重。

## 本機啟動

```bash
cd service
uv sync --group dev
export VISION_BUNDLE_DIR=../models/service/v43
export VISION_SERVICE_TOKEN='replace-with-a-random-32-character-or-longer-token'
export VISION_FEEDBACK_DB=../models/service/feedback.sqlite
uv run uvicorn vision_api.app:app --host 127.0.0.1 --port 8000
```

另開 terminal：

```bash
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/v1/predict \
  -H "X-Service-Token: $VISION_SERVICE_TOKEN" \
  -H 'X-File-Id: PMS-fileId' \
  -H 'X-Source-Policy: clean-raw-v1' \
  -H 'Content-Type: image/jpeg' \
  --data-binary @photo.jpg
```

回 `predictionId`、前三候選、未校準 `modelScore`、`modelVersion`、模型／輸入 hash、分類表與編碼器版本。第一版只收 2026-08-14 後「膠囊未烘進像素」的乾淨原圖；不支援舊圖直接重判。

`POST /v1/predict-batch` 一次收同一施作項目（同日報 × 同標題）1–8 張原圖（JSON，`dataBase64`），回每張單張結果、`fused`（參考兄弟照後的建議）、整個工項的 `workItem`、`lowConfidence` 與打底／粉光的 `stageDecision`；可選填日報 `title` 供階段判斷。欄位與限制見 `docs/PMS-FEEDBACK-API.md`「工項批次推論與缺失旗標」。現場顯示建議請用 `fused`；`status` 一律 `review_required`。

`POST /v1/feedback` 由 PMS 後端轉送回饋事件，以 `eventId` 冪等收件、寫入 SQLite 持久化。API 會核對已持久化的 `predictionId`、照片 hash、模型版本與分類表，並驗證修正類別是否存在於預測當時的分類表；因此換模型後，舊預測的回饋仍可入庫。此收件匣只標記 `pending`，不能把現場「正確」直接當真值或即時回訓。第一版採單一服務實例，部署需掛持久 volume `/data`；若要多實例與完整複核流程，改用共用資料庫及 PMS outbox。

回饋 JSON 必填 `eventId`、`predictionId`、`fileId`、`imageSha256`、`modelVersion`、`modelSha256`、`catalogVersion`、`action`。`action` 可為 `accept`、`correct`、`new_candidate`、`uncertain`、`out_of_scope`；`correct` 另填現有類別 `selectedClass`，`new_candidate` 另填 `candidateLabel`。PMS 必須保存預測與人員操作紀錄，並由具權限的專家複核後才匯入訓練資料。

## Docker

在 repo 根目錄、已匯出模型包後：

```bash
docker build -f service/Dockerfile -t eagle-vision-api:v43 .
docker run --rm -p 127.0.0.1:8000:8000 \
  -e VISION_SERVICE_TOKEN="$VISION_SERVICE_TOKEN" \
  -v eagle-vision-feedback:/data \
  eagle-vision-api:v43
```

正式部署不要發佈 host port；讓 PMS NestJS 與此 container 位於同一私有網路，或由 NestJS 經私有 HTTPS 呼叫另一台主機。NestJS 傳原圖 bytes 與服務間 token，瀏覽器不直連模型 API。Linux container 沒有 macOS MPS，正式主機要另測 CPU／CUDA 延遲與記憶體。

### 資源與延遲（v43 so400m，2026-09-28 實測）

本機 Apple Silicon 上的 OrbStack Linux container（linux/arm64、12 vCPU、CPU 推論，無 CUDA／MPS）：

| 項目 | 數值 |
|---|---|
| image 大小 | 12.9 GB（`uv sync` 在 linux 會裝 torch 的 CUDA 相依約 11 GB；模型包 1.7 GB） |
| 記憶體峰值（cgroup `memory.peak`，含載入權重時的檔案快取） | 7.1 GB；穩定後約 4.5 GB |
| `/v1/predict` 單張 | 中位數 2.1 s（p90 2.2 s，n=10） |
| `/v1/predict-batch` 同工項 2 張 | 中位數 3.9 s（n=5） |

同一台 Mac 直接跑（非 container）so400m 為 CPU 約 0.4–0.65 s／張、MPS 0.16 s／張，container 內明顯較慢。
**x86 正式主機的延遲與記憶體需在正式主機實測**，量測方式（repo 根目錄，模型包已匯出）：

```bash
docker build -f service/Dockerfile -t eagle-vision-api:v43 .
docker run -d --name ev-bench -e VISION_SERVICE_TOKEN="$VISION_SERVICE_TOKEN" \
  -p 127.0.0.1:18000:8000 --tmpfs /data:uid=10001 eagle-vision-api:v43
python3 service/bench.py http://127.0.0.1:18000 "$VISION_SERVICE_TOKEN" photo1.jpg photo2.jpg 10   # 單張＋2 張批次
docker exec ev-bench cat /sys/fs/cgroup/memory.peak
```

（`service/bench.py` 只用標準庫：對 `/v1/predict` 送 N 次、對 `/v1/predict-batch` 送 N/2 次，印中位數與 p90。）
容器記憶體上限建議至少 8 GB。只需 CPU 推論時，可改用 CPU 版 torch wheel 縮小 image（未實作）。

## 驗證與發布

```bash
cd service
uv run --group dev pytest -q tests
uv run --group dev ruff check .
```

公開 GitHub 僅存程式與建置設定；CI 若要建立正式 image，必須先從私有 artifact storage 取得 `models/service/v43`，再依 `metadata.json` 驗證 SHA256。GitHub repo 本身不會執行 API，也不能用公開 Actions artifact 發布這份模型包。
