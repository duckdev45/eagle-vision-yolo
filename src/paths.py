"""所有路徑集中一處，其餘模組不要自己拼字串。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"
PHOTOS = RAW / "photos"  # 不可變層：{fileId}.{ext}
REPORTS_JSON = RAW / "reports"  # 不可變層：日報原始 JSON 快照
MANIFEST = RAW / "manifest.csv"
INDEX = RAW / "report_index.csv"  # 同步狀態（version 水位線）

# 人工複核結果：{fileId, cls, note, reviewedAt}。這是**唯一**照片層級的真標籤來源
# ——title 標的是整個工項，一個工項的兩張照片常常在拍不同階段。人手打的，不可重算，
# 所以放 data/ 根層跟 raw 同級，不進 derived（重跑 pipeline 不會把它洗掉）。
REVIEW = ROOT / "data" / "review.csv"

DERIVED = ROOT / "data" / "derived"
IMAGES = DERIVED / "images"  # 遮蔽後 jpg
FEATURES = DERIVED / "features"
SPLITS = DERIVED / "splits"
TREE = DERIVED / "tree"  # 有結構的照片瀏覽樹（symlink）

# --- 缺失軸資料契約（docs/specs/reject-qs-code.md §2）------------------------
# 與 review.csv 同級不進 derived：append-only 的人寫層，重跑管線不會洗掉。
# review.csv 是工種 cls 裁決，defects.csv 是缺失回報＋CVAT 框——正交維度分開放。
DEFECTS = ROOT / "data" / "defects.csv"
GOLDEN = ROOT / "data" / "golden"  # G1 黃金集：manifest + 兩位標註者的裁決
CVAT = ROOT / "data" / "cvat"  # CVAT 進出：task manifest（匯出）與 XML（匯入暫存）

# --- QMS 稽核照（另一個系統、另一套標籤，刻意不與日報混在同一棵樹）---------
# --- 舊版 pptx 進度報告（第三個資料源）------------------------------------
# 與日報同分佈（同一批工地主任、同一種構圖），但**不是**同一個系統產的，
# 所以分開放：來源要看得出來，混在 raw/photos 裡就再也分不清哪張是哪來的。
LEGACY = ROOT / "data" / "legacy"
LEGACY_PHOTOS = LEGACY / "raw" / "photos"  # {sha1}.{ext}，檔名就是內容雜湊
LEGACY_MANIFEST = LEGACY / "raw" / "manifest.csv"
LEGACY_IMAGES = LEGACY / "derived" / "images"

QMS = ROOT / "data" / "qms"
QMS_PHOTOS = QMS / "raw" / "photos"  # {fileId}.jpg，浮水印烤死在畫面上
QMS_MANIFEST = QMS / "raw" / "manifest.csv"
QMS_CELLS = QMS / "raw" / "cells.csv"  # 母體清單（抽樣前的全部格子）
QMS_IMAGES = QMS / "derived" / "images"
QMS_TREE = QMS / "derived" / "tree"

# --- Field Reports 缺失照片語料（樂氧森 2026-08-28 快照，2026-09-01 進站）------
# 第四資料源，與日報/QMS/LEGACY sha1 零重疊。單日單場 → 只當煙霧測試＋標註練兵，
# 不可做 train/test 切分（詳 data/field_reports/README.md）。
FIELD_REPORTS = ROOT / "data" / "field_reports"
FR_PHOTOS = FIELD_REPORTS / "raw" / "photos"  # {photoId}.webp（1600px 衍生層，非 raw）
FR_MANIFEST = FIELD_REPORTS / "raw" / "manifest.csv"  # site=樂氧森（原值留在 siteRaw）
FR_GDINO = FIELD_REPORTS / "derived" / "gdino"  # GDINO pre-annotations（AI_GUESS）
FR_VLM = FIELD_REPORTS / "derived" / "vlm"  # gemma4:e4b 缺失判定（AI_GUESS）

VLM = DERIVED / "vlm"  # gemma4:e4b 判定輸出（尺入鏡 PoC 等，AI_GUESS 層）

REPORTS_OUT = ROOT / "reports"
MODELS = ROOT / "models"
LABELS_YAML = ROOT / "labels.yaml"


def ensure_dirs() -> None:
    for p in (
        PHOTOS,
        REPORTS_JSON,
        IMAGES,
        FEATURES,
        SPLITS,
        TREE,
        REPORTS_OUT,
        MODELS,
        QMS_PHOTOS,
        QMS_IMAGES,
        QMS_TREE,
        LEGACY_PHOTOS,
        LEGACY_IMAGES,
        FR_GDINO,
    ):
        p.mkdir(parents=True, exist_ok=True)
