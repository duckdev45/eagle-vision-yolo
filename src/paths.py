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

# --- 舊版 pptx 進度報告 ----------------------------------------------------
# 與日報同分佈但不同系統產的，分開放。2026-10-09 起不進訓練，只供 G1 黃金集／G2 考卷／CVAT。
# QMS 稽核照、Jev、VLM PoC、convnext 微調的資料已封存到 data/archive/（清單見其 README.md）。
LEGACY = ROOT / "data" / "legacy"
LEGACY_PHOTOS = LEGACY / "raw" / "photos"  # {sha1}.{ext}，檔名就是內容雜湊
LEGACY_MANIFEST = LEGACY / "raw" / "manifest.csv"
LEGACY_IMAGES = LEGACY / "derived" / "images"

# --- Field Reports 缺失照片語料（樂氧森 2026-08-28 快照，2026-09-01 進站）------
# 第四資料源，與日報/QMS/LEGACY sha1 零重疊。單日單場 → 只當煙霧測試＋標註練兵，
# 不可做 train/test 切分（詳 data/field_reports/README.md）。
FIELD_REPORTS = ROOT / "data" / "field_reports"
FR_PHOTOS = FIELD_REPORTS / "raw" / "photos"  # {photoId}.webp（1600px 衍生層，非 raw）
FR_MANIFEST = FIELD_REPORTS / "raw" / "manifest.csv"  # site=樂氧森（原值留在 siteRaw）
FR_GDINO = FIELD_REPORTS / "derived" / "gdino"  # GDINO pre-annotations（AI_GUESS；yolo_dataset 量測用）

# --- 每日分流（core/routing.py、src/daily.py）---------------------------------
# 分流結果是 derived（隨時可由規則＋模型重算）；抽查紀錄與每日紀錄是人看的歷史，
# 放 data/ 根層不進 derived，砍 derived 重跑也不會丟。
ROUTE = DERIVED / "route"  # latest.csv（每張照片在哪一桶）＋ latest.json（摘要）
ROUTE_HISTORY = ROOT / "data" / "route-history.jsonl"  # 每次分流的摘要（週報趨勢用）
ROUTE_AUDIT = ROOT / "data" / "route-audit.csv"  # 自動桶抽查樣本：量自動桶的真實準確率
QUEUE_EXPORT = ROOT / "data" / "exports" / "queue"  # 給未來標註平台的待審清單
DAILY_LOG = ROOT / "data" / "daily-log.jsonl"  # 每日編排的結果（同步／分流／重訓／切換）
PROMOTION_LOG = ROOT / "data" / "promotion-log.jsonl"  # 自動切換的每一次考卷與決定

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
        LEGACY_PHOTOS,
        LEGACY_IMAGES,
        FR_GDINO,
    ):
        p.mkdir(parents=True, exist_ok=True)
