"""去識別化判準包匯出器——引擎 B（賣「經過營建專家審核的判斷」）的最小可賣單位。

    uv run src/export_label_pack.py --version p1              # 內部用
    uv run src/export_label_pack.py --version p1 --licensed    # 資料授權已簽，可外送

白名單在 `docs/DATA-BOUNDARY.md` 的 `boundary-spec` 區塊，**不在這支程式裡**：
本檔只負責執行它、驗證它、把它的 sha256 釘進包裡。文件不在 → 拒跑（文件即白名單，
沒白名單不給匯）。

包裡有什麼：人審裁決對照（人把規則改成什麼）、人框的缺失框、QS 判準結構、工種類別名單。
包裡沒有什麼：照片、`fileId`、案場名、人名、自由文字、QS／合約原文。理由逐條寫在那份文件。

照片在客戶內網的承諾與「賣判準」必須切開，所以 `fileId` 也不給——改成加鹽雜湊
`sampleId`：**包內可 join，包外不可回推**。鹽與對照表寫在 `<pack>.local.json`
（0600，不進包、不進 git），只有我們自己追溯得回去。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from core import paths
from core.defects import BUTTON_ONLY, DEFECT_PATTERNS, load_defects
from core.labeler import Labeler, load_boxes, load_reviews

BOUNDARY_DOC = paths.ROOT / "docs" / "DATA-BOUNDARY.md"
PACK_SCHEMA = 1

_RE_HASH16 = re.compile(r"^[0-9a-f]{16}$")
_RE_BUCKET = re.compile(r"^site-\d{3}$")
_RE_YEAR_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_RE_QS_CODE = re.compile(r"^QS\d{4}(-[0-9]+(\.[0-9]+)*)?$")
_RE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_RE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# 身分／結構欄位不許空白；描述欄位可以空（規則沒意見、沒選 QS 項都是正常狀態）。
_NON_EMPTY_TYPES = {"hash16", "bucket", "yearMonth", "box1000", "flag"}


def load_boundary(doc: Path | None = None) -> dict:
    """讀 docs/DATA-BOUNDARY.md 的 boundary-spec。回傳 spec dict，另附 `_sha256`。"""
    doc = doc or BOUNDARY_DOC
    if not doc.exists():
        raise FileNotFoundError(f"找不到資料邊界文件：{doc}。白名單在文件裡，沒有它不准匯出判準包。")
    text = doc.read_text(encoding="utf-8")
    head, _, tail = text.partition("## boundary-spec")
    if not tail:
        raise ValueError(f"{doc} 缺少 `## boundary-spec` 區塊。")
    block = re.search(r"```yaml\n(.*?)```", tail, re.S)
    if not block:
        raise ValueError(f"{doc} 的 boundary-spec 區塊裡沒有 yaml。")
    spec = yaml.safe_load(block.group(1)) or {}
    if spec.get("schemaVersion") != PACK_SCHEMA:
        raise ValueError(f"邊界文件 schemaVersion={spec.get('schemaVersion')}，程式只懂 {PACK_SCHEMA}。")
    if not spec.get("tables"):
        raise ValueError("邊界文件沒有宣告任何 table。")
    forbidden = set(spec.get("forbiddenColumns") or [])
    for name, table in spec["tables"].items():
        cols = table.get("columns") or {}
        if not cols:
            raise ValueError(f"table {name} 沒有欄位。")
        if bad := forbidden & set(cols):
            raise ValueError(f"table {name} 列了禁用欄位：{sorted(bad)}")
    spec["_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    spec["_doc"] = str(doc.relative_to(paths.ROOT)) if doc.is_relative_to(paths.ROOT) else doc.name
    del head
    return spec


def taxonomy(lab: Labeler | None = None) -> set[str]:
    """可外售的類別集合＝labels.yaml 的類別 ∪ fallback ∪ 已核准新類。"""
    from core.pms_store import approved_classes

    lab = lab or Labeler.load()
    out = {label for _, label in lab.rules} | {lab.fallback}
    try:
        out |= set(approved_classes())
    except Exception:  # sqlite 不在（乾淨環境）不該讓匯出掛掉
        pass
    return {c for c in out if c}


def _valid(value, kind: str, allowed: dict) -> bool:
    if value == "" or value is None:
        return kind not in _NON_EMPTY_TYPES
    text = str(value)
    if kind == "hash16":
        return bool(_RE_HASH16.match(text))
    if kind == "bucket":
        return bool(_RE_BUCKET.match(text))
    if kind == "yearMonth":
        return bool(_RE_YEAR_MONTH.match(text))
    if kind == "qsCode":
        return bool(_RE_QS_CODE.match(text))
    if kind == "token":
        return bool(_RE_TOKEN.match(text))
    if kind == "flag":
        return text in {"0", "1"}
    if kind == "taxonomy":
        return text in allowed["taxonomy"]
    if kind == "defectPattern":
        return text in allowed["defectPattern"]
    if kind == "box1000":
        try:
            box = json.loads(text)
        except Exception:
            return False
        return (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(c, int) and 0 <= c <= 1000 for c in box)
        )
    return False  # 未知型別＝不放行（白名單制）


def validate_rows(table: str, rows: list[dict], spec: dict, allowed: dict) -> None:
    """型別即防線：任何一格不符就整份拒絕匯出，不靜默清洗。"""
    cols = spec["tables"][table]["columns"]
    forbidden = set(spec.get("forbiddenColumns") or [])
    for n, row in enumerate(rows):
        if extra := set(row) - set(cols):
            raise ValueError(f"{table} 第 {n} 列有白名單外的欄位：{sorted(extra)}")
        if leak := set(row) & forbidden:
            raise ValueError(f"{table} 第 {n} 列出現禁用欄位：{sorted(leak)}")
        for col, kind in cols.items():
            if not _valid(row.get(col, ""), kind, allowed):
                # 刻意不印值：值本身可能就是不該出現的公司資料
                raise ValueError(f"{table} 第 {n} 列的 {col} 不符型別 {kind}，匯出中止。")


def _sample_id(file_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{file_id}".encode()).hexdigest()[:16]


def _clean(value) -> str:
    """CSV 的空格會被 pandas 讀成 NaN，`str(nan)` 是 `"nan"`——會混成假值，先清掉。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _manifest_context() -> pd.DataFrame:
    """fileId → (constrId, reportDate, title)。title 只用來算我們自己的規則類別，不外流。"""
    if not paths.MANIFEST.exists():
        return pd.DataFrame(columns=["fileId", "constrId", "reportDate", "title"])
    df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    for col in ("fileId", "constrId", "reportDate", "title"):
        if col not in df.columns:
            df[col] = ""
    return df[["fileId", "constrId", "reportDate", "title"]].drop_duplicates("fileId", keep="last")


def build_judgements(salt: str, buckets: dict[str, str], lab: Labeler, allowed: dict) -> tuple:
    """人審裁決對照（只收有人審的照片）。回傳 (rows, stats)。"""
    reviews = load_reviews()
    boxed = set(load_boxes())
    ctx = _manifest_context().set_index("fileId")
    rows, skipped_no_manifest, skipped_orphan = [], 0, 0
    for file_id, human in sorted(reviews.items()):
        if file_id not in ctx.index:
            skipped_no_manifest += 1  # 非 PMS 日報源（legacy/QMS）v1 不收
            continue
        if human not in allowed["taxonomy"]:
            skipped_orphan += 1  # 類別改名留下的孤兒裁決，見 `make queue`
            continue
        meta = ctx.loc[file_id]
        constr = str(meta.constrId or "")
        date = str(meta.reportDate or "")
        if constr not in buckets or not _RE_YEAR_MONTH.match(date[:7]):
            skipped_no_manifest += 1
            continue
        rule = lab.label(meta.title) if isinstance(meta.title, str) else None
        rule = rule if rule in allowed["taxonomy"] else ""
        rows.append(
            {
                "sampleId": _sample_id(file_id, salt),
                "siteBucket": buckets[constr],
                "yearMonth": date[:7],
                "humanClass": human,
                "ruleClass": rule,
                "humanOverrode": "1" if rule and rule != human else "0",
                "hasBox": "1" if file_id in boxed else "0",
            }
        )
    return rows, {"skippedNoManifest": skipped_no_manifest, "skippedOrphanClass": skipped_orphan}


def build_defect_boxes(salt: str, allowed: dict) -> tuple:
    """人框的缺失框（defects.csv 全是 HUMAN 層，AI_GUESS 不進那張表）。"""
    df = load_defects()
    rows, skipped = [], 0
    for _, r in df.iterrows():
        raw_box = _clean(r.box)
        if not raw_box:
            continue  # 照片層級的缺陷回報沒有框，不進這張表
        try:
            boxes = json.loads(raw_box)
        except Exception:
            skipped += 1
            continue
        pattern = _clean(r.defectType)
        if pattern not in allowed["defectPattern"]:
            skipped += 1  # 不認得的樣態一律不外售（它可能根本是別處貼進來的文字）
            continue
        source = _clean(r.source).lower()
        for one in boxes if isinstance(boxes, list) else []:
            if len(one) != 4:
                skipped += 1
                continue
            rows.append(
                {
                    "sampleId": _sample_id(_clean(r.fileId), salt),
                    "defectPattern": pattern,
                    "qsCode": _clean(r.qsCode),
                    "box": json.dumps([int(c) for c in one]),
                    "source": source if _RE_TOKEN.match(source) else "",
                }
            )
    return rows, {"skippedBoxes": skipped}


def build_qs_criteria() -> tuple:
    """QS 判準的結構：條號 + A~E 分派 + O/R + 三個旗標。**絕不含 `name`（標準原文）。**"""
    from core import qs_data

    if not qs_data.RAW_DIR.exists():
        return [], {"qsDocs": 0}  # reference/ 是公司資料，不在就跳過（同 make test 紀律）
    # 一定要把 raw_dir 傳進去：`load(raw_dir=RAW_DIR)` 的預設值在 import 時就凍住了，
    # 不傳的話測試把 RAW_DIR 指到別處也沒用——它照樣去讀真的 reference/（踩過）。
    docs = qs_data.load(qs_data.RAW_DIR)  # {docNo: Doc}
    rows = []
    for doc in docs.values():
        for item in doc.items:
            if not _RE_QS_CODE.match(item.key):
                continue
            rows.append(
                {
                    "qsKey": item.key,
                    "kind": item.kind.lower(),
                    "status": item.status.lower(),
                    "isBilling": "1" if item.is_billing else "0",
                    "isPenalty": "1" if item.is_penalty else "0",
                    "isContract": "1" if item.is_contract else "0",
                }
            )
    return rows, {"qsDocs": len(docs)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_PACK_README = """# 判準包 {version}

營建缺失與工種判斷的**人審判準**資料集。

包含：人審裁決對照（規則判 A、懂現場的人改成 B）、人框的缺失框、QS 判準結構、工種類別名單。
**不包含**：工地照片或任何影像、客戶案場與日報識別碼、人員姓名、自由文字、品質標準與合約原文。

照片識別碼一律是加鹽雜湊（`sampleId`）：同一包內可以把裁決與框 join 起來，包外無法回推來源照片。
資料邊界與每個欄位的理由見發行方的 `docs/DATA-BOUNDARY.md`（sha256 記在 `manifest.json`）。

散佈狀態：**{distribution}**
"""


def export_pack(
    version: str,
    out: Path | None = None,
    *,
    licensed: bool = False,
    boundary: Path | None = None,
    log=print,
) -> Path:
    if not _RE_VERSION.match(version):
        raise ValueError("判準包版本請用英數字、連字號或底線，最多 64 字。")
    spec = load_boundary(boundary)
    out = out or paths.ROOT / "data" / "exports" / "label-packs" / version
    if out.exists():
        raise FileExistsError(f"判準包已存在，避免覆蓋：{out}")
    sidecar = out.with_name(out.name + ".local.json")

    lab = Labeler.load()
    allowed = {
        "taxonomy": taxonomy(lab),
        "defectPattern": set(DEFECT_PATTERNS) | set(BUTTON_ONLY),
    }
    # 鹽：sidecar 有就沿用（同版本重匯 sampleId 不變），沒有就新生一把。
    salt = ""
    if sidecar.exists():
        salt = str(json.loads(sidecar.read_text(encoding="utf-8")).get("salt") or "")
    salt = salt or secrets.token_hex(16)

    ctx = _manifest_context()
    sites = sorted({str(s) for s in ctx.constrId.tolist() if str(s).strip()})
    buckets = {site: f"site-{n:03d}" for n, site in enumerate(sites, start=1)}

    judgements, stats = build_judgements(salt, buckets, lab, allowed)
    boxes, box_stats = build_defect_boxes(salt, allowed)
    qs_rows, qs_stats = build_qs_criteria()
    tables = {"judgements": judgements, "defect_boxes": boxes, "qs_criteria": qs_rows}
    for name, rows in tables.items():
        validate_rows(name, rows, spec, allowed)
    if not judgements:
        raise ValueError("沒有任何人審裁決可以外售（判準包只收人審）。")
    # 包的賣點是「人改了機器什麼」。這三個數字就是它的厚度，直接寫進 manifest 讓買方與我們
    # 自己都看得到——overridden 太少的包，敘事上不要吹成「專家判準資料集」。
    breakdown = {
        "ruleSilent": sum(1 for r in judgements if not r["ruleClass"]),
        "agree": sum(1 for r in judgements if r["ruleClass"] and r["humanOverrode"] == "0"),
        "overridden": sum(1 for r in judgements if r["humanOverrode"] == "1"),
    }

    temp = out.with_name(out.name + ".building")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        files = []
        for name, rows in tables.items():
            cols = list(spec["tables"][name]["columns"])
            path = temp / f"{name}.csv"
            pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
            files.append({"name": path.name, "rows": len(rows), "sha256": _sha256(path)})
        tax_path = temp / "taxonomy.json"
        tax_path.write_text(
            json.dumps(
                {
                    # 類別名單可外售，規則 pattern 不可（裡面是日報用語語料）
                    "labelsVersion": yaml.safe_load(paths.LABELS_YAML.read_text(encoding="utf-8")).get(
                        "version"
                    ),
                    "tradeClasses": sorted(allowed["taxonomy"]),
                    "defectPatterns": list(DEFECT_PATTERNS),
                    "buttonOnlyPatterns": sorted(BUTTON_ONLY),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        files.append({"name": tax_path.name, "rows": None, "sha256": _sha256(tax_path)})
        distribution = "licensed-external" if licensed else "internal-only"
        readme = temp / "README.md"
        readme.write_text(_PACK_README.format(version=version, distribution=distribution), encoding="utf-8")
        files.append({"name": readme.name, "rows": None, "sha256": _sha256(readme)})
        manifest = {
            "schemaVersion": PACK_SCHEMA,
            "packVersion": version,
            "createdAt": datetime.now(UTC).isoformat(timespec="seconds"),
            "distribution": distribution,
            "boundaryDoc": {"path": spec["_doc"], "sha256": spec["_sha256"]},
            "contains": {"photos": False, "fileIds": False, "freeText": False, "standardText": False},
            "counts": {name: len(rows) for name, rows in tables.items()},
            "judgementBreakdown": breakdown,
            "stats": {**stats, **box_stats, **qs_stats, "siteBuckets": len(buckets)},
            "files": files,
        }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp.rename(out)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    # sidecar：鹽與回推對照表。不進包、不進 git，0600。
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "packVersion": version,
                "warning": "鹽與對照表：外流等於把去識別化作廢。不得與判準包一起交付。",
                "salt": salt,
                "siteBuckets": {bucket: site for site, bucket in buckets.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(sidecar, 0o600)

    log(f"判準包 {version} → {out}（{distribution}）")
    log(
        f"  人審裁決 {len(judgements)} 列："
        f"人改機器 {breakdown['overridden']} · 人機一致 {breakdown['agree']} · 規則沒意見 {breakdown['ruleSilent']}"
    )
    log(f"  缺失框 {len(boxes)} 列 · QS 判準 {len(qs_rows)} 列 · 案場代號 {len(buckets)} 個")
    if stats["skippedOrphanClass"]:
        log(f"  ⚠ {stats['skippedOrphanClass']} 筆裁決指向已不存在的類別，未外售（`make queue` 會列出）")
    if stats["skippedNoManifest"]:
        log(f"  · {stats['skippedNoManifest']} 筆非 PMS 日報源或缺日期，v1 不收")
    log(f"  回推對照表（不可外流）：{sidecar}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="匯出去識別化判準包（白名單＝docs/DATA-BOUNDARY.md）")
    ap.add_argument("--version", required=True, help="包版本名，如 p1")
    ap.add_argument("--out", type=Path, help="輸出目錄（預設 data/exports/label-packs/<version>）")
    ap.add_argument(
        "--licensed",
        action="store_true",
        help="資料授權條款已簽核，包可外送；未給則標成 internal-only",
    )
    args = ap.parse_args()
    export_pack(args.version, args.out, licensed=args.licensed)
