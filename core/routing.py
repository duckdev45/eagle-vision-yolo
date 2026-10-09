"""每日分流：機器先收掉沒爭議的照片，人只看有爭議的那一小撮。

    uv run src/daily.py --only-route     # 只重算分流（不同步、不重訓）

三個訊號、三個桶：
    規則   labels.yaml 看標題給的類別（純標題，不套人工覆寫）
    模型   圖像分類器的類別與信心（工項融合＋打底／粉光分層，與工作台同一套）
    人工   review.csv（有就結案，不再分流）

    自動確認  規則＝模型 且 信心 ≥ AUTO_CONFIDENCE，且沒有任何「依定義要看圖」的旗標
    抽查      自動確認裡固定 5% 送人看——量自動桶的真實準確率，人改掉的就是判準包要賣的
    人工佇列  其餘：規則≠模型、標題沒規則、信心不足、階段待人工、缺失改善照、模型沒學過這類
    隔離      沒有原圖、沒有圖像特徵、日期不合理（manifest 裡真的有 2082～2094 年的）

🔴 **模型訊號不能來自背過這張照片的模型。** 目前 split 的 labels 裡，沒人審的照片本來就是用
規則當答案在訓練——拿那顆模型去看自己的訓練照，「規則＝模型」是背出來的，不是兩個訊號一致
（2026-10-09 總覽頁第一版的 90% 就是這樣灌水的）。所以：split 裡有標籤的照片一律用分組
out-of-fold（案場×日期整組切，兄弟照同折），只有 split 外的新照片才用上線模型直接看。

自動確認**不寫 review.csv**：自動桶的答案就是規則答案，而沒人審的照片本來就用規則答案訓練，
寫進去不會讓模型多學到什麼，只會讓「人審」這個欄位不再等於人審（判準包、考卷都會被污染）。
分流只決定「誰需要看」，不決定「答案是什麼」。
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from core import model_registry as registry
from core import paths, pms_review
from core import pms_store as store
from core.evaluation_metrics import fuse_work_items
from core.labeler import Labeler, load_reviews

AUTO_CONFIDENCE = 0.9  # 2026-10-09 定案：寧可多看一點，也不要自動桶錯太多
AUDIT_PERCENT = 5  # 自動桶固定抽查比例（依 fileId 雜湊決定，同一張永遠同一個結果）
OOF_FOLDS = 5
DEFECT_CLASS = "雜項-缺失改善"  # v17 起只收「看不出工種的缺改照」，依定義要看圖，不自動收
HELD_STATES = {"candidate", "uncertain", "excluded"}  # 工作台已有人處理過的暫緩狀態

BUCKETS = {
    "auto": "自動確認",
    "audit": "抽查",
    "queue": "人工佇列",
    "quarantine": "隔離",
    "held": "已暫緩",
    "done": "已人審",
}
# 佇列排序：越前面越該先看（規則與模型吵架的那些，人一眼就能裁、價值也最高）
REASONS = [
    "規則與模型不同",
    "標題沒有對應規則",
    "模型尚未涵蓋此類",
    "打底／粉光階段待人工",
    "缺失改善照：工種待看圖",
    "模型信心不足",
    "自動桶抽查",
]
COLUMNS = [
    "fileId",
    "bucket",
    "reason",
    "priority",
    "ruleClass",
    "modelClass",
    "modelConfidence",
    "signal",
    "stageSource",
    "reportDate",
]


def audit_pick(file_id: str) -> bool:
    """固定抽樣：同一張照片今天抽到，明天也抽到——抽查集合才不會隨每天重算漂移。"""
    return int(hashlib.sha256(file_id.encode()).hexdigest()[:8], 16) % 100 < AUDIT_PERCENT


def oof_proba(X, y, groups, classes: list[str], *, folds: int = OOF_FOLDS, C: float | None = None):
    """分組 out-of-fold 機率。每列的預測來自沒看過它（也沒看過同組兄弟）的模型。

    某一折的訓練集若缺某類，那一欄就是 0；整折只剩一類時整折留 0（呼叫端當「沒有預測」）。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    if C is None:
        C = registry.PROBE_C
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        raise ValueError("out-of-fold 至少需要兩組（案場×日期）。")
    col = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(y), len(classes)))
    for tr, te in GroupKFold(n_splits=min(folds, n_groups)).split(X, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced").fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        for j, c in enumerate(clf.classes_):
            out[te, col[str(c)]] = p[:, j]
    return out


def _resolve(proba, classes: list[str], titles: list[str]) -> list[tuple[str, float, str]]:
    """(類別, 信心, 決定來源)。打底／粉光由標題定階段時，信心用「泥作打底粉光群」的總機率——
    圖像只負責判群，階段交給標題（與工作台同一條規則，見 pms_review.resolve_stage）。"""
    out = []
    for p, title in zip(proba, titles):
        if not p.any():
            out.append(("", 0.0, ""))
            continue
        pred, source = pms_review.resolve_stage(p, classes, title)
        conf = float(p.max())
        if source == "title":
            conf = float(sum(p[classes.index(c)] for c in pms_review.STAGE_CLASSES))
        out.append((pred, conf, source))
    return out


def model_signals(log=print) -> tuple[pd.DataFrame, dict]:
    """每張有特徵的施作照：模型類別、信心、訊號來源（oof｜live）。"""
    name = registry.current()
    meta = {"model": name, "modelClasses": []}
    empty = pd.DataFrame(columns=["fileId", "modelClass", "modelConfidence", "stageSource", "signal"])
    feature = registry.feature_path(registry.encoder(name))
    if not (registry.split_path(name).exists() and feature.exists() and registry.probe_path(name).exists()):
        log(f"⚠ {name} 缺 split／特徵／探針，模型訊號暫缺（全部照片會進隔離或佇列）")
        return empty, meta
    sp = registry.load_split(name)
    pool = pms_review.load_pool()
    titles = dict(zip(pool.fileId, pool.title))
    groups = {r.fileId: f"{r.constrId}|{r.reportDate}" for r in pool.itertuples()}
    keys = pms_review.work_item_keys(pool)
    with np.load(feature, allow_pickle=True) as z:
        ids = [str(f) for f in z["fileIds"]]
        emb = z["emb"]
    row = {f: i for i, f in enumerate(ids)}
    in_pool = [f for f in ids if f in titles]
    labeled = [f for f in in_pool if f in sp.get("labels", {})]
    unseen = [f for f in in_pool if f not in sp.get("labels", {})]
    rows = []

    if len(labeled) >= 2:
        y = [sp["labels"][f] for f in labeled]
        classes = sorted(set(y))
        proba = oof_proba(emb[[row[f] for f in labeled]], y, [groups[f] for f in labeled], classes)
        proba = fuse_work_items(proba, [keys.get(f, "") for f in labeled])
        for f, (pred, conf, src) in zip(labeled, _resolve(proba, classes, [titles[f] for f in labeled])):
            rows.append((f, pred, conf, src, "oof"))
        log(f"模型訊號：{len(labeled)} 張 split 內照片用 {OOF_FOLDS} 折分組 out-of-fold")

    clf = registry.load_probe(name)
    classes = [str(c) for c in clf.classes_]
    meta["modelClasses"] = classes
    if unseen:
        proba = fuse_work_items(
            clf.predict_proba(emb[[row[f] for f in unseen]]), [keys.get(f, "") for f in unseen]
        )
        for f, (pred, conf, src) in zip(unseen, _resolve(proba, classes, [titles[f] for f in unseen])):
            rows.append((f, pred, conf, src, "live"))
        log(f"模型訊號：{len(unseen)} 張新照片用上線模型 {name}")
    df = pd.DataFrame(rows, columns=["fileId", "modelClass", "modelConfidence", "stageSource", "signal"])
    return df, meta


def _bad_date(value: str, today: date) -> str:
    try:
        day = datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return "日報日期無法解析"
    return "日報日期在未來" if day > today + timedelta(days=1) else ""


def route_frame(df: pd.DataFrame, model_classes: set[str], today: date | None = None) -> pd.DataFrame:
    """純函式：每列 → bucket、reason、priority。欄位見 COLUMNS（加上 humanClass/reviewState/hasPhoto）。"""
    today = today or date.today()
    buckets, reasons, priorities = [], [], []
    for r in df.to_dict("records"):
        file_id = str(r["fileId"])
        rule, model = str(r.get("ruleClass") or ""), str(r.get("modelClass") or "")
        raw_conf = r.get("modelConfidence")
        conf = float(raw_conf) if isinstance(raw_conf, (int, float)) and raw_conf == raw_conf else 0.0
        bad = _bad_date(str(r.get("reportDate") or ""), today)
        why: list[str] = []
        if r.get("humanClass"):
            bucket = "done"
        elif r.get("reviewState") in HELD_STATES:
            bucket = "held"
        elif not r.get("hasPhoto") or not model or bad:
            bucket = "quarantine"
            why = ["沒有原圖" if not r.get("hasPhoto") else "尚無圖像特徵" if not model else bad]
        else:
            if not rule:
                why.append("標題沒有對應規則")
            elif rule not in model_classes:
                why.append("模型尚未涵蓋此類")
            elif rule != model:
                why.append("規則與模型不同")
            if r.get("stageSource") == "manual":
                why.append("打底／粉光階段待人工")
            if rule == DEFECT_CLASS:
                why.append("缺失改善照：工種待看圖")
            if conf < AUTO_CONFIDENCE and "規則與模型不同" not in why:
                why.append("模型信心不足")
            if why:
                bucket = "queue"
            elif audit_pick(file_id):
                bucket, why = "audit", ["自動桶抽查"]
            else:
                bucket = "auto"
        buckets.append(bucket)
        reasons.append("；".join(why))
        priorities.append(min((REASONS.index(w) for w in why if w in REASONS), default=len(REASONS)))
    out = df.copy()
    out["bucket"], out["reason"], out["priority"] = buckets, reasons, priorities
    return out


def build(today: date | None = None, log=print) -> dict:
    """算一次分流並落檔。回傳摘要（每日編排與總覽頁都讀它）。"""
    today = today or date.today()
    lab = Labeler.load()
    pool = pms_review.load_pool()
    human = load_reviews()
    decisions = store.active_decisions()
    signals, meta = model_signals(log)
    df = pool[["fileId", "title", "reportDate"]].merge(signals, on="fileId", how="left")
    df["ruleClass"] = [
        "" if (c := lab.label(t)) in (None, lab.fallback) else c for t in df.title.fillna("").astype(str)
    ]
    df["humanClass"] = [human.get(f, "") for f in df.fileId]
    df["reviewState"] = [decisions.get(f, {}).get("action", "") for f in df.fileId]
    df["hasPhoto"] = [pms_review.photo_path(f) is not None for f in df.fileId]
    for col in ("modelClass", "stageSource", "signal"):
        df[col] = df[col].fillna("")
    routed = route_frame(df, set(meta["modelClasses"]), today)

    paths.ROUTE.mkdir(parents=True, exist_ok=True)
    routed[COLUMNS].to_csv(paths.ROUTE / "latest.csv", index=False)
    counts = {k: int((routed.bucket == k).sum()) for k in BUCKETS}
    open_items = routed[routed.bucket.isin(["queue", "audit", "quarantine"])]
    summary = {
        "routedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": meta["model"],
        "autoConfidence": AUTO_CONFIDENCE,
        "auditPercent": AUDIT_PERCENT,
        "counts": counts,
        "openReasons": {
            w: int(open_items.reason.str.contains(w, regex=False).sum())
            for w in [*REASONS, "沒有原圖", "尚無圖像特徵", "日報日期在未來", "日報日期無法解析"]
            if open_items.reason.str.contains(w, regex=False).any()
        },
        "signals": routed.signal.value_counts().to_dict(),
        "auditPrecision": audit_precision(),
    }
    (paths.ROUTE / "latest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with paths.ROUTE_HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    _record_audit(routed[routed.bucket == "audit"], meta["model"])
    export_queue(routed, meta["model"])
    undecided = counts["queue"] + counts["audit"] + counts["quarantine"]
    log(
        f"分流（{meta['model']}，門檻 {AUTO_CONFIDENCE}）：自動確認 {counts['auto']}・抽查 {counts['audit']}"
        f"・人工佇列 {counts['queue']}・隔離 {counts['quarantine']}・已人審 {counts['done']}"
        f"・已暫緩 {counts['held']} → 人要看 {undecided} 張"
    )
    return summary


def _record_audit(rows: pd.DataFrame, model: str) -> None:
    """第一次進抽查桶時記下「當時自動桶會給的答案」。之後人審了，才算得出自動桶準不準。"""
    if rows.empty:
        return
    old = (
        pd.read_csv(paths.ROUTE_AUDIT, dtype=str, keep_default_na=False)
        if paths.ROUTE_AUDIT.exists()
        else pd.DataFrame(columns=["fileId"])
    )
    fresh = rows[~rows.fileId.isin(set(old.fileId))]
    if fresh.empty:
        return
    new = pd.DataFrame(
        {
            "fileId": fresh.fileId,
            "autoClass": fresh.ruleClass,
            "modelConfidence": fresh.modelConfidence.round(4),
            "model": model,
            "sampledAt": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    new.to_csv(paths.ROUTE_AUDIT, mode="a", header=not paths.ROUTE_AUDIT.exists(), index=False)


def audit_precision() -> dict:
    """抽查樣本裡已有人審的：人同意自動答案的比例。這是自動桶唯一可信的準確率。"""
    if not paths.ROUTE_AUDIT.exists():
        return {"sampled": 0, "reviewed": 0, "agree": 0, "precision": None}
    sample = pd.read_csv(paths.ROUTE_AUDIT, dtype=str, keep_default_na=False)
    human = load_reviews()
    done = sample[sample.fileId.isin(set(human))]
    agree = int(sum(human[f] == c for f, c in zip(done.fileId, done.autoClass)))
    return {
        "sampled": len(sample),
        "reviewed": len(done),
        "agree": agree,
        "precision": round(agree / len(done), 4) if len(done) else None,
    }


def latest() -> tuple[pd.DataFrame, dict]:
    path = paths.ROUTE / "latest.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS), {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["priority"] = pd.to_numeric(df.priority, errors="coerce").fillna(len(REASONS)).astype(int)
    df["modelConfidence"] = pd.to_numeric(df.modelConfidence, errors="coerce")
    meta_path = paths.ROUTE / "latest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return df, meta


def queue_items() -> pd.DataFrame:
    """收件匣：最近一次分流的佇列＋抽查，扣掉之後已經有人處理的（人審或暫緩），依優先序排。"""
    df, _ = latest()
    if df.empty:
        return df
    human = load_reviews()
    held = {f for f, d in store.active_decisions().items() if d.get("action") in HELD_STATES}
    df = df[df.bucket.isin(["queue", "audit"]) & ~df.fileId.isin(set(human) | held)]
    return df.sort_values(["priority", "reportDate"], ascending=[True, False]).reset_index(drop=True)


def resolve(
    file_id: str, *, reviewer: str, label: str = "", action: str = "classified", reason: str = ""
) -> None:
    """收件匣與未來標註平台寫回人審結果的唯一入口——走工作台同一條 decide()，不另開寫入路徑。"""
    pms_review.decide(
        file_id,
        action,
        reviewer=reviewer,
        label=label,
        reason=reason or ("inbox" if action == "classified" else ""),
    )


def export_queue(routed: pd.DataFrame, model: str) -> None:
    """給未來標註平台的待審清單（內部用，含 fileId 與標題；契約見 docs/QUEUE-CONTRACT.md）。"""
    items = routed[routed.bucket.isin(["queue", "audit"])].sort_values(
        ["priority", "reportDate"], ascending=[True, False]
    )
    paths.QUEUE_EXPORT.mkdir(parents=True, exist_ok=True)

    def _conf(value) -> float | None:
        return round(float(value), 4) if isinstance(value, (int, float)) and value == value else None

    payload = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model,
        "autoConfidence": AUTO_CONFIDENCE,
        "catalog": sorted(pms_review.catalog()),
        "items": [
            {
                "fileId": str(r["fileId"]),
                "bucket": str(r["bucket"]),
                "reasons": [w for w in str(r["reason"]).split("；") if w],
                "ruleClass": str(r["ruleClass"] or ""),
                "modelClass": str(r["modelClass"] or ""),
                "modelConfidence": _conf(r["modelConfidence"]),
                "title": str(r.get("title") or ""),
                "reportDate": str(r["reportDate"] or ""),
            }
            for r in items.to_dict("records")
        ],
    }
    (paths.QUEUE_EXPORT / "queue.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def recent_runs(limit: int = 7) -> dict:
    """總覽頁用：最近幾次每日編排與自動切換的紀錄（新的在前）。"""

    def tail(path, n):
        if not path.exists():
            return []
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [json.loads(ln) for ln in reversed(lines[-n:])]

    return {"daily": tail(paths.DAILY_LOG, limit), "promotions": tail(paths.PROMOTION_LOG, limit)}
