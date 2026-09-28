"""PMS 照片分類與新工種候選服務，供操作台與 CLI 共用。

規則、圖像分類器、AI 建議、人工裁決各自留欄位。未知與無標題照片留在複核母體；
只有人確認的類別寫回 review.csv。新分類定義不會替任何照片自動作答。
"""

from __future__ import annotations

import io
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import paths
from core import pms_store as store
from core.evaluation_metrics import REVIEW_CONFIDENCE, fuse_work_items
from core.labeler import Labeler, load_boxes, load_reviews, save_review
from core.pms_source import work_items

PHOTO_COLUMNS = ["fileId", "title", "constrId", "constrName", "reportDate", "chipsOn", "specKey", "syncedAt"]
ROUTES = {
    "known": "既有工種",
    "known_untrained": "既有類別・模型尚未涵蓋",
    "unknown": "尚無分類・待看圖",
    "invalid": "既有裁決的類名待核對",
}
STATES = {
    "pending": "待複核",
    "classified": "已確認",
    "candidate": "新類候選",
    "uncertain": "資訊不足",
    "excluded": "非本輪工種語料",
}


def catalog(lab: Labeler | None = None) -> dict[str, dict]:
    lab = lab or Labeler.load()
    result: dict[str, dict] = {}
    for pattern, label in lab.rules:
        result.setdefault(label, {"label": label, "origin": "labels.yaml", "patterns": []})[
            "patterns"
        ].append(pattern.pattern)
    for label, row in store.approved_classes().items():
        result.setdefault(label, {k: v for k, v in row.items() if not k.startswith("_")})
    return result


def catalog_version() -> str:
    return store.digest([paths.LABELS_YAML.read_text(), catalog()])


def load_pool() -> pd.DataFrame:
    """只讀 PMS 施作項目；小類、fallback、空標題保留，出工照不進分類。"""
    if not paths.MANIFEST.exists():
        return pd.DataFrame(columns=PHOTO_COLUMNS)
    df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    if "fileId" not in df:
        raise ValueError("PMS manifest 缺 fileId 欄位。")
    df = work_items(df)
    for col in PHOTO_COLUMNS:
        if col not in df:
            df[col] = ""
    if "active" in df:
        df = df[df.active.str.lower().isin(["true", "1"])]
    df = Labeler.load().drop_excluded(df)
    return df[df.fileId != ""].drop_duplicates("fileId", keep="last").reset_index(drop=True)


def _photos(file_ids: list[str]) -> pd.DataFrame:
    if not file_ids:
        raise ValueError("請至少選一張 PMS 照片。")
    for fid in file_ids:
        store.validate_file_id(fid)
    pool = load_pool().set_index("fileId", drop=False)
    missing = set(file_ids) - set(pool.index)
    if missing:
        raise ValueError(f"照片不在有效 PMS 施作項目母體（WORK_ITEM）：{sorted(missing)[:5]}")
    return pool.loc[list(dict.fromkeys(file_ids))]


def photo_path(file_id: str) -> Path | None:
    store.validate_file_id(file_id)
    for p in sorted(paths.PHOTOS.glob(f"{file_id}.*")):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            return p
    prepared = paths.IMAGES / f"{file_id}.jpg"
    return prepared if prepared.exists() else None


def photo_bytes(file_id: str) -> bytes:
    """審阅圖統一方向與 JPEG 格式，保留到 1600px；不附加文字或模型框。"""
    path = photo_path(file_id)
    if path is None:
        raise ValueError(f"找不到照片：{file_id}")
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def work_item_keys(df: pd.DataFrame) -> dict[str, str]:
    """fileId → 工項 key（同日報 × 同標題）。缺日報 id 的照片不融合（key 空字串）。"""
    if "dailyReportInfoId" not in df:
        return {}
    rid = df.dailyReportInfoId.fillna("").astype(str)
    title = df.title.fillna("").astype(str) if "title" in df else ""
    keys = (rid + "|" + title).where(rid != "", "")
    return dict(zip(df.fileId.astype(str), keys))


# ── 缺失旗標（v17：與工種正交）──────────────────────────────────────────
# 與 labels.yaml 雜項-缺失改善規則同一組字。旗標＝人工（最優先）＞標題缺失字｜defect-probe 圖像分數。
DEFECT_TITLE = "缺失|缺改|美容|修繕"
DEFECT_SOURCES = {"human": "人工", "title": "標題", "image": "圖像"}


def defect_title(title) -> bool:
    return isinstance(title, str) and bool(re.search(DEFECT_TITLE, title))


def set_defect(file_id: str, defect: bool, *, reviewer: str) -> None:
    """人工改缺失旗標。只寫事件庫，不動 review.csv 的工種裁決。"""
    _photos([file_id])
    store.append([("defect", file_id, {"defect": bool(defect), "reviewer": _reviewer(reviewer)})])


# ── 泥作打底／粉光分層判斷 ────────────────────────────────────────────────
# 兩者是 QS0402 的前後兩個工序，計價分開，不合併。圖像只負責判「泥作打底/粉光群」；
# 群內信心夠（≥ STAGE_GROUP）但階段不確定（較大那個佔群內 < STAGE_SHARE）時看標題：
# 標題明確寫打底或粉光就用標題，沒寫就送人工。門檻用 v42 分組 OOF 在人工裁決照上選
# （半切選參、另一半評：人工照準確率 +1.9pt，CI [0, +4.0]；人工照上「標題有階段字」
# 與人工答案 0 衝突）。
STAGE_CLASSES = ("泥作-打底", "泥作-粉光")
STAGE_GROUP = 0.6
STAGE_SHARE = 0.8
STAGE_MANUAL = "泥作打底／粉光階段待人工"


def title_stage(title) -> str:
    """標題明確只寫一個階段才算數；兩個都寫（打底/粉光）或都沒寫 → 空字串。"""
    t = title if isinstance(title, str) else ""
    base, finish = "打底" in t, bool(re.search("粉光|粉刷", t))
    return STAGE_CLASSES[0] if base and not finish else STAGE_CLASSES[1] if finish and not base else ""


def resolve_stage(proba, classes, title) -> tuple[str, str]:
    """(最終類別, 決定來源 model|title|manual)。manual 時類別仍回模型的 top1 供參考。"""
    classes = [str(c) for c in classes]
    top = classes[int(np.argmax(proba))]
    if not all(c in classes for c in STAGE_CLASSES):
        return top, "model"
    pb, pf = (float(proba[classes.index(c)]) for c in STAGE_CLASSES)
    group = pb + pf
    if group < STAGE_GROUP or max(pb, pf) / group >= STAGE_SHARE:
        return top, "model"
    stage = title_stage(title)
    return (stage, "title") if stage else (top, "manual")


def local_model() -> dict:
    """只使用 PMS 特徵主檔，回傳真正的 train/test/unseen 與模型類別。

    預測用工項融合（同日報×同標題的兄弟照一起看，core.evaluation_metrics.fuse_work_items），
    v40 同卷 top1 0.852 → 0.911；信心與邊際也是融合後的值。
    """
    import split as split_mod

    name = split_mod.current()
    result = {
        "name": name,
        "classes": [],
        "scores": {},
        "stage": {},
        "defectScore": {},
        "defectThreshold": None,
        "train": [],
        "test": [],
        "warning": "",
    }
    spath = paths.SPLITS / f"{name}.json"
    model = split_mod.probe_path(name)
    feature = paths.FEATURES / f"{split_mod.encoder(name)}.npz"
    if spath.exists():
        sp = json.loads(spath.read_text())
        result.update(train=sp.get("train", []), test=sp.get("test", []))
    if not model.exists():
        result["warning"] = "尚無目前版本的分類器；仍可看圖、人工分類及整理候選。"
        return result
    with model.open("rb") as fh:
        clf = pickle.load(fh)["clf"]
    result["classes"] = list(clf.classes_)
    if not feature.exists():
        result["warning"] = "PMS 圖像特徵尚未建立，模型預測暫缺。"
        return result
    pool = load_pool()
    keys = work_item_keys(pool)
    titles = dict(zip(pool.fileId, pool.title)) if "title" in pool else {}
    with np.load(feature, allow_pickle=True) as z:
        file_ids = [str(f) for f in z["fileIds"]]
        emb = z["emb"]
    item_keys = [keys.get(f, "") for f in file_ids]
    proba = fuse_work_items(clf.predict_proba(emb), item_keys)
    order = np.sort(proba, axis=1)
    for i, (fid, p) in enumerate(zip(file_ids, proba)):
        pred, source = resolve_stage(p, clf.classes_, titles.get(fid, ""))
        result["scores"][fid] = (
            pred,
            float(p.max()),
            float(order[i, -1] - order[i, -2]) if len(p) > 1 else 0.0,
        )
        if source != "model":
            result["stage"][fid] = source
    result.update(_defect_scores(file_ids, emb, item_keys, split_mod.encoder(name)))
    return result


def _defect_scores(file_ids: list[str], emb, item_keys: list[str], encoder: str) -> dict:
    """defect-probe（src/defect_probe.py 存的 models/defect-probe-{encoder}.pkl）→ 工項融合分數。"""
    path = paths.MODELS / f"defect-probe-{encoder}.pkl"
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        art = pickle.load(fh)  # 本機受信任的訓練產物
    p = art["clf"].predict_proba(emb)[:, 1]
    fused = fuse_work_items(np.c_[1 - p, p], item_keys)[:, 1]
    return {"defectScore": dict(zip(file_ids, fused.astype(float))), "defectThreshold": art["threshold"]}


def revision(file_id: str) -> str:
    row = _photos([file_id]).iloc[0].to_dict()
    image = photo_path(file_id)
    return store.digest(
        [
            store.review_signatures().get(file_id, ""),
            store.latest("decision").get(file_id, {}).get("_seq", 0),
            catalog_version(),
            row,
            [str(image), image.stat().st_mtime_ns, image.stat().st_size] if image else None,
        ]
    )


def snapshot(model: dict | None = None) -> tuple[pd.DataFrame, dict]:
    pool = load_pool()
    model = local_model() if model is None else model
    known = catalog()
    lab = Labeler.load()
    human = load_reviews()
    decisions = store.active_decisions()
    suggestions = {r["fileId"]: r for r in store.events("suggestion")}
    defect_marks = store.latest("defect")
    trained, train_ids, test_ids = set(model["classes"]), set(model["train"]), set(model["test"])
    version = catalog_version()
    rows = []
    for r in pool.to_dict("records"):
        fid = r["fileId"]
        rule = lab.label(r["title"])
        rule = "" if rule in (None, lab.fallback) else rule
        reviewed = human.get(fid, "")
        label = reviewed or rule
        action = decisions.get(fid, {}).get("action") or ("classified" if reviewed else "pending")
        route = (
            "invalid"
            if reviewed and reviewed not in known
            else "unknown"
            if not label
            else "known_untrained"
            if label not in trained
            else "known"
        )
        pred, conf, margin = model["scores"].get(fid, ("", None, None))
        part = "test" if fid in test_ids else "train" if fid in train_ids else "unseen"
        reasons = []
        if route != "known":
            reasons.append(ROUTES[route])
        if pred and label and pred != label:
            reasons.append("照片模型與標籤不同")
        if margin is not None and margin < 0.25 and part != "train":
            reasons.append("照片模型難分")
        elif conf is not None and conf < REVIEW_CONFIDENCE and part != "train":
            reasons.append("照片模型信心低")
        if not pred:
            reasons.append("尚無照片模型預測")
        if model.get("stage", {}).get(fid) == "manual" and not reviewed:
            reasons.append(STAGE_MANUAL)
        if label == "雜項-缺失改善" and not reviewed:
            reasons.append("缺失改善照：工種待看圖確認")
        score = model.get("defectScore", {}).get(fid)
        threshold = model.get("defectThreshold")
        human_defect = defect_marks.get(fid)
        if human_defect is not None:
            defect, defect_source = bool(human_defect["defect"]), "human"
        elif defect_title(r["title"]):
            defect, defect_source = True, "title"
        elif score is not None and threshold is not None and score >= threshold:
            defect, defect_source = True, "image"
        else:
            defect, defect_source = False, ""
        suggestion = suggestions.get(fid, {})
        current_suggestion = bool(suggestion) and suggestion.get("catalogVersion") == version
        if current_suggestion:
            from core.pms_exchange import context_of

            current_suggestion = suggestion.get("contextHash") == store.digest(context_of(r))
        if suggestion and action == "pending":
            reasons.append("有看圖審閱建議待確認" if current_suggestion else "看圖審閱建議已過期")
        rows.append(
            {
                **r,
                "ruleClass": rule,
                "humanClass": reviewed,
                "modelClass": pred,
                "stageDecision": model.get("stage", {}).get(fid, "model" if pred else ""),
                "defectFlag": defect,
                "defectSource": defect_source,
                "defectScore": score,
                "modelConfidence": conf,
                "modelMargin": margin,
                "part": part,
                "route": route,
                "reviewState": action,
                "reviewReason": "；".join(reasons),
                "needsReview": (action == "pending" and bool(reasons)) or route == "invalid",
                "suggestion": suggestion,
                "suggestionCurrent": current_suggestion,
            }
        )
    extra = [
        "ruleClass",
        "humanClass",
        "modelClass",
        "stageDecision",
        "defectFlag",
        "defectSource",
        "defectScore",
        "modelConfidence",
        "modelMargin",
        "part",
        "route",
        "reviewState",
        "reviewReason",
        "needsReview",
        "suggestion",
        "suggestionCurrent",
    ]
    return pd.DataFrame(rows, columns=list(pool.columns) + extra), model


def _reviewer(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("請填寫確認者名稱。")
    return name


def decide(
    file_id: str,
    action: str,
    *,
    reviewer: str,
    label: str = "",
    reason: str = "",
    proposal_id: str = "",
    expected_revision: str | None = None,
) -> None:
    _photos([file_id])
    reviewer = _reviewer(reviewer)
    if expected_revision is not None and revision(file_id) != expected_revision:
        raise ValueError("照片裁決或分類表已更新，請重新載入後確認。")
    if action not in {"classified", "uncertain", "excluded"}:
        raise ValueError("不支援的照片裁決。新類候選請用候選流程。")
    if action == "classified" and label not in catalog():
        raise ValueError("類名不在分類表中；請先提出並核准新類候選。")
    if action != "classified" and not reason.strip():
        raise ValueError("暫緩或排除時請記下原因。")
    if proposal_id:
        proposal = store.latest("suggestion").get(proposal_id)
        if not proposal or proposal["fileId"] != file_id:
            raise ValueError("AI 建議與照片不相符。")
        if proposal["catalogVersion"] != catalog_version():
            raise ValueError("AI 建議使用舊分類表，請重新看圖或重新審閱。")
        from core.pms_exchange import validate_proposal

        validate_proposal(proposal)
    note = json.dumps(
        {"workflow": "pms", "reviewer": reviewer, "proposalId": proposal_id, "reason": reason},
        ensure_ascii=False,
    )
    if action == "classified":
        # 只改照片類別；保留舊複核介面曾畫過的證據框。
        save_review(file_id, label, note, boxes=load_boxes().get(file_id))
    store.append(
        [
            (
                "decision",
                file_id,
                {
                    "action": action,
                    "label": label if action == "classified" else "",
                    "reason": reason,
                    "reviewer": reviewer,
                    "proposalId": proposal_id,
                    "reviewSignature": store.review_signatures().get(file_id, ""),
                },
            )
        ]
    )


def propose_candidate(
    label: str,
    file_ids: list[str],
    *,
    reviewer: str,
    definition: str,
    basis: str = "",
    excludes: str = "",
    proposal_id: str = "",
) -> str:
    label = store.validate_label(label)
    reviewer = _reviewer(reviewer)
    _photos(file_ids)
    if label in catalog():
        raise ValueError("分類表已有此類，請確認為既有類別；未進模型不等於新工種。")
    if not definition.strip():
        raise ValueError("請說明候選照片共同的可見特徵。")
    if proposal_id:
        from core.pms_exchange import validate_proposal

        proposal = store.latest("suggestion").get(proposal_id)
        if not proposal or proposal["fileId"] not in file_ids or proposal["decision"] != "new_candidate":
            raise ValueError("AI 建議與新類候選不相符。")
        validate_proposal(proposal)
    key = store.digest(label)[:20]
    previous = store.latest("candidate").get(key, {})
    if previous.get("status") in {"approved", "rejected"}:
        raise ValueError("此候選已結案，請核對既有決定。")
    signatures = store.review_signatures()
    payload = {
        "label": label,
        "definition": definition.strip(),
        "basis": basis.strip(),
        "excludes": excludes.strip(),
        "reviewer": reviewer,
        "status": "proposed",
        "fileIds": sorted(set(file_ids) | set(previous.get("fileIds", []))),
        "proposalIds": sorted(
            set(previous.get("proposalIds", [])) | ({proposal_id} if proposal_id else set())
        ),
    }
    records = [("candidate", key, payload)]
    records.extend(
        (
            "decision",
            fid,
            {
                "action": "candidate",
                "candidateId": key,
                "reason": definition,
                "reviewer": reviewer,
                "proposalId": proposal_id,
                "reviewSignature": signatures.get(fid, ""),
            },
        )
        for fid in set(file_ids)
    )
    store.append(records, expected=("candidate", key, previous.get("_seq", 0)))
    return key


def resolve_candidate(
    candidate_id: str,
    *,
    approve: bool,
    reviewer: str,
    definition: str,
    excludes: str,
    basis: str,
    expected_seq: int,
) -> None:
    reviewer = _reviewer(reviewer)
    old = store.latest("candidate").get(candidate_id)
    if not old or old["status"] != "proposed":
        raise ValueError("候選不存在或已結案。")
    if approve and (not definition.strip() or not excludes.strip() or not basis.strip()):
        raise ValueError("核准新類須填定義、排除條件與分類樹／QS 依據。")
    if approve and old["label"] in catalog():
        raise ValueError("此類已存在，請重新載入分類表。")
    updated = {
        **old,
        "status": "approved" if approve else "rejected",
        "definition": definition,
        "excludes": excludes,
        "basis": basis,
        "reviewer": reviewer,
    }
    records = [("candidate", candidate_id, updated)]
    if approve:
        records.append(
            (
                "class",
                old["label"],
                {
                    "label": old["label"],
                    "origin": "human-approved",
                    "definition": definition,
                    "excludes": excludes,
                    "basis": basis,
                    "reviewer": reviewer,
                    "candidateId": candidate_id,
                },
            )
        )
    # 核准定義不替照片寫 review.csv；每张仍需人工確認，張數門檻留給訓練。
    store.append(records, expected=("candidate", candidate_id, expected_seq))


def class_inventory(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    human = df[df.reviewState == "classified"].humanClass.value_counts()
    weak = df.ruleClass.value_counts()
    lab = Labeler.load()
    effective = lab.apply(load_pool(), drop_small=False).cls.value_counts()
    return pd.DataFrame(
        [
            {
                "類別": label,
                "人工確認": int(human.get(label, 0)),
                "標題規則命中": int(weak.get(label, 0)),
                "有效標籤": int(effective.get(label, 0)),
                "樣本門檻": lab.min_class_size,
                "還差": max(0, lab.min_class_size - int(effective.get(label, 0))),
                "目前模型": "已涵蓋" if label in model["classes"] else "尚未涵蓋",
                "定義來源": row["origin"],
            }
            for label, row in sorted(catalog().items())
        ],
        columns=["類別", "人工確認", "標題規則命中", "有效標籤", "樣本門檻", "還差", "目前模型", "定義來源"],
    )


def discover(df: pd.DataFrame) -> list[dict]:
    """有效 PMS 未知照片的文字群組提示；不把關鍵字或相似度當新類定義。"""
    from newclass import candidates

    pending = df[(df.route == "unknown") & df.reviewState.isin(["pending", "uncertain"])]
    pending = pending[pending.title.str.strip() != ""]
    fb = Counter(pending.title)
    ok = Counter(df.loc[df.ruleClass != "", "title"])
    groups = []
    for group in candidates(fb, ok, min_photos=2, top=20):
        titles = {title for title, _ in group["titles"]}
        rows = pending[pending.title.isin(titles)]
        groups.append(
            {
                "term": group["term"],
                "fileIds": rows.fileId.tolist(),
                "photos": len(rows),
                "sites": rows.loc[rows.constrId != "", "constrId"].nunique(),
                "days": rows.loc[rows.reportDate != "", "reportDate"].nunique(),
                "titles": sorted(titles),
            }
        )
    return groups


def calibration() -> dict:
    """歷史裁決對照純標題規則；不是黃金集，也不以人工答案覆蓋規則輸出。"""
    pool = load_pool()
    human = load_reviews()
    selected = pool[pool.fileId.isin(human) & ~pool.fileId.isin(store.blocked_ids())].copy()
    selected["final"] = selected.fileId.map(human)
    selected["rule"] = selected.title.map(Labeler.load().label)
    matches = int((selected.rule == selected.final).sum())
    return {
        "reviewed": len(selected),
        "ruleMatches": matches,
        "ruleAccuracy": matches / len(selected) if len(selected) else None,
        "note": "歷史已裁 PMS 子集；非獨立測試集，不作自動確認門檻。",
    }
