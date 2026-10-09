"""PMS 工作台的檢視：逐張快照、類別盤點、新工種線索、標題規則校準。

只組裝畫面要的表，不判斷「要不要人看」（那是 core/routing.py 的分流結果），也不寫入
（寫入走 core/pms_decisions.py）。core 的其他模組不得 import 本模組（tests/test_architecture.py 擋），
所以這裡可以直接依賴 routing 而不會循環。

拆分（2026-10-09）：母體與分類表 → pms_photos；模型判讀（泥作分層、缺失旗標、本機預測）→ pms_model；
人審寫回 → pms_decisions。
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from core import pms_store as store
from core import routing
from core.labeler import Labeler, load_reviews
from core.pms_exchange import context_of
from core.pms_model import defect_title, local_model
from core.pms_photos import catalog, catalog_version, load_pool

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


def snapshot(model: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """工作台的逐張檢視。「要不要人看」不在這裡判斷——一律採用最近一次分流（core/routing.py），
    收件匣、工作台、`make pms-status`、`make queue` 才會是同一個數字（2026-10-09 前三處各算一套，
    同一批照片給出 478／190／376 三個答案）。"""
    open_items = routing.queue_items()
    routed = dict(zip(open_items.fileId, open_items.reason))
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
        reasons = [routed[fid]] if fid in routed and action == "pending" else []
        if route == "invalid":
            reasons.append(ROUTES[route])
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
            current_suggestion = suggestion.get("contextHash") == store.digest(context_of(r))
        if suggestion and action == "pending":  # 只是提示，不影響「要不要人看」
            notes = [*reasons, "有看圖審閱建議待確認" if current_suggestion else "看圖審閱建議已過期"]
        else:
            notes = reasons
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
                "reviewReason": "；".join(notes),
                "needsReview": bool(reasons),
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
    from core.rule_candidates import candidates

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
