"""人審寫回：照片裁決、缺失旗標、新工種候選的提出與核決。

**所有人審結果只經這裡寫入**（收件匣、工作台、標框頁、CLI、未來標註平台）：
工種裁決同時寫 data/review.csv 與事件庫，缺失旗標與候選只寫事件庫。
每個入口都先驗證照片在有效母體、確認者有填、類名在分類表裡。
2026-10-09 自 pms_review.py 拆出。
"""

from __future__ import annotations

import json

from core import pms_store as store
from core.labeler import load_boxes, save_review
from core.pms_exchange import validate_proposal
from core.pms_photos import catalog, catalog_version, photo_path, require_photos


def _reviewer(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("請填寫確認者名稱。")
    return name


def revision(file_id: str) -> str:
    row = require_photos([file_id]).iloc[0].to_dict()
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


def decide(
    file_id: str,
    action: str,
    *,
    reviewer: str,
    label: str = "",
    reason: str = "",
    proposal_id: str = "",
    expected_revision: str | None = None,
    boxes: list | None = None,
) -> None:
    """人審寫回的唯一入口（收件匣、工作台、標框頁、未來標註平台都走這裡）。

    `boxes` 給了就換成這組證據框（0~1000 比例）；不給就保留這張照片既有的框。
    """
    require_photos([file_id])
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
        validate_proposal(proposal)
    note = json.dumps(
        {"workflow": "pms", "reviewer": reviewer, "proposalId": proposal_id, "reason": reason},
        ensure_ascii=False,
    )
    if action == "classified":
        save_review(file_id, label, note, boxes=boxes if boxes is not None else load_boxes().get(file_id))
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


def set_defect(file_id: str, defect: bool, *, reviewer: str) -> None:
    """人工改缺失旗標。只寫事件庫，不動 review.csv 的工種裁決。"""
    require_photos([file_id])
    store.append([("defect", file_id, {"defect": bool(defect), "reviewer": _reviewer(reviewer)})])


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
    require_photos(file_ids)
    if label in catalog():
        raise ValueError("分類表已有此類，請確認為既有類別；未進模型不等於新工種。")
    if not definition.strip():
        raise ValueError("請說明候選照片共同的可見特徵。")
    if proposal_id:
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
