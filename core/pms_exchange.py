"""ChatGPT 看圖審閱包：本機匯出、嚴格驗證回覆、只保存 AI 建議。

不連線、不自動送照片。封包保存图片與分類表版本，過期、漏圖或多圖回覆整批拒收。
"""

from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile

from core import pms_photos
from core import pms_store as store

PROMPT = """# PMS 日報照片工種複核

這批只做照片工種分類與新工種候選，不做缺失框、YOLO 或工程合格判定。
先逐張讀 images/ 的實際圖片，描述可見材料、工具、部位和動作，再讀 context.json。
日報標題與查驗重點只是參考，不能取代圖片證據。看不到圖片就回答 uncertain，勿猜。
catalog.json 是完整已知分類表，包含模型尚未學會的類別；不要把已知類別當新工種。

每張選一個 decision：
- existing：圖片支持某個已知類別，label 必須逐字等於 catalog 的鍵。
- new_candidate：可見內容一致但已知分類未涵蓋。label 用「工種-施作內容」，
  candidateDefinition 說明共同視覺特徵與鄰近舊類的差異。這只是候選，不能自動開類。
- uncertain：影像太小、遮擋、多工種並列，或無足夠線索；label 留空。
- not_construction：圖片明確不是這次的施工工種語料；label 留空。無標題不能當排除理由。

結果存成 JSON，沿用 response-template.json 的 packetId/catalogVersion/fileId/imageHash。
model 填實際可確認的名稱；只能確認是 ChatGPT 時填 ChatGPT，不捏造版本。
逐張填 visualEvidence、reason，existing/new_candidate 填 label，new_candidate 另填
candidateDefinition；alternatives 最多三個已知類別。所有照片恰好各一筆。
不要输出未看照片的判斷，也不要把自行宣稱的信心當成正確率。
"""


def context_of(row: dict) -> dict:
    return {
        k: str(row.get(k, ""))
        for k in ("fileId", "source", "title", "chipsOn", "specKey", "constrId", "reportDate")
    }


def export_packet(file_ids: list[str]) -> tuple[str, bytes]:
    if len(file_ids) > 50:
        raise ValueError("每批最多 50 張，請分批審閱。")
    selected = pms_photos.require_photos(file_ids)
    packet_id = uuid.uuid4().hex
    version = pms_photos.catalog_version()
    items, contexts = [], []
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for row in selected.to_dict("records"):
            fid = row["fileId"]
            blob = pms_photos.photo_bytes(fid)
            context = context_of(row)
            contexts.append(context)
            items.append(
                {
                    "fileId": fid,
                    "imageHash": hashlib.sha256(blob).hexdigest(),
                    "contextHash": store.digest(context),
                }
            )
            z.writestr(f"images/{fid}.jpg", blob)
        packet = {
            "schemaVersion": 1,
            "packetId": packet_id,
            "catalogVersion": version,
            "source": "pms",
            "items": items,
        }
        template = {
            "schemaVersion": 1,
            "packetId": packet_id,
            "catalogVersion": version,
            "model": "",
            "results": [
                {
                    "fileId": item["fileId"],
                    "imageHash": item["imageHash"],
                    "decision": "uncertain",
                    "label": "",
                    "visualEvidence": "",
                    "reason": "",
                    "alternatives": [],
                    "candidateDefinition": "",
                }
                for item in items
            ],
        }
        for name, obj in (
            ("packet.json", packet),
            ("catalog.json", pms_photos.catalog()),
            ("context.json", contexts),
            ("response-template.json", template),
        ):
            z.writestr(name, json.dumps(obj, ensure_ascii=False, indent=2))
        z.writestr("PROMPT.md", PROMPT)
    # 圖片全部可讀後才登記；失敗不留下半套封包。
    store.append([("packet", packet_id, packet)])
    return packet_id, archive.getvalue()


def validate_fresh(item: dict, row: dict) -> None:
    if item["contextHash"] != store.digest(context_of(row)):
        raise ValueError(f"{row['fileId']} 的日報內容已更新，請重新匯出審閱包。")
    if item["imageHash"] != hashlib.sha256(pms_photos.photo_bytes(row["fileId"])).hexdigest():
        raise ValueError(f"{row['fileId']} 的圖片已更新，請重新匯出審閱包。")


def validate_proposal(proposal: dict) -> None:
    if proposal["catalogVersion"] != pms_photos.catalog_version():
        raise ValueError("建議的分類表版本已過期，請重新審閱。")
    row = pms_photos.require_photos([proposal["fileId"]]).iloc[0].to_dict()
    validate_fresh(proposal, row)


def import_suggestions(payload: dict) -> int:
    """匯入整批只產生 suggestion 事件；完全不改 review.csv 或分類定義。"""
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise ValueError("回覆需為 schemaVersion=1 的 JSON 物件。")
    packet_id = payload.get("packetId")
    if not isinstance(packet_id, str):
        raise ValueError("回覆缺 packetId。")
    packet = store.latest("packet").get(packet_id)
    if not packet:
        raise ValueError("找不到本機發出的審閱包，請先匯出。")
    version = pms_photos.catalog_version()
    if payload.get("catalogVersion") != version or packet["catalogVersion"] != version:
        raise ValueError("分類表已改版，請重新匯出審閱包。")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("請填寫建議來源 model。")
    results = payload.get("results")
    if not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
        raise ValueError("results 需為建議物件清單。")
    ids = [r.get("fileId") for r in results]
    if not all(isinstance(fid, str) for fid in ids):
        raise ValueError("每筆建議需帶 fileId。")
    items = {r["fileId"]: r for r in packet["items"]}
    if len(ids) != len(set(ids)) or set(ids) != set(items):
        raise ValueError("回覆必須包含封包內所有照片，且每張恰好一筆。")
    selected = pms_photos.require_photos(ids)
    known = pms_photos.catalog()
    saved = store.latest("suggestion")
    records = []
    for r in results:
        fid = r["fileId"]
        item = items[fid]
        if r.get("imageHash") != item["imageHash"]:
            raise ValueError(f"{fid} 圖片版本不相符。")
        validate_fresh(item, selected.loc[fid].to_dict())
        decision = r.get("decision")
        label = r.get("label", "")
        if not isinstance(decision, str) or decision not in {
            "existing",
            "new_candidate",
            "uncertain",
            "not_construction",
        }:
            raise ValueError(f"{fid} 的 decision 無效。")
        if not isinstance(label, str):
            raise ValueError("label 需為字串。")
        if decision == "existing" and label not in known:
            raise ValueError(f"{fid} 引用不存在的既有類別。")
        if decision == "new_candidate":
            label = store.validate_label(label)
            if label in known:
                raise ValueError(f"{label} 已在分類表，應使用 existing。")
            if not isinstance(r.get("candidateDefinition"), str) or not r["candidateDefinition"].strip():
                raise ValueError("新類候選需說明可見特徵。")
        if decision in {"uncertain", "not_construction"} and label:
            raise ValueError("資訊不足或非施工照片的 label 必須留空。")
        for field in ("visualEvidence", "reason"):
            if not isinstance(r.get(field), str) or not r[field].strip():
                raise ValueError(f"{fid} 缺少 {field}。")
        alternatives = r.get("alternatives", [])
        if (
            not isinstance(alternatives, list)
            or len(alternatives) > 3
            or not all(isinstance(c, str) and c in known for c in alternatives)
        ):
            raise ValueError("alternatives 只能列最多三個已知類別。")
        if not isinstance(r.get("candidateDefinition", ""), str):
            raise ValueError("candidateDefinition 必須是字串。")
        title_fields = {}
        if "titleEvidence" in r or "titleRelation" in r:
            if not isinstance(r.get("titleEvidence"), str) or not r["titleEvidence"].strip():
                raise ValueError("titleEvidence 必須說明標題提供的訊息或缺乏訊息。")
            if r.get("titleRelation") not in ("agrees", "conflicts", "insufficient_context"):
                raise ValueError("titleRelation 需為 agrees、conflicts 或 insufficient_context。")
            title_fields = {"titleEvidence": r["titleEvidence"], "titleRelation": r["titleRelation"]}
        result = {
            "fileId": fid,
            "packetId": packet_id,
            "catalogVersion": version,
            "model": model.strip(),
            **item,
            "decision": decision,
            "label": label,
            "visualEvidence": r["visualEvidence"],
            "reason": r["reason"],
            "alternatives": alternatives,
            "candidateDefinition": r.get("candidateDefinition", ""),
            **title_fields,
        }
        proposal_id = store.digest(result)
        result["proposalId"] = proposal_id
        if proposal_id not in saved:
            records.append(("suggestion", proposal_id, result))
    store.append(records)
    return len(records)
