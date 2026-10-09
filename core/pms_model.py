"""模型輸出怎麼讀：本機探針預測、泥作打底／粉光分層、缺失旗標。

這裡的判讀規則（`resolve_stage`、`defect_title`、`work_item_keys` 的工項融合）同時被
分流、公平考卷、批次推論與服務包使用，服務端的複本由 tests/test_service_fusion_parity.py 對齊。
2026-10-09 自 pms_review.py 拆出。
"""

from __future__ import annotations

import pickle
import re

import numpy as np

from core import model_registry as registry
from core.evaluation_metrics import fuse_work_items
from core.pms_photos import load_pool, work_item_keys

# ── 缺失旗標（v17：與工種正交）──────────────────────────────────────────
# 與 labels.yaml 雜項-缺失改善規則同一組字。旗標＝人工（最優先）＞標題缺失字｜defect-probe 圖像分數。
DEFECT_TITLE = "缺失|缺改|美容|修繕"
DEFECT_SOURCES = {"human": "人工", "title": "標題", "image": "圖像"}


def defect_title(title) -> bool:
    return isinstance(title, str) and bool(re.search(DEFECT_TITLE, title))


# ── 泥作打底／粉光分層判斷 ────────────────────────────────────────────────
# 兩者是 QS0402 的前後兩個工序，計價分開，不合併。圖像只負責判「泥作打底/粉光群」；
# 群內信心夠（≥ STAGE_GROUP）但階段不確定（較大那個佔群內 < STAGE_SHARE）時看標題：
# 標題明確寫打底或粉光就用標題，沒寫就送人工。門檻用 v42 分組 OOF 在人工裁決照上選
# （半切選參、另一半評：人工照準確率 +1.9pt，CI [0, +4.0]；人工照上「標題有階段字」
# 與人工答案 0 衝突）。
STAGE_CLASSES = ("泥作-打底", "泥作-粉光")
STAGE_GROUP = 0.6
STAGE_SHARE = 0.8


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
    name = registry.current()
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
    encoder = registry.encoder(name)
    feature = registry.feature_path(encoder)
    if registry.split_path(name).exists():
        sp = registry.load_split(name)
        result.update(train=sp.get("train", []), test=sp.get("test", []))
    if not registry.probe_path(name).exists():
        result["warning"] = "尚無目前版本的分類器；仍可看圖、人工分類及整理候選。"
        return result
    clf = registry.load_probe(name)
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
    result.update(_defect_scores(file_ids, emb, item_keys, encoder))
    return result


def _defect_scores(file_ids: list[str], emb, item_keys: list[str], encoder: str) -> dict:
    """defect-probe（src/defect_probe.py 訓練的缺失旗標探針）→ 工項融合分數。"""
    path = registry.defect_probe_path(encoder)
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        art = pickle.load(fh)  # 本機受信任的訓練產物
    p = art["clf"].predict_proba(emb)[:, 1]
    fused = fuse_work_items(np.c_[1 - p, p], item_keys)[:, 1]
    return {"defectScore": dict(zip(file_ids, fused.astype(float))), "defectThreshold": art["threshold"]}
