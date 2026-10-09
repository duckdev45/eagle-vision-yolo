"""候選模型自動上線：公平考卷過關才切換，沒過就留在原版。

2026-10-09 定案：考卷過關就自動切換（不再等人按）。所以「過關」必須是機械可判的定義，
而且每一次的考卷與決定都要留下紀錄（data/promotion-log.jsonl），事後能查、能回滾。

為什麼不能直接比兩版報告的 top1：每版 test 是滾動切的，考的照片不同，分數不可比
（操作台自己就這樣警告）。公平考卷＝

    候選版的 test  −  現行版的 train（現行版背過的照片不能拿來考它）
                   ∩  兩版都有特徵、答案是現行版認得的類

同一份考卷、同一套預測（工項融合＋打底／粉光分層，與上線完全相同），兩版各考一次。

為什麼不用 G2（src/g2_gate.py）：G2 要 ≥300 張仲裁過的黃金集，到今天是 0 份。黃金集有了以後
G2 應該接進來當第二道門；在那之前，公平考卷是唯一能自動跑的不作弊比較。

過關條件（非劣性：新版通常是「多學了東西」，要的是沒有變差，不是每次都要變好）：
    1. 考卷至少 MIN_EXAM 張，否則證據不足，不切
    2. top1 不低於現行版 − TOLERANCE
    3. macroF1 不低於現行版 − TOLERANCE
    4. 考卷上有足夠樣本（≥ MIN_CLASS_SUPPORT）的類別，召回不得掉超過 MAX_CLASS_DROP
       ——防止總分持平、某一類被犧牲
    5. 分類表版本必須與候選版訓練時一致（不然上線後類別對不上）
人審子集的分數另外記錄，但不當門檻：人審照是佇列挑出來的難題，張數少、各版重疊不一，當門檻會抖。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np

from core import model_registry as registry
from core import paths, pms_review
from core.evaluation_metrics import fuse_work_items
from core.labeler import load_reviews

MIN_EXAM = 100
TOLERANCE = 0.01
MIN_CLASS_SUPPORT = 10
MAX_CLASS_DROP = 0.15


def _load(name: str) -> tuple[dict, Any]:
    return registry.load_split(name), registry.load_probe(name)


def _predict(clf, emb, ids: list[str], titles: dict[str, str], keys: dict[str, str]) -> list[str]:
    classes = [str(c) for c in clf.classes_]
    proba = fuse_work_items(clf.predict_proba(emb), [keys.get(f, "") for f in ids])
    return [pms_review.resolve_stage(p, classes, titles.get(f, ""))[0] for p, f in zip(proba, ids)]


def score(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import f1_score

    labels = sorted(set(y_true))
    recall = {}
    for c in labels:
        idx = [i for i, t in enumerate(y_true) if t == c]
        recall[c] = {"support": len(idx), "recall": sum(y_pred[i] == c for i in idx) / len(idx)}
    if not y_true:  # None 而不是 NaN：紀錄是 JSON，NaN 不是合法 JSON
        return {"n": 0, "top1": None, "macroF1": None, "recall": {}}
    return {
        "n": len(y_true),
        "top1": float(np.mean([a == b for a, b in zip(y_true, y_pred)])),
        "macroF1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": recall,
    }


def exam(candidate: str, baseline: str) -> dict:
    """兩版在同一份公平考卷上的成績。"""

    cand_sp, cand_clf = _load(candidate)
    base_sp, base_clf = _load(baseline)
    pool = pms_review.load_pool()
    titles = dict(zip(pool.fileId, pool.title))
    keys = pms_review.work_item_keys(pool)
    base_classes = {str(c) for c in base_clf.classes_}
    labels = cand_sp["labels"]
    paper = [
        f
        for f in cand_sp["test"]
        if f not in set(base_sp["train"]) and labels.get(f) in base_classes and f in titles
    ]
    out: dict = {"candidate": candidate, "baseline": baseline, "examSize": 0}
    feats = {}
    for name in (candidate, baseline):
        enc = registry.encoder(name)
        if enc not in feats:
            ids, emb = registry.load_features(enc)
            feats[enc] = ({f: i for i, f in enumerate(ids)}, emb)
    for name in (candidate, baseline):
        row = feats[registry.encoder(name)][0]
        paper = [f for f in paper if f in row]
    out["examSize"] = len(paper)
    out["leakage"] = {
        "candidateTestInBaselineTrain": len(set(cand_sp["test"]) & set(base_sp["train"])),
        "candidateTest": len(cand_sp["test"]),
    }
    if not paper:
        return out
    y = [labels[f] for f in paper]
    human = load_reviews()
    human_idx = [i for i, f in enumerate(paper) if f in human]
    for role, name, clf in (("candidateScore", candidate, cand_clf), ("baselineScore", baseline, base_clf)):
        row, emb = feats[registry.encoder(name)]
        pred = _predict(clf, emb[[row[f] for f in paper]], paper, titles, keys)
        out[role] = score(y, pred)
        out[role]["humanSubset"] = {
            k: v
            for k, v in score([y[i] for i in human_idx], [pred[i] for i in human_idx]).items()
            if k != "recall"
        }
    return out


def verdict(result: dict, *, catalog_ok: bool = True) -> tuple[bool, list[str]]:
    """純函式：考卷結果 → (過關?, 理由)。理由逐條列出，失敗的那幾條會寫進每日紀錄與通知。"""
    reasons: list[str] = []
    if not catalog_ok:
        reasons.append("分類表已改版，候選版訓練時的類別與現在不符")
    n = result.get("examSize", 0)
    if n < MIN_EXAM:
        reasons.append(f"考卷只有 {n} 張（需要 ≥ {MIN_EXAM}），證據不足")
        return False, reasons
    cand, base = result["candidateScore"], result["baselineScore"]
    for metric in ("top1", "macroF1"):
        if cand[metric] < base[metric] - TOLERANCE:
            reasons.append(f"{metric} {cand[metric]:.3f} < 現行 {base[metric]:.3f} − {TOLERANCE}")
    for cls, b in base["recall"].items():
        if b["support"] < MIN_CLASS_SUPPORT:
            continue
        drop = b["recall"] - cand["recall"][cls]["recall"]
        if drop > MAX_CLASS_DROP:
            reasons.append(
                f"{cls} 召回 {b['recall']:.2f} → {cand['recall'][cls]['recall']:.2f}（掉 {drop:.0%}）"
            )
    return not reasons, reasons


def promote(
    candidate: str, *, export: Callable[[str], object], baseline: str | None = None, log=print
) -> dict:
    """考一次；過關就切換 CURRENT 並呼叫 `export(candidate)` 匯出服務包。紀錄同時附加到 promotion-log.jsonl。

    `export` 由編排端注入（src/daily.py 接 export_service_bundle，要 train extra 的 torch；測試注入替身）
    ——服務層不反向 import 腳本。匯出失敗不回滾切換——操作台用的是本機探針，服務包只是同步副本，失敗會寫進紀錄等人處理。
    """
    baseline = baseline or registry.current()
    record: dict = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "candidate": candidate,
        "baseline": baseline,
    }
    if candidate == baseline:
        record.update(promoted=False, reasons=["候選版就是現行版"])
    else:
        result = exam(candidate, baseline)
        catalog_ok = registry.load_split(candidate).get("pmsCatalogVersion") == pms_review.catalog_version()
        ok, reasons = verdict(result, catalog_ok=catalog_ok)
        record.update(promoted=ok, reasons=reasons, exam=result)
        if ok:
            registry.set_current(candidate)
            log(f"✅ {candidate} 公平考卷過關，已切換（原 {baseline}）")
            try:
                export(candidate)
                record["serviceBundle"] = candidate
            except Exception as exc:  # 匯出失敗要記下來，不該讓切換結果被吞掉
                record["serviceBundleError"] = f"{type(exc).__name__}: {exc}"
                log(f"⚠ 服務包匯出失敗：{exc}")
        else:
            log(f"⏸ {candidate} 未過關，維持 {baseline}：" + "；".join(reasons))
    paths.PROMOTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with paths.PROMOTION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")
    return record
