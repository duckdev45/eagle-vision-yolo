"""缺失軸最小可行 baseline：用既有 embedding 判「這張像不像缺失改善照」。

還沒有任何人審缺失框（data/defects.csv 不存在，CVAT 兩包 2026-09-06 匯出後未回收），
YOLO 樣態偵測無從訓練。能做的只有**照片層級的二元弱標籤**：
日報標題命中 labels.yaml 雜項-缺失改善 那條規則的字（缺失|缺改|美容|修繕）＝正例，
人工裁成 雜項-缺失改善 的也算正例，其餘施作照＝負例。

弱標籤的意思：正例是「缺失**改善**作業」照（常拍到補漆、補磚、打鑿），不等於畫面有缺失；
負例裡也有沒寫進標題的缺失。所以產物不是缺失偵測器，而是**標註優先序**——
CVAT 冷啟動按分數由高往低抽，比 cvat_export 的組內均勻抽更快碰到真缺失。

評估按 案場×日期 整組 StratifiedGroupKFold（split 鐵律），另列各案場內 AUC：
正例集中在收尾階段的兩個案場，全域 AUC 會含「案場／階段」的混淆，案場內的才是畫面訊號。

    uv run --extra train src/defect_probe.py [--model siglip]   # 預設 features.DEFAULT_ENCODER

產出：reports/defect-probe/summary.json、scores.csv（全部 PMS 施作照，分數高→低）
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))
import paths

# 缺失字與工作台缺失旗標同一個來源（core.pms_review.DEFECT_TITLE）。v17 起有工種的缺改照
# 歸工種，所以這裡直接看標題，不看 cls。
from core.pms_review import DEFECT_TITLE

DEFECT_CLASS = "雜項-缺失改善"
IMAGE_PRECISION = 0.8  # 圖像旗標門檻：OOF 上（對標題弱標籤）precision ≥ 此值的最低分數


def weak_labels(
    df: pd.DataFrame, reviews: dict[str, str], marks: dict[str, dict] | None = None
) -> np.ndarray:
    """標題命中缺失字或人工裁成缺失改善 → 1；其餘 0。人工改過的缺失旗標（marks）最優先。"""
    by_title = df.title.fillna("").astype(str).str.contains(DEFECT_TITLE, regex=True)
    by_human = df.fileId.map(reviews).eq(DEFECT_CLASS)
    y = (by_title | by_human).to_numpy(dtype=int)
    for i, fid in enumerate(df.fileId):
        if marks and fid in marks:
            y[i] = int(bool(marks[fid]["defect"]))
    return y


def _fuse(p: np.ndarray, keys: list[str]) -> np.ndarray:
    """同工項兄弟照的缺失分數融合（與工種同一套 fuse_work_items）。"""
    from core.evaluation_metrics import fuse_work_items

    return fuse_work_items(np.c_[1 - p, p], keys)[:, 1]


def run(model_key: str | None = None, seeds: int = 3, log=print) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    import features
    from core.labeler import load_reviews
    from core.pms_review import load_pool, work_item_keys

    model_key = model_key or features.DEFAULT_ENCODER
    z = np.load(paths.FEATURES / f"{model_key}.npz", allow_pickle=True)
    emb = dict(zip(z["fileIds"].tolist(), z["emb"]))
    df = load_pool()
    df = df[df.fileId.isin(emb)].reset_index(drop=True)
    from core.pms_store import latest

    y = weak_labels(df, load_reviews(), latest("defect"))
    g = (df.constrId.astype(str) + "|" + df.reportDate.astype(str)).to_numpy()
    x = np.stack([emb[f] for f in df.fileId])
    log(f"PMS 施作照 {len(df)} 張，缺失改善弱正例 {int(y.sum())} 張，{len(set(g))} 個案場×日期組")

    def clf():
        return LogisticRegression(C=10, max_iter=3000, class_weight="balanced")

    oof = np.zeros(len(y))
    for s in range(seeds):
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(x, y, g):
            oof[va] += clf().fit(x[tr], y[tr]).predict_proba(x[va])[:, 1] / seeds
    item_of = work_item_keys(df)
    keys = [item_of.get(f, "") for f in df.fileId]
    oof_raw, oof = oof, _fuse(oof, keys)  # 以下指標與排序都用工項融合後的分數

    order = np.argsort(-oof)
    metrics = {
        "encoder": model_key,
        "photos": len(y),
        "positives": int(y.sum()),
        "baseRate": round(float(y.mean()), 4),
        "cvAUC": round(float(roc_auc_score(y, oof)), 4),
        "cvAP": round(float(average_precision_score(y, oof)), 4),
        "cvAUCSinglePhoto": round(float(roc_auc_score(y, oof_raw)), 4),
        "precisionAtK": {k: round(float(y[order[:k]].mean()), 3) for k in (50, 100, 200) if k <= len(y)},
        "perSiteAUC": {
            site: {
                "n": int(m.sum()),
                "pos": int(y[m].sum()),
                "auc": round(float(roc_auc_score(y[m], oof[m])), 4),
            }
            for site in sorted(df.constrName.unique())
            if (m := (df.constrName == site).to_numpy()).any() and 5 <= y[m].sum() < m.sum()
        },
        "note": "弱標籤＝標題含缺失字；分數用途是 CVAT 標註優先序，不是缺失判定",
    }
    log(
        f"grouped CV（工項融合）：AUC {metrics['cvAUC']}  AP {metrics['cvAP']}（base {metrics['baseRate']}，"
        f"單張 AUC {metrics['cvAUCSinglePhoto']}）"
        f"  P@50 {metrics['precisionAtK'].get(50)}"
    )
    for site, m in metrics["perSiteAUC"].items():
        log(f"  {site} 案場內 AUC {m['auc']}（{m['pos']}/{m['n']}）")

    # 圖像旗標門檻：OOF 分數由高往低，precision ≥ IMAGE_PRECISION 的最低分數
    hit = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    ok = np.where(hit >= IMAGE_PRECISION)[0]
    threshold = float(oof[order][ok.max()]) if len(ok) else 1.0
    metrics["imageFlagThreshold"] = round(threshold, 4)
    metrics["imageFlagPrecision"] = IMAGE_PRECISION
    log(f"圖像缺失旗標門檻 {threshold:.3f}（OOF precision ≥ {IMAGE_PRECISION}）")

    # 全量重訓給分數：已是正例的也列出，標註時一樣需要框
    final = clf().fit(x, y)
    score = _fuse(final.predict_proba(x)[:, 1], keys)
    with (paths.MODELS / f"defect-probe-{model_key}.pkl").open("wb") as fh:
        pickle.dump({"clf": final, "encoder": model_key, "threshold": threshold}, fh)
    out = paths.REPORTS_OUT / "defect-probe"
    out.mkdir(parents=True, exist_ok=True)
    df.assign(defectScore=score.round(4), cvScore=oof.round(4), weakLabel=y)[
        ["fileId", "defectScore", "cvScore", "weakLabel", "title", "constrName", "reportDate"]
    ].sort_values("defectScore", ascending=False).to_csv(out / "scores.csv", index=False)
    # 不叫 metrics.json：journal.py 會把 reports/*/metrics.json 當成訓練 run
    (out / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    log(f"→ {out}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="預設 features.DEFAULT_ENCODER")
    a = ap.parse_args()
    run(a.model)
