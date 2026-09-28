"""階段 C 之一：線性探針。

凍結編碼器 → embedding → LogisticRegression。
價值不在準確率，在診斷：分不開通常代表 labels.yaml 有矛盾，不是模型不夠大。

    uv run --extra train src/train.py --probe [--split v1] [--model so400m]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import features
import paths
import split as split_mod


def dataset(split_name: str, model_key: str):
    """標籤直接讀 split 裡的 labels，不管這批來自日報還是 QMS。"""
    sp = split_mod.load(split_name)
    ids, emb = features.load(model_key)
    idx = {f: i for i, f in enumerate(ids)}
    cls = sp["labels"]

    def take(file_ids):
        keep = [f for f in file_ids if f in idx and f in cls]
        return np.stack([emb[idx[f]] for f in keep]), np.array([cls[f] for f in keep]), keep

    return take(sp["train"]), take(sp["test"])


# C=1 對 768 維 SigLIP embedding 是過度正則化：實測 v7-fix 上 C=1 只有 0.737，
# 用 GroupKFold（同工地同日不跨 fold）在 train 上選出 C=300 → test 0.847。
# 資料量一變就要重選，別把它當常數看。
DEFAULT_C = 300.0


def probe(split_name: str = "v1", model_key: str | None = None, C: float = DEFAULT_C, log=print):
    from sklearn.linear_model import LogisticRegression

    model_key = model_key or split_mod.encoder(split_name)

    (xtr, ytr, _), (xte, yte, ids_te) = dataset(split_name, model_key)
    log(f"train {xtr.shape} / test {xte.shape} / {len(set(ytr))} 類")
    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    clf.fit(xtr, ytr)

    paths.ensure_dirs()
    out = paths.MODELS / f"probe-{model_key}-{split_name}.pkl"
    with out.open("wb") as f:
        pickle.dump({"clf": clf, "split": split_name, "encoder": model_key}, f)

    acc = float((clf.predict(xte) == yte).mean()) if len(yte) else float("nan")
    log(f"test top-1 = {acc:.3f} → {out}")
    (paths.MODELS / f"probe-{model_key}-{split_name}.json").write_text(
        json.dumps({"top1": acc, "classes": list(clf.classes_), "C": C}, ensure_ascii=False)
    )
    return clf, acc, (xte, yte, ids_te)


def _group_map() -> dict[str, str]:
    """fileId → 案場×日期（PMS）；legacy 照全部同一組（它們永遠只在 train 側）。"""
    df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
    return {r.fileId: f"{r.constrId}|{r.reportDate}" for r in df.itertuples()}


def tune_c(
    split_name: str = "v1",
    model_key: str | None = None,
    cs: list[float] | None = None,
    folds: int = 5,
    log=print,
) -> float:
    """GroupKFold 重選 C——DEFAULT_C 的註釋就是「資料量一變就要重選」。

    v7 時代（train ~400 張、9 類）選出 C=300 之後，資料長了三倍、類別拆了五次，
    一直沒重選過。折用 GroupKFold 按 案場×日期——與 split 鐵律同一條，同工地
    同日不跨 fold，否則選出的 C 會被近重複照灌水。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    cs = cs or [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
    model_key = model_key or split_mod.encoder(split_name)
    (xtr, ytr, ids_tr), _ = dataset(split_name, model_key)
    gmap = _group_map()
    groups = [gmap.get(f, "legacy") for f in ids_tr]
    n_splits = min(folds, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(xtr, ytr, groups))
    log(f"C 值掃描：train {len(ytr)} 張 / {len(set(groups))} 群組 / {n_splits} 折")
    best_c, best_f1 = None, -1.0
    for c in cs:
        f1s = []
        for tr_i, va_i in splits:
            clf = LogisticRegression(max_iter=2000, C=c, class_weight="balanced")
            clf.fit(xtr[tr_i], ytr[tr_i])
            f1s.append(f1_score(ytr[va_i], clf.predict(xtr[va_i]), average="macro", zero_division=0))
        m = float(np.mean(f1s))
        marker = ""
        if m > best_f1:
            best_c, best_f1 = c, m
            marker = "  ← best"
        log(f"  C={c:<8} macroF1 {m:.4f}{marker}")
    log(f"→ 建議 C={best_c}（現行 DEFAULT_C={DEFAULT_C}）")
    return best_c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", default=True)
    ap.add_argument("--split", default="v1")
    ap.add_argument("--model", default=None, help="預設讀 split 檔的 encoder 欄")
    ap.add_argument("-C", type=float, default=DEFAULT_C)
    ap.add_argument("--tune-c", action="store_true", help="GroupKFold 掃 C，不訓練不存檔")
    a = ap.parse_args()
    if a.tune_c:
        tune_c(a.split, a.model)
    else:
        probe(a.split, a.model, a.C)
