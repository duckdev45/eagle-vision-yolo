"""階段 C 之一：線性探針。

凍結編碼器 → embedding → LogisticRegression。
價值不在準確率，在診斷：分不開通常代表 labels.yaml 有矛盾，不是模型不夠大。

    uv run --extra train src/train.py --probe [--split v1] [--model siglip]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import features  # noqa: E402
import paths  # noqa: E402
import split as split_mod  # noqa: E402


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


def probe(split_name: str = "v1", model_key: str = "siglip", C: float = DEFAULT_C, log=print):
    from sklearn.linear_model import LogisticRegression

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
        json.dumps({"top1": acc, "classes": list(clf.classes_), "C": C}, ensure_ascii=False))
    return clf, acc, (xte, yte, ids_te)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", default=True)
    ap.add_argument("--split", default="v1")
    ap.add_argument("--model", default="siglip")
    ap.add_argument("-C", type=float, default=DEFAULT_C)
    a = ap.parse_args()
    probe(a.split, a.model, a.C)
