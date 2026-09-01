# core/inference_utils.py
"""模型推理的計算服務（ML Inference Utilities）。

V2.0 草稿期這裡回傳 np.random 假 embedding；2026-09-01 起接真實檔案：
embedding 在 data/derived/features/{model_key}.npz（src/features.py 產出），
分類器在 models/probe-{model_key}-{split}.pkl（src/train.py 產出）。

推理前處理的契約在 src/predict.py（與訓練的 eval_tf 逐步一致）——本檔只做
「讀現成的、算分數」，不重新實作特徵抽取。
"""
from __future__ import annotations

import sys as _sys
import os as _os

for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json  # noqa: E402
import pickle  # noqa: E402
from typing import Any, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

import paths  # noqa: E402


def load_model_classifier(model_key: str = "siglip", split_name: str = "") -> Optional[Any]:
    """載入線性探針（sklearn clf）。檔案不存在回 None——呼叫端要自己擋。"""
    p = paths.MODELS / f"probe-{model_key}-{split_name}.pkl"
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)["clf"]


def calculate_embeddings(file_ids: list[str], model_key: str = "siglip") -> np.ndarray:
    """{fileId → embedding}。特徵不存在就回空陣列（呼叫端要擋維度）。"""
    p = paths.FEATURES / f"{model_key}.npz"
    if not p.exists():
        return np.zeros((0, 0))
    z = np.load(p, allow_pickle=True)
    idx = {f: n for n, f in enumerate(z["fileIds"].tolist())}
    rows = [idx[f] for f in file_ids if f in idx]
    return z["emb"][rows] if rows else np.zeros((0, 0))


def scores(model_key: str = "siglip", split_name: str = "") -> tuple[dict, set]:
    """{fileId: (預測, 信心, 邊際)}, {測試集 fileId}——與 review 佇列同一套算法。

    用邊際不用信心：信心 0.9 但第二名 0.85 = 在兩類間猶豫；0.5 vs 0.05 = 篤定。
    """
    try:
        clf = load_model_classifier(model_key, split_name)
        if clf is None:
            return {}, set()
        z = np.load(paths.FEATURES / f"{model_key}.npz", allow_pickle=True)
        test = set(json.loads((paths.SPLITS / f"{split_name}.json").read_text())["test"])
    except (FileNotFoundError, KeyError):
        return {}, set()
    p = clf.predict_proba(z["emb"])
    top = np.sort(p, axis=1)
    return ({f: (clf.classes_[i], float(top[n, -1]), float(top[n, -1] - top[n, -2]))
             for n, (f, i) in enumerate(zip(z["fileIds"].tolist(), p.argmax(1)))}, test)


def get_heatmaps(file_id: str, model_key: str = "siglip",
                 split_name: str = "") -> Tuple[Optional[np.ndarray], Optional[Any],
                                                 Optional[float], Optional[str]]:
    """遮擋法熱區（explanation）。

    實體在 src/explain.py（probe_cam：遮一格 → 重編碼 → 看答案掉多少）。
    這裡是薄轉發：模型訓練依賴（torch/open_clip）還沒裝的環境照樣 import 本檔。
    """
    try:
        from explain import load_cams
    except ImportError:
        return (None, None, None, None)
    cams = load_cams(split_name, model_key)
    return (cams.get(file_id), None, None, None)


if __name__ == "__main__":
    print("--- Inference utils self-check ---")
    print(f"features dir: {paths.FEATURES} exists={paths.FEATURES.exists()}")
    print(f"models dir:   {paths.MODELS} exists={paths.MODELS.exists()}")
    import split as split_mod
    cur = split_mod.current() if paths.SPLITS.exists() else "v1"
    sc, test = scores("siglip", cur)
    print(f"split={cur} · embeddings scored: {len(sc)} · test ids: {len(test)}")
