"""編碼器設定：split 檔說了算，舊 split 缺欄＝siglip；操作台預測走工項融合。"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

import features
import paths
import split as split_mod
from core import pms_review


class HalfProbe:
    """emb 1 → 微偏打底；-1 → 強烈偏油漆；0.5 → 邊際夠大但信心低於轉人工門檻。"""

    classes_ = np.array(["泥作-打底", "油漆-塗裝", "其他"])

    def predict_proba(self, emb):
        table = {1.0: [0.55, 0.40, 0.05], -1.0: [0.05, 0.90, 0.05], 0.5: [0.51, 0.245, 0.245]}
        return np.array([table[float(v)] for v in emb[:, 0]])


def test_legacy_split_without_encoder_field_stays_siglip(pms_env):
    (paths.SPLITS / "vold.json").write_text(json.dumps({"train": [], "test": []}))
    assert split_mod.encoder("vold") == features.LEGACY_ENCODER == "siglip"
    assert split_mod.probe_path("vold").name == "probe-siglip-vold.pkl"
    assert split_mod.encoder("missing") == "siglip"


def test_workbench_scores_use_split_encoder_and_work_item_fusion(pms_env):
    man = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False)
    man["dailyReportInfoId"] = ["r1", "r1", "", "", "", "", "", "", ""]
    man.loc[man.fileId == "b", "title"] = "打底施作"
    man.to_csv(paths.MANIFEST, index=False)
    enc = "so400m"
    (paths.SPLITS / "vnew.json").write_text(
        json.dumps({"train": [], "test": ["a", "b", "u1"], "encoder": enc})
    )
    (paths.SPLITS / "CURRENT").write_text("vnew")
    (paths.MODELS / f"probe-{enc}-vnew.pkl").write_bytes(pickle.dumps({"clf": HalfProbe()}))
    np.savez(
        paths.FEATURES / f"{enc}.npz",
        fileIds=np.array(["a", "b", "u1"]),
        emb=np.array([[1.0], [-1.0], [0.5]]),
    )
    scores = pms_review.local_model()["scores"]
    assert scores["a"][0] == "油漆-塗裝"  # 兄弟照 b 把 a 拉到油漆
    assert scores["u1"][0] == "泥作-打底" and round(scores["u1"][1], 2) == 0.51  # 單張工項不融合
    df, _ = pms_review.snapshot()
    reason = df.set_index("fileId").loc["u1", "reviewReason"]
    assert "照片模型信心低" in reason and "照片模型難分" not in reason
