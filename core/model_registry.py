"""模型版本登記處：哪一版上線、它用哪個編碼器、它的檔案在哪。

一個版本 = 三個檔，檔名拼法只在這裡寫一次：

    data/derived/splits/{name}.json        切分與標籤（含 `encoder` 欄）
    models/probe-{encoder}-{name}.pkl      分類器（線性探針）
    data/derived/features/{encoder}.npz    PMS 圖像特徵（同編碼器的版本共用）

以前這套規則住在 `src/split.py`（切分 CLI），core 的分流、考卷、工作台為了問一句
「現在是哪一版」都得反向 import 腳本層；探針路徑與「舊 split 缺 encoder 欄＝siglip」
又在 15 支檔案裡各拼一份。之後要改檔名或換編碼器預設，只改這支。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from core import paths

# 工種分類器用哪個編碼器——**全專案唯一的設定點**。新 split 會把當下的值寫進 split 檔
# （`encoder` 欄），之後 train/evaluate/explain/操作台/推論都從 split 讀，不再各寫一份。
# v41 以前的 split 沒有這欄 → 一律視為 LEGACY_ENCODER（舊 siglip 模型照常可用）。
# 2026-09-27 換 so400m：同卷 v40 top1 0.852→0.885、CV +4.5pt（CI [+2.9,+6.3]）；
# 代價是 CPU ~410ms/張（siglip ~30ms）、權重 ~1.7GB。
DEFAULT_ENCODER = "so400m"
LEGACY_ENCODER = "siglip"

# 線性探針的正則化強度。訓練（src/train.py）與分流的 out-of-fold（core/routing.py）必須同一個值，
# 否則「規則＝模型」的一致率量的是另一顆模型。
# C=1 對 768 維 SigLIP embedding 是過度正則化：實測 v7-fix 上 C=1 只有 0.737，
# 用 GroupKFold（同工地同日不跨 fold）在 train 上選出 C=300 → test 0.847。
# 資料量一變就要重選（`train.py --tune-c`），別把它當常數看。
PROBE_C = 300.0

# 操作台與解釋工具要看哪一組。存成檔案而不是原始碼常數，操作台的按鈕才改得動
# ——否則重跑完還要手改程式碼，那個手動步驟一定有人忘記
# （實測踩過：畫面上一直顯示 v1 的 0.712，實際模型已經 0.847）。
FALLBACK = "v8"


def current() -> str:
    f = paths.SPLITS / "CURRENT"
    return f.read_text().strip() if f.exists() else FALLBACK


def set_current(name: str) -> None:
    """切換操作台指向的模型。刻意是獨立一步，不塞進重訓流程——
    換掉所有人看到的答案是個決定，不該是跑完訓練的副作用（自動切換只經 core/promotion.py）。"""
    paths.ensure_dirs()
    (paths.SPLITS / "CURRENT").write_text(name.strip())


def split_path(name: str | None = None) -> Path:
    return paths.SPLITS / f"{name or current()}.json"


def load_split(name: str | None = None) -> dict:
    """讀 split 檔；不存在就丟 FileNotFoundError（缺檔要由呼叫端決定怎麼降級）。"""
    return json.loads(split_path(name).read_text(encoding="utf-8"))


def encoder_of(split: dict) -> str:
    """split 檔說了算；缺欄＝v41 以前的舊 split＝LEGACY_ENCODER。"""
    return split.get("encoder") or LEGACY_ENCODER


def encoder(name: str | None = None) -> str:
    f = split_path(name)
    return encoder_of(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else LEGACY_ENCODER


def probe_path(name: str | None = None, encoder_key: str | None = None) -> Path:
    """`models/probe-{encoder}-{split}.pkl`。encoder_key 只給「刻意用別的編碼器重訓同一份 split」的實驗。"""
    name = name or current()
    return paths.MODELS / f"probe-{encoder_key or encoder(name)}-{name}.pkl"


# 特徵檔的資料源前綴：日報（無前綴，PMS 主檔）、舊 pptx（G1/G2 黃金集用）、人標框裁切。
# QMS 實驗線 2026-10-09 移除，qms-*.npz 已歸檔（data/archive/README.md）。
FEATURE_SOURCES = ("", "legacy-", "crops-")


def feature_path(encoder_key: str, source: str = "") -> Path:
    """`features/{source}{encoder}.npz`。不給 source＝PMS 日報主檔。"""
    assert source in FEATURE_SOURCES, source
    return paths.FEATURES / f"{source}{encoder_key}.npz"


def load_features(encoder_key: str) -> tuple[list[str], np.ndarray]:
    """合併所有資料源的 embedding（fileId 是 uuid，不會撞）。"""
    ids: list[str] = []
    embs = []
    for source in FEATURE_SOURCES:
        f = feature_path(encoder_key, source)
        if f.exists():
            z = np.load(f, allow_pickle=True)
            if not len(z["fileIds"]):  # 空的裁切檔寫死 768 維，會跟 so400m 的 1152 維接不起來
                continue
            ids += z["fileIds"].tolist()
            embs.append(z["emb"])
    if not embs:
        raise SystemExit("沒有任何特徵檔，先跑 features.py")
    return ids, np.concatenate(embs)


def defect_probe_path(encoder_key: str) -> Path:
    """缺失旗標探針（src/defect_probe.py 訓練、工作台讀）。跟著編碼器走，不跟 split。"""
    return paths.MODELS / f"defect-probe-{encoder_key}.pkl"


def load_probe(name: str | None = None, encoder_key: str | None = None) -> Any:
    """分類器本體。pickle 只讀本機自己訓練出的檔案（受信任產物），不接外部上傳。"""
    with probe_path(name, encoder_key).open("rb") as fh:
        return pickle.load(fh)["clf"]
