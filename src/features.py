"""凍結編碼器抽 embedding，存 npz 快取。

抽一次可重複使用：改標籤規則、改切分、換分類器都不必重抽。
順帶產物——這批 embedding 可直接拿去做「檢索相似歷史照片塞進 prompt」。

    uv run --extra train src/features.py [--model so400m]   # 預設 DEFAULT_ENCODER
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import paths

MODELS = {
    # open_clip 名稱 → (model, pretrained)
    "siglip": ("ViT-B-16-SigLIP", "webli"),
    "clip": ("ViT-B-16", "openai"),
    # 2026-09-06 encoder 基準用：384px 解析版（工地照的小缺失與密集紋理吃解析度）、
    # SO400M（大四倍的線性探針天花板候選）。mps 上都能跑，384 版約 2.25× 時間。
    "siglip384": ("ViT-B-16-SigLIP-384", "webli"),
    "so400m": ("ViT-SO400M-14-SigLIP-384", "webli"),
}


# 工種分類器用哪個編碼器——**全專案唯一的設定點**。新 split 會把當下的值寫進 split 檔
# （`encoder` 欄），之後 train/evaluate/explain/操作台/推論都從 split 讀，不再各寫一份。
# v41 以前的 split 沒有這欄 → 一律視為 LEGACY_ENCODER（舊 siglip 模型照常可用）。
# 2026-09-27 換 so400m：同卷 v40 top1 0.852→0.885、CV +4.5pt（CI [+2.9,+6.3]）；
# 代價是 CPU ~410ms/張（siglip ~30ms）、權重 ~1.7GB。
DEFAULT_ENCODER = "so400m"
LEGACY_ENCODER = "siglip"


SRC = {"report": ("images", ""), "legacy": ("legacy_images", "legacy-")}


def extract(
    model_key: str = "siglip", batch: int = 32, src: str = "report", force: bool = False, log=print
) -> str:
    import open_clip  # 重相依，用到才載
    import torch
    from PIL import Image

    paths.ensure_dirs()
    img_dir = {"report": paths.IMAGES, "legacy": paths.LEGACY_IMAGES}[src]
    out = paths.FEATURES / f"{SRC[src][1]}{model_key}.npz"
    files = sorted(p for p in img_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"{img_dir} 是空的，先跑 prepare.py")

    have: dict[str, np.ndarray] = {}
    if out.exists() and not force:
        z = np.load(out, allow_pickle=True)
        have = dict(zip(z["fileIds"].tolist(), z["emb"]))
    todo = [f for f in files if f.stem not in have]
    log(f"{len(files)} 張，需新抽 {len(todo)}")

    if todo:
        name, pretrained = MODELS[model_key]
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        model = model.to(dev).eval()
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            x = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in chunk]).to(dev)
            with torch.no_grad():
                v = model.encode_image(x)
                v = v / v.norm(dim=-1, keepdim=True)
            for f, e in zip(chunk, v.cpu().numpy().astype(np.float32)):
                have[f.stem] = e
            log(f"  {min(i + batch, len(todo))}/{len(todo)}")

    ids = sorted(have)
    np.savez(out, fileIds=np.array(ids), emb=np.stack([have[i] for i in ids]))
    log(f"寫入 {out}")
    return str(out)


def extract_crops(model_key: str = "siglip", batch: int = 32, force: bool = False, log=print) -> str:
    """人標的框 → 裁切 → embedding。id 是 `{fileId}#{i}`，只當**額外的訓練樣本**。

    為什麼這樣用框：線性探針吃的是整張圖的 embedding，框塞不進去。但把框裡那塊
    單獨編碼一次、掛同一個標籤，等於告訴模型「這個類別長這樣，不是整張場景長這樣」。
    「1F人行道地磚貼飾」那張整張餵下去，最強的訊號是右側的行道樹（實測熱區壓在那）；
    只餵框住地磚的那塊，訊號就只剩地磚。

    刻意**只進 train**：測試集要維持「產品實際收到什麼」，那裡沒有人畫的框。
    """
    import open_clip  # 重相依，用到才載
    import torch
    from PIL import Image

    from labels import load_boxes

    paths.ensure_dirs()
    boxes = load_boxes()
    out = paths.FEATURES / f"crops-{model_key}.npz"
    if not boxes:
        log("還沒有人標框（複核佇列的「✎ 標框」）")
        return str(out)

    have: dict[str, np.ndarray] = {}
    if out.exists() and not force:
        z = np.load(out, allow_pickle=True)
        have = dict(zip(z["fileIds"].tolist(), z["emb"]))

    todo = []
    for fid, bs in boxes.items():
        src = paths.IMAGES / f"{fid}.jpg"
        if not src.exists():
            continue
        for i, b in enumerate(bs):
            key = f"{fid}#{i}"
            if key not in have:
                todo.append((key, src, b))
    log(f"{sum(len(b) for b in boxes.values())} 個框，需新抽 {len(todo)}")

    if todo:
        name, pretrained = MODELS[model_key]
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        model = model.to(dev).eval()
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            ims = []
            for _, src, b in chunk:
                im = Image.open(src).convert("RGB")
                w, h = im.size
                ims.append(im.crop((b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h)))
            x = torch.stack([preprocess(im) for im in ims]).to(dev)
            with torch.no_grad():
                v = model.encode_image(x)
                v = v / v.norm(dim=-1, keepdim=True)
            for (key, _, _), e in zip(chunk, v.cpu().numpy().astype(np.float32)):
                have[key] = e
            log(f"  {min(i + batch, len(todo))}/{len(todo)}")

    # 框刪掉之後對應的 embedding 也要消失，否則會拿舊框繼續訓練
    live = {f"{f}#{i}" for f, bs in boxes.items() for i in range(len(bs))}
    have = {k: v for k, v in have.items() if k in live}
    ids = sorted(have)
    np.savez(
        out,
        fileIds=np.array(ids),
        emb=np.stack([have[i] for i in ids]) if ids else np.zeros((0, 768), dtype=np.float32),
    )
    log(f"寫入 {out}（{len(ids)} 個裁切）")
    return str(out)


def load(model_key: str = "siglip") -> tuple[list[str], np.ndarray]:
    """合併所有資料源的 embedding（fileId 是 uuid，不會撞）。"""
    ids: list[str] = []
    embs = []
    for prefix in ("", "legacy-", "crops-"):
        f = paths.FEATURES / f"{prefix}{model_key}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            if not len(z["fileIds"]):  # 空的裁切檔寫死 768 維，會跟 so400m 的 1152 維接不起來
                continue
            ids += z["fileIds"].tolist()
            embs.append(z["emb"])
    if not embs:
        raise SystemExit("沒有任何特徵檔，先跑 features.py")
    return ids, np.concatenate(embs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_ENCODER, choices=list(MODELS))
    ap.add_argument("--src", default="report", choices=["report", "legacy", "all"])
    ap.add_argument("--force", action="store_true", help="影像重做過就要加這個")
    ap.add_argument("--crops", action="store_true", help="改成抽人標框裡那塊的 embedding")
    a = ap.parse_args()
    if a.crops:
        extract_crops(a.model, force=a.force)
    else:
        for s in list(SRC) if a.src == "all" else [a.src]:
            extract(a.model, src=s, force=a.force)
