"""用 ONNX 對整批照片跑推論，結果存 csv 供操作台讀。

為什麼不在操作台直接載 PyTorch：streamlit 每次互動都會重跑整個腳本，
載 timm + torch 要好幾秒；而且操作台不該逼使用者裝 `--extra train`。
ONNX 單張 CPU 19ms，2000 張約 40 秒，跑一次存起來就好。

    uv run --extra train src/predict.py --ckpt backbone-qms --src qms
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import paths  # noqa: E402

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path) -> np.ndarray:
    """必須與 finetune 的 eval_tf 逐步一致：resize 256 → center crop 224 → normalize。"""
    im = Image.open(path).convert("RGB")
    short = int(SIZE * 1.14)
    w, h = im.size
    scale = short / min(w, h)
    im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = im.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    im = im.crop((left, top, left + SIZE, top + SIZE))
    x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


def run(ckpt: str = "backbone-qms", src: str = "qms", batch: int = 32, log=print) -> str:
    import onnxruntime as ort

    img_dir = paths.QMS_IMAGES if src == "qms" else paths.IMAGES
    out = paths.MODELS / f"{ckpt}-preds-{src}.csv"
    classes = json.loads((paths.MODELS / f"{ckpt}-classes.json").read_text())
    sess = ort.InferenceSession(str(paths.MODELS / f"{ckpt}.onnx"))

    files = sorted(img_dir.glob("*.jpg"))
    rows = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        x = np.stack([preprocess(f) for f in chunk])
        logits = sess.run(None, {"image": x})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        prob = e / e.sum(1, keepdims=True)
        for f, p in zip(chunk, prob):
            k = int(p.argmax())
            rows.append({"fileId": f.stem, "pred": classes[k], "conf": round(float(p[k]), 4)})
        if (i // batch) % 20 == 0:
            log(f"  {min(i + batch, len(files))}/{len(files)}")
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"{len(rows)} 張 → {out}")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="backbone-qms")
    ap.add_argument("--src", default="qms", choices=["qms", "report"])
    a = ap.parse_args()
    run(a.ckpt, a.src)
