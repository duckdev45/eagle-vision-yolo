"""判斷依據：線上探針的遮擋法熱區＋佐證框（SPEC §11）。

回答的問題不是「模型準不準」，是「**模型是看哪裡決定的**」：遮一格 → 重新編碼 → 看答案掉多少。
佐證框刻意用 Gemini evidence 的同一份格式（[xMin,yMin,xMax,yMax]，0~1000，相對原圖），
才能跟人標框、Gemini 框算一致率（`iou`）。

    uv run --extra train src/explain.py --split v45     # 批次算熱區快取，操作台與複核頁讀它

2026-10-09：convnext 微調模型的 Grad-CAM 路徑（`--ckpt`）隨 finetune 實驗線移除，見 git 歷史。
"""

from __future__ import annotations

import argparse

import numpy as np

from core import model_registry as registry
from core import paths


def to_1000(b):
    """與 Gemini evidence 同格式：[xMin,yMin,xMax,yMax]，0~1000 整數。"""
    return [round(min(1000, max(0, v * 1000))) for v in b]


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def overlay(crop, cam: np.ndarray, box, label: str):
    """左：原圖＋框，右：熱區疊圖。並排才看得出「它在看哪」。

    cam 可以比圖小（遮擋法只有 grid×grid），會先拉到圖的尺寸。
    """
    from PIL import Image, ImageDraw

    base = crop.convert("RGB")
    w, h = base.size
    if cam.shape != (h, w):
        cam = (
            np.asarray(
                Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )
    # 熱區上色：藍→紅，用 cam 當 alpha，暗處保留原圖
    c = np.clip(cam, 0, 1)[..., None]
    heat = np.concatenate([c, np.clip(1.2 - 2 * c, 0, 1) * 0.4, np.clip(1 - 2 * c, 0, 1)], axis=2)
    arr = np.asarray(base, dtype=np.float32) / 255.0
    mixed = np.clip(arr * (1 - 0.55 * c) + heat * 0.55 * c, 0, 1)
    right = Image.fromarray((mixed * 255).astype(np.uint8))

    left = base.copy()
    if box[2] > box[0]:
        d = ImageDraw.Draw(left)
        d.rectangle([box[0] * w, box[1] * h, box[2] * w, box[3] * h], outline=(255, 60, 60), width=3)
    out = Image.new("RGB", (w * 2 + 6, h), (20, 20, 20))
    out.paste(left, (0, 0))
    out.paste(right, (w + 6, 0))
    return out


# --- 線性探針的依據：遮擋法 -------------------------------------------------
# 上線的是「凍結 SigLIP + LogisticRegression」，不是 ft-report-* 那個 convnext。
# Grad-CAM 要挑一層 feature map，ViT 的 attention pool 之後那個恆等式不成立
# （與檔頭 gradcam() 註解同一個理由，只是 ConvNeXt 還能靠梯度繞過去，ViT 不行）。
# 遮擋法只問「拿掉這一塊，這個答案掉多少」：不看架構、換 encoder 不必改，
# 而且量的直接就是因果，不是相關。代價：解析度只有 grid×grid，一張要 grid²+1 次 forward。
GRID = 6
# 與 prepare.FILL 同一個灰。遮擋用訓練時看過的顏色，才不會是「沒看過的東西」在扣分。
FILL_GRAY = (127, 127, 127)


def encoder(model_key: str | None = None):
    import open_clip
    import torch

    import features

    model_key = model_key or registry.encoder()

    name, pretrained = features.MODELS[model_key]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    return model.to(dev).eval(), pre, dev


def embed(enc, ims) -> np.ndarray:
    import torch

    model, pre, dev = enc
    x = torch.stack([pre(i) for i in ims]).to(dev)
    with torch.no_grad():
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)


def probe_cam(im, clf, enc, grid: int = GRID, win: int = 2):
    """回傳 (cam grid×grid, 預測類別, 信心, 最關鍵那塊的比例框)。

    cam[y][x] = 把以 (x,y) 為左上、win×win 格大的那塊塗灰之後，這個答案掉多少。
    win 要大於步距：只遮 1/36 張圖時 SigLIP 的全域 embedding 幾乎不動，
    熱區會是一片雜訊（實測過）。遮 2×2 格（約 11% 畫面）訊號才出得來。

    框不取「所有高熱格的外接矩形」——高熱格是散的，那個矩形每次都等於整張圖。
    取單一最高格的那個視窗：意思是「拿掉這一塊，它最不敢答」，這句話才成立。
    """
    from PIL import ImageDraw

    im = im.convert("RGB")
    w, h = im.size
    cw, ch = w / grid, h / grid
    rects = [
        (x, y, (x * cw, y * ch, min(w, (x + win) * cw), min(h, (y + win) * ch)))
        for y in range(grid)
        for x in range(grid)
    ]
    cells = []
    for _, _, r in rects:
        c = im.copy()
        ImageDraw.Draw(c).rectangle(r, fill=FILL_GRAY)
        cells.append(c)
    ims = [im, *cells]
    # 分批 forward：so400m 一次 37 張會把記憶體吃滿，系統開始 swap 就整個停住
    p = clf.predict_proba(np.concatenate([embed(enc, ims[i : i + 8]) for i in range(0, len(ims), 8)]))
    k = int(p[0].argmax())
    cam = np.clip(p[0, k] - p[1:, k], 0, None).reshape(grid, grid)
    top = int(np.argmax(cam))
    x0, y0, x1, y1 = rects[top][2]
    m = cam.max()
    return ((cam / m if m > 0 else cam), clf.classes_[k], float(p[0, k]), (x0 / w, y0 / h, x1 / w, y1 / h))


def cam_cache(split_name: str, model_key: str | None = None):
    model_key = model_key or registry.encoder(split_name)
    return paths.FEATURES / f"cam-{model_key}-{split_name}-g{GRID}.npz"


def load_cams(split_name: str, model_key: str | None = None) -> dict:
    """{fileId: (cam grid×grid, pred, conf, box)}。沒跑過批次就回空的。"""
    f = cam_cache(split_name, model_key)
    if not f.exists():
        return {}
    z = np.load(f, allow_pickle=True)
    return {
        i: (c, p, float(cf), tuple(b))
        for i, c, p, cf, b in zip(z["fileIds"], z["cams"], z["preds"], z["confs"], z["boxes"])
    }


def run_probe(
    split_name: str | None = None,
    model_key: str | None = None,
    grid: int = GRID,
    force: bool = False,
    log=print,
) -> str:
    """把所有日報照的熱區一次算完存起來，操作台就不必每張等編碼器。

    一張 grid²+1 次 forward，批次跑約 0.5 秒；593 張約 5 分鐘。
    存的是 grid×grid 的小陣列（593×36 個 float，~90KB），疊圖是看的時候才畫。
    """
    from PIL import Image

    split_name = split_name or registry.current()
    model_key = model_key or registry.encoder(split_name)
    clf = registry.load_probe(split_name, model_key)
    enc = encoder(model_key)

    out = cam_cache(split_name, model_key)
    have = {} if force else load_cams(split_name, model_key)
    files = sorted(paths.IMAGES.glob("*.jpg"))
    todo = [p for p in files if p.stem not in have]
    log(f"{len(files)} 張，需新算 {len(todo)}")

    def save():
        ids = sorted(have)
        np.savez(
            out,
            fileIds=np.array(ids),
            cams=np.stack([have[i][0] for i in ids]).astype(np.float32),
            preds=np.array([have[i][1] for i in ids]),
            confs=np.array([have[i][2] for i in ids], dtype=np.float32),
            boxes=np.array([have[i][3] for i in ids], dtype=np.float32),
        )

    for i, p in enumerate(todo, 1):
        cam, pred, conf, box = probe_cam(Image.open(p).convert("RGB"), clf, enc, grid)
        have[p.stem] = (cam, pred, conf, box)
        if i % 25 == 0:
            save()  # 中途存檔：被砍掉重跑時只補沒算的
            log(f"  {i}/{len(todo)}")

    save()
    log(f"寫入 {out}")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--probe", action="store_true", help="批次算線上那顆探針的熱區（預設行為，保留旗標給舊指令）"
    )
    ap.add_argument("--split", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_probe(a.split, force=a.force)
