"""判斷依據：Grad-CAM 熱區 + 佐證框（SPEC §11）。

回答的問題不是「模型準不準」，是「**模型是看哪裡決定的**」。

為什麼要有這支：
- 給人看的信任線索。「AI 說這是泥作-打底粉光，因為它在看這一塊牆面」比一個
  0.92 的數字有說服力，而且**是模型真正的依據**，不是另一個模型（Gemini）的猜測。
- 抓洩漏。分數高但學錯東西是這個專案最大的風險：四角遮罩、工人衣服、鷹架、
  浮水印殘影都可能成為捷徑。`cornerMass` 把「熱區有多少落在遮罩角落」量化，
  比人工翻 99 張可靠。
- 佐證框輸出**刻意用 Gemini evidence 的同一份格式**（[xMin,yMin,xMax,yMax]，
  0~1000，相對原圖），才能直接餵進同一個前端、同一份 anno 契約，也才能跟
  Gemini 的框算一致率。

    uv run --extra train src/explain.py --ckpt ft-report-scratch
    uv run --extra train src/explain.py --ckpt ft-report-scratch --set all --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import paths
import split as split_mod

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 佐證框：取熱區最大值的這個比例當門檻，再框住連通的高熱區。
# 0.5 是常規做法（CAM 文獻的 20% 太寬，實測會框掉整張）。
BOX_THRESH = 0.5
# 四角遮罩的幾何，必須與 prepare.py 的 CORNER_W/H 一致，否則洩漏指標算錯。
CORNER_W, CORNER_H = 0.30, 0.12


def _img_path(file_id: str):
    p = paths.IMAGES / f"{file_id}.jpg"
    return p if p.exists() else paths.QMS_IMAGES / f"{file_id}.jpg"


def eval_crop(im):
    """與 finetune.py 的 eval_tf 逐步一致：Resize(256) → CenterCrop(224)。

    同時回傳這個 crop 在原圖裡的相對位置，佐證框才能換算回原圖座標
    （不換算就無法跟 Gemini 的框比對，它給的是原圖座標）。
    """
    from PIL import Image

    w0, h0 = im.size
    short = int(SIZE * 1.14)
    scale = short / min(w0, h0)
    im2 = im.resize((round(w0 * scale), round(h0 * scale)), Image.BILINEAR)
    w, h = im2.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    crop = im2.crop((left, top, left + SIZE, top + SIZE))
    # crop 在原圖中的比例框（0~1）
    box01 = (left / w, top / h, (left + SIZE) / w, (top + SIZE) / h)
    return crop, box01


def to_tensor(crop):
    import torch

    x = (np.asarray(crop, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


def load_model(ckpt: str):
    import timm
    import torch

    meta = json.loads((paths.MODELS / f"{ckpt}.json").read_text())
    model = timm.create_model(meta["model"], pretrained=False, num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(paths.MODELS / f"{ckpt}.pt", map_location="cpu"))
    return model.eval(), meta


def gradcam(model, x, cls_idx: int) -> np.ndarray:
    """最後一個 stage 的 Grad-CAM，回傳 0~1 的 224×224 熱區。

    為什麼用梯度而不是經典 CAM：timm 的 ConvNeXt 在 global pool **之後**還有一層
    LayerNorm，`sum_k w_k·featuremap_k` 那個恆等式不成立。梯度版不管架構都對。
    """
    import torch.nn.functional as F

    acts, grads = {}, {}

    def fwd(m, i, o):
        acts["v"] = o  # 一定要 return None，否則 hook 會被當成「替換輸出」

    def bwd(m, gi, go):
        grads["v"] = go[0]  # 同上：return 值會被當成新的 grad_input，尺寸不符就炸

    target = model.stages[-1]
    h1 = target.register_forward_hook(fwd)
    h2 = target.register_full_backward_hook(bwd)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, cls_idx].backward()
        a, g = acts["v"], grads["v"]  # (1,C,h,w)
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().numpy()
    finally:
        h1.remove()
        h2.remove()
    m = cam.max()
    return cam / m if m > 0 else cam


def cam_box(cam: np.ndarray, thresh: float = BOX_THRESH):
    """熱區 → 一個框。回傳 crop 內的比例框 (x0,y0,x1,y1)，全 0 表示沒有明顯熱區。"""
    h, w = cam.shape  # 不寫死 SIZE：遮擋法的熱區是 grid×grid，不是 224×224
    ys, xs = np.where(cam >= thresh)
    if not len(xs):
        return (0.0, 0.0, 0.0, 0.0)
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def crop_box_to_orig(b, crop01):
    """crop 內的比例框 → 原圖比例框。crop01 是 crop 在原圖中的比例位置。"""
    cx0, cy0, cx1, cy1 = crop01
    return (
        cx0 + b[0] * (cx1 - cx0),
        cy0 + b[1] * (cy1 - cy0),
        cx0 + b[2] * (cx1 - cx0),
        cy0 + b[3] * (cy1 - cy0),
    )


def to_1000(b):
    """與 Gemini evidence 同格式：[xMin,yMin,xMax,yMax]，0~1000 整數。"""
    return [round(min(1000, max(0, v * 1000))) for v in b]


def deletion_test(model, x, cls_idx: int, b) -> tuple[float, int | None]:
    """把熱區框塗成灰色再問一次 —— 熱區到底是不是真的依據。

    這一步是「解釋可不可信」的檢查，不是「模型準不準」。若遮掉之後信心幾乎不動，
    那張熱圖只是裝飾（模型的依據在別處）；信心大跌或答案翻掉，才代表框對了。
    這是文獻上的 deletion metric，成本只是多一次 forward。
    """
    import torch

    if b[2] <= b[0]:
        return 0.0, None
    x2 = x.clone()
    # 灰色 = 訓練時遮罩用的 (127,127,127)，正規化後的值
    gray = torch.tensor((127 / 255.0 - MEAN) / STD, dtype=x.dtype).view(1, 3, 1, 1)
    x0, y0, x1, y1 = (round(v * SIZE) for v in b)
    x2[:, :, y0:y1, x0:x1] = gray
    with torch.no_grad():
        p = torch.softmax(model(x2), 1)[0]
    k2 = int(p.argmax())
    return float(p[cls_idx]), (None if k2 == cls_idx else int(k2))


def corner_mass(cam: np.ndarray) -> float:
    """熱區有多少比例落在四角遮罩裡。

    遮罩是灰色方塊，本身沒有工項資訊。若這個值明顯高於遮罩的面積佔比
    （4 × 30% × 12% = 14.4%），模型就是在看遮罩邊界猜答案 —— 洩漏。
    """
    cw, ch = int(SIZE * CORNER_W), int(SIZE * CORNER_H)
    total = cam.sum()
    if total <= 0:
        return 0.0
    s = cam[:ch, :cw].sum() + cam[:ch, -cw:].sum() + cam[-ch:, :cw].sum() + cam[-ch:, -cw:].sum()
    return float(s / total)


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
    import split as sm

    model_key = model_key or sm.encoder()

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
    import split as sm

    model_key = model_key or sm.encoder(split_name)
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
    import pickle

    from PIL import Image

    import split as sm

    split_name = split_name or sm.current()
    model_key = model_key or sm.encoder(split_name)
    with (paths.MODELS / f"probe-{model_key}-{split_name}.pkl").open("rb") as f:
        clf = pickle.load(f)["clf"]
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


def run(
    ckpt: str = "ft-report-scratch", which: str = "test", n: int = 0, run_tag: str | None = None, log=print
) -> str:
    from PIL import Image

    model, meta = load_model(ckpt)
    classes = meta["classes"]
    sp = split_mod.load(meta["split"])
    ids = sp["test"] if which == "test" else (sp["train"] + sp["test"])
    ids = [f for f in ids if _img_path(f).exists()]
    if n:
        ids = ids[:n]
    test_set = set(sp["test"])

    # 熱區解釋的是「模型的預測」,不需要真實標籤,所以不因標籤不合而濾掉照片。
    # 但類別集不合就代表這個 checkpoint 與現在的 split 不是同一個世代
    # (實測踩到:ft-report-* 是 labels.yaml v5 改名前訓的,類別名全對不上),
    # 那 top-1 一定是假的,必須大聲講,不能安靜地只剩 3 張還照樣報分數。
    split_classes = set(sp.get("classes") or sp["labels"].values())
    shared = split_classes & set(classes)
    if len(shared) < len(split_classes):
        log(
            f"⚠ 類別集不符:checkpoint {len(classes)} 類、split {len(split_classes)} 類、"
            f"共通只有 {len(shared)} 類。"
        )
        log(f"  checkpoint: {sorted(classes)}")
        log(f"  split:      {sorted(split_classes)}")
        log("  → 熱區照樣正確(它只解釋預測),但 top-1 與 truth 欄不可信,需重訓或改名對齊。")

    gem = gemini_boxes()
    outdir = paths.REPORTS_OUT / f"{date.today():%Y-%m-%d}-explain-{run_tag or ckpt}"
    (outdir / "overlay").mkdir(parents=True, exist_ok=True)

    rows = []
    for i, f in enumerate(ids, 1):
        im = Image.open(_img_path(f)).convert("RGB")
        crop, crop01 = eval_crop(im)
        x = to_tensor(crop)
        import torch

        with torch.no_grad():
            p = torch.softmax(model(x), 1)[0].numpy()
        k = int(p.argmax())
        cam = gradcam(model, x, k)
        b = cam_box(cam)
        b_orig = crop_box_to_orig(b, crop01) if b[2] > b[0] else b
        truth = sp["labels"].get(f)
        pred, conf = classes[k], float(p[k])
        comparable = truth in classes  # 標籤與這個 checkpoint 同世代才能算對錯

        conf_after, flip_to = deletion_test(model, x, k, b)
        g = gem.get(f) or []
        best_iou = max((iou(b_orig, gb) for gb in g), default=None)
        cx, cy = (b_orig[0] + b_orig[2]) / 2, (b_orig[1] + b_orig[3]) / 2
        in_gem = any(gb[0] <= cx <= gb[2] and gb[1] <= cy <= gb[3] for gb in g) if g else None

        rows.append(
            {
                "fileId": f,
                "set": "test" if f in test_set else "train",
                "truth": truth,
                "comparable": comparable,
                "hit": (truth == pred) if comparable else None,
                "pred": pred,
                "conf": round(conf, 4),
                "box": to_1000(b_orig),
                "cornerMass": round(corner_mass(cam), 4),
                "confAfterMaskingBox": round(conf_after, 4),
                "confDrop": round(conf - conf_after, 4),
                "flippedTo": None if flip_to is None else classes[flip_to],
                "geminiBoxes": len(g),
                "iouWithGemini": None if best_iou is None else round(best_iou, 3),
                "camCenterInsideGeminiBox": in_gem,
            }
        )
        # 檔名只算一次並寫進 row：先前是存檔與 contact 各自 format 一次,
        # round(conf,4) 與原始 float 在 0.925 這種邊界會格出 0.92 / 0.93 兩種名字,
        # contact 就有一張圖破圖。單一來源才不會再犯。
        tag = "-" if not comparable else ("O" if truth == pred else "X")
        fname = f"{tag}_猜{pred}_{rows[-1]['conf']:.2f}_{f[:8]}.jpg"
        rows[-1]["file"] = fname
        overlay(crop, cam, b, pred).save(outdir / "overlay" / fname, "JPEG", quality=88)
        if i % 20 == 0:
            log(f"  {i}/{len(ids)}")

    summary = summarize(rows)
    (outdir / "cam_stats.json").write_text(
        json.dumps(
            {
                "checkpoint": ckpt,
                "split": meta["split"],
                "set": which,
                "boxThresh": BOX_THRESH,
                "cornerAreaShare": 4 * CORNER_W * CORNER_H,
                "summary": summary,
                "photos": rows,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    contact_html(rows, outdir, ckpt, summary)
    log(f"依據報告 → {outdir}")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    return str(outdir)


def gemini_boxes() -> dict[str, list[tuple]]:
    """manifest 的 predBoxes（Gemini evidence）→ {fileId: [原圖比例框]}。

    Gemini 給的是 0~1000 相對原圖，正規化成 0~1 才能跟 CAM 的框比。
    """
    import pandas as pd

    if not paths.MANIFEST.exists():
        return {}
    df = pd.read_csv(paths.MANIFEST)
    if "predBoxes" not in df:
        return {}
    out = {}
    for r in df[df.predBoxes.notna()].itertuples():
        try:
            bs = json.loads(r.predBoxes)
        except Exception:
            continue
        got = []
        for b in bs:
            v = b.get("box") if isinstance(b, dict) else None
            if isinstance(v, list) and len(v) == 4:
                got.append(tuple(c / 1000 for c in v))
        if got:
            out[r.fileId] = got
    return out


def summarize(rows: list[dict]) -> dict:
    te = [r for r in rows if r["set"] == "test" and r["comparable"]]
    cm = [r["cornerMass"] for r in rows]
    withg = [r for r in rows if r["geminiBoxes"]]
    hit = [r for r in te if r["hit"]]
    miss = [r for r in te if not r["hit"]]

    def f(xs):
        return round(float(np.mean(xs)), 4) if len(xs) else None

    return {
        "photos": len(rows),
        "comparableTestPhotos": len(te),
        "testTop1": f([r["hit"] for r in te]),
        "cornerMassMean": f(cm),
        "cornerMassMeanCorrect": f([r["cornerMass"] for r in hit]),
        "cornerMassMeanWrong": f([r["cornerMass"] for r in miss]),
        "cornerMassOver30pct": int(sum(1 for v in cm if v > 0.30)),
        "boxAreaMean": f(
            [
                (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]) / 1e6
                for r in rows
                if r["box"][2] > r["box"][0]
            ]
        ),
        "confDropMean": f([r["confDrop"] for r in rows]),
        "predFlipRate": f([r["flippedTo"] is not None for r in rows]),
        "photosWithGeminiBoxes": len(withg),
        "camCenterInsideGeminiBoxRate": f([r["camCenterInsideGeminiBox"] for r in withg]),
        "iouWithGeminiMean": f([r["iouWithGemini"] for r in withg]),
    }


def contact_html(rows, outdir, ckpt: str, summary: dict) -> None:
    """可在瀏覽器翻閱的對照頁。左原圖右熱區，紅=判錯。"""

    def card(r):
        name = r["file"]
        cls = "na" if not r["comparable"] else ("hit" if r["hit"] else "miss")
        warn = ""
        if r["cornerMass"] > 0.30:
            warn = ' <b class="w">角落熱區 %.0f%%</b>' % (r["cornerMass"] * 100)
        gem = ""
        if r["iouWithGemini"] is not None:
            gem = " · 與 Gemini 框 IoU {:.2f}".format(r["iouWithGemini"])
        dele = " · 遮掉框後信心 {:.2f}".format(r["confAfterMaskingBox"])
        if r["flippedTo"]:
            dele += f' <b class="ok">→ 改答 {r["flippedTo"]}</b>'
        gem = dele + gem
        truth = "" if r["hit"] else (" ／ 日報 " + str(r["truth"]))
        return (
            f'<figure class="{cls}"><img src="overlay/{name}" loading="lazy">'
            f"<figcaption><b>{r['pred']}</b> {r['conf']:.2f}{truth}"
            f'<span class="m">{r["set"]}{gem}{warn}</span></figcaption></figure>'
        )

    s = summary
    head = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>判斷依據 · {ckpt}</title><style>
body{{margin:0;background:#111;color:#eee;font-family:"Noto Sans TC","PingFang TC",sans-serif}}
.w{{max-width:1400px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:21px;margin:0 0 4px}} .s{{color:#999;font-size:14px;margin:0 0 18px}}
.kpi{{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 22px}}
.k{{background:#1c1c1b;border:1px solid #333;border-radius:9px;padding:10px 14px;min-width:150px}}
.k i{{display:block;font-style:normal;font-size:11.5px;color:#999;margin-bottom:3px}}
.k b{{font-size:19px;font-variant-numeric:tabular-nums}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px}}
figure{{margin:0;background:#1c1c1b;border:1px solid #333;border-radius:9px;overflow:hidden}}
figure.miss{{border-color:#d03b3b}}
img{{width:100%;display:block}}
figcaption{{padding:8px 10px;font-size:13px;line-height:1.5}}
figure.miss figcaption b:first-child{{color:#e66767}}
figure.hit figcaption b:first-child{{color:#1baf7a}}
.m{{display:block;color:#8a8a86;font-size:11.5px;margin-top:2px}}
.w b.w{{color:#fab219}}
.note{{background:#1c1c1b;border-left:3px solid #fab219;border-radius:8px;padding:11px 15px;
  font-size:13.5px;color:#c3c2b7;margin:0 0 22px;line-height:1.65}}
</style></head><body><div class="w">
<h1>模型判斷依據 — Grad-CAM 熱區</h1>
<p class="s">每張圖：<b>左＝原圖與推出的佐證框，右＝熱區</b>。紅框卡片＝判錯。
熱區是模型自己的依據，不是 Gemini 的猜測。</p>
<div class="kpi">
 <div class="k"><i>照片</i><b>{s["photos"]}</b></div>
 <div class="k"><i>可比測試集 top-1</i><b>{s["testTop1"]} <small>(n={s["comparableTestPhotos"]})</small></b></div>
 <div class="k"><i>熱區落在遮罩角落</i><b>{s["cornerMassMean"]:.1%}</b></div>
 <div class="k"><i>角落面積佔比（基準）</i><b>14.4%</b></div>
 <div class="k"><i>角落熱區 &gt;30% 的照片</i><b>{s["cornerMassOver30pct"]}</b></div>
 <div class="k"><i>佐證框平均面積</i><b>{(s["boxAreaMean"] or 0):.1%}</b></div>
 <div class="k"><i>遮掉框後信心平均降幅</i><b>{(s["confDropMean"] or 0):.2f}</b></div>
 <div class="k"><i>遮掉框後答案翻掉</i><b>{(s["predFlipRate"] or 0):.0%}</b></div>
</div>
<div class="note"><b>怎麼讀「熱區落在遮罩角落」：</b>四個灰色角落合計佔畫面 14.4%，
本身沒有工項資訊。若這個數字明顯高於 14.4%，代表模型在拿遮罩邊界當捷徑猜答案，
分數再高也不可信。判對與判錯的照片分別是
{s["cornerMassMeanCorrect"]} / {s["cornerMassMeanWrong"]}。<br><br>
<b>怎麼讀「遮掉框後」：</b>把紅框塗成灰色再問一次同一個模型。信心大跌或答案翻掉，
代表這個框真的是它的依據；信心幾乎不動，代表那張熱圖只是裝飾、依據其實在別處。
這是驗證「解釋可不可信」，不是驗證「模型準不準」。</div>
<div class="grid">"""
    body = "".join(card(r) for r in rows)
    (outdir / "contact.html").write_text(head + body + "</div></div></body></html>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ft-report-scratch")
    ap.add_argument("--set", dest="which", default="test", choices=["test", "all"])
    ap.add_argument("--n", type=int, default=0, help="0 = 全部")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--probe", action="store_true", help="改成批次算線上那顆探針的熱區，存 npz 給操作台用")
    ap.add_argument("--split", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.probe:
        run_probe(a.split, force=a.force)
    else:
        run(a.ckpt, a.which, a.n, a.tag)
