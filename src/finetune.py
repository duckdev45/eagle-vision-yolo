"""階段二：微調編碼器（SPEC §9.2）。

為什麼要微調而不是繼續加資料給線性探針：QMS 那 36k 在凍結編碼器下**完全不遷移**
（只用 QMS 訓練測日報 top-1 0.283，混進去越多掉越兇）。領域差異存在於特徵空間裡，
只有讓編碼器自己動才可能吸收。

兩階段：
  1. QMS 26 個中類上訓練 → 拿到一個看過幾千張工地照的 backbone
  2. 換掉分類頭，在日報 10 類上微調 → 測試集**原封不動**用日報那 99 張

第二階段也可以直接從 ImageNet 權重開始（--no-pretrain-stage1），那就是對照組：
「先看過 QMS」到底有沒有用，差值就是這 36k 的價值。

    uv run --extra train src/finetune.py --stage1              # QMS 預訓練
    uv run --extra train src/finetune.py --stage2              # 接著微調日報
    uv run --extra train src/finetune.py --stage2 --scratch    # 對照組：不看 QMS
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import paths
import split as split_mod

MODEL = "convnext_tiny"
SIZE = 224
CKPT_STAGE1 = paths.MODELS / "backbone-qms.pt"


def _img_path(file_id: str):
    p = paths.IMAGES / f"{file_id}.jpg"
    return p if p.exists() else paths.QMS_IMAGES / f"{file_id}.jpg"


class PhotoDS:
    """必須是模組層級的類別，DataLoader 的 worker 要 pickle 它。"""

    def __init__(self, ids, labels, cidx, tf):
        self.ids = [f for f in ids if labels.get(f) in cidx and _img_path(f).exists()]
        self.labels, self.cidx, self.tf = labels, cidx, tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from PIL import Image

        f = self.ids[i]
        return self.tf(Image.open(_img_path(f)).convert("RGB")), self.cidx[self.labels[f]]


def _loaders(split_name: str, batch: int, workers: int = 4):
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    sp = split_mod.load(split_name)
    classes = sorted({sp["labels"][f] for f in sp["train"] if f in sp["labels"]})
    cidx = {c: i for i, c in enumerate(classes)}

    # 不可用旋轉：工地照有明確重力方向（SPEC §9.2）
    train_tf = T.Compose(
        [
            T.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize(int(SIZE * 1.14)),
            T.CenterCrop(SIZE),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 測試集可能含訓練集沒有的類別，PhotoDS 會直接排除（無從預測）
    tr = PhotoDS(sp["train"], sp["labels"], cidx, train_tf)
    te = PhotoDS(sp["test"], sp["labels"], cidx, eval_tf)
    va = PhotoDS(sp.get("val") or [], sp["labels"], cidx, eval_tf)

    def mk(ds, sh=False):
        return DataLoader(ds, batch, shuffle=sh, num_workers=workers, drop_last=sh and len(ds) > batch)

    return mk(tr, True), (mk(va) if len(va) else None), mk(te), classes


def train(
    split_name: str, epochs: int, lr: float, batch: int, init_from=None, out_name: str = "ft", log=print
) -> dict:
    import timm
    import torch
    import torch.nn as nn

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tr, va, te, classes = _loaders(split_name, batch)
    log(
        f"{split_name}: train {len(tr.dataset)}"
        f" / val {len(va.dataset) if va else 0} / test {len(te.dataset)}"
        f" / {len(classes)} 類 · {dev}"
    )

    model = timm.create_model(MODEL, pretrained=True, num_classes=len(classes))
    if init_from and init_from.exists():
        sd = torch.load(init_from, map_location="cpu")
        # 只載 backbone，分類頭形狀不同要丟掉
        own = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        log(f"  由 {init_from.name} 起手（載入 {len(keep)}/{len(own)} 個張量）")
    model = model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_test, hist = 0.0, 0.0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            seen += len(y)
        sched.step()

        model.eval()

        def acc_of(loader):
            ok = n = 0
            with torch.no_grad():
                for x, y in loader:
                    ok += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
                    n += len(y)
            return ok / max(n, 1)

        te_acc = acc_of(te)
        va_acc = acc_of(va) if va else None
        # 有 val 就用 val 挑；沒有的話只能用 test，那是模型選擇洩漏，比較時看 final
        pick = va_acc if va_acc is not None else te_acc
        hist.append(
            {
                "epoch": ep,
                "loss": round(tot / max(seen, 1), 4),
                "valTop1": None if va_acc is None else round(va_acc, 4),
                "testTop1": round(te_acc, 4),
            }
        )
        log(
            f"  ep{ep:>2} loss {tot / max(seen, 1):.3f}"
            + (f"  val {va_acc:.3f}" if va_acc is not None else "")
            + f"  test {te_acc:.3f}  ({time.time() - t0:.0f}s)"
        )
        if pick >= best:
            best, best_test = pick, te_acc
            torch.save(model.state_dict(), paths.MODELS / f"{out_name}.pt")

    final = hist[-1]["testTop1"] if hist else 0.0
    has_val = va is not None
    (paths.MODELS / f"{out_name}.json").write_text(
        json.dumps(
            {
                "split": split_name,
                "model": MODEL,
                "classes": classes,
                "size": SIZE,
                "selectedBy": "val" if has_val else "test",
                "testTop1AtSelected": round(best_test, 4),
                "finalTop1": final,
                "note": (
                    "用 val 挑 epoch，testTop1AtSelected 是誠實的數字"
                    if has_val
                    else "沒有 val，只能用 test 挑 → 偏樂觀，比較看 finalTop1"
                ),
                "epochs": epochs,
                "lr": lr,
                "batch": batch,
                "initFrom": init_from.name if init_from else None,
                "history": hist,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    log(
        f"選出的 epoch：test top-1 = {best_test:.3f}"
        f"（依 {'val' if has_val else 'test'} 挑）  最後一輪 {final:.3f}"
        f" → models/{out_name}.pt"
    )
    return {"best": best, "testAtSelected": best_test, "final": final, "classes": classes, "history": hist}


def report(ckpt: str, run_tag: str | None = None, log=print) -> str:
    """對某個 checkpoint 產出完整報告：指標、混淆矩陣、誤判樣本、信心門檻曲線。"""
    from datetime import date

    import numpy as np
    import timm
    import torch
    from sklearn.metrics import classification_report

    import evaluate as ev

    meta = json.loads((paths.MODELS / f"{ckpt}.json").read_text())
    classes = meta["classes"]
    _, _, te, _ = _loaders(meta["split"], 32)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model(MODEL, pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load(paths.MODELS / f"{ckpt}.pt", map_location="cpu"))
    model = model.to(dev).eval()

    probs, ys = [], []
    with torch.no_grad():
        for x, y in te:
            probs.append(torch.softmax(model(x.to(dev)), 1).cpu().numpy())
            ys.append(y.numpy())
    P = np.concatenate(probs)
    y = np.array(classes)[np.concatenate(ys)]
    pred = np.array(classes)[P.argmax(1)]

    outdir = paths.REPORTS_OUT / f"{date.today():%Y-%m-%d}-{run_tag or ckpt}"
    (outdir / "errors").mkdir(parents=True, exist_ok=True)
    labels = sorted(set(y) | set(pred))
    rep = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "checkpoint": ckpt,
        "split": meta["split"],
        "model": MODEL,
        "top1": round(float((pred == y).mean()), 4),
        "macroF1": round(float(rep["macro avg"]["f1-score"]), 4),
        "support": len(y),
        "coverageCurve": ev.coverage_curve(P, y, classes),
        "perClass": {k: v for k, v in rep.items() if k in labels},
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    ev.confusion_png(y, pred, labels, outdir / "confusion.png")

    import shutil

    ids = te.dataset.ids
    for f, t, p in [(f, t, p) for f, t, p in zip(ids, y, pred) if t != p][:24]:
        src = _img_path(f)
        if src.exists():
            shutil.copy2(src, outdir / "errors" / f"真{t}_猜{p}_{f[:8]}.jpg")

    log(f"報告 → {outdir}  top1={metrics['top1']} macroF1={metrics['macroF1']}")
    return str(outdir)


def export_onnx(ckpt: str, out: str | None = None, log=print) -> str:
    import timm
    import torch

    meta = json.loads((paths.MODELS / f"{ckpt}.json").read_text())
    model = timm.create_model(MODEL, pretrained=False, num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(paths.MODELS / f"{ckpt}.pt", map_location="cpu"))
    model.eval()
    dst = paths.MODELS / (out or f"{ckpt}.onnx")
    # dynamo=False：新版匯出器會把權重拆成 <name>.onnx.data，兩個檔得一起搬，
    # 交付給產品端時漏一個就是載不起來。這個模型 110MB 遠低於 2GB 單檔上限。
    torch.onnx.export(
        model,
        torch.randn(1, 3, SIZE, SIZE),
        str(dst),
        dynamo=False,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
    )
    (paths.MODELS / f"{(out or ckpt).replace('.onnx', '')}-classes.json").write_text(
        json.dumps(meta["classes"], ensure_ascii=False)
    )
    log(f"ONNX → {dst}")
    return str(dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="QMS 26 中類預訓練")
    ap.add_argument("--stage2", action="store_true", help="日報 10 類微調")
    ap.add_argument("--scratch", action="store_true", help="stage2 對照組：不吃 QMS backbone")
    ap.add_argument("--onnx", default=None, help="把某個 checkpoint 匯出 ONNX")
    ap.add_argument("--report", default=None, help="對某個 checkpoint 產出評估報告")
    # checkpoint / 報告名稱要能區分不同次實驗。沒有 tag 時同名會直接覆蓋，
    # 上一版的權重與報告就沒了（實測踩過：v2 蓋掉 v1 的 0.900 那版）。
    ap.add_argument("--tag", default=None, help="附加在 checkpoint 與報告名稱後，避免覆蓋")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    suffix = f"-{a.tag}" if a.tag else ""
    if a.onnx:
        export_onnx(a.onnx)
    elif a.report:
        report(a.report)
    elif a.stage1:
        train("qms-v1", a.epochs or 12, a.lr or 3e-4, a.batch, out_name=f"backbone-qms{suffix}")
    elif a.stage2:
        init = None if a.scratch else CKPT_STAGE1
        train(
            "v1",
            a.epochs or 20,
            a.lr or 1e-4,
            a.batch,
            init_from=init,
            out_name=("ft-report-scratch" if a.scratch else "ft-report-qms") + suffix,
        )
    else:
        ap.error("要 --stage1 / --stage2 / --onnx")
