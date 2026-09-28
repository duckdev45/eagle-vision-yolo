"""G2 promotion gate（WORKFLOW §7.2）：候選模型上線前的機械化檢查。

「什麼分數才能上線」有定義，就不必每次靠開會決定。本腳本跑 §7.2 裡
能離線算的三項（洩漏測試/ONNX 一致性/影子模式是部署期的事，不在此）：

  1. 黃金集 coverage @ precision ≥ 0.90，且不低於 baseline − 2pt（產品指標）
  2. 獨立黃金集分組 macro-F1 不低於 baseline − 1σ（防單點運氣；
     GroupKFold 按 案場×日期，同一折評估兩版已訓練模型）
  3. 沒有任何類別 recall 掉到 0（防小類被犧牲換總分）

黃金答案 = G1 仲裁後的 data/golden/golden_labels.csv（fileId,cls）。
至少 300 張 PMS WORK_ITEM，且與兩版 split 的訓練和測試案場日期完全分離。

    uv run src/g2_gate.py --candidate v36 --baseline v35

exit code 0 = 全過可上線；1 = 有檢查沒過。報告存 data/golden/g2_report.json。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import features as features_mod
import paths
import split as split_mod

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.evaluation_metrics import PRECISION_BAR, coverage_at_precision

COVERAGE_SLACK = 0.02  # 不低於 baseline − 2pt
F1_SLACK_SIGMA = 1.0  # 不低於 baseline − 1σ


def probe(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)["clf"]


def corpus(model_key: str = features_mod.LEGACY_ENCODER) -> tuple[dict, object]:
    """黃金集照片的 embedding 索引（兩版編碼器不同時各取各的）。"""
    ids, emb = features_mod.load(model_key)
    idx = {f: i for i, f in enumerate(ids)}
    return idx, emb


def golden(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"找不到人工仲裁答案 {path}；先完成 G1 雙人標註與仲裁。")
    g = pd.read_csv(path, dtype=str)
    if not {"fileId", "cls"} <= set(g.columns):
        raise ValueError("黃金答案需要 fileId,cls")
    if (
        g.fileId.isna().any()
        or g.cls.isna().any()
        or g.fileId.str.strip().eq("").any()
        or g.cls.str.strip().eq("").any()
        or g.fileId.duplicated().any()
    ):
        raise ValueError("黃金答案有空白或重複 fileId／cls")
    return g.set_index("fileId")


def eval_probe(clf, X, y) -> dict:
    if not len(y):
        raise ValueError("eval_probe requires at least one labeled sample")
    y = np.asarray(y)
    p = clf.predict_proba(X)
    classes = list(clf.classes_)
    idx = p.argmax(1)
    pred = np.array([classes[i] for i in idx])
    conf = p[np.arange(len(p)), idx]
    correct = pred == y
    cov = coverage_at_precision(correct, conf, precision_bar=PRECISION_BAR)
    per_class = {}
    for c in sorted(set(y)):
        m = y == c
        rec = float(correct[m].mean())
        per_class[c] = {
            "n": int(m.sum()),
            "recall": round(rec, 3),
            "本輪可評": "✅" if m.sum() >= 30 else f"n={m.sum()} <30",
        }
    return {"top1": round(float(correct.mean()), 4), "coverage": round(cov, 4), "per_class": per_class}


def validate_holdout(g: pd.DataFrame, candidate: str, baseline: str) -> dict[str, str]:
    """黃金集必須是 PMS 施作項目，且兩版模型都未見過同案場同日照片。"""
    if len(g) < 300:
        raise ValueError(f"黃金集只有 {len(g)} 張；G1 至少需要 300 張獨立人工答案。")
    manifest = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False).drop_duplicates(
        "fileId", keep="last"
    )
    by_id = manifest.set_index("fileId")
    unknown = set(g.index) - set(by_id.index)
    if unknown:
        raise ValueError(f"黃金集含非 PMS 照片：{sorted(unknown)[:5]}")
    selected = by_id.loc[g.index]
    valid = (
        selected.source.eq("WORK_ITEM")
        & selected.active.str.lower().isin(["true", "1"])
        & selected.constrId.ne("")
        & selected.reportDate.ne("")
        & selected.reportDate.le(date.today().isoformat())
    )
    if not valid.all():
        raise ValueError(f"黃金集含非有效 PMS 施作項目：{selected.index[~valid].tolist()[:5]}")
    groups = dict(zip(manifest.fileId, manifest.constrId + "|" + manifest.reportDate))
    gold_groups = {groups[f] for f in g.index}
    for name in (candidate, baseline):
        split_path = paths.SPLITS / f"{name}.json"
        sp = json.loads(split_path.read_text(encoding="utf-8"))
        foreign = set(sp.get("datasets", {}).values()) - {"pms", "crop"}
        if foreign:
            raise ValueError(f"{name} 使用非 PMS 訓練來源 {sorted(foreign)}；請用 PMS 專用流程建立比較版本。")
        used = {f.split("#", 1)[0] for f in sp["train"] + sp["test"]}
        overlap = gold_groups & {groups[f] for f in used if f in groups}
        if overlap:
            raise ValueError(
                f"黃金集與 {name} 訓練／測試集有 {len(overlap)} 個案場日期重疊；需先保留獨立資料再重訓。"
            )
    return {f: groups[f] for f in g.index}


def group_folds(golden_ids: list[str], groups: dict[str, str], n_splits: int) -> list[list[str]]:
    """所有黃金照片依案場×日期分到相同折；每張恰好評估一次。"""
    from sklearn.model_selection import GroupKFold

    n = min(n_splits, len(set(groups.values())))
    if n < 2:
        raise ValueError("黃金集至少需要兩個不同案場日期才能分組評估。")
    splitter = GroupKFold(n_splits=n)
    return [
        [golden_ids[i] for i in test]
        for _, test in splitter.split(golden_ids, groups=[groups[f] for f in golden_ids])
    ]


def grouped_eval(
    candidate_clf,
    baseline_clf,
    folds: list[list[str]],
    g: pd.DataFrame,
    idx: dict,
    emb,
    baseline_index: tuple[dict, object] | None = None,
) -> dict:
    """同一批獨立黃金折評估已訓練好的兩版模型，不在考卷上重訓。

    baseline_index：baseline 用不同編碼器時它自己的 (idx, emb)；預設與 candidate 共用。
    """
    from sklearn.metrics import f1_score

    b_idx, b_emb = baseline_index or (idx, emb)
    out = {"candidate": [], "baseline": []}
    for fold in folds:
        y = g.loc[fold, "cls"].to_numpy()
        labels = sorted(set(y))
        for name, clf, (i, e) in (
            ("candidate", candidate_clf, (idx, emb)),
            ("baseline", baseline_clf, (b_idx, b_emb)),
        ):
            X = np.stack([e[i[f]] for f in fold])
            out[name].append(
                round(float(f1_score(y, clf.predict(X), labels=labels, average="macro", zero_division=0)), 4)
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(paths.GOLDEN / "golden_labels.csv"))
    ap.add_argument("--candidate", required=True, help="候選模型（split 名，如 v36）")
    ap.add_argument("--baseline", required=True, help="現行模型（CURRENT 那版）")
    ap.add_argument("--cv", type=int, default=5, help="獨立黃金集分組評估折數上限")
    a = ap.parse_args()

    g = golden(a.labels)
    groups = validate_holdout(g, a.candidate, a.baseline)
    c_path, b_path = split_mod.probe_path(a.candidate), split_mod.probe_path(a.baseline)
    for p in (c_path, b_path):
        if not p.exists():
            raise SystemExit(f"找不到 {p}")
    c_enc, b_enc = split_mod.encoder(a.candidate), split_mod.encoder(a.baseline)
    idx, emb = corpus(c_enc)
    b_idx, b_emb = (idx, emb) if b_enc == c_enc else corpus(b_enc)

    missing = (set(g.index) - set(idx)) | (set(g.index) - set(b_idx))
    if missing:
        raise ValueError(
            f"黃金集有 {len(missing)} 張沒有 embedding，不可縮小考卷後照常宣稱通過：{sorted(missing)[:5]}"
        )
    X = np.stack([emb[idx[f]] for f in g.index])
    Xb = np.stack([b_emb[b_idx[f]] for f in g.index])
    y = g.cls.to_numpy()
    print(f"獨立 PMS 黃金集 {len(g)} 張")

    report: dict = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "labels": a.labels,
        "candidate": a.candidate,
        "baseline": a.baseline,
        "golden_n": len(g),
        "golden_dropped_no_emb": 0,
        "checks": {},
    }
    verdicts: list[bool] = []

    candidate_clf, baseline_clf = probe(c_path), probe(b_path)
    cand_c, base_c = eval_probe(candidate_clf, X, y), eval_probe(baseline_clf, Xb, y)
    report["golden_eval"] = {"candidate": cand_c, "baseline": base_c}
    print(f"\n== ① 黃金集 coverage @ precision ≥ {PRECISION_BAR}")
    print(f"   candidate {a.candidate}: coverage {cand_c['coverage']} (top1 {cand_c['top1']})")
    print(f"   baseline  {a.baseline}: coverage {base_c['coverage']} (top1 {base_c['top1']})")
    ok1 = cand_c["coverage"] > 0 and cand_c["coverage"] >= base_c["coverage"] - COVERAGE_SLACK
    print(f"   → {'PASS' if ok1 else 'FAIL'}")
    report["checks"]["coverage_at_precision"] = ok1
    verdicts.append(ok1)

    print("\n== ③ per-class recall（任何一類掉 0 = FAIL）")
    zeros = [c for c, s in cand_c["per_class"].items() if s["recall"] == 0 and s["n"] >= 30]
    small_zero = [c for c, s in cand_c["per_class"].items() if s["recall"] == 0 and s["n"] < 30]
    ok3 = not zeros
    print(f"   recall=0 的類（n≥30）：{zeros or '無'}；n<30 的 0 類（本輪不評）：{small_zero or '無'}")
    print(f"   → {'PASS' if ok3 else 'FAIL'}")
    report["checks"]["no_zero_recall"] = ok3
    report["zero_recall_small_n"] = small_zero
    verdicts.append(ok3)

    print(f"\n== ② 獨立黃金集分組評估 macro-F1（最多 {a.cv} 折，同折比較兩版）")
    folds = group_folds(list(g.index), groups, a.cv)
    cv = grouped_eval(candidate_clf, baseline_clf, folds, g, idx, emb, baseline_index=(b_idx, b_emb))
    cand_f1, base_f1 = np.array(cv["candidate"]), np.array(cv["baseline"])
    print(f"   candidate 各折：{cv['candidate']} → mean {cand_f1.mean():.4f} ± {cand_f1.std():.4f}")
    print(f"   baseline  各折：{cv['baseline']} → mean {base_f1.mean():.4f} ± {base_f1.std():.4f}")
    ok2 = bool(cand_f1.mean() >= base_f1.mean() - base_f1.std() * F1_SLACK_SIGMA)
    print(
        f"   → {'PASS' if ok2 else 'FAIL'}（門檻：不低於 baseline − 1σ = {base_f1.mean() - base_f1.std() * F1_SLACK_SIGMA:.4f}）"
    )
    report["checks"]["grouped_macro_f1"] = ok2
    report["grouped_eval"] = {k: v for k, v in cv.items()}
    verdicts.append(ok2)

    report["verdict"] = "PASS" if all(verdicts) else "FAIL"
    out = paths.GOLDEN / "g2_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n總結：{report['verdict']}（{sum(verdicts)}/{len(verdicts)} 項通過）→ {out}")
    print("§7.2 其餘檢查（洩漏測試/ONNX 一致性/影子模式）屬部署期，不在本腳本。")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
