"""跨 run 的學習紀錄：reports/*/metrics.json → reports/JOURNAL.md

每個 run 只知道自己的分數。這支把它們按時間串起來，回答三件事：

  今天這一版比上一版好在哪、壞在哪   → per-class F1 差
  哪些混淆是慣犯，不是這次運氣不好   → 跨 run 重複出現的「真X→猜Y」
  哪些照片每次都錯                   → 同一張圖出現在多個 run 的 errors/

    uv run src/journal.py              # 重寫 reports/JOURNAL.md
    uv run src/journal.py --last 10    # 慣犯統計只看最近 10 個 run

不重算任何東西、不讀模型，純粹讀已經生成的報告——所以隨時可以跑，
也可以在刪掉舊 report 之後跑（那一版就從紀錄裡消失，這是刻意的）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
import paths  # noqa: E402

# evaluate.py 寫的檔名：真{true}_猜{pred}_{fileId[:8]}.jpg
# 非貪婪 + 尾端固定 8 碼，才不會把 hash 的前幾碼算進標籤裡
ERR_RE = re.compile(r"^真(.+?)_猜(.+?)_(.{8})\.jpg$")


def parse_err(name: str) -> tuple[str, str, str] | None:
    m = ERR_RE.match(name)
    return m.groups() if m else None  # type: ignore[return-value]


def load_runs() -> list[dict]:
    """所有有 metrics.json 的 report 目錄，按完成時間排序（舊→新）。"""
    runs = []
    for d in paths.REPORTS_OUT.iterdir():
        mf = d / "metrics.json" if d.is_dir() else None
        if not mf or not mf.exists():
            continue
        m = json.loads(mf.read_text())
        cfg_file = d / "config.json"
        cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
        errs = [e for e in ((d / "errors").iterdir() if (d / "errors").is_dir() else [])]
        per = m.get("perClass", {})
        # macroF1 一律從 perClass 現算，不用存檔裡那個值：2026-08-25 以前的報告是用
        # 舊算法存的（support=0 的幽靈類別以 f1=0 計入平均），直接比會把指標 bug
        # 當成模型退步。現算才能讓 26 個歷史 run 跟新的擺在同一張表上。
        real = [v["f1-score"] for v in per.values() if v.get("support", 0) > 0]
        runs.append({
            "name": d.name,
            "when": datetime.fromtimestamp(mf.stat().st_mtime),
            "split": cfg.get("split", "?"),
            "labels": cfg.get("labelsVersion", "?"),
            "top1": m.get("top1"),
            "macroF1": round(sum(real) / len(real), 4) if real else None,
            "storedMacroF1": m.get("macroF1"),
            "ghosts": sorted(k for k, v in per.items() if v.get("support", 0) == 0),
            "support": m.get("support"),
            "perClass": per,
            "errs": [p for p in (parse_err(e.name) for e in errs) if p],
        })
    runs.sort(key=lambda r: r["when"])
    return runs


def history_table(runs: list[dict]) -> str:
    out = ["| 完成 | run | split | labels | 測試張數 | top1 | macroF1 | 類數 | Δtop1 | 測不到的類 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    prev = None
    for r in runs:
        d = "" if prev is None else f"{r['top1'] - prev:+.4f}"
        n = len([v for v in r["perClass"].values() if v.get("support", 0) > 0])
        g = "、".join(r["ghosts"]) if r["ghosts"] else ""
        out.append(f"| {r['when']:%m-%d %H:%M} | {r['name']} | {r['split']} | "
                   f"{r['labels']} | {r['support']} | {r['top1']} | {r['macroF1']} | {n} | {d} | {g} |")
        prev = r["top1"]
    return "\n".join(out)


def _real(per: dict) -> dict:
    """濾掉 support=0 的類別。那不是「學會了」或「退步了」，是這份考卷沒考到。"""
    return {k: v for k, v in per.items() if v.get("support", 0) > 0}


def class_delta(cur: dict, prev: dict, eps: float = 0.005) -> str:
    """per-class F1 差。分母（support）變了就標出來——標籤一改分母就不同，
    分數升降有一部分是換考卷換的，不是模型變強。"""
    a, b = _real(cur["perClass"]), _real(prev["perClass"])
    rows = []
    for k in sorted(set(a) | set(b)):
        if k not in b:
            rows.append((99.0, f"| {k} | — | {a[k]['f1-score']:.3f} | 新類別 | {a[k]['support']:.0f} |"))
        elif k not in a:
            rows.append((-99.0, f"| {k} | {b[k]['f1-score']:.3f} | — | **這版測不到** | 0 |"))
        else:
            d = a[k]["f1-score"] - b[k]["f1-score"]
            if abs(d) < eps:
                continue
            sup = f"{a[k]['support']:.0f}"
            if a[k]["support"] != b[k]["support"]:
                sup += f"（上版 {b[k]['support']:.0f}）"
            rows.append((d, f"| {k} | {b[k]['f1-score']:.3f} | {a[k]['f1-score']:.3f} "
                            f"| {d:+.3f} | {sup} |"))
    if not rows:
        return "兩版 per-class F1 差異都在 ±0.005 內。"
    rows.sort(key=lambda x: -x[0])
    return "\n".join(["| 類別 | 上一版 F1 | 這一版 F1 | Δ | support |", "|---|---|---|---|---|"]
                     + [r[1] for r in rows])


def weakest(run: dict, n: int = 6) -> str:
    rows = sorted(_real(run["perClass"]).items(), key=lambda kv: kv[1]["f1-score"])[:n]
    return "\n".join(
        ["| 類別 | f1 | precision | recall | support |", "|---|---|---|---|---|"]
        + [f"| {k} | {v['f1-score']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} "
           f"| {v['support']:.0f} |" for k, v in rows])


def repeats(runs: list[dict], min_runs: int = 2) -> str:
    """同一組「真X→猜Y」出現在幾個 run。一個 run 內重複只算一次——
    要找的是「每次重訓都還在錯」，不是「這次錯很多張」。"""
    c = Counter()
    for r in runs:
        for t, p, _ in set((t, p, "") for t, p, _ in r["errs"]):
            c[(t, p)] += 1
    hit = [(v, k) for k, v in c.items() if v >= min_runs]
    if not hit:
        return f"沒有混淆在 {min_runs} 個以上的 run 重複出現。"
    hit.sort(reverse=True)
    return "\n".join(["| 真 → 猜 | 出現在幾個 run |", "|---|---|"]
                     + [f"| {k[0]} → {k[1]} | {v} / {len(runs)} |" for v, k in hit[:20]])


def chronic(runs: list[dict], min_runs: int = 3) -> str:
    """每次都錯的照片。這批要嘛標錯了，要嘛真的難——兩種都值得人去看一眼。"""
    seen: dict[str, list[str]] = defaultdict(list)
    for r in runs:
        for t, p, fid in r["errs"]:
            seen[fid].append(f"{r['split']}: 真{t}→猜{p}")
    hit = [(len(v), k, v) for k, v in seen.items() if len(v) >= min_runs]
    if not hit:
        return f"沒有照片連錯 {min_runs} 個 run 以上。"
    hit.sort(reverse=True)
    return "\n".join(["| fileId 前 8 碼 | 錯了幾個 run | 最近一次 |", "|---|---|---|"]
                     + [f"| `{k}` | {n} | {v[-1]} |" for n, k, v in hit[:20]])


def build(last: int = 12) -> str:
    runs = load_runs()
    if not runs:
        return "# 學習紀錄\n\n還沒有任何 report。跑 `make model SPLIT=vN` 之後再來。\n"
    recent = runs[-last:]
    cur = runs[-1]
    body = [
        "# 學習紀錄",
        "",
        f"自動生成，勿手改（`uv run src/journal.py`）。生成於 {datetime.now():%Y-%m-%d %H:%M}，"
        f"共 {len(runs)} 個 run。",
        "",
        "## 一、每次重訓的分數",
        "",
        history_table(runs),
        "",
        f"## 二、最新這版學到／退步了什麼（{cur['name']} vs "
        f"{runs[-2]['name'] if len(runs) > 1 else '無'}）",
        "",
        class_delta(cur, runs[-2]) if len(runs) > 1 else "只有一個 run，沒有可比對象。",
        "",
        f"## 三、這一版最弱的類別（{cur['name']}）",
        "",
        weakest(cur),
        "",
        f"## 四、慣犯混淆（最近 {len(recent)} 個 run）",
        "",
        repeats(recent),
        "",
        f"## 五、每次都錯的照片（最近 {len(recent)} 個 run）",
        "",
        chronic(recent),
        "",
        "---",
        "",
        "註一：**top1 跨版不可直接比**。split 按「每個工地最新幾天」滾動切，每一版的考卷"
        "是不同照片（實測 v18∩v20 只重疊 83/141）。要比模型強弱請讓兩顆考同一份卷。",
        "",
        "註二：macroF1 一律從 `perClass` 現算，只平均 support>0 的類別，"
        "**不採用 `metrics.json` 裡存的值**——2026-08-25 以前的報告把 support=0 的類別"
        "以 f1=0 算進平均（v19 就這樣從 0.856 被壓成 0.778）。"
        "「測不到的類」那欄列的就是這種類別：模型還是會輸出它，只是這份考卷驗不了。",
        "",
        "註三：`errors/` 每個 run 只存前 24 張誤判（evaluate.py 的 `errors` 參數），"
        "所以四、五兩節會**低估**——沒出現不等於沒錯。"
        "要看完整誤判請開 `reports/<run>/confusion.png`。",
        "",
    ]
    return "\n".join(body)


def write(last: int = 12) -> str:
    """重寫 reports/JOURNAL.md，回傳路徑。命令列與操作台的按鈕共用同一條。"""
    paths.REPORTS_OUT.mkdir(parents=True, exist_ok=True)
    out = paths.REPORTS_OUT / "JOURNAL.md"
    out.write_text(build(last))
    return str(out)


def demo() -> None:
    assert parse_err("真泥作-地磚貼飾_猜油漆-批土塗裝_0e2b86ad.jpg") == (
        "泥作-地磚貼飾", "油漆-批土塗裝", "0e2b86ad")
    assert parse_err("confusion.png") is None
    assert parse_err("真A_猜B_1234567.jpg") is None      # hash 不是 8 碼 → 不認
    runs = [{"split": "v1", "errs": [("A", "B", "aaaaaaaa"), ("A", "B", "bbbbbbbb")]},
            {"split": "v2", "errs": [("A", "B", "aaaaaaaa")]}]
    assert "2 / 2" in repeats(runs)                       # 同 run 內重複只算一次
    assert "`aaaaaaaa`" in chronic(runs, min_runs=2)
    assert "沒有照片" in chronic(runs, min_runs=3)
    a = {"perClass": {"X": {"f1-score": 0.9, "precision": 1.0, "recall": 0.8, "support": 10.0}}}
    b = {"perClass": {"X": {"f1-score": 0.5, "precision": 0.5, "recall": 0.5, "support": 10.0}}}
    assert "+0.400" in class_delta(a, b)
    assert "都在 ±0.005 內" in class_delta(a, a)

    # 幽靈類別（support=0）不能算進 macro，也不能出現在「最弱」與「退步」表裡
    ghost = {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0.0}
    g = {"perClass": {**a["perClass"], "防水": ghost}}
    assert _real(g["perClass"]).keys() == {"X"}
    assert "防水" not in weakest(g)                       # 否則 f1=0 會排到第一名
    assert "都在 ±0.005 內" in class_delta(g, a)          # 幽靈不算「新類別」
    assert "測不到" in class_delta(a, {"perClass": {**a["perClass"],
                                                    "Y": {**ghost, "support": 3.0}}})
    print("ok journal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=12, help="慣犯統計看最近幾個 run")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        demo()
    else:
        print(f"學習紀錄 → {write(a.last)}")
