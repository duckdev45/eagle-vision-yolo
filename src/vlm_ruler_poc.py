"""gemma4:e4b「量測工具入鏡」判定 PoC（ROADMAP 執行序 #3：B→A 降格驗證）。

B 類（量測數值，佔 REQUIRED 19.3%）的判定基準多半是「照片裡有沒有尺」——
如 QS0701-4.14「防水高度以尺丈量並附相片存證」，標準要的憑證就是量測工具
入鏡。判「尺入鏡」遠比判「牆平不平」容易（ANALYSIS §NEXT_PHASE）；若 VLM
能穩定判這題，B 類大半降格成 A（純視覺），gemma4:e4b 就有第一個可上線的
判定角色。

資料：data/raw/photos（QMS 日報照）分層抽樣。specKey 防水 20 全收
（QS0701 正是防水）、鋼筋 8 / 連續壁 4 全收、磁磚 15、油漆 15、
specKey 空 18 隨機 → 預設 80 張。

提示詞採「先描述場景 → 再檢查 → 才判定」兩段式。教訓：初版一句式
prompt（直接問有沒有捲尺）在兩張無尺照片上 2/2 幻覺回 true，evidence
是含糊的「捲尺展開的樣子」——結構化 schema 逼它二選一反而催生假陽性。
兩段式強迫先客觀列物件再判定，同一批照片 2/2 修正。

    uv run src/vlm_ruler_poc.py --sample            # 80 張分層抽樣
    uv run src/vlm_ruler_poc.py --all               # 全量 988 張（約 1 小時）
    uv run src/vlm_ruler_poc.py --photos id1,id2    # 指定 fileId 補跑

產出：data/derived/vlm/ruler/{run_id}/judgments.jsonl + summary.json
每列 {fileId, specKey, title, scene, tool_check, tool_visible, tool_type,
evidence, elapsed_s}。status 一律 AI_GUESS——沿用 yolo_preprocessor 哲學，
這是候選判定，不是訓練級標籤；人審後才可信。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from paths import MANIFEST, PHOTOS, VLM

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"

# 兩段式 prompt（v2）。改 prompt 只能改這裡＋PROMPT_VERSION。
PROMPT_VERSION = "v2-desc-then-judge"
PROMPT = """你是最嚴謹的工地照片查驗員。任務：判斷照片中是否出現量測工具（捲尺/直尺/角尺/水平尺/雷射水平儀）。
步驟：1) 先客觀列出照片中的主要物件與人物動作；2) 逐一檢查清單中是否有量測工具；3) 只有在能明確指出工具外觀時才回答 true。
只輸出 JSON：{"scene": "一句話描述", "tool_check": "找工具的過程一句話", "tool_visible": true/false, "tool_type": "無則填none", "evidence": "工具在圖中何處；找不到工具就寫not found"}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "tool_check": {"type": "string"},
        "tool_visible": {"type": "boolean"},
        "tool_type": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["scene", "tool_check", "tool_visible", "tool_type", "evidence"],
}

# 分層抽樣：specKey → 收幾張（None = 全收）
STRATUM: dict[str, int | None] = {
    "防水": None,
    "鋼筋": None,
    "連續壁": None,
    "磁磚": 15,
    "油漆": 15,
    "": 18,  # specKey 空＝其餘日報照
}


def load_manifest() -> list[dict]:
    return list(csv.DictReader(MANIFEST.open(encoding="utf-8")))


def sample_photos(rows: list[dict], seed: int = 42) -> list[dict]:
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(r["specKey"], []).append(r)
    picked: list[dict] = []
    for key, n in STRATUM.items():
        pool = by_key.get(key, [])
        if n is None or len(pool) <= n:
            picked.extend(pool)
        else:
            rng = random.Random(seed)
            picked.extend(rng.sample(pool, n))
    return picked


def photo_path(row: dict) -> Path | None:
    ext = row["fileName"].rsplit(".", 1)[-1] if row.get("fileName") else "webp"
    p = PHOTOS / f"{row['fileId']}.{ext}"
    return p if p.exists() else None


def judge(img_path: Path) -> tuple[dict, float]:
    img = base64.b64encode(img_path.read_bytes()).decode()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT, "images": [img]}],
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": {"temperature": 0, "num_predict": 400},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=600))
    elapsed = time.time() - t0
    out = json.loads(resp["message"]["content"])
    return out, elapsed


def run(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / "judgments.jsonl"
    done = set()
    if jl.exists():  # 可續跑
        for line in jl.open():
            done.add(json.loads(line)["fileId"])
    with jl.open("a") as f:
        for i, r in enumerate(rows):
            if r["fileId"] in done:
                continue
            p = photo_path(r)
            if p is None:
                print(f"[{i + 1}/{len(rows)}] MISS {r['fileId']}")
                continue
            try:
                out, el = judge(p)
            except Exception as e:  # 單張失敗不中斷批次
                f.write(json.dumps({"fileId": r["fileId"], "error": str(e)}, ensure_ascii=False) + "\n")
                f.flush()
                continue
            rec = {
                "fileId": r["fileId"],
                "specKey": r["specKey"],
                "title": r["title"],
                **out,
                "elapsed_s": round(el, 1),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            flag = "RULER" if out["tool_visible"] else "-"
            print(f"[{i + 1}/{len(rows)}] {flag} {r['fileId'][:8]} {el:.1f}s | {out['tool_type']}")
    summarize(jl, out_dir)


def summarize(jl: Path, out_dir: Path) -> None:
    recs, errs = [], 0
    for line in jl.open():
        d = json.loads(line)
        if "error" in d:
            errs += 1
        else:
            recs.append(d)
    vis = [r for r in recs if r["tool_visible"]]
    by_key = Counter(r["specKey"] for r in recs)
    vis_key = Counter(r["specKey"] for r in vis)
    summary = {
        "run_id": out_dir.name,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "judged": len(recs),
        "errors": errs,
        "tool_visible": len(vis),
        "rate": round(len(vis) / len(recs), 3) if recs else None,
        "tool_types": dict(Counter(r["tool_type"] for r in vis)),
        "by_speckey": {k: f"{vis_key.get(k, 0)}/{v}" for k, v in sorted(by_key.items())},
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in recs) / len(recs), 1) if recs else None,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "note": "AI_GUESS 層：未人審，precision/recall 待人工驗證",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true", help="80 張分層抽樣")
    g.add_argument("--all", action="store_true", help="全量 988 張")
    g.add_argument("--photos", help="逗號分隔 fileId 補跑")
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y-%m-%d-%H%M"))
    args = ap.parse_args(argv)

    rows = load_manifest()
    if args.sample:
        rows = sample_photos(rows)
    elif args.photos:
        ids = set(args.photos.split(","))
        rows = [r for r in rows if r["fileId"] in ids]
    # --all 不動
    print(f"{len(rows)} 張 → {VLM}/ruler/{args.run_id}/")
    run(rows, VLM / "ruler" / args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
