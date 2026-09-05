"""樂氧森缺失照片 × gemma4:e4b「有無缺失」狀態判斷煙霧測試（ROADMAP 待決 #5）。

GDINO 結案（2026-09-03）後初標工具缺人：候選分工是「VLM 判有無缺失 →
GDINO 只做定位」。本腳本驗證 gemma4:e4b 能不能當那個 VLM。

Ground truth = 語料的**弱標籤**（每張照片掛一條人寫缺失描述，見
data/field_reports/README.md）：
    正例：描述命中強視覺樣態關鍵字（縫隙/髒污/破損/滲水/油漆/刮傷/裂縫/
          鏽蝕/不平整/保護…），刻意排除「建議加裝/未做」的物件缺席型
          ——單張照片判「不在畫面的東西」不是視覺判斷題。
    負例：讚美／純位置詞。（空描述原當負例，2026-09-05 仲裁發現空描述照片
          常帶紅筆圈缺失——負例噪音來源，改 excluded。）
弱標籤有噪音（描述可能對不上照片、負例照片也可能真有缺失），所以分歧樣本
要人工看圖仲裁——仲裁結果順便量弱標籤本身的噪音率。

判定 prompt 沿用尺入鏡 PoC 的教訓（src/vlm_ruler_poc.py）：先描述場景 →
再檢查 → 才判定；VLM 只看圖，絕不餵描述。

    uv run src/fr_vlm_defect.py --sample    # 正 25 + 負 20 分層抽樣
    uv run src/fr_vlm_defect.py --photos id1,id2

產出：data/field_reports/derived/vlm/{run_id}/judgments.jsonl + summary.json
status 一律 AI_GUESS，人審後才是可信標籤。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from paths import FIELD_REPORTS, FR_PHOTOS, FR_VLM

MANIFEST = FIELD_REPORTS / "raw" / "manifest.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"
PROMPT_VERSION = "v1-desc-then-judge"
PROMPT = """你是工程驗收巡查照片的查驗員。任務：判斷照片中是否看得到「施工缺失或狀態異常」。
缺失的例子：破損凹陷、裂縫、縫隙過大、刮傷撞痕、髒污殘留、滲水積水水痕、鏽蝕、掉漆油漆缺陷、批土收邊不平整、應保護而未保護。
不算缺失：正常施工作業中、正常完成面、正常設備家具、保護措施做得好。
注意：只判「照片裡看得到的狀態」，不要推測沒入鏡的東西；不確定就 has_defect=false 並在 evidence 寫清楚為何不確定。
步驟：先客觀描述場景，再逐一檢查上述缺失樣態，最後判定。
只輸出 JSON：{"scene": "一句話描述", "defect_check": "檢查過程一句話", "has_defect": true/false, "defect_type": "無則填none", "evidence": "缺失在圖中何處與外觀；沒有就寫not found"}"""

# v2（2026-09-05，唯一一輪修正）：v1 對 25 張弱標籤正例只中 3 張，人眼複看水龍頭鏽蝕/門框凹陷
# 都清楚可見——問題是「不確定就 false」把模型推向保守，且缺失多是寬景裡的小面積局部。
# v2 拿掉 false 偏置、強制先列「可疑細節候選」再判，小面積異常也算。
PROMPTS = {
    "v1-desc-then-judge": PROMPT,
    "v2-scan-candidates": """你是工程驗收巡查照片的查驗員。任務：找出照片中任何「施工缺失或狀態異常」，包括很小、很局部的異常。
缺失樣態：破損缺角、凹陷撞痕、裂縫、縫隙過大或填縫不全、刮傷、髒污殘留污漬、滲水積水水痕、鏽蝕變色、掉漆漆面缺陷、批土或收邊不平整不順、保護膜/保護板未拆或應保護而未保護、壁紙或貼皮不平。
不算缺失：正常施工作業中、正常完成面、正常設備家具、正常材質紋理與接縫。
巡查照片通常是為了拍某個缺失而拍的，缺失常常很小、在畫面局部（門框邊、水龍頭、磁磚角、接縫、收邊條）。請放大檢查每個物件的邊緣與表面。
步驟：1) 一句話描述場景 2) 列出所有看起來可疑的小細節（candidates，沒有就空陣列） 3) 逐一判斷是否真為缺失 4) 判定。
只輸出 JSON：{"scene": "一句話", "candidates": ["可疑細節1", ...], "defect_check": "逐一判斷一句話", "has_defect": true/false, "defect_type": "無則填none", "evidence": "缺失在圖中何處與外觀；沒有就寫not found"}""",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "candidates": {"type": "array", "items": {"type": "string"}},
        "defect_check": {"type": "string"},
        "has_defect": {"type": "boolean"},
        "defect_type": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["scene", "candidates", "defect_check", "has_defect", "defect_type", "evidence"],
}

# 強視覺樣態關鍵字（正例池）。物件缺席型（建議加裝/沒開關蓋）刻意不收。
DEFECT_RE = re.compile(
    r"裂|縫|刮傷|撞|凹陷|破損|破掉|髒|污染|殘留|滲水|積水|水痕|鏽|生鏽"
    r"|掉漆|油漆|批土|收邊|不平整|粉塵|保護|貼皮|破皮|缺口|污漬"
)
# 物件缺席型：命中 DEFECT_RE 也要排除（描述是「東西不在」不是「狀態異常」）
ABSENCE_RE = re.compile(r"建議加裝|無修飾蓋|沒開關蓋|未.*裝|加裝|增設|無.*蓋")
PRAISE_RE = re.compile(r"👍|很棒|Great|good|漂亮|感謝|舒適|氛圍|採光", re.I)
LOC_RE = re.compile(r"^(健身房|電梯|RF|B1F|1F|[0-9]+F)[、,，0-9F~之]*$")


def weak_label(desc: str) -> str:
    d = (desc or "").strip()
    if not d:
        return "excluded"  # 2026-09-05 仲裁：空描述照片常帶紅筆圈缺失，不是乾淨負例
    if PRAISE_RE.search(d) or LOC_RE.match(d):
        return "clean"
    if DEFECT_RE.search(d) and not ABSENCE_RE.search(d):
        return "defect"
    return "excluded"  # 物件缺席型/其他：不進抽樣池


def load_manifest() -> list[dict]:
    tiny = set()
    tiny_path = FIELD_REPORTS / "derived" / "tiny_images.json"
    if tiny_path.exists():
        tiny = {t["file"].removesuffix(".webp") for t in json.loads(tiny_path.read_text())}
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    return [r for r in rows if r["fileId"] not in tiny]


def sample(rows: list[dict], n_pos: int = 25, n_neg: int = 20, seed: int = 42) -> list[dict]:
    pos = [r for r in rows if weak_label(r["description"]) == "defect"]
    neg = [r for r in rows if weak_label(r["description"]) == "clean"]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:n_pos] + neg[:n_neg]


def judge(img_path: Path) -> tuple[dict, float]:
    img = base64.b64encode(img_path.read_bytes()).decode()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[PROMPT_VERSION], "images": [img]}],
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
    return json.loads(resp["message"]["content"]), time.time() - t0


def run(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / "judgments.jsonl"
    done = {json.loads(ln)["fileId"] for ln in jl.open()} if jl.exists() else set()
    with jl.open("a") as f:
        for i, r in enumerate(rows):
            if r["fileId"] in done:
                continue
            p = FR_PHOTOS / f"{r['fileId']}.webp"
            if not p.exists():
                print(f"[{i + 1}/{len(rows)}] MISS {r['fileId']}")
                continue
            try:
                out, el = judge(p)
            except Exception as e:
                f.write(json.dumps({"fileId": r["fileId"], "error": str(e)}, ensure_ascii=False) + "\n")
                f.flush()
                continue
            rec = {
                "fileId": r["fileId"],
                "weak_label": weak_label(r["description"]),
                "description": r["description"],
                **out,
                "elapsed_s": round(el, 1),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            flag = "DEF" if out["has_defect"] else "-"
            print(f"[{i + 1}/{len(rows)}] {flag} {r['fileId'][:8]} {el:.1f}s | {out['defect_type']}")
    summarize(jl, out_dir)


def summarize(jl: Path, out_dir: Path) -> None:
    recs, errs = [], 0
    for line in jl.open():
        d = json.loads(line)
        if "error" in d:
            errs += 1
        else:
            recs.append(d)
    if not recs:
        print("no records")
        return
    # 混淆矩陣（對弱標籤）
    tp = sum(1 for r in recs if r["weak_label"] == "defect" and r["has_defect"])
    fn = sum(1 for r in recs if r["weak_label"] == "defect" and not r["has_defect"])
    fp = sum(1 for r in recs if r["weak_label"] == "clean" and r["has_defect"])
    tn = sum(1 for r in recs if r["weak_label"] == "clean" and not r["has_defect"])
    summary = {
        "run_id": out_dir.name,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "judged": len(recs),
        "errors": errs,
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in recs) / len(recs), 1),
        "vs_weak_label": {
            "tp_defect_yes": tp,
            "fn_defect_missed": fn,
            "fp_clean_called_defect": fp,
            "tn_clean_ok": tn,
        },
        "disagreements": [
            {
                "fileId": r["fileId"],
                "weak": r["weak_label"],
                "vlm": r["has_defect"],
                "desc": r["description"][:50],
                "evidence": r["evidence"][:80],
            }
            for r in recs
            if (r["weak_label"] == "defect") != r["has_defect"]
        ],
        "defect_types": dict(Counter(r["defect_type"] for r in recs if r["has_defect"])),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "note": "弱標籤對比，分歧需人工看圖仲裁後才可下結論",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    global PROMPT_VERSION
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true", help="正 25 + 負 20 抽樣")
    g.add_argument("--photos", help="逗號分隔 photoId 補跑")
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y-%m-%d-%H%M"))
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default=PROMPT_VERSION)
    args = ap.parse_args(argv)
    PROMPT_VERSION = args.prompt

    rows = load_manifest()
    if args.sample:
        rows = sample(rows)
    else:
        ids = set(args.photos.split(","))
        rows = [r for r in rows if r["fileId"] in ids]
    n_pos = sum(1 for r in rows if weak_label(r["description"]) == "defect")
    print(f"{len(rows)} 張（正 {n_pos}）→ {FR_VLM}/{args.run_id}/")
    run(rows, FR_VLM / args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
