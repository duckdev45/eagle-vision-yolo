"""樂氧森缺失照片 × Grounding DINO 煙霧測試（ROADMAP 執行序 #1）。

資料：data/field_reports（469 張，單場單日 → 只當煙霧測試＋標註練兵，不可切分）。
產出：data/field_reports/derived/gdino/runs/{run_id}/
    annotations/{photoId}.json  每框 {pattern, phrase, box, score, status:"AI_GUESS"}
    summary.json               每樣態 photos_with / boxes / avg_score，詞彙命中統計

哲學沿用 src/yolo_preprocessor.py：AI_GUESS 只是候選框，人審（CVAT/主任）後
才是 HUMAN_REFINED 訓練級。詞彙表用英文（文字編碼器是 BERT）。

    uv run --extra gdino src/fr_gdino.py --sample 30    # 先小跑驗輸出
    uv run --extra gdino src/fr_gdino.py --all          # 全量 469 張
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from paths import FR_MANIFEST, FR_PHOTOS, ROOT

# --- 樣態 → 英文詞彙表 ----------------------------------------------------------
# key = 樣態代號（與 data/field_reports/README.md 的掃描表同系）；
# value = 餵 GDINO 的英文名詞短語。GDINO 文字編碼器是 BERT，中文 prompt 效果打折。
# 「未做/缺件」（物件不在畫面裡）無法框，刻意不設詞彙。
# 2026-09-01 12 張小跑後修訂：protective film 在樣品屋照片上滿框氾濫（full-frame
# 假陽性），watering/puddle 缺詞彙（中庭積水 no-hit）。
# 2026-09-01 43 框人審後修訂（precision 19%）：dirty floor 2/14、protective
# corner guard 2/14 語義錯位嚴重——GDINO 框到的是「同紋理區域」不是人指的缺失。
# dirty glass 0/3 砍掉；unfinished edge 2/2 全對續留。詞彙表的問題與模型的問題
# 分開驗證：這版砍 0% 命中詞，留有命中詞，重跑對照。
# 2026-09-02 82 框人審後定案（precision 10%）：protective corner guard 0/36、
# standing water 0/9、spider web 0/5、floor protective cover 0/3 全砍——「保護」
# 詞彙組在這批全滅。且 wrong 均分(0.401) > correct 均分(0.381)，調門檻無效
# （thr 0.40 → precision 反降為 3%）：分數沒有鑑別力，問題在模型不在門檻。
# 存活：unfinished edge 3/4、rusty metal 3/10（具體物件詞 > 抽象狀態詞）、
# crack in the floor 1/2。v3 只留存活組＋物件型詞。
# 2026-09-03 base 模型（同 v3 詞彙表）對照跑全量：crack 2→32 張大幅回血，但
# damage/scratch/paint 依然死透、rust 84 框（僅 5 張真標鏽蝕）假陽性照樣氾濫——
# 換大模型救不了死詞、也救不了 rust 的假陽性，問題在詞彙選字不在模型量級。
# 用 manifest description 撈已知真陽性照片（scratch 9 張/paint 11 張/damage 8 張）
# 小樣本測新詞：damage/scratch/paint 全部改「具體物件詞」（door frame／floor tile／
# wall）掛動詞子句後，6/6 真陽性全命中（原詞 0 命中）——但物件詞本身可能跟 rust
# 同個毛病（見得到物件、判不出狀態），v4 只是讓詞「活過來」，precision 仍待全量
# 人審驗證，不能只看真陽性測試。
PATTERN_PHRASES: dict[str, list[str]] = {
    "crack": ["crack", "crack in the wall", "crack in the floor"],
    "gap_finish": ["gap between tiles", "gap", "hole in the wall", "unfinished edge"],
    "damage": ["cracked floor tile", "damaged floor tile", "hole in the floor"],
    "scratch": ["scratched door frame", "scratched wood surface", "scuff mark"],
    "rust": ["rust", "rusty metal", "rusty faucet", "corrosion"],
    "paint": ["peeling paint on wall", "paint peeling off wall", "faded paint patch"],
    "dirt_residue": ["stain", "dust", "smudge", "debris", "garbage"],
}

# phrase（小寫）→ pattern 回查表；GDINO 回的 label 就是命中的短語
PHRASE2PATTERN: dict[str, str] = {p.lower(): pat for pat, phrases in PATTERN_PHRASES.items() for p in phrases}


def resolve_pattern(phrase: str) -> str | None:
    """phrase → pattern，含精確比對失敗時的救援。

    2026-09-03 發現：GDINO post-process 常把 prompt 短語截斷/疊字回傳（如
    "scratched wood surface" 只回 "wood surface"，或同 pattern 內兩個相似短語
    共用尾詞時回傳疊字垃圾如 "floor tile floor tile"），只用字典精確比對會把
    真命中當「認不得」全丟——先前 v1~v3 好幾個「死詞」疑似被這個過濾誤殺，不是
    真的詞彙選字問題。救援策略改用**詞集合**比對（忽略順序與重複詞）：回傳片段
    的詞集合是某個完整短語詞集合的子集（或反之），視為命中；只有唯一候選 pattern
    才採信（prompt 把全部 pattern 的短語接成一條，跨 pattern 誤配的風險真實存在）。
    """
    exact = PHRASE2PATTERN.get(phrase)
    if exact:
        return exact
    words = set(phrase.split())
    candidates = {
        pat
        for pat, phrases in PATTERN_PHRASES.items()
        for p in phrases
        if (pw := set(p.lower().split())) and (words <= pw or pw <= words)
    }
    return candidates.pop() if len(candidates) == 1 else None


MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TINY_LIST = ROOT / "data" / "field_reports" / "derived" / "tiny_images.json"


def load_manifest() -> list[dict[str, str]]:
    with open(FR_MANIFEST, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_prompt(patterns: list[str] | None = None) -> str:
    """把詞彙表接成 GDINO 的一條 prompt（短語間用 ' . ' 分隔）。"""
    keys = patterns or list(PATTERN_PHRASES)
    phrases = [p for k in keys for p in PATTERN_PHRASES[k]]
    return " . ".join(phrases)


class GdinoRunner:
    """GDINO-tiny 的薄封裝：device 自動偵測（mps 失敗/NaN 退 cpu）。"""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device or ("mps" if self._mps_ok() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model = model.to(self.device).eval()
        self._image_cls = Image.Image  # 僅標記用，避免頂層 import

        # 暖機：mps 上已知會炸（NaN / placeholder storage），先打一發再決定
        if self.device == "mps":
            try:
                probe = Image.new("RGB", (64, 64), "white")
                self.infer(probe, ["crack"], 0.35, 0.25)
            except Exception:
                self.device = "cpu"
                self.model = self.model.to("cpu")

    @staticmethod
    def _mps_ok() -> bool:
        try:
            import torch

            return torch.backends.mps.is_available()
        except Exception:
            return False

    def infer(
        self,
        image,  # PIL.Image
        patterns: list[str] | None,
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        """跑一張（詞彙表模式）：只收 PHRASE2PATTERN 認得的短語，其餘略過。"""
        prompt = build_prompt(patterns)
        return self._infer_raw(image, prompt, box_threshold, text_threshold, keep_unknown=False)

    def infer_raw(
        self,
        image,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        """跑一張（任意 prompt）：認不得的短語標 pattern='custom'，demo 現場推理用。"""
        return self._infer_raw(image, prompt, box_threshold, text_threshold, keep_unknown=True)

    def _infer_raw(
        self,
        image,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        keep_unknown: bool,
    ) -> list[dict[str, Any]]:
        """底層：回傳 AI_GUESS 候選框（座標=原圖像素 xyxy）。"""
        import torch

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes: list[dict[str, Any]] = []
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"], strict=False):
            phrase = str(label).strip().lower()
            pattern = resolve_pattern(phrase)
            if pattern is None:
                if not keep_unknown:  # 唯一候選都找不到——真的認不得，詞彙表模式下略過
                    continue
                pattern = "custom"
            x1, y1, x2, y2 = [round(float(v)) for v in box.tolist()]
            boxes.append(
                {
                    "pattern": pattern,
                    "phrase": phrase,
                    "box": [x1, y1, x2, y2],
                    "score": round(float(score), 3),
                    "status": "AI_GUESS",
                }
            )
        return boxes


def run(args: argparse.Namespace) -> Path:
    from PIL import Image

    manifest = load_manifest()
    tiny = set()
    if TINY_LIST.exists() and not args.include_tiny:
        tiny = {r["file"] for r in json.loads(TINY_LIST.read_text())}

    rows = [r for r in manifest if r["file"] not in tiny]
    if args.all:
        sample = rows
    elif args.sample:
        import random

        rng = random.Random(args.seed)
        sample = rng.sample(rows, min(args.sample, len(rows)))
    else:
        sample = rows

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "data" / "field_reports" / "derived" / "gdino" / "runs" / run_id
    ann_dir = run_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    runner = GdinoRunner(model_id=args.model)
    print(
        f"[run {run_id}] model={args.model} photos={len(sample)} device={runner.device} "
        f"box_thr={args.box_threshold} text_thr={args.text_threshold}"
    )

    per_pattern: dict[str, dict[str, Any]] = {
        pat: {"photos_with": 0, "boxes": 0, "score_sum": 0.0} for pat in PATTERN_PHRASES
    }
    per_phrase: dict[str, int] = {}
    no_hit = 0
    t0 = time.time()

    for i, r in enumerate(sample, 1):
        path = FR_PHOTOS / r["file"]
        with Image.open(path) as im:
            image = im.convert("RGB")
        ms0 = time.time()
        boxes = runner.infer(image, None, args.box_threshold, args.text_threshold)
        ms = round((time.time() - ms0) * 1000)

        photo_id = r["fileId"]
        ann = {
            "photoId": photo_id,
            "file": r["file"],
            "reportNo": r["reportNo"],
            "obsDisplayId": r["obsDisplayId"],
            "photoDisplayId": r["photoDisplayId"],
            "description": r["description"],
            "site": r["site"],
            "width": image.width,
            "height": image.height,
            "device": runner.device,
            "ms": ms,
            "boxes": boxes,
        }
        (ann_dir / f"{photo_id}.json").write_text(
            json.dumps(ann, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        if not boxes:
            no_hit += 1
        for b in boxes:
            per_pattern[b["pattern"]]["photos_with"] += 0  # 下方 photos_with 用 set 算
            per_pattern[b["pattern"]]["boxes"] += 1
            per_pattern[b["pattern"]]["score_sum"] += b["score"]
            per_phrase[b["phrase"]] = per_phrase.get(b["phrase"], 0) + 1
        # photos_with 要去重，改用 set
        for pat in {b["pattern"] for b in boxes}:
            per_pattern[pat]["photos_with"] += 1

        if i % 25 == 0 or i == len(sample):
            print(f"  {i}/{len(sample)} … boxes so far {sum(v['boxes'] for v in per_pattern.values())}")

    total_ms = round((time.time() - t0) * 1000)
    summary = {
        "runId": run_id,
        "modelId": args.model,
        "device": runner.device,
        "photos": len(sample),
        "excludedTiny": len(tiny),
        "boxThreshold": args.box_threshold,
        "textThreshold": args.text_threshold,
        "vocabulary": PATTERN_PHRASES,
        "elapsedMs": total_ms,
        "msPerPhoto": round(total_ms / max(len(sample), 1)),
        "photosWithNoBox": no_hit,
        "perPattern": {
            pat: {
                "photosWith": v["photos_with"],
                "boxes": v["boxes"],
                "avgScore": round(v["score_sum"] / v["boxes"], 3) if v["boxes"] else 0.0,
            }
            for pat, v in per_pattern.items()
        },
        "perPhrase": dict(sorted(per_phrase.items(), key=lambda kv: -kv[1])),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n[done] {run_dir}  elapsed={total_ms}ms  avg={summary['msPerPhoto']}ms/張")
    print(f"{'樣態':<14}{'有框照片':>8}{'框數':>6}{'均分':>7}")
    for pat, v in summary["perPattern"].items():
        print(f"{pat:<14}{v['photosWith']:>8}{v['boxes']:>6}{v['avgScore']:>7}")
    print(f"no-hit photos: {no_hit}/{len(sample)}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sample", type=int, default=30, help="隨機抽 N 張（預設 30）")
    g.add_argument("--all", action="store_true", help="跑全部（排除 tiny）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--include-tiny", action="store_true", help="包含長邊<=640 的 15 張")
    ap.add_argument("--device", choices=["cpu", "mps"], default=None)
    ap.add_argument("--model", default=MODEL_ID, help="HF model id（換大模型驗證是模型還是詞彙表問題）")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
