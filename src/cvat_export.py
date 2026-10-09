"""CVAT 匯出：把要標缺失框的照片按「案場×日期」打成 task manifest。

CVAT 上線後的冷啟動流程（ROADMAP 執行序 #6，初標全人工）：
照片 → 本腳本產出 task manifest → cvat-cli 建 task → 主任/標註者畫缺失框 →
匯出 CVAT 1.1 XML → `src/cvat_import.py` 寫回 data/defects.csv。

**為什麼 task 要按案場×日期切**：照片型缺失同案場同日有視覺相關性（同工班、
同批材料、同光源），匯出檔必須帶分組資訊，之後 split.py 才能整組切——
這是 ROADMAP 執行序 #6 的前置條件，不是方便而已。

    uv run src/cvat_export.py --source pms --sample 300     # 日報照抽 300 張
    uv run src/cvat_export.py --source golden               # 用 G1 的缺失框子集
    uv run src/cvat_export.py --source field --all          # 樂氧森全量

產出：data/cvat/<batch>/
    manifest.csv   fileId,taskId(=案場×日期),photoPath,title,constrName,reportDate
    tasks.json     {taskId: [fileId, ...]}（cvat-cli 一次建一個 task）
    README.md      當下環境的實際指令（照片複製 + cvat-cli 建議參數）
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from core import paths


def pool_pms(sample: int, seed: int) -> pd.DataFrame:
    """日報照。分層：每個案場×日期組內均勻抽，避免一個大組吃掉整個額度。"""
    df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
    df = df[df.active.astype(str).str.lower().isin(["true", "1"])]
    from labels import Labeler

    df = Labeler.load().apply(df)  # 有分類的才進冷啟動語料（junk/未來日已在裡面排掉）
    df = df.assign(group=df.constrId + "|" + df.reportDate.astype(str))
    if sample and sample < len(df):
        rng = random.Random(seed)
        per = max(1, sample // max(1, df.group.nunique()))
        picked = []
        for _g, rows in df.groupby("group"):
            idx = rows.index.tolist()
            rng.shuffle(idx)
            picked += idx[:per]
        if len(picked) < sample:  # 大組均勻完還有名額 → 從剩餘裡補
            rest = [i for i in df.index if i not in set(picked)]
            rng.shuffle(rest)
            picked += rest[: sample - len(picked)]
        df = df.loc[picked]
    return df


def pool_field(all_photos: bool, sample: int, seed: int) -> pd.DataFrame:
    """樂氧森缺失語料：全有缺失描述（弱標籤），排除 tiny 圖。"""
    df = pd.read_csv(
        paths.FIELD_REPORTS / "raw" / "manifest.csv", dtype=str, keep_default_na=False, na_values=[""]
    )
    tiny_path = paths.FIELD_REPORTS / "derived" / "tiny_images.json"
    if tiny_path.exists():
        tiny = {t["file"].rsplit(".", 1)[0] for t in json.loads(tiny_path.read_text())}
        df = df[~df.fileId.isin(tiny)]
    if not all_photos and sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    return df.assign(group="樂氧森|" + df.reportDate.astype(str))


def pool_golden() -> pd.DataFrame:
    """G1 缺失框子集（src/g1_sample.py 的產物）——黃金集與訓練語料同一批照片。"""
    m = paths.GOLDEN / "g1_manifest.csv"
    if not m.exists():
        raise SystemExit("沒有 G1 manifest，先跑 uv run src/g1_sample.py")
    g = pd.read_csv(m, dtype=str)
    g = g[g.subset == "defect"]
    return g.assign(group=g.fileId.map(_golden_group))


def _golden_group(file_id: str) -> str:
    df = pd.read_csv(
        paths.FIELD_REPORTS / "raw" / "manifest.csv", dtype=str, keep_default_na=False, na_values=[""]
    )
    row = df[df.fileId == file_id]
    return "樂氧森|" + (row.reportDate.iloc[0] if len(row) else "unknown")


def photo_path(row: pd.Series) -> Path | None:
    """四個源的實檔位置（golden 子集的照片來自 field_reports）。"""
    p = paths.FR_PHOTOS / f"{row.fileId}.webp"
    if p.exists():
        return p
    for ext in (".webp", ".jpg", ".png"):
        p = paths.PHOTOS / f"{row.fileId}{ext}"
        if p.exists():
            return p
    for ext in (".jpg", ".png", ".webp"):
        p = paths.LEGACY_PHOTOS / f"{row.fileId}{ext}"
        if p.exists():
            return p
    return None


def export(source: str, sample: int, all_photos: bool, seed: int, batch: str) -> Path:
    if source == "pms":
        df = pool_pms(sample, seed)
    elif source == "field":
        df = pool_field(all_photos, sample, seed)
    elif source == "golden":
        df = pool_golden()
    else:
        raise SystemExit(f"未知 source：{source}")

    out = paths.CVAT / batch
    if out.exists():
        raise SystemExit(f"{out} 已存在——換個 batch 名，別覆寫已發出去的標註包")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in df.iterrows():
        p = photo_path(r)
        if p is None:
            continue
        rows.append(
            {
                "fileId": r.fileId,
                "taskId": str(r.get("group", "unknown")),
                "photoPath": str(p),
                "title": str(r.get("title", ""))[:60],
                "constrName": str(r.get("constrName", "") or "樂氧森"),
                "reportDate": str(r.get("reportDate", "")),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)

    tasks: dict[str, list[str]] = {}
    for r in rows:
        tasks.setdefault(r["taskId"], []).append(r["fileId"])
    (out / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")

    # 把照片複製成 fileId.jpg——CVAT 匯入與匯出 XML 的 image name 都用這個名字，
    # 匯回時 name → fileId 的對應才不會斷
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    for r in rows:
        dst = img_dir / f"{r['fileId']}.jpg"
        if not dst.exists():
            shutil.copy(r["photoPath"], dst)

    (out / "README.md").write_text(_readme(batch, tasks), encoding="utf-8")
    sizes = Counter(len(v) for v in tasks.values())
    print(
        f"{len(rows)} 張 → {out}\n"
        f"  tasks {len(tasks)} 個（案場×日期）· 每任務張數 {dict(sorted(sizes.items())[:5])}\n"
        f"  建議：cvat-cli 一次建一個 task，labels 用 core.defects.DEFECT_PATTERNS 的 10 樣態"
    )
    return out


def _readme(batch: str, tasks: dict[str, list[str]]) -> str:
    sample_task = next(iter(tasks), "")
    return f"""# CVAT 標註包 {batch}（{datetime.now().date()}）

images/ 的檔名就是 fileId（.jpg），匯回時靠它對回 manifest。

## labels 設定（CVAT task labels，一行一個，純文字框）
{chr(10).join(paths_defects())}

## 建議 cvat-cli 流程（每個 task 一次）
```bash
# 1) 建 task（label 依上面清單；segment 帶案場×日期，之後 split 重切靠它）
cvat-cli --auth user:pass task create \\
  --name "{batch} {sample_task}" --labels "labels.tsv" \\
  --segment_size 40 {batch}-{sample_task[:8]}

# 2) 上傳該 task 的照片（manifest.csv 撈 taskId 過濾）
cvat-cli task frames {batch}-{sample_task[:8]} --upload-mode copy images/...

# 3) 標完匯出：Export task dataset → CVAT for images 1.1（XML）
#    存到 data/cvat/{batch}/export/ 之後跑：
uv run src/cvat_import.py --xml data/cvat/{batch}/export/*.xml --annotator 你的名字
```

## 標註規則（與 ROADMAP #7 同一組）
- 只框「看得到的缺失狀態」，整框緊貼缺失邊界；不確定是什麼樣態 → 其他（但要框）。
- 一張照片多個缺失就多個框，樣態不同各自開框。
- 紅筆圈（巡查員畫的）照框缺失本體，不要框紅筆——紅筆圈是 shortcut leak。
"""


def paths_defects() -> list[str]:
    from core.defects import DEFECT_PATTERNS

    return DEFECT_PATTERNS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="pms", choices=["pms", "field", "golden"])
    ap.add_argument("--sample", type=int, default=300, help="抽樣上限（--all 時忽略）")
    ap.add_argument("--all", action="store_true", help="不抽樣全量（field 源常用）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", default=datetime.now().strftime("coldstart-%Y%m%d"))
    a = ap.parse_args()
    export(a.source, a.sample, a.all, a.seed, a.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
