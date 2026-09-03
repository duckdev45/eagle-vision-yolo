"""合併 MBDD2025 + GYU-DET → 單一缺失偵測基準資料集（imgsz/tiling 定案後的下一步）。

類別映射（四組對齊 + MBDD 獨有 bulge + GYU 獨有 honeycomb/exposed_rebar，共 7 類）：

    crack/cracks → crack
    leakage/Seepage → leakage
    corrosion/rust → corrosion
    abscission/spalling → abscission
    bulge（MBDD 獨有）
    honeycomb_surface / exposed_rebar（GYU 獨有）

輸出是 derived 層：只建 symlink + 重新映射過 class id 的 label 副本，
不碰兩邊的原始標註（同 public_yolo/README.md「不改動原始標註」的約定）。
每次執行整個重建，不用 diff/patch。

    uv run src/fr_yolo_merge.py

輸出：data/public_yolo/merged/{train,val,test}/{images,labels}/、merged.yaml
"""
from __future__ import annotations

import shutil
from pathlib import Path

from paths import ROOT

MBDD_ROOT = Path("/Users/duck/datasets/MBDD2025/MBDD2025")
GYU_ROOT = Path("/Users/duck/datasets/gyu-det")
OUT = ROOT / "data" / "public_yolo" / "merged"

CLASSES = ["crack", "leakage", "corrosion", "abscission", "bulge", "honeycomb_surface", "exposed_rebar"]

# 舊 id（各自 dataset 的 yaml）→ 新 id（CLASSES 的 index）
MBDD_MAP = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}  # crack,leakage,abscission,corrosion,bulge
GYU_MAP = {0: 1, 1: 0, 2: 6, 3: 5, 4: 2, 5: 3}  # Seepage,cracks,exposed_rebar,honeycomb,rust,spalling


def remap_lines(src: Path, mapping: dict[int, int]) -> str:
    if not src.exists():
        return ""
    out = []
    for line in src.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        old = int(p[0])
        if old not in mapping:
            continue  # 該類沒對到映射表就整框丟（目前映射涵蓋所有已知類，理論不會發生）
        out.append(" ".join([str(mapping[old]), *p[1:]]))
    return "\n".join(out) + ("\n" if out else "")


def link_pair(img_src: Path, label_text: str, split_dir: Path, name: str) -> None:
    (split_dir / "images" / f"{name}{img_src.suffix}").symlink_to(img_src)
    (split_dir / "labels" / f"{name}.txt").write_text(label_text)


def build_mbdd(split_dir: Path, list_file: str) -> int:
    ids = [
        line.strip() for line in (MBDD_ROOT / list_file).read_text().splitlines() if line.strip()
    ]
    n = 0
    for rel in ids:
        stem = Path(rel).stem
        img = MBDD_ROOT / "JPEGImages" / Path(rel).name
        if not img.exists():
            continue
        lbl_text = remap_lines(MBDD_ROOT / "Labels" / f"{stem}.txt", MBDD_MAP)
        link_pair(img, lbl_text, split_dir, f"mbdd_{stem}")
        n += 1
    return n


def build_gyu(split_dir: Path, subdir: str) -> int:
    src_dir = GYU_ROOT / subdir
    n = 0
    for img in sorted((src_dir / "images").glob("*")):
        lbl_text = remap_lines(src_dir / "labels" / f"{img.stem}.txt", GYU_MAP)
        link_pair(img, lbl_text, split_dir, f"gyu_{img.stem}")
        n += 1
    return n


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)  # 全 derived、可重建，不需要保留舊版

    splits = [("train", "train.txt", "train"), ("val", "val.txt", "valid"), ("test", "test.txt", "test")]
    counts = {}
    for split, mbdd_list, gyu_dir in splits:
        split_dir = OUT / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        n_mbdd = build_mbdd(split_dir, mbdd_list)
        n_gyu = build_gyu(split_dir, gyu_dir)
        counts[split] = (n_mbdd, n_gyu)
        print(f"{split}: MBDD2025 {n_mbdd} + GYU-DET {n_gyu} = {n_mbdd + n_gyu}")

    yaml_text = (
        "# 合併資料集：MBDD2025（UAV 外牆）+ GYU-DET（橋梁近拍）— fr_yolo_merge.py 產生，勿手改\n"
        f"path: {OUT}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES)) + "\n"
    )
    (OUT / "merged.yaml").write_text(yaml_text)
    total = sum(a + b for a, b in counts.values())
    print(f"\nwrote {OUT / 'merged.yaml'} | 共 {total} 張 | {len(CLASSES)} 類")


if __name__ == "__main__":
    main()
