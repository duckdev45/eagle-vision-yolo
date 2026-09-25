"""G1 黃金集抽樣（WORKFLOW §6 / ROADMAP 執行序 #7）。

兩個子集、兩份用途：
  trade  工種分類：300–500 張、按當前類別分層（每類保底，弱類不被大宗吃掉）、
         **獨立於 labels.yaml**——抽樣用規則標籤只是選樣手段，黃金答案由兩位
         懂現場的人各標一次，一致度就是人類天花板。
  defect 缺失框子集：樂氧森語料按缺失樣態關鍵字分層，每樣態保底 30 框——
         不足 30 的樣態照抽（訓練語料反正需要），但 manifest 標 gateEligible=0，
         G2 的 per-class gate 對它「本輪不評」（n<30 的 recall 與 WORKFLOW §5
         同一紀律，要下結論 ≥50）。

    uv run src/g1_sample.py --n 400 --floor 20 --cap 30

產出：data/golden/g1_manifest.csv + README.md（標註紀則，給兩位標註者各一份）。
標註結果：trade 子集 → data/golden/labels-{名字}.csv（fileId,cls 兩欄）；
defect 子集 → CVAT 匯出後走 src/cvat_import.py --source g1-A / g1-B。
之後 uv run src/g1_gate.py 算一致度與仲裁清單。
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

import paths

# 缺失描述關鍵字 → 10 樣態（core.defects.DEFECT_PATTERNS）。順序即優先權：
# 「裂縫」先於「縫隙」，否則裂縫整批被「縫」接走；「凹陷」併入 破損/脫落。
PATTERN_KEYWORDS: list[tuple[str, str]] = [
    ("裂縫", r"裂縫|龜裂"),
    ("刮傷/撞痕", r"刮傷|撞痕"),
    ("破損/脫落", r"凹陷|破損|破掉|破皮|缺口|剝落"),
    ("鏽蝕", r"鏽|生鏽"),
    ("滲水/水痕", r"滲水|積水|水痕|壁癌"),
    ("掉漆/漆面", r"掉漆|油漆|批土|漆面"),
    ("髒污/殘留", r"髒|污染|殘留|污漬|粉塵"),
    ("保護不足", r"保護"),
    ("縫隙/收邊", r"縫|收邊"),
    ("不平整", r"不平整|凹凸"),
]


def pattern_of(desc) -> str | None:
    d = str(desc or "")
    for name, pat in PATTERN_KEYWORDS:
        if re.search(pat, d):
            return name
    return None


def trade_pool() -> pd.DataFrame:
    """訓練母體（PMS+legacy 已分類照片），帶當前 split 歸屬資訊。"""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import split as split_mod
    from labels import Labeler, labeled_manifest, legacy_manifest

    lab = Labeler.load()
    df = pd.concat([labeled_manifest(), legacy_manifest()], ignore_index=True)
    out = lab.apply(df, overrides={})
    try:
        cur = split_mod.current()
        sp = json.loads((paths.SPLITS / f"{cur}.json").read_text())
        used = {**{f: "train" for f in sp["train"]}, **{f: "test" for f in sp["test"]}}
        out = out.assign(inCurrentSplit=out.fileId.map(lambda f: used.get(f, "")))
    except (FileNotFoundError, ValueError, KeyError):
        out = out.assign(inCurrentSplit="")
    return out


def sample_trade(pool: pd.DataFrame, n: int, floor: int, seed: int) -> pd.DataFrame:
    """先保底，再按原類別比例分配剩餘名額；用最大餘數補齊整數配額。

    小類不足 floor 時全收；保底總額超過 n 則拒絕，避免默默犧牲某類。
    母體不足 n 時警告並全收。fileId 必須唯一，DataFrame index 可重複。
    """
    if n < 0 or floor < 0:
        raise ValueError("n and floor must be non-negative")
    if not {"fileId", "cls"} <= set(pool.columns):
        raise ValueError("pool requires fileId and cls columns")
    for col in ("fileId", "cls"):
        if not pool[col].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"pool {col} must contain non-empty strings")
    if not pool.fileId.is_unique:
        raise ValueError("pool fileId must be unique; resolve duplicate photos before sampling")
    if len(pool) < n:
        warnings.warn(
            f"Requested n={n}, but pool contains only {len(pool)} photos; returning all available photos",
            UserWarning,
            stacklevel=2,
        )
    if pool.empty:
        return pool.copy()

    # 以 ID 固定母體順序，避免來源列序或重複 index 改變抽樣結果。
    ordered = pool.sort_values("fileId").reset_index(drop=True)
    by_cls = {c: g.sample(frac=1, random_state=seed) for c, g in ordered.groupby("cls")}
    counts = {c: len(g) for c, g in by_cls.items()}
    take = {c: min(floor, size) for c, size in counts.items()}
    minimum = sum(take.values())
    if minimum > n:
        raise ValueError(f"n={n} cannot satisfy floor={floor}: at least {minimum} photos are required")

    rest = min(n, len(pool)) - minimum
    while rest:
        available = [c for c in by_cls if take[c] < counts[c]]
        weight = sum(counts[c] for c in available)
        quotas = {c: divmod(rest * counts[c], weight) for c in available}
        for c in available:
            extra = min(quotas[c][0], counts[c] - take[c])
            take[c] += extra
            rest -= extra
        for c in sorted(available, key=lambda c: (-quotas[c][1], -counts[c], c)):
            if rest and take[c] < counts[c]:
                take[c] += 1
                rest -= 1
        # 類別容量不足留下的名額，下一輪按尚有照片的類別重新分配。

    sel = pd.concat([g.head(take[c]) for c, g in by_cls.items()], ignore_index=True)
    return sel.sample(frac=1, random_state=seed).reset_index(drop=True)


def sample_defect(cap: int, seed: int) -> pd.DataFrame:
    """樂氧森按缺失樣態分層；<cap 的樣態 gateEligible=0（本輪不評，照抽照標）。"""
    df = pd.read_csv(
        paths.FIELD_REPORTS / "raw" / "manifest.csv", dtype=str, keep_default_na=False, na_values=[""]
    )
    tiny = {
        t["file"].rsplit(".", 1)[0]
        for t in json.loads((paths.FIELD_REPORTS / "derived" / "tiny_images.json").read_text())
    }
    df = df[~df.fileId.isin(tiny)].copy()
    df["stratum"] = df.description.map(pattern_of)
    df = df[df.stratum.notna()]
    picked = []
    for _s, g in df.groupby("stratum"):
        g = g.sample(frac=1, random_state=seed)
        picked.append(g.head(cap))
        if len(g) > cap:
            picked.append(g.iloc[cap:].head(max(0, cap // 2)))  # 大樣態多抓一半，框多的照片多練
    sel = pd.concat(picked)
    counts = sel.stratum.value_counts()
    sel = sel.assign(gateEligible=sel.stratum.map(lambda s: int(counts[s] >= 30)))
    return sel.sample(frac=1, random_state=seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400, help="trade 子集目標張數（WORKFLOW：300–500）")
    ap.add_argument("--floor", type=int, default=20, help="trade 每類保底")
    ap.add_argument("--cap", type=int, default=30, help="defect 每樣態上限（=保底門檻）")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    paths.GOLDEN.mkdir(parents=True, exist_ok=True)

    pool = trade_pool()
    trade = sample_trade(pool, a.n, a.floor, a.seed).assign(
        subset="trade", stratum=lambda d: d.cls, gateEligible=1
    )
    defect = sample_defect(a.cap, a.seed)
    defect["subset"], defect["inCurrentSplit"] = "defect", ""

    manifest = pd.DataFrame(
        {
            "fileId": pd.concat([trade.fileId, defect.fileId]),
            "subset": pd.concat([trade.subset, defect.subset]),
            "stratum": pd.concat([trade.stratum, defect.stratum]),
            "gateEligible": pd.concat([trade.gateEligible, defect.gateEligible]).astype(int),
            "inCurrentSplit": pd.concat([trade.inCurrentSplit, defect.inCurrentSplit]),
            "reportDate": pd.concat([trade.reportDate, defect.reportDate]),
            "title": pd.concat([trade.title, defect.description.fillna("")]),
        }
    )
    out = paths.GOLDEN / "g1_manifest.csv"
    manifest.to_csv(out, index=False)
    (paths.GOLDEN / "README.md").write_text(_readme(manifest), encoding="utf-8")

    print(f"G1 manifest → {out}")
    print(
        f"  trade {len(trade)} 張 / {trade.stratum.nunique()} 類（每類 {trade.stratum.value_counts().min()}~{trade.stratum.value_counts().max()}）"
    )
    dc = defect.stratum.value_counts()
    print(f"  defect {len(defect)} 張 / {len(dc)} 樣態：{dict(dc)}")
    ne = dc[dc < 30]
    if len(ne):
        print(f"  本輪不評（<30 框）：{dict(ne)}")
    return 0


def _readme(m: pd.DataFrame) -> str:
    tr = m[m.subset == "trade"]
    de = m[m.subset == "defect"]
    return f"""# G1 黃金集標註說明（{datetime.now().date()} 產出）

manifest：g1_manifest.csv（fileId, subset, stratum, gateEligible, inCurrentSplit）。
**兩位標註者各自獨立標一份**，不對答案、不討論。

## trade 子集（{len(tr)} 張）
- 看照片選「實際在拍什麼」：一個工種-施作內容類名（清單 = labels.yaml 當時的類別，
  但你的答案不進 labels.yaml——這份考卷獨立於規則）。
- 存成 data/golden/labels-{{你的名字}}.csv：fileId,cls 兩欄。
- 拍多個工種 → 選畫面占比最大那個；真的並列 → 挑一個並在備註欄寫下來。

## defect 子集（{len(de)} 張）
- CVAT 上框缺失（10 樣態：縫隙/收邊、髒污/殘留、破損/脫落、滲水/水痕、不平整、
  保護不足、刮傷/撞痕、鏽蝕、裂縫、掉漆/漆面），框緊貼缺失邊界。
- 匯出 XML → `uv run src/cvat_import.py --xml ... --annotator {{名字}} --source g1-{{A|B}}`
- 紅筆圈照框缺失本體，不要框紅筆。

## 裁決規則（標完之後的事，`src/g1_gate.py`）
- trade：兩人同類 = 一致；不同類 = 分歧 → 主任仲裁，仲裁結果就是黃金答案。
- defect：框 IoU ≥ 0.5 且同樣態 = 一致；一人有、一人無 = 模糊 → 主任仲裁。
- gateEligible=0 的樣態「本輪不評」：樣本 <30，recall 與 WORKFLOW §5 同一紀律，
  要下結論 ≥50。照抽照標（訓練語料反正需要），只是不進 G2 的 per-class gate。
"""


if __name__ == "__main__":
    raise SystemExit(main())
