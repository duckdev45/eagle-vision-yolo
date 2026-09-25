import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
import pandas as pd

import paths
import split as split_mod
from core.review_utils import scores
from labels import Labeler, human_refs, load_reviews
from sync import _truthy

df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
if "active" in df:
    df = df[_truthy(df.active)]
lab = Labeler.load()
base = lab.apply(df, drop_small=False)
out = base.join(human_refs(base, lab))

# 用「工種層級」重算 chips 反對（修正後）
gem = out.predWorkItem.map(lambda t: (lab.label(t) or lab.fallback) if isinstance(t, str) and t else None)
o = out.assign(gemNorm=gem)

sc, test_ids = scores("siglip", split_mod.current())
o = o.assign(mPred=o.fileId.map(lambda f: (sc.get(f) or (None,))[0]))

chips, spec = o.get("clsChips"), o.get("specTrade")
F = pd.Series(False, index=o.index)
d_chips = (
    (chips.notna() & (chips != lab.fallback) & (chips.str.split("-").str[0] != o.cls.str.split("-").str[0]))
    if chips is not None
    else F
)
d_spec = spec.notna() & (spec != o.cls.str.split("-").str[0]) if spec is not None else F
d_gem = gem.notna() & (gem != o.cls)
d_model = o.mPred.notna() & (o.mPred != o.cls)
is_test = o.fileId.isin(test_ids)
unsure = o.mMargin.notna() & (o.mMargin < 0.25) & is_test if "mMargin" in o else F

tier = pd.Series(0, index=o.index)
tier[unsure] = 1
tier[d_gem] = 2
tier[d_model & d_gem] = 3
tier[d_chips | d_spec] = 4
q = o.assign(tier=tier, why="", isTest=is_test)
done = load_reviews()
pend = q[(q.tier > 0) & (~q.fileId.isin(done))].copy()

n_anti = (
    (pend[["clsChips"]].notna().any(axis=1) & d_chips.reindex(pend.index)).astype(int)
    + (d_spec.reindex(pend.index)).astype(int)
    + (d_gem.reindex(pend.index)).astype(int)
    + (d_model.reindex(pend.index)).astype(int)
)
pend["n_anti"] = n_anti
# 「只有 Gemini 反對」= d_gem 且其他三個都不反對
only_gem = (
    d_gem.reindex(pend.index)
    & ~d_chips.reindex(pend.index)
    & ~d_spec.reindex(pend.index)
    & ~d_model.reindex(pend.index)
)
zero = pend.n_anti == 0
print(f"修正後待裁 {len(pend)}")
print(f"  零反對訊號        : {zero.sum():4}  ({zero.mean():.0%})  ← 規則標籤無人挑戰")
print(
    f"  只有 Gemini 反對  : {only_gem.sum():4}  ({only_gem.mean():.0%})  ← Gemini 歷史對率 16% vs 規則 100%"
)
print(f"  ─ 可自動確認小計  : {(zero | only_gem).sum():4}  ({(zero | only_gem).mean():.0%})")
print(f"  真有人工訊號反對  : {(~(zero | only_gem)).sum():4}  ← 仍需人/AI 裁決")
print()
print("=== 真正需要判斷的那批，反對訊號組合 ===")
hard = pend[~(zero | only_gem)]
print(f"  chips反對: {d_chips.reindex(hard.index).sum()}")
print(f"  specKey反對: {d_spec.reindex(hard.index).sum()}")
print(f"  模型反對: {d_model.reindex(hard.index).sum()}")
print(f"  模型難分(測試集): {unsure.reindex(hard.index).sum()}")
