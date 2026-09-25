import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
import pandas as pd

import paths
import split as split_mod
from core.review_utils import build, scores
from labels import Labeler, human_refs, load_reviews
from sync import _truthy

df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
if "active" in df:
    df = df[_truthy(df.active)]
lab = Labeler.load()
base = lab.apply(df, drop_small=False)
out = base.join(human_refs(base, lab))
sc, test_ids = scores("siglip", split_mod.current())
q = build(out, lab, sc, test_ids)
done = load_reviews()

# 已裁 265 筆的訊號組合 → 人類最終選了規則標籤的比例（校準自動裁決的精度）
q["final"] = q.fileId.map(done)
qd = q[q.final.notna()].copy()


def sigs(r):
    s = []
    if isinstance(r.clsChips, str):
        s.append("chips同工種" if r.clsChips.split("-")[0] == r.cls.split("-")[0] else "chips反對")
    if isinstance(r.specTrade, str):
        s.append("specKey同" if r.specTrade == r.cls.split("-")[0] else "specKey反對")
    if isinstance(r.gemNorm, str):
        s.append("gemini同意" if r.gemNorm == r.cls else "gemini反對")
    if isinstance(r.mPred, str):
        s.append("模型同意" if r.mPred == r.cls else "模型反對")
    return "+".join(s) or "無訊號"


qd["sigs"] = qd.apply(sigs, axis=1)
qd["rule_wins"] = qd.final == qd.cls

g = qd.groupby("sigs").agg(n=("rule_wins", "size"), wins=("rule_wins", "sum"))
g["rate"] = g.wins / g.n
print("=== 已裁 265 筆：每種訊號組合下「人類最終確認規則標籤」的比例 ===")
print("(這就是自動裁決的歷史精度校準表)")
print()
for k, row in g.sort_values("n", ascending=False).iterrows():
    print(f"  {row['n']:4}  規則勝 {row['rate']:6.0%}   {k}")
print()
n = g.n.sum()
wins = g.wins.sum()
print(f"  合計 {n} 筆，規則勝 {wins} ({wins / n:.0%})")
print()
# 反對訊號數 vs 規則勝率
qd["n_anti"] = qd.sigs.str.count("反對")
print("=== 反對訊號數 → 規則勝率（這是自動裁決的分界線）===")
for k in sorted(qd.n_anti.unique()):
    sub = qd[qd.n_anti == k]
    print(f"  {k} 個反對訊號: n={len(sub):4}  規則勝 {(sub.rule_wins).mean():6.0%}")
