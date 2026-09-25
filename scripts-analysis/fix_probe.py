import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
import json
from collections import Counter

import numpy as np
import pandas as pd

import paths
import split as split_mod
from core.review_utils import embedding_index, neighbors
from core.review_utils import orphans as build_orphans
from labels import Labeler, human_refs, load_reviews
from sync import _truthy

df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
if "active" in df:
    df = df[_truthy(df.active)]
lab = Labeler.load()
base = lab.apply(df, drop_small=False)
out = base.join(human_refs(base, lab))
done = load_reviews()

d = out[(out.clsChips.notna()) & (out.clsChips != lab.fallback) & (out.clsChips != out.cls)].copy()
print("tier4 全部(含已裁):", len(d), " 待裁:", len(d[~d.fileId.isin(done)]))
# trade-level compare: 只比前綴（工種）
d_trade = d[d.clsChips.str.split("-").str[0] != d.cls.str.split("-").str[0]]
print("→ 工種層級仍不一致:", len(d_trade), " 待裁:", len(d_trade[~d_trade.fileId.isin(done)]))
print("→ 工種一致、僅施作內容(階段)不一致 = 假異議:", len(d) - len(d_trade))
print()
print("工種層級仍不一致的（真異議）分布：")
print(d_trade.clsChips.value_counts().head(10).to_string())
print()
print("=== 孤兒 kNN 投票可解性 ===")

o = build_orphans(out, lab)
o = o[~o.fileId.isin(done)]
print("孤兒待裁:", len(o))
try:
    cur = split_mod.current()
    labels = json.loads((paths.SPLITS / f"{cur}.json").read_text())["labels"]
except Exception as e:
    labels = {}
    print("  (讀不到 split labels:", e, ")")
try:
    ids, emb = embedding_index("siglip")
    have = set(ids)
except Exception as e:
    have = set()
    print("  (embedding 讀不到:", e, ")")

vote_top1_agree = 0
sim_mean = []
for r in o.itertuples():
    nb = neighbors(r.fileId, labels, _index=(ids, emb)) if r.fileId in have else []
    if not nb:
        continue

    # 加權投票：cosine 當權重
    w = Counter()
    for _f, c, s in nb:
        w[c] += s
    top = w.most_common(1)[0]
    topcls, ws = top
    tot = sum(w.values())
    sim_mean.append(nb[0][2])
    print(
        f"  {r.fileId[:8]} {r.title[:24]:24} → {topcls:12} w={ws / tot:.2f} (top1sim={nb[0][2]:.2f}) n={len(nb)}"
    )
if sim_mean:
    print(f"  平均最近鄰 cosine: {np.mean(sim_mean):.3f}")
