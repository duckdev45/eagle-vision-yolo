"""抓 PMS 的工程分類樹（請款項目）到 reference/billing_items.json。

領域專家歸納出來的三層樹，是核對 labels.yaml 歸類的外部依據。
不進 labels 流程——它是請款用的，顆粒度與「照片看得出什麼」不同（見 README）。

    uv run src/taxonomy.py            # 更新快取
    uv run src/taxonomy.py 矽利康      # 查某個詞掛在哪一支
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import paths
from api import Pms

CACHE = paths.ROOT / "reference" / "billing_items.json"


def fetch() -> list:
    p = Pms()
    p.login()
    tree = p._get("/v1/insp-cat-item")
    p.close()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(tree, ensure_ascii=False, indent=1))
    return tree


def _nm(node) -> str:
    n = node["name"]
    return (n.get("zh-TW") if isinstance(n, dict) else n).strip()


def flatten(tree=None) -> list[tuple[str, str, str | None]]:
    """→ [(大類, 中類, 小類 or None)]"""
    tree = tree if tree is not None else json.loads(CACHE.read_text())
    out = []
    for a in tree:
        for b in a.get("children") or []:
            kids = b.get("children") or []
            out.append((_nm(a), _nm(b), None)) if not kids else None
            out += [(_nm(a), _nm(b), _nm(c)) for c in kids]
    return out


if __name__ == "__main__":
    rows = flatten(fetch() if not CACHE.exists() else None)
    if len(sys.argv) > 1:
        kw = sys.argv[1]
        for l1, l2, l3 in rows:
            if kw in (l3 or l2):
                print(f"{l1} > {l2}" + (f" > {l3}" if l3 else ""))
    else:
        fetch()
        rows = flatten()
        print(
            f"{len({r[0] for r in rows})} 大類 / {len({r[:2] for r in rows})} 中類 / "
            f"{len([r for r in rows if r[2]])} 小類 → {CACHE}"
        )
