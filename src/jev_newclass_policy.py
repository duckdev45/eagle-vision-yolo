"""「這個工種該不該現在就寫進 labels.yaml」——把判斷交給 Jev，分支留在程式碼。

問題背景（2026-09-19）：QS 樹有 77 支標準，labels.yaml 只有 37 條規則。
那些「樹上有、規則沒有」的工種，現在的照片被既有規則掃進某個鄰居類別。
該不該現在就替它們開規則？

**決策拆成三個輸入，只有第一個交給模型：**

  (a) 這批標題描述的施作，跟現在歸的那個類別是不是同一件事？   → Jev（看標題）
  (b) 公司標準樹上它是不是獨立一支？                          → 本機查，**不出境**
  (c) 張數夠不夠 min_class_size？                             → 本機算術

(b) 是公司營運資料（reference/ 在 .gitignore，從 git 歷史移除過），不送 API。
(c) 是算術，jaggedness 明說留在程式碼。模型只回答語義問題，這是它唯一的強項。

政策由 (a)(b)(c) 三者在 `decide()` 裡合成——模型不會看到「該不該開類」這個問題，
所以它也無從迎合。這是 skill 裡「把它當聰明的 if」的完整形態。

state 只送：標題範例 + 目前歸到的類名。工地名、constrId、照片路徑、QS 樹一律不送。
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

import jev_questions as Q
import paths
from core.labeler import Labeler

load_dotenv()

OUT = paths.REPORTS_OUT / "2026-09-19-jev"
CACHE = paths.DERIVED / "jev_policy_cache.jsonl"
WORKERS = 8

# 同一件事 / 不同件事的判準線。skill 的三段式：0.90 以上自動採用、0.50 以下不猜。
SAME_CONF = Q.AUTO_ACCEPT_CONF
FLOOR = Q.REVIEW_FLOOR_CONF


def candidates() -> list[dict]:
    """本機算出候選：規則掃得到、但沒有專屬類別的工種。

    探測關鍵字只在本機用來撈標題與算張數，不出境。
    """
    lab = Labeler.load()
    pms = lab.drop_excluded(pd.read_csv(paths.MANIFEST))
    pms = pms.assign(cls=pms.title.map(lab.label))
    lg = pd.read_csv(paths.LEGACY_MANIFEST, dtype=str)
    lg = lg.assign(cls=lg.title.astype(str).map(lab.label))
    al = pd.concat([pms[["title", "cls"]], lg[["title", "cls"]]], ignore_index=True)

    probe = yaml.safe_load((paths.ROOT / "data" / "qs_probe.yaml").read_text())
    out = []
    for name, spec in probe.items():
        hit = al[al.title.astype(str).str.contains(spec["pattern"], na=False, regex=True)]
        if not len(hit):
            continue
        now = hit.cls.value_counts(dropna=False)
        top = now.index[0]
        out.append(
            {
                "工種": name,
                "photos": len(hit),
                "titles": sorted(set(hit.title.astype(str)))[:10],
                "現歸": "（規則丟棄）" if pd.isna(top) else str(top),
                "現歸張數": int(now.iloc[0]),
                "散佈類數": int(now.notna().sum()),
                "_independent": bool(spec["independent"]),  # 本機閘門，不進 state
            }
        )
    return sorted(out, key=lambda r: -r["photos"])


def build_questions() -> dict:
    """兩題：語義同一性 + 是否為行政活動。都不問「該不該開類」。"""
    from typesafe_sdk import Choice, Noul

    return {
        "same_work": Choice(
            instructions={
                "question": "這批工地照片標題描述的施作內容，與「目前歸屬類別」是同一種工程作業嗎？",
                "focus": "看標題裡的**施作動作**。標題同時出現兩個工種詞時，以動作的主體為準——"
                "例如「油漆踢腳」的動作是油漆，「踢腳板安裝」的動作是安裝踢腳板。",
            },
            criteria={
                "same": {
                    "what": "是同一種作業。標題的施作動作就是目前歸屬類別在做的事",
                    "examples": ["木扶手油漆 歸 油漆（動作是油漆，不是做扶手）"],
                },
                "different": {
                    "what": "不是同一種作業。標題描述的是另一個工種的施作，只是被關鍵字掃進來",
                    "examples": ["土方清運 被歸 清潔（動作是土方運棄，不是打掃）"],
                },
                "mixed": {
                    "what": "這批標題本身就混了多種作業，無法一概而論",
                    "examples": ["同一批裡有本體安裝、有油漆收尾、有缺失改善"],
                },
            },
        ),
        # junk 規則的對照組。放樣那 58 張就是靠這題驗證。
        "not_construction": Noul(
            instructions="這批標題描述的是查驗/放樣/行政活動（自主查驗、人數清點、會勘、"
            "放樣、進度回報）而不是實際施工畫面。"
        ),
    }


def decide(row: dict, ans: dict, min_size: int) -> dict:
    """三輸入合成政策。**模型不參與這一步**——它只提供 (a)。"""
    same, conf = ans["same_work"], ans["confidence"]
    indep, n = row["_independent"], row["photos"]

    if ans["not_construction"] >= Q.OOD_NOUL:
        return {
            "policy": "維持丟棄",
            "why": f"模型判為行政活動 {ans['not_construction']:.2f}，junk 規則是對的",
        }
    if conf < FLOOR:
        return {"policy": "人工複核", "why": f"信心 {conf:.2f} < {FLOOR}，不猜"}
    if same == "same":
        return {"policy": "不動", "why": f"與 {row['現歸']} 同一種作業（信心 {conf:.2f}）"}
    if same == "mixed":
        return {"policy": "看圖分桶", "why": f"標題混多種作業（信心 {conf:.2f}），先分桶再談開類"}
    # different：語義上確實是另一支 → 樹與張數決定寫不寫
    if not indep:
        return {"policy": "不開類", "why": "語義不同，但公司標準樹上不是獨立一支"}
    if n >= min_size:
        return {"policy": "★ 現在就寫規則", "why": f"語義不同 + 樹上獨立一支 + {n} 張 ≥ {min_size}"}
    return {
        "policy": "寫規則等放行",
        "why": f"語義不同 + 樹上獨立一支，但 {n} 張 < {min_size}（同 v13 連續壁）",
    }


def _cache() -> dict:
    if not CACHE.exists():
        return {}
    return {r["key"]: r["answer"] for r in map(json.loads, filter(str.strip, CACHE.read_text().splitlines()))}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lab = Labeler.load()
    cands = candidates()
    print(f"候選 {len(cands)} 個工種 / {sum(c['photos'] for c in cands)} 張照片\n")

    states = [{"標題範例": c["titles"], "全庫張數": c["photos"], "目前歸屬類別": c["現歸"]} for c in cands]
    if a.dry_run:
        print(json.dumps(states[0], ensure_ascii=False, indent=1))
        print(f"\n[dry-run] 會送出 {len(states)} 筆，未呼叫 API")
        return

    import hashlib

    from typesafe_sdk import ChoiceAnswer, NoulAnswer, TypeSafeClient

    cached, qs, client = _cache(), build_questions(), TypeSafeClient()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    fh = CACHE.open("a")

    def ask(state: dict) -> dict:
        k = hashlib.sha1(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if k in cached:
            return cached[k]
        r = client.system_one(state=state, questions=qs, model=Q.MODEL)
        sw, nc = r.answers["same_work"], r.answers["not_construction"]
        assert isinstance(sw, ChoiceAnswer), f"same_work 應為 Choice，收到 {type(sw).__name__}"
        assert isinstance(nc, NoulAnswer), "not_construction 應為 Noul"
        ans = {
            "same_work": sw.choice,
            "confidence": float(sw.confidence),
            "probabilities": {k2: float(v) for k2, v in sw.probabilities.items()},
            "not_construction": float(nc.noul),
        }
        fh.write(json.dumps({"key": k, "answer": ans}, ensure_ascii=False) + "\n")
        fh.flush()
        return ans

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        answers = list(pool.map(ask, states))
    fh.close()

    rows = []
    for c, ans in zip(cands, answers, strict=True):
        d = decide(c, ans, lab.min_class_size)
        rows.append({**{k: v for k, v in c.items() if not k.startswith("_")}, **ans, **d})

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "newclass_policy.csv", index=False)

    for pol, g in df.groupby("policy", sort=False):
        print(f"\n### {pol}（{len(g)} 個 / {g.photos.sum()} 張）")
        for _, r in g.iterrows():
            print(f"  {r['工種']:16s} {r['photos']:4d} 張  現歸 {r['現歸']:12s}  {r['why']}")
    print(f"\n→ {OUT / 'newclass_policy.csv'}")


if __name__ == "__main__":
    main()
