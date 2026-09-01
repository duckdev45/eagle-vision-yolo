"""沒被規則命中的標題 → 下一批新工種的候選。

`drop_fallback: true` 讓規則沒命中的照片不進訓練，那是對的：把互不相干的東西
倒進一個「其他」類，等於教模型「其他 = 一袋雜物」（實測 top-1 0.760 → 0.699）。

**但「不倒進其他」不等於「該丟掉」。** 照片還在 `data/raw/photos/`，
少的只是一條規則。這支就是把「少哪條規則」算出來：

    uv run src/newclass.py                 # PMS + legacy 一起看
    uv run src/newclass.py --src pms       # 只看 PMS（線上真的會收到的照片）
    uv run src/newclass.py --min 12        # 只列蓋得到 12 張以上的（= min_class_size）

印出來的 yaml 區塊可以直接貼進 labels.yaml 的 rules。**貼之前要看過**——
這裡只負責「哪個詞蓋得到最多沒人認領的照片」，不負責決定那是不是一個工種。
命名規約（`工種-施作內容`）與掛在工程分類樹哪一支，是人的判斷。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paths
from labels import Labeler

CJK = re.compile(r"[一-鿿]+")


def terms(title: str, lo: int = 2, hi: int = 4) -> set[str]:
    """標題裡所有 2~4 字的中文連續片段。

    只取中文：樓層與編號（`1F`、`A7`、`7-3F`）在每個工種的標題裡都會出現，
    當規則會把整批不相干的照片一起攔走。
    """
    out = set()
    for run in CJK.findall(title):
        for n in range(lo, hi + 1):
            for i in range(len(run) - n + 1):
                out.add(run[i : i + n])
    return out


def candidates(
    fb: Counter, ok: Counter, min_photos: int = 12, purity: float = 0.9, top: int = 25
) -> list[dict]:
    """挑出值得變成新工種的詞。貪婪集合覆蓋，一張照片只算給第一個蓋到它的詞。

    好詞要同時滿足兩件事：

    1. **蓋得夠多** —— 少於 min_class_size 的話，加了規則也會被門檻擋下來，
       白忙一場（現在就有 9 個類別卡在這個狀態）。
    2. **夠純** —— 「施作」「安裝」「工程」這種詞在已分類的標題裡也一大堆，
       拿去當規則會把既有類別一起攔走。規則是**由上而下第一個命中者勝**，
       一條髒規則插進去，底下所有類別的張數都會變，而且不會有人發現。

    fb / ok 都是 {標題: 張數}，不是 {標題: 出現次數}——同一個標題底下有幾張照片
    才是我們在乎的，那決定它過不過 min_class_size。
    """
    fb_terms: dict[str, int] = Counter()
    ok_terms: dict[str, int] = Counter()
    for title, n in fb.items():
        for t in terms(str(title)):
            fb_terms[t] += n
    for title, n in ok.items():
        for t in terms(str(title)):
            ok_terms[t] += n

    pure = {t: c for t, c in fb_terms.items() if c >= min_photos and c / (c + ok_terms.get(t, 0)) >= purity}

    # 「這個字有多通用」：出現在**已分類**標題裡的張數。施、作、安、裝、工、程
    # 這些字每個工種都在用，含它們的片段（`子施`、`裝施`）是跨詞界的碎片，不是工種。
    # 用資料算而不是硬編停用詞表——換個工地的用字習慣，表就過期了。
    generic: Counter = Counter()
    for title, n in ok.items():
        for ch in set(str(title)):
            generic[ch] += n

    def noise(t: str) -> int:
        return sum(generic[ch] for ch in t)

    # 貪婪覆蓋：每輪挑「還沒被認領的照片」蓋最多的那個詞
    left = dict(fb)
    picked: list[dict] = []
    while len(picked) < top:
        best, best_key, best_titles = None, None, []
        for t in pure:
            hit = [(ti, n) for ti, n in left.items() if t in str(ti)]
            n = sum(x[1] for x in hit)
            if n < min_photos:
                continue
            # 蓋得多優先；一樣多挑通用字最少的（擋掉 `子施` 這種跨詞界碎片）；
            # 再一樣挑**長**的（`抿石子` 比 `抿石` 精確，當規則誤傷的機會小）。
            key = (n, -noise(t), len(t))
            if best_key is None or key > best_key:
                best, best_key, best_titles = t, key, hit
        if not best:
            break
        best_n = best_key[0]
        picked.append({"term": best, "photos": best_n, "titles": sorted(best_titles, key=lambda x: -x[1])})
        for ti, _ in best_titles:
            left.pop(ti, None)
        pure.pop(best, None)
    return picked


def pools(src: str = "all") -> tuple[Counter, Counter, Labeler]:
    """({沒命中的標題: 張數}, {命中的標題: 張數}, Labeler)。"""
    from sync import _truthy

    lab = Labeler.load()
    frames = []
    if src in ("all", "pms"):
        m = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
        frames.append(m[_truthy(m.active)][["title"]].assign(src="pms"))
    if src in ("all", "legacy") and paths.LEGACY_MANIFEST.exists():
        lg = pd.read_csv(paths.LEGACY_MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
        frames.append(lg[["title"]].assign(src="legacy"))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["title"])
    cls = df.title.map(lab.label)
    fb = Counter(df.title[cls == lab.fallback].dropna())
    ok = Counter(df.title[cls.notna() & (cls != lab.fallback)].dropna())
    return fb, ok, lab


def report(src: str = "all", min_photos: int = 0, top: int = 25, log=print) -> list[dict]:
    fb, ok, lab = pools(src)
    min_photos = min_photos or lab.min_class_size
    total = sum(fb.values())
    log(f"來源 {src} · 沒被規則命中 {total} 張 / {len(fb)} 種標題 · 門檻 {min_photos} 張（= min_class_size）")
    if not total:
        log("規則全部命中，沒有待認領的照片。")
        return []
    cand = candidates(fb, ok, min_photos=min_photos, top=top)
    covered = sum(c["photos"] for c in cand)
    log(
        f"挑出 {len(cand)} 個候選詞，蓋掉 {covered}/{total} 張"
        f"（剩 {total - covered} 張是零散的，還不值得開類別）\n"
    )
    for i, c in enumerate(cand, 1):
        log(f"{i:>2}. 「{c['term']}」 {c['photos']} 張")
        for ti, n in c["titles"][:4]:
            log(f"      {n:>3}  {ti}")
        if len(c["titles"]) > 4:
            log(f"      … 另 {len(c['titles']) - 4} 種標題")
    if cand:
        log("\n貼進 labels.yaml 的 rules（**標籤名要自己改**，這裡只填得出詞）：")
        for c in cand:
            # 詞常常是跨詞界的碎片（`櫃安` = 櫥櫃+安裝），因為它蓋得比完整詞多
            # ——`櫥櫃` 漏掉「廚櫃」那種錯字。附上最常見的標題，人才看得出這是什麼工種。
            log(
                f'  - {{ pattern: "{c["term"]}", label: 待命名-{c["term"]} }}'
                f"   # {c['photos']} 張，例：{c['titles'][0][0]}"
            )
        log(
            "\n順序即優先權，第一個命中者勝——新規則插在哪一行會改變既有類別的張數，"
            "\n插完務必 `make model SPLIT=vN` 看分數，別直接 `make use`。"
        )
    return cand


def demo() -> None:
    assert terms("1F大廳公設裝修") >= {"大廳", "公設", "裝修", "大廳公設"}
    assert not any(c.isascii() for t in terms("B3FEpoxy地坪打磨") for c in t)  # 只取中文

    fb = Counter({"A區抿石子施作": 10, "B區抿石子施作": 8, "頂樓吊模": 5})
    # 已分類的標題也有「施作」「區」「樓」——通用字的分數是從這裡算出來的，
    # 所以這個 fixture 必須長得像真的標題，只放一筆會讓每個字看起來都很獨特
    ok = Counter({"3F油漆施作": 60, "B1區壁磚施作": 30, "頂樓防水施作": 10})
    got = candidates(fb, ok, min_photos=12, top=5)
    # 蓋一樣多（18 張）的有 石子/抿石/抿石子/區抿石子/抿石子施/子施…
    # 含 施、區 的被通用字扣掉，剩 石子/抿石/抿石子 同分 → 取最長
    assert [c["term"] for c in got] == ["抿石子"], got
    assert got[0]["photos"] == 18
    assert not candidates(fb, ok, min_photos=30, top=5)  # 蓋不到門檻就別提

    # 「施作」蓋 23 張比「抿石子」多，但它在已分類標題裡有 100 張 → 必須被純度擋掉
    assert "施作" not in [c["term"] for c in candidates(fb, ok, min_photos=5, top=5)]
    print("ok newclass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="all", choices=["all", "pms", "legacy"])
    ap.add_argument("--min", type=int, default=0, help="蓋不到這麼多張就不提，預設 min_class_size")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        demo()
    else:
        report(a.src, a.min, a.top)
