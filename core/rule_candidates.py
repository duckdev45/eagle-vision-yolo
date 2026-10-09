"""標題 → 新工種候選詞（純函式）。

規則沒命中的標題裡，哪個中文片段「蓋得夠多又夠純」，就是下一條 labels.yaml 規則的候選。
CLI `src/newclass.py`（`make newclass`）與工作台的新類提示（core/pms_review.discover）共用這一份。
這裡只負責算詞，不決定那是不是一個工種——命名與掛樹是人的判斷。
"""

from __future__ import annotations

import re
from collections import Counter

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
