"""孤兒院（異類隔離區）：規則沒接住的照片 × 看鄰居裁決。

`Labeler.apply` 的 `drop_fallback` 把規則沒命中的照片排在訓練外——那對訓練是對的
（「其他 = 一袋雜物」實測 top-1 0.760 → 0.699），但那些照片從此沒有任何佇列看得到
它們：複核佇列的母體是已分類的照片，孤兒就這樣靜靜躺在語料裡。

這頁把孤兒撈回來，給每張兩個裁決依據：
  1. **SigLIP 最近鄰**——隔壁五張已分類照片長什麼樣、人給它們標了什麼。
     「棄土坑施作」的鄰居全是 基礎-舊基礎切削（cosine 0.90），一眼就知道歸哪。
  2. **探針 top-3**——模型只在既有類別裡挑，信心低正是「不屬於任何一類」的訊號。

裁決寫進 data/review.csv，`Labeler.apply` 蓋掉規則，照片就從孤兒院畢業進訓練。
新類的誕生（湊滿 min_class_size 的孤兒群）走另一條路：下面的候選詞表把
`src/newclass.py` 的標題詞挖礦搬進來，該開類的還是要人改 labels.yaml。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

import paths

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in __import__("sys").path:
        __import__("sys").path.insert(0, _p)

from core.review_utils import embedding_index, neighbors  # noqa: E402
from core.review_utils import orphans as build_orphans  # noqa: E402
from labels import Labeler, legacy_manifest, load_reviews, save_review  # noqa: E402

from .common import txt  # noqa: E402


@st.cache_data(show_spinner=False)
def _orphans(_mtime: float, legacy: bool) -> pd.DataFrame:
    """孤兒母體。mtime 當 cache key（manifest / labels.yaml 改了就重算）。"""
    if legacy:
        df = legacy_manifest()
    else:
        df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
        if "active" in df:
            df = df[df.active.astype(str).str.lower().isin(["true", "1"])]
    return build_orphans(df, Labeler.load())


@st.cache_data(show_spinner=False)
def _emb_index(_mtime: float):
    return embedding_index("siglip")


@st.cache_data(show_spinner=False)
def _split_labels(split_name: str) -> dict:
    import json

    return json.loads((paths.SPLITS / f"{split_name}.json").read_text())["labels"]


def _img(file_id: str) -> str | None:
    """孤兒/鄰居的縮圖：先找遮蔽後 jpg，再退回 raw。"""
    for p in (paths.IMAGES / f"{file_id}.jpg", paths.LEGACY_IMAGES / f"{file_id}.jpg"):
        if p.exists():
            return str(p)
    for d in (paths.PHOTOS, paths.LEGACY_PHOTOS):
        for ext in (".webp", ".jpg", ".png"):
            p = d / f"{file_id}{ext}"
            if p.exists():
                return str(p)
    return None


def _candidate_terms(orph: pd.DataFrame, legacy: bool) -> pd.DataFrame:
    """newclass 同款的標題詞挖礦，套在孤兒上。詞是提示，命名與掛樹是人的判斷。"""
    if legacy:
        base = legacy_manifest()
    else:
        base = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
        if "active" in base:
            base = base[base.active.astype(str).str.lower().isin(["true", "1"])]
    lab = Labeler.load()
    allr = lab.apply(base, drop_small=False)
    ok_side = allr[~(allr.cls.isna() | (allr.cls == lab.fallback))]
    ok = Counter(ok_side.title.value_counts().to_dict())
    fb = Counter(orph.title.value_counts().to_dict())

    from newclass import candidates

    got = candidates(fb, ok, min_photos=3, top=15)
    return pd.DataFrame(
        [
            {
                "候選詞": c["term"],
                "孤兒照片": c["photos"],
                "過 12 張門檻": "✅" if c["photos"] >= 12 else "",
                "涵蓋標題": "、".join(t for t, _ in c["titles"][:4]),
            }
            for c in got
        ]
    )


def orphan_queue() -> None:
    import review as review_mod
    import split as split_mod

    st.subheader("孤兒院（異類隔離區）")
    st.caption(
        "規則沒接住的照片在這裡等裁決。看鄰居（SigLIP 最近已分類照）決定歸哪個既有類，"
        "裁決寫進 review.csv 後照片進訓練；同一批孤兒湊滿 12 張且樹上找得到位置，才值得開新類——"
        "開類走 ⑥ 標籤規則頁人手改，不在此頁。"
    )

    legacy = st.segmented_control("來源", ["PMS 日報", "舊 pptx"], default="PMS 日報") == "舊 pptx"
    orph = _orphans(paths.MANIFEST.stat().st_mtime if paths.MANIFEST.exists() else 0.0, legacy)
    if not len(orph):
        st.success("沒有孤兒。")
        return
    done = load_reviews()
    labels = _split_labels(split_mod.current())
    idx = _emb_index(_emb_mtime())
    scores, _ = review_mod.scores(None, split_mod.current())

    with st.expander(f"候選詞挖礦（{len(orph)} 張孤兒的標題 × newclass 同款演算法）", expanded=False):
        got = _candidate_terms(orph, legacy)
        if len(got):
            st.dataframe(got, width="stretch", hide_index=True)
        else:
            st.caption("沒有湊到 3 張的候選詞——孤兒是散彈，不是新類。")
        st.caption("詞是提示不是答案：命名與掛樹的位置是人的判斷（v6 紀律：樹上查得到才收）。")

    c1, c2, c3 = st.columns([3, 2, 2])
    why_pick = c1.multiselect("看哪種", ["fallback", "無規則命中"], default=sorted(orph.orphanWhy.unique()))
    view = orph[orph.orphanWhy.isin(why_pick)]
    if c2.toggle("只看還沒裁的", value=True):
        view = view[~view.fileId.isin(done)]
    per = c3.select_slider("一頁幾張", [4, 8, 16], value=4)
    view = view.sort_values("reportDate", ascending=False, kind="stable")
    pages = max(1, -(-len(view) // per))
    page = st.number_input(f"第幾頁（共 {pages}）", 1, pages, 1)
    st.caption(
        f"孤兒 {len(orph)}（{'、'.join(f'{k} {v}' for k, v in orph.orphanWhy.value_counts().items())}）"
        f" · 已裁 {len(set(done) & set(orph.fileId))} · 符合篩選 {len(view)}"
    )
    if not len(view):
        st.success("這批裁完了。")
        return

    classes = sorted(set(labels.values()))
    page_rows = list(view.iloc[(page - 1) * per : page * per].itertuples())
    for row0 in range(0, len(page_rows), 2):
        for col, r in zip(st.columns(2), page_rows[row0 : row0 + 2]):
            with col, st.container(border=True):
                _orphan_card(r, classes, done, labels, idx, scores)


def _orphan_card(r, classes: list[str], done: dict, labels: dict, idx, scores: dict) -> None:
    img = _img(r.fileId)
    if img:
        st.image(img, width="stretch")
    st.markdown(f"**{txt(r.title)}**")
    st.caption(
        f"`{r.orphanWhy}`　{txt(r.reportDate)} · {txt(getattr(r, 'constrName', ''))}"
        + ("　·　舊 pptx" if getattr(r, "dataset", "") == "legacy" else "")
    )

    # 探針 top-1：它只會在既有類別裡挑——信心低本身就是「不屬於任何一類」的訊號
    hit = scores.get(r.fileId)
    if hit:
        pred, conf, _margin = hit
        st.caption(f"探針硬猜：`{pred}`（信心 {conf:.2f}）——只供參考，孤兒本來就不在它的類別裡")

    # 鄰居縮圖：裁決的真正依據
    nb = neighbors(r.fileId, labels, _index=idx)
    if nb:
        cols = st.columns(5)
        for c, (fid, cls, sim) in zip(cols, nb):
            t = _img(fid)
            if t:
                c.image(t, width="content")
            c.caption(f"{cls}　{sim:.2f}")

    if r.fileId in done:
        st.info(f"已裁：{done[r.fileId]}")
    a, c2 = st.columns([4, 1])
    pick = a.selectbox("實際在拍什麼", classes, key=f"or_{r.fileId}")
    c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c2.button(
        "畢業",
        key=f"ob_{r.fileId}",
        type="primary",
        width="stretch",
        help="裁決寫進 review.csv，照片離開孤兒院、進訓練",
    ):
        save_review(r.fileId, pick)
        st.cache_data.clear()
        # 刻意不 st.rerun()：跟複核佇列同一個理由，rerun 會把分頁彈回 ①
        st.success(f"已紀錄 {pick}")


def _emb_mtime() -> float:
    """全源特徵檔的最新 mtime（任一個重抽，鄰居索引就重算）。"""
    mt = 0.0
    for prefix in ("", "qms-", "legacy-", "crops-"):
        p = paths.FEATURES / f"{prefix}siglip.npz"
        if p.exists():
            mt = max(mt, p.stat().st_mtime)
    return mt
