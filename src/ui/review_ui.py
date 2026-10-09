"""進階複核：收件匣同一份佇列，多了熱區與標框畫布。依賴 pipeline._cams（熱區快取）。

「哪些照片要人看」只有一份規則（core/routing.py）。這頁以前用自己的四層分類
（core/review_utils.py，2026-10-09 刪除），跟收件匣給不同答案；現在只差在「能畫證據框」。
寫入走 routing.resolve()（＝工作台同一條 decide），框跟著裁決一起存。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core import model_registry as registry
from core import paths, routing
from core import pms_review as review

from .common import reviewer_input, txt
from .data import labeled
from .pipeline import _cams
from .pms_ai import ai_panel, latest_suggestions, suggestion_card

CANVAS_W = 460  # 畫布寬度（px）。框存的是 0~1000 比例，換裝置不會跑掉
_REF_COLUMNS = ["fileId", "clsChips", "specTrade", "predWorkItem"]
_SIGNAL = {"oof": "交叉驗證", "live": "上線模型"}


def _queue() -> pd.DataFrame:
    """收件匣佇列＋標題、案場與人寫的參考答案（查驗重點、specKey、Gemini 作答）。"""
    from labels import Labeler

    items = routing.queue_items()
    if items.empty:
        return items
    pool = review.load_pool()[["fileId", "title", "constrName", "chipsOn"]]
    refs = labeled()
    refs = refs[[c for c in _REF_COLUMNS if c in refs]] if len(refs) else pd.DataFrame(columns=["fileId"])
    q = items.merge(pool, on="fileId", how="left").merge(refs, on="fileId", how="left")
    lab = Labeler.load()
    q["gemNorm"] = [
        (lab.label(t) or None) if isinstance(t, str) and t else None for t in q.get("predWorkItem", [])
    ]
    return q


def review_queue() -> None:
    from labels import orphan_reviews

    st.subheader("進階複核（含證據框）")
    st.caption(
        "佇列與收件匣相同；這裡多了模型熱區與標框畫布。框會餵進訓練（人標框裁切），不是只給人看的註記。"
    )
    orphan = orphan_reviews()
    if orphan:
        st.error(
            f"{len(orphan)} 筆裁決指到已不存在的類別 {sorted(set(orphan.values()))}"
            " —— 那些照片會被丟掉。類別改過名的話，重裁一次補上新名字。"
        )
    q = _queue()
    if q.empty:
        st.success("佇列是空的（或還沒分流：收件匣那頁可以立即分流）。")
        return
    classes = sorted(review.catalog())

    c1, c3, c4 = st.columns([4, 2, 2])
    present = [w for w in routing.REASONS if q.reason.str.contains(w, regex=False).any()]
    pick = c1.multiselect("看哪些原因", present, default=present)
    q = q[q.reason.apply(lambda r: any(w in r for w in pick))]
    per = c3.select_slider("一頁幾張", [6, 12, 24, 48], value=12)
    pages = max(1, -(-len(q) // per))
    page = c4.number_input(f"第幾頁（共 {pages}）", 1, pages, 1)
    st.caption(f"符合篩選 {len(q)} 張 · 排序與收件匣相同（規則與模型吵架的先看）")
    if not len(q):
        st.info("這個條件下沒有照片。")
        return

    if ai_panel(q, key="pms_queue_ai") is not None:
        st.cache_data.clear()
        st.rerun()
    reviewer = reviewer_input()
    suggestions = latest_suggestions()
    page_rows = list(q.iloc[(page - 1) * per : page * per].itertuples())
    for row0 in range(0, len(page_rows), 2):
        for col, r in zip(st.columns(2), page_rows[row0 : row0 + 2]):
            with col, st.container(border=True):
                _review_card(r, classes, reviewer)
                if proposal := suggestions.get(r.fileId):
                    revision_key = f"pms_revision:queue:{r.fileId}"
                    expected = st.session_state.setdefault(revision_key, review.revision(r.fileId))
                    if suggestion_card(
                        proposal, reviewer, key=f"pms_queue_accept_{r.fileId}", expected_revision=expected
                    ):
                        st.session_state.pop(revision_key, None)
                        st.cache_data.clear()
                        st.rerun()


def _patch_canvas() -> None:
    """補上 streamlit-drawable-canvas 0.9.3 要的 `streamlit.elements.image.image_to_url`。

    Streamlit 1.4x 把它搬到 `lib.image_utils`，而且第二個參數從 `width: int`
    改成 `LayoutConfig`。畫布套件還沒跟上。接一層轉接比把 Streamlit 錨在舊版划算——
    錨舊版等於整個操作台跟著退回去。哪天套件修好了，這整段可以刪。
    """
    import streamlit.elements.image as mod  # 模組還在，只是 image_to_url 被搬走了

    if hasattr(mod, "image_to_url"):
        return
    from streamlit.elements.lib.image_utils import image_to_url as _new
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def image_to_url(image, width, clamp, channels, output_format, image_id):
        return _new(image, LayoutConfig(width=int(width)), clamp, channels, output_format, image_id)

    mod.image_to_url = image_to_url


def _draw_boxes(r, img, cam_box):
    """用滑鼠拖框標「證據在這」。回傳 [[x0,y0,x1,y1], ...]，0~1000 相對原圖。

    為什麼要人來標而不是信模型的熱區：熱區是**模型現在**看的地方，人標的是
    **應該**看的地方。兩者的差距就是要學的東西——「人行道地磚貼飾」那張，
    模型壓在行道樹上、人會框在地磚上，這個差就是訓練訊號。
    """
    from PIL import Image

    _patch_canvas()
    from streamlit_drawable_canvas import st_canvas

    im = Image.open(img).convert("RGB")
    w, h = im.size
    scale = CANVAS_W / w
    ch = round(h * scale)
    old = _boxes_now().get(r.fileId) or []
    # 已標過的框畫回去，人才知道自己標過什麼（canvas 吃像素座標，要從 0~1000 換回來）
    initial = (
        {
            "version": "4.4.0",
            "objects": [
                {
                    "type": "rect",
                    "left": b[0] / 1000 * CANVAS_W,
                    "top": b[1] / 1000 * ch,
                    "width": (b[2] - b[0]) / 1000 * CANVAS_W,
                    "height": (b[3] - b[1]) / 1000 * ch,
                    "fill": "rgba(255,80,80,0.18)",
                    "stroke": "#ff5050",
                    "strokeWidth": 2,
                }
                for b in old
            ],
        }
        if old
        else None
    )
    res = st_canvas(
        fill_color="rgba(255,80,80,0.18)",
        stroke_color="#ff5050",
        stroke_width=2,
        background_image=im,
        drawing_mode="rect",
        update_streamlit=True,
        width=CANVAS_W,
        height=ch,
        initial_drawing=initial,
        key=f"cv_{r.fileId}",
    )
    out = []
    for o in ((res.json_data or {}).get("objects") or []) if res else []:
        if o.get("type") != "rect":
            continue
        x0 = o["left"] / CANVAS_W * 1000
        y0 = o["top"] / ch * 1000
        x1 = x0 + o["width"] * o.get("scaleX", 1) / CANVAS_W * 1000
        y1 = y0 + o["height"] * o.get("scaleY", 1) / ch * 1000
        if x1 - x0 > 8 and y1 - y0 > 8:  # 誤點出來的小框丟掉
            out.append([int(max(0, min(1000, v))) for v in (x0, y0, x1, y1)])
    if cam_box:
        st.caption(f"模型自己看的：{cam_box}　·　紅框是你標的（拖曳新增，側欄工具可刪）")
    return out


@st.cache_data(show_spinner=False)
def _boxes_cached(_mtime: float) -> dict:
    from labels import load_boxes

    return load_boxes()


def _boxes_now() -> dict:
    return _boxes_cached(paths.REVIEW.stat().st_mtime if paths.REVIEW.exists() else 0.0)


def _review_card(r, classes, reviewer: str) -> None:
    img = paths.IMAGES / f"{r.fileId}.jpg"
    hit = _cams(registry.current()).get(r.fileId)
    n_old = len(_boxes_now().get(r.fileId) or [])
    # 用 toggle 不用 button：button 要配 st.rerun() 才切得動狀態，而 st.rerun()
    # 會把 st.tabs 彈回第一頁——標一張框就跳走一次，沒人受得了。
    mark = st.toggle(
        f"✎ 標框{f'（已有 {n_old}）' if n_old else ''}",
        key=f"mk_{r.fileId}",
        help="拖曳框出「應該看這裡」。框會餵進訓練，不是只給人看的註記",
    )
    cam_box = None
    if hit is not None:
        from PIL import Image

        import explain

        cam, pred, _conf, box = hit
        cam_box = explain.to_1000(box)
        if not mark:
            st.image(explain.overlay(Image.open(img).convert("RGB"), cam, box, pred), width="stretch")
    elif img.exists() and not mark:
        st.image(str(img), width="stretch")
    drawn = _draw_boxes(r, img, cam_box) if (mark and img.exists()) else None

    st.markdown(f"**{txt(r.title)}**")
    # 抽查照不告訴看的人「這是抽查」（同收件匣）：知道機器確認過，人就容易照單全收
    why = "例行確認" if r.bucket == "audit" else r.reason
    signal = _SIGNAL.get(r.signal, "")  # 兩種都是「沒背過這張」的模型給的，信心可信
    st.caption(
        f"{why}　·　{txt(r.reportDate)} · {txt(r.constrName)}"
        + (f"　·　模型訊號：{signal}" if signal else "")
    )

    src = [("規則", r.ruleClass or None, True)]
    if isinstance(getattr(r, "clsChips", None), str):
        src.append(("查驗項目", r.clsChips, True))
    if isinstance(getattr(r, "specTrade", None), str):
        src.append(("specKey", r.specTrade, False))
    if isinstance(r.gemNorm, str):
        src.append(("Gemini", r.gemNorm, True))
    if isinstance(r.modelClass, str) and r.modelClass:
        conf = r.modelConfidence
        src.append(("模型", r.modelClass + ("" if conf != conf else f"（信心 {conf:.2f}）"), True))
    st.markdown(" ".join(f"<span class=src>{k}</span> `{txt(v)}`" for k, v, _ in src), unsafe_allow_html=True)

    with st.expander("原文"):
        st.write(f"Gemini：{txt(getattr(r, 'predWorkItem', None))}")
        chips_txt = getattr(r, "chipsOn", None)
        if isinstance(chips_txt, str):
            st.write("人手打的查驗重點：")
            st.write("\n".join(f"- {x}" for x in chips_txt.split("|")))

    # 建議選項放前面（specKey 只到工種，不能直接當答案，所以不進建議）
    head = [
        v.split("（")[0]
        for _, v, full in src
        if full and isinstance(v, str) and v.split("（")[0] in set(classes)
    ]
    seen, opts = set(), []
    for c in head + classes:
        if c not in seen:
            seen.add(c)
            opts.append(c)
    a, c2 = st.columns([4, 1])
    pickd = a.selectbox(
        "實際在拍什麼", opts, key=f"rv_{r.fileId}", help="前面幾個是各來源的建議，後面是全部類別"
    )
    c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c2.button("儲存紀錄", key=f"bt_{r.fileId}", type="primary", width="stretch"):
        # 沒開標框模式就別動既有的框——沒展開畫布不代表要把框清掉（boxes=None＝保留既有）
        try:
            routing.resolve(
                r.fileId, reviewer=reviewer, label=pickd, reason="進階複核", boxes=drawn if mark else None
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.cache_data.clear()
        # 刻意不 st.rerun()：那會把分頁彈回 ①。佇列下次互動才刷新，換來不會跳走
        st.success(f"已紀錄 {pickd}" + (f"，含 {len(drawn)} 個框" if drawn else ""))
