"""複核佇列 + 標框畫布。依賴 pipeline._cams（熱區快取）。"""

from __future__ import annotations

import streamlit as st

import paths
import split as split_mod

from .common import badge, txt
from .data import labeled
from .pipeline import _cams

CANVAS_W = 460  # 畫布寬度（px）。框存的是 0~1000 比例，換裝置不會跑掉


def review_queue() -> None:
    """值得人看的照片排成佇列。四種訊號，照「證據強度」分層。

    參考答案（誰說這張是什麼）：
      1. `chipsOn`  工地主任自己打的查驗重點——**人寫的**
      2. `specKey`  前端點選的工種——**人選的**，與規則一致率 98.6%
      3. `predWorkItem` Gemini 的作答——**機器答的**，只能當提示，不能當答案

    模型訊號（主動學習）：
      4. 上線那顆探針的預測與**邊際**（top1 機率 − top2 機率）

    前三種問的是「標籤對不對」，第四種問的是「模型會不會」。兩件事不一樣：
    實測 520 張裡只有 58 張有標籤分歧，剩下 462 張就算模型答錯也沒人會看到。
    補上模型訊號才補得到那個洞。裁決寫進 data/review.csv，`Labeler.apply` 蓋掉規則。
    """
    from labels import Labeler, load_reviews, orphan_reviews, save_review

    st.subheader("複核佇列")
    orphan = orphan_reviews()
    if orphan:
        st.error(
            f"{len(orphan)} 筆裁決指到已不存在的類別 {sorted(set(orphan.values()))}"
            " —— 那些照片會被丟掉。類別改過名的話，重裁一次補上新名字。"
        )
    df = labeled()
    if not len(df):
        st.info("還沒有照片。")
        return
    lab = Labeler.load()
    classes = sorted(df.cls.unique().tolist())
    cur = split_mod.current()
    # 分層規則在 review.py，操作台與 `uv run src/review.py` 共用同一份——
    # 規則抄兩份的話，畫面上看到的佇列跟命令列印的會慢慢對不起來。
    import review as review_mod

    scores, test_ids = review_mod.scores("siglip", cur)
    q = review_mod.build(df, lab, scores, test_ids)
    names = review_mod.TIER_NAMES
    tier_n = q.tier.value_counts().to_dict()  # 先存，下面的篩選會改 q
    done = load_reviews()

    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    pick_tiers = c1.multiselect(
        "看哪幾批", [names[t] for t in (4, 3, 2, 1)], default=[names[t] for t in (4, 3, 2, 1)]
    )
    keep = {t for t, n in names.items() if n in pick_tiers}
    q = q[q.tier.isin(keep)]
    if c2.toggle("只看還沒裁的", value=True):
        q = q[~q.fileId.isin(done)]
    per = c3.select_slider("一頁幾張", [6, 12, 24, 48], value=12)
    # 同一層裡最新的排前面：舊照片的標籤問題多半已經在前幾輪裁過了
    q = q.sort_values(["tier", "syncedAt", "reportDate"], ascending=False, kind="stable")
    pages = max(1, -(-len(q) // per))
    page = c4.number_input(f"第幾頁（共 {pages}）", 1, pages, 1)

    st.caption(
        " · ".join(f"**{names[t]}** {tier_n.get(t, 0)}" for t in (4, 3, 2, 1))
        + f" · 已裁 {len(done)} · 符合篩選 {len(q)}"
    )
    if not len(q):
        st.success("這批裁完了。要讓裁決進到模型，回 ① 同步那頁重跑一次。")
        return

    page_rows = list(q.iloc[(page - 1) * per : page * per].itertuples())
    for row0 in range(0, len(page_rows), 2):
        for col, r in zip(st.columns(2), page_rows[row0 : row0 + 2]):
            with col, st.container(border=True):
                _review_card(r, classes, done, save_review, names)


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


def _review_card(r, classes, done, save_review, names) -> None:
    img = paths.IMAGES / f"{r.fileId}.jpg"
    hit = _cams(split_mod.current()).get(r.fileId)
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

    st.markdown(f"{badge('test' if r.isTest else 'train')} **{txt(r.title)}**", unsafe_allow_html=True)
    st.caption(
        f"`{names[r.tier]}`　{r.why}　·　{txt(r.reportDate)} · {txt(r.constrName)}"
        + ("" if r.isTest else "　·　模型背過這張，它的信心偏樂觀")
    )

    src = [("規則", r.cls, True)]
    if isinstance(getattr(r, "clsChips", None), str):
        src.append(("查驗項目", r.clsChips, True))
    if isinstance(getattr(r, "specTrade", None), str):
        src.append(("specKey", r.specTrade, False))
    if isinstance(r.gemNorm, str):
        src.append(("Gemini", r.gemNorm, True))
    if isinstance(r.mPred, str):
        m = f"{r.mPred}（信心 {r.mConf:.2f}・邊際 {r.mMargin:.2f}）"
        src.append(("模型", m, True))
    st.markdown(" ".join(f"<span class=src>{k}</span> `{txt(v)}`" for k, v, _ in src), unsafe_allow_html=True)

    with st.expander("原文"):
        st.write(f"Gemini：{txt(r.predWorkItem)}")
        chips_txt = getattr(r, "chipsOn", None)
        if isinstance(chips_txt, str):
            st.write("人手打的查驗重點：")
            st.write("\n".join(f"- {x}" for x in chips_txt.split("|")))

    if r.fileId in done:
        st.info(f"已裁：{done[r.fileId]}")
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
        # 沒開標框模式就別動既有的框——沒展開畫布不代表要把框清掉
        save_review(r.fileId, pickd, "", boxes=drawn if mark else (_boxes_now().get(r.fileId) or None))
        st.cache_data.clear()
        # 刻意不 st.rerun()：那會把分頁彈回 ①。佇列下次互動才刷新，換來不會跳走
        st.success(f"已紀錄 {pickd}" + (f"，含 {len(drawn)} 個框" if drawn else ""))
