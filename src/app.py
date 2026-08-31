"""操作台。最上層先選系統：PMS 日報 / QMS 稽核。

兩邊的資料、標籤、模型全部各自獨立（實測混訓有害，見 README），所以在最上層就
分開，而不是塞成同一排分頁裡的一格——那會讓人以為它們是同一件事的兩個面向。

    uv run streamlit run src/app.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import date

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import boxes  # noqa: E402
import organize  # noqa: E402
import paths  # noqa: E402
import prepare  # noqa: E402
import split as split_mod  # noqa: E402
import sync as sync_mod  # noqa: E402
from labels import Labeler  # noqa: E402

st.set_page_config(page_title="eagle-vision", layout="wide", page_icon="🦅")

# 照片卡樣式。放最上層而非某個分頁裡——分頁的程式碼會因為「那批資料還沒有」
# 而整段跳過，樣式就跟著消失，其他分頁的卡片會裸奔。
st.markdown("""<style>
.pc{height:114px;line-height:1.5;font-size:12px;overflow:hidden}
.pc6{height:132px;line-height:1.5;font-size:12px;overflow:hidden}
.pc div,.pc6 div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pc .m,.pc6 .m{color:#8b949e}
.pc .ok,.pc6 .ok{color:#3fb950}
.pc .no,.pc6 .no{color:#e5534b}
.bdg{display:inline-block;padding:0 6px;border-radius:3px;font-size:11px;
     font-weight:700;color:#fff;letter-spacing:.5px}
.bdg.tr{background:#2f9e44}.bdg.te{background:#c92a2a}
/* 複核卡片：來源標籤。名稱與值成對，一眼看得出誰說了什麼 */
.src{display:inline-block;background:#2a2a28;color:#9aa0a6;border-radius:3px;
     padding:0 5px;margin-right:2px;font-size:11px}
</style>""", unsafe_allow_html=True)


# ---------- 資料 -------------------------------------------------------
@st.cache_data(show_spinner=False)
def _manifest(_mtime: float) -> pd.DataFrame:
    if not paths.MANIFEST.exists():
        return pd.DataFrame()
    df = pd.read_csv(paths.MANIFEST)
    return df[df.active.astype(str).str.lower().isin(["true", "1"])] if "active" in df else df


def load_manifest() -> pd.DataFrame:
    return _manifest(paths.MANIFEST.stat().st_mtime if paths.MANIFEST.exists() else 0.0)


def labeled() -> pd.DataFrame:
    """加上 cls，並帶上人寫的兩個參考答案（clsChips / specTrade）給複核佇列用。"""
    from labels import human_refs

    df = load_manifest()
    if not len(df):
        return df
    lab = Labeler.load()
    out = lab.apply(df)
    return out.join(human_refs(out, lab))


@st.cache_data(show_spinner=False)
def local_preds(model_key: str = "siglip",
                split_name: str = "") -> tuple[dict, set]:
    """{fileId: 本地模型預測}, {測試集 fileId}。缺模型或特徵就回空的。

    訓練集照片的預測是它自己背過的，偏樂觀，所以要標出來。
    """
    import pickle

    import numpy as np
    try:
        z = np.load(paths.FEATURES / f"{model_key}.npz", allow_pickle=True)
        with (paths.MODELS / f"probe-{model_key}-{split_name}.pkl").open("rb") as f:
            clf = pickle.load(f)["clf"]
        test = set(json.loads((paths.SPLITS / f"{split_name}.json").read_text())["test"])
    except (FileNotFoundError, KeyError):
        return {}, set()
    return dict(zip(z["fileIds"].tolist(), clf.predict(z["emb"]))), test



@st.cache_data(show_spinner=False)
def _qms(_mtime: float) -> pd.DataFrame:
    """QMS manifest + 中類標籤 + ONNX 預測（有的話）。"""
    import qms
    try:
        df = qms.labeled()
    except FileNotFoundError:
        return pd.DataFrame()
    pred = paths.MODELS / "backbone-qms-preds-qms.csv"
    if pred.exists():
        df = df.merge(pd.read_csv(pred), on="fileId", how="left")
    sp = paths.SPLITS / "qms-v1.json"
    if sp.exists():
        j = json.loads(sp.read_text())
        part = {f: "test" for f in j["test"]}
        part.update({f: "val" for f in j.get("val") or []})
        df["part"] = df.fileId.map(part).fillna("train")
    return df


def load_qms() -> pd.DataFrame:
    return _qms(paths.QMS_MANIFEST.stat().st_mtime if paths.QMS_MANIFEST.exists() else 0.0)


def nav(names: list[str], key: str) -> list[bool]:
    """分頁選單。刻意不用 `st.tabs`：它的選取是純前端狀態，元件樹一變
    （例如標框畫布出現）就重置回第一頁——標一次框就被彈走一次。
    radio 的值存在 session_state，任何 rerun 都不會掉。
    """
    pick = st.radio(" ", names, horizontal=True, key=key, label_visibility="collapsed")
    return [pick == n for n in names]


def run_step(fn, **kw):
    buf = io.StringIO()
    with st.spinner("執行中…"), redirect_stdout(buf):
        try:
            fn(log=lambda *a: print(*a), **kw)
        except Exception as e:  # 操作台不該因為一個步驟炸掉就整頁死掉
            print(f"錯誤：{e}")
    st.code(buf.getvalue() or "(無輸出)")
    st.cache_data.clear()


def txt(v) -> str:
    """欄位可能是 NaN（pandas 讀 CSV 的空值）——NaN 是 truthy，`v or '－'` 擋不住。"""
    return "－" if v is None or v != v else str(v)


def verdict(label: str, p, truth) -> str:
    if p is None or p != p:
        return f'<div class=m>{label}：－</div>'
    return f'<div class={"ok" if p == truth else "no"}>{label}：{p}</div>'


def gem_line(raw, norm, truth, conf_s: str = "") -> str:
    """Gemini 那一行：顯示它自己寫的字，但用正規化後的結果決定綠/紅。

    正規化（`景觀草皮鋪設` → `植栽-景觀`）是我們為了跟 cls 比對硬折過去的，
    看板上要看的是它到底講了什麼。折出來的結果不在訓練類別裡時補一個「類別外」
    ——那不是它答錯，是我們沒有那個類別可以接。
    """
    if raw is None or raw != raw or not str(raw).strip():
        return '<div class=m>Gemini判斷：－</div>'
    cls_ = "ok" if norm == truth else "no"
    tail = " ·類別外" if norm == "類別外" else ""
    return (f'<div class={cls_} title="{txt(raw)}">Gemini判斷：{txt(raw)}{conf_s}'
            f'<span class=m>{tail}</span></div>')


@st.cache_data(show_spinner=False)
def with_boxes(path: str, boxes_json: str) -> "object":
    return boxes.draw(path, boxes_json)


box_area = boxes.area


def badge(part: str) -> str:
    """test 紅、其餘綠：紅色代表「模型沒背過，這些才算數」。"""
    return (f'<span class="bdg {"te" if part == "test" else "tr"}">{part.upper()}</span>')


@st.cache_resource(show_spinner=False)
def _encoder(model_key: str = "siglip"):
    import explain
    return explain.encoder(model_key)


@st.cache_resource(show_spinner=False)
def _probe(model_key: str = "siglip", split_name: str = ""):
    import pickle
    with (paths.MODELS / f"probe-{model_key}-{split_name}.pkl").open("rb") as f:
        return pickle.load(f)["clf"]


@st.cache_resource(show_spinner=False)
def _cams(split_name: str = ""):
    """explain.py --probe 批次算好的熱區。沒跑過就是空的，改走現算。"""
    import explain
    return explain.load_cams(split_name)


@st.cache_data(show_spinner=False)
def _evidence(file_id: str, grid: int):
    """先吃批次快取；沒有才現算（grid²+1 次 encoder forward）。"""
    import explain
    from PIL import Image

    im = Image.open(paths.IMAGES / f"{file_id}.jpg").convert("RGB")
    hit = _cams(split_mod.current()).get(file_id) if grid == explain.GRID else None
    cam, pred, conf, box = hit or explain.probe_cam(
        im, _probe("siglip", split_mod.current()), _encoder(), grid)
    return explain.overlay(im, cam, box, pred), pred, conf, explain.to_1000(box)


def evidence_view(d2) -> None:
    """「它是看哪裡決定的」。解釋的是上面那個系統判斷，不是另一個模型的猜測。"""
    st.divider()
    st.subheader("判斷依據")
    cur = split_mod.current()
    if not (paths.MODELS / f"probe-siglip-{cur}.pkl").exists():
        st.info(f"沒有 probe-siglip-{cur}.pkl，先跑 train.py。")
        return
    opts = d2.fileId.tolist()
    if not opts:
        return
    look = {r.fileId: f"{r.cls} · {txt(r.title)} · {r.reportDate}" for r in d2.itertuples()}
    pick = st.selectbox("挑一張看它在看哪裡", opts, format_func=lambda f: look.get(f, f))
    grid = st.select_slider("解析度（格數愈多愈細，也愈慢）", [4, 6, 8], value=6)
    if not st.button("🔍 算依據", type="primary"):
        st.caption("遮住一格 → 重新編碼 → 看這個答案掉多少。掉最多的那格就是依據。"
                   f"要跑 {grid * grid + 1} 次編碼，約數秒。")
        return
    with st.spinner("遮擋中…"):
        img, pred, conf, box = _evidence(pick, grid)
    st.image(img, caption=f"左＝最關鍵的一塊，右＝熱區　·　判斷 {pred}（{conf:.2f}）",
             width="stretch")
    st.caption(f"框（與 Gemini evidence 同格式 0~1000）：{box}　·　"
               "紅色愈深＝遮掉那塊、這個答案掉愈多。熱區落在天空、工人、鷹架、"
               "行道樹上，分數再高也不可信 —— 那是捷徑，不是依據。")


def _next_split_name() -> str:
    """v8 → v9。看不懂的名字就退回加 -b，不要猜。"""
    import re
    have = {p.stem for p in paths.SPLITS.glob("v*.json")}
    cur = split_mod.current()
    m = re.fullmatch(r"v(\d+)", cur)
    if not m:
        return f"{cur}-b" if f"{cur}-b" not in have else f"{cur}-c"
    n = int(m.group(1))
    while f"v{n}" in have:
        n += 1
    return f"v{n}"


def run_pipeline(name: str, with_data: bool) -> None:
    """README「新照片進來之後」那條鏈，跑在同一個 process 裡。

    ponytail: 同步阻塞，不做背景工作佇列。這是本機單人操作台，全鏈約 5~10 分鐘，
    Streamlit 撐得住；真的要邊跑邊用再說（那時候該用 make，不是加 job queue）。
    """
    import evaluate as ev
    import explain
    import features
    import journal
    import train as tr

    # 這串必須與 Makefile 的 `data` + `model` 逐步對齊。兩邊漂開的話，
    # 按鈕跑出來的 split 跟 `make model` 跑出來的不是同一份，分數就沒得比。
    steps = []
    if with_data:
        steps += [("同步日報 + 照片", lambda log: sync_mod.sync(log=log)),
                  ("前處理", lambda log: prepare.run(kind="report", log=log)),
                  ("抽 embedding", lambda log: features.extract(log=log))]
    steps += [
        # 人標框裁出來的塊也要有 embedding，否則新裁的框會靜靜被丟掉
        # （實測 split 說 train 649，載進去只有 633）
        ("抽裁切框 embedding", lambda log: features.extract_crops(log=log)),
        (f"切分 {name}", lambda log: split_mod.build(name=name, log=log)),
        ("訓練探針", lambda log: tr.probe(split_name=name, log=log)),
        ("評估 + Gemini 基準線", lambda log: ev.run(split_name=name, run_tag=name, log=log)),
        ("重算熱區快取", lambda log: explain.run_probe(split_name=name, log=log)),
        ("重寫學習紀錄", lambda log: log("學習紀錄 →", journal.write())),
    ]
    for title, fn in steps:
        with st.status(title, expanded=False) as box:
            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    fn(lambda *a: print(*a))
                except Exception as e:
                    print(f"錯誤：{e}")
                    box.update(label=f"{title} — 失敗", state="error")
                    st.code(buf.getvalue())
                    st.error("這一步失敗，後面沒跑。修好再按一次。")
                    return
            box.update(label=f"{title} ✔", state="complete")
            st.code(buf.getvalue() or "(無輸出)")
    st.cache_data.clear()
    st.session_state["pipeline_done"] = name


def pipeline_panel() -> None:
    st.markdown("**2. 重跑模型** — 兩顆的差別只在「要不要先抓新照片」")
    cur = split_mod.current()
    st.caption(f"操作台現在看的是 **{cur}**。跑完**不會自動切換**，"
               "分數看過覺得可以，再按最下面那顆。")
    c1, c2, c3 = st.columns([2, 2, 2])
    name = c1.text_input("新的 split 名字", _next_split_name(),
                         help="每次給新名字。原地覆蓋的話，同名舊報告的分母就對不上了。")
    full = c2.button("▶ 全鏈重跑", type="primary",
                     help="有新照片時用。同步 → 前處理 → embedding → 裁切框 → "
                          "切分 → 訓練 → 評估 → 熱區 → 學習紀錄（含上面第 1 顆的工作）")
    only = c3.button("▶ 只重跑模型",
                     help="只改過 labels.yaml 或複核裁決時用。"
                          "跳過前三步（與標籤無關），其餘一樣")
    c2.caption("有新照片")
    c3.caption("只改了標籤或裁決")
    if full or only:
        if name in {p.stem for p in paths.SPLITS.glob("*.json")}:
            st.error(f"`{name}` 已經存在。換一個名字，別覆蓋舊的。")
        else:
            run_pipeline(name, with_data=full)

    done = st.session_state.get("pipeline_done")
    if done and done != cur:
        rep = paths.REPORTS_OUT / f"{date.today():%Y-%m-%d}-{done}"
        m = json.loads((rep / "metrics.json").read_text()) \
            if (rep / "metrics.json").exists() else {}
        st.success(f"`{done}` 跑完了"
                   + (f" — top-1 {m['top1']} / macro-F1 {m['macroF1']} / {m['support']} 張"
                      if m else ""))
        st.caption("切換是獨立一步：換掉所有人看到的答案是個決定，"
                   "不該是跑完訓練的副作用。")
        st.caption(f"⚠ 這個 top-1 **不能直接跟 {cur} 的比**。split 按「每個工地最新幾天」"
                   "滾動切，每一版的考卷是不同照片（實測 v18∩v20 只重疊 83/141），"
                   "分數升降有一部分是換考卷換的。要比模型強弱，看 "
                   "`reports/JOURNAL.md` 的 per-class 那節，或讓兩顆考同一份卷。")
        if st.button(f"✔ 把操作台切到 {done}"):
            split_mod.set_current(done)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("pipeline_done", None)
            st.rerun()


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
    from labels import load_reviews, orphan_reviews, save_review

    st.subheader("複核佇列")
    orphan = orphan_reviews()
    if orphan:
        st.error(f"{len(orphan)} 筆裁決指到已不存在的類別 {sorted(set(orphan.values()))}"
                 " —— 那些照片會被丟掉。類別改過名的話，重裁一次補上新名字。")
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
    tier_n = q.tier.value_counts().to_dict()   # 先存，下面的篩選會改 q
    done = load_reviews()

    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    pick_tiers = c1.multiselect("看哪幾批", [names[t] for t in (4, 3, 2, 1)],
                                default=[names[t] for t in (4, 3, 2, 1)])
    keep = {t for t, n in names.items() if n in pick_tiers}
    q = q[q.tier.isin(keep)]
    if c2.toggle("只看還沒裁的", value=True):
        q = q[~q.fileId.isin(done)]
    per = c3.select_slider("一頁幾張", [6, 12, 24, 48], value=12)
    # 同一層裡最新的排前面：舊照片的標籤問題多半已經在前幾輪裁過了
    q = q.sort_values(["tier", "syncedAt", "reportDate"], ascending=False, kind="stable")
    pages = max(1, -(-len(q) // per))
    page = c4.number_input(f"第幾頁（共 {pages}）", 1, pages, 1)

    st.caption(" · ".join(
        f"**{names[t]}** {tier_n.get(t, 0)}" for t in (4, 3, 2, 1))
               + f" · 已裁 {len(done)} · 符合篩選 {len(q)}")
    if not len(q):
        st.success("這批裁完了。要讓裁決進到模型，回 ① 同步那頁重跑一次。")
        return

    page_rows = list(q.iloc[(page - 1) * per: page * per].itertuples())
    for row0 in range(0, len(page_rows), 2):
        for col, r in zip(st.columns(2), page_rows[row0:row0 + 2]):
            with col, st.container(border=True):
                _review_card(r, classes, done, save_review, names)


CANVAS_W = 460  # 畫布寬度（px）。框存的是 0~1000 比例，換裝置不會跑掉


def _patch_canvas() -> None:
    """補上 streamlit-drawable-canvas 0.9.3 要的 `streamlit.elements.image.image_to_url`。

    Streamlit 1.4x 把它搬到 `lib.image_utils`，而且第二個參數從 `width: int`
    改成 `LayoutConfig`。畫布套件還沒跟上。接一層轉接比把 Streamlit 釘回舊版划算——
    釘舊版等於整個操作台跟著退回去。哪天套件修好了，這整段可以刪。
    """
    import streamlit.elements.image as mod  # 模組還在，只是 image_to_url 被搬走了

    if hasattr(mod, "image_to_url"):
        return
    from streamlit.elements.lib.image_utils import image_to_url as _new
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def image_to_url(image, width, clamp, channels, output_format, image_id):
        return _new(image, LayoutConfig(width=int(width)), clamp, channels,
                    output_format, image_id)

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
    ch = int(round(h * scale))
    old = _boxes_now().get(r.fileId) or []
    # 已標過的框畫回去，人才知道自己標過什麼（canvas 吃像素座標，要從 0~1000 換回來）
    initial = {"version": "4.4.0", "objects": [
        {"type": "rect", "left": b[0] / 1000 * CANVAS_W, "top": b[1] / 1000 * ch,
         "width": (b[2] - b[0]) / 1000 * CANVAS_W, "height": (b[3] - b[1]) / 1000 * ch,
         "fill": "rgba(255,80,80,0.18)", "stroke": "#ff5050", "strokeWidth": 2}
        for b in old]} if old else None
    res = st_canvas(
        fill_color="rgba(255,80,80,0.18)", stroke_color="#ff5050", stroke_width=2,
        background_image=im, drawing_mode="rect", update_streamlit=True,
        width=CANVAS_W, height=ch, initial_drawing=initial, key=f"cv_{r.fileId}")
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
        st.caption(f"模型自己看的：{cam_box}　·　紅框是你標的（拖曳新增，"
                   "側欄工具可刪）")
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
    mark = st.toggle(f"✎ 標框{f'（已有 {n_old}）' if n_old else ''}",
                     key=f"mk_{r.fileId}",
                     help="拖曳框出「應該看這裡」。框會餵進訓練，不是只給人看的註記")
    cam_box = None
    if hit is not None:
        import explain
        from PIL import Image
        cam, pred, conf, box = hit
        cam_box = explain.to_1000(box)
        if not mark:
            st.image(explain.overlay(Image.open(img).convert("RGB"), cam, box, pred),
                     width="stretch")
    elif img.exists() and not mark:
        st.image(str(img), width="stretch")
    drawn = _draw_boxes(r, img, cam_box) if (mark and img.exists()) else None

    st.markdown(f'{badge("test" if r.isTest else "train")} **{txt(r.title)}**',
                unsafe_allow_html=True)
    st.caption(f"`{names[r.tier]}`　{r.why}　·　{txt(r.reportDate)} · {txt(r.constrName)}"
               + ("" if r.isTest else "　·　模型背過這張，它的信心偏樂觀"))

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
    st.markdown(" ".join(f"<span class=src>{k}</span> `{txt(v)}`" for k, v, _ in src),
                unsafe_allow_html=True)

    with st.expander("原文"):
        st.write(f"Gemini：{txt(r.predWorkItem)}")
        chips_txt = getattr(r, "chipsOn", None)
        if isinstance(chips_txt, str):
            st.write("人手打的查驗重點：")
            st.write("\n".join(f"- {x}" for x in chips_txt.split("|")))

    if r.fileId in done:
        st.info(f"已裁：{done[r.fileId]}")
    # 建議選項放前面（specKey 只到工種，不能直接當答案，所以不進建議）
    head = [v.split("（")[0] for _, v, full in src
            if full and isinstance(v, str) and v.split("（")[0] in set(classes)]
    seen, opts = set(), []
    for c in head + classes:
        if c not in seen:
            seen.add(c)
            opts.append(c)
    a, c2 = st.columns([4, 1])
    pickd = a.selectbox("實際在拍什麼", opts, key=f"rv_{r.fileId}",
                        help="前面幾個是各來源的建議，後面是全部類別")
    c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c2.button("儲存紀錄", key=f"bt_{r.fileId}", type="primary", width="stretch"):
        # 沒開標框模式就別動既有的框——沒展開畫布不代表要把框清掉
        save_review(r.fileId, pickd, "", boxes=drawn if mark else
        (_boxes_now().get(r.fileId) or None))
        st.cache_data.clear()
        # 刻意不 st.rerun()：那會把分頁彈回 ①。佇列下次互動才刷新，換來不會跳走
        st.success(f"已紀錄 {pickd}" + (f"，含 {len(drawn)} 個框" if drawn else ""))


@st.cache_data(show_spinner=False)
def _legacy(_mtime: float) -> pd.DataFrame:
    from labels import Labeler, legacy_manifest

    lg = legacy_manifest()
    if not len(lg):
        return lg
    raw = pd.read_csv(paths.LEGACY_MANIFEST, dtype=str)
    lab = Labeler.load()
    return lg.assign(
        cls=lg.title.map(lab.label),
        dupCount=pd.to_numeric(raw.dupCount, errors="coerce").fillna(1).astype(int),
        pptx=raw.pptx, slide=raw.slide)


def legacy_page() -> None:
    """歷史資料（舊 pptx 進度報告）—— 它補了什麼、有沒有進訓練集。"""
    st.subheader("歷史資料（舊版 pptx 進度報告）")
    if not paths.LEGACY_MANIFEST.exists():
        st.info("還沒抽取。跑 `make legacy`（或 "
                "`uv run --extra train src/legacy.py --root ~/Downloads/115年日報表`）。")
        return
    d = _legacy(paths.LEGACY_MANIFEST.stat().st_mtime)
    lab = Labeler.load()

    # 第一件要講清楚的事：這批到底有沒有在訓練
    cur = split_mod.current()
    sp = split_mod.load(cur) if (paths.SPLITS / f"{cur}.json").exists() else {}
    n_in = int(sp.get("trainLegacy") or 0)
    if n_in:
        st.success(f"線上的 **{cur}** 有收這批：train 裡有 {n_in} 張歷史照片"
                   f"（{len(sp.get('classes') or [])} 類）。歷史照片**只進 train，不進 test**。")
    else:
        st.warning(f"線上的 **{cur}** 沒有收這批（`source={sp.get('source', '?')}`）。"
                   "要收：`make model SPLIT=vN LEGACY=1`，或 ① 同步那頁重跑時加上。")
        st.caption("預設不收是實測結果：只加那 11 個新類別就掉 22.6pt（0.815 → 0.589），"
                   "代價來自類別空間變大——那些類別完全沒有 PMS 照片可以定界線。")

    stat = {}
    sf = paths.LEGACY_MANIFEST.parent / "stats.json"
    if sf.exists():
        stat = (json.loads(sf.read_text()).get("stat") or {})
    k = st.columns(5)
    k[0].metric("留下的照片", len(d))
    k[1].metric("pptx", stat.get("pptx", "—"))
    k[2].metric("工地", d.constrName.nunique())
    k[3].metric("可訓練類別", int(d.cls.notna().sum() and
                                  d[d.cls.notna() & (d.cls != lab.fallback)].cls.nunique()))
    dates = d.reportDate.dropna()
    dates = dates[dates != ""]
    k[4].metric("日期範圍", f"{dates.min()[5:]} ~ {dates.max()[5:]}" if len(dates) else "—")

    if stat:
        st.markdown("**去重：抽出來一半是重複的**")
        st.dataframe(pd.DataFrame([
            {"擋掉的": "同一份 pptx 裡重複引用", "張數": stat.get("dupPhotos"),
             "靠什麼": "sha1（位元組完全相同）"},
            {"擋掉的": "不同 pptx 重壓縮過的同一張", "張數": stat.get("dupVisual"),
             "靠什麼": "dhash（sha1 抓不到）"},
            {"擋掉的": "與 PMS 已下載的撞畫面", "張數": stat.get("dupWithPms"),
             "靠什麼": "dhash 比對 data/raw/photos"},
            {"擋掉的": "太小（logo/圖示）", "張數": stat.get("tinySkipped"),
             "靠什麼": "< 200×200 px"},
            {"擋掉的": "沒有工項標題的投影片", "張數": stat.get("noTitleSlides"),
             "靠什麼": "版面垂直位置認不到標題"},
        ]), width="stretch", hide_index=True)
        st.caption(f"{stat.get('slides')} 張投影片 → {stat.get('workItemSlides')} 張是工項頁 "
                   f"→ 留下 {len(d)} 張照片。pptx 一定會重新編碼貼進去的圖，"
                   "所以 sha1 對跨檔重複無效。")

    st.divider()
    st.markdown("**它補了哪些類別** —— 打勾的是 PMS 湊不到門檻、只有這批撐得起來的")
    pms = labeled()
    a = d[d.cls.notna() & (d.cls != lab.fallback)].cls.value_counts().rename("歷史")
    b = pms.cls.value_counts().rename("PMS") if len(pms) else pd.Series(dtype=int, name="PMS")
    cmp = pd.concat([a, b], axis=1).fillna(0).astype(int).sort_values("歷史", ascending=False)
    cmp["只有歷史資料有"] = cmp.PMS.eq(0)
    st.dataframe(cmp, width="stretch")
    drop = int((d.cls == lab.fallback).sum()) + int(d.cls.isna().sum())
    st.caption(f"落到 fallback 或被 junk 排除的 {drop} 張不進訓練"
               f"（`drop_fallback: true`）——那是一袋互不相干的東西，不是一個類別。"
               "PMS 欄是**可訓練張數**（已套 min_class_size），所以未達門檻的顯示 0。")

    st.divider()
    c1, c2, c3 = st.columns([2, 2, 2])
    site = c1.selectbox("工地", ["全部"] + sorted(d.constrName.dropna().unique().tolist()),
                        key="lg_site")
    v = d if site == "全部" else d[d.constrName == site]
    kls = c2.selectbox("類別", ["全部"] + sorted(v.cls.dropna().unique().tolist()), key="lg_cls")
    v = v if kls == "全部" else v[v.cls == kls]
    only_dup = c3.toggle("只看被反覆引用的（dupCount > 1）", value=False,
                         help="同一張出現在多份 pptx = 被當代表照反覆使用，通常畫面較好")
    if only_dup:
        v = v[v.dupCount > 1]
    v = v.sort_values(["reportDate", "slide"], ascending=False, kind="stable")
    st.caption(f"{len(v)} 張 · 最新的前 60")
    cols = st.columns(6)
    for i, r in enumerate(v.head(60).itertuples()):
        img = paths.LEGACY_IMAGES / f"{r.fileId}.jpg"
        with cols[i % 6]:
            if img.exists():
                st.image(str(img), width="stretch")
            st.markdown(f'<div class=pc6><span class=m>{txt(r.cls)} · {txt(r.reportDate)}</span>'
                        f'<div class=m title="{txt(r.title)}">{txt(r.title)}</div>'
                        f'<div class=m>{txt(r.constrName)} · 引用 {r.dupCount} 次</div>'
                        f'<div class=m title="{txt(r.pptx)}">{txt(r.pptx)} p.{txt(r.slide)}</div>'
                        '</div>', unsafe_allow_html=True)


def report_view(runs, key: str):
    """報告分頁 —— PMS 與 QMS 共用同一份報告格式。"""
    if not runs:
        st.info("還沒有報告。")
        return
    pick = st.selectbox("run", runs, format_func=lambda p: p.name, key=key)
    if (pick / "comparison.md").exists():
        st.markdown((pick / "comparison.md").read_text())
    m = json.loads((pick / "metrics.json").read_text()) \
        if (pick / "metrics.json").exists() else {}
    if m:
        a, b, c = st.columns(3)
        a.metric("top-1", m["top1"])
        b.metric("macro-F1", m["macroF1"])
        c.metric("測試張數", m["support"])
        if m.get("coverageCurve"):
            st.caption("信心門檻 → 自動處理比例與該批準確率（決定何時能少靠人工）")
            st.dataframe(pd.DataFrame(m["coverageCurve"]).rename(columns={
                "threshold": "門檻", "coverage": "自動處理",
                "accuracyOnCovered": "該批準確率", "n": "張數"}),
                width="stretch", hide_index=True)
        if m.get("perClass"):
            st.caption("每類指標")
            st.dataframe(pd.DataFrame(m["perClass"]).T.sort_values("support", ascending=False),
                         width="stretch", height=420)
    if (pick / "confusion.png").exists():
        st.image(str(pick / "confusion.png"))
    errs = sorted((pick / "errors").glob("*.jpg")) if (pick / "errors").exists() else []
    if errs:
        st.caption("誤判樣本（檔名 = 真實_預測）")
        cols = st.columns(6)
        for i, e in enumerate(errs):
            cols[i % 6].image(str(e), caption=e.stem[:28], width="stretch")


# ======================= PMS 日報 =======================================
def pms_page(tabs):
    if tabs[0]:
        st.subheader("每日同步")
        st.caption(f"來源 {os.getenv('API_BASE_URL', 'https://pms.example.invalid/')}"
                   f" · raw 只抄不改 · derived 隨時可砍")

        # 這頁只放三顆按鈕，各自負責一件別人不做的事。
        # 之前還有一顆「前處理（遮蔽 + 縮圖）」，它是「重跑模型」第 2 步的子集、
        # 沒有任何自己的選項，連說明都寫著「整條鏈看下面」——那不是按鈕，是陷阱：
        # 按了以為有進度，其實下面還得整條再跑一次。已移除，要單步跑用 make。
        with st.container(border=True):
            st.markdown("**1. 抓新資料** — 只抓，不重訓")
            st.caption("想知道今天有沒有新照片就按這顆。**抓完模型還是舊的**，"
                       "要讓新照片進到模型得再跑下面的「全鏈重跑」。")
            c1, c2 = st.columns([1, 2])
            limit = c1.number_input("只抓前 N 份日報（0 = 全部）", 0, 5000, 0, step=10)
            full = c1.checkbox("忽略快取全部重抓", value=False,
                               help="平常不用。SUBMITTED 且 version 沒變的日報預設不重抓 detail。")
            if c2.button("▶ 同步日報 + 下載照片", type="primary"):
                run_step(sync_mod.sync, full=full, limit=int(limit) or None)

        with st.container(border=True):
            pipeline_panel()

        with st.container(border=True):
            st.markdown("**3. 建照片結構樹** — 跟訓練無關")
            st.caption(f"重建 `{paths.TREE}` 的 symlink（by-date / by-site / by-class），"
                       "只影響用檔案總管翻照片。不在上面任何一條鏈裡，想跑再跑。")
            if st.button("▶ 建照片結構樹"):
                run_step(organize.build)

        st.divider()
        m = load_manifest()
        a, b, c, d = st.columns(4)
        a.metric("manifest 列數", len(m))
        b.metric("raw 照片", len(list(paths.PHOTOS.glob("*"))) if paths.PHOTOS.exists() else 0)
        c.metric("已前處理", len(list(paths.IMAGES.glob("*.jpg"))) if paths.IMAGES.exists() else 0)
        d.metric("日報快照",
                 len(list(paths.REPORTS_JSON.glob("*.json"))) if paths.REPORTS_JSON.exists() else 0)

    if tabs[1]:
        df = labeled()
        if not len(df):
            st.info("還沒有資料，先到「① 同步」跑一次。")
        else:
            st.subheader(f"{len(df)} 張 / {df.cls.nunique()} 類 / {df.constrId.nunique()} 個工地")
            c1, c2 = st.columns(2)
            c1.caption("各類別張數")
            c1.bar_chart(df.cls.value_counts())
            c2.caption("每天照片數")
            c2.bar_chart(df.groupby("reportDate").size())
            st.caption("工地 × 類別（切分策略的依據：類別與工地高度綁定）")
            st.dataframe(pd.crosstab(df.cls, df.constrName.fillna(df.constrId)), width="stretch")
            with st.expander("原始 manifest"):
                st.dataframe(df, width="stretch", height=400)

    if tabs[2]:
        df = labeled()
        if not len(df):
            st.info("還沒有照片。")
            return
        lab = Labeler.load()
        preds, test_ids = local_preds("siglip", split_mod.current())
        df = df.assign(local=df.fileId.map(preds), gem=df.predWorkItem)
        df["gemNorm"] = df.gem.map(
            lambda t: (lab.label(t) or lab.fallback) if isinstance(t, str) and t else None)
        # Gemini 正規化後可能命中一個我們根本沒在訓練的類（張數不足被排除），
        # 那是不存在的答案，標成「類別外」而不是硬塞進某一類
        df.loc[~df.gemNorm.isin(set(df.cls.unique())) & df.gemNorm.notna(), "gemNorm"] = "類別外"

        c1, c2, c3, c4 = st.columns(4)
        site = c1.selectbox("工地", ["全部"] + sorted(df.constrName.dropna().unique().tolist()))
        d2 = df if site == "全部" else df[df.constrName == site]
        day = c2.selectbox("日期", ["全部"] + sorted(d2.reportDate.dropna().unique(), reverse=True))
        d2 = d2 if day == "全部" else d2[d2.reportDate == day]
        cls = c3.selectbox("類別（人工日報標題推出的真值）",
                           ["全部"] + sorted(d2.cls.unique().tolist()))
        d2 = d2 if cls == "全部" else d2[d2.cls == cls]
        pvs = sorted(d2.promptVersion.dropna().unique().tolist())
        pv = c4.selectbox("prompt 版本", ["全部"] + pvs)
        d2 = d2 if pv == "全部" else d2[d2.promptVersion == pv]

        if not preds:
            st.warning("尚無本地模型預測。先跑 features.py → train.py。")
        c1, c2, c3 = st.columns([2, 2, 1])
        show_masked = c1.toggle("顯示遮蔽後版本", value=True)
        show_box = c2.toggle("畫出 Gemini 佐證框", value=False,
                             help="v3 prompt 才有。框住整張圖的那種對訓練沒用，"
                                  "卡片上會標出最大框的面積佔比")
        # 最新加入的排最前面。manifest 是 append-only，原順序等於「最舊的在最前」——
        # 每天看的都是同一批老照片，新同步進來的要翻到最後才看得到。
        d2 = d2.sort_values(
            [c for c in ("syncedAt", "reportDate", "pageSort", "serial") if c in d2],
            ascending=False, kind="stable")
        c3.caption(f"{len(d2)} 張 · 最新的前 60")
        st.caption("綠色 TRAIN = 這張在訓練集，模型背過，它的判斷不能當成績；"
                   "紅色 TEST = 沒背過，只有這些算數。"
                   "判斷結果綠字代表與人工日報的分類一致，紅字代表不一致。"
                   "`v3` 標記 = 2026-08-17 新 prompt（多了信心值、位置、次要工項、佐證框）。")

        cols = st.columns(6)
        for i, r in enumerate(d2.head(60).itertuples()):
            src = (paths.IMAGES / f"{r.fileId}.jpg") if show_masked else None
            if src is None or not src.exists():
                hits = list(paths.PHOTOS.glob(f"{r.fileId}.*"))
                src = hits[0] if hits else None
            has_box = isinstance(getattr(r, "predBoxes", None), str)
            with cols[i % 6]:
                if src and src.exists():
                    st.image(with_boxes(str(src), r.predBoxes) if (show_box and has_box)
                             else str(src), width="stretch")
                conf = getattr(r, "predConf", None)
                conf_s = "" if conf is None or conf != conf else f"（{conf:.2f}）"
                sec = getattr(r, "predSecondary", None)
                extra = []
                if isinstance(sec, str):
                    extra.append("次要：" + "、".join(json.loads(sec)))
                if has_box:
                    extra.append(f"框{len(json.loads(r.predBoxes))}"
                                 f"·最大{box_area(r.predBoxes):.0%}")
                is_v3 = str(getattr(r, "promptVersion", "")).startswith("v3")
                st.markdown(
                    f'<div class=pc6>'
                    f'<div>{badge("test" if r.fileId in test_ids else "train")}'
                    f'<span class=m> {r.cls} · {r.reportDate}'
                    + ('<b> · v3</b>' if is_v3 else '') + '</span></div>'
                    + verdict("系統判斷", r.local, r.cls)
                    # Gemini 顯示它**自己寫的字**，不是正規化後的類別——正規化是我們
                    # 為了比對硬折過去的，看板上要看的是它到底講了什麼。
                    # 綠/紅仍然依正規化後的結果上色，一致與否照樣一眼看得出來。
                    + gem_line(r.gem, r.gemNorm, r.cls, conf_s)
                    + (f'<div class=m>{" · ".join(extra)}</div>' if extra else
                       '<div class=m>&nbsp;</div>')
                    + f'<div class=m>G判斷位置：{txt(getattr(r, "predLocation", None))}</div>'
                    + f'<div class=m title="{txt(r.title)}">日報標題：{txt(r.title)}</div>'
                      f'</div>', unsafe_allow_html=True)

        evidence_view(d2.head(60))

    if tabs[3]:
        review_queue()

    if tabs[4]:
        legacy_page()

    if tabs[5]:
        st.subheader("labels.yaml")
        st.caption("順序即優先權，第一個命中者勝。改完存檔 → 重跑組織/切分即可，不必重新下載。")
        text = st.text_area("規則", paths.LABELS_YAML.read_text(), height=320)
        if st.button("💾 存檔並重算"):
            paths.LABELS_YAML.write_text(text)
            st.cache_data.clear()
            st.rerun()
        probe = st.text_input("試打一個標題", "13F外牆打底粉光")
        if probe:
            st.write("→", Labeler.load().label(probe) or "（排除）")
        st.divider()
        # 照片沒進訓練只有兩種原因，擺在一起才看得出全貌：
        #   規則認得但張數不夠 → 在排隊（下面第一塊）
        #   規則根本沒認領     → 缺一條規則（下面第二塊）
        from labels import pending_classes, unclaimed as _unclaimed
        import newclass
        lab_now = Labeler.load()
        pend = pending_classes()
        st.markdown(f"**排隊中的類別 {len(pend)} 類** — 規則認得，張數還不夠")
        st.caption(f"門檻 `min_class_size: {lab_now.min_class_size}`，算的是 **train + test 全部**"
                   "（切分之前就先砍了），不是 train。張數一過門檻**自動**進訓練，"
                   "不用改任何程式——`金屬-欄杆鐵件` 就是這樣在 v18 自己冒出來的。"
                   "別跟 `MIN_TRAIN` 搞混，那個是進場之後的切分保底。")
        if len(pend):
            st.dataframe(pend.rename(columns={
                "cls": "類別", "photos": "現有", "need": "還差", "latest": "最近一張"}),
                width="stretch", hide_index=True)
            st.progress(min(1.0, float(pend.photos.sum()) / float(pend.need.sum() + pend.photos.sum())),
                        text=f"這 {len(pend)} 類合計 {int(pend.photos.sum())} 張，"
                             f"全部上線還差 {int(pend.need.sum())} 張")
        else:
            st.success("沒有類別卡在門檻下。")

        st.divider()
        # 這裡刻意**不用** labeled()。它走 apply(drop_small=True)，配上
        # drop_fallback: true，fallback 那些列在到達這裡之前就被濾掉了——
        # 這個面板從寫出來到 2026-08-25 為止一直顯示 0 張，等於沒人看得到新工種。
        other = _unclaimed()
        st.markdown(f"**沒有規則認領的 {len(other)} 張** — 下一批新工種在這裡")
        st.caption("照片沒被刪，都在 `raw/photos/`，只是沒進訓練集（`drop_fallback: true`）。"
                   "少的是一條規則，不是資料。")
        if len(other):
            st.dataframe(other.title.value_counts().rename("張數"), width="stretch")
        with st.expander("候選新工種（含舊 pptx 那批一起算）", expanded=not len(other)):
            buf = []
            cand = newclass.report(src="all", log=lambda *a: buf.append(" ".join(map(str, a))))
            st.code("\n".join(buf) or "(無)")
            if cand:
                st.caption("貼進上面的規則框之前先看標題明細——詞是照「蓋得到幾張」挑的，"
                           "常常是跨詞界的碎片（`櫃安` = 櫥櫃+安裝，因為它連「廚櫃」的錯字一起收）。")

    if tabs[6]:
        runs = sorted([p for p in paths.REPORTS_OUT.glob("*") if p.is_dir()
                       and "qms" not in p.name], reverse=True) \
            if paths.REPORTS_OUT.exists() else []
        report_view(runs, "pms_run")


# ======================= QMS 稽核 =======================================
def qms_page(tabs):
    df = load_qms()

    if tabs[0]:
        st.subheader("QMS 稽核照同步")
        st.caption(f"來源 {os.getenv('QMS_API_BASE_URL', 'https://qms.example.invalid/api/qms')}"
                   f" · 正式環境，只讀 · 母體 36,653 張")
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.number_input("抽樣張數", 500, 40000, 12000, step=500)
            cap = st.number_input("每中類上限（0 = 純隨機）", 0, 5000, 600, step=100)
            if st.button("▶ 抽樣 + 下載", type="primary"):
                import qms
                run_step(qms.sample, n=int(n), per_class_cap=int(cap))
        with c2:
            if st.button("▶ 重掃母體（只 query）"):
                import qms
                run_step(qms.scan_cells)
            if st.button("▶ 前處理（遮左下浮水印）"):
                run_step(prepare.run, kind="qms")
        with c3:
            if st.button("▶ 重建切分（按格切）"):
                run_step(split_mod.build_qms, name="qms-v1")
            if st.button("▶ 重跑 ONNX 推論"):
                import predict
                run_step(predict.run, ckpt="backbone-qms", src="qms")

        st.divider()
        a, b, c, d = st.columns(4)
        a.metric("manifest 列數", len(df))
        b.metric("raw 照片",
                 len(list(paths.QMS_PHOTOS.glob("*"))) if paths.QMS_PHOTOS.exists() else 0)
        c.metric("已前處理",
                 len(list(paths.QMS_IMAGES.glob("*.jpg"))) if paths.QMS_IMAGES.exists() else 0)
        d.metric("格數", df.constructionInsId.nunique() if len(df) else 0)

    if not len(df):
        for t in tabs[1:]:
            with t:
                st.info("還沒有 QMS 資料。到「① 同步」抽樣下載，或跑 `uv run src/qms.py --sample 12000`")
        return

    has_pred = "pred" in df and df.pred.notna().any()

    if tabs[1]:
        st.subheader(f"{len(df)} 張 / {df.cls.nunique()} 個中類 / "
                     f"{df.constructionInsId.nunique()} 格")
        c1, c2 = st.columns(2)
        c1.caption("各中類張數")
        c1.bar_chart(df.cls.value_counts())
        c2.caption("建案 × 大類")
        c2.dataframe(pd.crosstab(df.l1, df.constrName), width="stretch")
        if has_pred:
            d = df[df.pred.notna()]
            st.caption("每類正確率（含訓練集，僅供概觀；正式數字看「④ 報告」）")
            # 用聚合不用 apply：apply(include_groups=False) 會把 cls 從 group 裡拿掉
            acc = (d.assign(hit=d.pred == d.cls).groupby("cls")
                   .agg(張數=("hit", "size"), 正確率=("hit", "mean"), 平均信心=("conf", "mean"))
                   .round(3).sort_values("正確率"))
            st.dataframe(acc, width="stretch", height=420)
        else:
            st.warning("尚無模型預測。到「① 同步」按「重跑 ONNX 推論」。")

    if tabs[2]:
        c1, c2, c3, c4 = st.columns(4)
        site = c1.selectbox("建案", ["全部"] + sorted(df.constrName.dropna().unique().tolist()),
                            key="q_site")
        d2 = df if site == "全部" else df[df.constrName == site]
        cls = c2.selectbox("中類", ["全部"] + sorted(d2.cls.unique().tolist()), key="q_cls")
        d2 = d2 if cls == "全部" else d2[d2.cls == cls]
        part = c3.selectbox("資料分割", ["全部", "test", "val", "train"], key="q_part")
        if part != "全部" and "part" in d2:
            d2 = d2[d2.part == part]
        view = c4.selectbox("看什麼", ["全部", "只看判錯的", "只看低信心"], key="q_view")
        if has_pred and view == "只看判錯的":
            d2 = d2[d2.pred.notna() & (d2.pred != d2.cls)]
        elif has_pred and view == "只看低信心":
            d2 = d2[d2.conf < 0.8]
        st.caption(f"{len(d2)} 張 · 顯示前 60 · 灰塊是遮掉的浮水印 · "
                   "紅色 TEST = 模型沒背過，只有這些算數")

        cols = st.columns(6)
        for i, r in enumerate(d2.head(60).itertuples()):
            src = paths.QMS_IMAGES / f"{r.fileId}.jpg"
            with cols[i % 6]:
                if src.exists():
                    st.image(str(src), width="stretch")
                conf = getattr(r, "conf", None)
                st.markdown(
                    f'<div class=pc>'
                    f'<div>{badge(str(getattr(r, "part", "train")))}'
                    f'<span class=m> {r.cls}</span></div>'
                    + verdict("系統判斷", getattr(r, "pred", None), r.cls)
                    + f'<div class=m>信心：{"－" if conf is None or conf != conf else f"{conf:.2f}"}'
                      f'</div>'
                    + f'<div class=m title="{txt(r.l3)}">項目：{txt(r.l3)}</div>'
                    + f'<div class=m>{txt(r.floor)} · {txt(r.room)}</div>'
                      f'</div>', unsafe_allow_html=True)

    if tabs[3]:
        runs = sorted([p for p in paths.REPORTS_OUT.glob("*qms*") if p.is_dir()], reverse=True) \
            if paths.REPORTS_OUT.exists() else []
        report_view(runs, "qms_run")


def standards_page(tabs) -> None:
    """規範庫：QS 公司標準 + 合約工作約定。

    這一頁不碰照片也不碰模型——它是**判定基準**的來源。
    先前 qsdata/contractdata 兩層寫好了卻沒有任何界面用得到，
    這頁把它們接上來，讓複核時查得到「這一項的標準數值是多少」。
    """
    import contractdata
    import qsdata

    if tabs[0]:                                    # ① QS 標準
        docs = qsdata.load()
        items = [i for i in qsdata.all_items(docs) if i.status == "R"]
        c = st.columns(4)
        c[0].metric("標準份數", len(docs))
        c[1].metric("檢查項", len(items))
        vision = sum(1 for i in items if i.kind in ("A", "B"))
        c[2].metric("vision 可判", f"{vision / len(items) * 100:.0f}%",
                    help="A 純視覺 + B 量測。C 文件/D 時序/E 儀器走別的路")
        c[3].metric("請款靶", len(qsdata.billing_items(docs)),
                    help="明文「須拍照存證，做為請款之憑證」")

        pick = st.selectbox("挑一份標準", sorted(docs),
                            format_func=lambda d: f"{d} {docs[d].name}")
        d = docs[pick]
        if d.phases:
            st.caption("工序階段（標準自己定義的 OPTIONAL 節點）")
            st.markdown(" → ".join(f"**{p.name.rstrip('：:')}**" for p in d.phases))
        else:
            st.caption("⚠ 這份完全扁平（無 OPTIONAL 節點），工序需另行推論")
        st.dataframe(
            pd.DataFrame([{"項次": i.item_no, "類型": i.kind, "檢查項": i.name,
                           "請款": "◆" if i.is_billing else "",
                           "罰則": "◆" if i.is_penalty else "",
                           "看合約": "◆" if i.is_contract else ""}
                          for i in d.required]),
            width="stretch", hide_index=True)

    if tabs[1]:                                    # ② 合約工作約定
        cdocs = contractdata.load()
        cs = contractdata.all_clauses(cdocs)
        c = st.columns(4)
        c[0].metric("合約份數", len(cdocs))
        c[1].metric("條款", len(cs))
        c[2].metric("付款節點", len(contractdata.payment_terms(cdocs)))
        c[3].metric("罰則", len(contractdata.penalties(cdocs)))
        st.caption("⚠ 工作約定**逐案有效**——主鍵帶案名，不同案的同一工種數值可能不同")

        opts = [f"{d.project}／{d.trade}（{d.vendor}·{d.doc_date}）" for d in cdocs]
        i = st.selectbox("挑一份合約", range(len(cdocs)), format_func=lambda x: opts[x])
        d = cdocs[i]
        if d.kind == "物明":
            st.warning("這份是**物料明細**（材料供應），不是施工工作約定——"
                       "回答不了施工類的 QS 檢查項")
        st.dataframe(
            pd.DataFrame([{"條號": x.no, "分類": x.kind, "來源": x.sheet, "條文": x.text}
                          for x in d.clauses]),
            width="stretch", hide_index=True)

    if tabs[2]:                                    # ③ 衝突比對
        st.caption("同工種內，QS 與合約都給了數值的項目。"
                   "**這是候選不是結論**——程式只能證明兩邊在談同一主題且都有數字。")
        rs = contractdata.cross_check()
        if not rs:
            st.info("沒有候選。可能是該工種的合約沒給數值，不代表沒有衝突。")
        for r in rs:
            with st.expander(f"{r['project']}／{r['trade']} · {r['qsDoc']} · {r['topic']}"):
                a, b = st.columns(2)
                a.markdown("**QS 公司標準**")
                for k, t in r["qs"]:
                    a.markdown(f"`{k}`　{t}")
                b.markdown("**合約工作約定**")
                for k, t in r["contract"]:
                    b.markdown(f"`{k.split('/')[-1]}`　{t}")
        st.divider()
        st.markdown(
            "**判定基準用哪個：合約優先。** 不是因為數字比較嚴，而是 "
            "`QS0302-5` 自己寫「依**合約中工作約定要點**施工」——QS 把裁量權讓給合約；"
            "且合約帶罰則（扣款、止付），QS 只是檢查表。"
            "已人工確認的三處衝突見 `reference/contract/CONTRACT_VS_QS.md`。")

    if tabs[3]:                                    # ④ 工種介面
        st.caption("QS 按**單一工種**編排，交界被切碎（實測 28 個介面項散在 9 份標準裡）。"
                   "合約是逐工種簽的，同一個交界會在兩份合約各出現一次——"
                   "**兩邊都拿到才是完整的**。")
        for it in contractdata.INTERFACES:
            cs = contractdata.interface_clauses(it["name"])
            with st.expander(f"{it['name']}（{len(cs)} 條）"):
                st.info(it["note"])
                for c in cs:
                    st.markdown(f"**{c.trade}** `{c.no}`　{c.text}")

    if tabs[4]:                                    # ⑤ 合約相依缺口
        docs = qsdata.load()
        ci = qsdata.contract_items(docs)
        answered = contractdata.QS_ANSWERS
        st.metric("合約相依項", f"{len(answered)} / {len(ci)}",
                  help="判定基準指向合約而非 QS 的檢查項，目前已對應到合約條款的比例")
        st.caption("這些項目**單靠 QS 答不出來**。RAG 若只灌 QS，"
                   "檢索會命中但回答不了「合不合格」。")
        rows = []
        for i in ci:
            got = answered.get(i.key, [])
            rows.append({"狀態": "✓" if got else "—", "QS 項": i.key,
                         "檢查項": i.name,
                         "合約條款": "、".join(got) if got else "尚未取得"})
        st.dataframe(pd.DataFrame(rows).sort_values("狀態", ascending=False),
                     width="stretch", hide_index=True)


# ---------- 進入點 -----------------------------------------------------
SYSTEM = st.segmented_control(
    "系統", ["PMS 日報", "QMS 稽核", "規範庫"], default="PMS 日報",
    label_visibility="collapsed")

if SYSTEM == "QMS 稽核":
    qms_page(nav(["① 同步", "② 資料總覽", "③ 照片", "④ 報告"], "qms_nav"))
elif SYSTEM == "規範庫":
    standards_page(nav(["① QS 標準", "② 工作約定", "③ 衝突比對", "④ 工種介面",
                        "⑤ 合約缺口"], "std_nav"))
else:
    pms_page(nav(["① 同步", "② 資料總覽", "③ 照片", "④ 複核佇列",
                  "⑤ 歷史資料", "⑥ 標籤規則", "⑦ 報告"], "pms_nav"))
