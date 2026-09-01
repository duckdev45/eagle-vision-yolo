"""重跑管線 + 「判斷依據」遮擋法解釋。"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from datetime import date

import streamlit as st

import paths
import split as split_mod

from .common import txt


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
    from PIL import Image

    import explain

    im = Image.open(paths.IMAGES / f"{file_id}.jpg").convert("RGB")
    hit = _cams(split_mod.current()).get(file_id) if grid == explain.GRID else None
    cam, pred, conf, box = hit or explain.probe_cam(
        im, _probe("siglip", split_mod.current()), _encoder(), grid
    )
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
        st.caption(
            "遮住一格 → 重新編碼 → 看這個答案掉多少。掉最多的那格就是依據。"
            f"要跑 {grid * grid + 1} 次編碼，約數秒。"
        )
        return
    with st.spinner("遮擋中…"):
        img, pred, conf, box = _evidence(pick, grid)
    st.image(img, caption=f"左＝最關鍵的一塊，右＝熱區　·　判斷 {pred}（{conf:.2f}）", width="stretch")
    st.caption(
        f"框（與 Gemini evidence 同格式 0~1000）：{box}　·　"
        "紅色愈深＝遮掉那塊、這個答案掉愈多。熱區落在天空、工人、鷹架、"
        "行道樹上，分數再高也不可信 —— 那是捷徑，不是依據。"
    )


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
    import prepare
    import sync as sync_mod
    import train as tr

    # 這串必須與 Makefile 的 `data` + `model` 逐步對齊。兩邊漂開的話，
    # 按鈕跑出來的 split 跟 `make model` 跑出來的不是同一份，分數就沒得比。
    steps = []
    if with_data:
        steps += [
            ("同步日報 + 照片", lambda log: sync_mod.sync(log=log)),
            ("前處理", lambda log: prepare.run(kind="report", log=log)),
            ("抽 embedding", lambda log: features.extract(log=log)),
        ]
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
    st.caption(f"操作台現在看的是 **{cur}**。跑完**不會自動切換**，分數看過覺得可以，再按最下面那顆。")
    c1, c2, c3 = st.columns([2, 2, 2])
    name = c1.text_input(
        "新的 split 名字", _next_split_name(), help="每次給新名字。原地覆蓋的話，同名舊報告的分母就對不上了。"
    )
    full = c2.button(
        "▶ 全鏈重跑",
        type="primary",
        help="有新照片時用。同步 → 前處理 → embedding → 裁切框 → "
        "切分 → 訓練 → 評估 → 熱區 → 學習紀錄（含上面第 1 顆的工作）",
    )
    only = c3.button(
        "▶ 只重跑模型", help="只改過 labels.yaml 或複核裁決時用。跳過前三步（與標籤無關），其餘一樣"
    )
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
        m = json.loads((rep / "metrics.json").read_text()) if (rep / "metrics.json").exists() else {}
        st.success(
            f"`{done}` 跑完了"
            + (f" — top-1 {m['top1']} / macro-F1 {m['macroF1']} / {m['support']} 張" if m else "")
        )
        st.caption("切換是獨立一步：換掉所有人看到的答案是個決定，不該是跑完訓練的副作用。")
        st.caption(
            f"⚠ 這個 top-1 **不能直接跟 {cur} 的比**。split 按「每個工地最新幾天」"
            "滾動切，每一版的考卷是不同照片（實測 v18∩v20 只重疊 83/141），"
            "分數升降有一部分是換考卷換的。要比模型強弱，看 "
            "`reports/JOURNAL.md` 的 per-class 那節，或讓兩顆考同一份卷。"
        )
        if st.button(f"✔ 把操作台切到 {done}"):
            split_mod.set_current(done)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("pipeline_done", None)
            st.rerun()
