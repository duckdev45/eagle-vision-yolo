"""PMS 日報系統頁（① 同步 ② 資料總覽 ③ 照片 ④ 複核 ⑤ 歷史 ⑥ 標籤 ⑦ 報告）。"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

import boxes
import organize
import paths
import split as split_mod
import sync as sync_mod
from labels import Labeler

from .common import badge, gem_line, run_step, txt, verdict
from .data import labeled, load_manifest, local_preds
from .legacy_ui import legacy_page
from .pipeline import evidence_view, pipeline_panel
from .report_view import report_view
from .review_ui import review_queue

box_area = boxes.area


@st.cache_data(show_spinner=False)
def with_boxes(path: str, boxes_json: str) -> object:
    return boxes.draw(path, boxes_json)


def pms_page(tabs):
    if tabs[0]:
        st.subheader("每日同步")
        st.caption(
            f"來源 {os.getenv('API_BASE_URL', '（未設 API_BASE_URL）')} · raw 只抄不改 · derived 隨時可砍"
        )

        # 這頁只放三顆按鈕，各自負責一件別人不做的事。
        # 之前還有一顆「前處理（遮蔽 + 縮圖）」，它是「重跑模型」第 2 步的子集、
        # 沒有任何自己的選項，連說明都寫著「整條鏈看下面」——那不是按鈕，是陷阱：
        # 按了以為有進度，其實下面還得整條再跑一次。已移除，要單步跑用 make。
        with st.container(border=True):
            st.markdown("**1. 抓新資料** — 只抓，不重訓")
            st.caption(
                "想知道今天有沒有新照片就按這顆。**抓完模型還是舊的**，"
                "要讓新照片進到模型得再跑下面的「全鏈重跑」。"
            )
            c1, c2 = st.columns([1, 2])
            limit = c1.number_input("只抓前 N 份日報（0 = 全部）", 0, 5000, 0, step=10)
            full = c1.checkbox(
                "忽略快取全部重抓",
                value=False,
                help="平常不用。SUBMITTED 且 version 沒變的日報預設不重抓 detail。",
            )
            if c2.button("▶ 同步日報 + 下載照片", type="primary"):
                run_step(sync_mod.sync, full=full, limit=int(limit) or None)

        with st.container(border=True):
            pipeline_panel()

        with st.container(border=True):
            st.markdown("**3. 建照片結構樹** — 跟訓練無關")
            st.caption(
                f"重建 `{paths.TREE}` 的 symlink（by-date / by-site / by-class），"
                "只影響用檔案總管翻照片。不在上面任何一條鏈裡，想跑再跑。"
            )
            if st.button("▶ 建照片結構樹"):
                run_step(organize.build)

        st.divider()
        m = load_manifest()
        a, b, c, d = st.columns(4)
        a.metric("manifest 列數", len(m))
        b.metric("raw 照片", len(list(paths.PHOTOS.glob("*"))) if paths.PHOTOS.exists() else 0)
        c.metric("已前處理", len(list(paths.IMAGES.glob("*.jpg"))) if paths.IMAGES.exists() else 0)
        d.metric(
            "日報快照", len(list(paths.REPORTS_JSON.glob("*.json"))) if paths.REPORTS_JSON.exists() else 0
        )

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
            lambda t: (lab.label(t) or lab.fallback) if isinstance(t, str) and t else None
        )
        # Gemini 正規化後可能命中一個我們根本沒在訓練的類（張數不足被排除），
        # 那是不存在的答案，標成「類別外」而不是硬塞進某一類
        df.loc[~df.gemNorm.isin(set(df.cls.unique())) & df.gemNorm.notna(), "gemNorm"] = "類別外"

        c1, c2, c3, c4 = st.columns(4)
        site = c1.selectbox("工地", ["全部", *sorted(df.constrName.dropna().unique().tolist())])
        d2 = df if site == "全部" else df[df.constrName == site]
        day = c2.selectbox("日期", ["全部", *sorted(d2.reportDate.dropna().unique(), reverse=True)])
        d2 = d2 if day == "全部" else d2[d2.reportDate == day]
        cls = c3.selectbox("類別（人工日報標題推出的真值）", ["全部", *sorted(d2.cls.unique().tolist())])
        d2 = d2 if cls == "全部" else d2[d2.cls == cls]
        pvs = sorted(d2.promptVersion.dropna().unique().tolist())
        pv = c4.selectbox("prompt 版本", ["全部", *pvs])
        d2 = d2 if pv == "全部" else d2[d2.promptVersion == pv]

        if not preds:
            st.warning("尚無本地模型預測。先跑 features.py → train.py。")
        c1, c2, c3 = st.columns([2, 2, 1])
        show_masked = c1.toggle("顯示遮蔽後版本", value=True)
        show_box = c2.toggle(
            "畫出 Gemini 佐證框",
            value=False,
            help="v3 prompt 才有。框住整張圖的那種對訓練沒用，卡片上會標出最大框的面積佔比",
        )
        # 最新加入的排最前面。manifest 是 append-only，原順序等於「最舊的在最前」——
        # 每天看的都是同一批老照片，新同步進來的要翻到最後才看得到。
        d2 = d2.sort_values(
            [c for c in ("syncedAt", "reportDate", "pageSort", "serial") if c in d2],
            ascending=False,
            kind="stable",
        )
        c3.caption(f"{len(d2)} 張 · 最新的前 60")
        st.caption(
            "綠色 TRAIN = 這張在訓練集，模型背過，它的判斷不能當成績；"
            "紅色 TEST = 沒背過，只有這些算數。"
            "判斷結果綠字代表與人工日報的分類一致，紅字代表不一致。"
            "`v3` 標記 = 2026-08-17 新 prompt（多了信心值、位置、次要工項、佐證框）。"
        )

        cols = st.columns(6)
        for i, r in enumerate(d2.head(60).itertuples()):
            src = (paths.IMAGES / f"{r.fileId}.jpg") if show_masked else None
            if src is None or not src.exists():
                hits = list(paths.PHOTOS.glob(f"{r.fileId}.*"))
                src = hits[0] if hits else None
            has_box = isinstance(getattr(r, "predBoxes", None), str)
            with cols[i % 6]:
                if src and src.exists():
                    st.image(
                        with_boxes(str(src), r.predBoxes) if (show_box and has_box) else str(src),
                        width="stretch",
                    )
                conf = getattr(r, "predConf", None)
                conf_s = "" if conf is None or conf != conf else f"（{conf:.2f}）"
                sec = getattr(r, "predSecondary", None)
                extra = []
                if isinstance(sec, str):
                    extra.append("次要：" + "、".join(json.loads(sec)))
                if has_box:
                    extra.append(f"框{len(json.loads(r.predBoxes))}·最大{box_area(r.predBoxes):.0%}")
                is_v3 = str(getattr(r, "promptVersion", "")).startswith("v3")
                st.markdown(
                    f"<div class=pc6>"
                    f"<div>{badge('test' if r.fileId in test_ids else 'train')}"
                    f"<span class=m> {r.cls} · {r.reportDate}"
                    + ("<b> · v3</b>" if is_v3 else "")
                    + "</span></div>"
                    + verdict("系統判斷", r.local, r.cls)
                    # Gemini 顯示它**自己寫的字**，不是正規化後的類別——正規化是我們
                    # 為了比對硬折過去的，看板上要看的是它到底講了什麼。
                    # 綠/紅仍然依正規化後的結果上色，一致與否照樣一眼看得出來。
                    + gem_line(r.gem, r.gemNorm, r.cls, conf_s)
                    + (f"<div class=m>{' · '.join(extra)}</div>" if extra else "<div class=m>&nbsp;</div>")
                    + f"<div class=m>G判斷位置：{txt(getattr(r, 'predLocation', None))}</div>"
                    + f'<div class=m title="{txt(r.title)}">日報標題：{txt(r.title)}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

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
        import newclass
        from labels import pending_classes
        from labels import unclaimed as _unclaimed

        lab_now = Labeler.load()
        pend = pending_classes()
        st.markdown(f"**排隊中的類別 {len(pend)} 類** — 規則認得，張數還不夠")
        st.caption(
            f"門檻 `min_class_size: {lab_now.min_class_size}`，算的是 **train + test 全部**"
            "（切分之前就先砍了），不是 train。張數一過門檻**自動**進訓練，"
            "不用改任何程式——`金屬-欄杆鐵件` 就是這樣在 v18 自己冒出來的。"
            "別跟 `MIN_TRAIN` 搞混，那個是進場之後的切分保底。"
        )
        if len(pend):
            st.dataframe(
                pend.rename(columns={"cls": "類別", "photos": "現有", "need": "還差", "latest": "最近一張"}),
                width="stretch",
                hide_index=True,
            )
            st.progress(
                min(1.0, float(pend.photos.sum()) / float(pend.need.sum() + pend.photos.sum())),
                text=f"這 {len(pend)} 類合計 {int(pend.photos.sum())} 張，"
                f"全部上線還差 {int(pend.need.sum())} 張",
            )
        else:
            st.success("沒有類別卡在門檻下。")

        st.divider()
        # 這裡刻意**不用** labeled()。它走 apply(drop_small=True)，配上
        # drop_fallback: true，fallback 那些列在到達這裡之前就被濾掉了——
        # 這個面板從寫出來到 2026-08-25 為止一直顯示 0 張，等於沒人看得到新工種。
        other = _unclaimed()
        st.markdown(f"**沒有規則認領的 {len(other)} 張** — 下一批新工種在這裡")
        st.caption(
            "照片沒被刪，都在 `raw/photos/`，只是沒進訓練集（`drop_fallback: true`）。"
            "少的是一條規則，不是資料。"
        )
        if len(other):
            st.dataframe(other.title.value_counts().rename("張數"), width="stretch")
        with st.expander("候選新工種（含舊 pptx 那批一起算）", expanded=not len(other)):
            buf = []
            cand = newclass.report(src="all", log=lambda *a: buf.append(" ".join(map(str, a))))
            st.code("\n".join(buf) or "(無)")
            if cand:
                st.caption(
                    "貼進上面的規則框之前先看標題明細——詞是照「蓋得到幾張」挑的，"
                    "常常是跨詞界的碎片（`櫃安` = 櫥櫃+安裝，因為它連「廚櫃」的錯字一起收）。"
                )

    if tabs[6]:
        runs = (
            sorted(
                [p for p in paths.REPORTS_OUT.glob("*") if p.is_dir() and "qms" not in p.name], reverse=True
            )
            if paths.REPORTS_OUT.exists()
            else []
        )
        report_view(runs, "pms_run")
