"""PMS 照片工種分類、新類候選及看圖審閱包的操作介面。"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import paths
import split as split_mod
from core import pms_exchange as exchange
from core import pms_review as review
from core import pms_store as store

from .pms_ai import ai_panel, suggestion_card


@st.cache_data(show_spinner=False)
def _model(stamp: tuple) -> dict:
    return review.local_model()


def _snapshot():
    name = split_mod.current()
    files = [
        paths.SPLITS / f"{name}.json",
        paths.MODELS / f"probe-siglip-{name}.pkl",
        paths.FEATURES / "siglip.npz",
    ]
    stamp = tuple(
        (str(p), p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (str(p), 0, 0) for p in files
    )
    return review.snapshot(_model(stamp))


def _refresh(message: str = "") -> None:
    for key in list(st.session_state):
        if key.startswith("pms_revision:"):
            del st.session_state[key]
    st.session_state["pms_flash"] = message
    st.cache_data.clear()
    st.rerun()


def _image(fid: str) -> None:
    try:
        st.image(review.photo_bytes(fid), width="stretch")
    except (OSError, ValueError) as exc:
        st.warning(f"圖片暫時無法讀取：{exc}")


def _photo_name(row: dict) -> str:
    return f"{row['reportDate']} · {row['title'] or '無標題'} · {row['fileId'][:8]}"


def _packet_panel(view: pd.DataFrame) -> None:
    with st.expander("看圖審閱包與 AI 建議"):
        st.caption("選取照片建立本機 ZIP，包含圖片、分類表、日報參考資訊及回覆範本。匯入回覆後仍需逐張確認。")
        options = view.head(200).fileId.tolist()
        lookup = {r["fileId"]: _photo_name(r) for r in view.head(200).to_dict("records")}
        ids = st.multiselect(
            "本批照片（最多 50 張）", options, format_func=lambda fid: lookup[fid], key="pms_packet_ids"
        )
        if st.button("建立看圖審閱包", disabled=not ids, key="pms_export"):
            try:
                key, blob = exchange.export_packet(ids)
                st.session_state["pms_packet"] = (key, blob)
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        packet = st.session_state.get("pms_packet")
        if packet:
            st.download_button(
                "取得審閱包 ZIP",
                packet[1],
                file_name=f"pms-review-{packet[0]}.zip",
                mime="application/zip",
                key="pms_download",
            )
        uploaded = st.file_uploader("匯入審閱結果 JSON", type=["json"], key="pms_response_file")
        pasted = st.text_area("或貼上審閱結果 JSON", key="pms_response_text", height=100)
        if st.button("驗證並匯入建議", disabled=uploaded is None and not pasted.strip(), key="pms_import"):
            try:
                data = uploaded.getvalue() if uploaded is not None else pasted.encode()
                if len(data) > 2_000_000:
                    raise ValueError("回覆超過 2 MB，請分批匯入。")
                count = exchange.import_suggestions(json.loads(data))
                _refresh(f"已匯入 {count} 筆新建議；正式照片標籤尚未變更。")
            except (ValueError, OSError) as exc:
                st.error(str(exc))


def _candidate_form(ids: list[str], reviewer: str, key: str, proposal: dict | None = None) -> None:
    proposal = proposal or {}
    with st.form(f"pms_candidate_{key}"):
        label = st.text_input("候選名稱（工種-施作內容）", value=proposal.get("label", ""))
        definition = st.text_area("共同可見特徵", value=proposal.get("candidateDefinition", ""))
        excludes = st.text_input("與哪些既有類別不同？")
        basis = st.text_input("分類樹或 QS 依據（提出時可稍後補）")
        submit = st.form_submit_button("提出新工種候選", disabled=not ids)
    if submit:
        try:
            review.propose_candidate(
                label,
                ids,
                reviewer=reviewer,
                definition=definition,
                basis=basis,
                excludes=excludes,
                proposal_id=proposal.get("proposalId", ""),
            )
            _refresh(f"已提出 {label}，候選照片暫停進入訓練，等待核准與逐張確認。")
        except (ValueError, OSError) as exc:
            st.error(str(exc))


def _photo_card(row: dict, reviewer: str, known: list[str]) -> None:
    fid = row["fileId"]
    st.markdown(f"**{row['title'] or '無標題照片'}**")
    st.caption(f"{row['reportDate']} · {row['constrName'] or row['constrId']} · {fid}")
    _image(fid)
    st.caption(f"{review.STATES[row['reviewState']]} · {review.ROUTES[row['route']]} · {row['part'].upper()}")
    st.write(f"標題規則：{row['ruleClass'] or '無分類'}")
    st.write(f"人工確認：{row['humanClass'] or '尚未確認'}")
    model = row["modelClass"] or "尚無預測"
    if row["modelConfidence"] is not None and pd.notna(row["modelConfidence"]):
        model += f"（分數 {row['modelConfidence']:.2f}）"
    st.write(f"照片模型：{model}")
    if row["reviewReason"]:
        st.caption(row["reviewReason"])
    with st.expander("日報參考資訊"):
        st.write({"標題": row["title"], "查驗重點": row["chipsOn"], "前端工種": row["specKey"]})
    rev_key = f"pms_revision:{fid}"
    expected = st.session_state.setdefault(rev_key, review.revision(fid))
    proposal = row["suggestion"]
    if suggestion_card(proposal, reviewer, key=f"pms_accept_{fid}", expected_revision=expected):
        _refresh("已保存照片分類與 AI 建議來源。")
    with st.form(f"pms_decide_{fid}"):
        action = st.selectbox(
            "判斷",
            ["classified", "uncertain", "excluded"],
            format_func=lambda value: {
                "classified": "確認工種",
                "uncertain": "資訊不足",
                "excluded": "非本輪工種語料",
            }[value],
        )
        label = st.selectbox(
            "照片實際工種", known, index=None, placeholder="請看圖後選擇", key=f"pms_class_{fid}"
        )
        reason = st.text_input("判斷依據／暫緩原因", key=f"pms_reason_{fid}")
        submitted = st.form_submit_button("儲存判斷")
    if submitted:
        try:
            review.decide(
                fid, action, reviewer=reviewer, label=label or "", reason=reason, expected_revision=expected
            )
            _refresh("已保存判斷；下次訓練會使用更新後的標籤與排除狀態。")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    with st.expander("這張可能是新工種"):
        proposed = (
            proposal if proposal.get("decision") == "new_candidate" and row["suggestionCurrent"] else None
        )
        _candidate_form([fid], reviewer, fid, proposed)


def _photos_page(df: pd.DataFrame, reviewer: str) -> None:
    a, b = st.columns(2)
    kind = a.selectbox(
        "查看照片",
        [
            "待複核",
            "全部",
            "尚無分類",
            "模型尚未涵蓋",
            *[v for k, v in review.STATES.items() if k != "pending"],
        ],
        key="pms_photo_filter",
    )
    site = b.selectbox("案場", ["全部", *sorted(set(df.constrName) - {""})], key="pms_photo_site")
    view = df
    if kind == "待複核":
        view = view[view.needsReview.astype(bool)]
    elif kind == "尚無分類":
        view = view[view.route == "unknown"]
    elif kind == "模型尚未涵蓋":
        view = view[view.route == "known_untrained"]
    elif kind != "全部":
        state = next(key for key, label in review.STATES.items() if label == kind)
        view = view[view.reviewState == state]
    if site != "全部":
        view = view[view.constrName == site]
    query = st.text_input("搜尋標題或照片識別碼", key="pms_photo_search")
    if query:
        view = view[
            view.title.str.contains(query, regex=False) | view.fileId.str.contains(query, regex=False)
        ]
    view = view.sort_values(["reportDate", "fileId"], ascending=[False, True])
    st.caption(f"符合條件 {len(view)} 張。TRAIN 表示模型訓練用過；TEST 是該版測試照；UNSEEN 尚未進該版切分。")
    if ai_panel(view, key="pms_photo_ai") is not None:
        _refresh("AI 分類已處理；結果與失敗明細可在 AI 自動分類面板查看。")
    _packet_panel(view)
    if view.empty:
        st.info("這個篩選條件下沒有照片，可改選全部或其他狀態。")
        return
    pages = max(1, (len(view) + 3) // 4)
    if st.session_state.get("pms_photo_page", 1) > pages:
        st.session_state["pms_photo_page"] = 1
    page = st.number_input("頁數", min_value=1, max_value=pages, step=1, key="pms_photo_page")
    rows = view.iloc[(page - 1) * 4 : page * 4].to_dict("records")
    for offset in range(0, len(rows), 2):
        for col, row in zip(st.columns(2), rows[offset : offset + 2]):
            with col, st.container(border=True):
                _photo_card(row, reviewer, sorted(review.catalog()))


def _candidates_page(df: pd.DataFrame, model: dict, reviewer: str) -> None:
    st.caption("先檢查完整分類表；既有類別缺樣本或模型尚未涵蓋時，直接分類並累積資料即可。")
    with st.expander("完整分類表與待訓練類別"):
        st.dataframe(review.class_inventory(df, model), hide_index=True, width="stretch")
    groups = review.discover(df)
    chosen = []
    if groups:
        st.markdown("**未知照片的文字群組提示**")
        idx = st.selectbox(
            "選擇一組查看",
            range(len(groups)),
            format_func=lambda i: (
                f"{groups[i]['term']} · {groups[i]['photos']} 張 · {groups[i]['sites']} 案場 · {groups[i]['days']} 日期"
            ),
        )
        group = groups[idx]
        chosen = group["fileIds"]
        st.write("、".join(group["titles"]))
        for col, fid in zip(st.columns(4), chosen[:4]):
            with col:
                _image(fid)
        st.caption("這些群組只來自標題用詞；先看圖片，再判斷是否需要新類。")
    else:
        st.info("尚無重複出現的未知標題群組；仍可從照片頁或下方手動提出候選。")
    with st.expander("建立或補充候選"):
        lookup = {r["fileId"]: _photo_name(r) for r in df.to_dict("records")}
        ids = st.multiselect(
            "候選代表照片",
            df.fileId.tolist(),
            default=chosen,
            format_func=lambda fid: lookup[fid],
            key="pms_candidate_ids",
        )
        _candidate_form(ids, reviewer, "group")
    saved = store.latest("candidate")
    st.markdown("**已提出的候選**")
    if not saved:
        st.info("尚未建立候選。")
        return
    statuses = {"proposed": "待核准", "approved": "已核准定義", "rejected": "不開新類"}
    key = st.selectbox(
        "選擇候選",
        list(saved),
        format_func=lambda value: f"{saved[value]['label']} · {statuses[saved[value]['status']]}",
    )
    candidate = saved[key]
    members = df[df.fileId.isin(candidate["fileIds"])]
    st.caption(
        f"目前有效代表照 {len(members)} 張 · {members.constrId.nunique()} 案場 · {members.reportDate.nunique()} 日期"
    )
    st.dataframe(members[["fileId", "title", "humanClass", "reviewState"]], hide_index=True, width="stretch")
    st.write(candidate["definition"])
    if candidate["status"] == "proposed":
        seq = st.session_state.setdefault(f"pms_revision:candidate:{key}", candidate["_seq"])
        with st.form(f"pms_resolve_{key}"):
            definition = st.text_area("新類定義", value=candidate["definition"])
            excludes = st.text_area("不包含的情況／容易混淆的類別", value=candidate["excludes"])
            basis = st.text_input("分類樹或 QS 依據", value=candidate["basis"])
            approve = st.form_submit_button("核准此新類")
            reject = st.form_submit_button("不開新類，保留紀錄")
        if approve or reject:
            try:
                review.resolve_candidate(
                    key,
                    approve=approve,
                    reviewer=reviewer,
                    definition=definition,
                    excludes=excludes,
                    basis=basis,
                    expected_seq=seq,
                )
                _refresh("已保存候選決定。請回照片工種頁，逐張確認工種或記錄資訊不足。")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    else:
        st.info("候選定義已結案。請在照片工種頁逐張確認；核准定義不會自動替代表照片貼標籤。")


def workbench(section: str = "photos") -> None:
    titles = {"photos": "照片工種分類", "candidates": "新工種候選", "overview": "PMS 工種資料總覽"}
    st.subheader(titles[section])
    if st.session_state.get("pms_flash"):
        st.success(st.session_state.pop("pms_flash"))
    try:
        df, model = _snapshot()
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"無法載入 PMS 工種資料：{exc}")
        return
    if model["warning"]:
        st.info(model["warning"])
    a, b, c, d = st.columns(4)
    a.metric("施作項目照片", len(df))
    b.metric("有訊號待複核", int(df.needsReview.sum()))
    c.metric("人工確認", int((df.reviewState == "classified").sum()))
    d.metric("尚無分類", int((df.route == "unknown").sum()))
    st.caption(
        f"目前模型 {model['name']} · 完整分類表 {len(review.catalog())} 類 · 模型涵蓋 {len(model['classes'])} 類"
    )
    st.caption("本頁只收 PMS 施作項目（WORK_ITEM）；出工紀錄照片保留於原始資料，不列入工種分類與新類候選。")
    if st.button("重新整理資料", key="pms_refresh"):
        _refresh()
    if section == "overview":
        st.dataframe(review.class_inventory(df, model), hide_index=True, width="stretch")
        st.caption("有效標籤包含尚未複核的標題規則結果；樣本達門檻僅表示可進下輪訓練，模型切換需另行確認。")
        with st.expander("已裁照片與純標題規則的一致度"):
            st.json(review.calibration())
        return
    if df.empty:
        st.info("尚無有效 PMS 照片，請先到同步頁抓取日報。")
        return
    reviewer = st.text_input("確認者", key="pms_reviewer", placeholder="保存判斷時需要填寫")
    if section == "photos":
        _photos_page(df, reviewer)
    else:
        _candidates_page(df, model, reviewer)
