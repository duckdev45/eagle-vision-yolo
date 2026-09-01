"""規範庫頁：QS 公司標準 + 合約工作約定。

這一頁不碰照片也不碰模型——它是**判定基準**的來源。
先前 qsdata/contractdata 兩層寫好了卻沒有任何界面用得到，
這頁把它們接上來，讓複核時查得到「這一項的標準數值是多少」。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import contractdata
import qsdata


def standards_page(tabs) -> None:
    if tabs[0]:  # ① QS 標準
        docs = qsdata.load()
        items = [i for i in qsdata.all_items(docs) if i.status == "R"]
        c = st.columns(4)
        c[0].metric("標準份數", len(docs))
        c[1].metric("檢查項", len(items))
        vision = sum(1 for i in items if i.kind in ("A", "B"))
        c[2].metric(
            "vision 可判",
            f"{vision / len(items) * 100:.0f}%",
            help="A 純視覺 + B 量測。C 文件/D 時序/E 儀器走別的路",
        )
        c[3].metric("請款靶", len(qsdata.billing_items(docs)), help="明文「須拍照存證，做為請款之憑證」")

        pick = st.selectbox("挑一份標準", sorted(docs), format_func=lambda d: f"{d} {docs[d].name}")
        d = docs[pick]
        if d.phases:
            st.caption("工序階段（標準自己定義的 OPTIONAL 節點）")
            st.markdown(" → ".join(f"**{p.name.rstrip('：:')}**" for p in d.phases))
        else:
            st.caption("⚠ 這份完全扁平（無 OPTIONAL 節點），工序需另行推論")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "項次": i.item_no,
                        "類型": i.kind,
                        "檢查項": i.name,
                        "請款": "◆" if i.is_billing else "",
                        "罰則": "◆" if i.is_penalty else "",
                        "看合約": "◆" if i.is_contract else "",
                    }
                    for i in d.required
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    if tabs[1]:  # ② 合約工作約定
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
            st.warning("這份是**物料明細**（材料供應），不是施工工作約定——回答不了施工類的 QS 檢查項")
        st.dataframe(
            pd.DataFrame(
                [{"條號": x.no, "分類": x.kind, "來源": x.sheet, "條文": x.text} for x in d.clauses]
            ),
            width="stretch",
            hide_index=True,
        )

    if tabs[2]:  # ③ 衝突比對
        st.caption(
            "同工種內，QS 與合約都給了數值的項目。"
            "**這是候選不是結論**——程式只能證明兩邊在談同一主題且都有數字。"
        )
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
            "已人工確認的三處衝突見 `reference/contract/CONTRACT_VS_QS.md`。"
        )

    if tabs[3]:  # ④ 工種介面
        st.caption(
            "QS 按**單一工種**編排，交界被切碎（實測 28 個介面項散在 9 份標準裡）。"
            "合約是逐工種簽的，同一個交界會在兩份合約各出現一次——"
            "**兩邊都拿到才是完整的**。"
        )
        for it in contractdata.load_mappings().get("INTERFACES", []):
            cs = contractdata.interface_clauses(it["name"])
            with st.expander(f"{it['name']}（{len(cs)} 條）"):
                st.info(it["note"])
                for c in cs:
                    st.markdown(f"**{c.trade}** `{c.no}`　{c.text}")

    if tabs[4]:  # ⑤ 合約相依缺口
        docs = qsdata.load()
        ci = qsdata.contract_items(docs)
        answered = contractdata.load_mappings().get("QS_ANSWERS", {})
        st.metric(
            "合約相依項",
            f"{len(answered)} / {len(ci)}",
            help="判定基準指向合約而非 QS 的檢查項，目前已對應到合約條款的比例",
        )
        st.caption("這些項目**單靠 QS 答不出來**。RAG 若只灌 QS，檢索會命中但回答不了「合不合格」。")
        rows = []
        for i in ci:
            got = answered.get(i.key, [])
            rows.append(
                {
                    "狀態": "✓" if got else "—",
                    "QS 項": i.key,
                    "檢查項": i.name,
                    "合約條款": "、".join(got) if got else "尚未取得",
                }
            )
        st.dataframe(
            pd.DataFrame(rows).sort_values("狀態", ascending=False), width="stretch", hide_index=True
        )
