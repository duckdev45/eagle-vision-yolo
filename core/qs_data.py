# core/qs_data.py
"""公司 ISO（QS）品質標準的資料層（服務層實作，src/qsdata.py 的 CLI 相容層指向這裡）。

`reference/iso/raw/*.tsv` → 可查詢的結構。所有未來的訓練菜單都要經過這裡：

    工序階段標籤   phases(docNo)        ← OPTIONAL 節點
    缺失分類表     required(docNo)      ← REQUIRED 項，主鍵 f"{docNo}-{itemNo}"
    請款照片靶     billing_items()      ← 明文「須拍照存證」
    合約相依警示   contract_items()     ← 判定基準不在 QS 裡，RAG 單灌 QS 答不出來
    工具分派       classify(item)       ← A~E，決定該用 VLM / OCR / 規則 / 儀器報告

    make qs                          # 統計總表（src/qsdata.py 的 CLI）
    uv run src/qsdata.py --self-check
    uv run src/qsdata.py --emit-phases   # 寫出 reference/iso/phases.yaml

刻意不依賴 pandas/torch：這層要能在任何環境跑（含只裝基本相依的 CI）。
"""
from __future__ import annotations

import sys as _sys
import os as _os

for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import re  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import paths  # noqa: E402

RAW_DIR = paths.ROOT / "reference" / "iso" / "raw"
PHASES_OUT = paths.ROOT / "reference" / "iso" / "phases.yaml"


# ── 工具分派規則 ────────────────────────────────────────────────────────
# 一個檢查項「憑什麼判定」決定該用哪種工具。順序即優先權：先命中者勝。
# ⚠ 這是啟發式規則，不是人工標註。跑 --self-check 看 UNSURE 比例，
#   佔比過高就別拿它當統計依據，要人工複核。
RULES: list[tuple[str, str, str]] = [
    # (類型, 說明, regex)
    ("E", "儀器試驗",
     r"試驗|抗壓|氯離子|坍度|扭力|水壓|蓄水|非破壞|試水|含水率|強度.*試|試.*強度"),
    ("C", "文件憑證",
     r"證明|證照|報告|圖說|圖審|樣品|簽認|計劃書|計畫書|名冊|技術資料|合約|採發"
     r"|規範|出廠|CNS|認可|核准|審核|存查|備查|提送"),
    ("D", "時序流程",
     r"每月|每半月|每日|天後|小時|分鐘|日內|週內|一週|之前|先行|完成後|時機"
     r"|同時|再行|方可|後方|次數|定期"),
    ("B", "量測數值",
     r"\d+\s*(?:mm|cm|m²|m2|kg|µ|%|度|公分|公尺|米|倍|分|寸)"
     r"|誤差|間距|厚度|坡度|垂直度|水平|高程|尺寸|長度|直徑|深度|寬度|重量"
     r"|不超過|不得小於|不得大於|以上|以下|至少"),
]
UNSURE = "A"  # 落不到上面任何一條 → 純視覺（最可能是 VLM 能吃的）

BILLING_RE = re.compile(r"請款|存證|憑證")
PENALTY_RE = re.compile(r"扣款|罰款|止付|保固")
CONTRACT_RE = re.compile(r"合約|採發|圖說|圖審|樣品|簽認|工作約定")


@dataclass
class Item:
    doc_no: str
    item_no: str
    status: str          # "O" = 階段節點 / "R" = 檢查項
    name: str

    @property
    def key(self) -> str:
        """缺失代碼主鍵。人看得懂，可直接當 RAG 的 chunk id。"""
        return f"{self.doc_no}-{self.item_no}"

    @property
    def depth(self) -> int:
        return self.item_no.count(".") + 1

    @property
    def kind(self) -> str:
        """A~E 工具分派。"""
        for tag, _, pat in RULES:
            if re.search(pat, self.name):
                return tag
        return UNSURE

    @property
    def is_billing(self) -> bool:
        return bool(BILLING_RE.search(self.name))

    @property
    def is_penalty(self) -> bool:
        return bool(PENALTY_RE.search(self.name))

    @property
    def is_contract(self) -> bool:
        return bool(CONTRACT_RE.search(self.name))


@dataclass
class Doc:
    doc_no: str
    iso_info_id: str
    name: str
    items: list[Item] = field(default_factory=list)

    @property
    def phases(self) -> list[Item]:
        """深度 1 的 OPTIONAL 節點 = 現成的工序階段。"""
        return [i for i in self.items if i.status == "O" and i.depth == 1]

    @property
    def required(self) -> list[Item]:
        return [i for i in self.items if i.status == "R"]

    @property
    def orphan_depth1_required(self) -> list[Item]:
        """深度 1 卻是 REQUIRED 的葉節點。

        QS0501 的「1 清潔」就是這種：語意上是第一個工序階段，但格式上是檢查項。
        **不自動當成階段**——列出來讓人決定，免得靜默猜錯。
        """
        has_child = {i.item_no.rsplit(".", 1)[0] for i in self.items if i.depth > 1}
        return [i for i in self.items
                if i.status == "R" and i.depth == 1 and i.item_no not in has_child]

    @property
    def shape(self) -> str:
        """工序型態：A 有時序階段 / C 完全扁平。B（部位分支）需人工判定，這裡只分 A/C。"""
        return "A" if self.phases else "C"


def load(raw_dir: Path = RAW_DIR) -> dict[str, Doc]:
    """讀所有 TSV。回傳 {docNo: Doc}。"""
    files = sorted(raw_dir.glob("*.tsv"))
    if not files:
        raise SystemExit(f"{raw_dir} 沒有 tsv——先落地 QS 資料")
    docs: dict[str, Doc] = {}
    cur: Doc | None = None
    for fp in files:
        for ln, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            if line.startswith("#DOC"):
                p = line.split("\t")
                if len(p) < 4:
                    raise ValueError(f"{fp.name}:{ln} #DOC 欄位不足：{line!r}")
                cur = Doc(doc_no=p[1].strip(), iso_info_id=p[2].strip(), name=p[3].strip())
                if cur.doc_no in docs:
                    prev = docs[cur.doc_no]
                    raise ValueError(
                        f"{fp.name}:{ln} docNo 重複：{cur.doc_no}\n"
                        f"  既有：{prev.name}（{prev.iso_info_id}）\n"
                        f"  新增：{cur.name}（{cur.iso_info_id}）\n"
                        f"  → 兩份不同標準搶同一個編號。dict 會靜默覆蓋，所以這裡直接擋下。\n"
                        f"  若確定是兩份不同文件，其中一份要用消歧碼（如 {cur.doc_no}B），"
                        f"並在 catalog.yaml 的 conflicts 區塊記錄原因。")
                docs[cur.doc_no] = cur
                continue
            if line.startswith("#"):        # 註解
                continue
            p = line.split("\t")
            if len(p) < 3:
                raise ValueError(f"{fp.name}:{ln} 欄位不足：{line!r}")
            if cur is None:
                raise ValueError(f"{fp.name}:{ln} 檢查項出現在任何 #DOC 之前")
            st = p[1].strip()
            if st not in ("O", "R"):
                raise ValueError(f"{fp.name}:{ln} status 只能是 O/R，得到 {st!r}")
            cur.items.append(Item(cur.doc_no, p[0].strip(), st, p[2].strip()))
    return docs


# ── 跨文件查詢 ──────────────────────────────────────────────────────────
def all_items(docs=None) -> list[Item]:
    docs = docs or load()
    return [i for d in docs.values() for i in d.items]


def billing_items(docs=None) -> list[Item]:
    """明文要求照片作為請款憑證 → vision 模型最高價值的靶。"""
    return [i for i in all_items(docs) if i.status == "R" and i.is_billing]


def penalty_items(docs=None) -> list[Item]:
    return [i for i in all_items(docs) if i.status == "R" and i.is_penalty]


def contract_items(docs=None) -> list[Item]:
    """判定基準指向合約而非 QS → RAG 只灌 QS 會答不出來的那批。"""
    return [i for i in all_items(docs) if i.status == "R" and i.is_contract]


def kind_stats(docs=None) -> Counter:
    return Counter(i.kind for i in all_items(docs) if i.status == "R")


def emit_phases(docs=None, out=PHASES_OUT) -> str:
    """把工序階段寫成 yaml，供 labels/review 那一側引用。

    只寫**確定的**（OPTIONAL 節點）。扁平的標 needs_inference: true，
    不在這裡塞我推論的階段——推論要人核過才能進資料層。
    """
    docs = docs or load()
    lines = [
        "# 工序階段（自動生成，勿手改 —— `uv run src/qsdata.py --emit-phases`）",
        "#",
        "# phases 只收 checklist 裡 status=OPTIONAL 的深度 1 節點，即標準自己認定的階段。",
        "# needs_inference: true 的那些是完全扁平的標準（O 節點 0 個），階段必須另外推論，",
        "# 且**推論結果不寫在這裡**——要先經人工核對再進 review.csv 的 work_phase 欄。",
        "#",
        "# orphan_depth1_required：深度 1 但格式是 REQUIRED 的葉節點。語意上常常是階段",
        "# （例 QS0501「1 清潔」），但不自動採用，列出來讓人決定。",
        "",
        "docs:",
    ]
    for dn in sorted(docs):
        d = docs[dn]
        lines.append(f"  {dn}:")
        lines.append(f"    name: {d.name}")
        lines.append(f"    shape: {d.shape}")
        if d.phases:
            lines.append("    phases:")
            for i, ph in enumerate(d.phases, 1):
                nm = ph.name.rstrip(":：").replace('"', "'")
                lines.append(f'      - {{ seq: {i}, itemNo: "{ph.item_no}", name: "{nm}" }}')
        else:
            lines.append("    needs_inference: true")
        orph = d.orphan_depth1_required
        if orph and d.phases:
            lines.append("    orphan_depth1_required:")
            for o in orph:
                nm = o.name.replace('"', "'")[:40]
                lines.append(f'      - {{ itemNo: "{o.item_no}", name: "{nm}" }}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)


def report(docs=None, log=print) -> None:
    docs = docs or load()
    items = all_items(docs)
    req = [i for i in items if i.status == "R"]
    log(f"標準 {len(docs)} 份 / 節點 {len(items)}（REQUIRED {len(req)} · OPTIONAL "
        f"{len(items) - len(req)}）")

    log("\n── 工序型態 ──")
    shapes = Counter(d.shape for d in docs.values())
    log(f"  A 有現成階段：{shapes['A']} 份")
    log(f"  C 完全扁平　：{shapes['C']} 份（階段需推論，勿當成沒有工序）")

    log("\n── 工具分派（A~E，啟發式）──")
    ks = kind_stats(docs)
    names = {t: n for t, n, _ in RULES}
    names[UNSURE] = "純視覺"
    for tag in "ABCDE":
        n = ks.get(tag, 0)
        bar = "█" * round(n / max(ks.values()) * 28) if ks else ""
        log(f"  {tag} {names.get(tag,''):<6} {n:>4} ({n/len(req)*100:4.1f}%) {bar}")
    vision = ks.get("A", 0) + ks.get("B", 0)
    log(f"  → vision 守備範圍 A+B = {vision}/{len(req)} ({vision/len(req)*100:.1f}%)")

    log("\n── 請款照片靶（明文須拍照存證）──")
    for i in billing_items(docs):
        log(f"  {i.key:<12} {i.name[:56]}")

    log("\n── 罰則／付款閘門 ──")
    for i in penalty_items(docs):
        log(f"  {i.key:<12} {i.name[:56]}")

    log(f"\n── 合約相依（判定基準不在 QS 裡）：{len(contract_items(docs))} 項 ──")
    log("  單灌 QS 進 RAG 這批答不出來，需併合約側文件")

    log("\n── 各份摘要 ──")
    for dn in sorted(docs):
        d = docs[dn]
        ph = "、".join(p.name.rstrip(":：") for p in d.phases) if d.phases else "—"
        log(f"  {dn:<10} {d.shape} R={len(d.required):<3} {d.name:<18} {ph[:44]}")


def self_check() -> int:
    """跑得起來就等於驗證了解析、階層、分派三件事。

    回傳值只反映「硬性」失敗（bad），不含 warnings（如未解決的 docNo 衝突提醒）——
    後者要能被看見，但不該讓 CI／make test 卡住，因為它是已知狀態，不是新壞掉的東西。
    """
    bad = 0
    warn = 0

    def ck(cond, msg):
        nonlocal bad
        if not cond:
            bad += 1
            print(f"  ✗ {msg}")
        else:
            print(f"  ✓ {msg}")

    def warn_if(cond, msg):
        """cond 為 True 時印警告，不計入失敗數（用於「已知、待辦、不阻斷」的狀態）。"""
        nonlocal warn
        if cond:
            warn += 1
            print(f"  ⚠ {msg}")

    docs = load()
    ck(len(docs) >= 37, f"至少 37 份標準（實得 {len(docs)}）")

    # 階層深度：QS0203 連續壁有三層（2.2.2.1）
    d = docs.get("QS0203")
    ck(d is not None and max(i.depth for i in d.items) >= 4,
       "QS0203 解析出 4 層階層（2.2.2.1）")

    # QS0501 油漆：3 個 O 節點（補土/整平磨平/底漆面漆）+ 1 個孤兒（清潔）
    d = docs["QS0501"]
    ck(len(d.phases) == 3, f"QS0501 有 3 個 OPTIONAL 階段（實得 {len(d.phases)}）")
    ck(any("清潔" in o.name for o in d.orphan_depth1_required),
       "QS0501「1 清潔」被列為 orphan 而非靜默當成階段")

    # QS0402 泥作粉刷：完全扁平，32 項
    d = docs["QS0402"]
    ck(d.shape == "C" and len(d.required) == 32,
       f"QS0402 扁平且 32 項（shape={d.shape} R={len(d.required)}）")

    # 主鍵格式。注意 required 的順序是 API 回傳序（非排序），所以用查找不用索引
    it = next(i for i in docs["QS0404"].required if "10~15mm" in i.name)
    ck(it.key == "QS0404-4", f"主鍵格式 QS0404-4（實得 {it.key}）")

    # 請款項：明文「須拍照存證，做為請款之憑證」。全庫應為 8 項。
    b = billing_items(docs)
    ck(len(b) >= 5, f"請款照片項 ≥5（實得 {len(b)}）")
    ck({i.doc_no for i in b} <= {"QS0301", "QS0302", "QS0303", "QS0606",
                                 "QS0701", "QS0907B"},
       "請款項落在結構工程／門窗／防水／機電")

    # 分派：厚度項該是 B，出廠證明該是 C
    ck(next(i for i in docs["QS0404"].required if "10~15mm" in i.name).kind == "B",
       "「粘貼厚度10~15mm」分派到 B 量測")
    ck(next(i for i in docs["QS0501"].required if "出廠証明" in i.name).kind == "C",
       "「材料進場是否附出廠証明」分派到 C 文件")

    # 罰則
    ck(len(penalty_items(docs)) >= 2, f"罰則項 ≥2（實得 {len(penalty_items(docs))}）")

    # QS0104 施工電梯：2026-08-27 對過公司系統 API 全量 dump，22/22 項逐字一致。
    # 這裡只驗證項數與階段數，不是重跑那次比對——真正的驗證是那次人工核對本身。
    d = docs.get("QS0104")
    ck(d is not None and len(d.required) == 14 and len(d.phases) == 8,
       f"QS0104 施工電梯 14R+8O（實得 R={len(d.required) if d else '?'} "
       f"O={len(d.phases) if d else '?'}）—— 已比對公司 API 全量 dump 逐字一致")

    # ── catalog.yaml ↔ raw/ 一致性 ──────────────────────────────────
    # 2026-08-28 的教訓：爬蟲重建 raw/ 時把「停車場排風」寫成 QS0907，
    # 覆蓋了 catalog 裡原本屬於「電梯工程標準」的位置。因為 load() 回傳 dict，
    # 後寫的蓋前面的，**整件事靜默發生**——目錄與內容各說各話卻沒人報錯。
    # 這三條檢查就是為了讓同類問題下次自己叫出來。
    cat_path = paths.ROOT / "reference" / "iso" / "catalog.yaml"
    if cat_path.exists():
        import yaml
        cat = yaml.safe_load(cat_path.read_text(encoding="utf-8"))
        entries = cat.get("docs", [])
        cat_ids = {}
        dupes = []
        for e in entries:
            dn = e["docNo"]
            if dn in cat_ids:
                dupes.append(dn)
            cat_ids[dn] = e

        ck(not dupes,
           f"catalog.yaml 無重複 docNo（重複：{sorted(set(dupes))}）"
           if dupes else "catalog.yaml 無重複 docNo")

        only_cat = sorted(set(cat_ids) - set(docs))
        only_raw = sorted(set(docs) - set(cat_ids))
        ck(not only_cat and not only_raw,
           f"catalog ↔ raw 份數對齊（僅目錄有：{only_cat[:5]}；僅內容有：{only_raw[:5]}）"
           if (only_cat or only_raw) else "catalog ↔ raw 份數對齊")

        # isoInfoId 是身分證：docNo 可能被人記錯，UUID 不會
        mismatch = [(k, docs[k].iso_info_id[:8], cat_ids[k]["id"][:8])
                    for k in docs if k in cat_ids
                    and docs[k].iso_info_id != cat_ids[k]["id"]]
        ck(not mismatch,
           f"isoInfoId 全部一致（不符 {len(mismatch)} 份：{mismatch[:3]}）"
           if mismatch else "isoInfoId 全部一致")

    # QS0907 編號歸屬（2026-08-28 定案，見 catalog.yaml 的 numbering 區塊）：
    #   QS0907  = 電梯工程標準（PDF 原件，7 查驗項）
    #   QS0907B = 停車場排風（本專案固定編號，非待確認狀態）
    # 這兩條是正向斷言不是警告——編號已定，任何一邊被改動都該讓測試失敗。
    d = docs.get("QS0907")
    ck(d is not None and len(d.required) == 7 and "電梯" in d.name,
       f"QS0907 = 電梯工程標準 7 項（實得 {d.name if d else '缺'} "
       f"{len(d.required) if d else 0} 項）")

    d2 = docs.get("QS0907B")
    ck(d2 is not None and "排風" in d2.name,
       f"QS0907B = 停車場排風（實得 {d2.name if d2 else '缺'}）"
       "——本專案固定編號，勿改回 QS0907")

    print(f"\n{'✓ 全部通過' if not bad else f'✗ {bad} 項失敗'}"
          + (f"（另有 {warn} 項待辦警告，不計入失敗）" if warn else ""))
    return bad


def main(argv: list[str] | None = None) -> int:
    """CLI 進入點（src/qsdata.py 相容層會直接呼叫）。"""
    import argparse

    ap = argparse.ArgumentParser(prog="qsdata")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--emit-phases", action="store_true")
    a = ap.parse_args(argv)
    if a.self_check:
        return 1 if self_check() else 0
    docs = load()
    if a.emit_phases:
        print(f"寫出 → {emit_phases(docs)}")
    else:
        report(docs)
    return 0
