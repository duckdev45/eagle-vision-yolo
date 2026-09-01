# core/contractdata.py
"""合約側工作約定的資料層（服務層實作，src/contractdata.py 的 CLI 相容層指向這裡）。

與 `core/qs_data.py` 對稱：QS 是**通用標準**（跨案適用），工作約定是**逐案合約**
（同一工種在不同案、不同版本，數值會不一樣）。兩者不可混在同一個 index，
否則會拿 A 案的條款去答 B 案的問題。

    工作約定條款   clauses(trade)         ← 逐條，帶 project/vendor/date
    付款節點       payment_terms()        ← 請款比例、保留款（L5 的輸入）
    罰則           penalties()            ← 扣款倍數、罰金（L5 的閘門）
    驗收數值       tolerances()           ← 與 QS 衝突時，合約優先（見 reference/contract/CONTRACT_VS_QS.md）
    QS 交叉引用    answers_qs(qs_key)     ← 回答 qsdata.contract_items() 的合約相依項

    make contract                            # 統計總表（src/contractdata.py 的 CLI）
    uv run src/contractdata.py --self-check
    uv run src/contractdata.py --conflicts   # QS vs 合約的數值衝突候選

⚠ 這層的資料**逐案有效**。任何查詢都必須帶 project 或明確接受「跨案通用」的風險。
"""
from __future__ import annotations

import sys as _sys
import os as _os

for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import re  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import paths  # noqa: E402

RAW_DIR = paths.ROOT / "reference" / "contract" / "raw"
MAPPINGS = paths.ROOT / "reference" / "contract" / "mappings.yaml"

# 條款分類（決定該進哪個子系統）——順序即優先權
KINDS: list[tuple[str, str, str]] = [
    ("payment", "付款節點", r"請款|付款|計價|估驗|保留款|即期票|月結|付\s*\d+|%|％|﹪"),
    ("penalty", "罰則扣款", r"扣款|罰款|罰\s*\d|倍工資|倍計|不予計價|不予請款|沒收|賠償|終止合約"),
    ("tolerance", "驗收數值", r"誤差|公差|以內|不得小於|不得超過|平整度|垂直線|\d+\s*(?:mm|cm|公分|米|M|㎜)"),
    ("material", "材料指定", r"廠牌|品牌|型號|編號|規格|出廠證明|樣品|認可|指定|限用|採用"),
    ("process", "工序要求", r"施工|批土|噴漆|面漆|打底|粉光|貼|舖|養護|順序|完成後|始可|方可"),
    ("duty", "權責分工", r"乙方負責|甲方提供|甲方負責|乙方自備|由乙方|由甲方|派駐|保固"),
]
UNSURE = "other"


@dataclass
class Clause:
    project: str          已外移（與 QS 分開的主鍵軸）
    trade: str            # 工種，如「泥作工程」
    vendor: str           已外移
    doc_date: str         # 文件日期
    sheet: str            # 來源分頁，如「工約」
    no: str               # 條號，如「38」或「61(3)」
    text: str

    @property
    def key(self) -> str:
        """主鍵。與 QS 的 `QS0404-4` 對稱，但帶案名——因為逐案有效。"""
        return f"{self.project}/{self.trade}/{self.no}"

    @property
    def kind(self) -> str:
        for tag, _, pat in KINDS:
            if re.search(pat, self.text):
                return tag
        return UNSURE

    @property
    def has_number(self) -> bool:
        """含具體數值 → 可能與 QS 衝突，需人工核對。"""
        return bool(re.search(r"\d", self.text))


@dataclass
class ContractDoc:
    project: str
    trade: str
    vendor: str
    doc_date: str
    kind: str             # 工明（施工）/ 物明（材料供應）
    clauses: list[Clause] = field(default_factory=list)
    note: str = ""


def load(raw_dir: Path = RAW_DIR) -> list[ContractDoc]:
    """讀 reference/contract/raw/*.tsv。

    格式:  #DOC <TAB> project <TAB> trade <TAB> vendor <TAB> date <TAB> kind <TAB> note
           條號 <TAB> sheet <TAB> 條文
    """
    files = sorted(raw_dir.glob("*.tsv"))
    if not files:
        raise SystemExit(f"{raw_dir} 沒有 tsv——先落地工作約定")
    docs: list[ContractDoc] = []
    cur: ContractDoc | None = None
    for fp in files:
        for ln, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            if line.startswith("#DOC"):
                p = line.split("\t")
                if len(p) < 6:
                    raise ValueError(f"{fp.name}:{ln} #DOC 欄位不足")
                cur = ContractDoc(project=p[1].strip(), trade=p[2].strip(),
                                  vendor=p[3].strip(), doc_date=p[4].strip(),
                                  kind=p[5].strip(),
                                  note=p[6].strip() if len(p) > 6 else "")
                docs.append(cur)
                continue
            if line.startswith("#"):
                continue
            p = line.split("\t")
            if len(p) < 3:
                raise ValueError(f"{fp.name}:{ln} 欄位不足：{line[:60]!r}")
            if cur is None:
                raise ValueError(f"{fp.name}:{ln} 條文出現在 #DOC 之前")
            cur.clauses.append(Clause(cur.project, cur.trade, cur.vendor,
                                      cur.doc_date, p[1].strip(), p[0].strip(),
                                      p[2].strip()))
    return docs


def all_clauses(docs=None) -> list[Clause]:
    docs = docs or load()
    return [c for d in docs for c in d.clauses]


def by_kind(kind: str, docs=None) -> list[Clause]:
    return [c for c in all_clauses(docs) if c.kind == kind]


def payment_terms(docs=None) -> list[Clause]:
    return by_kind("payment", docs)


def penalties(docs=None) -> list[Clause]:
    return by_kind("penalty", docs)


def tolerances(docs=None) -> list[Clause]:
    return by_kind("tolerance", docs)


# ── QS 交叉引用（公司映射，不入 git）──────────────────────────────────
已外移/廠商/介面配對是公司營運資訊，2026-09-01 起移到 reference/contract/mappings.yaml
# （與 TSV 同進退）。這裡只留載入器；mappings 缺席時查詢回空、自檢跳過——
# 裸 clone 的行為由 tests 的 skip 邏輯接管。
def load_mappings() -> dict:
    """QS_ANSWERS / TRADE_TO_QS / INTERFACES 三張映射表。缺檔回空 dict。"""
    if not MAPPINGS.exists():
        return {"QS_ANSWERS": {}, "TRADE_TO_QS": {}, "INTERFACES": []}
    import yaml
    return yaml.safe_load(MAPPINGS.read_text(encoding="utf-8")) or {}


def answers_qs(qs_key: str) -> list[Clause]:
    """給一個 QS 檢查項代碼，回傳能回答它的合約條款。"""
    keys = load_mappings().get("QS_ANSWERS", {}).get(qs_key, [])
    if not keys:
        return []
    idx = {c.key: c for c in all_clauses()}
    return [idx[k] for k in keys if k in idx]


# ── 合約工種→QS 對照與介面條款：資料面在 mappings.yaml（不入 git）────────
# 交叉比對只在**同工種**內做才有意義。跨工種比會出現「鷹架壁拉桿間距」
# 配「灌漿牆自攻螺絲間距」這種誤配——都是「間距」，但毫無關係。
# 介面（同案兩個工種講同一個交界）也是公司配對資料，一併外移。


def interface_clauses(name: str, docs=None) -> list[Clause]:
    """取某個介面涉及的所有條款（跨工種）。mappings 缺席時回空。"""
    docs = docs or load()
    idx = {c.key: c for c in all_clauses(docs)}
    for it in load_mappings().get("INTERFACES", []):
        if it.get("name") == name:
            return [idx[k] for k in it.get("clauses", []) if k in idx]
    return []


# 比對主題：兩邊講同一件事才比得下去
CONFLICT_TOPICS: dict[str, str] = {
    "磁磚縫隙": r"間隙|抹縫|勾縫|對縫|密接|收邊",
    "平整度誤差": r"平整|誤差|押尺|高低差",
    "厚度": r"厚度|打底|粘貼|黏著",
    "洩水坡度": r"坡度|洩水",
    "切割最小": r"切割|切細|小塊",
    "固定間距": r"間距|螺絲|擊釘",
}

_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm|㎜|公分|公尺|米|%)")


def cross_check(docs=None, qs_docs=None) -> list[dict]:
    """同工種內，QS 與合約都給了數值的檢查項 → 待人工判定的衝突候選。

    回傳的是**候選**不是結論：程式只能證明「兩邊都在談這個主題且都有數字」，
    真正是不是衝突（還是根本在講不同東西）必須人看。已確認的結論寫在
    `reference/contract/CONTRACT_VS_QS.md`。
    """
    from core import qs_data

    docs = docs or load()
    qd = qs_docs or qs_data.load()
    tmap = load_mappings().get("TRADE_TO_QS", {})
    out = []
    for d in docs:
        for qdoc_no in TRADE_TO_QS.get(d.trade, []):
            if qdoc_no not in qd:
                continue
            for topic, pat in CONFLICT_TOPICS.items():
                qs_hits = [i for i in qd[qdoc_no].required
                           if re.search(pat, i.name) and _NUM.search(i.name)]
                c_hits = [c for c in d.clauses
                          if re.search(pat, c.text) and _NUM.search(c.text)]
                if qs_hits and c_hits:
                    out.append({"trade": d.trade, "project": d.project,
                                "qsDoc": qdoc_no, "topic": topic,
                                "qs": [(i.key, i.name) for i in qs_hits],
                                "contract": [(c.key, c.text) for c in c_hits]})
    return out


def report(docs=None, log=print) -> None:
    docs = docs or load()
    cs = all_clauses(docs)
    log(f"合約 {len(docs)} 份 / 條款 {len(cs)} 條")

    log("\n── 各份摘要 ──")
    for d in docs:
        log(f"  {d.project:<8} {d.trade:<12} {d.vendor:<6} {d.doc_date:<10} "
            f"{d.kind:<4} {len(d.clauses):>3} 條  {d.note}")

    log("\n── 條款分類 ──")
    from collections import Counter
    ks = Counter(c.kind for c in cs)
    names = {t: n for t, n, _ in KINDS}
    names[UNSURE] = "其他"
    mx = max(ks.values()) if ks else 1
    for tag, _, _ in KINDS + [(UNSURE, "", "")]:
        n = ks.get(tag, 0)
        if not n:
            continue
        log(f"  {names.get(tag, tag):<6} {n:>3} ({n/len(cs)*100:4.1f}%) {'█' * round(n/mx*24)}")

    log(f"\n── 付款節點 {len(payment_terms(docs))} 條（L5 判定的輸入）──")
    for c in payment_terms(docs)[:8]:
        log(f"  {c.key:<28} {c.text[:52]}")

    log(f"\n── 罰則 {len(penalties(docs))} 條（L5 的閘門）──")
    for c in penalties(docs)[:8]:
        log(f"  {c.key:<28} {c.text[:52]}")

    log(f"\n── 驗收數值 {len(tolerances(docs))} 條（與 QS 衝突時合約優先）──")
    for c in tolerances(docs)[:8]:
        log(f"  {c.key:<28} {c.text[:52]}")

    qa = load_mappings().get("QS_ANSWERS", {})
    log("\n── QS 交叉引用（合約相依項，已對應幾項）──")
    log(f"  已對應 {len(qa)} 項")
    for qk in sorted(qa):
        log(f"  {qk:<14} ← {'、'.join(qa[qk])}")


def self_check() -> int:
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
        nonlocal warn
        if cond:
            warn += 1
            print(f"  ⚠ {msg}")

    docs = load()
    ck(len(docs) >= 4, f"至少 4 份合約（實得 {len(docs)}）")

    cs = all_clauses(docs)
    ck(len(cs) >= 100, f"條款總數 ≥100（實得 {len(cs)}）")

    # 泥作那份 64 條是最完整的
    ni = [d for d in docs if d.trade == "泥作工程"]
    ck(len(ni) == 1 and len(ni[0].clauses) >= 60,
       f"泥作工約 ≥60 條（實得 {len(ni[0].clauses) if ni else 0}）")

    # 主鍵格式帶案名
    c = cs[0]
    ck(c.key.count("/") == 2, f"主鍵格式 案名/工種/條號（實得 {c.key}）")

    # 分類：付款、罰則、數值都要抓得到
    ck(len(payment_terms(docs)) >= 5, f"付款節點 ≥5（實得 {len(payment_terms(docs))}）")
    ck(len(penalties(docs)) >= 8, f"罰則 ≥8（實得 {len(penalties(docs))}）")
    ck(len(tolerances(docs)) >= 3, f"驗收數值 ≥3（實得 {len(tolerances(docs))}）")

    # 鋼筋那份是物明不是工明——這個區分很重要，不能當成施工工約用
    st = [d for d in docs if "鋼筋" in d.trade]
    ck(bool(st) and st[0].kind == "物明",
       "鋼筋那份標記為「物明」（材料供應），非施工工約")

    # 防水：合約明文要求照片存證，這是 vision 的靶
    wp = [d for d in docs if d.trade == "防水工程"]
    ck(bool(wp), "防水工程工約已落地")
    if wp:
        ck(any("照相存證" in c.text for c in wp[0].clauses),
           "防水 14(b)「每一道施工完成應照相存證」——合約層級的請款照片要求")
        ck(any("不同顏色" in c.text for c in wp[0].clauses),
           "防水 15「每度施工需使用不同顏色塗佈」——可用顏色判斷施作到第幾度")

    ifs = load_mappings().get("INTERFACES", [])
    qa = load_mappings().get("QS_ANSWERS", {})

    # 介面條款：跨工種配對必須解析得到
    for it in ifs:
        got = interface_clauses(it["name"], docs)
        ck(len(got) == len(it["clauses"]),
           f"介面「{it['name']}」解析得到 {len(got)}/{len(it['clauses'])} 條")

    # QS 交叉引用要指得到真實條款
    for qk in qa:
        got = answers_qs(qk)
        ck(len(got) == len(qa[qk]),
           f"{qk} 的合約對應解析得到 {len(got)}/{len(qa[qk])} 條")

    from core import qs_data
    n_ci = len(qs_data.contract_items(qs_data.load()))
    warn_if(len(qa) < n_ci,
            f"qsdata 有 {n_ci} 項合約相依，目前只對應了 {len(qa)} 項"
            "——其餘待更多合約落地（尤其 QS0302-5 鋼筋綁紮，現有那份是材料供應非施工）")

    print(f"\n{'✓ 全部通過' if not bad else f'✗ {bad} 項失敗'}"
          + (f"（另有 {warn} 項待辦警告）" if warn else ""))
    return bad


def main(argv: list[str] | None = None) -> int:
    """CLI 進入點（src/contractdata.py 相容層會直接呼叫）。"""
    import argparse

    ap = argparse.ArgumentParser(prog="contractdata")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--conflicts", action="store_true",
                    help="同工種內 QS vs 合約的數值衝突候選（需人工判定）")
    a = ap.parse_args(argv)
    if a.self_check:
        return 1 if self_check() else 0
    if a.conflicts:
        rs = cross_check()
        print(f"衝突候選 {len(rs)} 組（同工種、同主題、兩邊都有數值）\n")
        for r in rs:
            print(f"【{r['project']}/{r['trade']}】{r['qsDoc']} · {r['topic']}")
            for k, t in r["qs"]:
                print(f"   QS   {k:<14} {t[:58]}")
            for k, t in r["contract"]:
                print(f"   合約 {k.split('/')[-1]:<14} {t[:58]}")
            print()
        print("⚠ 這是候選不是結論。已人工確認的三處衝突見 "
              "reference/contract/CONTRACT_VS_QS.md")
    else:
        report()
    return 0
