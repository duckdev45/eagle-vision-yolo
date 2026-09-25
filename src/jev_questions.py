"""TypeSafe / Jev 的問題與閾值——**全部常數集中在這一支**。

TypeSafe 官方建議：questions 與 thresholds 放單一檔案，人要 review 的就是這些，
不要散在呼叫端（docs.typesafe.ai/agent-skill〈It's difficult to review TypeSafe code〉）。

設計紀律（與 WORKFLOW.md §2 四條鐵律同源）：

  **criteria 不手寫，從 labels.yaml 反推。**
  每個類別的說明取自該類 regex 的關鍵字（= 現場真實寫法），而不是我們對建築工法的
  通用認知。理由與 v11 的「看圖仲裁才收編」同一條：這個 repo 裡凡是語義判斷，
  來源必須是人裁決過的紀錄。憑通用知識生成建築語義曾經與 QS 原文衝突過，不再犯。

  所以 labels.yaml 改版 → 這裡的 criteria 自動跟著改，不會兩邊各說各話。
"""

from __future__ import annotations

import re

import yaml

import paths

# --- 閾值（人要 review 的就是這幾個數字）-----------------------------------
# g2_gate.PRECISION_BAR 是同一條線：模型自動標的那段要 ≥ 0.90 precision。
AUTO_ACCEPT_CONF = 0.90  # 以上 → 自動採用
REVIEW_FLOOR_CONF = 0.50  # 以下 → 不猜，直接進 review 佇列
OOD_NOUL = 0.50  # 「這不屬於任何已知類別」的 noul 門檻

MODEL = "jev-latest"  # 實驗要可複現時改釘版本號，例：jev-1.13.0

# regex 元字元：帶這些的 alternative 不是人看得懂的現場寫法，不當 example 給模型
_META = re.compile(r"[\[\]().*+?^$\\{}]")


def _keywords(pattern: str) -> list[str]:
    """regex → 現場寫法清單。`[Ee][Pp][Oo][Xx]` 這種字元類直接丟掉，只留白話詞。"""
    return [a for a in (p.strip() for p in pattern.split("|")) if a and not _META.search(a)]


def class_criteria(classes: list[str], labels_yaml=None) -> dict[str, dict]:
    """split 凍住的 classes → Jev 的 Choice criteria。

    labels.yaml 的規則**順序即優先權**，這個資訊 Jev 吃不到（它平行評分、沒有先後），
    所以同一組關鍵字餵給兩支類別時，靠 examples 的差異區分而不是靠順序。
    外牆磁磚 / 壁磚貼飾 這種官方文件說的「容易混淆的鄰居」，用 not_for 明講。
    """
    cfg = yaml.safe_load((labels_yaml or paths.LABELS_YAML).read_text())
    by_label: dict[str, list[str]] = {}
    for r in cfg.get("rules") or []:
        by_label.setdefault(r["label"], []).extend(_keywords(r["pattern"]))

    out: dict[str, dict] = {}
    for c in classes:
        kws = by_label.get(c, [])
        entry: dict = {"工種": c.split("-")[0], "施作內容": c.split("-", 1)[-1]}
        if kws:
            entry["現場寫法"] = kws
        # 鄰居互斥：labels.yaml 註解裡已經寫明的幾組，明講給模型聽。
        for a, b in _NEIGHBOURS:
            if c == a:
                entry["不是"] = b
            elif c == b:
                entry["不是"] = a
        out[c] = entry

    # 沒有這一項，模型會被迫從 17 類裡硬挑一個。官方文件明說：
    # 清單可能蓋不全時要給 other，讓它能回答「都不是」。
    out[OTHER] = {
        "what": "以上皆非：非施工畫面（查驗表單、人數清點、會勘），或現有類別沒涵蓋的工種",
        "examples": _keywords(" ".join(cfg.get("junk_patterns") or [])) or ["自主查驗", "人數清點"],
    }
    return out


OTHER = "其他"

# labels.yaml 註解裡已經裁定過的混淆對。只列「文件明文寫過」的，不自行擴充。
_NEIGHBOURS = [
    ("泥作-外牆磁磚", "泥作-壁磚貼飾"),  # v8：外牆 vs 室內，工序位置不同
    ("泥作-打底", "泥作-粉光"),  # v9：QS0402 粗底層 vs 完成面，兩階段
    ("輕隔間-灌漿牆", "輕隔間-輕質磚"),  # v7：QS0512 vs QS0406
    ("木作-天花封板", "木作-暗架天花板"),  # v12：角材木夾板 vs 金屬格柵
]


def build(classes: list[str]) -> dict:
    """一次問完所有問題（官方：平行評估，多問一題幾乎不影響延遲）。"""
    from typesafe_sdk import Choice, Noul

    return {
        "trade": Choice(
            instructions={
                "question": "這張工地照片的標題描述的是哪一個工種施作內容？",
                "focus": "依標題文字判斷施作項目。選項名格式為「工種-施作內容」。",
            },
            criteria=class_criteria(classes),
        ),
        # regex 的 junk_patterns 對照組：這批東西 drop 掉是對的，Jev 該同意。
        "not_construction": Noul(
            instructions="這個標題描述的是查驗/行政活動（自主查驗、人數清點、會勘、放樣、"
            "進度回報）而不是實際施工畫面。"
        ),
        # drop_fallback 丟掉的那 1,140 張，真正該問的是這一題。
        "out_of_taxonomy": Noul(instructions="這是實際施工畫面，但施作內容不屬於提供的任何一個工種類別。"),
    }


def state_of(row, *, with_human_refs: bool) -> dict:
    """照片 → state。

    兩組 arm 刻意不同：
      title_only  = 與 regex 完全同樣的輸入，這樣 A/B 才公平。
      with_refs   = 加上 chipsOn（主任自打的查驗重點）與 specKey（前端點選工種）。
                    這兩欄是**人寫的**（WORKFLOW.md S7 ④），regex 引擎根本沒在讀，
                    是 Jev 能贏的地方，但贏了不能算在「取代 regex」帳上。
    """
    s: dict = {"標題": str(row.get("title") or "")}
    if with_human_refs:
        if row.get("specKey"):
            s["前端點選工種"] = str(row["specKey"])
        if row.get("chipsOn"):
            s["現場查驗重點"] = [c for c in str(row["chipsOn"]).split("|") if c.strip()]
    return s
