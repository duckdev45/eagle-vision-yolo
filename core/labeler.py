# core/labeler.py
"""labels.yaml → 標籤的規則引擎（服務層實作）。

標籤屬 derived 層：隨時可改、隨時重算。`Labeler`、`human_refs` 的實體在這裡；
manifest 組裝（labeled_manifest / legacy_manifest / pending_classes / unclaimed）
留在 src/labels.py——它們吃的是 data/ 的檔案，屬資料編排層。
"""
from __future__ import annotations

import sys as _sys
import os as _os

for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json  # noqa: E402
import re  # noqa: E402
from datetime import date  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import paths  # noqa: E402


class Labeler:
    def __init__(self, cfg: dict):
        self.junk = set(cfg.get("junk") or [])
        self.junk_pats = [re.compile(p) for p in cfg.get("junk_patterns") or []]
        self.min_len = int(cfg.get("min_title_length", 0))
        self.rules = [(re.compile(r["pattern"]), r["label"]) for r in cfg.get("rules") or []]
        self.fallback = cfg.get("fallback", "其他")
        self.drop_fallback = bool(cfg.get("drop_fallback", False))
        self.min_class_size = int(cfg.get("min_class_size", 0))
        self.small_class = cfg.get("small_class", "merge")   # drop | merge
        self.version = cfg.get("version")
        excl = cfg.get("exclude") or {}
        self.excl_dates = {str(d) for d in (excl.get("dates") or [])}
        self.excl_created_by = {str(u) for u in (excl.get("createdBy") or [])}
        self.excl_status = {str(s) for s in (excl.get("status") or [])}
        self.drop_future = bool(excl.get("future_dates", True))

    @classmethod
    def load(cls, path=None) -> "Labeler":
        return cls(yaml.safe_load((path or paths.LABELS_YAML).read_text()))

    def label(self, title) -> str | None:
        """None = 該張排除（測試資料 / 無語意標題）。順序即優先權，第一個命中者勝。

        title 可能是 NaN（CSV 空欄）、None 或非字串，一律先正規化成字串。
        """
        t = "" if title is None or (isinstance(title, float) and title != title) else str(title)
        t = t.strip()
        if not t or t in self.junk or len(t) < self.min_len:
            return None
        if any(p.search(t) for p in self.junk_pats):
            return None
        for pat, lab in self.rules:
            if pat.search(t):
                return lab
        return self.fallback

    def drop_excluded(self, df: pd.DataFrame, today: str | None = None) -> pd.DataFrame:
        """整份日報層級的排除：未來日期、壞日期、測試帳號、未定稿狀態。"""
        out = df
        # 未來日期一律是髒資料（實測 2082/2085/2092/2094 四種）。硬編清單追不上——
        # 每冒一個新的就要改一次 yaml，中間那幾天的統計已經被污染了。
        if self.drop_future and "reportDate" in out:
            out = out[out.reportDate.astype(str) <= (today or date.today().isoformat())]
        if self.excl_dates and "reportDate" in out:
            out = out[~out.reportDate.astype(str).isin(self.excl_dates)]
        if self.excl_created_by and "createdBy" in out:
            out = out[~out.createdBy.astype(str).isin(self.excl_created_by)]
        if self.excl_status and "status" in out:
            out = out[~out.status.astype(str).isin(self.excl_status)]
        return out

    def apply(self, df: pd.DataFrame, drop_small: bool = True,
              today: str | None = None, overrides: dict | None = None) -> pd.DataFrame:
        """加上 cls 欄；排除壞日報與 junk 標題；小類別併入 fallback 或整批丟掉。

        overrides = 人工複核的照片層級標籤，蓋掉 title 推出來的那個。
        None（預設）= 去讀 data/review.csv；傳 {} 可關掉（測試用）。
        """
        out = self.drop_excluded(df, today).copy()
        # 標籤只能來自 title。出工照的 tradeName 定義不同（混用會製造矛盾標籤），
        # predWorkItem 更是不能碰——那是 Gemini 的作答，拿它當標籤等於用受測者的
        # 答案當考卷答案。title 空的那些是「待預測」，不是訓練樣本，直接落到 None。
        out["cls"] = out["title"].map(self.label)
        # 人工複核蓋在規則之上。要在 notna 過濾**之前**——被 junk 規則排掉的照片，
        # 人看過說它其實是某一類，那就該收回來訓練。
        ov = load_reviews() if overrides is None else overrides
        if ov and "fileId" in out:      # 測試用的小 DataFrame 沒有 fileId 欄
            hit = out.fileId.map(ov)
            out.loc[hit.notna(), "cls"] = hit[hit.notna()]
        out = out[out.cls.notna()]
        # 人工複核裁的類別不會是 fallback，所以放在覆蓋之後排除是安全的
        if drop_small and self.drop_fallback:
            out = out[out.cls != self.fallback]
        if drop_small and self.min_class_size:
            n = out.cls.value_counts()
            small = set(n[n < self.min_class_size].index)
            if self.small_class == "drop":
                # fallback 也一起檢查：拆完之後剩下的「其他」若也不夠張數，
                # 留著只是一袋互不相干的東西，訓練與評估都學不到
                out = out[~out.cls.isin(small)]
            else:
                out.loc[out.cls.isin(small - {self.fallback}), "cls"] = self.fallback
        return out


# ── data/review.csv 存取（照片層級的真標籤）────────────────────────────
def load_reviews() -> dict[str, str]:
    """data/review.csv → {fileId: cls}。空字串或 NaN 視為「還沒判」，不覆蓋。"""
    if not paths.REVIEW.exists():
        return {}
    r = pd.read_csv(paths.REVIEW, dtype=str)
    if not {"fileId", "cls"} <= set(r.columns):
        return {}
    r = r[r.cls.notna() & (r.cls.str.strip() != "")]
    # 同一張複核兩次以最後一次為準（append-only，不覆寫舊列）
    return dict(zip(r.fileId, r.cls.str.strip()))


def orphan_reviews(lab: "Labeler | None" = None) -> dict[str, str]:
    """複核裁決指到「規則已經產不出來」的類別 → {fileId: 那個類別}。

    類別改名（實測：`設備-電梯停車` 拆成 `設備-電梯` / `設備-機械停車`）會讓舊裁決
    變成孤兒：它照樣蓋掉規則，但蓋成一個沒人認得的類別，接著被 min_class_size
    當成小類別丟掉——**人審過的照片就這樣無聲消失**。這個函式讓它出聲。
    """
    lab = lab or Labeler.load()
    valid = {lb for _, lb in lab.rules} | {lab.fallback}
    return {f: c for f, c in load_reviews().items() if c not in valid}


def load_boxes() -> dict[str, list[list[int]]]:
    """data/review.csv → {fileId: [[x0,y0,x1,y1], ...]}，0~1000 相對原圖。

    格式刻意與 Gemini `anno.evidence` 的框完全一致，`explain.iou()` 才能直接拿去比。
    空字串 = 這張看過但沒標框（跟「還沒看」不一樣，所以照樣以最後一列為準）。
    """
    if not paths.REVIEW.exists():
        return {}
    r = pd.read_csv(paths.REVIEW, dtype=str)
    if "box" not in r.columns:
        return {}
    out = {}
    for f, b in zip(r.fileId, r.box):          # append-only，後面的蓋前面的
        if not isinstance(b, str) or not b.strip():
            out.pop(f, None)
            continue
        try:
            v = json.loads(b)
        except Exception:
            continue
        if isinstance(v, list) and v:
            out[f] = [[int(c) for c in one] for one in v if len(one) == 4]
    return {f: v for f, v in out.items() if v}


def save_review(file_id: str, cls: str, note: str = "", when: str | None = None,
                boxes: list | None = None) -> None:
    """append 一列。刻意不覆寫舊列：判斷改過就是要留痕，出事才追得回來。"""
    from datetime import datetime, timezone

    ts = when or datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = pd.DataFrame([{"fileId": file_id, "cls": cls, "note": note, "reviewedAt": ts,
                         "box": json.dumps(boxes, ensure_ascii=False) if boxes else ""}])
    # 舊檔沒有 box 欄。直接 append 會讓欄位錯位，所以欄位對不上時整份重寫一次。
    if paths.REVIEW.exists():
        old = pd.read_csv(paths.REVIEW, dtype=str)
        if list(old.columns) != list(new.columns):
            pd.concat([old.reindex(columns=new.columns), new],
                      ignore_index=True).to_csv(paths.REVIEW, index=False)
            return
    new.to_csv(paths.REVIEW, mode="a", header=not paths.REVIEW.exists(), index=False)


# feMeta.specKey 是人在前端**點選**的工種，不是自由文字。實測與 title 推出的類別
# 一致率 136/138（98.6%），是目前最乾淨的人工訊號——但它只到「工種」這一層，
# 分不出 泥作-地磚 / 泥作-壁磚，所以只拿來驗前綴，不直接當標籤。
SPEC_KEY_TRADE = {"油漆": "油漆", "磁磚": "泥作", "防水": "防水",
                  "鋼筋": "結構", "連續壁": "基礎", "模板": "結構", "木作": "木作"}


def human_refs(df: pd.DataFrame, lab: "Labeler") -> pd.DataFrame:
    """把日報裡**人寫的**兩個欄位也算成參考答案，回傳 (clsChips, specTrade) 兩欄。

    - `chipsOn`：feMeta.pool 裡 `custom: true` 的查驗重點，工地主任自己打的字
      （全庫 543 條）。套同一份 labels.yaml 就得到第二個獨立的人工答案。
    - `specKey`：前端點選的工種。

    為什麼這兩個能當參考答案、`predWorkItem` 不行：**這兩個是人寫的，那個是 Gemini 答的。**
    拿受測者的作答當考卷答案是 SPEC §7.2 明文禁止的；人寫的不在此限。

    它們的用途不是取代 title——實測只能多救回 4 張。用途是**對帳**：
    與 title 推出的類別不一致的那些（8%），就是規則錯或照片與 title 講的不是同一件事。
    """
    chips = df.get("chipsOn")
    spec = df.get("specKey")
    return pd.DataFrame({
        "clsChips": (chips.map(lambda t: lab.label(t.replace("|", ""))
                               if isinstance(t, str) else None)
                     if chips is not None else None),
        "specTrade": (spec.map(lambda k: SPEC_KEY_TRADE.get(k) if isinstance(k, str) else None)
                      if spec is not None else None),
    }, index=df.index)
