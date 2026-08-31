"""labels.yaml → 標籤。屬 derived 層，隨時可改、隨時重算。"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import date

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
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


def pending_classes(df: pd.DataFrame = None, lab: "Labeler" = None) -> pd.DataFrame:
    """規則認得、但張數還沒到 `min_class_size` 的類別 → 每列一類。

    這些不是錯誤，是**在排隊**：張數一過門檻就自動進訓練，不用改任何程式
    （`金屬-欄杆鐵件` 就是這樣在 v18 自己冒出來的）。

    門檻算的是 **train + test 全部**，不是 train——`apply()` 在切分之前就先砍了。
    別跟 Makefile 的 `MIN_TRAIN` 搞混，那個是進場之後的切分保底。

    回傳欄位：cls / photos（現有）/ need（還差幾張）/ latest（最近一張的日期）。
    """
    lab = lab or Labeler.load()
    if df is None:
        from sync import _truthy
        df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
        if "active" in df:
            df = df[_truthy(df.active)]
    full = lab.apply(df, drop_small=False)
    full = full[full.cls != lab.fallback]
    n = full.cls.value_counts()
    small = n[n < lab.min_class_size]
    if not len(small):
        return pd.DataFrame(columns=["cls", "photos", "need", "latest"])
    latest = (full[full.cls.isin(small.index)].groupby("cls").reportDate.max()
              if "reportDate" in full else pd.Series(dtype=str))
    return pd.DataFrame({
        "cls": small.index,
        "photos": small.values,
        "need": lab.min_class_size - small.values,
        "latest": [latest.get(c, "") for c in small.index],
    }).sort_values("need").reset_index(drop=True)


def unclaimed(df: pd.DataFrame = None, lab: "Labeler" = None) -> pd.DataFrame:
    """規則沒認領、因此不會進訓練的照片（`cls == fallback`）。

    `apply()` 預設就把它們濾掉了，所以誰也看不到——操作台 ⑥ 那個
    「落入 fallback 的標題 —— 這裡就是下一條規則的來源」面板，實測永遠顯示 0 張，
    因為它拿到的 DataFrame 早就被濾過。這個函式繞開那層濾網。

    照片沒有被刪，都還在 raw/photos/。少的只是一條規則。
    """
    lab = lab or Labeler.load()
    if df is None:
        from sync import _truthy
        df = pd.read_csv(paths.MANIFEST, dtype=str, keep_default_na=False, na_values=[""])
        if "active" in df:
            df = df[_truthy(df.active)]
    out = lab.apply(df, drop_small=False)
    return out[out.cls == lab.fallback]


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


def orphan_reviews(lab: "Labeler" = None) -> dict[str, str]:
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


def legacy_manifest() -> pd.DataFrame:
    """舊 pptx 那批，欄位對齊到 PMS 的最小集合。

    `constrId` 用 constrName 對回 PMS——切分要按工地分組，名字對不上就分不了組。
    對不到的（PMS 沒有那個案場）用 `legacy:<名字>` 當自己的 id，同名的至少不會拆開。
    """
    if not paths.LEGACY_MANIFEST.exists():
        return pd.DataFrame()
    d = pd.read_csv(paths.LEGACY_MANIFEST, dtype=str)
    name2id = {}
    if paths.MANIFEST.exists():
        p = pd.read_csv(paths.MANIFEST, dtype=str, usecols=["constrName", "constrId"])
        name2id = dict(p.dropna().drop_duplicates().values)
    return pd.DataFrame({
        "fileId": d.fileId,
        "dataset": "legacy",
        "title": d.title,
        "reportDate": d.reportDate.fillna(""),
        "constrName": d.constrName,
        "constrId": d.constrName.map(lambda n: name2id.get(n, f"legacy:{n}")),
    })


def labeled_manifest(active_only: bool = True, with_legacy: bool = False) -> pd.DataFrame:
    """PMS 日報（＋選配的舊 pptx）→ 加上 cls 欄。

    兩批**合併後才套規則**，不是各自套：`min_class_size` 的門檻是對整個訓練集算的，
    分開套會得到兩套不同的類別集合，接不起來（舊資料正好補的就是原本不足被 drop 的類）。
    """
    df = pd.read_csv(paths.MANIFEST)
    if active_only and "active" in df:
        df = df[df.active.astype(str).str.lower().isin(["true", "1"])]
    df = df.assign(dataset="pms")
    df = df.join(human_refs(df, Labeler.load()))
    if with_legacy:
        lg = legacy_manifest()
        if len(lg):
            df = pd.concat([df, lg], ignore_index=True)
    return Labeler.load().apply(df)


if __name__ == "__main__":
    d = labeled_manifest()
    print(d.cls.value_counts().to_string())
    print(f"\n合計 {len(d)} 張 / {d.cls.nunique()} 類")
