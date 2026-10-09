"""切分：PMS 日報照片 → train/test（JSON）。

    {name, train:[fileId], test:[fileId], labels:{fileId: cls}, ...}

labels 直接寫進 split，train/evaluate 不必再讀標籤規則。

切分鐵律：**畫面幾乎相同的照片不可跨組**——同工地同一天按 constrId × reportDate 整天切（SPEC §8.3）。
訓練鏈的唯一入口是 pipeline/pms_workflow.py（`make pms-model`／每日排程）；這支 CLI 只切不訓。

    uv run src/split.py --name v46

2026-10-09：QMS 自訓／混合實驗（B/C）與舊 pptx（legacy）進訓練的選項已移除——主線自 v40 起只收 PMS，
那兩條路的資料原地保留（見 data/archive/README.md），程式在 git 歷史。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))
import paths
from labels import labeled_manifest

# 操作台與解釋工具要看哪一組。存成檔案而不是原始碼常數，操作台的按鈕才改得動
# ——否則重跑完還要手改程式碼，那個手動步驟一定有人忘記
# （實測踩過：畫面上一直顯示 v1 的 0.712，實際模型已經 0.847）。
FALLBACK = "v8"

# 稀有類別保底的預設值。放這裡而不是各呼叫端各寫一份——CLI 與 pms_workflow 必須切出同一份 split。
DEFAULT_MIN_TRAIN = 12  # train 少於這個數 → 把含它的測試日整天搬回 train


def current() -> str:
    f = paths.SPLITS / "CURRENT"
    return f.read_text().strip() if f.exists() else FALLBACK


def set_current(name: str) -> None:
    """切換操作台指向的模型。刻意是獨立一步，不塞進重訓流程——
    換掉所有人看到的答案是個決定，不該是跑完訓練的副作用。"""
    paths.ensure_dirs()
    (paths.SPLITS / "CURRENT").write_text(name.strip())


def encoder(name: str | None = None) -> str:
    """這個 split 的模型用哪個編碼器——split 檔說了算（features.DEFAULT_ENCODER 只管新切的）。

    v41 以前的 split 沒寫 `encoder` 欄，那時只有 siglip，所以缺欄＝LEGACY_ENCODER；
    舊版模型因此不必重訓就能照常被操作台與推論找到。
    """
    import features

    f = paths.SPLITS / f"{name or current()}.json"
    if not f.exists():
        return features.LEGACY_ENCODER
    return json.loads(f.read_text()).get("encoder") or features.LEGACY_ENCODER


def probe_path(name: str | None = None):
    """`models/probe-{encoder}-{split}.pkl`——所有讀探針的地方都走這裡，別再自己拼。"""
    name = name or current()
    return paths.MODELS / f"probe-{encoder(name)}-{name}.pkl"


def _write(payload: dict, log=print) -> dict:
    # 類別集合是 min_class_size 現算出來的，資料一長就會變。不寫進 split 的話，
    # 前後兩次評估的分母不同卻長得一樣，數字不可比。凍在這裡，evaluate 才有據可查。
    payload.setdefault("classes", sorted(set(payload["labels"].values())))
    paths.ensure_dirs()
    (paths.SPLITS / f"{payload['name']}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    log(
        f"split {payload['name']}: train {len(payload['train'])} / test {len(payload['test'])}"
        f" / {len(set(payload['labels'].values()))} 類"
    )
    return payload


def split_dates(df, test_frac: float = 0.2, today: str | None = None) -> dict[str, list[str]]:
    """回傳 {constrId: [測試日期...]}。至少留一天給測試，也至少留一天給訓練。

    未來日期是已知髒資料（實測有 2085-04-20、2092-09-13）。若不排除，
    它會變成該工地「最晚的日子」而霸佔整個測試集。這類列自動落到訓練側。
    """
    today = today or date.today().isoformat()
    out = {}
    for cid, g in df.groupby("constrId"):
        days = sorted(d for d in g.reportDate.dropna().unique() if str(d) <= today)
        n = min(max(1, math.ceil(len(days) * test_frac)), max(len(days) - 1, 0))
        out[str(cid)] = list(days[len(days) - n :]) if n else []
    return out


def rare_class_floor(
    df, is_test, test_days: dict[str, list[str]], floor: int, max_shrink: float = 0.15, log=print
):
    """稀有類別訓練保底：把含有稀有類別的測試日**整天**搬回 train。

    按天切分完全不看類別分佈，稀有類別的照片往往集中在少數幾天，那幾天一落進
    測試窗口就整批被抽走（實測 v20：金屬-欄杆鐵件 train 6 / test 10，考的比訓練的多，
    recall 0.700 是餓出來的，不是模型笨）。

    只能以**整天**為單位搬。搬單張就違反鐵律三——同一天同一個工地的另一張照片
    還留在 test，那兩張常常是同一面牆前後拍的，等於把答案偷渡進訓練側。

    三道護欄：每個工地至少留一天測試日；測試集最多縮 `max_shrink`；
    只搬**划算**的天（那天有多少比例是這個稀有類別）。

    護欄比保底重要。第一版沒有 max_shrink，實測 min_train=20 把測試集從 132 砍到 70
    ——搬 4 整天 62 張，只換到金屬-欄杆鐵件 8 張，而且還是沒到 20。
    測試集是唯一的量尺，為了餵飽一個類把量尺鋸掉一半，代價遠大於收益。
    救不動就認了：16 張的類別是資料不夠，切分變不出照片來。
    """
    moved: list[dict] = []
    days = {k: list(v) for k, v in test_days.items()}
    quota = int(int(is_test.sum()) * max_shrink)  # 測試集最多賠掉這麼多張
    stuck: set[str] = set()  # 救不動的，別在迴圈裡打轉

    pms_only = df.dataset == "pms" if "dataset" in df else df.cls.notna()

    while quota > 0:
        counts = df[~is_test & pms_only].cls.value_counts().to_dict()
        hungry = [c for c in df[pms_only].cls.unique() if counts.get(c, 0) < floor and c not in stuck]
        if not hungry:
            break
        cls_ = min(hungry, key=lambda c: counts.get(c, 0))  # 最餓的先救

        # 候選：還在測試側、含這個類別、且賠得起的天。
        # 排序看**密度**（這天有幾成是稀有類別）不是張數——20 張裡有 2 張稀有的那天，
        # 賠 20 換 2；4 張裡有 2 張的那天賠 4 換 2。後者才該先搬。
        cand = []
        for cid, dl in days.items():
            if len(dl) <= 1:  # 護欄：每工地至少留一天測試日
                continue
            for d in dl:
                day = df[is_test & (df.constrId.astype(str) == cid) & (df.reportDate == d)]
                got, cost = int((day.cls == cls_).sum()), len(day)
                if got and cost <= quota:
                    cand.append((got / cost, got, -cost, cid, d))
        if not cand:
            log(
                f"  ⚠ {cls_} train {counts.get(cls_, 0)} < {floor}，沒有划算的測試日可搬"
                f"（剩餘配額 {quota} 張）——切分救不了，要嘛補資料要嘛認了"
            )
            stuck.add(cls_)
            continue

        _, got, neg_cost, cid, d = max(cand)
        quota += neg_cost  # neg_cost 是 -len(day)
        days[cid].remove(d)
        back = is_test & (df.constrId.astype(str) == cid) & (df.reportDate == d)
        is_test = is_test & ~back
        moved.append(
            {"constrId": cid, "reportDate": d, "photos": int(back.sum()), "for": cls_, "gained": got}
        )

    if moved:
        log(f"  稀有類別保底：搬回 {len(moved)} 個測試日 / {sum(m['photos'] for m in moved)} 張")
        for m in moved:
            log(
                f"    {m['reportDate']} @ {m['constrId'][:8]} → train"
                f"（{m['for']} +{m['gained']}，整天 {m['photos']} 張）"
            )
    return is_test, days, moved


def build(
    name: str = "v1",
    test_frac: float = 0.2,
    min_train: int = DEFAULT_MIN_TRAIN,
    encoder_key: str | None = None,
    log=print,
) -> dict:
    """PMS 日報，工地 × 日期整天切。`min_train`：稀有類別把含它的測試日整天搬回 train（0＝關）。"""
    from labels import orphan_reviews, pending_classes, unclaimed

    orphan = orphan_reviews()
    if orphan:
        log(
            f"  ⚠ {len(orphan)} 筆複核裁決指到已不存在的類別，那些照片會被丟掉："
            f"{sorted(set(orphan.values()))}。改名之後要在 review.csv 補一列新名字。"
        )
    # 規則沒命中的照片不進訓練（drop_fallback: true）。那是對的，但**不能靜靜地做**
    # ——新工種進來只會表現成「張數沒長」，沒人會聯想到是規則缺一條。這裡讓它出聲。
    un = unclaimed()
    if len(un):
        top = un.title.value_counts()
        log(
            f"  ⚠ {len(un)} 張沒有任何規則認領，不會進訓練（{len(top)} 種標題）。"
            f"最多的：{'、'.join(f'{k}×{v}' for k, v in top.head(3).items())}"
        )
        log("     這些是新工種的候選，不是垃圾。跑 `make newclass` 看該補哪條規則。")
    # 規則認得、只是張數還沒到門檻的：這些在排隊，過門檻會自動上線。
    # 印出來是為了讓「差 2 張」這種情況被看見——那可能只要多同步一天就解決了。
    pend = pending_classes()
    if len(pend):
        near = "、".join(f"{r.cls} {r.photos}/{r.photos + r.need}" for r in pend.head(4).itertuples())
        log(
            f"  排隊中 {len(pend)} 類（規則認得、張數不足 min_class_size，過門檻自動上線）：{near}"
            + ("…" if len(pend) > 4 else "")
        )
    df = labeled_manifest()
    if "dataset" not in df:
        df = df.assign(dataset="pms")
    test_days = split_dates(df, test_frac)

    def in_test_day(r) -> bool:
        return r.reportDate in test_days.get(str(r.constrId), [])

    is_test = df.apply(in_test_day, axis=1).astype(bool)

    moved: list[dict] = []
    if min_train:
        is_test, test_days, moved = rare_class_floor(df, is_test, test_days, min_train, log=log)

    # 人標框裁出來的那些塊：同一個標籤、**只進 train**（測試集沒有人畫的框）。
    # 母張若落在測試日，它的裁切塊也不能進 train——那等於把答案偷渡到訓練側。
    from labels import load_boxes

    cls_of = dict(zip(df.fileId, df.cls))
    train_ids = set(df[~is_test].fileId)
    crops = {}
    for f, bs in load_boxes().items():
        if f in train_ids:
            crops.update({f"{f}#{i}": cls_of[f] for i in range(len(bs))})
    if crops:
        log(f"  人標框裁切 {len(crops)} 塊併入 train")

    import features

    payload = {
        "name": name,
        "source": "report",
        # 這份 split 的模型用哪個編碼器；train/evaluate/操作台/推論都從這裡讀（見 encoder()）
        "encoder": encoder_key or features.DEFAULT_ENCODER,
        "testFrac": test_frac,
        "minTrain": min_train,
        # 搬過哪幾天要留證據：測試集比 testFrac 算出來的小，看到這個才知道為什麼
        "floorMovedDays": moved,
        "testDaysByConstr": test_days,
        "train": sorted(df[~is_test].fileId.tolist()) + sorted(crops),
        "test": sorted(df[is_test].fileId.tolist()),
        "labels": {**dict(zip(df.fileId, df.cls)), **crops},
        "datasets": {**dict(zip(df.fileId, df.dataset)), **{k: "crop" for k in crops}},
        "trainCrops": len(crops),
        "trainClassCounts": df[~is_test].cls.value_counts().to_dict(),
        "testClassCounts": df[is_test].cls.value_counts().to_dict(),
    }
    out = _write(payload, log)
    missing = set(payload["trainClassCounts"]) - set(payload["testClassCounts"])
    if missing:
        log(f"  ⚠ 測試集缺少類別：{sorted(missing)}——這些類模型會輸出但這份考卷驗不了")
    # 考的比訓練的多是餓死的徵兆，v20 的金屬-欄杆鐵件就是這樣（train 6 / test 10）
    starved = {
        c: (payload["trainClassCounts"].get(c, 0), n)
        for c, n in payload["testClassCounts"].items()
        if payload["trainClassCounts"].get(c, 0) < n
    }
    if starved:
        log(f"  ⚠ 測試張數 > 訓練張數：{ {c: f'train {a} / test {b}' for c, (a, b) in starved.items()} }")
    return out


def load(name: str = "v1") -> dict:
    return json.loads((paths.SPLITS / f"{name}.json").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument(
        "--min-train",
        type=int,
        default=DEFAULT_MIN_TRAIN,
        help="稀有類別訓練保底：train 少於 N 張就把含它的測試日整天搬回 train（0 = 關）",
    )
    ap.add_argument("--encoder", default=None, help="模型編碼器（預設 features.DEFAULT_ENCODER）")
    a = ap.parse_args()
    build(a.name, a.test_frac, min_train=a.min_train, encoder_key=a.encoder)
