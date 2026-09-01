"""切分。三種資料組態，共用同一個 JSON 格式。

    {name, train:[fileId], test:[fileId], labels:{fileId: cls}, ...}

labels 直接寫進 split，train/evaluate 就不必知道這批資料來自日報還是 QMS，
三個實驗（A 日報 / B QMS / C 混合）走同一條程式碼路徑。

切分鐵律都是同一條：**畫面幾乎相同的照片不可跨組**。
- 日報：同工地同一天 → 按 constrId × reportDate 切（SPEC §8.3）
- QMS：同一格是同位置同時間拍的 2~3 張 → 按 constructionInsId 整格切

    uv run src/split.py                     # A：日報 v1
    uv run src/split.py --qms               # B：QMS 自己的 train/test
    uv run src/split.py --merged            # C：日報 train + QMS → 測日報 test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))
import paths
from labels import labeled_manifest

# 操作台與解釋工具要看哪一組。存成檔案而不是原始碼常數，操作台的按鈕才改得動
# ——否則重跑完還要手改程式碼，那個手動步驟一定有人忘記
# （實測踩過：畫面上一直顯示 v1 的 0.712，實際模型已經 0.847）。
FALLBACK = "v8"

# 稀有類別保底的預設值。放這裡而不是各呼叫端各寫一份——操作台的「重跑模型」按鈕
# 與 `make model` 必須切出同一份 split，否則畫面上的分數跟命令列跑的對不起來。
DEFAULT_MIN_TRAIN = 12  # train 少於這個數 → 把含它的測試日整天搬回 train
DEFAULT_LEGACY_FILL = 40  # 切完還不足 → 只從 legacy 補那幾類到這個數


def current() -> str:
    f = paths.SPLITS / "CURRENT"
    return f.read_text().strip() if f.exists() else FALLBACK


def set_current(name: str) -> None:
    """切換操作台指向的模型。刻意是獨立一步，不塞進重訓流程——
    換掉所有人看到的答案是個決定，不該是跑完訓練的副作用。"""
    paths.ensure_dirs()
    (paths.SPLITS / "CURRENT").write_text(name.strip())


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
    救不動就認了：16 張的類別是資料不夠，切分變不出照片來（那是 legacy_topup 的事）。
    """
    moved: list[dict] = []
    days = {k: list(v) for k, v in test_days.items()}
    quota = int(int(is_test.sum()) * max_shrink)  # 測試集最多賠掉這麼多張
    stuck: set[str] = set()  # 救不動的，別在迴圈裡打轉

    # 只數 PMS：這一步治的是「按天切分把 PMS 稀有類別整批抽進測試側」。
    # legacy 那幾千張永遠在 train，把它們算進來會讓每個類別都看起來吃飽，保底變空轉
    # （實測 --min-train 12 --legacy-fill 40 一天都沒搬）。
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


def legacy_topup(df, is_test, floor: int, log=print):
    """只用 legacy 補**PMS 訓練張數不足**的類別，其餘 legacy 全丟。

    全開 legacy 實測有害（README：同一份 124 張測試集 0.823 → 0.782）——5599 張
    legacy 淹掉 641 張 PMS，訓練分佈整個歪成另一個管道（有人挑過、重壓縮過）。

    但「分佈歪掉」是給有東西學的類別的煩惱。PMS 只有 16 張的金屬-欄杆鐵件，
    問題不是學歪，是根本沒得學。所以按類別補到門檻就停，不是全開或全關。

    **只補 PMS 已經有的類別**。legacy 那 9 個 PMS 沒有的類（模板、廚具、石材…）
    不在這裡開——解鎖新類別是 `--with-legacy` 的事，是產品決定。
    第一版沒擋，`--legacy-fill 40` 順手把類別數從 11 開成 21，其中 9 類測試集
    一張都沒有，等於偷偷換了一個沒人要求的模型。
    """
    pms_train = df[(df.dataset == "pms") & ~is_test].cls.value_counts().to_dict()
    keep, note = [], []
    for cls_, g in df[df.dataset == "legacy"].groupby("cls"):
        if cls_ not in pms_train:  # PMS 沒有的類別不在這裡開
            continue
        need = floor - pms_train.get(cls_, 0)
        if need <= 0:
            continue
        # 固定取排序後的前 n 張——同樣的資料要切出同樣的 split，不能靠亂數
        take = sorted(g.fileId)[:need]
        keep += take
        note.append(f"{cls_} {pms_train.get(cls_, 0)}+{len(take)}")
    if note:
        log(f"  legacy 補訓練（門檻 {floor}）：{'、'.join(note)}")
    drop = (df.dataset == "legacy") & ~df.fileId.isin(keep)
    n = int(drop.sum())
    if n:
        log(f"  其餘 legacy {n} 張丟掉——沒缺的類別不補，補了會把訓練分佈拉去別的管道")
    return df[~drop], is_test[~drop]


def build(
    name: str = "v1",
    test_frac: float = 0.2,
    with_legacy: bool = False,
    min_train: int = DEFAULT_MIN_TRAIN,
    legacy_fill: int = DEFAULT_LEGACY_FILL,
    log=print,
) -> dict:
    """A：日報，工地 × 日期。舊 pptx（`dataset == legacy`）只進 train。

    為什麼舊資料不進 test：測試集要代表**產品實際會收到的照片**，那是 PMS 上傳的。
    舊 pptx 是同分佈但不同管道（有人挑過、貼進投影片、重壓縮過），
    放進 test 等於把考卷換成另一份，跨次實驗就再也比不了。

    為什麼 with_legacy 預設是關的：實測它**沒有提升**現有類別的準確率
    （同一份 124 張測試集、限縮到測試集有的 12 類：0.823 → 0.782）。
    它的價值不在分數，在於解鎖 PMS 張數不夠的 9 個類別（模板、鋼筋、石材、廚具…）。
    要那些類別就開，要現有 10 類的最高分就別開——這是產品決定，不是預設值該替人做的。

    `min_train` / `legacy_fill` 是稀有類別的兩道保底，預設都關（給 0 就是舊行為）：
    前者調切分（把含稀有類別的測試日整天搬回 train），後者調資料
    （只對缺的類別從 legacy 補到門檻）。先切分後補——切分是免費的，補 legacy
    要付分佈歪掉的代價，能不補就不補。
    """
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
    # legacy_fill 要有 legacy 列可挑，先全部載進來，補完再把沒用到的丟掉
    df = labeled_manifest(with_legacy=with_legacy or bool(legacy_fill))
    if "dataset" not in df:
        df = df.assign(dataset="pms")
    if legacy_fill and not with_legacy:
        # min_class_size 是**合併後**才算的（labels.py 的設計），所以光是載進 legacy
        # 就會讓幾個 PMS 張數不足的類別復活，考卷跟著變（實測 test 132 → 136）。
        # legacy_fill 只該加訓練資料，不該改考卷——把類別集合鎖回 PMS 自己算的那組。
        keep = set(labeled_manifest(with_legacy=False).cls)
        if extra := sorted(set(df.cls) - keep):
            log(f"  載入 legacy 讓 {len(extra)} 個類別復活，鎖回 PMS 的類別集合：{extra}")
        df = df[df.cls.isin(keep)]
    pms = df[df.dataset == "pms"]
    test_days = split_dates(pms, test_frac)  # 測試日只從 PMS 這邊挑

    def in_test_day(r) -> bool:
        return r.reportDate in test_days.get(str(r.constrId), [])

    is_test = df.apply(lambda r: r.dataset == "pms" and in_test_day(r), axis=1)

    moved: list[dict] = []
    if min_train:
        is_test, test_days, moved = rare_class_floor(df, is_test, test_days, min_train, log=log)

    # 舊資料若落在 PMS 的測試日、同一個工地，就是同一天同一面牆的另一張照片 → 丟掉。
    # 不丟的話它進了 train，畫面幾乎相同的那張在 test，這正是鐵律三禁止的事。
    # 必須在保底搬完之後才算——測試日搬回 train 了，撞上的就不算撞。
    leak = df.apply(lambda r: r.dataset == "legacy" and in_test_day(r), axis=1)
    if leak.any():
        log(f"  舊資料撞到 PMS 測試日，丟掉 {int(leak.sum())} 張")
    df = df[~leak]
    is_test = is_test[~leak]

    if legacy_fill and not with_legacy:
        df, is_test = legacy_topup(df, is_test, legacy_fill, log)

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

    n_legacy = int((df.dataset == "legacy").sum())
    payload = {
        "name": name,
        "source": "report+legacy" if n_legacy else "report",
        "testFrac": test_frac,
        "minTrain": min_train,
        "legacyFill": legacy_fill,
        # 搬過哪幾天要留證據：測試集比 testFrac 算出來的小，看到這個才知道為什麼
        "floorMovedDays": moved,
        "testDaysByConstr": test_days,
        "train": sorted(df[~is_test].fileId.tolist()) + sorted(crops),
        "test": sorted(df[is_test].fileId.tolist()),
        "labels": {**dict(zip(df.fileId, df.cls)), **crops},
        "datasets": {**dict(zip(df.fileId, df.dataset)), **{k: "crop" for k in crops}},
        "trainLegacy": n_legacy,
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


def split_cells(df, test_frac: float = 0.2, seed: int = 0) -> set[str]:
    """QMS：每個中類抽 test_frac 的**格子**（不是照片）進測試集。

    同一格是同位置同時間拍的，拆開就等於把近乎同一張分到兩邊，數字會漂亮到假。
    """
    test = set()
    for _, g in df.groupby("cls"):
        cells = sorted(g.constructionInsId.unique())
        k = min(max(1, math.ceil(len(cells) * test_frac)), max(len(cells) - 1, 0))
        # 用 hash 而非 random，換機器結果一樣
        test |= set(sorted(cells, key=lambda c: hash((seed, c)))[:k])
    return test


def build_qms(
    name: str = "qms-v1", test_frac: float = 0.2, val_frac: float = 0.1, min_class_size: int = 30, log=print
) -> dict:
    """B：QMS 自己的 train/val/test（中類標籤，按格切）。

    有 val 才能誠實挑 epoch。日報那邊只有 441 張切不出三份，所以只有 QMS 有這個。
    """
    import qms

    df = qms.labeled(min_class_size=min_class_size)
    test_cells = split_cells(df, test_frac, seed=0)
    rest = df[~df.constructionInsId.isin(test_cells)]
    val_cells = split_cells(rest, val_frac / (1 - test_frac), seed=1)
    is_test = df.constructionInsId.isin(test_cells)
    is_val = df.constructionInsId.isin(val_cells) & ~is_test
    payload = {
        "name": name,
        "source": "qms",
        "testFrac": test_frac,
        "valFrac": val_frac,
        "testCells": sorted(test_cells),
        "valCells": sorted(val_cells),
        "train": sorted(df[~is_test & ~is_val].fileId.tolist()),
        "val": sorted(df[is_val].fileId.tolist()),
        "test": sorted(df[is_test].fileId.tolist()),
        "labels": dict(zip(df.fileId, df.cls)),
        "trainClassCounts": df[~is_test & ~is_val].cls.value_counts().to_dict(),
        "testClassCounts": df[is_test].cls.value_counts().to_dict(),
    }
    out = _write(payload, log)
    log(f"  val {len(payload['val'])} 張")
    return out


ALIAS = paths.ROOT / "reference" / "qms_to_report.yaml"


def qms_aliased(min_class_size: int = 30):
    """QMS 中類經 reference/qms_to_report.yaml 折進日報的類別；沒映到的整批丟掉。"""
    import yaml

    import qms

    m = (yaml.safe_load(ALIAS.read_text()) or {}).get("map") or {}
    q = qms.labeled(min_class_size=min_class_size)
    q = q[q.cls.isin(m)].assign(cls=lambda d: d.cls.map(m))
    return q


def build_merged(
    name: str = "mix-v1", report_split: str = "v1", alias: bool = True, min_class_size: int = 30, log=print
) -> dict:
    """C：日報 train + QMS → 測試集**原封不動用日報的 test**。

    測試集不動是這個實驗唯一的意義：幾條線的分母完全一樣才比得出來。

    alias=False 時直接把 QMS 中類當成新類別（實測有害，留著當對照組）。
    """
    rep = load(report_split)
    import qms

    q = qms_aliased(min_class_size) if alias else qms.labeled(min_class_size=min_class_size)
    payload = {
        "name": name,
        "source": "merged",
        "reportSplit": report_split,
        "alias": alias,
        "train": sorted(set(rep["train"]) | set(q.fileId)),
        "test": rep["test"],
        "labels": {**rep["labels"], **dict(zip(q.fileId, q.cls))},
        "qmsPhotos": len(q),
        "qmsClassCounts": q.cls.value_counts().to_dict(),
    }
    return _write(payload, log)


def build_qms_only(
    name: str = "qms-transfer", report_split: str = "v1", min_class_size: int = 30, log=print
) -> dict:
    """B'：**只用 QMS 訓練**，測日報的 test。純粹的領域遷移測試。"""
    rep = load(report_split)
    q = qms_aliased(min_class_size)
    payload = {
        "name": name,
        "source": "qms-transfer",
        "reportSplit": report_split,
        "train": sorted(q.fileId),
        "test": rep["test"],
        "labels": {**rep["labels"], **dict(zip(q.fileId, q.cls))},
        "qmsPhotos": len(q),
        "qmsClassCounts": q.cls.value_counts().to_dict(),
    }
    return _write(payload, log)


def load(name: str = "v1") -> dict:
    return json.loads((paths.SPLITS / f"{name}.json").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--qms", action="store_true", help="B：QMS 自己的 train/test")
    ap.add_argument("--transfer", action="store_true", help="B'：只用 QMS 訓練，測日報 test")
    ap.add_argument("--merged", action="store_true", help="C：日報 + QMS 混合")
    ap.add_argument(
        "--with-legacy", action="store_true", help="加入舊 pptx 那批（只進 train，會多出 9 個類別）"
    )
    ap.add_argument(
        "--min-train",
        type=int,
        default=DEFAULT_MIN_TRAIN,
        help="稀有類別訓練保底：train 少於 N 張就把含它的測試日整天搬回 train（0 = 關）",
    )
    ap.add_argument(
        "--legacy-fill",
        type=int,
        default=DEFAULT_LEGACY_FILL,
        help="切完還不足 N 張的類別，從 legacy 補到 N（只補缺的類，不是全開；0 = 關）",
    )
    ap.add_argument("--no-alias", action="store_true", help="C 的對照組：不折標籤")
    a = ap.parse_args()
    if a.qms:
        build_qms(a.name or "qms-v1", a.test_frac)
    elif a.transfer:
        build_qms_only(a.name or "qms-transfer")
    elif a.merged:
        build_merged(a.name or ("mix-raw" if a.no_alias else "mix-v1"), alias=not a.no_alias)
    else:
        build(
            a.name or "v1",
            a.test_frac,
            with_legacy=a.with_legacy,
            min_train=a.min_train,
            legacy_fill=a.legacy_fill,
        )
