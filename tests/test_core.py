"""最小自檢：邏輯壞掉就會炸。不需要 pytest，直接 uv run tests/test_core.py。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from PIL import Image

from labels import Labeler
from prepare import mask_corners, mask_watermark
from split import split_dates
from sync import KNOWN_ANNO_KEYS, anno_keys, flatten, merge_index, merge_manifest

CFG = {
    "junk": ["TEST"],
    "min_title_length": 3,
    "fallback": "其他",
    "min_class_size": 2,
    "rules": [
        {"pattern": "防水|止漏", "label": "防水"},
        {"pattern": "外牆磁磚|壁磚", "label": "壁磚貼飾"},
        {"pattern": "地磚", "label": "地磚貼飾"},
        {"pattern": "油漆|批土", "label": "油漆批土"},
        {"pattern": "打底|粉光|泥作", "label": "泥作"},
    ],
}


def test_labels():
    L = Labeler(CFG)
    assert L.label("TEST") is None  # junk
    assert L.label("7F") is None  # 太短
    assert L.label(None) is None
    assert L.label(float("nan")) is None  # CSV 空欄讀回來是 NaN
    assert L.label("B1 防水施作") == "防水"
    assert L.label("13F外牆磁磚") == "壁磚貼飾"
    # 順序即優先權：外牆磁磚規則在地磚之前，含兩者時前者勝
    assert L.label("外牆磁磚與地磚") == "壁磚貼飾"
    assert L.label("13F~7F外牆打底粉光") == "泥作"
    assert L.label("某某不明工項") == "其他"


def test_exclude_bad_reports():
    L = Labeler({**CFG, "exclude": {"dates": ["2026-07-04"], "createdBy": ["u-test"]}})
    df = pd.DataFrame(
        {
            "title": ["B1 防水施作"] * 4,
            "reportDate": ["2026-08-01", "2026-07-04", "2026-08-02", "2026-08-03"],
            "createdBy": ["u-ok", "u-ok", "u-test", "u-ok"],
        }
    )
    kept = L.apply(df, drop_small=False, today="2026-08-17")
    assert list(kept.reportDate) == ["2026-08-01", "2026-08-03"], kept.to_dict()


def test_exclude_future_dates():
    """未來日期是髒資料，且硬編清單追不上（實測 2082/2085/2092/2094 四種）。"""
    L = Labeler(CFG)
    df = pd.DataFrame(
        {
            "title": ["B1 防水施作"] * 4,
            "reportDate": ["2026-08-01", "2085-04-20", "2094-01-05", "2026-08-17"],
        }
    )
    kept = L.apply(df, drop_small=False, today="2026-08-17")
    assert list(kept.reportDate) == ["2026-08-01", "2026-08-17"], kept.to_dict()


def test_exclude_status():
    L = Labeler({**CFG, "exclude": {"status": ["DRAFT"]}})
    df = pd.DataFrame({"title": ["B1 防水施作"] * 2, "status": ["SUBMITTED", "DRAFT"]})
    assert list(L.apply(df, drop_small=False).status) == ["SUBMITTED"]


def test_rule_order_is_the_contract():
    """順序即優先權。這些組合都是實測踩過的坑，改 labels.yaml 時不准回歸。"""
    L = Labeler.load()  # 刻意讀真的 labels.yaml，不是 CFG
    # 景觀必須排在清潔前面：「整理」會被清潔規則先吃掉（實測 14 張）
    assert L.label("中庭植栽整理") == "植栽-景觀"
    # 烤漆玻璃/明鏡排在油漆前面，歸玻璃（v10 翻案：QS0606 查驗表 chipsOn 逐字對上）
    assert L.label("1F 烤漆玻璃安裝") == "門窗-玻璃"
    assert L.label("1F-2F各戶室內烤漆及明鏡安裝") == "門窗-玻璃"
    # 矽利康自成一支，不歸門窗
    assert L.label("鋁門窗周邊矽利康") == "防水-矽利康"
    # 打底歸泥作、批土歸油漆（工程分類樹核對過）
    assert L.label("外牆打底") == "泥作-打底"
    assert L.label("天花批土") == "油漆-批土塗裝"
    # 粉光排在打底前面：兩者關鍵字不重疊，但粉光是完成面查驗，寫在先讀起來順（v9 拆分）
    assert L.label("13F內部牆面粉光") == "泥作-粉光"
    # 判不出階段的殘餘字樣（泥作/抹灰/砌/吊線）預設歸打底，不強猜
    assert L.label("泰舜初驗泥作缺失改善") == "泥作-打底"
    # 三個泥作類是兄弟，前綴必須一致
    assert L.label("室內地磚貼飾") == "泥作-地磚貼飾"
    assert L.label("9F浴室壁磚貼飾") == "泥作-壁磚貼飾"
    # 外牆磁磚必須排在壁磚前面，否則被「壁磚」子字串接走，外牆永遠拆不出來（v8 拆分）
    assert L.label("13F~7F外牆磁磚貼飾") == "泥作-外牆磁磚"
    assert L.label("外牆貼磁磚") == "泥作-外牆磁磚"
    # 批土排在輕隔間前面：「15F輕隔間批土」照片上就是批土（實測 14 張）
    assert L.label("15F輕隔間批土") == "油漆-批土塗裝"
    assert L.label("14F輕隔間擊釘") == "輕隔間-灌漿牆"  # 沒批土、沒輕質磚 → 預設灌漿牆（v7 拆分）
    # 輕質磚必須排在灌漿牆前面，否則被「輕隔間」吃掉，這類永遠拆不出來
    assert L.label("15F輕質磚砌築") == "輕隔間-輕質磚"
    # 鋼筋排在清潔前面：「整理」會把鋼筋加工場吃進清潔（實測 10 張）
    assert L.label("鋼筋加工場場地整理") == "結構-鋼筋"
    # 電梯與機械停車是樹上兩個中類，不可併回大類；兩詞都有時停車設備勝
    assert L.label("電梯安裝") == "設備-電梯"
    assert L.label("B2 機械停車設備安裝") == "設備-機械停車"
    assert L.label("電梯式機械停車設備") == "設備-機械停車"


def test_no_rule_is_permanently_shadowed():
    """規則順序即優先權。較晚規則的某個關鍵字若整個「包含」較早規則的某個關鍵字
    （較早的是較晚的子字串），較晚那個關鍵字就永遠不可能命中——凡是含較早關鍵字
    的字串，較早的規則必定先比對到。方向反過來（較晚的關鍵字是較早的子字串，
    例如「烤漆玻璃」排在「烤漆」前面）是刻意的特例覆蓋，較晚規則的其餘關鍵字
    照樣可觸發，不算死規則。同一個 label 的兩條規則也不算真衝突（結果一樣）。
    這是機械可查的死規則，不需要真實資料，跟 test_rule_order_is_the_contract
    那種要人工判斷「該歸哪類」的案例不同。
    """
    L = Labeler.load()
    kws = [(lab, pat.pattern.split("|")) for pat, lab in L.rules]
    dead = []
    for i, (lab_i, kw_i) in enumerate(kws):
        for lab_j, kw_j in kws[i + 1 :]:
            if lab_i == lab_j:
                continue
            for kj in kw_j:
                for ki in kw_i:
                    if ki and kj and ki in kj:
                        dead.append(f"{lab_i}[{ki}] 擋住 {lab_j}[{kj}]")
    assert not dead, "發現永遠不會命中的規則，補一條 test_rule_order_is_the_contract 案例或調整關鍵字：\n" + "\n".join(dead)


def test_class_names_follow_convention():
    """`工種-施作內容`。新增類別時照著寫，不然又會冒出「磁磚_地」那種名字。"""
    labs = {lab for _, lab in Labeler.load().rules}
    odd = {c for c in labs if "-" not in c}
    assert not odd, f"沒照命名規約的類別：{sorted(odd)}"
    assert not {c for c in labs if "_" in c}, "底線是程式設計師的複合鍵，不是中文"


def test_flatten():
    report = {
        "dailyReportInfoId": "r1",
        "constrId": "c1",
        "reportDate": "2026-08-01",
        "status": "SUBMITTED",
        "stage": "F1_TO_TOPPING",
        "constr": {"name": "SITE-A"},
        "createdBy": "u1",
        "pages": [
            {
                "pageSort": 1,
                "contentKind": "WORK_ITEM",
                "content": {
                    "title": "3F 地磚鋪設",
                    "feMeta": {
                        "specKey": "SPEC-1",
                        "location": "3F",
                        "pool": [{"label": "平整度", "on": True}, {"label": "x", "custom": "自填項"}],
                    },
                    "photos": [
                        {
                            "id": "f1",
                            "name": "a.webp",
                            "type": "image/webp",
                            "size": 10,
                            "pathCategory": "daily-report",
                            "url": "http://x/1",
                            "serial": 1,
                            "anno": {
                                "v": 1,
                                "raw": False,
                                "workItem": "地磚鋪貼工程",
                                "wmPos": "TOP_LEFT",
                                "nat": {"w": 400, "h": 300},
                            },
                        }
                    ],
                },
            },
            {
                "pageSort": 2,
                "contentKind": "WORKFORCE",
                "content": {
                    "title": "出工",
                    "items": [
                        {
                            "workforceTradeId": "t1",
                            "tradeName": "鋼筋工",
                            "photos": [
                                {
                                    "id": "f2",
                                    "name": "b.jpg",
                                    "type": "image/jpeg",
                                    "pathCategory": "line-bot/media",
                                }
                            ],
                        }
                    ],
                },
            },
            {
                "pageSort": 3,
                "contentKind": "FREE_CONTENT",
                "content": {"title": "LINE 截圖", "photos": [{"id": "f9"}]},
            },
        ],
    }
    rows = flatten(report)
    assert [r["fileId"] for r in rows] == ["f1", "f2"], "FREE_CONTENT 必須排除"
    r = rows[0]
    assert r["objectKey"] == "daily-report/f1.webp"  # 完整 pathCategory，不可寫死前綴
    assert r["predWorkItem"] == "地磚鋪貼工程" and r["natW"] == 400
    assert r["chipsOn"] == "平整度" and r["chipsCustom"] == "自填項"
    assert r["constrName"] == "SITE-A" and r["createdBy"] == "u1"
    assert rows[1]["objectKey"] == "line-bot/media/f2.jpg"
    assert rows[1]["tradeName"] == "鋼筋工" and rows[1]["source"] == "WORKFORCE"
    # 有 workItem 沒 pv → 只能猜；f2 連 anno 都沒有，那不是「舊版 prompt」，是沒送過
    assert r["promptVersion"] == "v1-guess"
    assert rows[1]["promptVersion"] is None


def test_prompt_version_not_guessed_without_model_output():
    def pv(anno):
        return flatten(
            {"pages": [{"contentKind": "WORK_ITEM", "content": {"photos": [{"id": "x", "anno": anno}]}}]}
        )[0]["promptVersion"]

    assert pv({"pv": "v3-2026-08-17", "workItem": "泥作"}) == "v3-2026-08-17"  # 後端寫的真版本
    assert pv({"workItem": "泥作", "workAction": "粉刷"}) == "v2-guess"
    assert pv({"workItem": "泥作"}) == "v1-guess"
    assert pv({"v": 1, "nat": {"w": 1}}) is None  # 有 anno 但沒經過 Gemini
    assert pv({}) is None


def test_anno_keys_catches_new_frontend_fields():
    """anno 是 additionalProperties:true 的自由 blob，前端加欄位不會通知我們。"""
    report = {
        "pages": [
            {
                "contentKind": "WORK_ITEM",
                "content": {"photos": [{"id": "f1", "anno": {"workItem": "泥作", "pv": "v3-2026-08-17"}}]},
            },
            {
                "contentKind": "WORKFORCE",
                "content": {  # items[] 底下的也要看到
                    "items": [{"photos": [{"id": "f2", "anno": {"新欄位": 1}}]}]
                },
            },
        ]
    }
    keys = anno_keys(report)
    assert keys == {"workItem", "pv", "新欄位"}
    assert keys - KNOWN_ANNO_KEYS == {"新欄位"}


def test_partial_sync_does_not_deactivate_everything():
    """--constr/--limit 只看了母體的一片，不得把沒列到的判成「已消失」。"""
    old = pd.DataFrame({"fileId": ["f1", "f2", "f3"], "active": ["True"] * 3})
    new = pd.DataFrame({"fileId": ["f1"], "active": [True]})

    part = merge_manifest(old, new, partial=True)
    assert set(part.fileId) == {"f1", "f2", "f3"}
    assert list(part.active.astype(str)) == ["True", "True", "True"], part.to_dict()

    full = merge_manifest(old, new, partial=False)  # 全量跑才判定消失
    assert dict(zip(full.fileId, full.active.astype(str))) == {"f1": "True", "f2": "False", "f3": "False"}


def test_index_upsert_keeps_other_reports_watermark():
    """水位線被覆寫成子集的話，下次全量會把所有 detail 重抓一遍。"""
    old = pd.DataFrame({"dailyReportInfoId": ["r1", "r2"], "version": ["1", "5"]})
    fresh = pd.DataFrame({"dailyReportInfoId": ["r1"], "version": ["2"]})
    out = merge_index(old, fresh)
    assert dict(zip(out.dailyReportInfoId, out.version)) == {"r1": "2", "r2": "5"}
    assert len(merge_index(old, pd.DataFrame(columns=old.columns))) == 2  # 空清單不清檔


def test_split_by_site_and_date():
    df = pd.DataFrame(
        {
            "constrId": ["a"] * 10 + ["b"] * 3,
            "reportDate": [f"2026-08-{d:02d}" for d in range(1, 11)] + ["2026-08-01"] * 3,
        }
    )
    days = split_dates(df, 0.2, today="2026-08-31")
    assert days["a"] == ["2026-08-09", "2026-08-10"], days  # 最晚 20% 的天數
    assert days["b"] == [], days  # 只有一天 → 不能全切給測試
    # 同一天不跨組
    assert set(days["a"]).isdisjoint(set(df.reportDate[:8]))

    # 髒的未來日期不得霸佔測試集（實測有 2085-04-20）
    dirty = pd.concat([df, pd.DataFrame({"constrId": ["a"], "reportDate": ["2085-04-20"]})])
    assert split_dates(dirty, 0.2, today="2026-08-31")["a"] == ["2026-08-09", "2026-08-10"]


def test_unclaimed_survives_the_drop_fallback_filter():
    """規則沒認領的照片不進訓練是對的，但要看得到。

    `apply()` 預設會濾掉 fallback，操作台那個「下一條規則的來源」面板因此
    從寫出來就一直顯示 0 張——新工種進來只表現成「張數沒長」，沒人聯想得到。"""
    from labels import Labeler, unclaimed

    lab = Labeler(
        {
            "rules": [{"pattern": "油漆", "label": "油漆-批土塗裝"}],
            "fallback": "其他",
            "drop_fallback": True,
            "min_class_size": 0,
        }
    )
    df = pd.DataFrame(
        {
            "fileId": ["a", "b", "c"],
            "title": ["3F油漆", "棄土坑施作", "木門扇安裝"],
            "reportDate": ["2026-08-01"] * 3,
        }
    )

    assert lab.fallback not in set(lab.apply(df).cls), "apply() 本來就該濾掉，這是前提"
    un = unclaimed(df, lab)
    assert set(un.title) == {"棄土坑施作", "木門扇安裝"}, un.title.tolist()


def test_pending_classes_counts_train_plus_test():
    """入場門檻算的是**全部**照片，不是 train。`apply()` 在切分之前就先砍了，
    所以「差幾張」要拿全部去比——拿 train 去比會少算，讓人以為還很遠。"""
    from labels import Labeler, pending_classes

    lab = Labeler(
        {
            "rules": [{"pattern": "油漆", "label": "油漆"}, {"pattern": "欄杆", "label": "欄杆"}],
            "fallback": "其他",
            "drop_fallback": True,
            "min_class_size": 3,
        }
    )
    df = pd.DataFrame(
        {
            "fileId": list("abcdef"),
            "title": ["3F油漆", "4F油漆", "5F油漆", "1F欄杆", "2F欄杆", "棄土坑"],
            "reportDate": [f"2026-08-0{i}" for i in range(1, 7)],
        }
    )

    p = pending_classes(df, lab)
    assert p.cls.tolist() == ["欄杆"], p  # 油漆 3 張已達標；其他 是 fallback 不算
    assert p.need.iloc[0] == 1 and p.photos.iloc[0] == 2
    assert p.latest.iloc[0] == "2026-08-05"  # 最近一張，判斷是不是還在持續進來

    # 過門檻就該從排隊名單消失（= 自動上線，不用改程式）
    df2 = pd.concat([df, pd.DataFrame([{"fileId": "g", "title": "6F欄杆", "reportDate": "2026-08-07"}])])
    assert not len(pending_classes(df2, lab))


def test_newclass_ignores_generic_words():
    """候選詞是拿來當規則的。規則由上而下第一個命中者勝，
    插一條含「施作」的進去會把既有類別整批攔走，而且不會有人發現。"""
    from collections import Counter

    from newclass import candidates

    fb = Counter({"A區抿石子施作": 10, "B區抿石子施作": 8})
    ok = Counter({"3F油漆施作": 60, "B1區壁磚施作": 30})
    got = [c["term"] for c in candidates(fb, ok, min_photos=12, top=5)]
    assert got == ["抿石子"], got  # 不是「施作」也不是「子施」
    assert not candidates(fb, ok, min_photos=30, top=5), "蓋不到 min_class_size 就別提"


def test_merge_manifest_only_deactivates_same_host():
    """換 API 主機不等於照片消失。2026-08-25 切正式版，dev 那 81 張（60 張已標註）
    在 prod 清單裡查無此人，照舊邏輯會被集體作廢，訓練集無聲少 9%。"""
    from sync import UNKNOWN_HOST, merge_manifest

    old = pd.DataFrame(
        [
            {"fileId": "a", "active": "True", "apiHost": "api-dev.example"},
            {"fileId": "b", "active": "True", "apiHost": "api.example"},
            {"fileId": "c", "active": "True", "apiHost": UNKNOWN_HOST},
        ]
    )
    new = pd.DataFrame([{"fileId": "z", "active": True, "apiHost": "api.example"}])

    out = merge_manifest(old, new, partial=False, api_host="api.example").set_index("fileId")
    assert out.loc["b", "active"] is False or out.loc["b", "active"] == False  # noqa: E712
    assert out.loc["a", "active"] == "True", "別台主機的照片不該被作廢"
    assert out.loc["c", "active"] == "True", "切換前抄的（unknown）不該被作廢"

    # --limit / --constr 只看了母體一片，一張都不准作廢
    out = merge_manifest(old, new, partial=True, api_host="api.example").set_index("fileId")
    assert (out.loc[["a", "b", "c"], "active"] == "True").all()


def _floor_df():
    """兩個工地。稀有類別 R 集中在 a 的最後三天（會整批落進測試側）。"""
    from split import split_dates

    rows = []
    for d in range(1, 16):  # 工地 a：15 天，每天 4 張 → 3 天測試日
        for i in range(4):
            cls = "R" if (d >= 13 and i == 0) else "C"
            rows.append(
                {
                    "constrId": "a",
                    "reportDate": f"2026-08-{d:02d}",
                    "fileId": f"a{d:02d}{i}",
                    "cls": cls,
                    "dataset": "pms",
                }
            )
    for d in range(1, 6):  # 工地 b：5 天，每天 2 張，全是 C
        for i in range(2):
            rows.append(
                {
                    "constrId": "b",
                    "reportDate": f"2026-08-{d:02d}",
                    "fileId": f"b{d:02d}{i}",
                    "cls": "C",
                    "dataset": "pms",
                }
            )
    df = pd.DataFrame(rows)
    days = split_dates(df, 0.2, today="2026-08-31")
    is_test = df.apply(lambda r: r.reportDate in days.get(str(r.constrId), []), axis=1)
    return df, is_test, days


def test_rare_class_floor_moves_whole_days():
    from split import rare_class_floor

    df, is_test, days = _floor_df()
    assert int((df[~is_test].cls == "R").sum()) == 0  # 前提：R 整批被抽進測試側

    out, new_days, moved = rare_class_floor(df, is_test, days, floor=2, max_shrink=1.0, log=lambda *_: None)
    assert int((df[~out].cls == "R").sum()) >= 2, "保底沒把 R 補到門檻"
    # 鐵律三：搬的是整天，不是挑照片。被搬的那幾天不可以還有照片留在測試側
    for m in moved:
        day = (df.constrId.astype(str) == m["constrId"]) & (df.reportDate == m["reportDate"])
        assert not out[day].any(), f"{m['reportDate']} 只搬了一半，同天照片跨組"
        assert m["reportDate"] not in new_days[m["constrId"]]


def test_rare_class_floor_guards():
    from split import rare_class_floor

    df, is_test, days = _floor_df()
    n0 = int(is_test.sum())

    # 護欄一：配額 0 → 一天都不准搬，寧可餓著也不動量尺
    out, _, moved = rare_class_floor(df, is_test, days, floor=99, max_shrink=0.0, log=lambda *_: None)
    assert not moved and int(out.sum()) == n0

    # 護欄二：門檻高到救不動也要停得下來（不能無窮迴圈），且每個工地至少留一天測試日
    out, new_days, _ = rare_class_floor(df, is_test, days, floor=99, max_shrink=1.0, log=lambda *_: None)
    assert all(len(v) >= 1 for v in new_days.values()), new_days


def test_legacy_topup_only_fills_short_classes():
    from split import legacy_topup

    df = pd.DataFrame(
        [{"fileId": f"p{i}", "cls": "C", "dataset": "pms"} for i in range(5)]
        + [{"fileId": f"r{i}", "cls": "R", "dataset": "pms"} for i in range(1)]
        + [{"fileId": f"lc{i}", "cls": "C", "dataset": "legacy"} for i in range(50)]
        + [{"fileId": f"lr{i}", "cls": "R", "dataset": "legacy"} for i in range(50)]
        + [{"fileId": f"lx{i}", "cls": "X", "dataset": "legacy"} for i in range(50)]
    )
    is_test = pd.Series(False, index=df.index)

    out, _ = legacy_topup(df, is_test, floor=4, log=lambda *_: None)
    n = out.groupby(["cls", "dataset"]).size().to_dict()
    assert n.get(("R", "legacy")) == 3, n  # R 有 1 張 → 補到 4
    assert ("C", "legacy") not in n, n  # C 已有 5 張 ≥ 4 → 一張都不補
    assert "X" not in set(out.cls), n  # legacy 專屬的類別不在這裡開，那是 --with-legacy 的事


def test_mask_corners_hits_four_corners():
    im = Image.new("RGB", (1000, 1000), (255, 0, 0))
    out = mask_corners(im)
    for x, y in [(5, 5), (995, 5), (5, 995), (995, 995)]:
        assert out.getpixel((x, y)) == (127, 127, 127), (x, y)
    assert out.getpixel((500, 500)) == (255, 0, 0)  # 中心主體保留


def test_clean_ids_only_takes_sane_dates_from_cutoff(tmp_path, monkeypatch):
    """CLEAN_FROM 之後才不遮；髒的未來日期（reportDate > syncedAt）照樣要遮。"""
    import prepare

    m = tmp_path / "manifest.csv"
    m.write_text(
        "fileId,reportDate,syncedAt\n"
        "old,2026-08-13,2026-08-18T07:26:42+00:00\n"
        "new,2026-08-14,2026-08-18T07:26:42+00:00\n"
        "future,2085-04-20,2026-08-18T07:26:42+00:00\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare.paths, "MANIFEST", m)
    assert prepare.clean_ids() == {"new"}


def test_process_mask_false_keeps_corners(tmp_path):
    import prepare

    src, dst = tmp_path / "a.jpg", tmp_path / "a.out.jpg"
    Image.new("RGB", (100, 100), (255, 0, 0)).save(src)
    prepare.process(src, dst, "report", mask=False)
    assert Image.open(dst).getpixel((2, 2))[0] > 200  # 左上角沒被灰掉
    prepare.process(src, dst, "report", mask=True)
    assert Image.open(dst).getpixel((2, 2))[0] < 200


def test_mask_watermark_covers_qms_box_only():
    """QMS 只遮左下浮水印（實測 x 2%~52% / y 65%~97%），四角不動。"""
    im = Image.new("RGB", (1000, 1000), (255, 0, 0))
    out = mask_watermark(im)
    for x, y in [(20, 650), (520, 970), (300, 800)]:  # 浮水印代表點
        assert out.getpixel((x, y)) == (127, 127, 127), (x, y)
    for x, y in [(5, 5), (995, 5), (995, 995), (700, 500)]:  # 四角與主體保留
        assert out.getpixel((x, y)) == (255, 0, 0), (x, y)


def test_review_overrides_beat_the_title_rule():
    """人工複核是照片層級的真標籤，蓋過 title 推出來的那個。"""
    L = Labeler(CFG)
    df = pd.DataFrame(
        {
            "fileId": ["a", "b", "c"],
            "title": ["15F輕隔間批土", "9F浴室壁磚貼飾", "TEST"],
            "reportDate": ["2026-08-01"] * 3,
        }
    )
    base = L.apply(df, drop_small=False, overrides={})
    assert "c" not in set(base.fileId)  # junk 標題本來就被排掉

    out = L.apply(df, drop_small=False, overrides={"a": "泥作-打底粉光", "c": "木作-天花封板"})
    got = dict(zip(out.fileId, out.cls))
    assert got["a"] == "泥作-打底粉光"  # 蓋掉規則
    assert got["b"] == dict(zip(base.fileId, base.cls))["b"]  # 沒複核的不動
    assert got["c"] == "木作-天花封板"  # 被 junk 排掉的可以救回來


def test_boxes_roundtrip_and_last_write_wins(tmp_path, monkeypatch):
    """人標的框：0~1000 相對原圖，與 Gemini evidence 同格式；append-only 最後一列為準。"""
    import labels

    monkeypatch.setattr(labels.paths, "REVIEW", tmp_path / "review.csv")
    labels.save_review("a", "防水", when="2026-08-19T00:00:00+00:00", boxes=[[100, 200, 300, 400]])
    assert labels.load_boxes() == {"a": [[100, 200, 300, 400]]}
    # 重標成兩個框
    labels.save_review(
        "a", "防水", when="2026-08-19T00:00:01+00:00", boxes=[[0, 0, 500, 500], [500, 500, 900, 900]]
    )
    assert labels.load_boxes()["a"] == [[0, 0, 500, 500], [500, 500, 900, 900]]
    # 標成空的 = 把框清掉（跟「還沒標」不一樣，所以要真的消失）
    labels.save_review("a", "防水", when="2026-08-19T00:00:02+00:00", boxes=None)
    assert labels.load_boxes() == {}
    # 舊檔沒有 box 欄時不能欄位錯位
    assert list(pd.read_csv(tmp_path / "review.csv").columns) == [
        "fileId",
        "cls",
        "note",
        "reviewedAt",
        "box",
    ]


def test_crops_never_leak_from_test_photos(tmp_path, monkeypatch):
    """裁切塊只進 train。母張若在測試集，它的裁切塊等於把答案偷渡到訓練側。"""
    import pytest

    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "photos")):
        pytest.skip("data/ 不在（本機執行期資料，不入 git）")
    import labels
    import split as sp_mod

    monkeypatch.setattr(labels.paths, "REVIEW", tmp_path / "review.csv")
    monkeypatch.setattr(sp_mod.paths, "SPLITS", tmp_path)
    labels.save_review("tr1", "防水", when="2026-08-19T00:00:00+00:00", boxes=[[0, 0, 9, 9]])
    labels.save_review("te1", "防水", when="2026-08-19T00:00:01+00:00", boxes=[[0, 0, 9, 9]])

    df = pd.DataFrame(
        {
            "fileId": ["tr1", "te1"],
            "title": ["B1 防水施作"] * 2,
            "reportDate": ["2026-08-01", "2026-08-02"],
            "constrId": ["c1", "c1"],
        }
    )
    monkeypatch.setattr(sp_mod, "labeled_manifest", lambda **kw: Labeler(CFG).apply(df))
    out = sp_mod.build("t", test_frac=0.5, log=lambda *a: None)
    assert "te1" in out["test"] and "tr1" in out["train"]
    assert "tr1#0" in out["train"], "訓練側的框要進來"
    assert "te1#0" not in out["train"], "測試側的框不得進來"
    assert out["trainCrops"] == 1


def test_orphan_reviews_catches_renamed_classes(tmp_path, monkeypatch):
    """類別改名會讓舊裁決指向不存在的類別，然後被當小類別靜靜丟掉。要出聲。"""
    import labels

    monkeypatch.setattr(labels.paths, "REVIEW", tmp_path / "review.csv")
    L = Labeler(CFG)
    labels.save_review("a", "防水", when="2026-08-19T00:00:00+00:00")
    labels.save_review("b", "設備-電梯停車", when="2026-08-19T00:00:01+00:00")  # 已改名
    assert labels.orphan_reviews(L) == {"b": "設備-電梯停車"}
    # 補一列新名字之後就不再是孤兒（append-only，最後一列為準）
    labels.save_review("b", "防水", when="2026-08-19T00:00:02+00:00")
    assert labels.orphan_reviews(L) == {}


def test_review_csv_roundtrip(tmp_path, monkeypatch):
    """append-only：同一張改過兩次，以最後一次為準，舊列仍留在檔案裡。"""
    import labels

    monkeypatch.setattr(labels.paths, "REVIEW", tmp_path / "review.csv")
    assert labels.load_reviews() == {}  # 檔案不存在 = 沒有覆蓋
    labels.save_review("a", "泥作-打底粉光", "第一次", when="2026-08-18T00:00:00+00:00")
    labels.save_review("a", "油漆-批土塗裝", "看錯了", when="2026-08-18T01:00:00+00:00")
    labels.save_review("b", "", when="2026-08-18T02:00:00+00:00")  # 空的 = 還沒判
    assert labels.load_reviews() == {"a": "油漆-批土塗裝"}
    assert len(pd.read_csv(tmp_path / "review.csv")) == 3  # 舊列沒被洗掉


def test_human_refs_reads_chips_and_speckey():
    """日報裡人寫的兩欄也是參考答案：chipsOn（自打的查驗重點）、specKey（點選的工種）。

    Gemini 的 predWorkItem 不在此列——那是受測者的作答，SPEC §7.2 禁止當考卷答案。
    """
    from labels import human_refs

    L = Labeler(CFG)
    df = pd.DataFrame(
        {
            "chipsOn": ["1.壁磚貼飾是否平整。|2.留縫是否一致。", "1.素地清理是否乾淨。", None],
            "specKey": ["磁磚", None, "油漆"],
        }
    )
    got = human_refs(df, L)
    assert got.clsChips[0] == "壁磚貼飾"  # 查驗重點裡有工種詞 → 標得出來
    assert got.clsChips[1] == L.fallback  # 只有驗收條件 → 沒訊號，不是有異議
    assert pd.isna(got.clsChips[2])  # 沒填 → NaN，不參與比對
    assert got.specTrade[0] == "泥作" and got.specTrade[2] == "油漆"
    assert pd.isna(got.specTrade[1])


def test_dhash_survives_recompression_but_separates_photos():
    """跨來源去重的前提：同一張照片重壓縮後 dhash 仍相同，不同照片不能撞。

    sha1 對重壓縮完全無效（pptx 一定會重編碼），這條就是在守 dhash 有沒有做到。
    """
    import io

    from legacy import dhash

    a = Image.new("RGB", (400, 300))
    for x in range(400):  # 有梯度才有明暗結構可比
        for y in range(0, 300, 30):
            a.paste((x % 256, (x * 2) % 256, y % 256), (x, y, x + 1, y + 30))
    buf = io.BytesIO()
    a.save(buf, "JPEG", quality=30)  # 重壓縮 + 縮小，模擬貼進 pptx
    a2 = Image.open(buf).resize((260, 195))
    assert dhash(a) == dhash(a2)

    b = a.transpose(Image.FLIP_LEFT_RIGHT)  # 不同畫面就該不同
    assert dhash(a) != dhash(b)


def test_legacy_site_alias_matches_pms_names():
    """工地名對不上 PMS 的話，「同工地不跨組」這條鐵律就形同虛設。"""
    import legacy
    from legacy import site_from

    # 別名表外移 reference/（公司資料不入 git）——測試自帶臨時表驗證套用邏輯
    legacy.SITE_ALIAS = {"甲": "甲案", "乙區": "乙案"}
    assert site_from("甲案進度報告115.01.12", "x.pptx") == "甲案"
    assert site_from("乙區進度報告", "x.pptx") == "乙案"
    assert site_from("", "乙區_115.08.10.pptx") == "乙案"
    assert site_from("新工地進度報告", "x.pptx") == "新工地"  # 不在表上：原樣


def test_drop_fallback_removes_the_junk_bag():
    """fallback 不管張數一律排除——它是一袋互不相干的東西，不是一個類別。"""
    df = pd.DataFrame({"title": ["B1 防水施作"] * 3 + ["中庭雨遮吊裝"] * 3, "reportDate": ["2026-08-01"] * 6})
    keep = Labeler({**CFG, "min_class_size": 1}).apply(df, overrides={})
    assert set(keep.cls) == {"防水", "其他"}  # 預設不開，行為不變
    dropped = Labeler({**CFG, "min_class_size": 1, "drop_fallback": True}).apply(df, overrides={})
    assert set(dropped.cls) == {"防水"}


def test_probe_cam_finds_the_block_that_matters():
    """遮擋法：熱區必須落在真正決定答案的那一塊，框要是那一塊、不是整張圖。

    用假的 encoder/分類器（只看左上角有多綠）才測得動——真的 SigLIP 太慢，
    而且這裡要驗的是遮擋與座標換算的邏輯，不是模型好壞。
    """
    import numpy as np

    import explain

    im = Image.new("RGB", (120, 120), (200, 200, 200))
    im.paste(Image.new("RGB", (60, 60), (0, 255, 0)), (0, 0))  # 唯一的綠塊，佔左上 1/4

    class FakeClf:
        classes_ = np.array(["綠", "灰"])

        def predict_proba(self, x):
            return np.stack([x[:, 0], 1 - x[:, 0]], axis=1)

    # embed 回傳 [綠色比例]；explain.embed 會被 monkeypatch 掉，enc 只是佔位
    def fake_embed(enc, ims):
        return np.array([[float((np.asarray(i)[:, :, 1] > 200).mean() * 4)] for i in ims], dtype=np.float32)

    orig, explain.embed = explain.embed, fake_embed
    try:
        cam, pred, _conf, box = explain.probe_cam(im, FakeClf(), None, grid=4, win=2)
    finally:
        explain.embed = orig

    assert pred == "綠"
    assert cam.shape == (4, 4)
    assert cam[0, 0] == 1.0  # 蓋住綠塊的那格掉最多
    assert box == (0.0, 0.0, 0.5, 0.5)  # 框 = 那格的 2×2 視窗
    # 完全碰不到綠塊的格子不該有掉幅
    assert cam[2, 2] == 0.0 and cam[3, 3] == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("全部通過")


# ── core/ 服務層真實性（2026-09-01 接線後的守門）────────────────────────
def test_core_services_are_real_not_stubs():
    """src/ 相容層必須指向 core/ 的真實實作，不許漂回兩份平行邏輯。"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import contractdata
    import labels
    import qsdata
    import review
    from core import contractdata as c_contract
    from core import labeler as c_labeler
    from core import qs_data as c_qs
    from core import review_utils as c_review

    assert labels.Labeler is c_labeler.Labeler
    assert labels.save_review is c_labeler.save_review
    assert review.build is c_review.build and review.scores is c_review.scores
    assert qsdata.load is c_qs.load and qsdata.emit_phases is c_qs.emit_phases
    assert contractdata.load is c_contract.load
    assert contractdata.cross_check is c_contract.cross_check


def test_core_qs_service_loads_real_data():
    """core/qs_data 必須載到 reference/iso 的真檔案（V2.0 草稿期曾是空殼）。

    reference/ 是公司資料、不入 git（2026-09-01 起）——裸 clone 上跳過，
    有資料的機器上這條就是守門。
    """
    import pytest

    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "reference", "iso", "raw")):
        pytest.skip("reference/ 不在（公司資料不入 git）")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from core import qs_data

    docs = qs_data.load()
    assert len(docs) >= 75, f"QS 應 ≥75 份（實得 {len(docs)}）"
    assert docs["QS0907"].required and "電梯" in docs["QS0907"].name  # 編號定案
    it = next(i for i in docs["QS0404"].required if "10~15mm" in i.name)
    assert it.key == "QS0404-4" and it.kind == "B"


def test_core_contract_service_loads_real_data():
    """core/contractdata 必須載到 reference/contract 的真條款（不入 git，缺席則跳過）。"""
    import pytest

    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "reference", "contract", "raw")):
        pytest.skip("reference/ 不在（公司資料不入 git）")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from core import contractdata as cd

    docs = cd.load()
    assert len(docs) >= 4 and len(cd.all_clauses(docs)) >= 100
    st_ = [d for d in docs if "鋼筋" in d.trade]
    assert st_ and st_[0].kind == "物明"  # 物明≠工明，這條判斷不能丟
    assert cd.answers_qs("QS0701-4.13"), "QS 交叉引用必須解析得到條款"


def test_contract_cross_check_and_mappings_smoke():
    """cross_check 走 mappings.yaml（2026-09-01 去識別化外移後曾因殘留引用炸過——
    規範庫 ③ 衝突比對分頁在跑這條路徑，self-check 與 tests 當時都沒蓋到它）。"""
    import pytest

    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "reference", "contract", "raw")):
        pytest.skip("reference/ 不在（公司資料不入 git）")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from core import contractdata as cd

    # 缺 mappings 的降級：查詢回空、不炸
    rs = cd.cross_check()
    assert isinstance(rs, list)
    if cd.load_mappings().get("TRADE_TO_QS"):
        assert len(rs) >= 1, "有映射時應產出衝突候選（本機實測 7 組）"
        r0 = rs[0]
        assert {"trade", "project", "qsDoc", "topic", "qs", "contract"} <= set(r0)
    # answers_qs / interface_clauses 同一條映射鏈
    assert cd.answers_qs("QS0701-4.13") or not cd.load_mappings().get("QS_ANSWERS")
    names = [it["name"] for it in cd.load_mappings().get("INTERFACES", [])]
    for n in names:
        assert cd.interface_clauses(n), f"介面「{n}」應解析得到條款"
