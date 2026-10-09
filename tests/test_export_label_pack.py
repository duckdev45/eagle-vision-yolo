"""判準包匯出的去識別化保證（docs/DATA-BOUNDARY.md 是白名單，這裡是它的回歸測試）。

這支測試的立場：**寧可整份拒絕匯出，也不要靜默清洗**。每一條都對應那份文件裡的一條紅線，
文件改了這裡要跟著改——不然紅線就只是文字。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

import export_label_pack as pack
import paths
from core.labeler import Labeler

# 毒餌：這些值一個都不該出現在包裡（案場名、人名、自由文字、QS 原文）
POISON = ["翠屏三", "王大明", "13F外牆打底粉光", "粉刷面應平整不得有裂縫"]


@pytest.fixture(autouse=True)
def _isolate_qs(tmp_path, monkeypatch):
    """QS 的 `RAW_DIR` 是 import 時定下的，會指到真的 `reference/`。

    測試一律指向空目錄，否則 import 順序一變，這支測試就改成在掃公司真資料
    （實測：單獨跑過、全套跑就炸）。要驗 QS 欄位的那支測試自己餵合成 tsv。
    """
    from core import qs_data

    monkeypatch.setattr(qs_data, "RAW_DIR", tmp_path / "no-reference")


def _seed_reviews() -> None:
    """兩筆人審裁決：一筆與規則一致、一筆人改過；note 欄塞毒餌。"""
    pd.DataFrame(
        [
            {"fileId": "a", "cls": "泥作-打底", "note": POISON[2], "reviewedAt": "x", "box": ""},
            {
                "fileId": "b",
                "cls": "泥作-打底",  # 規則判油漆，人改成打底 → humanOverrode=1
                "note": POISON[1],
                "reviewedAt": "x",
                "box": "[[10,10,200,200]]",
            },
        ]
    ).to_csv(paths.REVIEW, index=False)


def _export(tmp_path, **kw):
    return pack.export_pack("p1", tmp_path / "pack", log=lambda *a: None, **kw)


def test_boundary_doc_is_the_whitelist(pms_env):
    spec = pack.load_boundary()
    assert spec["schemaVersion"] == pack.PACK_SCHEMA
    assert set(spec["tables"]) == {"judgements", "defect_boxes", "qs_criteria"}
    assert len(spec["_sha256"]) == 64
    # 禁用欄位與白名單不得交集（文件自我矛盾時 load 就該擋）
    for table in spec["tables"].values():
        assert not set(table["columns"]) & set(spec["forbiddenColumns"])


def test_no_boundary_doc_means_no_export(pms_env, tmp_path):
    missing = tmp_path / "nope.md"
    with pytest.raises(FileNotFoundError):
        pack.load_boundary(missing)


def test_pack_columns_match_the_whitelist_exactly(pms_env, tmp_path):
    _seed_reviews()
    out = _export(tmp_path)
    spec = pack.load_boundary()
    for name, table in spec["tables"].items():
        got = list(pd.read_csv(out / f"{name}.csv").columns)
        assert got == list(table["columns"]), f"{name} 欄位與白名單不符"


def test_file_ids_and_free_text_never_ship(pms_env, tmp_path):
    _seed_reviews()
    out = _export(tmp_path)
    blob = "\n".join(p.read_text(encoding="utf-8") for p in sorted(out.rglob("*")) if p.is_file())
    for poison in POISON:
        assert poison not in blob, f"毒餌外流：{poison}"
    # fileId 逐格比對（不用 substring：單字母 id 會與 enum 值互撞，驗不準）
    real_ids = {"a", "b", "u1", "u2", "c", "d", "inactive", "future", "foreign"}
    for csv_path in out.glob("*.csv"):
        table = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        for col in table.columns:
            assert not (set(table[col]) & real_ids), f"{csv_path.name}:{col} 出現原始 fileId"
    # 案場只以包內序號出現，真的 constrId 留在 sidecar
    assert "site-001" in blob
    assert "s1" not in blob
    sidecar = json.loads((out.with_name(out.name + ".local.json")).read_text(encoding="utf-8"))
    assert sidecar["siteBuckets"]["site-001"] == "s1"


def test_qs_criteria_ships_codes_but_never_the_standard_text(pms_env, tmp_path, monkeypatch):
    """QS 條號是路由鍵（可外售），查驗項名稱是標準原文（絕不外售）。"""
    from core import qs_data

    raw = tmp_path / "qs-raw"
    raw.mkdir()
    (raw / "qs04.tsv").write_text(
        "\n".join(
            [
                "#DOC\tQS0402\tuuid-1\t泥作粉刷工程",
                f"1\tO\t{POISON[3]}",  # 階段節點，名稱是原文
                "1.1\tR\t表面應平整，且須拍照存證做為請款之憑證",
                "2\tR\t未依規定者按日扣款二倍工資",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(qs_data, "RAW_DIR", raw)
    _seed_reviews()
    out = _export(tmp_path)
    rows = pd.read_csv(out / "qs_criteria.csv", dtype=str, keep_default_na=False)
    assert list(rows.qsKey) == ["QS0402-1", "QS0402-1.1", "QS0402-2"]
    assert set(rows.status) == {"o", "r"}
    assert list(rows.isBilling) == ["0", "1", "0"]  # 「拍照存證…請款之憑證」＝請款靶
    assert list(rows.isPenalty) == ["0", "0", "1"]  # 「扣款二倍工資」＝罰則
    blob = "\n".join(p.read_text(encoding="utf-8") for p in sorted(out.rglob("*")) if p.is_file())
    for sentence in (POISON[3], "表面應平整", "二倍工資", "泥作粉刷工程"):
        assert sentence not in blob, f"標準原文外流：{sentence}"


def test_sample_id_joins_inside_the_pack_but_is_salted(pms_env, tmp_path):
    _seed_reviews()
    out = _export(tmp_path)
    rows = pd.read_csv(out / "judgements.csv", dtype=str)
    assert len(rows) == 2
    assert rows.sampleId.str.fullmatch(r"[0-9a-f]{16}").all()
    assert rows.sampleId.nunique() == 2
    sidecar = json.loads((out.with_name(out.name + ".local.json")).read_text(encoding="utf-8"))
    assert len(sidecar["salt"]) == 32
    # 同一包內 sampleId 必須可重現（鹽沿用 sidecar），否則裁決與框 join 不起來
    assert pack._sample_id("a", sidecar["salt"]) in set(rows.sampleId)


def test_human_override_is_the_product(pms_env, tmp_path):
    _seed_reviews()
    out = _export(tmp_path)
    rows = pd.read_csv(out / "judgements.csv", dtype=str)
    assert set(rows.humanOverrode) == {"0", "1"}  # a 與規則一致、b 是人改的
    overridden = rows[rows.humanOverrode == "1"]
    assert len(overridden) == 1
    assert overridden.iloc[0].humanClass == "泥作-打底"
    assert overridden.iloc[0].ruleClass == "油漆-塗裝"  # 機器原本的答案也要留，對照才有價值
    assert overridden.iloc[0].hasBox == "1"
    # 包的厚度要寫在 manifest：人改幾筆就是幾筆，敘事不能靠感覺
    breakdown = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["judgementBreakdown"]
    assert breakdown == {"ruleSilent": 0, "agree": 1, "overridden": 1}


def test_orphan_verdicts_are_dropped_not_exported(pms_env, tmp_path):
    pd.DataFrame(
        [
            {"fileId": "a", "cls": "泥作-打底", "note": "", "reviewedAt": "x", "box": ""},
            {"fileId": "b", "cls": "已經不存在的類別", "note": "", "reviewedAt": "x", "box": ""},
        ]
    ).to_csv(paths.REVIEW, index=False)
    out = _export(tmp_path)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["judgements"] == 1
    assert manifest["stats"]["skippedOrphanClass"] == 1


def test_defect_boxes_only_take_known_patterns(pms_env, tmp_path):
    _seed_reviews()
    pd.DataFrame(
        [
            {
                "fileId": "a",
                "source": "cvat",
                "defectType": "裂縫",
                "qsCode": "QS0402-7",
                "box": "[[1,2,3,4]]",
            },
            {"fileId": "a", "source": "cvat", "defectType": POISON[3], "qsCode": "", "box": "[[5,6,7,8]]"},
            {
                "fileId": "b",
                "source": "cvat",
                "defectType": "髒污/殘留",
                "qsCode": "",
                "box": "[[9,9,99,99]]",
            },
        ]
    ).to_csv(paths.DEFECTS, index=False)
    out = _export(tmp_path)
    rows = pd.read_csv(out / "defect_boxes.csv", dtype=str, keep_default_na=False)
    assert len(rows) == 2  # 不認得的樣態（這裡塞的是標準原文）整列不外售
    assert set(rows.defectPattern) == {"裂縫", "髒污/殘留"}
    assert list(rows.qsCode) == ["QS0402-7", ""]
    assert json.loads((out / "manifest.json").read_text())["stats"]["skippedBoxes"] == 1
    # 框與裁決要能在包內 join（這是買方拿到東西能用的前提）
    judgements = pd.read_csv(out / "judgements.csv", dtype=str)
    assert set(rows.sampleId) <= set(judgements.sampleId)


def test_out_of_range_box_refuses_the_whole_export(pms_env, tmp_path):
    """座標超界＝上游 CVAT 匯入壞了。這種情況要整份拒絕，不是悄悄夾帶出去。"""
    _seed_reviews()
    pd.DataFrame(
        [{"fileId": "a", "source": "cvat", "defectType": "裂縫", "qsCode": "", "box": "[[0,0,9999,9]]"}]
    ).to_csv(paths.DEFECTS, index=False)
    with pytest.raises(ValueError):
        _export(tmp_path)
    assert not (tmp_path / "pack").exists()  # 失敗不留半成品


def test_validator_refuses_the_whole_export_on_a_bad_cell(pms_env):
    spec = pack.load_boundary()
    allowed = {"taxonomy": pack.taxonomy(Labeler.load()), "defectPattern": {"裂縫"}}
    good = {
        "sampleId": "0" * 16,
        "siteBucket": "site-001",
        "yearMonth": "2026-08",
        "humanClass": "泥作-打底",
        "ruleClass": "",
        "humanOverrode": "0",
        "hasBox": "0",
    }
    pack.validate_rows("judgements", [good], spec, allowed)  # 基準線：這列是合法的
    for col, bad in [
        ("sampleId", "a"),  # 不是 16 碼雜湊
        ("siteBucket", "翠屏三"),  # 真案場名
        ("yearMonth", "2026-08-04"),  # 給到日＝可回推單一日報
        ("humanClass", "不在分類樹裡"),
        ("hasBox", ""),  # 結構欄位不許空
    ]:
        with pytest.raises(ValueError):
            pack.validate_rows("judgements", [{**good, col: bad}], spec, allowed)
    with pytest.raises(ValueError):  # 多塞一個白名單外的欄位
        pack.validate_rows("judgements", [{**good, "fileId": "a"}], spec, allowed)


def test_error_message_never_echoes_the_offending_value(pms_env):
    """驗證器的訊息自己也不能洩漏——印出值等於把公司資料寫進 log。"""
    spec = pack.load_boundary()
    allowed = {"taxonomy": {"泥作-打底"}, "defectPattern": {"裂縫"}}
    row = {
        "sampleId": "0" * 16,
        "siteBucket": "site-001",
        "yearMonth": "2026-08",
        "humanClass": POISON[0],
        "ruleClass": "",
        "humanOverrode": "0",
        "hasBox": "0",
    }
    with pytest.raises(ValueError) as err:
        pack.validate_rows("judgements", [row], spec, allowed)
    assert POISON[0] not in str(err.value)


def test_distribution_defaults_to_internal_only(pms_env, tmp_path):
    _seed_reviews()
    out = _export(tmp_path)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["distribution"] == "internal-only"  # 授權沒簽就不是可外送的包
    assert manifest["contains"] == {
        "photos": False,
        "fileIds": False,
        "freeText": False,
        "standardText": False,
    }
    assert manifest["boundaryDoc"]["sha256"] == pack.load_boundary()["_sha256"]
    assert {f["name"] for f in manifest["files"]} == {
        "judgements.csv",
        "defect_boxes.csv",
        "qs_criteria.csv",
        "taxonomy.json",
        "README.md",
    }


def test_licensed_flag_is_explicit(pms_env, tmp_path):
    _seed_reviews()
    out = _export(tmp_path, licensed=True)
    assert json.loads((out / "manifest.json").read_text())["distribution"] == "licensed-external"
    assert "licensed-external" in (out / "README.md").read_text(encoding="utf-8")


def test_taxonomy_json_ships_labels_but_not_the_rule_file(pms_env, tmp_path):
    _seed_reviews()
    out = _export(tmp_path)
    tax = json.loads((out / "taxonomy.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(paths.LABELS_YAML.read_text(encoding="utf-8"))
    assert tax["labelsVersion"] == cfg["version"]
    assert "泥作-打底" in tax["tradeClasses"]
    assert set(tax) == {"labelsVersion", "tradeClasses", "defectPatterns", "buttonOnlyPatterns"}
    blob = json.dumps(tax, ensure_ascii=False)
    # 規則本體不外售：pattern 是日報用語語料，版本註記裡還有案場觀察
    assert "rules" not in tax
    assert "pattern" not in blob
    assert paths.LABELS_YAML.read_text(encoding="utf-8") not in blob


def test_export_refuses_to_overwrite(pms_env, tmp_path):
    _seed_reviews()
    out = _export(tmp_path)
    with pytest.raises(FileExistsError):
        pack.export_pack("p1", out, log=lambda *a: None)


def test_no_human_review_means_no_pack(pms_env, tmp_path):
    with pytest.raises(ValueError):
        _export(tmp_path)  # review.csv 不存在＝沒有人審，判準包不該憑空產生
