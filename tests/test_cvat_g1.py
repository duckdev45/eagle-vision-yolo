"""CVAT 橋接與 G1 工具的測試。全部純函式，defects.csv 用 tmp 路径，不碰真資料。"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cvat_import
import paths
from core.defects import append_rows, load_defects
from g1_gate import defect_gate, trade_gate
from g1_sample import pattern_of, sample_trade

FIXTURE_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta/>
  <image id="0" name="11111111-2222-3333-4444-555555555555.jpg" width="1000" height="500">
    <box label="鏽蝕" source="manual" xtl="100" ytl="50" xbr="300" ybr="150"/>
    <box label="裂縫" source="manual" xtl="-50" ytl="0" xbr="20" ybr="10"/>
    <box label="鏽蝕" source="manual" xtl="900" ytl="400" xbr="902" ybr="402"/>
  </image>
  <image id="1" name="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jpg" width="800" height="600">
    <box label="喵喵" source="manual" xtl="0" ytl="0" xbr="100" ybr="100"/>
  </image>
</annotations>
"""


@pytest.fixture()
def tmp_defects(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DEFECTS", tmp_path / "defects.csv")
    yield tmp_path / "defects.csv"


def test_parse_cvat_xml_normalizes_and_clips(tmp_path):
    xml = tmp_path / "tasks.xml"
    xml.write_text(FIXTURE_XML, encoding="utf-8")
    rows = cvat_import.parse_cvat_xml(str(xml))
    # image 0：正常框 0~1000 換算
    first = [r for r in rows if r["fileId"] == "11111111-2222-3333-4444-555555555555"]
    assert first[0] == {
        "fileId": "11111111-2222-3333-4444-555555555555",
        "defectType": "鏽蝕",
        "box": [100, 100, 300, 300],
    }
    # 出界框夾回 0~1000；寬高 <6/1000 的誤點丟掉
    tiny = [r for r in rows if r["fileId"] == "11111111-2222-3333-4444-555555555555" and r["box"][0] == 0]
    assert tiny and tiny[0]["box"] == [0, 0, 20, 20]
    assert all(not (r["box"][2] - r["box"][0] < 6 and r["box"][3] - r["box"][1] < 6) for r in rows)


def test_import_rejects_unknown_label(tmp_defects, tmp_path, capsys):
    xml = tmp_path / "tasks.xml"
    xml.write_text(FIXTURE_XML, encoding="utf-8")
    n = cvat_import.import_xml([str(xml)], annotator="測試", source="cvat-test")
    # image 0 的 鏽蝕/裂縫 是合法樣態（2 框）；「喵喵」拒收並印出來
    assert n == 2
    out = capsys.readouterr().out
    assert "喵喵" in out
    df = load_defects()
    assert sorted(df.defectType) == ["裂縫", "鏽蝕"]
    assert (df.source == "cvat-test").all() and (df.reviewedBy == "測試").all()


def test_defects_append_dedup(tmp_defects):
    row = {
        "fileId": "f1",
        "source": "cvat",
        "defectType": "鏽蝕",
        "box": "[[0,0,100,100]]",
        "reviewedBy": "a",
    }
    assert append_rows([row, dict(row)]) == 1  # 同框重複只進一筆
    assert append_rows([dict(row)]) == 0  # 再匯一次不翻倍
    df = load_defects()
    assert len(df) == 1 and df.iloc[0].reviewedBy == "a"


def test_defect_gate_matching(tmp_defects):
    rows = [
        # 一致：同位置同樣態
        {"fileId": "p1", "source": "g1-A", "defectType": "鏽蝕", "box": "[[100,100,300,300]]"},
        {"fileId": "p1", "source": "g1-B", "defectType": "鏽蝕", "box": "[[110,100,310,300]]"},
        # 樣態歧義：同位置不同樣態
        {"fileId": "p2", "source": "g1-A", "defectType": "鏽蝕", "box": "[[0,0,100,100]]"},
        {"fileId": "p2", "source": "g1-B", "defectType": "裂縫", "box": "[[0,0,100,100]]"},
        # 單邊：B 有 A 無
        {"fileId": "p3", "source": "g1-B", "defectType": "髒污/殘留", "box": "[[200,200,400,400]]"},
    ]
    df = pd.DataFrame(rows)
    out, summary = defect_gate(df, "g1-A", "g1-B")
    kinds = out.kind.value_counts().to_dict()
    assert kinds.get("一致") == 1 and kinds.get("樣態歧義") == 1 and kinds.get("B 有 A 無") == 1
    assert summary["summary"]["agree"] == 1 and summary["summary"]["one_sided"] == 1


def test_trade_gate():
    a = pd.DataFrame({"fileId": ["x", "y", "z"], "cls": ["泥作-打底", "設備-電梯", "泥作-打底"]}).set_index(
        "fileId"
    )
    b = pd.DataFrame(
        {"fileId": ["x", "y", "z"], "cls": ["泥作-打底", "油漆-批土塗裝", "泥作-打底"]}
    ).set_index("fileId")
    rows, rep = trade_gate(a, b)
    assert rep["summary"]["ceiling"] == pytest.approx(2 / 3, abs=1e-3)
    dis = rows[rows.kind == "分歧"]
    assert list(dis.fileId) == ["y"] and list(dis.A) == ["設備-電梯"] and list(dis.B) == ["油漆-批土塗裝"]


def test_pattern_of_order():
    assert pattern_of("磁磚地板有裂縫") == "裂縫"  # 裂縫先於縫隙
    assert pattern_of("窗框刮傷") == "刮傷/撞痕"
    assert pattern_of("牆面滲水痕跡") == "滲水/水痕"
    assert pattern_of("正常完成面") is None


def test_sample_trade_floor():
    # 3 類：兩類 5 張、一類 40 張 → floor 4 全類有份；剩餘名額按類別比例補到 20
    rows = []
    for i in range(5):
        rows.append({"fileId": f"a{i}", "cls": "A類"})
        rows.append({"fileId": f"b{i}", "cls": "B類"})
    for i in range(40):
        rows.append({"fileId": f"c{i}", "cls": "C類"})
    pool = pd.DataFrame(rows)
    sel = sample_trade(pool, n=20, floor=4, seed=1)
    vc = sel.cls.value_counts()
    # floor 4×3=12，rest 8 按比例（5/50→1、5/50→1、40/50→6）
    assert vc["A類"] == 5 and vc["B類"] == 5 and vc["C類"] == 10
    assert len(sel) == 20 and sel.fileId.is_unique
