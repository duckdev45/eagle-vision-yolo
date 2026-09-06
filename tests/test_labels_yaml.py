"""labels.yaml 的載重順序與路由回歸測試。

規則是「由上而下第一個命中者勝」——順序本身就是語意。版本註記裡每一條
「必須排在 XX 前面」都是踩過坑換來的（v6 植栽被清潔搶走、v7 輕質磚被灌漿牆
吃掉、v8 外牆磁磚被壁磚子字串接走、v10 烤漆玻璃歸錯油漆、v12 木門被鋁門窗
的「門框」攔截、v13 切削混進連續壁）。這裡把它們釘死：任何人或腳本重排規則，
測試立刻叫。

路由測試用**真實標題**（語料裡出現過的寫法），不是編造的最小案例——
`BFEproxy`（typo）、`天花噴漆`（動詞勝受體）這種坑只有真標題踩得出來。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from labels import Labeler


@pytest.fixture(scope="module")
def lab() -> Labeler:
    return Labeler.load()


def test_rules_compile_and_unique(lab: Labeler) -> None:
    assert lab.rules, "沒有規則"
    seen: set = set()
    for pat, _label in lab.rules:
        assert pat.pattern, "空 pattern"
        pat.search("測試字串")  # 編譯過的 pattern 要能 search
        key = pat.pattern
        assert key not in seen, f"重複 pattern：{key}"
        seen.add(key)


def test_naming_convention(lab: Labeler) -> None:
    """命名規約：`工種-施作內容`，正好一個 dash。fallback「其他」不在此限。"""
    labels = {label for _, label in lab.rules}
    for name in labels:
        assert name.count("-") == 1, f"{name} 不符合 工種-施作內容"


def test_order_specific_before_general(lab: Labeler) -> None:
    """版本註記裡載明的先後關係，一條都不能翻。"""
    order = [label for _, label in lab.rules]
    must_precede = [
        ("泥作-外牆磁磚", "泥作-壁磚貼飾", "v8：外牆磁磚會被壁磚的子字串接走"),
        ("門窗-玻璃", "油漆-批土塗裝", "v10：烤漆玻璃/明鏡曾歸錯油漆"),
        ("輕隔間-輕質磚", "輕隔間-灌漿牆", "v7：輕質磚會被輕隔間子字串吃掉"),
        ("植栽-景觀", "雜項-清潔", "v6：中庭植栽整理的「整理」會被清潔搶走"),
        ("結構-鋼筋", "雜項-清潔", "v6：鋼筋加工場整理"),
        ("防水-嵌縫", "門窗-鋁門窗", "嵌縫 rule：窗框崁縫不能歸鋁門窗"),
        ("門窗-木門", "門窗-鋁門窗", "v12：木門框會被「門框」子字串攔走"),
        ("木作-暗架天花板", "木作-天花封板", "v12：暗架天花板先判"),
        ("木作-公設裝修", "木作-天花封板", "v12：公設木作裝修先判"),
        ("基礎-舊基礎切削", "基礎-連續壁", "v13：切削正名，先判"),
    ]
    for before, after, why in must_precede:
        assert before in order, f"{before} 不在規則裡"
        assert after in order, f"{after} 不在規則裡"
        assert order.index(before) < order.index(after), f"順序翻了：{why}"


@pytest.mark.parametrize(
    "title,expected",
    [
        # v11：大小寫與關鍵字補漏
        ("B3FEpoxy地坪修平", "裝修-耐磨地坪"),
        ("BFEproxy地坪(中塗層)施作", "裝修-耐磨地坪"),
        ("B1F入口金剛砂施作", "裝修-耐磨地坪"),
        ("B1F地坪整平施作", "裝修-耐磨地坪"),
        ("1F中庭板模拆模", "結構-模板"),
        ("15F頂版吊模", "結構-模板"),
        ("8F~9F鋁窗安裝", "門窗-鋁門窗"),
        ("B1F線槽管線清理", "雜項-清潔"),
        ("中庭水溝清理", "雜項-清潔"),
        ("空中花園花土及稻殼施作", "植栽-景觀"),
        ("空中花園喬灌木進場", "植栽-景觀"),
        ("1F中庭噴灌系統施作", "植栽-景觀"),
        ("1F車道貼磚", "泥作-地磚貼飾"),
        ("1F沿街步道磁磚施作", "泥作-地磚貼飾"),
        ("B2F-B3FEpoxy車格線", "油漆-批土塗裝"),
        ("R3F~R2F及13F~11F室內噴漆", "油漆-批土塗裝"),
        ("2~15F防火門安裝", "門窗-住戶大門"),
        # v12：木作拆三類 + 木門
        ("15F暗架天花板施作", "木作-暗架天花板"),
        ("9F廁所及廚房暗架天花板施作", "木作-暗架天花板"),
        ("1F公設木作裝修施作", "木作-公設裝修"),
        ("RF梯廳木作裝修施作", "木作-公設裝修"),
        ("B1F第二門廳木作", "木作-公設裝修"),
        ("27F梯廳木作天花", "木作-天花封板"),
        ("7F小門廳木作天花施作", "木作-天花封板"),
        ("5F木門按裝施作", "門窗-木門"),
        ("15F~2F木門框膠條施作", "門窗-木門"),
        # v13：切削正名
        ("舊基礎切削作業", "基礎-舊基礎切削"),
        ("舊基礎切削施作圖", "基礎-舊基礎切削"),
        ("連續壁機具進場及組裝", "基礎-連續壁"),
        # 動詞勝受體：噴漆/批土/補漆贏過 天花/木作/欄杆/木門
        ("B5F~B1F天花噴漆", "油漆-批土塗裝"),
        ("29F-24F梯廳走道木作天花噴漆", "油漆-批土塗裝"),
        ("6F天花板油漆批土", "油漆-批土塗裝"),
        ("25F梯廳木作天花批土", "油漆-批土塗裝"),
        ("各戶室內木門框補漆", "油漆-批土塗裝"),
        ("B1F車格線缺改", "油漆-批土塗裝"),
        ("B1Fepoxy車格線缺失改善", "油漆-批土塗裝"),
        ("B2FEpoxy地坪(中塗)缺失改善施工", "裝修-耐磨地坪"),
        # 既有基準（v4 時代就在的行為，防手滑）
        ("13F~7F外牆磁磚貼飾", "泥作-外牆磁磚"),
        ("10F浴室壁磚貼飾", "泥作-壁磚貼飾"),
        ("15F浴室防水施作", "防水-防水施作"),
        ("12F廁所及廚房崁縫", "防水-嵌縫"),
        ("22F廚櫃安裝", "裝修-廚具"),
        ("1F大廳公設裝修施作", "其他"),
    ],
)
def test_title_routing(lab: Labeler, title: str, expected: str) -> None:
    got = lab.label(title)
    assert got == expected, f"{title!r} → {got}，預期 {expected}"
