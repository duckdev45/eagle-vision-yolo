"""缺失弱標籤：標題缺失字或人工裁成缺失改善才算正例。"""

import pandas as pd

from defect_probe import weak_labels


def test_weak_labels_title_or_human_verdict():
    df = pd.DataFrame(
        {
            "fileId": ["a", "b", "c", "d"],
            "title": ["泥作缺失改善", "磁磚美容", "1F地磚貼飾", None],
        }
    )
    got = weak_labels(df, {"c": "泥作-地磚貼飾", "d": "雜項-缺失改善"})
    # 泥作缺失改善被 labels.yaml 泥作規則接走成打底，這裡照標題仍算缺失改善
    assert got.tolist() == [1, 1, 0, 1]


def test_human_defect_flag_overrides_title():
    df = pd.DataFrame({"fileId": ["a", "b"], "title": ["泥作缺失改善", "1F地磚貼飾"]})
    got = weak_labels(df, {}, {"a": {"defect": False}, "b": {"defect": True}})
    assert got.tolist() == [0, 1]
