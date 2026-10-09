"""G1 regression tests using synthetic tables and temporary files."""

import pandas as pd
import pytest

import g1_gate
import g1_sample


def trade(rows=()):
    return pd.DataFrame(rows, columns=["fileId", "cls"]).set_index("fileId")


def box(fid, source, pattern="rust"):
    return {"fileId": fid, "source": source, "defectType": pattern, "box": "[[100,100,300,300]]"}


def pool(counts=None):
    counts = counts if counts is not None else {"A": 5, "B": 5, "C": 40}
    return pd.DataFrame(
        [{"fileId": f"{c}{i:03d}", "cls": c} for c, n in counts.items() for i in range(n)],
        columns=["fileId", "cls"],
    )


@pytest.mark.parametrize("n", [15, 30])
def test_perfect_agreement_counts_each_photo_and_pair_once(n):
    a = trade([(f"p{i}", "A") for i in range(n)])
    _, report = g1_gate.trade_gate(a, a.copy())
    per = report["per_class"].set_index("cls").loc["A"]
    assert per["n"] == n and per["一致率"] == 1.0
    assert (per["本輪可評"] == "✅") == (n >= 30)
    df = pd.DataFrame([box(f"p{i}", src) for i in range(n) for src in ("A", "B")])
    _, report = g1_gate.defect_gate(df, "A", "B")
    per = report["per_pattern"].set_index("樣態").loc["rust"]
    assert per["n"] == per["配對框數"] == n
    assert per["一致率"] == 1.0
    assert (per["本輪可評"] == "✅") == (n >= 30)


def test_trade_missing_ids_are_arbitrated_and_count_in_denominator():
    a = trade([("shared", "A"), ("a-only", "A")])
    b = trade([("shared", "A"), ("b-only", "B")])
    rows, report = g1_gate.trade_gate(a, b)
    assert rows.set_index("fileId").kind.to_dict() == {
        "shared": "一致",
        "a-only": "A 有 B 無",
        "b-only": "B 有 A 無",
    }
    assert report["summary"]["total"] == 3 and report["summary"]["pair"] == 1
    assert report["summary"]["one_sided"] == 2
    assert report["summary"]["ceiling"] == pytest.approx(1 / 3, abs=0.001)
    per = report["per_class"].set_index("cls")
    assert per.loc["A", "n"] == 2 and per.loc["A", "一致率"] == 0.5
    assert per.loc["B", "n"] == 1 and per.loc["B", "一致率"] == 0.0


def test_trade_disagreement_counts_each_mentioned_class_once():
    _, report = g1_gate.trade_gate(trade([("x", "A"), ("y", "A")]), trade([("x", "A"), ("y", "B")]))
    per = report["per_class"].set_index("cls")
    assert per.loc["A", "n"] == 2 and per.loc["A", "一致率"] == 0.5
    assert per.loc["B", "n"] == 1 and per.loc["B", "一致率"] == 0.0


@pytest.mark.parametrize("missing", [None, "", "  ", float("nan")])
def test_trade_blank_answers_are_not_agreement(missing):
    rows, report = g1_gate.trade_gate(trade([("x", missing)]), trade([("x", missing)]))
    assert rows.kind.tolist() == ["雙方未標"]
    assert report["summary"]["unlabeled"] == 1 and report["summary"]["agree"] == 0
    assert report["summary"]["ceiling"] == 0.0 and report["per_class"].empty
    rows, _ = g1_gate.trade_gate(trade([("x", "A")]), trade([("x", missing)]))
    assert rows.kind.tolist() == ["A 有 B 無"]


def test_trade_duplicate_id_is_rejected():
    with pytest.raises(ValueError, match="重複"):
        g1_gate.trade_gate(trade([("x", "A"), ("x", "B")]), trade([("x", "A")]))


def test_empty_results_keep_schema_without_claiming_agreement():
    rows, report = g1_gate.trade_gate(trade(), trade())
    assert rows.empty and "kind" in rows
    assert report["per_class"].empty and "n" in report["per_class"]
    assert report["summary"]["ceiling"] is None
    rows, report = g1_gate.defect_gate(pd.DataFrame(), "A", "B")
    assert rows.empty and "kind" in rows
    assert report["per_pattern"].empty and "n" in report["per_pattern"]


@pytest.mark.parametrize("n", [20, 40, 50])
def test_sample_fills_available_capacity(n):
    result = g1_sample.sample_trade(pool(), n=n, floor=4, seed=1)
    assert len(result) == n and result.fileId.is_unique
    assert result.cls.value_counts().min() >= 4
    if n == 40:
        assert result.cls.value_counts().to_dict() == {"C": 30, "A": 5, "B": 5}


def test_sample_rejects_impossible_floor():
    with pytest.raises(ValueError, match="at least 12"):
        g1_sample.sample_trade(pool(), n=10, floor=4, seed=1)
