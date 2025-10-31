import math
import pytest

from retrieve import build_resume_collection, retrieve_for_requirements


_RESUME_LINES = [
    "Delivered 5 projects using Scheduling, ERP, Oracle with measurable KPIs.",
    "Improved reliability by 30%.",
    "Collaborated with 10 stakeholders to ship on schedule.",
    "Delivered 5 projects using Project Planning, SAP, Lean with measurable KPIs.",
    "Collaborated with 7 stakeholders to ship on schedule. Six Sigma exposure.",
]

_REQS = ["Proficiency in Six Sigma", "Proficiency in Lean"]


def _requirements_to_queries(reqs):
    # Minimal mirror of parse logic in tests: only the literal requirement strings are used here
    return reqs


def test_build_and_count():
    client, coll = build_resume_collection(_RESUME_LINES, collection_name="t_count")
    assert coll.count() == len(set([ln.strip() for ln in _RESUME_LINES]))


def test_retrieve_contains_expected_tokens():
    client, coll = build_resume_collection(_RESUME_LINES, collection_name="t_tokens")
    out = retrieve_for_requirements(coll, _requirements_to_queries(_REQS), k=3)

    # For "Lean", at least one retrieved doc should include "Lean"
    lean_docs = out["Proficiency in Lean"]
    assert any("lean" in item["text"].lower() for item in lean_docs)

    # For "Six Sigma", token should appear in at least one doc
    six_docs = out["Proficiency in Six Sigma"]
    assert any("six sigma" in item["text"].lower() for item in six_docs)


def test_k_bound():
    client, coll = build_resume_collection(_RESUME_LINES, collection_name="t_k")
    out2 = retrieve_for_requirements(coll, _requirements_to_queries(_REQS), k=2)
    assert len(out2["Proficiency in Lean"]) == 2


def test_determinism_same_inputs_same_results():
    client, coll = build_resume_collection(_RESUME_LINES, collection_name="t_det")
    out1 = retrieve_for_requirements(coll, _requirements_to_queries(_REQS), k=3)
    out2 = retrieve_for_requirements(coll, _requirements_to_queries(_REQS), k=3)

    # IDs should match exactly
    for req in _REQS:
        ids1 = [x["id"] for x in out1[req]]
        ids2 = [x["id"] for x in out2[req]]
        assert ids1 == ids2
        # Distances should match to a small tolerance
        d1 = [round(x["distance"], 8) for x in out1[req]]
        d2 = [round(x["distance"], 8) for x in out2[req]]
        assert d1 == d2
