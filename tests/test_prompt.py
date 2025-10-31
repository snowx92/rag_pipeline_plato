import json
import pytest

from prompt import build_prompt
from schema import get_schema


def _sample_jd():
    return {
        "title": "Program Manager",
        "sector": "Operations & Supply Chain",
        "location": "Hybrid – Cairo",
        "description": "We are hiring a Program Manager...",
        "requirements": [
            "Proficiency in Six Sigma",
            "Proficiency in Lean",
            "1+ years of relevant experience",
        ],
    }


def _sample_parsed():
    return {
        "skills": ["Six Sigma", "Lean"],
        "experience_years": 4.2,
        "evidence_lines": [
            "Delivered 5 projects using Project Planning, SAP, Lean with measurable KPIs.",
            "Collaborated with 7 stakeholders to ship on schedule. Six Sigma exposure.",
        ],
    }


def _sample_retrieval():
    return {
        "Proficiency in Lean": [
            {"id": "res-0003", "text": "... SAP, Lean with measurable KPIs.", "distance": 0.123456789, "metadata": {"index": 3}},
            {"id": "res-0000", "text": "Delivered 5 projects using Scheduling, ERP, Oracle.", "distance": 0.234567891, "metadata": {"index": 0}},
        ],
        "Proficiency in Six Sigma": [
            {"id": "res-0004", "text": "... Six Sigma exposure.", "distance": 0.111111119, "metadata": {"index": 4}},
        ],
    }


def test_prompt_contains_schema_and_sections():
    prompt = build_prompt(_sample_jd(), _sample_parsed(), _sample_retrieval(), get_schema())
    # Must contain all section headers
    for header in ["SYSTEM:", "JSON_SCHEMA:", "JOB:", "PARSED_RESUME:", "RETRIEVAL:", "TASK:"]:
        assert header in prompt
    # Schema keys present
    assert "overallScore" in prompt
    assert "technicalSkillsScore" in prompt


def test_prompt_is_deterministic():
    p1 = build_prompt(_sample_jd(), _sample_parsed(), _sample_retrieval(), get_schema())
    p2 = build_prompt(_sample_jd(), _sample_parsed(), _sample_retrieval(), get_schema())
    assert p1 == p2


def test_prompt_orders_requirements_and_hits_deterministically():
    # Shuffle requirements to ensure sorting works
    jd = _sample_jd()
    jd["requirements"] = [
        "1+ years of relevant experience",
        "Proficiency in Lean",
        "Proficiency in Six Sigma",
    ]
    prompt = build_prompt(jd, _sample_parsed(), _sample_retrieval(), get_schema())
    # Requirements should appear sorted by casefolded string (1+..., Lean, Six Sigma)
    reqs_order = prompt.split("JOB:\n", 1)[1].split("}\n\nPARSED_RESUME:", 1)[0]
    # Simple checks for ordering by index within the JOB JSON
    assert reqs_order.index("1+ years of relevant experience") < reqs_order.index("Proficiency in Lean")
    assert reqs_order.index("Proficiency in Lean") < reqs_order.index("Proficiency in Six Sigma")


def test_prompt_includes_retrieval_hits_sorted_by_distance_then_id():
    # Note distances in sample_retrieval, Lean hits are 0.123456789 (id res-0003) then 0.234567891 (id res-0000)
    prompt = build_prompt(_sample_jd(), _sample_parsed(), _sample_retrieval(), get_schema())
    retrieval_block = prompt.split("RETRIEVAL:\n", 1)[1].split("\n\nTASK:", 1)[0]
    first_idx = retrieval_block.index("res-0003")
    second_idx = retrieval_block.index("res-0000")
    assert first_idx < second_idx