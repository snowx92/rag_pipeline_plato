import pytest

from scorer import score_rule_based
from schema import validate_json


def _jd():
    return {
        "title": "Program Manager",
        "sector": "Operations & Supply Chain",
        "location": "Hybrid – Cairo",
        "description": "We are hiring a PM...",
        "requirements": [
            "Proficiency in Six Sigma",
            "Proficiency in Lean",
            "1+ years of relevant experience",
        ],
    }


def _parsed_has_both():
    return {
        "skills": ["Six Sigma", "Lean"],
        "experience_years": 4.2,
        "evidence_lines": [
            "Collaborated with 7 stakeholders to ship on schedule. Six Sigma exposure.",
            "Delivered 5 projects using SAP, Lean with measurable KPIs.",
        ],
    }


def _parsed_missing_one():
    return {
        "skills": ["Six Sigma"],
        "experience_years": 4.2,
        "evidence_lines": [
            "Collaborated with stakeholders; delivered projects with measurable KPIs.",
        ],
    }


def _retrieval_dummy():
    return {
        "Proficiency in Six Sigma": [{"id": "res-0004", "text": "Six Sigma exposure.", "distance": 0.1, "metadata": {}}],
        "Proficiency in Lean": [{"id": "res-0003", "text": "... SAP, Lean with measurable KPIs.", "distance": 0.2, "metadata": {}}],
    }


def test_schema_valid_when_both_skills_present():
    out = score_rule_based(_jd(), _parsed_has_both(), _retrieval_dummy())
    ok, errs = validate_json(out)
    assert ok, errs
    assert out["technicalSkillsScore"] == 100
    assert out["experienceScore"] >= 80  # rough floor
    assert out["culturalFitScore"] >= 60


def test_tech_score_halves_when_one_skill_missing():
    out = score_rule_based(_jd(), _parsed_missing_one(), _retrieval_dummy())
    assert out["technicalSkillsScore"] == 50


def test_deterministic_same_inputs_same_output():
    out1 = score_rule_based(_jd(), _parsed_has_both(), _retrieval_dummy())
    out2 = score_rule_based(_jd(), _parsed_has_both(), _retrieval_dummy())
    assert out1 == out2
