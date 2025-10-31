import json
import pytest

from pipeline import run_pipeline, PipelineConfig


class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("m", (), {"content": content})


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        class _Chat:
            def __init__(self, outer):
                class _Comps:
                    def __init__(self, outer):
                        self._outer = outer
                    def create(self, *args, **kwargs):
                        if not outer._responses:
                            raise RuntimeError("no more fake responses")
                        return _FakeResp(outer._responses.pop(0))
                self.completions = _Comps(outer)
        self.chat = _Chat(self)


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


def _resume_text():
    return (
        "Program Manager at Crestel Systems (2017-05 to 2019-11)\n"
        "Delivered 5 projects using Scheduling, ERP, Oracle with measurable KPIs.\n"
        "Collaborated with 10 stakeholders to ship on schedule.\n\n"
        "Associate Program Manager at Lumena Group (2019-06 to 2021-03)\n"
        "Delivered 5 projects using Project Planning, SAP, Lean with measurable KPIs.\n"
        "Collaborated with 7 stakeholders to ship on schedule. Six Sigma exposure.\n"
    )


def test_happy_path_valid_json(monkeypatch):
    # Fake LLM returns already-valid JSON
    valid_json = json.dumps({
        "overallScore": 88,
        "technicalSkillsScore": 100,
        "experienceScore": 90,
        "culturalFitScore": 70,
        "matchSummary": "ok",
        "strengthsHighlights": ["A"],
        "improvementAreas": ["B"],
        "detailedBreakdown": {
            "technicalSkills": [],
            "experience": [],
            "educationAndCertifications": [],
            "culturalFitAndSoftSkills": []
        }
    })
    fake = _FakeClient([valid_json])
    out = run_pipeline(_jd(), _resume_text(), cfg=PipelineConfig(k=2, model="dummy"), client=fake)
    assert isinstance(out, dict)
    assert all(k in out for k in ["overallScore", "technicalSkillsScore"])  # schema keys present


def test_invalid_then_repaired(monkeypatch):
    # First return invalid JSON (missing keys), then a repaired valid JSON
    bad = json.dumps({"overallScore": 50})
    repaired = json.dumps({
        "overallScore": 80,
        "technicalSkillsScore": 90,
        "experienceScore": 85,
        "culturalFitScore": 65,
        "matchSummary": "ok",
        "strengthsHighlights": ["A"],
        "improvementAreas": ["B"],
        "detailedBreakdown": {
            "technicalSkills": [],
            "experience": [],
            "educationAndCertifications": [],
            "culturalFitAndSoftSkills": []
        }
    })
    fake = _FakeClient([bad, repaired])
    out = run_pipeline(_jd(), _resume_text(), cfg=PipelineConfig(k=2, model="dummy"), client=fake)
    assert out["overallScore"] == 80


def test_double_failure_falls_back_to_rule_based():
    # Two junk responses trigger fallback to rule-based (which should be schema-valid)
    junk1 = "not json"
    junk2 = json.dumps({"not": "valid per schema"})
    fake = _FakeClient([junk1, junk2])
    out = run_pipeline(_jd(), _resume_text(), cfg=PipelineConfig(k=2, model="dummy"), client=fake)
    # We can't assert exact numbers, but it must be schema-like
    assert isinstance(out, dict)
    assert all(k in out for k in ["overallScore", "technicalSkillsScore", "experienceScore", "culturalFitScore"])  # schema keys present

