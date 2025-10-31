import pytest
from schema import get_schema, validate_json, assert_valid

def _valid_payload():
    return {
        "overallScore": 87,
        "technicalSkillsScore": 100,
        "experienceScore": 90,
        "culturalFitScore": 57,
        "matchSummary": "Alignment on 2/2 core tools; experience ≈4.2 vs 1+ requirement.",
        "strengthsHighlights": [
            "Coverage across key tools with explicit skill evidence.j",
            "Meets experience bar.",
        ],
        "improvementAreas": ["Increase quantification of impact to strengthen seniority signal."],
        "detailedBreakdown": {
            "technicalSkills": [
                {
                    "requirement": "Proficiency in Six Sigma",
                    "present": True,
                    "evidence": "Skills list includes Six Sigma",
                    "gapPercentage": 0,
                    "missingDetail": "0% gap because requirement fully met with explicit evidence.",
                },
                {
                    "requirement": "Proficiency in Lean",
                    "present": True,
                    "evidence": "Skills list includes Lean",
                    "gapPercentage": 0,
                    "missingDetail": "0% gap because requirement fully met with explicit evidence.",
                },
            ],
            "experience": [
                {
                    "requirement": "1+ years of relevant experience",
                    "present": True,
                    "evidence": "Total years ≈ 4.2 via roles listed",
                    "gapPercentage": 0,
                    "missingDetail": "0% gap because requirement satisfied.",
                }
            ],
            "educationAndCertifications": [
                {
                    "requirement": "Formal education or equivalent experience",
                    "present": True,
                    "evidence": "Experience considered sufficient",
                    "gapPercentage": 0,
                    "missingDetail": "0% gap.",
                }
            ],
            "culturalFitAndSoftSkills": [
                {
                    "requirement": "Cross-functional collaboration",
                    "present": True,
                    "evidence": "Worked with product/ops/finance/design",
                    "gapPercentage": 0,
                    "missingDetail": "0% gap because collaboration evidenced.",
                }
            ],
        },
        "redFlags": [
            {
                "issue": "Seniority mismatch",
                "evidence": "Years ≈ 0.8 (< 3+ req)",
                "reason": "Limited autonomy risk.",
            }
        ],
    }


def test_schema_object_shape():
    schema = get_schema()
    assert schema["type"] == "object"
    assert "properties" in schema


def test_valid_payload_passes():
    ok, errs = validate_json(_valid_payload())
    assert ok, f"Expected valid payload, got errors: {errs}"


def test_score_bounds_enforced():
    bad = _valid_payload()
    bad["overallScore"] = 101  # out of range
    ok, errs = validate_json(bad)
    assert not ok
    assert any(
        ("greater than the maximum" in e and "100" in e)
        or ("maximum" in e and "100" in e)
        for e in errs
    ), f"Unexpected error messages: {errs}"


def test_score_type_enforced():
    bad = _valid_payload()
    bad["experienceScore"] = "90"  # wrong type
    ok, errs = validate_json(bad)
    assert not ok and any("is not of type 'integer'" in e for e in errs)


def test_breakdown_item_requires_fields():
    bad = _valid_payload()
    del bad["detailedBreakdown"]["technicalSkills"][0]["evidence"]
    ok, errs = validate_json(bad)
    assert not ok and any("'evidence' is a required property" in e for e in errs)


def test_assert_valid_raises_on_invalid():
    bad = _valid_payload()
    bad["overallScore"] = -1
    with pytest.raises(AssertionError):
        assert_valid(bad)
