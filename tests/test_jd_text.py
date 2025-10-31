from jd_text import parse_job_text

_SAMPLE = """
Program Manager
We are hiring a Program Manager in Operations & Supply Chain to drive outcomes.
Location: Hybrid in Cairo.
Requirements:
- Proficiency in Six Sigma
- Proficiency in Lean
At least 1+ years of relevant experience. Bachelor's degree preferred.
""".strip()


def test_parse_job_text_basic_fields():
    jd = parse_job_text(_SAMPLE)
    assert jd["title"] == "Program Manager"
    assert jd["sector"] in ("Operations & Supply Chain", "Unknown")
    assert isinstance(jd["description"], str) and len(jd["description"]) > 0
    # location heuristic
    assert jd["location"].startswith("Hybrid") or jd["location"] in ("Cairo, Egypt", "Unknown")


def test_parse_job_text_requirements_extracted_and_sorted():
    jd = parse_job_text(_SAMPLE)
    reqs = jd["requirements"]
    assert any(r.lower().startswith("proficiency in six sigma") for r in reqs)
    assert any(r.lower().startswith("proficiency in lean") for r in reqs)
    assert any(r.endswith("+ years of relevant experience") for r in reqs)
    assert reqs == sorted(reqs, key=lambda x: x.lower())
