import pytest
from parse_resume import extract_required_skills_from_jd, parse_resume

def _jd_requirements():
    return [
        "Proficiency in Six Sigma",
        "Proficiency in Lean",
        "1+ years of relevant experience",
        "Evidence of ownership and collaboration with stakeholders",
    ]

def test_extract_required_skills_from_jd_simple():
    skills = extract_required_skills_from_jd(_jd_requirements())
    assert skills == ["Lean", "Six Sigma"]

def test_parse_resume_matches_skills_and_evidence():
    resume = """
    Program Manager at Crestel Systems (2017-05 to 2019-11)
    Delivered 5 projects using Scheduling, ERP, Oracle with measurable KPIs.
    Improved reliability by 30%.
    Collaborated with 10 stakeholders to ship on schedule.

    Associate Program Manager at Lumena Group (2019-06 to 2021-03)
    Delivered 5 projects using Project Planning, SAP, Lean with measurable KPIs.
    Improved process efficiency by 24%.
    Collaborated with 7 stakeholders to ship on schedule. Six Sigma exposure.
    """.strip()

    out = parse_resume(resume, _jd_requirements())

    assert out["skills"] == ["Lean", "Six Sigma"]
    # evidence lines should include the lines that mention Lean / Six Sigma
    assert any("Lean" in ln for ln in out["evidence_lines"])
    assert any("Six Sigma" in ln for ln in out["evidence_lines"])

def test_experience_years_from_date_ranges():
    resume = """
    Program Manager (2017-05 to 2019-11)
    Associate Program Manager (2019-06 to 2021-03)
    """.strip()

    out = parse_resume(resume, _jd_requirements())
    # (2017-05→2019-11)=30 months, (2019-06→2021-03)=21 months, total=51 → 4.25 years → 4.2 rounded
    assert abs(out["experience_years"] - 4.2) < 1e-6

def test_deterministic_output_for_same_input():
    resume = "Program Manager (2017-05 to 2019-11)\nLean initiatives; Six Sigma projects."
    out1 = parse_resume(resume, _jd_requirements())
    out2 = parse_resume(resume, _jd_requirements())
    assert out1 == out2
