"""
Rule-based scorer (v0) that produces a schema-valid JSON output without using an LLM.

Inputs:
  - jd: dict with title/sector/location/description/requirements
  - parsed_resume: dict from parse_resume(...) {skills, experience_years, evidence_lines}
  - retrieval_hits: dict from retrieve_for_requirements(...)

Outputs:
  - dict matching schema.get_schema() (validated in tests)

Deterministic and minimal: good for unit tests and as a fallback path.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import re

from schema import assert_valid

_SOFT_POSITIVE = [
    "collaborated", "stakeholder", "stakeholders", "ownership", "owned", "mentor", "mentored",
    "lead", "led", "leadership", "cross-functional",
]
_IMPACT_TOKENS = ["%", "kpi", "kpis", "improved", "delivered"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _extract_years_req(requirements: List[str]) -> int:
    for r in requirements or []:
        m = re.search(r"(\d+)\+\s*years\s+of\s+relevant\s+experience", r, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


def _tech_requirements(requirements: List[str]) -> List[str]:
    out = []
    for r in requirements or []:
        m = re.match(r"^\s*Proficiency in\s+(.+?)\s*$", r, flags=re.IGNORECASE)
        if m:
            out.append(m.group(1).strip())
    # stable sort by lowercase
    return sorted(out, key=lambda x: x.lower())


def _present_in_skills(skill: str, skills: List[str]) -> bool:
    n = _norm(skill)
    return any(_norm(s) == n for s in (skills or []))


def _has_any(tokens: List[str], lines: List[str]) -> bool:
    txt = "\n".join(lines or [])
    low = _norm(txt)
    return any(t in low for t in tokens)


def _exp_score(cand_years: float, years_req: int) -> Tuple[int, Dict[str, Any]]:
    if years_req <= 0:
        # Make a conservative score that still allows high values for long tenures
        score = int(round(min(100, 60 + min(40, cand_years * 5))))
        present = True
        gap_pct = 0
        evidence = f"Total years ≈ {cand_years:.1f} via roles listed"
    else:
        ratio = cand_years / max(1, years_req)
        score = int(round(min(96, 55 + min(35, ratio * 25))))
        present = cand_years >= years_req
        deficit = max(0.0, years_req - cand_years)
        gap_pct = 0 if present else min(40, int(round(deficit * 10)))
        evidence = f"Total years ≈ {cand_years:.1f} via roles listed"
    breakdown = {
        "requirement": f"{years_req}+ years of relevant experience" if years_req > 0 else "Years of relevant experience",
        "present": present,
        "evidence": evidence,
        "gapPercentage": int(gap_pct),
        "missingDetail": (
            "0% gap because requirement satisfied." if gap_pct == 0
            else f"{int(gap_pct)}% gap because role expects {years_req}+ years; resume totals ≈ {cand_years:.1f}."
        ),
    }
    return score, breakdown


def score_rule_based(jd: Dict[str, Any], parsed_resume: Dict[str, Any], retrieval_hits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    reqs = jd.get("requirements", []) or []
    tech_reqs = _tech_requirements(reqs)
    years_req = _extract_years_req(reqs)

    skills = parsed_resume.get("skills", []) or []
    evidence_lines = parsed_resume.get("evidence_lines", []) or []
    cand_years = float(parsed_resume.get("experience_years", 0.0))

    # Technical score: fraction of tech requirements found in skills
    if tech_reqs:
        hits = sum(1 for t in tech_reqs if _present_in_skills(t, skills))
        tech_score = int(round((hits / len(tech_reqs)) * 100))
    else:
        tech_score = 50  # neutral when no tech reqs found
        hits = 0

    # Experience score & breakdown item
    exp_score, exp_break_item = _exp_score(cand_years, years_req)

    # Cultural: look for soft signals in evidence lines
    cultural_present = _has_any(_SOFT_POSITIVE, evidence_lines)
    cultural_score = 70 if cultural_present else 55

    # Overall
    overall = int(round(0.4 * tech_score + 0.4 * exp_score + 0.2 * cultural_score))

    # Detailed breakdown: technical items (cap to 4 items) with simple gaps
    tech_items: List[Dict[str, Any]] = []
    for t in tech_reqs[:4]:
        present = _present_in_skills(t, skills)
        gap = 0 if present else 20
        tech_items.append({
            "requirement": f"Proficiency in {t}",
            "present": present,
            "evidence": (f"Skills list includes {t}" if present else "No mention or usage evidence in resume"),
            "gapPercentage": gap,
            "missingDetail": (
                "0% gap because requirement fully met with explicit evidence." if present else f"{gap}% gap because {t} is required; resume lists no {t} evidence."
            ),
        })

    # Experience breakdown includes an impact/scope check
    scope_present = _has_any(_IMPACT_TOKENS, evidence_lines)
    scope_gap = 0 if scope_present else 15
    exp_items = [exp_break_item, {
        "requirement": "Demonstrated scope and measurable impact",
        "present": scope_present,
        "evidence": ("Bullets quantify outcomes (%, KPIs)" if scope_present else "Bullets lack metrics"),
        "gapPercentage": scope_gap,
        "missingDetail": ("0% gap because KPIs/metrics present." if scope_gap == 0 else f"{scope_gap}% gap because measurable impact required."),
    }]

    # Education/certs placeholder for v0 (we didn't parse education yet)
    edu_items = [{
        "requirement": "Formal education or equivalent experience",
        "present": True,
        "evidence": "Experience considered sufficient",
        "gapPercentage": 0,
        "missingDetail": "0% gap.",
    }]

    # Cultural breakdown
    collab_present = _has_any(["collaborated", "cross-functional", "stakeholder"], evidence_lines)
    leader_present = _has_any(["lead", "led", "owned", "mentored", "ownership"], evidence_lines)
    soft_items = [
        {
            "requirement": "Cross-functional collaboration",
            "present": collab_present,
            "evidence": ("Worked with stakeholders / cross-functional teams" if collab_present else "No collaboration examples"),
            "gapPercentage": 0 if collab_present else 15,
            "missingDetail": ("0% gap because collaboration evidenced." if collab_present else "15% gap because cross-functional work is critical."),
        },
        {
            "requirement": "Leadership/ownership behaviors",
            "present": leader_present,
            "evidence": ("Owned initiatives or mentored juniors" if leader_present else "Ownership not evidenced"),
            "gapPercentage": 0 if leader_present else 10,
            "missingDetail": ("0% gap because ownership signals present." if leader_present else "10% gap because role expects initiative ownership."),
        },
    ]

    # Strengths / improvement areas
    strengths: List[str] = []
    if tech_score >= 60 and tech_reqs:
        strengths.append("Coverage across key tools with explicit skill evidence.")
    if cand_years >= (years_req or 0):
        strengths.append(f"Meets experience bar (≈{cand_years:.1f} vs {years_req}+).")
    if cultural_present:
        strengths.append("Collaboration/ownership evidenced via stakeholder work.")
    if not strengths:
        strengths.append("Adequate baseline with growth potential.")

    improvements: List[str] = []
    for t in tech_reqs:
        if not _present_in_skills(t, skills) and len(improvements) < 3:
            improvements.append(f"{t} missing – 20% gap because it is a core requirement; resume has no {t}.")
    if cand_years < (years_req or 0) and len(improvements) < 3:
        diff = (years_req or 0) - cand_years
        g = min(40, int(round(diff * 10)))
        improvements.append(f"Years short by ≈{diff:.1f} – {g}% gap vs {years_req}+; resume totals ≈{cand_years:.1f}.")
    if not improvements:
        improvements.append("Increase quantification of impact to strengthen seniority signal.")

    result: Dict[str, Any] = {
        "overallScore": overall,
        "technicalSkillsScore": tech_score,
        "experienceScore": exp_score,
        "culturalFitScore": cultural_score,
        "matchSummary": (
            f"Alignment on {hits}/{len(tech_reqs)} core tools; experience ≈{cand_years:.1f} vs {years_req}+ requirement. "
            + ("Strengths include collaboration and ownership; " if cultural_present else "")
            + ("gaps focus on " + ", ".join([t for t in tech_reqs if not _present_in_skills(t, skills)][:2]) + "." if any(not _present_in_skills(t, skills) for t in tech_reqs) else "minor scope signals.")
        ),
        "strengthsHighlights": strengths[:3],
        "improvementAreas": improvements[:3],
        "detailedBreakdown": {
            "technicalSkills": tech_items,
            "experience": exp_items,
            "educationAndCertifications": edu_items,
            "culturalFitAndSoftSkills": soft_items,
        },
    }

    # Optional red flag: strong seniority mismatch
    if years_req > 0 and cand_years + 2 < years_req:
        result["redFlags"] = [{
            "issue": "Seniority mismatch",
            "evidence": f"Years ≈ {cand_years:.1f} (< {years_req}+ req)",
            "reason": "Limited autonomy risk.",
        }]

    # Validate before returning
    assert_valid(result)
    return result
