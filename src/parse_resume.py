"""
Minimal, deterministic resume parser (v0).
- Extract required skills by exact normalized match from JD requirements of the form
  "Proficiency in X".
- Compute experience years from date ranges like "YYYY-MM to YYYY-MM".
- Collect evidence lines that mention matched skills.

Favor simplicity and determinism.
"""
from __future__ import annotations
from typing import List, Dict, Any
import re

__all__ = ["extract_required_skills_from_jd", "parse_resume"]

def _norm(s: str) -> str:
    """Normalize for stable, case-insensitive matching."""
    return re.sub(r"\s+", " ", s.strip().lower())

def extract_required_skills_from_jd(jd_requirements: List[str]) -> List[str]:
    """Return canonical skill names referenced by JD requirements.

    We only support the simple pattern: "Proficiency in <Skill>".
    Results are de-duplicated and sorted for determinism.
    """
    out: List[str] = []
    for r in jd_requirements or []:
        m = re.match(r"^\s*Proficiency in\s+(.+?)\s*$", r, flags=re.IGNORECASE)
        if m:
            skill = m.group(1).strip()
            out.append(skill)
    # de-dup while preserving first occurrence order, then sort for stable output
    seen = {}
    ordered = [seen.setdefault(_norm(x), x) for x in out if _norm(x) not in seen]
    return sorted(ordered, key=lambda x: _norm(x))

_DATE_PAIR = re.compile(
    r"(\d{4}-\d{2})\s*(?:to|–|-|—)\s*(\d{4}-\d{2})",
    flags=re.IGNORECASE,
)

def _months_between(start: str, end: str) -> int:
    ys, ms = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    months = (ye - ys) * 12 + (me - ms)
    return max(0, months)

def _collect_experience_months(text: str) -> int:
    total = 0
    for s, e in _DATE_PAIR.findall(text or ""):
        total += _months_between(s, e)
    return total

def parse_resume(text: str, jd_requirements: List[str]) -> Dict[str, Any]:
    """Parse resume text against JD requirements.

    Returns dict with keys:
      - skills: sorted list of matched skills (strings)
      - experience_years: float (months / 12, rounded to 1 decimal)
      - evidence_lines: list[str] of lines that mention matched skills
    """
    required_skills = extract_required_skills_from_jd(jd_requirements)
    # Build normalized set for matching
    norm_targets = { _norm(s): s for s in required_skills }

    # Evidence lines: any line that includes a required skill token (case-insensitive)
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    evidence_lines: List[str] = []
    present_norms = set()
    for ln in lines:
        ln_norm = _norm(ln)
        for t_norm, orig in norm_targets.items():
            # word-boundary-ish match (simple contains to keep it minimal)
            if t_norm in ln_norm:
                evidence_lines.append(ln)
                present_norms.add(t_norm)
                break  # avoid duplicating same line for multiple skills

    matched_skills = sorted([norm_targets[n] for n in present_norms], key=_norm)

    months = _collect_experience_months(text)
    years = round(months / 12.0, 1)

    return {
        "skills": matched_skills,
        "experience_years": years,
        "evidence_lines": evidence_lines,
    }
