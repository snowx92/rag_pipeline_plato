"""
Plain-text Job Description → JD dict (minimal, deterministic heuristics).
"""
from __future__ import annotations
from typing import Dict, List
import re

# Tiny sector/location heuristics
_SECTOR_RULES = [
    (re.compile(r"operations|supply\s*chain", re.I), "Operations & Supply Chain"),
    (re.compile(r"data|analytics", re.I), "Data & Analytics"),
    (re.compile(r"software|engineering|developer", re.I), "Software Engineering"),
    (re.compile(r"product\s+manager|product\s+management", re.I), "Product Management"),
]

_CITY_RULES = [
    (re.compile(r"\bcairo\b", re.I), "Cairo, Egypt"),
    (re.compile(r"\bgiza\b", re.I), "Giza, Egypt"),
    (re.compile(r"\balexandria\b", re.I), "Alexandria, Egypt"),
]

_DEF_LOCATION = "Unknown"
_DEF_SECTOR = "Unknown"


def _first_nonempty(lines: List[str]) -> str:
    for ln in lines:
        s = ln.strip().lstrip("-•* ")
        if s:
            return s
    return ""


def _collect_requirements(lines: List[str], full_text: str) -> List[str]:
    reqs: List[str] = []
    # 1) Explicit "Proficiency in ..." lines
    for ln in lines:
        m = re.match(r"\s*[-•*]?\s*(Proficiency in\s+.+?)\s*$", ln, flags=re.I)
        if m:
            reqs.append(m.group(1).strip())
    # 2) Years of experience
    m2 = re.search(r"(\d+)\+\s*years\s+of\s+relevant\s+experience", full_text, flags=re.I)
    if m2:
        reqs.append(f"{m2.group(1)}+ years of relevant experience")
    # 3) Degree hint
    if re.search(r"bachelor\w*|mba|degree", full_text, flags=re.I):
        reqs.append("Bachelor's degree or equivalent experience")
    # Dedup + deterministic order
    return sorted(dict.fromkeys([r.strip() for r in reqs if r.strip()]), key=lambda x: x.lower())


def _infer_location(full_text: str) -> str:
    # Hybrid/Remote labels first
    if re.search(r"hybrid", full_text, flags=re.I):
        for rx, city in _CITY_RULES:
            if rx.search(full_text):
                return f"Hybrid – {city.split(',')[0]}"
        return "Hybrid"
    if re.search(r"remote", full_text, flags=re.I):
        return "Remote – EMEA"
    for rx, city in _CITY_RULES:
        if rx.search(full_text):
            return city
    return _DEF_LOCATION


def _infer_sector(full_text: str) -> str:
    for rx, name in _SECTOR_RULES:
        if rx.search(full_text):
            return name
    return _DEF_SECTOR


def parse_job_text(text: str) -> Dict[str, object]:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    full = "\n".join(lines)
    title = _first_nonempty(lines) or "Unknown Role"
    requirements = _collect_requirements(lines, full)
    sector = _infer_sector(full)
    location = _infer_location(full)

    # Description: everything after the title
    try:
        first_idx = next(i for i, ln in enumerate(lines) if ln.strip())
    except StopIteration:
        first_idx = 0
    desc = "\n".join(lines[first_idx + 1:]).strip()

    return {
        "title": title,
        "sector": sector,
        "location": location,
        "description": desc or title,
        "requirements": requirements,
    }
