from __future__ import annotations
from typing import Any, Dict, List
import json


def _stable_sorted_strs(items: List[str]) -> List[str]:
    return sorted([s for s in (items or []) if isinstance(s, str) and s.strip()], key=lambda x: x.lower())


def _stable_hits(retrieval: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    # Sort requirements by case-insensitive order; within each requirement, sort hits by (distance, id)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for req in _stable_sorted_strs(list(retrieval.keys())):
        items = retrieval.get(req, []) or []
        normed = []
        for it in items:
            normed.append({
                "id": str(it.get("id", "")),
                "text": str(it.get("text", "")),
                "distance": float(it.get("distance", 0.0)),
                "metadata": it.get("metadata", {}) or {},
            })
        out[req] = sorted(normed, key=lambda x: (round(x["distance"], 8), x["id"]))
    return out


def build_prompt(
    jd: Dict[str, Any],
    parsed_resume: Dict[str, Any],
    retrieval_hits: Dict[str, List[Dict[str, Any]]],
    schema_dict: Dict[str, Any],
) -> str:
    """Return a deterministic prompt string.

    Sections:
      - SYSTEM (JSON-only guardrails)
      - JSON_SCHEMA (strict Draft-07 schema)
      - JOB (title, sector, requirements, description)
      - PARSED_RESUME (skills, experience_years, evidence_lines)
      - RETRIEVAL (top-k hits per requirement)
      - TASK (clear instruction to output a single JSON object only)
    """
    reqs = _stable_sorted_strs(jd.get("requirements", []))
    job_obj = {
        "title": jd.get("title", ""),
        "sector": jd.get("sector", ""),
        "location": jd.get("location", ""),
        "description": jd.get("description", ""),
        "requirements": reqs,
    }

    parsed_obj = {
        "skills": _stable_sorted_strs(parsed_resume.get("skills", [])),
        "experience_years": float(parsed_resume.get("experience_years", 0.0)),
        "evidence_lines": _stable_sorted_strs(parsed_resume.get("evidence_lines", [])),
    }

    retrieval_obj = _stable_hits({k: retrieval_hits.get(k, []) for k in reqs})

    system_block = (
        "You are an evaluation service.\n"
        "Return a SINGLE JSON object ONLY that strictly validates against the provided JSON_SCHEMA.\n"
        "Do not include explanations, markdown, or any extra text before or after the JSON.\n"
        "If uncertain, make the best deterministic judgment using only the provided evidence."
    )

    task_block = (
        "Using JOB, PARSED_RESUME, and RETRIEVAL, produce scores and explanations that match JSON_SCHEMA.\n"
        "- Be faithful to the evidence.\n"
        "- Scores are integers 0..100.\n"
        "- All arrays and fields required by the schema must be present.\n"
        "- If an item is missing, explain the gap in missingDetail.\n"
        "- No extra keys."
    )

    parts = [
        "SYSTEM:\n" + system_block,
        "JSON_SCHEMA:\n" + json.dumps(schema_dict, ensure_ascii=False, sort_keys=True),
        "JOB:\n" + json.dumps(job_obj, ensure_ascii=False, sort_keys=True),
        "PARSED_RESUME:\n" + json.dumps(parsed_obj, ensure_ascii=False, sort_keys=True),
        "RETRIEVAL:\n" + json.dumps(retrieval_obj, ensure_ascii=False, sort_keys=True),
        "TASK:\n" + task_block,
    ]

    return "\n\n".join(parts)
