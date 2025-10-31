"""
Minimal JSON schema and validation helpers for the assignment output.
Stage 1 goal: define a strict schema and a tiny validation API.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

try:
    from jsonschema import Draft7Validator
except Exception as e:  # pragma: no cover
    raise RuntimeError("jsonschema is required. Install with `pip install jsonschema`.") from e


# ---- Public API -------------------------------------------------------------

def get_schema() -> Dict[str, Any]:
    """Return the JSON Schema (Draft-07) for the model's output.

    The schema is intentionally strict to enforce deterministic structure.
    """
    base_req_item = {
        "type": "object",
        "properties": {
            "requirement": {"type": "string", "minLength": 1},
            "present": {"type": "boolean"},
            "evidence": {"type": "string"},
            "gapPercentage": {"type": "integer", "minimum": 0, "maximum": 100},
            "missingDetail": {"type": "string"},
        },
        "required": ["requirement", "present", "evidence", "gapPercentage", "missingDetail"],
        "additionalProperties": False,
    }

    schema: Dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AssignmentOutput",
        "type": "object",
        "properties": {
            "overallScore": {"type": "integer", "minimum": 0, "maximum": 100},
            "technicalSkillsScore": {"type": "integer", "minimum": 0, "maximum": 100},
            "experienceScore": {"type": "integer", "minimum": 0, "maximum": 100},
            "culturalFitScore": {"type": "integer", "minimum": 0, "maximum": 100},
            "matchSummary": {"type": "string"},
            "strengthsHighlights": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 3,
            },
            "improvementAreas": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 3,
            },
            "detailedBreakdown": {
                "type": "object",
                "properties": {
                    "technicalSkills": {"type": "array", "items": base_req_item},
                    "experience": {"type": "array", "items": base_req_item},
                    "educationAndCertifications": {"type": "array", "items": base_req_item},
                    "culturalFitAndSoftSkills": {"type": "array", "items": base_req_item},
                },
                "required": [
                    "technicalSkills",
                    "experience",
                    "educationAndCertifications",
                    "culturalFitAndSoftSkills",
                ],
                "additionalProperties": False,
            },
            # Optional red flags
            "redFlags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue": {"type": "string"},
                        "evidence": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["issue", "evidence", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "overallScore",
            "technicalSkillsScore",
            "experienceScore",
            "culturalFitScore",
            "matchSummary",
            "strengthsHighlights",
            "improvementAreas",
            "detailedBreakdown",
        ],
        "additionalProperties": False,
    }
    return schema


def validate_json(payload: Dict[str, Any]) -> Tuple[bool, Tuple[str, ...]]:
    """Validate a payload against the schema.

    Returns (is_valid, errors) where errors is a tuple of readable messages.
    """
    validator = Draft7Validator(get_schema())
    errors = tuple(sorted((e.message for e in validator.iter_errors(payload))))
    return (len(errors) == 0, errors)


def assert_valid(payload: Dict[str, Any]) -> None:
    """Raise AssertionError with a nice message if payload is invalid."""
    ok, errors = validate_json(payload)
    if not ok:
        raise AssertionError("Invalid output: " + "; ".join(errors))
