"""
LLM evaluator (Step 6)
- Uses environment variable OPENAI_API_KEY (do NOT hardcode keys).
- JSON mode call, temperature=0, top_p=1, optional seed for determinism.
- One repair attempt helper.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, List
import json
import os
import sys


class LLMConfig:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed


def _create_openai_client():
    # Local import so tests can run without the package installed
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment")
    return OpenAI(api_key=api_key)


def _json_mode_kwargs(cfg: LLMConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "response_format": {"type": "json_object"},
    }
    if cfg.seed is not None:
        kwargs["seed"] = cfg.seed
    return kwargs


def generate_scores(prompt: str, cfg: Optional[LLMConfig] = None, client=None) -> Dict[str, Any]:
    """
    Call the LLM in JSON-mode and parse the JSON into a dict.
    Raises RuntimeError on provider/parse issues (so caller can decide to repair/fallback).
    """
    cfg = cfg or LLMConfig()
    client = client or _create_openai_client()

    kwargs = _json_mode_kwargs(cfg)
    messages = [
        {"role": "system", "content": "You return one JSON object that validates against the given schema. No extra text."},
        {"role": "user", "content": prompt},
    ]

    try:
        # proof-of-call debug
        print("[llm] chat.completions.create(...) called", file=sys.stderr)
        resp = client.chat.completions.create(messages=messages, **kwargs)
        content = resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e

    try:
        return json.loads(content or "{}")
    except Exception as e:
        raise RuntimeError(f"LLM returned non-JSON content: {str(content)[:200]}...") from e


def repair_json(bad_json_text: str, schema_errors: List[str], cfg: Optional[LLMConfig] = None, client=None) -> Dict[str, Any]:
    """One-shot repair request: provide previous JSON and schema errors, ask for corrected JSON-only output."""
    cfg = cfg or LLMConfig()
    client = client or _create_openai_client()

    kwargs = _json_mode_kwargs(cfg)
    repair_prompt = (
        "Your previous JSON did not validate against the schema.\n"
        "Here is your last JSON:\n" + bad_json_text + "\n\n"
        "Here are validation errors (bulleted):\n- " + "\n- ".join(schema_errors[:10]) + "\n\n"
        "Return a corrected JSON object ONLY that fixes these issues."
    )
    messages = [
        {"role": "system", "content": "You return one corrected JSON object. No extra text."},
        {"role": "user", "content": repair_prompt},
    ]

    try:
        print("[llm] chat.completions.create(...) called (repair)", file=sys.stderr)
        resp = client.chat.completions.create(messages=messages, **kwargs)
        content = resp.choices[0].message.content
        return json.loads(content or "{}")
    except Exception as e:
        raise RuntimeError(f"Repair attempt failed: {e}") from e
