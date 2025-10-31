from __future__ import annotations
from typing import Any, Dict, Optional
import json
import sys
from uuid import uuid4

from parse_resume import parse_resume, extract_required_skills_from_jd
from retrieve import build_resume_collection, retrieve_for_requirements
from prompt import build_prompt
from schema import get_schema, validate_json, assert_valid
from scorer import score_rule_based
from llm_evaluator import LLMConfig, generate_scores, repair_json


class PipelineConfig:
    def __init__(self, *, k: int = 3, model: str = "gpt-4o-mini", seed: Optional[int] = 42):
        self.k = k
        self.model = model
        self.seed = seed


def run_pipeline(
    jd: Dict[str, Any],
    resume_text: str,
    *,
    cfg: Optional[PipelineConfig] = None,
    client=None,
    debug: bool = False,
    print_prompt: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or PipelineConfig()

    # 1) Parse
    parsed = parse_resume(resume_text, jd.get("requirements", []))

    # 2) Build collection & retrieve (use parsed evidence lines; fallback to raw lines)
    all_lines = parsed.get("evidence_lines", [])
    if len(all_lines) < 2:
        all_lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]

    unique_name = f"resume_v0_{uuid4().hex[:8]}"
    client_vs, coll = build_resume_collection(all_lines, collection_name=unique_name)
    if debug:
        try:
            nvec = coll.count()
        except Exception:
            nvec = len(all_lines)
        print(f"[pipeline] collection built: name={coll.name!r} vectors={nvec}", file=sys.stderr)

    # Requirements to query = keep the original phrasing that starts with "Proficiency in "
    raw_reqs = [r for r in jd.get("requirements", []) if r.lower().startswith("proficiency in ")]
    hits = retrieve_for_requirements(coll, raw_reqs, k=cfg.k, debug=debug)

    # 3) Prompt
    prompt = build_prompt(jd, parsed, hits, get_schema())
    if print_prompt or debug:
        print(f"[pipeline] prompt length: {len(prompt)} chars", file=sys.stderr)
        if print_prompt:
            print("----- BEGIN PROMPT -----", file=sys.stderr)
            print(prompt, file=sys.stderr)
            print("----- END PROMPT -----", file=sys.stderr)

    # 4) LLM evaluate → JSON
    llm_cfg = LLMConfig(model=cfg.model, seed=cfg.seed)
    try:
        if debug:
            print(f"[pipeline] LLM call: model={cfg.model} seed={cfg.seed}", file=sys.stderr)
        result = generate_scores(prompt, cfg=llm_cfg, client=client)
        ok, errs = validate_json(result)
        if not ok:
            if debug:
                print(f"[pipeline] schema invalid; attempting repair (errors={len(errs)})", file=sys.stderr)
            # 5) One repair attempt
            repaired = repair_json(json.dumps(result), errs, cfg=llm_cfg, client=client)
            ok2, errs2 = validate_json(repaired)
            if ok2:
                if debug:
                    print("[pipeline] repair succeeded; returning LLM(repaired) result", file=sys.stderr)
                assert_valid(repaired)
                return repaired
            if debug:
                print("[pipeline] repair failed; falling back to rule-based scorer", file=sys.stderr)
            # 6) Fallback to rule-based
            rb = score_rule_based(jd, parsed, hits)
            assert_valid(rb)
            return rb
        if debug:
            print("[pipeline] LLM result valid; returning LLM output", file=sys.stderr)
        assert_valid(result)
        return result
    except Exception as e:
        if debug:
            print(f"[pipeline] exception during LLM flow: {e}; falling back to rule-based scorer", file=sys.stderr)
        # On any error, fallback to rule-based
        rb = score_rule_based(jd, parsed, hits)
        assert_valid(rb)
        return rb
