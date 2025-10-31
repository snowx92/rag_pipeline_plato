from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

# --- Ensure local 'src/' imports work even when running `python -m main` ---
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# ---------------------------------------------------------------------------

from schema import validate_json, assert_valid
from parse_resume import parse_resume
from retrieve import build_resume_collection, retrieve_for_requirements
from scorer import score_rule_based
from pipeline import run_pipeline, PipelineConfig

# Optional (only if you added plain-text JD support)
try:
    from jd_text import parse_job_text
except Exception:
    parse_job_text = None


def _read_jd(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    # Support either a full record {"job": {...}} or just the job dict directly
    if isinstance(obj, dict) and "job" in obj and isinstance(obj["job"], dict):
        return obj["job"]
    if not isinstance(obj, dict):
        raise SystemExit("JD file must be a JSON object or contain a 'job' object")
    required = ["title", "sector", "location", "description", "requirements"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise SystemExit(f"JD JSON missing keys: {missing}")
    return obj


def _read_jd_txt(path: Path) -> dict:
    if parse_job_text is None:
        raise SystemExit("--jd-txt was provided but jd_text.py is not available.")
    text = path.read_text(encoding="utf-8")
    return parse_job_text(text)


def _read_resume(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_rules(jd: dict, resume_text: str, k: int = 3, debug: bool = False) -> dict:
    # 1) Parse
    parsed = parse_resume(resume_text, jd.get("requirements", []))
    # 2) Build collection
    lines = parsed.get("evidence_lines", [])
    if len(lines) < 2:
        lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    name = f"resume_v0_{uuid4().hex[:8]}"
    _, coll = build_resume_collection(lines, collection_name=name)
    # 3) Retrieve (only "Proficiency in ..." requirements)
    raw_reqs = [r for r in jd.get("requirements", []) if r.lower().startswith("proficiency in ")]
    hits = retrieve_for_requirements(coll, raw_reqs, k=k, debug=debug)
    # 4) Score (rule-based)
    out = score_rule_based(jd, parsed, hits)
    assert_valid(out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the RAG pipeline and emit schema-valid JSON")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jd", type=Path, help="Path to job JSON (either full record with 'job' or just the job object)")
    src.add_argument("--jd-txt", dest="jd_txt", type=Path, help="Path to job description as plain text")

    p.add_argument("--resume", required=True, type=Path, help="Path to resume text file")
    p.add_argument("--out", type=Path, default=None, help="Optional path to write the result JSON; stdout if omitted")
    p.add_argument("--mode", choices=["llm", "rules"], default="llm", help="Use LLM (default) or rule-based only")
    p.add_argument("--k", type=int, default=3, help="Top-k evidence per requirement (default: 3)")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name (LLM mode only)")
    p.add_argument("--seed", type=int, default=42, help="Seed for determinism if provider supports it (LLM mode)")
    p.add_argument("--print-prompt", action="store_true", help="Echo the assembled prompt to stderr (LLM mode)")
    p.add_argument("--debug", action="store_true", help="Verbose debug logs to stderr (retrieval hits, LLM calls, fallbacks)")

    args = p.parse_args(argv)

    jd = _read_jd(args.jd) if args.jd else _read_jd_txt(args.jd_txt)
    resume_text = _read_resume(args.resume)

    try:
        if args.mode == "rules":
            result = _run_rules(jd, resume_text, k=args.k, debug=args.debug)
        else:
            # Delegate to pipeline (parse → retrieve → prompt → LLM → validate/repair → fallback)
            result = run_pipeline(
                jd,
                resume_text,
                cfg=PipelineConfig(k=args.k, model=args.model, seed=args.seed),
                debug=args.debug,
                print_prompt=args.print_prompt,
            )

        ok, errs = validate_json(result)
        if not ok:
            print("Result failed schema validation: ", errs, file=sys.stderr)
            return 2

        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
