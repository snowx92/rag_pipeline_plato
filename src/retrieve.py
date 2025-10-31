from __future__ import annotations
from typing import List, Dict, Any, Tuple
import sys
import uuid

import chromadb
from chromadb.config import Settings

# If you rely on sentence-transformers, ensure it's installed.
# The tests typically use the default embedding function or a simple one.


def build_resume_collection(
    resume_lines: List[str],
    collection_name: str | None = None,
) -> Tuple[chromadb.Client, Any]:
    """
    Create a Chroma collection and add each resume line as a separate document.
    """
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        # Defaults to sqlite in-memory if you don't set a path/persist dir
    ))

    name = collection_name or f"resume_v0"
    # If a collection with this name exists and get_or_create=False, Chroma will raise.
    # For tests/pipeline runs, pass a unique name to avoid collisions.
    coll = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    # Add documents
    ids = [f"res-{i:04d}" for i in range(len(resume_lines))]
    metadatas = [{"idx": i, "id": f"res-{i:04d}"} for i in range(len(resume_lines))]
    coll.add(documents=resume_lines, ids=ids, metadatas=metadatas)
    return client, coll


def _normalize_requirement_to_query(rq: str) -> str:
    # For very simple pipeline, the requirement itself is the query;
    # strip leading "Proficiency in ".
    t = rq.strip()
    if t.lower().startswith("proficiency in "):
        t = t[len("proficiency in "):]
    return t


def retrieve_for_requirements(
    collection,
    requirements: List[str],
    k: int = 3,
    debug: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each requirement, run a top-k vector query and return text+distance+meta.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for idx, rq in enumerate(requirements):
        q = _normalize_requirement_to_query(rq)
        res = collection.query(
            query_texts=[q],
            n_results=k,
            include=["documents", "distances", "metadatas"],  # no "ids" (Chroma complains)
        )
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out[rq] = [
            {
                "id": metas[i].get("id", metas[i].get("idx")),  # stable id for tests
                "text": docs[i],
                "distance": float(dists[i]),
                "meta": metas[i],
            }
            for i in range(len(docs))
        ]
        if debug:
            print(f"[retrieve] req[{idx}] {rq!r} -> top-{len(docs)}", file=sys.stderr)
            for i, (d, dist) in enumerate(zip(docs, dists)):
                snippet = (d[:120] + "…") if len(d) > 120 else d
                print(f"   {i+1:>2}. dist={dist:.4f}  {snippet}", file=sys.stderr)
    return out
