"""Manual single-query runner: all 3 pipelines + real Gemini API + LLM judge.

Runs one user-entered query through Pipeline 1 (Baseline), Pipeline 2
(Dictionary) and Pipeline 3 (Ontology), generates each answer with
gemini-3.8-flash, grades each with the strict blinded judge, then prints to
stdout, in order:

  1. a metric table with exactly: Pipeline, Concept-F1, Recall, Judge correct
  2. the full generated answer (output) of every pipeline

The full JSON object (traces, doc ids, answers, judge grades) is NOT printed
in normal mode; pass --json-only when you need just the JSON (e.g. to save
to a file or pipe into jq).

Progress / rate-limit / library chatter goes to stderr, so screen output stays
readable.

Nothing is ever fabricated: missing key, empty vector store, API errors after
bounded retries, or malformed judge output all abort loudly.

Caching guarantees (enforced, silent):
  * This script is STATELESS - it never reads results/metrics/* records or
    any stored answer/judge output. Every invocation re-generates every
    answer and re-grades it with fresh API calls, including retries of the
    same query.
  * The only cache in the path is the deterministic query-embedding cache,
    keyed by SHA-256 of (model, task_type, exact text). A NEW query text can
    therefore never read another query's embedding (always a fresh API
    embed); re-running the same text reuses only the embedding, never
    answers or grades.

Usage:
    uv run python -m scripts.ask_query --query "What is a heart attack?"
    uv run python -m scripts.ask_query                # interactive prompt
    uv run python -m scripts.ask_query --query "..." --no-judge
    uv run python -m scripts.ask_query --query "..." --json-only   # pure JSON
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import NoReturn

from src.config import (
    GENERATOR_MODEL,
    GOOGLE_API_KEY,
    LLM_JUDGE_MODEL,
    TOP_K_EXTENDED,
    TOP_K_PRIMARY,
)
from src.corpus.vectorstore import get_or_create_collection
from src.evaluation.answer_metrics import ConceptResolver, concept_coverage_f1
from src.generation.generator import generate_answer
from src.pipelines.baseline import run_baseline
from src.pipelines.dictionary import run_dictionary
from src.pipelines.ontology_enhanced import run_ontology_enhanced
from scripts.run_full_eval import _format_trace, judge_answer_strict, retry_call

PIPELINES = [
    ("baseline", "1 Baseline", run_baseline),
    ("dictionary", "2 Dictionary", run_dictionary),
    ("ontology", "3 Ontology", run_ontology_enhanced),
]

_ASCII_TABLE = str.maketrans({
    "—": "-", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "\u00a0": " ",
    "→": "->", "←": "<-", "≥": ">=", "≤": "<=", "×": "x",
})


def _safe(text: object) -> str:
    return str(text).translate(_ASCII_TABLE).encode("ascii", "replace").decode()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


_REAL_STDOUT = sys.stdout
_JSON_MODE = False


def _fatal(msg: str) -> NoReturn:
    _log(f"[FATAL] {msg}")
    if _JSON_MODE:
        _REAL_STDOUT.write(json.dumps({"error": msg}) + "\n")
        _REAL_STDOUT.flush()
    sys.exit(1)


def preflight() -> None:
    if not GOOGLE_API_KEY:
        _fatal("GOOGLE_API_KEY missing. Add it to .env (see .env.example).")
    collection = get_or_create_collection()
    n_indexed = collection.count()
    if n_indexed == 0:
        _fatal("vector store is empty - build it with `uv run python -m scripts.build_vectorstore`.")
    _log(f"[preflight] vector store : {n_indexed} chunks ")
    _log(f"[preflight] generator    : {GENERATOR_MODEL}")
    _log(f"[preflight] llm judge    : {LLM_JUDGE_MODEL}")


def _read_query(cli_query: str | None) -> str:
    if cli_query:
        return cli_query.strip()
    print("Enter query: ", file=sys.stderr, end="")
    sys.stderr.flush()
    query = sys.stdin.readline().strip()
    if not query:
        _fatal("empty query - provide --query \"...\" or type one at the prompt.")
    return query


def run(query: str, top_k: int, use_judge: bool) -> dict:
    resolver = ConceptResolver()
    # Ground truth for a manual query: the ontology concepts objectively
    # matched in the question itself (no relevance judgment file exists).
    expected_concepts = resolver.extract_concepts(query)
    results: dict[str, dict] = {}

    for key, label, fn in PIPELINES:
        _log(f"--- {label} ---")
        # retrieval is also retry-wrapped: a cache MISS here means a live
        # embedding API call, which can hit the same 429s as generation.
        res = retry_call(lambda: fn(query, generate=False, top_k=top_k))
        if not res.retrieved_texts:
            _fatal(f"[{label}] retrieval returned no chunks - cannot answer from empty context.")

        trace = res.metadata.get("reasoning_trace")
        ontology_context = _format_trace(trace) if trace else None

        answer = retry_call(
            lambda: generate_answer(
                question=query,
                context_chunks=res.retrieved_texts[:TOP_K_PRIMARY],
                ontology_context=ontology_context,
            )
        )
        if not answer or not answer.strip():
            _fatal(f"[{label}] generator returned an empty answer.")

        judge = None
        if use_judge:
            judge = retry_call(lambda: judge_answer_strict(query, expected_concepts, answer))
            _log(f"    judge: {judge['score']}/10 correct={judge['correct']}")

        entry = {
            "pipeline": label,
            "expanded_terms": res.metadata.get("expanded_terms", []),
            "retrieved_doc_ids": res.retrieved_doc_ids,
            "answer": answer,
            "expected_concepts": expected_concepts,
            "concepts_in_answer": resolver.extract_concepts(answer),
            "concept_metrics": concept_coverage_f1(answer, expected_concepts, resolver),
            "judge": judge,
        }
        if trace:
            entry["reasoning_trace"] = trace
        results[key] = entry

    return results


def _print_table(results: dict[str, dict], use_judge: bool) -> None:
    header = (
        f"{'Pipeline':<14} {'Concept-F1':>11} {'Recall':>8} {'Judge correct':>14}"
    )
    print("=" * len(header))
    print(" METRIC TABLE - all pipelines")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key, label, _ in PIPELINES:
        r = results[key]
        metrics = r.get("concept_metrics", {})
        f1_s = f"{metrics.get('f1', 0.0):.3f}"
        recall_s = f"{metrics.get('recall', 0.0):.3f}"
        if use_judge and r.get("judge"):
            correct_s = f"{r['judge']['score'] / 10.0:.3f}"
        else:
            correct_s = "-"
        print(
            f"{label:<14} {f1_s:>11} {recall_s:>8} {correct_s:>14}"
        )
    print("-" * len(header))
    print(
        "Concept-F1 / Recall = coverage of the ontology concepts matched in "
        "your question; Judge correct = blinded LLM score (0-10) / 10."
    )


def _print_outputs(results: dict[str, dict]) -> None:
    print()
    print("=" * 100)
    print(" OUTPUTS - all pipelines")
    print("=" * 100)
    for key, label, _ in PIPELINES:
        r = results[key]
        print(f"\n[{label}]")
        print("  " + _safe(r["answer"]).replace("\n", "\n  "))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one query through all 3 pipelines: metric table + outputs + JSON"
    )
    parser.add_argument("--query", default=None, help='the query (otherwise prompted on stderr)')
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="print ONLY the JSON object (normal mode prints table + outputs, no JSON)",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K_EXTENDED, help="chunks to retrieve")
    args = parser.parse_args()

    global _JSON_MODE
    _JSON_MODE = args.json_only

    query = _read_query(args.query)
    use_judge = not args.no_judge
    preflight()

    _log(f"[run] query: {query}")
    with contextlib.redirect_stdout(sys.stderr):
        results = run(query, top_k=args.top_k, use_judge=use_judge)

    payload = {
        "query": query,
        "generator_model": GENERATOR_MODEL,
        "judge_model": LLM_JUDGE_MODEL if use_judge else None,
        "expected_concepts": results["baseline"]["expected_concepts"],
        "results": results,
    }

    if args.json_only:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print()
        _print_table(results, use_judge)
        _print_outputs(results)


if __name__ == "__main__":
    main()
