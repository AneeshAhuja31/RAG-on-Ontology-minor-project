"""Full end-to-end evaluation: 3 poster approaches, real Gemini API, LLM judge.

Approaches evaluated (from Ontology-Enhanced-RAG-for-Medical-Question-Answering):
  Pipeline 1  Baseline   - no expansion, semantic retrieval only
  Pipeline 2  Dictionary - synonym-dictionary expansion (52 term pairs)
  Pipeline 3  Ontology   - rdflib graph traversal (equivalence, hierarchy,
                           domain properties) + reasoning trace in prompt

Per query, per pipeline:
  1. retrieve from the EXISTING vector store (never rebuilt, never re-embedded)
  2. generate the answer with gemini-3.8-flash (real API call)
  3. compute objective concept-coverage Precision / Recall / F1
  4. grade with a strict-JSON, blinded LLM judge (gemini-3.8-flash)

At the end the script prints the final results table, pairwise statistical
tests (Wilcoxon, Cohen's d, Cliff's delta), the per-category judge breakdown,
and an "ontology usage" section proving the graph traversal actually fired.
Results are saved to results/metrics/full_eval_summary.json.

Nothing is ever faked: a missing API key, an empty vector store, API errors
after bounded retries, or malformed judge output all abort loudly.

Resumable: per-query records persist after every API call. Records are tagged
with the model that produced them; changing the model regenerates them.

Usage:
    uv run python -m scripts.run_full_eval                # full run + judge
    uv run python -m scripts.run_full_eval --limit 3      # smoke test
    uv run python -m scripts.run_full_eval --no-judge     # skip the judge
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter

from google import genai
from google.genai import types

from src.config import (
    GENERATOR_MODEL,
    GOOGLE_API_KEY,
    LLM_JUDGE_MODEL,
    METRICS_DIR,
    QUERIES_FILE,
    QUERY_CATEGORIES,
    RELEVANCE_JUDGMENTS_FILE,
    TOP_K_EXTENDED,
    TOP_K_PRIMARY,
    ensure_directories,
)
from src.corpus.loader import load_and_chunk_corpus
from src.corpus.vectorstore import get_or_create_collection
from src.evaluation.answer_metrics import ConceptResolver, concept_coverage_f1
from src.evaluation.retrieval_metrics import aggregate_metrics, compute_all_retrieval_metrics
from src.evaluation.statistical_tests import full_comparison
from src.generation.generator import generate_answer
from src.pipelines.baseline import run_baseline
from src.pipelines.dictionary import run_dictionary
from src.pipelines.ontology_enhanced import run_ontology_enhanced
from src.rate_limiter import acquire_generation

PIPELINES = {
    "pipeline_1_baseline": (run_baseline, "1 Baseline"),
    "pipeline_2_dictionary": (run_dictionary, "2 Dictionary"),
    "pipeline_3_ontology_full": (run_ontology_enhanced, "3 Ontology"),
}

PIPELINE_LABELS = {
    "pipeline_1_baseline": "Pipeline 1 (Baseline)",
    "pipeline_2_dictionary": "Pipeline 2 (Dictionary)",
    "pipeline_3_ontology_full": "Pipeline 3 (Ontology)",
}

PAIR_SPECS = [
    ("ontology_vs_baseline", "pipeline_3_ontology_full", "pipeline_1_baseline"),
    ("ontology_vs_dictionary", "pipeline_3_ontology_full", "pipeline_2_dictionary"),
    ("dictionary_vs_baseline", "pipeline_2_dictionary", "pipeline_1_baseline"),
]

_ASCII_TABLE = str.maketrans({
    "—": "-", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "\u00a0": " ",
    "→": "->", "←": "<-", "≥": ">=", "≤": "<=", "×": "x",
})


def _safe(text: object) -> str:
    return str(text).translate(_ASCII_TABLE).encode("ascii", "replace").decode()


def preflight() -> None:
    if not GOOGLE_API_KEY:
        sys.exit(
            "[FATAL] GOOGLE_API_KEY missing. Add it to .env "
            "(see .env.example). This script never fabricates results, "
            "so it refuses to run without the real API key."
        )
    ensure_directories()
    collection = get_or_create_collection()
    n_indexed = collection.count()
    if n_indexed == 0:
        sys.exit(
            "[FATAL] vector store is empty. Build it once with "
            "`uv run python -m scripts.build_vectorstore` (this script "
            "will never build or rebuild it automatically)."
        )
    n_expected = len(load_and_chunk_corpus())
    if n_indexed != n_expected:
        print(
            f"[WARN] vector store has {n_indexed} chunks but the corpus "
            f"chunks to {n_expected} - using the existing store as-is "
            "(no rebuild, per instructions)."
        )
    print(f"[preflight] vector store : {n_indexed} chunks ready (no rebuild)")
    print(f"[preflight] generator    : {GENERATOR_MODEL}")
    print(f"[preflight] llm judge    : {LLM_JUDGE_MODEL}")
    print(f"[preflight] embeddings   : cache-first (index untouched)")


def retry_call(fn, retries: int = 12, base_delay: float = 20.0, max_delay: float = 90.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - surface any transient API error
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "high demand" in msg:
                delay = min(base_delay * (2 ** attempt), max_delay)
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
                if match:
                    delay = max(delay, float(match.group(1)) + 5.0)
                print(f"    [quota/server] retrying in {delay:.0f}s ...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Persistent rate limiting after retries.")


def _format_trace(trace: dict) -> str:
    lines = [f"Query: {trace.get('original_query', '')}"]
    for match in trace.get("matched_concepts", []):
        lines.append(
            f"  Matched: '{match.get('term', '')}' -> "
            f"{match.get('concept_id', '')} ({match.get('label', '')})"
        )
    for exp in trace.get("expansions", []):
        lines.append(
            f"  Expansion ({exp.get('relation', '')}): "
            f"{exp.get('source', '')} -> {exp.get('target', '')} "
            f"({exp.get('target_label', '')})"
        )
    return "\n".join(lines)


JUDGE_SYSTEM_PROMPT = """You are a strict, blinded medical answer grader.

You will be given a medical question, the set of expected medical concepts, and an answer.
You are NOT told which system produced the answer - grade it purely on quality.

An answer is CORRECT if it directly addresses the question AND covers the expected concepts
without major factual errors or unsupported claims.

Respond with ONLY a JSON object in exactly this shape and nothing else
(no markdown, no code fences, no commentary):
{"correct": true or false, "score": integer from 0 to 10, "reason": "one short sentence"}"""


def _parse_strict_judge_output(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = [cleaned]
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if "correct" not in data or "score" not in data:
            continue

        raw_correct = data["correct"]
        if isinstance(raw_correct, bool):
            correct = raw_correct
        elif isinstance(raw_correct, str) and raw_correct.strip().lower() in ("true", "false"):
            correct = raw_correct.strip().lower() == "true"
        else:
            continue

        try:
            score = int(float(data["score"]))
        except (TypeError, ValueError):
            continue
        if not 0 <= score <= 10:
            continue

        return {
            "correct": correct,
            "score": score,
            "reason": str(data.get("reason", "")).strip(),
        }

    raise ValueError(f"unparseable judge output: {cleaned[:200]}")


def judge_answer_strict(question: str, expected_concepts: list[str], answer: str) -> dict:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    concepts_line = (
        ", ".join(expected_concepts)
        if expected_concepts
        else "(none provided - grade solely on correctness and completeness)"
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Expected concepts: {concepts_line}\n\n"
        f"Answer: {answer}\n"
    )

    last_error: Exception | None = None
    for _ in range(3):
        acquire_generation(LLM_JUDGE_MODEL)
        response = client.models.generate_content(
            model=LLM_JUDGE_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        try:
            return _parse_strict_judge_output(response.text or "")
        except ValueError as e:
            last_error = e
            print("    [judge] malformed JSON, re-calling ...")
            continue

    raise RuntimeError(f"Judge returned malformed output after 3 attempts: {last_error}")


def _out_file(pipeline_key: str):
    return METRICS_DIR / f"full_eval_{pipeline_key}.json"


def _load_records(pipeline_key: str) -> dict[str, dict]:
    path = _out_file(pipeline_key)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["query_id"]: r for r in data}


def _save_records(pipeline_key: str, records: dict[str, dict]) -> None:
    with open(_out_file(pipeline_key), "w", encoding="utf-8") as f:
        json.dump(list(records.values()), f, indent=2, ensure_ascii=False)


def run_pipeline(
    pipeline_key: str,
    pipeline_fn,
    label: str,
    queries: list[dict],
    judgments: dict[str, list[str]],
    resolver: ConceptResolver,
    use_judge: bool,
) -> dict[str, dict]:
    records = _load_records(pipeline_key)
    fresh = 0
    judged = 0

    for i, q in enumerate(queries, 1):
        rec = records.get(q["id"])
        needs_generation = (
            rec is None
            or rec.get("generator_model") != GENERATOR_MODEL
            or not rec.get("answer")
            or rec.get("retrieval_metrics") is None
        )

        if needs_generation:
            res = pipeline_fn(q["text"], generate=False, top_k=TOP_K_EXTENDED)
            if not res.retrieved_texts:
                raise RuntimeError(
                    f"[{pipeline_key}] {q['id']}: retrieval returned no chunks - "
                    "refusing to generate an answer from empty context."
                )

            ontology_context = None
            trace = res.metadata.get("reasoning_trace")
            if trace:
                ontology_context = _format_trace(trace)

            answer = retry_call(
                lambda: generate_answer(
                    question=q["text"],
                    context_chunks=res.retrieved_texts[:TOP_K_PRIMARY],
                    ontology_context=ontology_context,
                )
            )
            if not answer or not answer.strip():
                raise RuntimeError(f"[{pipeline_key}] {q['id']}: empty answer from generator.")

            records[q["id"]] = {
                "query_id": q["id"],
                "category": q["category"],
                "query": q["text"],
                "generator_model": GENERATOR_MODEL,
                "retrieved_doc_ids": res.retrieved_doc_ids,
                "retrieval_metrics": compute_all_retrieval_metrics(
                    res.retrieved_doc_ids, judgments.get(q["id"], [])
                ),
                "expanded_terms": res.metadata.get("expanded_terms", []),
                "reasoning_trace": trace,
                "answer": answer,
                "concept_metrics": concept_coverage_f1(
                    answer, q.get("target_concepts", []), resolver
                ),
            }
            _save_records(pipeline_key, records)
            fresh += 1

        rec = records[q["id"]]
        if use_judge and (
            not rec.get("judge") or rec.get("judge_model") != LLM_JUDGE_MODEL
        ):
            rec["judge"] = retry_call(
                lambda: judge_answer_strict(
                    q["text"], q.get("target_concepts", []), rec["answer"]
                )
            )
            rec["judge_model"] = LLM_JUDGE_MODEL
            _save_records(pipeline_key, records)
            judged += 1

        if i % 5 == 0 or i == len(queries):
            print(f"    {label}: {i}/{len(queries)} queries ({fresh} generated, {judged} judged)")

    return records


def _aggregate(records: list[dict], use_judge: bool) -> dict:
    agg: dict = {
        "retrieval": aggregate_metrics([r["retrieval_metrics"] for r in records]),
        "concept_f1": aggregate_metrics(
            [{"v": r["concept_metrics"]["f1"]} for r in records]
        ).get("v", {}),
        "concept_precision": aggregate_metrics(
            [{"v": r["concept_metrics"]["precision"]} for r in records]
        ).get("v", {}),
        "concept_recall": aggregate_metrics(
            [{"v": r["concept_metrics"]["recall"]} for r in records]
        ).get("v", {}),
    }

    if use_judge:
        judged = [r for r in records if r.get("judge")]
        if judged:
            correct = [1.0 if r["judge"]["correct"] else 0.0 for r in judged]
            agg["judge_correctness"] = {
                "mean": sum(correct) / len(correct),
                "n": len(correct),
            }
            agg["judge_score"] = aggregate_metrics(
                [{"v": float(r["judge"]["score"])} for r in judged]
            ).get("v", {})
    return agg


def _category_breakdown(records: list[dict], use_judge: bool) -> dict:
    result = {}
    for cat in QUERY_CATEGORIES:
        cat_records = [r for r in records if r["category"] == cat]
        if not cat_records:
            continue
        entry = {
            "n": len(cat_records),
            "concept_f1": sum(r["concept_metrics"]["f1"] for r in cat_records) / len(cat_records),
        }
        if use_judge:
            judged = [r for r in cat_records if r.get("judge")]
            if judged:
                entry["judge_correctness"] = (
                    sum(1.0 if r["judge"]["correct"] else 0.0 for r in judged) / len(judged)
                )
        result[cat] = entry
    return result


def _ontology_usage(records: list[dict]) -> dict:
    relation_counter: Counter = Counter()
    per_category: dict[str, dict] = {}
    example = None
    n_with_matches = 0
    n_with_expansions = 0

    for r in records:
        trace = r.get("reasoning_trace") or {}
        matched = trace.get("matched_concepts", [])
        expansions = trace.get("expansions", [])
        cat_entry = per_category.setdefault(
            r["category"], {"queries": 0, "with_expansion": 0, "total_expansions": 0}
        )
        cat_entry["queries"] += 1
        if matched:
            n_with_matches += 1
        if expansions:
            n_with_expansions += 1
            cat_entry["with_expansion"] += 1
            cat_entry["total_expansions"] += len(expansions)
            if example is None:
                example = r
        relation_counter.update(e.get("relation", "") for e in expansions)

    return {
        "n_queries": len(records),
        "queries_with_matched_concepts": n_with_matches,
        "queries_with_expansions": n_with_expansions,
        "total_expansions": sum(relation_counter.values()),
        "relation_counts": dict(relation_counter.most_common()),
        "per_category": per_category,
        "example": example,
    }


def _mean(agg: dict, path: tuple[str, ...], default: float | None = None):
    node = agg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if isinstance(node, (int, float)) else default


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, float) else "-"


def _print_results(
    aggregates: dict[str, dict],
    stats: dict,
    use_judge: bool,
    n_queries: int,
) -> None:
    print()
    print("=" * 100)
    print(f" FINAL RESULTS ({n_queries} queries per pipeline, "
          f"generator = judge = {GENERATOR_MODEL})")
    print("=" * 100)
    header = (
        f"{'Pipeline':<14} {'Recall@5':>9} {'Recall@10':>10} {'nDCG@5':>8} "
        f"{'P@5':>7} {'Conj-F1':>8} {'Conj-R':>7}"
    )
    if use_judge:
        header += f" {'Judge-correct':>14} {'Judge-score':>12}"
    print(header)
    print("-" * len(header))

    for key, (_, short_label) in PIPELINES.items():
        agg = aggregates[key]
        line = (
            f"{short_label:<14} "
            f"{_fmt(_mean(agg, ('retrieval', 'recall@5', 'mean'))):>9} "
            f"{_fmt(_mean(agg, ('retrieval', 'recall@10', 'mean'))):>10} "
            f"{_fmt(_mean(agg, ('retrieval', 'ndcg@5', 'mean'))):>8} "
            f"{_fmt(_mean(agg, ('retrieval', 'precision@5', 'mean'))):>7} "
            f"{_fmt(_mean(agg, ('concept_f1', 'mean'))):>8} "
            f"{_fmt(_mean(agg, ('concept_recall', 'mean'))):>7}"
        )
        if use_judge:
            correctness = _mean(agg, ("judge_correctness", "mean"))
            score = _mean(agg, ("judge_score", "mean"))
            line += (
                f" {_fmt(correctness):>14} "
                f"{_fmt(score):>12}"
            )
        print(line)

    print("-" * len(header))
    print("Concept-F1 = objective concept coverage; Judge = blinded LLM rubric (0-10, correct y/n).")

    for metric_name in ("Recall@5", "Judge score (0-10)"):
        block = stats.get(metric_name)
        if not block:
            continue
        print()
        print(f" Statistical tests - {metric_name}")
        print(f" {'Comparison':<47} {'p':>8} {'Cohens d':>10} {'Cliffs delta':>14}")
        for pair_key, a_key, b_key in PAIR_SPECS:
            test = block.get(pair_key)
            if not test:
                continue
            label = f"{PIPELINE_LABELS[a_key]} vs {PIPELINE_LABELS[b_key]}"
            print(
                f" {label:<47} "
                f"{test['wilcoxon']['p_value']:>8.4f} "
                f"{test['cohens_d']:>10.3f} "
                f"{test['cliffs_delta']['cliffs_delta']:>14.3f}"
            )

    print()
    print(" Per-category breakdown (judge correctness | concept F1)")
    print(f" {'Category':<24}", end="")
    for key, (_, short_label) in PIPELINES.items():
        print(f" {short_label:>20}", end="")
    print()
    for cat in QUERY_CATEGORIES:
        print(f" {cat:<24}", end="")
        for key in PIPELINES:
            cat_entry = aggregates[key].get("_categories", {}).get(cat, {})
            if use_judge:
                jc = cat_entry.get("judge_correctness")
                f1 = cat_entry.get("concept_f1")
                cell = (
                    f"{jc:.2f} | {f1:.3f}"
                    if isinstance(jc, float) and isinstance(f1, float)
                    else "-"
                )
            else:
                f1 = cat_entry.get("concept_f1")
                cell = f"{f1:.3f}" if isinstance(f1, float) else "-"
            print(f" {cell:>20}", end="")
        print()


def _print_ontology_usage(usage: dict) -> None:
    print()
    print("=" * 100)
    print(" ONTOLOGY USAGE (proof the knowledge graph actually fired)")
    print("=" * 100)
    print(f"  Queries run through graph traversal : {usage['n_queries']}")
    print(f"  Queries matching an ontology concept: {usage['queries_with_matched_concepts']}")
    print(f"  Queries with >=1 graph expansion    : {usage['queries_with_expansions']}")
    print(f"  Total expansions added to queries   : {usage['total_expansions']}")
    print("  Expansions by relation type:")
    for relation, count in usage["relation_counts"].items():
        print(f"      {relation:<34} {count:>4}")
    print("  Per-category (queries with expansion / total):")
    for cat in QUERY_CATEGORIES:
        entry = usage["per_category"].get(cat)
        if entry:
            print(f"      {cat:<34} {entry['with_expansion']:>3}/{entry['queries']:<3} "
                  f"({entry['total_expansions']} expansions)")

    example = usage.get("example")
    if example:
        trace = example.get("reasoning_trace") or {}
        print()
        print(f"  Example trace - {_safe(example['query'])}")
        for match in trace.get("matched_concepts", [])[:4]:
            print(f"      matched : '{_safe(match.get('term', ''))}' -> "
                  f"{match.get('concept_id', '')} ({_safe(match.get('label', ''))})")
        for exp in trace.get("expansions", [])[:8]:
            print(f"      {_safe(exp.get('source', ''))} --[{_safe(exp.get('relation', ''))}]--> "
                  f"{_safe(exp.get('target', ''))} ({_safe(exp.get('target_label', ''))})")
        terms = trace.get("expanded_terms", [])
        print(f"      expanded terms: {_safe(', '.join(terms[:12]))}"
              f"{' ...' if len(terms) > 12 else ''}")


def _print_sample_outputs(per_pipeline: dict[str, list[dict]], query_id: str) -> None:
    print()
    print("=" * 100)
    print(f" SAMPLE OUTPUTS - all pipelines (query {query_id})")
    print("=" * 100)
    for key, (_, short_label) in PIPELINES.items():
        rec = next((r for r in per_pipeline.get(key, []) if r["query_id"] == query_id), None)
        if not rec:
            continue
        terms = rec.get("expanded_terms") or []
        judge = rec.get("judge")
        judge_str = (
            f"{judge['score']}/10, correct={'yes' if judge['correct'] else 'no'}"
            if judge else "not judged"
        )
        print(f"\n[{short_label}]  query: {_safe(rec['query'])}")
        print(f"    expanded terms ({len(terms)}): "
              f"{_safe(', '.join(terms[:10])) if terms else '(none)'}"
              f"{' ...' if len(terms) > 10 else ''}")
        print(f"    retrieved docs: {len(rec.get('retrieved_doc_ids', []))} | judge: {judge_str}")
        if judge and judge.get("reason"):
            print(f"    judge reason : {_safe(judge['reason'])}")
        print("    answer:")
        print("      " + _safe(rec.get("answer", "")).replace("\n", "\n      "))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full evaluation: 3 approaches, real Gemini API calls, LLM judge"
    )
    parser.add_argument("--limit", type=int, default=None, help="only first N queries (smoke test)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

    preflight()

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if args.limit:
        queries = queries[: args.limit]

    with open(RELEVANCE_JUDGMENTS_FILE, "r", encoding="utf-8") as f:
        judgments = json.load(f)

    use_judge = not args.no_judge
    print()
    print("=" * 100)
    print(f" FULL EVAL: 3 approaches x {len(queries)} queries "
          f"(judge={'ON' if use_judge else 'OFF'})")
    print("=" * 100)

    resolver = ConceptResolver()
    per_pipeline: dict[str, list[dict]] = {}
    aggregates: dict[str, dict] = {}

    for key, (pipeline_fn, short_label) in PIPELINES.items():
        print(f"\n--- {PIPELINE_LABELS[key]} ---")
        records_by_id = run_pipeline(
            key, pipeline_fn, short_label, queries, judgments, resolver, use_judge
        )
        ordered = [records_by_id[q["id"]] for q in queries if q["id"] in records_by_id]
        per_pipeline[key] = ordered
        aggregates[key] = _aggregate(ordered, use_judge)
        aggregates[key]["_categories"] = _category_breakdown(ordered, use_judge)

    stats: dict[str, dict] = {}
    recall_scores = {
        key: [r["retrieval_metrics"]["recall@5"] for r in records]
        for key, records in per_pipeline.items()
    }
    stats["Recall@5"] = {
        pair_key: full_comparison(
            recall_scores[a_key], recall_scores[b_key],
            PIPELINE_LABELS[a_key], PIPELINE_LABELS[b_key],
        )
        for pair_key, a_key, b_key in PAIR_SPECS
    }

    if use_judge:
        judge_scores = {
            key: [float(r["judge"]["score"]) for r in records]
            for key, records in per_pipeline.items()
        }
        stats["Judge score (0-10)"] = {
            pair_key: full_comparison(
                judge_scores[a_key], judge_scores[b_key],
                PIPELINE_LABELS[a_key], PIPELINE_LABELS[b_key],
            )
            for pair_key, a_key, b_key in PAIR_SPECS
        }

    _print_results(aggregates, stats, use_judge, len(queries))

    ontology_usage = None
    if per_pipeline.get("pipeline_3_ontology_full"):
        ontology_usage = _ontology_usage(per_pipeline["pipeline_3_ontology_full"])
        _print_ontology_usage(ontology_usage)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator_model": GENERATOR_MODEL,
        "judge_model": LLM_JUDGE_MODEL if use_judge else None,
        "n_queries": len(queries),
        "judge_enabled": use_judge,
        "pipelines": {
            key: {
                "label": PIPELINE_LABELS[key],
                "aggregate": {k: v for k, v in aggregates[key].items() if k != "_categories"},
                "category_breakdown": aggregates[key]["_categories"],
            }
            for key in PIPELINES
        },
        "statistical_tests": stats,
        "ontology_usage": (
            {k: v for k, v in ontology_usage.items() if k != "example"}
            if ontology_usage else None
        ),
    }

    summary_file = METRICS_DIR / "full_eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"[OK] Summary saved to {summary_file}")
    print(f"[OK] Per-query records: {METRICS_DIR / 'full_eval_<pipeline>.json'}")

    if queries:
        _print_sample_outputs(per_pipeline, queries[0]["id"])


if __name__ == "__main__":
    main()
