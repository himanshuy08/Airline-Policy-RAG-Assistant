from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

import rag
from rag import PolicyAnswer, load_config

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
QUESTIONS_PATH = BASE_DIR / "data" / "questions.csv"
CACHE_PATH = RESULTS_DIR / "llm_cache.json"
logger = logging.getLogger(__name__)
_MODEL_CACHE: Dict[str, str] = {}
_RAW_MODEL_CALL = rag._model_call
_CACHE_HITS = 0
_CACHE_MISSES = 0


def load_questions(path: str | Path = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    dataframe = pd.read_csv(path).fillna("")
    required = {"id", "question", "type", "airline", "expected_sections", "answerable", "notes"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return dataframe.to_dict(orient="records")


def parse_expected_sections(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _parse_answerable(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid answerable value: {value!r}")


def _split_section_ref(value: str) -> Tuple[str, str]:
    doc_id, separator, section_id = value.partition(":")
    if not separator or not doc_id or not section_id:
        raise ValueError(f"Expected section reference 'doc_id:section_id', got {value!r}")
    return doc_id.strip(), section_id.strip()


def validate_questions(questions: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> None:
    known = {(str(chunk["doc_id"]), str(chunk["section_id"])) for chunk in chunks}
    issues: List[str] = []
    for row in questions:
        try:
            answerable = _parse_answerable(row["answerable"])
            expected = parse_expected_sections(row["expected_sections"])
            if answerable and not expected:
                issues.append(f"{row['id']}: answerable row has no expected_sections")
            for reference in expected:
                if _split_section_ref(reference) not in known:
                    issues.append(f"{row['id']}: unknown expected section {reference!r}")
            if row["type"] == "comparison" and answerable:
                airlines = {doc.get("airline") for doc in load_config().get("documents", [])}
                expected_doc_ids = {_split_section_ref(item)[0] for item in expected}
                expected_airlines = {
                    doc.get("airline")
                    for doc in load_config().get("documents", [])
                    if doc.get("doc_id") in expected_doc_ids
                }
                if len(expected_airlines & airlines) < 2:
                    issues.append(f"{row['id']}: comparison must label an expected section for both airlines")
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"{row.get('id', '<unknown>')}: {error}")
    if issues:
        raise ValueError("Question dataset validation failed:\n- " + "\n- ".join(issues))


def hit_at_k(retrieved_sections: Iterable[str], expected_sections: Iterable[str], k: int) -> int:
    return int(any(section in set(expected_sections) for section in list(retrieved_sections)[:k]))


def mean_reciprocal_rank(retrieved_sections: Iterable[str], expected_sections: Iterable[str]) -> float:
    expected = set(expected_sections)
    for rank, section in enumerate(retrieved_sections, start=1):
        if section in expected:
            return 1.0 / rank
    return 0.0


def comparison_full_hit(
    retrieved_by_airline: Dict[str, Iterable[str]],
    expected_by_airline: Dict[str, Iterable[str]],
    k: int,
) -> int:
    return int(bool(expected_by_airline) and all(
        hit_at_k(retrieved_by_airline.get(airline, []), expected, k)
        for airline, expected in expected_by_airline.items()
    ))


def refusal_metrics(predicted_refused: Iterable[bool], answerable: Iterable[bool]) -> Dict[str, float]:
    actuals = list(answerable)
    predicts = list(predicted_refused)
    if len(actuals) != len(predicts):
        raise ValueError("predicted_refused and answerable must have equal lengths")
    tp = sum(1 for actual, pred in zip(actuals, predicts) if not actual and pred)
    fp = sum(1 for actual, pred in zip(actuals, predicts) if actual and pred)
    tn = sum(1 for actual, pred in zip(actuals, predicts) if actual and not pred)
    fn = sum(1 for actual, pred in zip(actuals, predicts) if not actual and not pred)
    return {
        "accuracy": (tp + tn) / max(1, len(actuals)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "false_refusal_rate": fp / max(1, sum(actuals)),
    }


def citation_precision(cited_sections: Iterable[str], expected_sections: Iterable[str]) -> float:
    cited = list(cited_sections)
    expected = set(expected_sections)
    return sum(1 for section in cited if section in expected) / len(cited) if cited else 0.0


def groundedness_from_claims(supported_claims: int, total_claims: int) -> Optional[float]:
    if total_claims <= 0:
        return None
    return supported_claims / total_claims


def _cache_key(prompt: str, model: str, temperature: float) -> str:
    payload = json.dumps(
        {"prompt": prompt, "model": model, "temperature": temperature},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cache() -> None:
    global _MODEL_CACHE
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict) or any(not isinstance(value, str) for value in loaded.values()):
            raise ValueError(f"Invalid LLM cache format in {CACHE_PATH}")
        _MODEL_CACHE = loaded


def _cached_model_call(prompt: str, model_name: str, temperature: float = 0.0) -> str:
    global _CACHE_HITS, _CACHE_MISSES
    key = _cache_key(prompt, model_name, temperature)
    if key in _MODEL_CACHE:
        _CACHE_HITS += 1
    else:
        _CACHE_MISSES += 1
        _MODEL_CACHE[key] = _RAW_MODEL_CALL(prompt, model_name, temperature)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path = CACHE_PATH.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(_MODEL_CACHE, handle, ensure_ascii=False, indent=2)
        temporary_path.replace(CACHE_PATH)
    return _MODEL_CACHE[key]


def _expected_by_airline(expected: List[str], documents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    doc_to_airline = {str(doc["doc_id"]): str(doc["airline"]) for doc in documents}
    grouped: Dict[str, List[str]] = {}
    for reference in expected:
        doc_id, section_id = _split_section_ref(reference)
        airline = doc_to_airline.get(doc_id, doc_id)
        grouped.setdefault(airline, []).append(section_id)
    return grouped


def _section_key(chunk: Dict[str, Any]) -> str:
    return f"{chunk.get('doc_id', '')}:{chunk.get('section_id', '')}"


def _judge_groundedness(
    answer: PolicyAnswer,
    retrieved: List[Dict[str, Any]],
    model: str,
    temperature: float,
) -> Tuple[Optional[int], Optional[int]]:
    if not answer.answerable or not answer.citations:
        return None, None
    cited_ids = {citation.chunk_id for citation in answer.citations}
    cited_passages = [
        f"{chunk['chunk_id']}: {chunk.get('text', '')}"
        for chunk in retrieved
        if str(chunk.get("chunk_id")) in cited_ids
    ]
    if not cited_passages:
        return None, None
    prompt = (
        "Evaluate whether each factual claim in the answer is supported by the cited passages. "
        "Use ONLY those passages. Split the answer into atomic factual claims. Return JSON only: "
        '{"claims":[{"claim":"...", "supported":true}]} . '
        "Do not count stylistic language or disclaimers as claims.\n\n"
        f"Answer:\n{answer.answer}\n\nCited passages:\n" + "\n\n".join(cited_passages)
    )
    response = _cached_model_call(prompt, model, temperature)
    try:
        payload = json.loads(response)
        claims = payload["claims"]
        if not isinstance(claims, list):
            return None, None
        supported = sum(1 for item in claims if isinstance(item, dict) and item.get("supported") is True)
        return supported, len(claims)
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Groundedness judge returned invalid output; excluding this answer from judge metrics")
        return None, None


def _run_question(
    row: Dict[str, Any],
    strategy: str,
    mode: str,
    threshold: float,
    top_k: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    answerable = _parse_answerable(row["answerable"])
    airline = str(row["airline"]).strip()
    airline_filter = airline if airline not in {"both", "none", ""} and row["type"] != "comparison" else None
    misses_before_answer = _CACHE_MISSES
    answer, metadata = rag.answer_question(
        str(row["question"]),
        strategy=strategy,
        mode=mode,
        airline_filter=airline_filter,
        comparison=row["type"] == "comparison",
        top_k=top_k,
        min_similarity=threshold,
    )
    answer_was_generated = _CACHE_MISSES > misses_before_answer
    retrieved = metadata.get("retrieved", [])
    expected = parse_expected_sections(row["expected_sections"])
    expected_airlines = _expected_by_airline(expected, config.get("documents", []))
    per_airline = metadata.get("per_airline") or {}
    retrieved_by_airline = {
        name: [_section_key(chunk).split(":", 1)[1] for chunk in chunks]
        for name, chunks in per_airline.items()
    }
    citation_keys = []
    for citation in answer.citations:
        doc_id = next(
            (doc["doc_id"] for doc in config.get("documents", []) if doc["airline"] == citation.airline),
            citation.airline,
        )
        citation_keys.append(f"{doc_id}:{citation.section_id}")
    support, total = _judge_groundedness(
        answer,
        retrieved,
        str(config.get("settings", {}).get("chat_model", "gemma3:4b")),
        float(config.get("settings", {}).get("temperature", 0.0)),
    )
    rank = next(
        (index for index, chunk in enumerate(retrieved, start=1) if _section_key(chunk) in expected),
        None,
    )
    invalid_errors = [error for error in answer.citation_errors if not error.startswith("Corrected chunk id ")]
    return {
        "id": row["id"],
        "question": row["question"],
        "type": row["type"],
        "airline": row["airline"],
        "expected_sections": ";".join(expected),
        "gold_answerable": answerable,
        "predicted_answerable": answer.answerable,
        "refused": not answer.answerable,
        "retrieved_sections": ";".join(_section_key(chunk) for chunk in retrieved),
        "retrieved_sections_by_airline": json.dumps(
            {name: [f"{name}:{section}" for section in sections] for name, sections in retrieved_by_airline.items()},
            ensure_ascii=False,
        ),
        "expected_by_airline": json.dumps(expected_airlines, ensure_ascii=False),
        "hit_at_1": hit_at_k([_section_key(chunk) for chunk in retrieved], expected, 1) if answerable else "",
        "hit_at_3": hit_at_k([_section_key(chunk) for chunk in retrieved], expected, 3) if answerable else "",
        "hit_at_5": hit_at_k([_section_key(chunk) for chunk in retrieved], expected, 5) if answerable else "",
        "mrr": mean_reciprocal_rank([_section_key(chunk) for chunk in retrieved], expected) if answerable else "",
        "comparison_full_hit_at_3": comparison_full_hit(retrieved_by_airline, expected_airlines, 3) if row["type"] == "comparison" and answerable else "",
        "answer": answer.answer,
        "citations": json.dumps([citation.model_dump() for citation in answer.citations], ensure_ascii=False),
        "citation_sections": ";".join(citation_keys),
        "citation_errors": ";".join(answer.citation_errors),
        "invalid_citation_count": len(invalid_errors),
        "judge_supported_claims": "" if support is None else support,
        "judge_total_claims": "" if total is None else total,
        "judge_groundedness": "" if support is None or total is None or total == 0 else support / total,
        "retrieval_ms": metadata.get("retrieval_ms", 0.0),
        "generation_ms": float(metadata.get("generation_ms", 0.0)) if answer_was_generated else "",
        "latency_ms": (
            float(metadata.get("retrieval_ms", 0.0)) + float(metadata.get("generation_ms", 0.0))
            if answer_was_generated else ""
        ),
        "retrieval_rank": rank if rank is not None else "",
        "refusal_reason": answer.refusal_reason or "",
    }


def _run_retrieval_only(
    row: Dict[str, Any],
    strategy: str,
    mode: str,
    threshold: float,
    top_k: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    if row["type"] == "comparison":
        per_airline_k = int(config.get("settings", {}).get("retrieval", {}).get("per_airline_k", 3))
        by_airline = rag.retrieve_per_airline(
            str(row["question"]),
            strategy=strategy,
            mode=mode,
            top_k=top_k,
            per_airline_k=min(top_k, per_airline_k),
        )
        retrieved = [chunk for airline_chunks in by_airline.values() for chunk in airline_chunks]
    else:
        airline = str(row["airline"]).strip()
        filters = {"airline": airline} if airline not in {"both", "none", ""} else None
        by_airline = None
        retrieved = rag.retrieve(
            str(row["question"]),
            strategy=strategy,
            mode=mode,
            top_k=top_k,
            filters=filters,
        )
    retrieval_ms = (time.perf_counter() - started) * 1000.0
    expected = parse_expected_sections(row["expected_sections"])
    expected_airlines = _expected_by_airline(expected, config.get("documents", []))
    retrieved_by_airline = {
        name: [chunk["section_id"] for chunk in airline_chunks]
        for name, airline_chunks in (by_airline or {}).items()
    }
    answerable = _parse_answerable(row["answerable"])
    refused = rag.should_refuse(retrieved, min_similarity=threshold)
    sections = [_section_key(chunk) for chunk in retrieved]
    first_rank = next((rank for rank, key in enumerate(sections, start=1) if key in expected), None)
    return {
        "id": row["id"],
        "question": row["question"],
        "type": row["type"],
        "airline": row["airline"],
        "expected_sections": ";".join(expected),
        "gold_answerable": answerable,
        "predicted_answerable": "",
        "refused": refused,
        "retrieved_sections": ";".join(sections),
        "retrieved_sections_by_airline": json.dumps(retrieved_by_airline, ensure_ascii=False),
        "expected_by_airline": json.dumps(expected_airlines, ensure_ascii=False),
        "hit_at_1": hit_at_k(sections, expected, 1) if answerable else "",
        "hit_at_3": hit_at_k(sections, expected, 3) if answerable else "",
        "hit_at_5": hit_at_k(sections, expected, 5) if answerable else "",
        "mrr": mean_reciprocal_rank(sections, expected) if answerable else "",
        "comparison_full_hit_at_3": comparison_full_hit(retrieved_by_airline, expected_airlines, 3) if row["type"] == "comparison" and answerable else "",
        "answer": "",
        "citations": "",
        "citation_sections": "",
        "citation_errors": "",
        "invalid_citation_count": 0,
        "judge_supported_claims": "",
        "judge_total_claims": "",
        "judge_groundedness": "",
        "retrieval_ms": retrieval_ms,
        "generation_ms": "",
        "latency_ms": "",
        "retrieval_rank": first_rank if first_rank is not None else "",
        "refusal_reason": "retrieval_threshold" if refused else "",
    }


def _mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [float(value) for value in values if value != "" and value is not None]
    return statistics.mean(numeric) if numeric else None


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected_answerability = [bool(row["gold_answerable"]) for row in rows]
    predicted_refusals = [bool(row["refused"]) for row in rows]
    metrics = refusal_metrics(predicted_refusals, expected_answerability)
    answerable_rows = [row for row in rows if row["gold_answerable"]]
    has_answer_results = any(row.get("answer", "") for row in rows)
    metrics.update({
        "hit@1": _mean(row["hit_at_1"] for row in answerable_rows),
        "hit@3": _mean(row["hit_at_3"] for row in answerable_rows),
        "hit@5": _mean(row["hit_at_5"] for row in answerable_rows),
        "mrr": _mean(row["mrr"] for row in answerable_rows),
        "comparison_full_hit@3": _mean(
            row["comparison_full_hit_at_3"] for row in answerable_rows
            if row["comparison_full_hit_at_3"] != ""
        ),
        "citation_coverage": _mean(
            int(bool(row["citation_sections"]) and any(ref in row["citation_sections"].split(";") for ref in row["expected_sections"].split(";")))
            for row in answerable_rows if row["predicted_answerable"]
        ),
        "citation_precision": _mean(
            citation_precision(row["citation_sections"].split(";"), row["expected_sections"].split(";"))
            for row in rows if row["citation_sections"]
        ),
        "invalid_citation_rate": (
            sum(int(row["invalid_citation_count"]) for row in rows)
            / max(1, sum(len(row["citation_sections"].split(";")) + int(row["invalid_citation_count"]) for row in rows))
        ) if has_answer_results else None,
        "judge_groundedness": (
            sum(int(row["judge_supported_claims"]) for row in rows if row["judge_supported_claims"] != "")
            / max(1, sum(int(row["judge_total_claims"]) for row in rows if row["judge_total_claims"] != ""))
        ) if has_answer_results else None,
        "mean_latency_ms": (
            _mean(row["latency_ms"] for row in rows)
            if all(row["latency_ms"] != "" for row in rows) else None
        ),
        "p95_latency_ms": (
            float(pd.Series([float(row["latency_ms"]) for row in rows]).quantile(0.95))
            if rows and all(row["latency_ms"] != "" for row in rows) else None
        ),
    })
    return metrics


def _format_metric(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def _threshold_sweep(
    questions: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    vectors: List[Tuple[float, bool]] = []
    config = load_config()
    for row in questions:
        answerable = _parse_answerable(row["answerable"])
        airline_filter = row["airline"] if row["airline"] not in {"both", "none", ""} else None
        filters = {"airline": airline_filter} if airline_filter and row["type"] != "comparison" else None
        retrieved = rag.retrieve(
            str(row["question"]),
            strategy="section",
            mode="hybrid",
            top_k=top_k,
            filters=filters,
            chunks=chunks,
        )
        best = max((float(item.get("vector_similarity", 0.0)) for item in retrieved), default=0.0)
        vectors.append((best, answerable))
    results = []
    for step in range(15):
        threshold = round(0.10 + step * 0.05, 2)
        refusal = refusal_metrics(
            [best < threshold for best, _ in vectors],
            [answerable for _, answerable in vectors],
        )
        balanced_accuracy = (refusal["recall"] + 1.0 - refusal["false_refusal_rate"]) / 2.0
        results.append({"threshold": threshold, "balanced_accuracy": balanced_accuracy, **refusal})
    return results


def _write_manual_review(rows: List[Dict[str, Any]], output_path: Path) -> None:
    candidates = [row for row in rows if row["answer"]]
    sample = random.Random(42).sample(candidates, min(20, len(candidates)))
    fields = [
        "id", "question", "answer", "citations", "judge_supported_claims",
        "judge_total_claims", "human_supported_claims", "human_total_claims",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sample:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_summary(experiments: List[Dict[str, Any]], question_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    lines = [
        "# Airline Policy RAG Evaluation",
        "",
        "Results generated from the configured local policy PDFs and `data/questions.csv`.",
        "In `run --all`, retrieval and refusal metrics cover all four configurations. To avoid repeating slow local-model calls, answer, citation, and judge metrics are generated for `section_vector` only; use `run --config <name>` to fully evaluate another configuration. Latency is omitted when model responses came from the cache.",
        "",
        "| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR | Refusal accuracy | False-refusal rate | Citation coverage | Judge groundedness | Mean latency ms | P95 latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in experiments:
        lines.append(
            f"| {result['config']} | {_format_metric(result['hit@1'])} | {_format_metric(result['hit@3'])} | "
            f"{_format_metric(result['hit@5'])} | {_format_metric(result['mrr'])} | {_format_metric(result['accuracy'])} | "
            f"{_format_metric(result['false_refusal_rate'])} | {_format_metric(result['citation_coverage'])} | "
            f"{_format_metric(result['judge_groundedness'])} | {_format_metric(result['mean_latency_ms'], 1)} | "
            f"{_format_metric(result['p95_latency_ms'], 1)} |"
        )
    base_name = "section_vector" if "section_vector" in question_rows else next(iter(question_rows), "")
    if base_name:
        ranked = sorted(
            (row for row in question_rows[base_name] if row["gold_answerable"]),
            key=lambda row: (int(row["retrieval_rank"]) if row["retrieval_rank"] != "" else 10_000, row["id"]),
            reverse=True,
        )[:10]
        lines.extend(["", "## Ten lowest retrieval ranks", "", "| ID | Question | Best expected-section rank |", "|---|---|---:|"])
        for row in ranked:
            lines.append(f"| {row['id']} | {row['question']} | {row['retrieval_rank'] or 'not retrieved'} |")
    lines.append("")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(config_names: List[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _load_cache()
    rag._model_call = _cached_model_call
    config = load_config()
    questions = load_questions()
    top_k = int(config.get("settings", {}).get("retrieval", {}).get("top_k", 5))
    experiment_rows = []
    all_question_results: Dict[str, List[Dict[str, Any]]] = {}

    answer_generation_config = "section_vector" if "section_vector" in config_names else config_names[0]
    for config_name in config_names:
        try:
            strategy, mode = config_name.split("_", 1)
        except ValueError as error:
            raise ValueError(f"Unknown configuration {config_name!r}; expected strategy_mode") from error
        if strategy not in {"section", "fixed"} or mode not in {"vector", "hybrid"}:
            raise ValueError(f"Unknown configuration {config_name!r}")
        corpus = rag.ingest_documents(strategy=strategy).get(strategy, [])
        validate_questions(questions, corpus)
        config_results = []
        for row in questions:
            logger.info("Evaluating %s: %s", config_name, row["id"])
            if config_name == answer_generation_config:
                result = _run_question(
                    row,
                    strategy=strategy,
                    mode=mode,
                    threshold=float(config.get("settings", {}).get("min_similarity", 0.35)),
                    top_k=top_k,
                    config=config,
                )
            else:
                result = _run_retrieval_only(
                    row,
                    strategy=strategy,
                    mode=mode,
                    threshold=float(config.get("settings", {}).get("min_similarity", 0.35)),
                    top_k=top_k,
                    config=config,
                )
            config_results.append(result)
        all_question_results[config_name] = config_results
        pd.DataFrame(config_results).to_csv(RESULTS_DIR / f"per_question_{config_name}.csv", index=False)
        aggregate = {"config": config_name, **_aggregate(config_results)}
        experiment_rows.append(aggregate)

    pd.DataFrame(experiment_rows).to_csv(RESULTS_DIR / "experiments.csv", index=False)
    _write_summary(experiment_rows, all_question_results)
    if answer_generation_config in all_question_results:
        _write_manual_review(
            all_question_results[answer_generation_config],
            RESULTS_DIR / "manual_review.csv",
        )

    sweep_corpus = rag.ingest_documents(strategy="section").get("section", [])
    sweep_rows = _threshold_sweep(questions, sweep_corpus, top_k)
    pd.DataFrame(sweep_rows).to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    best_threshold = max(sweep_rows, key=lambda row: (row["balanced_accuracy"], -row["threshold"]))
    print(
        f"Best measured threshold: {best_threshold['threshold']:.2f} "
        f"(balanced_accuracy={best_threshold['balanced_accuracy']:.3f})"
    )
    print(f"Wrote evaluation outputs to {RESULTS_DIR}")


def run_human_review() -> None:
    path = RESULTS_DIR / "manual_review.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `python evaluate.py run --all` first")
    dataframe = pd.read_csv(path)
    required = {"human_supported_claims", "human_total_claims", "judge_supported_claims", "judge_total_claims"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"manual_review.csv is missing columns: {sorted(missing)}")
    completed = dataframe.dropna(subset=["human_supported_claims", "human_total_claims"])
    if completed.empty:
        print("No human review rows have been filled in yet.")
        return
    human_supported = pd.to_numeric(completed["human_supported_claims"], errors="raise").sum()
    human_total = pd.to_numeric(completed["human_total_claims"], errors="raise").sum()
    judge_supported = pd.to_numeric(completed["judge_supported_claims"], errors="coerce").sum()
    judge_total = pd.to_numeric(completed["judge_total_claims"], errors="coerce").sum()
    human_score = groundedness_from_claims(int(human_supported), int(human_total))
    judge_score = groundedness_from_claims(int(judge_supported), int(judge_total))
    print(f"Human groundedness: {human_score:.3f}" if human_score is not None else "Human groundedness unavailable: zero human claims")
    print(f"Judge groundedness on reviewed rows: {judge_score:.3f}" if judge_score is not None else "Judge groundedness unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate airline policy RAG against a labelled question set")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--all", action="store_true")
    run_parser.add_argument("--config", choices=["section_vector", "section_hybrid", "fixed_vector", "fixed_hybrid"])
    subparsers.add_parser("sweep")
    subparsers.add_parser("human")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "run":
        if args.all:
            names = ["section_vector", "section_hybrid", "fixed_vector", "fixed_hybrid"]
        elif args.config:
            names = [args.config]
        else:
            parser.error("run requires --all or --config")
        run_evaluation(names)
    elif args.command == "sweep":
        questions = load_questions()
        corpus = rag.ingest_documents(strategy="section").get("section", [])
        validate_questions(questions, corpus)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        rows = _threshold_sweep(
            questions,
            corpus,
            int(load_config().get("settings", {}).get("retrieval", {}).get("top_k", 5)),
        )
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
        print(json.dumps(rows, indent=2))
    elif args.command == "human":
        run_human_review()


if __name__ == "__main__":
    main()
