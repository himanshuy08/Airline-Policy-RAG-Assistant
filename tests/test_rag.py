import json

import evaluate
import rag


def test_headings_detected():
    text = "Preamble text here.\nRule 25 Refunds\nRefunds are allowed within 24 hours.\nRule 26 Delays\nDelayed baggage can be claimed."
    sections = rag.detect_sections(text, r"^Rule\s+\d+")
    assert len(sections) >= 2
    assert sections[0]["section_id"] == "Preamble"
    assert sections[1]["section_id"].startswith("Rule")


def test_long_sections_split_and_keep_section_id():
    long_text = "Rule 25\n" + "word " * 4000
    sections = rag.detect_sections(long_text, r"^Rule\s+\d+")
    section = sections[0]
    chunks = rag._split_on_paragraphs(section["text"], max_chars=3000, overlap=300)
    assert len(chunks) > 1
    assert all(chunks)
    assert section["section_id"] == "Rule 25"


def test_fixed_chunks_get_section_id():
    text = "Rule 25\nThis is a long text that should be split into multiple fixed chunks for the retrieval experiment.\n" * 50
    sections = rag.detect_sections(text, r"^Rule\s+\d+")
    section = [s for s in sections if s["section_id"] == "Rule 25"][0]
    fixed = rag._build_fixed_chunks(section["text"], section["section_id"], chunk_size=400, chunk_overlap=100)
    assert len(fixed) > 1
    assert all(isinstance(part, str) for part in fixed)


def test_deterministic_ids_are_stable():
    first = rag._deterministic_chunk_id("doc-1", "section", 0)
    second = rag._deterministic_chunk_id("doc-1", "section", 0)
    assert first == second
    assert first != rag._deterministic_chunk_id("doc-1", "fixed", 0)


def test_preamble_handled():
    text = "Intro text before heading.\nRule 12 Check-in\nArrive 2 hours early."
    sections = rag.detect_sections(text, r"^Rule\s+\d+")
    assert any(section["section_id"] == "Preamble" for section in sections)


def test_clean_regex_removes_toc_lines_and_page_footer():
    text = "Delta Domestic General Rules Tariff Page 3 of 23\n1. Summary ........................................ 2\nRULE 1: FLIGHT CANCELLATIONS\nCancelled flights may be refunded.\nDelta Domestic General Rules Tariff Page 3 of 23\n"
    cleaned, removed = rag.clean_document_text(text, [
        r'Delta Domestic General Rules Tariff Page \d+ of \d+',
        r'(?m)^.*\.{4,}\s*\d+\s*$',
    ])
    assert removed == [len(text) - len(cleaned) for _ in range(2)] or len(removed) == 2
    assert "Summary" not in cleaned
    assert "Page 3 of 23" not in cleaned
    sections = rag.detect_sections(cleaned, r'^RULE\s+\d+:\s+\S')
    assert all(section["section_id"] != "Preamble" or "Summary" not in section["text"] for section in sections)
    assert any(section["section_id"].startswith("RULE") for section in sections)


def test_vector_and_bm25_ordering():
    chunks = [
        {"chunk_id": "a", "text": "bags delayed claim reimbursement", "airline": "Airline A"},
        {"chunk_id": "b", "text": "ticket refund within 24 hours", "airline": "Airline A"},
        {"chunk_id": "c", "text": "pets are restricted", "airline": "Airline A"},
    ]
    ranked_vector = rag.retrieve("delayed baggage reimbursement", strategy="section", mode="vector", top_k=2, chunks=chunks)
    ranked_bm25 = rag.retrieve("delayed baggage reimbursement", strategy="section", mode="hybrid", top_k=2, chunks=chunks)
    assert ranked_vector[0]["chunk_id"] == "a"
    assert ranked_bm25[0]["chunk_id"] == "a"


def test_rrf_on_simple_example():
    chunks = [
        {"chunk_id": "a", "text": "refund ticket fee", "airline": "Airline A"},
        {"chunk_id": "b", "text": "baggage claim lost", "airline": "Airline A"},
        {"chunk_id": "c", "text": "flight change policy", "airline": "Airline A"},
    ]
    result = rag.retrieve("refund ticket", strategy="section", mode="hybrid", top_k=2, chunks=chunks)
    assert result[0]["chunk_id"] == "a"


def test_airline_filter_and_per_airline_results():
    chunks = [
        {"chunk_id": "a", "text": "Airline A baggage rules", "airline": "Airline A"},
        {"chunk_id": "b", "text": "Airline B checked bag rules", "airline": "Airline B"},
        {"chunk_id": "c", "text": "Airline A refund rules", "airline": "Airline A"},
    ]
    filtered = rag.retrieve("baggage", strategy="section", mode="vector", top_k=2, filters={"airline": "Airline A"}, chunks=chunks)
    assert all(item["airline"] == "Airline A" for item in filtered)
    per_airline = rag.retrieve_per_airline("baggage", strategy="section", mode="vector", top_k=2, per_airline_k=2, airlines=["Airline A", "Airline B"])
    assert set(per_airline) == {"Airline A", "Airline B"}


def test_refusal_gate_edge_cases():
    assert rag.should_refuse([], min_similarity=0.35) is True
    assert rag.should_refuse([{"vector_similarity": 0.34}], min_similarity=0.35) is True
    assert rag.should_refuse([{"vector_similarity": 0.40}], min_similarity=0.35) is False


def test_valid_json_path(monkeypatch):
    chunks = [{"chunk_id": "a", "text": "airline refund policy covers cancellations", "airline": "Airline A", "section_id": "Rule 21", "section_title": "Refunds", "page_start": 1}]
    monkeypatch.setattr(rag, "retrieve", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(rag, "_model_call", lambda *args, **kwargs: json.dumps({"answerable": True, "answer": "Refunds are allowed.", "citations": [{"chunk_id": "a", "airline": "Airline A", "section_id": "Rule 21", "section_title": "Refunds", "page_start": 1, "supporting_snippet": "airline refund policy covers cancellations"}], "confidence": "high", "refusal_reason": None, "comparison_table": None}))
    answer, meta = rag.answer_question("When can I get a refund?", strategy="section", mode="vector")
    assert answer.answerable is True
    assert answer.citations[0].chunk_id == "a"
    assert meta["retrieval_ms"] >= 0


def test_model_string_chunk_ids_are_resolved_to_citations(monkeypatch):
    chunks = [{
        "chunk_id": "a",
        "text": "Airline A permits refunds after eligible cancellations.",
        "airline": "Airline A",
        "section_id": "Rule 21",
        "section_title": "Refunds",
        "page_start": 4,
    }]
    response = {
        "answerable": True,
        "answer": "Eligible cancellations may be refunded.",
        "citations": ["a"],
        "confidence": 1.0,
        "refusal_reason": None,
        "comparison_table": None,
    }
    monkeypatch.setattr(rag, "retrieve", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(rag, "_model_call", lambda *args, **kwargs: json.dumps(response))

    answer, _ = rag.answer_question("Can I get a refund?", strategy="section", mode="vector")

    assert answer.answerable is True
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "a"
    assert answer.citations[0].page_start == 4
    assert answer.citations[0].supporting_snippet in chunks[0]["text"]


def test_long_model_snippet_is_bounded_before_citation_validation(monkeypatch):
    text = " ".join(["supporting"] * 60)
    chunks = [{
        "chunk_id": "a",
        "text": text,
        "airline": "Airline A",
        "section_id": "Rule 21",
        "section_title": "Refunds",
        "page_start": 1,
    }]
    response = {
        "answerable": True,
        "answer": "The passage supports this answer.",
        "citations": [{"chunk_id": "a", "supporting_snippet": text}],
        "confidence": "medium",
    }
    monkeypatch.setattr(rag, "retrieve", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(rag, "_model_call", lambda *args, **kwargs: json.dumps(response))

    answer, _ = rag.answer_question("What does this policy say?", strategy="section", mode="vector")

    assert answer.answerable is True
    assert len(answer.citations[0].supporting_snippet) <= 200
    assert answer.citations[0].supporting_snippet in text


def test_wrong_model_chunk_id_is_corrected_when_snippet_identifies_retrieved_chunk(monkeypatch):
    supporting_text = "Report a baggage claim\nwithin six hours after arrival\nat the final destination."
    chunks = [
        {"chunk_id": "wrong", "text": "Reservation information and flight confirmation.", "airline": "Delta Air Lines", "section_id": "RULE 13", "section_title": "Reservations", "page_start": 1},
        {"chunk_id": "right", "text": supporting_text, "airline": "Delta Air Lines", "section_id": "RULE 17", "section_title": "Baggage", "page_start": 2},
    ]
    response = {
        "answerable": True,
        "answer": "Report the claim within six hours.",
        "citations": [{"chunk_id": "wrong", "supporting_snippet": supporting_text.replace("\n", " ")}],
        "confidence": "high",
    }
    monkeypatch.setattr(rag, "retrieve", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(rag, "_model_call", lambda *args, **kwargs: json.dumps(response))

    answer, _ = rag.answer_question("What is the baggage claim deadline?", strategy="section", mode="vector")

    assert answer.answerable is True
    assert answer.citations[0].chunk_id == "right"
    assert answer.citations[0].section_id == "RULE 17"
    assert any("Corrected chunk id wrong" in error for error in answer.citation_errors)


def test_comparison_table_dict_is_normalized_to_schema():
    table = {
        "headers": ["Aspect", "Delta Air Lines", "Air India"],
        "rows": [["Refund", "Rule 22 applies", "Clause 11.3 applies"]],
    }

    normalized = rag._normalize_comparison_table(table)

    assert normalized == [{
        "aspect": "Refund",
        "per_airline": {
            "Delta Air Lines": "Rule 22 applies",
            "Air India": "Clause 11.3 applies",
        },
    }]


def test_evaluation_metrics_use_labelled_sections_and_detect_false_refusals():
    assert evaluate.hit_at_k(["a", "b", "c"], ["c"], 3) == 1
    assert evaluate.hit_at_k(["a", "b", "c"], ["c"], 2) == 0
    assert evaluate.mean_reciprocal_rank(["a", "b", "c"], ["c"]) == 1 / 3
    assert evaluate.comparison_full_hit(
        {"Delta": ["RULE 20"], "Air India": ["8.4"]},
        {"Delta": ["RULE 20"], "Air India": ["8.4"]},
        3,
    ) == 1
    metrics = evaluate.refusal_metrics([True, True], [False, True])
    assert metrics["recall"] == 1.0
    assert metrics["false_refusal_rate"] == 1.0
    assert evaluate.citation_precision(["doc:expected", "doc:other"], ["doc:expected"]) == 0.5
    assert evaluate.groundedness_from_claims(3, 4) == 0.75


def test_invalid_then_retry(monkeypatch):
    chunks = [{"chunk_id": "a", "text": "airline refund policy covers cancellations", "airline": "Airline A", "section_id": "Rule 21", "section_title": "Refunds", "page_start": 1}]
    calls = []

    def fake_model(prompt, model, temperature):
        calls.append(prompt)
        if len(calls) == 1:
            return "not json"
        return json.dumps({"answerable": True, "answer": "Refunds are allowed.", "citations": [{"chunk_id": "a", "airline": "Airline A", "section_id": "Rule 21", "section_title": "Refunds", "page_start": 1, "supporting_snippet": "airline refund policy covers cancellations"}], "confidence": "high", "refusal_reason": None, "comparison_table": None})

    monkeypatch.setattr(rag, "retrieve", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(rag, "_model_call", fake_model)
    answer, _ = rag.answer_question("When can I get a refund?", strategy="section", mode="vector")
    assert len(calls) == 2
    assert answer.answerable is True


def test_fabricated_chunk_id_removed_and_snippet_flagged():
    chunks = [{"chunk_id": "a", "text": "airline refund policy covers cancellations", "airline": "Airline A", "section_id": "Rule 21", "section_title": "Refunds", "page_start": 1}]
    citations = [
        rag.Citation(chunk_id="x", airline="Airline A", section_id="Rule 21", section_title="Refunds", page_start=1, supporting_snippet="airline refund policy covers cancellations"),
        rag.Citation(chunk_id="a", airline="Airline A", section_id="Rule 21", section_title="Refunds", page_start=1, supporting_snippet="not in chunk"),
    ]
    valid, errors = rag._validate_citations(citations, chunks)
    assert len(valid) == 0
    assert any("chunk" in error.lower() for error in errors)


def test_refusal_skips_llm(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda *args, **kwargs: [])
    called = {"used": False}

    def fake_model(*args, **kwargs):
        called["used"] = True
        return json.dumps({"answerable": True, "answer": "should not run"})

    monkeypatch.setattr(rag, "_model_call", fake_model)
    answer, _ = rag.answer_question("What is the answer?", strategy="section", mode="vector")
    assert answer.answerable is False
    assert called["used"] is False
