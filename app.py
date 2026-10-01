from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from rag import answer_question, load_config

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def get_config():
    return load_config(BASE_DIR / "config.yaml")


def build_examples() -> list[str]:
    config = get_config()
    documents = config.get("documents", [])
    examples = [
        "What is the baggage claim deadline?",
        "How do I request a refund after cancellation?",
        "When do I need to check in for a flight?",
        "What should I do if my baggage is delayed?",
    ]
    if documents:
        return examples[: min(len(examples), 4)]
    return examples


st.set_page_config(page_title="Airline Policy Assistant", layout="wide")
st.title("Airline Policy Assistant")
st.caption("Information only, not legal advice. Based on the documents listed below; policies change.")

config = get_config()
documents = config.get("documents", [])
questions = build_examples()

with st.sidebar:
    st.selectbox("Chunking strategy", ["section", "fixed"], index=0, key="strategy")
    st.selectbox("Retrieval mode", ["vector", "hybrid"], index=1, key="mode")
    st.slider("Top K", 1, 10, 5, key="top_k")
    st.selectbox("Airline filter", ["all", *[item["airline"] for item in documents]], index=0, key="airline")
    st.checkbox("Comparison mode", value=False, key="comparison")
    st.markdown("### Indexed documents")
    for item in documents:
        st.markdown(f"- **{item['airline']}**: {item['doc_title']} ({item['doc_revision_date']} / {item['date_downloaded']})")

question = st.text_input("Ask a question about airline policy", value=questions[0])
if st.button("Ask"):
    airline_filter = None if st.session_state.airline == "all" else st.session_state.airline
    answer, meta = answer_question(question, strategy=st.session_state.strategy, mode=st.session_state.mode, airline_filter=airline_filter, comparison=st.session_state.comparison)
    if answer.answerable:
        st.success("Answer")
        st.write(answer.answer)
        st.markdown("**Confidence:** " + answer.confidence)
        for citation in answer.citations:
            with st.expander(f"{citation.airline} — {citation.section_id}"):
                st.write(citation.model_dump())
    else:
        st.warning("Not found in the provided documents")
        st.write(answer.refusal_reason)
    st.json({"retrieved": meta.get("retrieved", []), "retrieval_ms": meta.get("retrieval_ms"), "generation_ms": meta.get("generation_ms")})

if Path("results/summary.md").exists():
    with st.tabs(["Chat", "Evaluation Summary"])[1]:
        st.markdown(Path("results/summary.md").read_text(encoding="utf-8"))
else:
    st.caption("Run `python evaluate.py run --all` to generate evaluation results.")
