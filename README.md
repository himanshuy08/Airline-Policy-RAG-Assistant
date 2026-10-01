# Airline Policy RAG Assistant

A citation-grounded question-answering application for airline policies. The assistant retrieves relevant sections from Delta Air Lines' U.S. domestic tariff and Air India's Conditions of Carriage, then generates an answer with citations to the source documents.

## Features

- **PDF ingestion and cleanup:** Extracts policy text and supports configured cleanup patterns for running headers, footers, and table-of-contents entries.
- **Section-aware indexing:** Organizes Delta content by rules and Air India content by articles and numbered clauses.
- **Multiple retrieval strategies:** Supports vector and hybrid retrieval, airline-specific questions, and cross-airline comparisons.
- **Grounded answers:** Returns source-section citations and refuses questions that are not supported by the indexed documents.
- **Local model inference:** Uses Ollama for chat generation and embeddings.
- **Evaluation workflow:** Includes a 43-question benchmark, retrieval and refusal metrics, threshold analysis, and a manual-review workflow.
- **Web interface:** Provides a Streamlit application for asking policy questions.

## Tech Stack

`Python` · `LangChain` · `ChromaDB` · `Ollama` · `Gemma 3` · `nomic-embed-text` · `Streamlit`

## Requirements

- Python and pip
- [Ollama](https://ollama.com/) installed and running
- The configured models:

  ```bash
  ollama pull gemma3:4b
  ollama pull nomic-embed-text:latest
  ```

## Setup

1. Install the Python dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Place the policy PDFs in `data/raw/` and configure their paths and metadata in `config.yaml`.

3. Ingest the documents:

   ```bash
   python rag.py ingest
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

## Evaluation

Run the benchmark across all configured retrieval strategies:

```bash
python evaluate.py run --all
```

The 43-question benchmark covers single-airline questions, cross-airline comparisons, and questions that should not be answerable from the policy documents. In the recorded evaluation, the best retrieval configuration achieved **84.8% Hit@5** and **0.681 MRR**. Generated-answer groundedness was **97.9%** for the baseline configuration; this was measured separately from the best retrieval result.

Evaluation outputs are written to `results/`.

Run the test suite with:

```bash
pytest
```

## Document Scope

Delta's configured tariff covers U.S. domestic travel. Air India's Conditions of Carriage cover domestic and international travel, and some terms refer to separate carrier regulations. Answers and comparisons should be interpreted within the scope of the cited document sections.

## Configured Source Documents

- **Delta Air Lines:** Domestic General Rules Tariff; revision date August 18, 2026; downloaded October 2, 2026.
- **Air India:** Conditions of Carriage; no revision date stated in the downloaded document; downloaded October 2, 2026.

The source PDFs are not included in this repository. Download them from the URLs in `config.yaml` and place them in `data/raw/` before ingestion.

## Disclaimer

This application is for informational purposes only and is not legal advice. Always check the current airline documents and applicable regulations for authoritative information.
