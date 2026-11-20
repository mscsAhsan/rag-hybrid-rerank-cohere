# 🔀 Hybrid RAG: BM25 + Dense + RRF + Cohere Rerank (Streamlit)

**Approach 2 of 2** for the *RAG & Power BI* assignment.
A **hybrid retrieval** RAG pipeline: keyword search (BM25) and semantic vector search run in parallel, their rankings are merged with **Reciprocal Rank Fusion**, a **Cohere Rerank** cross-encoder re-scores the candidates, and Cohere Command writes the final cited answer. The Streamlit dashboard shows every retrieval stage side by side.

## Approach

```
                    ┌─► BM25 keyword search ───────┐
 Question ──────────┤                              ├─► RRF fusion ─► Cohere Rerank ─► top-N chunks
                    └─► Cohere embed + cosine ─────┘                                      │
                                                                                 Cohere Command
                                                                                          │
                                                                                 Answer + citations
```

| Stage | What happens | Where |
|---|---|---|
| Load / Chunk | PDF/TXT/MD → cleaned text → overlapping word chunks | `rag/loader.py`, `rag/chunker.py` |
| Sparse index | `BM25Okapi` over tokenised chunks | `rag/hybrid.py` |
| Dense index | Cohere `embed-english-v3.0`, normalised NumPy matrix | `rag/hybrid.py` |
| Fusion | `score = Σ 1/(60 + rank)` across both lists | `rag/hybrid.py` |
| Rerank | Cohere `rerank-v3.5` on fused candidates | `rag/hybrid.py` |
| Generate | Grounded prompt with citations | `rag/cohere_utils.py` |

**How it differs from Approach 1 (vanilla dense RAG):** adds a keyword retriever, rank fusion and a reranker, so exact-term questions (IDs, names, constants) and semantic questions are both handled. The app has a **"Use Cohere Rerank"** toggle and a rank-comparison table so you can see the effect.

## Document collection
`data/sample/` has four short articles on RAG, vector search, keyword/hybrid search and RAG evaluation. You can also **upload your own PDF/TXT/MD** files in the sidebar.

## Setup

**Requirements:** Python 3.10+ and a free Cohere API key from <https://dashboard.cohere.com/api-keys>.

```bash
git clone <this-repo-url>
cd rag-hybrid-rerank-cohere

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env        # Windows: copy .env.example .env
# open .env and paste your key:  COHERE_API_KEY=...
```

> 🔒 Never commit `.env` or your key. `.gitignore` already excludes `.env`.

## Run

```bash
streamlit run app.py
```

1. Pick **Sample documents** (or upload files) and click **Build indexes**.
2. **1️⃣ Process & Index** shows documents, chunks and embedding dimension.
3. **2️⃣ Retrieve, Fuse & Generate**: ask a question and compare the BM25, Dense, RRF and Reranked lists, the rank table, the chunks sent to the LLM, the answer and the prompt.

Try: *"What constant is used in Reciprocal Rank Fusion?"* (keyword-friendly) and *"How can I make sure the model does not answer from memory?"* (semantic).

## Tests (offline, no API key)
```bash
python -m pytest -q
```

## Dependencies
`streamlit`, `cohere`, `rank-bm25`, `numpy`, `pandas`, `pypdf`, `python-dotenv` (see `requirements.txt`).

## Project structure
```
rag-hybrid-rerank-cohere/
├── app.py              # Streamlit dashboard
├── rag/
│   ├── loader.py       # file loading
│   ├── chunker.py      # chunking
│   ├── cohere_utils.py # embed / chat helpers
│   └── hybrid.py       # BM25 + dense + RRF + rerank
├── data/sample/        # sample document collection
├── tests/test_core.py
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration
Override models via env vars if needed: `COHERE_CHAT_MODEL`, `COHERE_EMBED_MODEL`, `COHERE_RERANK_MODEL`.

## Troubleshooting
- **429 / rate limit** – trial keys are limited per minute; the app retries automatically, or wait a minute. Untick *Use Cohere Rerank* to save calls.
- **Model not found** – set the matching env var in `.env` to a currently available model.
- **Empty PDF text** – scanned PDFs have no text layer.
