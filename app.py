"""Streamlit demo for Approach 2: Hybrid RAG (BM25 + dense + RRF + Cohere Rerank)."""
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag.chunker import chunk_documents
from rag.cohere_utils import CHAT_MODEL, EMBED_MODEL, RERANK_MODEL, build_prompt, generate_answer, get_client
from rag.hybrid import HybridRAG
from rag.loader import load_folder, read_bytes

load_dotenv()
st.set_page_config(page_title="Hybrid RAG - BM25 + Dense + Rerank", page_icon="🔀", layout="wide")
st.title("🔀 Hybrid RAG — BM25 + Dense + Reciprocal Rank Fusion + Cohere Rerank")
st.caption("Load → Chunk → (BM25 index ‖ Embedding index) → Fuse (RRF) → Rerank → Generate")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Cohere API key", value=os.getenv("COHERE_API_KEY", ""), type="password")
    source = st.radio("Document collection", ["Sample documents", "Upload my own"])
    uploads = None
    if source == "Upload my own":
        uploads = st.file_uploader("PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    chunk_size = st.slider("Chunk size (words)", 50, 500, 100, 10)
    overlap = st.slider("Chunk overlap (words)", 0, 100, 20, 5)
    candidate_k = st.slider("Candidates per retriever", 3, 30, 8)
    top_n = st.slider("Final chunks sent to LLM", 1, 8, 3)
    use_rerank = st.checkbox("Use Cohere Rerank", value=True)
    build = st.button("🔨 Build indexes", type="primary", use_container_width=True)
    st.caption(f"Embed: `{EMBED_MODEL}`  \nRerank: `{RERANK_MODEL}`  \nChat: `{CHAT_MODEL}`")

if build:
    try:
        if source == "Sample documents":
            docs = load_folder("data/sample")
        else:
            docs = [{"name": f.name, "text": read_bytes(f.name, f.getvalue())} for f in (uploads or [])]
            docs = [d for d in docs if d["text"]]
        if not docs:
            st.sidebar.error("No readable documents found.")
        else:
            chunks = chunk_documents(docs, chunk_size, overlap)
            with st.spinner(f"Building BM25 + embedding indexes for {len(chunks)} chunks…"):
                rag = HybridRAG(get_client(api_key))
                rag.build_index(chunks)
            st.session_state.update(rag=rag, docs=docs, chunks=chunks, client=rag.client)
            st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(docs)} document(s).")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Indexing failed: {exc}")

tab1, tab2, tab3 = st.tabs(["1️⃣ Process & Index", "2️⃣ Retrieve, Fuse & Generate", "ℹ️ How it works"])

with tab1:
    if "rag" not in st.session_state:
        st.info("Choose a collection in the sidebar and click **Build indexes**.")
    else:
        rag, docs, chunks = st.session_state.rag, st.session_state.docs, st.session_state.chunks
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", len(docs))
        c2.metric("Chunks", len(chunks))
        c3.metric("Embedding dimension", rag.dim)
        st.dataframe(
            pd.DataFrame(
                {
                    "chunk id": [c["id"] for c in chunks],
                    "source": [c["source"] for c in chunks],
                    "words": [len(c["text"].split()) for c in chunks],
                    "preview": [c["text"][:110] + "…" for c in chunks],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab2:
    if "rag" not in st.session_state:
        st.info("Build the indexes first.")
    else:
        question = st.text_input("Ask a question", placeholder="What is the constant used in Reciprocal Rank Fusion?")
        if st.button("Ask", type="primary") and question.strip():
            try:
                with st.spinner("Searching (BM25 + dense), fusing and reranking…"):
                    res = st.session_state.rag.search(question, candidate_k, top_n, use_rerank)

                st.subheader("🔎 Retrieval stages")
                cols = st.columns(4)
                titles = [
                    ("BM25 (keyword)", "bm25", "BM25 score"),
                    ("Dense (vector)", "dense", "cosine"),
                    ("Fused (RRF)", "fused", "RRF score"),
                    ("Reranked (final)" if use_rerank else "Final (no rerank)", "final", "relevance"),
                ]
                for col, (title, key, label) in zip(cols, titles):
                    with col:
                        st.markdown(f"**{title}**")
                        if not res[key]:
                            st.caption("no matches")
                        for h in res[key]:
                            st.caption(f"#{h['rank']} · chunk {h['id']} · {label} {h['score']:.3f}")

                # rank comparison table
                all_ids = sorted({h["id"] for k in ("bm25", "dense", "fused", "final") for h in res[k]})
                rank_of = {k: {h["id"]: h["rank"] for h in res[k]} for k in ("bm25", "dense", "fused", "final")}
                st.markdown("**Rank of each candidate chunk at every stage** (– = not retrieved)")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "chunk id": all_ids,
                            "source": [st.session_state.chunks[i]["source"] for i in all_ids],
                            "BM25 rank": [rank_of["bm25"].get(i, "–") for i in all_ids],
                            "Dense rank": [rank_of["dense"].get(i, "–") for i in all_ids],
                            "RRF rank": [rank_of["fused"].get(i, "–") for i in all_ids],
                            "Final rank": [rank_of["final"].get(i, "–") for i in all_ids],
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("📄 Chunks sent to the LLM")
                for h in res["final"]:
                    with st.expander(f"#{h['rank']} · {h['source']} · chunk {h['id']} · score {h['score']:.3f}", expanded=h["rank"] == 1):
                        st.write(h["text"])

                with st.spinner("Generating answer with Cohere…"):
                    answer = generate_answer(st.session_state.client, question, res["final"])
                st.subheader("💬 Answer")
                st.write(answer)
                with st.expander("Prompt sent to the LLM"):
                    st.code(build_prompt(question, res["final"]))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Query failed: {exc}")

with tab3:
    st.markdown(
        """
**Why hybrid?** BM25 matches *exact words* (IDs, names, numbers); dense vectors match *meaning*. Each misses what the other finds.

1. **BM25** ranks chunks by keyword statistics.
2. **Dense search** ranks chunks by cosine similarity of Cohere embeddings.
3. **Reciprocal Rank Fusion** merges both lists: `score = Σ 1 / (60 + rank)`.
4. **Cohere Rerank** (cross-encoder) reads question + candidate together and re-scores the fused candidates.
5. The best *N* chunks go to **Cohere Command**, which answers with citations.
"""
    )
