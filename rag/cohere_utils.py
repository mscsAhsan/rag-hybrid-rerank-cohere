"""Thin helpers around the Cohere v2 API (embed, chat, rerank)."""
import os
import time

import cohere

EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025")
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

SYSTEM_PROMPT = (
    "You are a careful assistant in a Retrieval-Augmented Generation system. "
    "Answer ONLY from the numbered context passages provided. Cite the passages "
    "you used like [1], [2]. If the context does not contain the answer, say "
    "that you could not find it in the documents."
)


def get_client(api_key: str) -> "cohere.ClientV2":
    if not api_key or not api_key.strip():
        raise ValueError("Cohere API key is missing. Set COHERE_API_KEY or paste it in the sidebar.")
    return cohere.ClientV2(api_key=api_key.strip())


def _with_retry(fn, attempts: int = 4):
    """Retry on rate limits (trial keys are limited per minute)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if ("429" in msg or "rate" in msg or "too many" in msg) and i < attempts - 1:
                time.sleep(8 * (i + 1))
                continue
            raise


def embed_texts(client, texts, input_type: str, batch_size: int = 96):
    """Embed texts. input_type: 'search_document' for chunks, 'search_query' for questions."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _with_retry(
            lambda b=batch: client.embed(
                texts=b,
                model=EMBED_MODEL,
                input_type=input_type,
                embedding_types=["float"],
            )
        )
        emb = resp.embeddings
        vecs = getattr(emb, "float_", None) or getattr(emb, "float")
        vectors.extend(vecs)
    return vectors


def build_prompt(question: str, contexts) -> str:
    numbered = "\n\n".join(f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(contexts, 1))
    return f"Context passages:\n{numbered}\n\nQuestion: {question}\n\nAnswer:"


def generate_answer(client, question: str, contexts) -> str:
    resp = _with_retry(
        lambda: client.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, contexts)},
            ],
            temperature=0.2,
        )
    )
    return resp.message.content[0].text
