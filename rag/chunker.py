"""Sliding-window word chunker with overlap."""


def chunk_text(text: str, source: str, chunk_size: int = 150, overlap: int = 30, start_id: int = 0):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, max(len(words), 1), step):
        piece = words[start : start + chunk_size]
        if not piece:
            break
        chunks.append({"id": start_id + len(chunks), "source": source, "text": " ".join(piece)})
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(docs, chunk_size: int = 150, overlap: int = 30):
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc["text"], doc["name"], chunk_size, overlap, start_id=len(chunks)))
    return chunks
