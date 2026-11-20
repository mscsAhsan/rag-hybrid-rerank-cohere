"""Document loading: PDF, TXT and Markdown."""
import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_bytes(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(BytesIO(data))
        return _clean("\n".join((p.extract_text() or "") for p in reader.pages))
    if ext in {".txt", ".md"}:
        return _clean(data.decode("utf-8", errors="ignore"))
    raise ValueError(f"Unsupported file type: {ext}")


def load_folder(folder: str):
    docs = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() in SUPPORTED:
            text = read_bytes(path.name, path.read_bytes())
            if text:
                docs.append({"name": path.name, "text": text})
    return docs
