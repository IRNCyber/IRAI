"""Offline knowledge base with local vector search (RAG)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import KnowledgeConfig

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Local RAG knowledge base using FAISS and sentence-transformers."""

    def __init__(self, config: KnowledgeConfig, project_root: Path) -> None:
        self._config = config
        self._root = project_root
        self._embedder = None
        self._index = None
        self._documents: list[str] = []
        self._calendar: list[dict] = []

    def load(self) -> None:
        """Load the embedding model and build/load the FAISS index."""
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model for knowledge base...")
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self._load_calendar()
        self._load_documents()
        self._build_index()
        logger.info(
            "Knowledge base ready: %d documents indexed.", len(self._documents)
        )

    def _load_calendar(self) -> None:
        calendar_path = self._root / self._config.calendar
        if calendar_path.exists():
            with open(calendar_path) as f:
                self._calendar = json.load(f)
            for event in self._calendar:
                self._documents.append(
                    f"Calendar event: {event.get('title', 'Untitled')} "
                    f"on {event.get('date', 'unknown date')} "
                    f"at {event.get('time', 'unknown time')}. "
                    f"{event.get('notes', '')}"
                )
            logger.info("Loaded %d calendar events.", len(self._calendar))

    def _load_documents(self) -> None:
        docs_dir = self._root / self._config.user_docs
        if not docs_dir.exists():
            return

        for pdf_path in docs_dir.glob("*.pdf"):
            self._load_pdf(pdf_path)

        for txt_path in docs_dir.glob("*.txt"):
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            chunks = self._chunk_text(text)
            self._documents.extend(chunks)
            logger.info("Loaded %d chunks from %s", len(chunks), txt_path.name)

    def _load_pdf(self, path: Path) -> None:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            full_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

            chunks = self._chunk_text(full_text)
            self._documents.extend(chunks)
            logger.info("Loaded %d chunks from %s", len(chunks), path.name)
        except Exception:
            logger.warning("Failed to load PDF: %s", path.name, exc_info=True)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def _build_index(self) -> None:
        if not self._documents or self._embedder is None:
            return

        import faiss

        embeddings = self._embedder.encode(
            self._documents, show_progress_bar=False, convert_to_numpy=True
        )
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Search the knowledge base for relevant documents.

        Args:
            query: Natural language query.
            top_k: Number of results to return.

        Returns:
            List of relevant document chunks.
        """
        if self._index is None or self._embedder is None:
            return []

        import faiss

        q_emb = self._embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = self._index.search(q_emb, min(top_k, len(self._documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0.3:
                results.append(self._documents[idx])
        return results

    def get_calendar_summary(self) -> str:
        """Get a plain-text summary of upcoming calendar events."""
        if not self._calendar:
            return "No calendar events loaded."

        lines = []
        for event in self._calendar[:7]:
            title = event.get("title", "Untitled")
            date = event.get("date", "unknown")
            time_ = event.get("time", "")
            lines.append(f"- {title} on {date} {time_}".strip())
        return "Upcoming events:\n" + "\n".join(lines)
