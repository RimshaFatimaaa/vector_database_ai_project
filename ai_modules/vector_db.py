"""
Vector database utility module for the Streamlit app.

This module mirrors the logic originally prototyped inside
`notebooks/vector_database_test.ipynb` and exposes a reusable
API that the app can call for semantic lookups against the
interview knowledge base stored in ChromaDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


SEED_DATA = [
    {
        "question": "Tell me about yourself",
        "ideal_answer": (
            "A structured response focusing on background, experience, "
            "and relevance to the role."
        ),
    },
    {
        "question": "Describe a challenge you overcame",
        "ideal_answer": (
            "Use STAR method: Situation, Task, Action, Result. Highlight learning and impact."
        ),
    },
    {
        "question": "Explain OOP concepts",
        "ideal_answer": (
            "OOP includes Encapsulation, Inheritance, Polymorphism, "
            "Abstraction. Provide examples."
        ),
    },
    {
        "question": "What are Python decorators?",
        "ideal_answer": (
            "Decorators wrap a function to extend behavior without modifying the function."
        ),
    },
]


@dataclass
class VectorSearchResult:
    """Convenience container for vector search returns."""

    score: float
    question: str
    ideal_answer: str
    document_id: str


class InterviewVectorStore:
    """Lightweight wrapper around ChromaDB for knowledge lookups."""

    def __init__(
        self,
        db_subdir: str | Path | None = None,
        collection_name: str = "interview_knowledge",
        seed_data: Sequence[dict] | None = None,
    ) -> None:
        base_path = Path(__file__).resolve().parent.parent
        default_db_dir = base_path / "notebooks" / "vector_db"
        self.db_path = Path(db_subdir) if db_subdir else default_db_dir
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.seed_data = list(seed_data) if seed_data else list(SEED_DATA)

        # Heavy resources
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._client = PersistentClient(path=str(self.db_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._ensure_seed_documents()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, text: str, top_k: int = 3) -> List[VectorSearchResult]:
        """Return the closest knowledge entries for the provided text."""
        if not text.strip():
            return []

        k = max(1, min(top_k, len(self.seed_data)))
        query_embedding = self._embed(text)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        results: List[VectorSearchResult] = []
        documents = raw.get("documents", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        ids = raw.get("ids", [[]])[0]

        for doc, score, meta, doc_id in zip(documents, distances, metadatas, ids):
            results.append(
                VectorSearchResult(
                    score=score,
                    question=meta.get("question", "Unknown question"),
                    ideal_answer=doc,
                    document_id=doc_id,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _embed(self, text: str):
        return self._model.encode(text).tolist()

    def _ensure_seed_documents(self) -> None:
        """Populate the collection with baseline documents if missing."""
        try:
            existing_count = self._collection.count()
        except AttributeError:
            # Older Chroma versions may not have count(); fallback using get()
            existing_count = len(self._collection.get()["ids"])

        if existing_count >= len(self.seed_data):
            return

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for idx, item in enumerate(self.seed_data):
            doc_id = f"doc_{idx}"
            ids.append(doc_id)
            documents.append(item["ideal_answer"])
            embeddings.append(self._embed(item["ideal_answer"]))
            metadatas.append({"question": item["question"]})

        # upsert avoids duplicate errors if docs already exist
        add_fn = getattr(self._collection, "upsert", None) or self._collection.add
        add_fn(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )


__all__ = ["InterviewVectorStore", "VectorSearchResult"]

