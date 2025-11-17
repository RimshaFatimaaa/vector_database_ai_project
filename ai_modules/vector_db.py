"""
Vector Database Module - Step 5 Implementation
Purpose: Store and retrieve interview data (questions, answers, examples, transcripts)

According to PRD Step 5:
- Store canonical answers, good/bad examples, transcripts as embeddings
- Enable retrieval of similar examples for feedback
- Use Chroma for local dev; Sentence-Transformers for embeddings
- Deliverable: A searchable knowledge base of interview material to improve evaluations
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional, Dict, Literal
from datetime import datetime

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


# Comprehensive seed data with different example types
CANONICAL_ANSWERS = [
    {
        "question": "Tell me about yourself",
        "answer": (
            "A structured response focusing on background, experience, "
            "and relevance to the role."
        ),
        "type": "canonical",
        "category": "hr_behavioral",
    },
    {
        "question": "Describe a challenge you overcame",
        "answer": (
            "Use STAR method: Situation, Task, Action, Result. Highlight learning and impact."
        ),
        "type": "canonical",
        "category": "hr_behavioral",
    },
    {
        "question": "Explain OOP concepts",
        "answer": (
            "OOP includes Encapsulation, Inheritance, Polymorphism, "
            "Abstraction. Provide examples."
        ),
        "type": "canonical",
        "category": "technical",
    },
    {
        "question": "What are Python decorators?",
        "answer": (
            "Decorators wrap a function to extend behavior without modifying the function."
        ),
        "type": "canonical",
        "category": "technical",
    },
]

GOOD_EXAMPLES = [
    {
        "question": "Tell me about yourself",
        "answer": (
            "I'm a software engineer with 5 years of experience in full-stack development. "
            "I specialize in Python and JavaScript, and I've led several projects that improved "
            "system performance by 40%. I'm excited about this role because it aligns with my "
            "passion for building scalable applications."
        ),
        "type": "good_example",
        "category": "hr_behavioral",
        "score": 95,
        "strengths": ["Clear structure", "Quantified achievements", "Shows enthusiasm"],
    },
    {
        "question": "Explain OOP concepts",
        "answer": (
            "Object-Oriented Programming has four main principles. Encapsulation bundles data "
            "and methods together, like a class in Python. Inheritance allows classes to inherit "
            "properties from parent classes. Polymorphism lets objects of different types be "
            "treated through the same interface. Abstraction hides complex implementation details. "
            "For example, when you use a car, you don't need to know how the engine works."
        ),
        "type": "good_example",
        "category": "technical",
        "score": 92,
        "strengths": ["Complete coverage", "Real-world example", "Clear explanation"],
    },
]

BAD_EXAMPLES = [
    {
        "question": "Tell me about yourself",
        "answer": "I'm a developer. I code stuff. I like programming.",
        "type": "bad_example",
        "category": "hr_behavioral",
        "score": 35,
        "weaknesses": ["Too vague", "No specific details", "Lacks structure"],
    },
    {
        "question": "Explain OOP concepts",
        "answer": "OOP is about classes and objects. It's useful for programming.",
        "type": "bad_example",
        "category": "technical",
        "score": 40,
        "weaknesses": ["Incomplete", "No examples", "Too brief"],
    },
]

# Combine all seed data
SEED_DATA = CANONICAL_ANSWERS + GOOD_EXAMPLES + BAD_EXAMPLES


@dataclass
class VectorSearchResult:
    """Container for vector search results with enhanced metadata."""

    score: float
    question: str
    answer: str
    document_id: str
    example_type: str  # canonical, good_example, bad_example, transcript
    category: Optional[str] = None
    metadata: Optional[Dict] = None


class InterviewVectorStore:
    """
    Enhanced Vector Store for Interview Knowledge Base
    
    Features:
    - Stores canonical answers, good/bad examples, and transcripts
    - Enables retrieval of similar examples for feedback
    - Uses ChromaDB for persistence and Sentence-Transformers for embeddings
    """

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

        # Initialize embedding model (Sentence-Transformers)
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize ChromaDB client
        self._client = PersistentClient(path=str(self.db_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._ensure_seed_documents()

    # ------------------------------------------------------------------
    # Public API - Search and Retrieval
    # ------------------------------------------------------------------
    
    def search(
        self, 
        text: str, 
        top_k: int = 3,
        example_type: Optional[Literal["canonical", "good_example", "bad_example", "transcript"]] = None,
        category: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar examples in the knowledge base.
        
        Args:
            text: Query text to search for
            top_k: Number of results to return
            example_type: Filter by example type (canonical, good_example, bad_example, transcript)
            category: Filter by category (hr_behavioral, technical)
        
        Returns:
            List of VectorSearchResult objects
        """
        if not text.strip():
            return []

        query_embedding = self._embed(text)
        
        # Build filter if needed
        where_filter = {}
        if example_type:
            where_filter["type"] = example_type
        if category:
            where_filter["category"] = category
        
        # Query ChromaDB
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where_filter:
            query_kwargs["where"] = where_filter
        
        raw = self._collection.query(**query_kwargs)

        results: List[VectorSearchResult] = []
        documents = raw.get("documents", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        ids = raw.get("ids", [[]])[0]

        for doc, score, meta, doc_id in zip(documents, distances, metadatas, ids):
            # Convert distance to similarity score (cosine distance -> similarity)
            similarity_score = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)
            
            results.append(
                VectorSearchResult(
                    score=similarity_score,
                    question=meta.get("question", "Unknown question"),
                    answer=doc,
                    document_id=doc_id,
                    example_type=meta.get("type", "unknown"),
                    category=meta.get("category"),
                    metadata={
                        k: v for k, v in meta.items() 
                        if k not in ["question", "type", "category"]
                    },
                )
            )

        return results

    def get_canonical_answers(
        self, 
        question: str, 
        top_k: int = 1
    ) -> List[VectorSearchResult]:
        """Retrieve canonical answers for a given question."""
        return self.search(question, top_k=top_k, example_type="canonical")

    def get_good_examples(
        self, 
        text: str, 
        top_k: int = 2
    ) -> List[VectorSearchResult]:
        """Retrieve good example answers for reference."""
        return self.search(text, top_k=top_k, example_type="good_example")

    def get_bad_examples(
        self, 
        text: str, 
        top_k: int = 1
    ) -> List[VectorSearchResult]:
        """Retrieve bad example answers to show what to avoid."""
        return self.search(text, top_k=top_k, example_type="bad_example")

    def get_examples_for_feedback(
        self, 
        candidate_answer: str,
        question: str,
        top_k: int = 3
    ) -> Dict[str, List[VectorSearchResult]]:
        """
        Retrieve examples for feedback generation.
        Returns canonical, good, and bad examples together.
        """
        query_text = f"{question} {candidate_answer}"
        
        return {
            "canonical": self.get_canonical_answers(question, top_k=1),
            "good_examples": self.get_good_examples(query_text, top_k=min(2, top_k)),
            "bad_examples": self.get_bad_examples(query_text, top_k=min(1, top_k)),
        }

    # ------------------------------------------------------------------
    # Public API - Storage
    # ------------------------------------------------------------------
    
    def add_transcript(
        self,
        question: str,
        answer: str,
        category: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Store a transcript from an interview session.
        
        Args:
            question: Interview question
            answer: Candidate's answer
            category: Question category (hr_behavioral, technical)
            session_id: Optional session identifier
            metadata: Additional metadata (scores, feedback, etc.)
        
        Returns:
            Document ID of the stored transcript
        """
        doc_id = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._collection.get()['ids'])}"
        if session_id:
            doc_id = f"transcript_{session_id}_{len(self._collection.get()['ids'])}"
        
        embedding = self._embed(answer)
        
        meta = {
            "question": question,
            "type": "transcript",
            "category": category,
            "timestamp": datetime.now().isoformat(),
        }
        if session_id:
            meta["session_id"] = session_id
        if metadata:
            meta.update(metadata)
        
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[answer],
            metadatas=[meta],
        )
        
        return doc_id

    def add_example(
        self,
        question: str,
        answer: str,
        example_type: Literal["canonical", "good_example", "bad_example"],
        category: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Add a new example to the knowledge base.
        
        Args:
            question: Interview question
            answer: Example answer
            example_type: Type of example (canonical, good_example, bad_example)
            category: Question category
            metadata: Additional metadata
        
        Returns:
            Document ID
        """
        doc_id = f"{example_type}_{len(self._collection.get()['ids'])}"
        
        embedding = self._embed(answer)
        
        meta = {
            "question": question,
            "type": example_type,
            "category": category,
        }
        if metadata:
            meta.update(metadata)
        
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[answer],
            metadatas=[meta],
        )
        
        return doc_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    
    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text using Sentence-Transformers."""
        return self._model.encode(text).tolist()

    def _ensure_seed_documents(self) -> None:
        """Populate the collection with seed documents if missing."""
        try:
            existing_count = self._collection.count()
        except AttributeError:
            existing_count = len(self._collection.get()["ids"])

        if existing_count >= len(self.seed_data):
            return

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for idx, item in enumerate(self.seed_data):
            doc_id = f"{item.get('type', 'doc')}_{idx}"
            ids.append(doc_id)
            documents.append(item["answer"])
            embeddings.append(self._embed(item["answer"]))
            
            meta = {
                "question": item["question"],
                "type": item.get("type", "canonical"),
                "category": item.get("category", "general"),
            }
            # Add additional metadata if present
            for key in ["score", "strengths", "weaknesses"]:
                if key in item:
                    meta[key] = str(item[key])  # ChromaDB requires string values
            
            metadatas.append(meta)

        # Use upsert to avoid duplicates
        add_fn = getattr(self._collection, "upsert", None) or self._collection.add
        add_fn(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )


__all__ = ["InterviewVectorStore", "VectorSearchResult"]
