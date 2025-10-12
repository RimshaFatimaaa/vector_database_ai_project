#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
"""
step2_llms.py
Single-file implementation for Step 2 (LLMs / Transformers) of:
AI-Powered Job Interview Coach — PRD

Features:
- Dynamic question generation (HR & Technical) via OpenAI or local FLAN-T5.
- Answer evaluation:
    - Embedding-based similarity to canonical answers (sentence-transformers).
    - LLM-driven rubric scoring & textual feedback (OpenAI or FLAN-T5).
- Minimal FastAPI endpoints for integration.

Usage:
  - Optional: set OPENAI_API_KEY to use OpenAI Chat completions for generation/evaluation.
  - Otherwise the code falls back to local HF models (may download model files on first run).
  - Run server:
      uvicorn step2_llms:app --reload
"""

from typing import List, Dict, Optional
import os
import json
import re
import math
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
import logging

# OPTIONAL: OpenAI
try:
    import openai
except Exception:
    openai = None

# sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# transformers pipeline for local generation (FLAN-T5)
try:
    from transformers import pipeline
except Exception:
    pipeline = None

# ---------- Configuration ----------
def get_openai_key():
    """Get OpenAI key from environment or Streamlit secrets"""
    # Try environment variable first
    key = os.getenv("OPENAI_API_KEY", None)
    if key:
        return key
    
    # Try Streamlit secrets
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except:
        pass
    
    return None

# Get the key dynamically
OPENAI_KEY = get_openai_key()

if OPENAI_KEY and openai:
    openai.api_key = OPENAI_KEY

# Model choices (change if you prefer other models)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # sentence-transformers (light + fast). :contentReference[oaicite:2]{index=2}
LOCAL_TEXT2TEXT_MODEL = "google/flan-t5-base"   # used when OpenAI not configured. :contentReference[oaicite:3]{index=3}

# Caching model objects
_embedding_model = None
_generation_pipeline = None

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Utilities ----------

def safe_load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required. Install with `pip install sentence-transformers`.")
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (this may download files)...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

def safe_load_generation_pipeline():
    global _generation_pipeline
    if _generation_pipeline is None:
        if pipeline is None:
            raise RuntimeError("transformers is required. Install with `pip install transformers[torch]`.")
        logger.info(f"Loading generation pipeline model '{LOCAL_TEXT2TEXT_MODEL}' (this may download files)...")
        # Use text2text-generation pipeline for FLAN-T5
        _generation_pipeline = pipeline(task="text2text-generation", model=LOCAL_TEXT2TEXT_MODEL, device=0 if _has_cuda() else -1)
    return _generation_pipeline

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # safe cosine similarity
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ---------- Sample canonical Q&A store ----------
# In production this would be in your Step 5 vector DB; here we keep a small in-memory sample.
CANONICAL_QA = [
    {"id": "hr_01", "round": "hr", "question": "Tell me about a time you worked in a team and faced conflict. How did you handle it?",
     "answer": "I used the STAR method: Situation - our team missed a deadline; Task - I coordinated communication; Action - scheduled a meeting, redistributed tasks, negotiated new timelines; Result - we delivered with acceptable quality and learned to align better."},
    {"id": "tech_01", "round": "technical", "question": "Explain the concept of object-oriented programming and its main principles.",
     "answer": "OOP organizes code into objects that combine data and behavior. Main principles: Encapsulation (bundling data + methods), Abstraction (exposing only necessary details), Inheritance (reuse and extend classes), Polymorphism (same interface, different implementations)."},
    {"id": "tech_02", "round": "technical", "question": "What are Python decorators and when would you use them?",
     "answer": "Decorators are higher-order functions that modify other functions or methods. Use them for logging, access control, caching, or adding behavior without changing the original function."},
    # Add more canonical items as needed...
]

# Precompute embeddings for canonical answers (lazy)
_canonical_embeddings = None
_canonical_loaded = False

def ensure_canonical_embeddings():
    global _canonical_embeddings, _canonical_loaded
    if _canonical_loaded:
        return
    model = safe_load_embedding_model()
    texts = [normalize_text(item["answer"]) for item in CANONICAL_QA]
    logger.info("Computing embeddings for canonical answers...")
    _canonical_embeddings = model.encode(texts, convert_to_numpy=True)
    _canonical_loaded = True

# ---------- Question Generation ----------

def _generate_question_openai(round_type: str, context: Optional[str] = None) -> str:
    """
    Generate a question using OpenAI Chat API (if available).
    """
    assert openai is not None, "OpenAI library not available."
    
    # Get the key from environment variable
    current_key = os.getenv("OPENAI_API_KEY")
    if not current_key:
        # Try Streamlit secrets as fallback
        try:
            import streamlit as st
            current_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass
    
    if not current_key:
        raise ValueError("OpenAI API key not available")
    
    # Create specific prompts for different question types
    if round_type.lower() in ["hr_behavioral", "hr"]:
        system_prompt = "You are an expert interviewer. Generate behavioral interview questions that ask about past experiences, challenges, teamwork, or leadership."
        user_prompt = "Generate one behavioral interview question. Use phrases like 'Tell me about a time when...', 'Describe a situation where...', or 'Give me an example of...'. Return only the question text."
    elif round_type.lower() == "technical":
        system_prompt = "You are an expert technical interviewer. Generate technical interview questions about programming, problem-solving, algorithms, or software engineering."
        user_prompt = "Generate one technical interview question about programming, algorithms, system design, or software engineering concepts. Return only the question text."
    else:
        system_prompt = "You are an expert interviewer."
        user_prompt = f"Generate one {round_type} interview question. Return only the question text."
    
    if context:
        user_prompt += f" Context: {context}"

    logger.info("Calling OpenAI to generate question...")
    
    try:
        # Use the new OpenAI client API (v1.0+)
        client = openai.OpenAI(api_key=current_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=150,
        )
        
        # Extract text
        text = resp.choices[0].message.content.strip()
        logger.info(f"OpenAI generated: {text}")
        return text
        
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise e

def _generate_question_local(round_type: str, context: Optional[str] = None) -> str:
    """
    Generate a question using local FLAN-T5 text2text pipeline.
    """
    pipe = safe_load_generation_pipeline()
    
    # Create specific prompts for different question types
    if round_type.lower() == "hr_behavioral" or round_type.lower() == "hr":
        prompt = "Generate a behavioral interview question about past experiences, challenges, teamwork, or leadership."
    elif round_type.lower() == "technical":
        prompt = "Generate a technical interview question about programming, problem-solving, or software engineering."
    else:
        prompt = f"Generate a {round_type} interview question."
    
    if context:
        prompt += f" Context: {context}"
    prompt += " Keep it concise. Output only the question."
    
    logger.info("Generating question with local model...")
    out = pipe(prompt, max_length=128, do_sample=False)
    text = out[0]["generated_text"].strip()
    # FLAN-T5 may echo the prompt; attempt to clean to single sentence
    return text

def generate_question(round_type: str = "hr", context: Optional[str] = None) -> Dict:
    """
    Public API for generating a single interview question.
    round_type: "hr" or "technical"
    """
    round_type = round_type.lower()
    
    # Force OpenAI usage - get key from environment
    current_openai_key = os.getenv("OPENAI_API_KEY")
    logger.info(f"Generating {round_type} question. OpenAI key available: {bool(current_openai_key)}")
    
    # If no key in environment, try to get from Streamlit secrets
    if not current_openai_key:
        try:
            import streamlit as st
            current_openai_key = st.secrets.get("OPENAI_API_KEY")
            logger.info(f"Got key from Streamlit secrets: {bool(current_openai_key)}")
        except Exception as e:
            logger.warning(f"Could not get key from Streamlit secrets: {e}")
    
    if current_openai_key and openai:
        try:
            logger.info("Using OpenAI for question generation")
            q = _generate_question_openai(round_type, context)
            logger.info(f"Successfully generated question: {q[:100]}...")
            return {"question": normalize_text(q), "source": "openai"}
        except Exception as e:
            logger.error(f"OpenAI generation failed with error: {str(e)}")
            logger.warning("Falling back to local model")
    else:
        logger.info("OpenAI not available, using local model")
    
    # fallback to local
    logger.info("Using local FLAN-T5 for question generation")
    q = _generate_question_local(round_type, context)
    return {"question": normalize_text(q), "source": "local_flan_t5"}

# ---------- Answer Evaluation ----------

def _llm_evaluate_openai(question: str, candidate_answer: str, top_canonical: Dict, similarity_score: float) -> Dict:
    """
    Use OpenAI to produce a structured evaluation. We ask for a JSON output with numeric scores.
    """
    system = "You are an objective interview evaluator. Given the candidate's answer and a canonical example answer, evaluate using a rubric and output strict JSON with these keys: relevance (0-1), completeness (0-1), clarity (0-1), feedback (string), suggestions (string). Also return matched_canonical_id and similarity (0-1). IMPORTANT: For clarity scoring, very short answers (1-3 words) should score 0.1-0.2, short answers (4-10 words) should score 0.3-0.5, medium answers (11-25 words) should score 0.6-0.7, and detailed answers (25+ words) should score 0.8-0.9. Single letters or very poor answers should get very low clarity scores."
    user_prompt = (
        f"Question: {question}\n\n"
        f"Candidate answer: {candidate_answer}\n\n"
        f"Canonical answer (best match): {top_canonical['answer']}\n\n"
        f"Similarity (embedding-based): {similarity_score:.4f}\n\n"
        "Evaluate using the rubric. Provide JSON only."
    )
    try:
        # Use the new OpenAI client API (v1.0+)
        client = openai.OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user_prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        content = resp.choices[0].message.content.strip()
        # Try to extract JSON from the model output
        json_text = _extract_json_from_text(content)
        result = json.loads(json_text)
        
        # Post-process clarity score to ensure it follows strict rules
        word_count = len(candidate_answer.split())
        if word_count == 0:
            result["clarity"] = 0.0
        elif word_count == 1:
            result["clarity"] = min(result.get("clarity", 0.5), 0.1)  # Cap at 0.1 for single word
        elif word_count <= 3:
            result["clarity"] = min(result.get("clarity", 0.5), 0.2)  # Cap at 0.2 for very short
        elif word_count <= 10:
            result["clarity"] = min(result.get("clarity", 0.5), 0.4)  # Cap at 0.4 for short
        
        return result
    except Exception as e:
        logger.exception("OpenAI evaluation failed: %s", e)
        # fallback to simple heuristic evaluator below
        return _heuristic_evaluation(candidate_answer, top_canonical, similarity_score)

def _llm_evaluate_local(question: str, candidate_answer: str, top_canonical: Dict, similarity_score: float) -> Dict:
    """
    Use local FLAN-T5 to produce a JSON-like evaluation. FLAN may not output strict JSON; attempt to parse.
    """
    pipe = safe_load_generation_pipeline()
    prompt = (
        "You are an objective interview evaluator. Given the question, candidate answer, and canonical answer, "
        "produce a JSON object with keys: relevance (0-1), completeness (0-1), clarity (0-1), feedback, suggestions, matched_canonical_id, similarity.\n\n"
        f"Question: {question}\n\nCandidate answer: {candidate_answer}\n\nCanonical answer: {top_canonical['answer']}\n\n"
        f"Similarity (embedding): {similarity_score:.4f}\n\n"
        "Return only valid JSON."
    )
    logger.info("Running local LLM evaluator (may be less strict than OpenAI)...")
    out = pipe(prompt, max_length=512, do_sample=False)
    text = out[0]["generated_text"].strip()
    json_text = _extract_json_from_text(text)
    if not json_text:
        return _heuristic_evaluation(candidate_answer, top_canonical, similarity_score)
    try:
        return json.loads(json_text)
    except Exception:
        return _heuristic_evaluation(candidate_answer, top_canonical, similarity_score)

def _heuristic_evaluation(candidate_answer: str, top_canonical: Dict, similarity_score: float) -> Dict:
    """
    If LLM-based evaluation isn't available, return heuristic scores:
      - relevance: based on cosine similarity
      - completeness: similarity adjusted for length
      - clarity: based on answer length, structure, and quality
    """
    cand = normalize_text(candidate_answer)
    word_count = len(cand.split())
    char_count = len(cand.strip())
    
    # heuristics
    relevance = float(np.clip(similarity_score, 0.0, 1.0))
    completeness = float(np.clip(similarity_score * min(1.0, math.log(1 + word_count) / 3.0), 0.0, 1.0))
    
    # Improved clarity measure based on length and structure
    if word_count == 0:
        clarity = 0.0  # No answer
    elif word_count == 1:
        clarity = 0.1  # Single word/letter - very poor clarity
    elif word_count <= 3:
        clarity = 0.2  # Very short - poor clarity
    elif word_count <= 10:
        clarity = 0.4  # Short - limited clarity
    elif word_count <= 25:
        clarity = 0.6  # Medium - adequate clarity
    elif word_count <= 50:
        clarity = 0.8  # Long - good clarity
    else:
        clarity = 0.9  # Very long - excellent clarity
    
    # Penalize for filler words
    filler_count = len(re.findall(r"\bu(m+|uh+|like|you know)\b", cand.lower()))
    clarity = float(np.clip(clarity - (filler_count * 0.1), 0.0, 1.0))
    
    # Penalize for very short answers
    if char_count < 10:
        clarity = min(clarity, 0.2)
    
    feedback = f"Similarity to canonical answer: {similarity_score:.2f}. Expand more on examples and outcomes."
    suggestions = "Use STAR structure for behavioral answers; add specific examples and metrics."
    return {
        "relevance": relevance,
        "completeness": completeness,
        "clarity": clarity,
        "feedback": feedback,
        "suggestions": suggestions,
        "matched_canonical_id": top_canonical.get("id"),
        "similarity": similarity_score
    }

def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Try to extract a JSON object from a string (first {...} occurrence).
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    # Attempt to fix common issues: trailing commas
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*\]", "]", candidate)
    return candidate

def evaluate_answer(question: str, candidate_answer: str, top_k: int = 1) -> Dict:
    """
    Main evaluation function:
     - computes embedding similarity to canonical answers
     - selects best matching canonical answer(s)
     - asks LLM (OpenAI or local) to produce structured evaluation JSON
    Returns a dict with:
       question, candidate_answer, matched_canonical, similarity, evaluation (scores+feedback)
    """
    candidate_answer = normalize_text(candidate_answer)
    ensure_canonical_embeddings()
    model = safe_load_embedding_model()
    cand_emb = model.encode(candidate_answer, convert_to_numpy=True)

    # Compute similarities against canonical embeddings
    sims = [cosine_sim(cand_emb, emb) for emb in _canonical_embeddings]
    # Top K
    idx_sorted = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    top_idx = idx_sorted[0]
    sim_score = float(sims[top_idx])
    top_canonical = CANONICAL_QA[top_idx]

    # Call LLM to evaluate using a prompt-based rubric
    if OPENAI_KEY and openai:
        try:
            eval_result = _llm_evaluate_openai(question, candidate_answer, top_canonical, sim_score)
        except Exception as e:
            logger.warning("OpenAI evaluation failed; falling back to local or heuristic. Error: %s", e)
            eval_result = _llm_evaluate_local(question, candidate_answer, top_canonical, sim_score)
    else:
        eval_result = _llm_evaluate_local(question, candidate_answer, top_canonical, sim_score)

    # Normalize numeric fields to floats in [0,1]
    for k in ("relevance", "completeness", "clarity", "similarity"):
        if k in eval_result:
            try:
                val = float(eval_result[k])
                eval_result[k] = max(0.0, min(1.0, val))
            except Exception:
                pass

    return {
        "question": question,
        "candidate_answer": candidate_answer,
        "matched_canonical": {"id": top_canonical["id"], "answer": top_canonical["answer"]},
        "similarity": sim_score,
        "evaluation": eval_result
    }

# ---------- FastAPI app for integration ----------
app = FastAPI(title="Step2 LLMs — QuestionGen & Evaluator")

class GenRequest(BaseModel):
    round_type: str = "hr"
    context: Optional[str] = None

class EvalRequest(BaseModel):
    question: str
    candidate_answer: str

@app.post("/generate_question")
def api_generate_question(req: GenRequest):
    return generate_question(req.round_type, req.context)

@app.post("/evaluate_answer")
def api_evaluate_answer(req: EvalRequest):
    return evaluate_answer(req.question, req.candidate_answer)

# ---------- CLI demo ----------
if __name__ == "__main__":
    # Quick demo usage when run as script
    print("Step 2 — LLMs demo (generate + evaluate).")
    # 1) Generate a technical question about Python decorators
    gen = generate_question(round_type="technical", context="Candidate knows Python and web frameworks")
    print("Generated question:", gen["question"], "(source:", gen["source"], ")")

    # 2) Example candidate answer (short)
    candidate = "Decorators are functions that wrap another function to extend its behavior without modifying it. They are useful for logging and caching."
    result = evaluate_answer(gen["question"], candidate)
    print(json.dumps(result, indent=2))


# In[ ]:





# In[ ]:




