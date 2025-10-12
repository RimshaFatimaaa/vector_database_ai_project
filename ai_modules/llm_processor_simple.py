"""
Simplified LLM Processor Module - Step 2
Core features: Question generation, answer evaluation, and structured feedback
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# LLM imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """Types of interview questions"""
    HR_BEHAVIORAL = "hr_behavioral"
    TECHNICAL = "technical"

class DifficultyLevel(Enum):
    """Difficulty levels for questions"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class Question:
    """Structured question data"""
    question_text: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    expected_keywords: List[str] = None

@dataclass
class EvaluationResult:
    """Structured evaluation result"""
    overall_score: float  # 0-100
    relevance_score: float  # 0-100
    clarity_score: float  # 0-100
    correctness_score: float  # 0-100
    feedback: str
    suggestions: List[str]

class SimpleLLMProcessor:
    """Simplified LLM processor for core interview features"""
    
    def __init__(self, use_openai: bool = True, openai_api_key: Optional[str] = None):
        """
        Initialize simplified LLM processor
        
        Args:
            use_openai: Whether to use OpenAI API (True) or local models (False)
            openai_api_key: OpenAI API key (if using OpenAI)
        """
        self.use_openai = use_openai
        self.openai_client = None
        self.local_llm = None
        
        # Initialize based on preference
        if use_openai:
            self._init_openai(openai_api_key)
        else:
            self._init_local_models()
        
        # Canonical answers database (manual for now)
        self.canonical_answers = {
            "teamwork": "Teamwork involves collaborating effectively with others to achieve common goals. It requires communication, active listening, conflict resolution, and supporting team members. A good team player contributes ideas, helps others when needed, and maintains a positive attitude.",
            "leadership": "Leadership is the ability to guide and inspire others toward achieving shared objectives. It involves setting clear goals, making decisions, motivating team members, providing feedback, and leading by example. Effective leaders communicate vision, delegate tasks appropriately, and support team development.",
            "problem_solving": "Problem-solving is the process of identifying, analyzing, and resolving issues systematically. It involves defining the problem clearly, gathering relevant information, generating multiple solutions, evaluating options, implementing the best solution, and monitoring results for continuous improvement.",
            "technical_skills": "Technical skills refer to the specific knowledge and abilities required to perform job-related tasks. This includes programming languages, software tools, methodologies, and domain expertise. Continuous learning and staying updated with industry trends are essential for maintaining technical competency.",
            "communication": "Communication is the ability to convey information clearly and effectively. It includes verbal, written, and non-verbal communication skills. Good communicators listen actively, ask clarifying questions, adapt their message to the audience, and provide constructive feedback.",
            "adaptability": "Adaptability is the ability to adjust to new conditions, environments, or challenges. It involves being flexible, open to change, learning new skills quickly, and maintaining performance under pressure. Adaptable individuals embrace uncertainty and view change as an opportunity for growth."
        }
        
        # Interview session state
        self.current_session = {
            "questions_asked": [],
            "current_difficulty": DifficultyLevel.MEDIUM,
            "session_score": 0.0,
            "total_questions": 0
        }

    def _init_openai(self, api_key: Optional[str] = None):
        """Initialize OpenAI client"""
        try:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found. Falling back to local models.")
                self.use_openai = False
                self._init_local_models()
                return
            
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self.use_openai = False
            self._init_local_models()

    def _init_local_models(self):
        """Initialize local Hugging Face models"""
        try:
            # Use a smaller, faster model for local inference
            model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create pipeline
            self.local_llm = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Local models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize local models: {e}")
            raise

    def generate_question(
        self, 
        question_type: QuestionType, 
        role: str = "Software Engineer",
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    ) -> Question:
        """
        Generate a dynamic interview question using DialoGPT-Medium
        
        Args:
            question_type: Type of question to generate
            role: Job role/position
            difficulty: Difficulty level
            
        Returns:
            Generated Question object
        """
        try:
            # Use DialoGPT-Medium for question generation
            if self.local_llm:
                question_text = self._generate_question_with_dialogpt(question_type, role, difficulty)
            else:
                # Fallback to template-based generation
                question_text = self._generate_question_template(question_type, role, difficulty)
            
            # Extract expected keywords
            expected_keywords = self._extract_keywords_from_question(question_text, question_type)
            
            # Create question object
            question = Question(
                question_text=question_text,
                question_type=question_type,
                difficulty=difficulty,
                expected_keywords=expected_keywords
            )
            
            # Update session
            self.current_session["questions_asked"].append(question_text)
            self.current_session["total_questions"] += 1
            
            return question
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            # Return fallback question
            return Question(
                question_text="Tell me about yourself and your relevant experience.",
                question_type=question_type,
                difficulty=difficulty,
                expected_keywords=["experience", "skills", "background"]
            )

    def _generate_question_with_dialogpt(self, question_type: QuestionType, role: str, difficulty: DifficultyLevel) -> str:
        """Generate question using DialoGPT-Medium"""
        import random
        
        # Base topics for each question type
        if question_type == QuestionType.HR_BEHAVIORAL:
            topics = ["teamwork", "leadership", "problem solving", "communication", "adaptability", "conflict resolution", "time management"]
            topic = random.choice(topics)
            base_prompt = f"Interviewer: Create a {difficulty.value} level behavioral interview question about {topic} for a {role} position. The question should ask for a specific example or experience."
        else:  # TECHNICAL
            topics = ["programming", "problem solving", "technical challenges", "learning", "projects", "algorithms", "system design"]
            topic = random.choice(topics)
            base_prompt = f"Interviewer: Create a {difficulty.value} level technical interview question about {topic} for a {role} position. The question should ask for specific technical details."
        
        base_prompt += f"\nInterviewer: Question:"
        
        # Generate using DialoGPT-Medium
        result = self.local_llm(
            base_prompt,
            max_length=len(base_prompt.split()) + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            truncation=True
        )
        
        generated_text = result[0]['generated_text']
        question = generated_text[len(base_prompt):].strip()
        
        # Clean up the question
        if "Interviewer:" in question:
            question = question.split("Interviewer:")[0].strip()
        if "Candidate:" in question:
            question = question.split("Candidate:")[0].strip()
        
        # If the question is too short or unclear, fall back to template
        if len(question) < 10 or question == "?" or not question.strip():
            return self._generate_question_template(question_type, role, difficulty)
        
        # Ensure it's a proper question
        if not question.endswith('?'):
            question += '?'
        
        return question

    def _generate_question_template(self, question_type: QuestionType, role: str, difficulty: DifficultyLevel) -> str:
        """Fallback template-based question generation"""
        import random
        
        if question_type == QuestionType.HR_BEHAVIORAL:
            topics = ["teamwork", "leadership", "problem solving", "communication", "adaptability"]
            topic = random.choice(topics)
            question_text = f"Tell me about a time when you demonstrated {topic} in your previous role."
        else:  # TECHNICAL
            topics = ["programming", "problem solving", "technical challenges", "learning", "projects"]
            topic = random.choice(topics)
            question_text = f"Describe a {topic} project you worked on and the technical challenges you faced."
        
        # Adjust difficulty
        if difficulty == DifficultyLevel.EASY:
            question_text += " Please provide a brief overview."
        elif difficulty == DifficultyLevel.HARD:
            question_text += " Please provide specific details about your approach, technologies used, and lessons learned."
        
        return question_text

    def _extract_keywords_from_question(self, question: str, question_type: QuestionType) -> List[str]:
        """Extract expected keywords from the generated question"""
        question_lower = question.lower()
        
        # Base keywords for each question type
        base_keywords = {
            QuestionType.HR_BEHAVIORAL: ["experience", "situation", "result", "learned", "challenge", "team", "leadership"],
            QuestionType.TECHNICAL: ["project", "technical", "challenges", "solution", "technologies", "approach", "implementation"]
        }
        
        keywords = base_keywords.get(question_type, [])
        
        # Add topic-specific keywords from the question
        topic_keywords = {
            'teamwork': ['team', 'collaborate', 'collaboration', 'together', 'group'],
            'leadership': ['lead', 'leader', 'manage', 'management', 'direct', 'guide'],
            'problem_solving': ['problem', 'solve', 'solution', 'challenge', 'difficult'],
            'communication': ['communicate', 'communication', 'present', 'presentation', 'explain'],
            'adaptability': ['adapt', 'adaptability', 'change', 'flexible', 'flexibility'],
            'programming': ['programming', 'code', 'coding', 'development', 'software'],
            'technical': ['technical', 'technology', 'project', 'develop', 'development']
        }
        
        for topic, topic_words in topic_keywords.items():
            if any(word in question_lower for word in topic_words):
                keywords.extend(topic_words)
        
        return list(set(keywords))  # Remove duplicates

    def evaluate_answer(
        self, 
        question: str, 
        candidate_answer: str,
        cleaned_answer: str = None
    ) -> EvaluationResult:
        """
        Evaluate candidate's answer using DialoGPT-Medium enhanced analysis
        
        Args:
            question: The interview question
            candidate_answer: Original candidate's answer
            cleaned_answer: Cleaned answer from NLP module (optional)
            
        Returns:
            EvaluationResult with scores and feedback
        """
        try:
            # Use cleaned answer if provided, otherwise use original
            answer_to_evaluate = cleaned_answer if cleaned_answer else candidate_answer
            
            # Try DialoGPT-Medium evaluation first, fallback to rule-based
            if self.local_llm:
                try:
                    return self._evaluate_with_dialogpt(question, answer_to_evaluate)
                except Exception as e:
                    logger.warning(f"DialoGPT evaluation failed, falling back to rule-based: {e}")
            
            # Fallback to rule-based evaluation
            return self._evaluate_rule_based(question, answer_to_evaluate)
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return self._create_fallback_evaluation()

    def _evaluate_with_dialogpt(self, question: str, answer: str) -> EvaluationResult:
        """Evaluate answer using DialoGPT-Medium"""
        # Create a more direct evaluation prompt
        conversation_prompt = f"""Rate this interview answer from 0-100:

Question: {question}
Answer: {answer}

Rate on:
Relevance: How well does it answer the question?
Clarity: How clear and well-structured is it?
Correctness: How accurate and appropriate is it?

Give scores like this:
Relevance: 75/100
Clarity: 80/100
Correctness: 70/100
Overall: 75/100
Feedback: Good answer with specific examples
Suggestions: Add more technical details

Your rating:"""
        
        result = self.local_llm(
            conversation_prompt,
            max_length=len(conversation_prompt.split()) + 150,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.1,
            truncation=True
        )
        
        generated_text = result[0]['generated_text']
        evaluation = generated_text[len(conversation_prompt):].strip()
        
        # Debug: Log the generated evaluation
        logger.info(f"DialoGPT generated evaluation: {evaluation}")
        
        # Clean up the response
        if "Interviewer:" in evaluation:
            evaluation = evaluation.split("Interviewer:")[0].strip()
        if "Candidate:" in evaluation:
            evaluation = evaluation.split("Candidate:")[0].strip()
        
        # Parse the evaluation
        return self._parse_dialogpt_evaluation(evaluation, question, answer)

    def _parse_dialogpt_evaluation(self, evaluation_text: str, question: str, answer: str) -> EvaluationResult:
        """Parse DialoGPT evaluation response"""
        try:
            import re
            
            # Debug: Log the evaluation text being parsed
            logger.info(f"Parsing evaluation text: {evaluation_text}")
            
            # More comprehensive patterns for score extraction
            relevance_patterns = [
                r'Relevance:\s*(\d+)', r'relevance[:\s]*(\d+)', r'Relevance[:\s]*(\d+)',
                r'relevance\s*(\d+)', r'Relevance\s*(\d+)', r'Relevance\s*(\d+)/100'
            ]
            clarity_patterns = [
                r'Clarity:\s*(\d+)', r'clarity[:\s]*(\d+)', r'Clarity[:\s]*(\d+)',
                r'clarity\s*(\d+)', r'Clarity\s*(\d+)', r'Clarity\s*(\d+)/100'
            ]
            correctness_patterns = [
                r'Correctness:\s*(\d+)', r'correctness[:\s]*(\d+)', r'Correctness[:\s]*(\d+)',
                r'correctness\s*(\d+)', r'Correctness\s*(\d+)', r'Correctness\s*(\d+)/100'
            ]
            overall_patterns = [
                r'Overall:\s*(\d+)', r'overall[:\s]*(\d+)', r'Overall[:\s]*(\d+)',
                r'overall\s*(\d+)', r'Overall\s*(\d+)', r'Overall\s*(\d+)/100'
            ]
            
            # Initialize with rule-based scores as fallback
            relevance_score = self._calculate_relevance_score(question.lower(), answer.lower())
            clarity_score = self._calculate_clarity_score(answer)
            correctness_score = self._calculate_correctness_score(answer.lower(), "")
            overall_score = (relevance_score + clarity_score + correctness_score) / 3
            
            # Try to extract relevance score
            for pattern in relevance_patterns:
                match = re.search(pattern, evaluation_text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if 0 <= score <= 100:
                        relevance_score = score
                        break
            
            # Try to extract clarity score
            for pattern in clarity_patterns:
                match = re.search(pattern, evaluation_text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if 0 <= score <= 100:
                        clarity_score = score
                        break
            
            # Try to extract correctness score
            for pattern in correctness_patterns:
                match = re.search(pattern, evaluation_text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if 0 <= score <= 100:
                        correctness_score = score
                        break
            
            # Try to extract overall score
            for pattern in overall_patterns:
                match = re.search(pattern, evaluation_text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if 0 <= score <= 100:
                        overall_score = score
                        break
            
            # If no overall score found, calculate it
            if overall_score == (relevance_score + clarity_score + correctness_score) / 3:
                overall_score = (relevance_score + clarity_score + correctness_score) / 3
            
            # Extract feedback and suggestions with more flexible patterns
            feedback_patterns = [
                r'Feedback:\s*([^\n]+)', r'feedback[:\s]*([^\n]+)', r'Feedback[:\s]*([^\n]+)',
                r'feedback\s*([^\n]+)', r'Feedback\s*([^\n]+)'
            ]
            suggestions_patterns = [
                r'Suggestions:\s*([^\n]+)', r'suggestions[:\s]*([^\n]+)', r'Suggestions[:\s]*([^\n]+)',
                r'suggestions\s*([^\n]+)', r'Suggestions\s*([^\n]+)'
            ]
            
            feedback = self._generate_feedback(relevance_score, clarity_score, correctness_score, "general")
            suggestions_text = "Continue practicing and providing specific examples."
            
            # Try to extract feedback
            for pattern in feedback_patterns:
                match = re.search(pattern, evaluation_text, re.IGNORECASE)
                if match:
                    feedback = match.group(1).strip()
                    break
            
            # Try to extract suggestions
            for pattern in suggestions_patterns:
                match = re.search(pattern, evaluation_text, re.IGNORECASE)
                if match:
                    suggestions_text = match.group(1).strip()
                    break
            
            # Convert suggestions to list
            suggestions = [s.strip() for s in suggestions_text.split(',') if s.strip()]
            if not suggestions:
                suggestions = self._generate_suggestions(relevance_score, clarity_score, correctness_score)
            
            # Update session score
            if self.current_session["total_questions"] > 0:
                self.current_session["session_score"] = (
                    (self.current_session["session_score"] * (self.current_session["total_questions"] - 1) + overall_score) 
                    / self.current_session["total_questions"]
                )
            else:
                self.current_session["session_score"] = overall_score
            
            logger.info(f"Parsed scores - Relevance: {relevance_score}, Clarity: {clarity_score}, Correctness: {correctness_score}, Overall: {overall_score}")
            
            return EvaluationResult(
                overall_score=overall_score,
                relevance_score=relevance_score,
                clarity_score=clarity_score,
                correctness_score=correctness_score,
                feedback=feedback,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error parsing DialoGPT evaluation: {e}")
            # Fallback to rule-based evaluation
            return self._evaluate_rule_based(question, answer)

    def _evaluate_rule_based(self, question: str, answer: str) -> EvaluationResult:
        """Fallback rule-based evaluation"""
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Extract topic from question
        topic = None
        question_words = question_lower.split()
        
        topic_mapping = {
            'teamwork': ['teamwork', 'team', 'collaborate', 'collaboration'],
            'leadership': ['leadership', 'lead', 'leader', 'manage', 'management'],
            'problem_solving': ['problem', 'solve', 'challenge', 'difficult'],
            'technical_skills': ['technical', 'programming', 'code', 'development', 'project'],
            'communication': ['communication', 'communicate', 'present', 'presentation'],
            'adaptability': ['adapt', 'adaptability', 'change', 'flexible', 'flexibility']
        }
        
        for topic_key, keywords in topic_mapping.items():
            if any(keyword in question_words for keyword in keywords):
                topic = topic_key
                break
        
        # Get canonical answer for comparison
        canonical_answer = self.canonical_answers.get(topic, "")
        
        # Calculate scores
        relevance_score = self._calculate_relevance_score(question_lower, answer_lower)
        clarity_score = self._calculate_clarity_score(answer)
        correctness_score = self._calculate_correctness_score(answer_lower, canonical_answer)
        
        # Calculate overall score
        overall_score = (relevance_score + clarity_score + correctness_score) / 3
        
        # Generate feedback
        feedback = self._generate_feedback(relevance_score, clarity_score, correctness_score, topic)
        suggestions = self._generate_suggestions(relevance_score, clarity_score, correctness_score)
        
        # Update session score
        if self.current_session["total_questions"] > 0:
            self.current_session["session_score"] = (
                (self.current_session["session_score"] * (self.current_session["total_questions"] - 1) + overall_score) 
                / self.current_session["total_questions"]
            )
        else:
            self.current_session["session_score"] = overall_score
        
        return EvaluationResult(
            overall_score=overall_score,
            relevance_score=relevance_score,
            clarity_score=clarity_score,
            correctness_score=correctness_score,
            feedback=feedback,
            suggestions=suggestions
        )

    def _calculate_relevance_score(self, question: str, answer: str) -> float:
        """Calculate relevance score based on content analysis"""
        # Extract key terms from question
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Remove common stop words for better analysis
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'tell', 'me', 'about', 'time', 'when', 'you', 'demonstrated', 'your', 'previous', 'role'}
        question_terms = set(question_lower.split()) - stop_words
        answer_terms = set(answer_lower.split()) - stop_words
        
        # Check if answer addresses the question
        common_terms = question_terms.intersection(answer_terms)
        
        # More sophisticated relevance scoring
        answer_length = len(answer.strip())
        
        # Very short answers get very low scores
        if answer_length < 5:
            return 5.0  # Almost no score for very short answers
        elif answer_length < 10:
            return 10.0
        elif answer_length < 20:
            return 20.0
        
        # Check for semantic relevance using topic keywords
        topic_keywords = {
            'teamwork': ['team', 'collaborate', 'collaboration', 'together', 'group', 'colleagues', 'peers', 'co-workers', 'partnership', 'cooperation', 'united', 'joint', 'collective', 'shared', 'mutual', 'support', 'help', 'assist', 'work with', 'team members', 'team project', 'cross-functional', 'teamwork', 'team player'],
            'leadership': ['lead', 'leader', 'manage', 'management', 'direct', 'guide', 'mentor', 'supervise', 'oversee', 'coordinate', 'facilitate', 'initiate', 'drive', 'champion', 'head', 'captain', 'boss', 'supervisor', 'manager', 'director', 'led', 'leading', 'leadership'],
            'problem_solving': ['problem', 'solve', 'solution', 'challenge', 'difficult', 'issue', 'troubleshoot', 'resolve', 'fix', 'address', 'overcome', 'tackle', 'approach', 'strategy', 'method', 'technique', 'process', 'solved', 'solving'],
            'technical': ['technical', 'technology', 'project', 'develop', 'development', 'programming', 'code', 'coding', 'software', 'application', 'system', 'architecture', 'database', 'api', 'framework', 'language', 'programming', 'engineer', 'engineering', 'built', 'created', 'implemented', 'designed', 'developed'],
            'communication': ['communicate', 'communication', 'present', 'presentation', 'explain', 'discuss', 'talk', 'speak', 'write', 'email', 'meeting', 'conversation', 'dialogue', 'interact', 'express', 'convey', 'share', 'inform'],
            'adaptability': ['adapt', 'adaptability', 'change', 'flexible', 'flexibility', 'adjust', 'modify', 'evolve', 'transform', 'shift', 'transition', 'accommodate', 'versatile', 'dynamic', 'responsive']
        }
        
        # Find the main topic from the question - improved detection
        main_topic = None
        topic_scores = {}
        
        for topic, keywords in topic_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in question_lower:
                    # Give more weight to longer, more specific keywords
                    score += len(keyword.split()) * 2
                    score += 1
            topic_scores[topic] = score
        
        # Find the topic with the highest score
        if topic_scores:
            main_topic = max(topic_scores, key=topic_scores.get)
            # Only use topic if it has a reasonable score
            if topic_scores[main_topic] < 2:
                main_topic = None
        
        # Calculate relevance based on topic-specific keywords in answer
        if main_topic and main_topic in topic_keywords:
            # First check for vague indicators - these override keyword count
            vague_indicators = ['i guess', 'i think', 'maybe', 'not sure', 'dont know', 'stuff', 'things', 'okay', 'fine', 'like', 'um', 'uh']
            vague_count = sum(1 for vague in vague_indicators if vague in answer_lower)
            
            if vague_count >= 3:  # Multiple vague indicators - very low score regardless of keywords
                return 20.0
            elif vague_count >= 1:  # Some vague indicators - low score regardless of keywords
                return 35.0
            
            # If not vague, then check keyword count
            topic_keywords_in_answer = sum(1 for keyword in topic_keywords[main_topic] if keyword in answer_lower)
            
            # High relevance if answer contains many topic-specific keywords
            if topic_keywords_in_answer >= 8:
                return 95.0
            elif topic_keywords_in_answer >= 5:
                return 85.0
            elif topic_keywords_in_answer >= 3:
                return 75.0
            elif topic_keywords_in_answer >= 1:
                return 65.0
        
        # Fallback to keyword overlap scoring
        if len(common_terms) > 8:  # Very high keyword overlap
            return 85.0
        elif len(common_terms) > 5:  # High keyword overlap
            return 75.0
        elif len(common_terms) > 3:  # Medium-high keyword overlap
            return 65.0
        elif len(common_terms) > 1:  # Medium keyword overlap
            return 55.0
        elif len(common_terms) > 0:  # Some keyword overlap
            return 45.0
        else:  # No keyword overlap
            return 25.0

    def _calculate_clarity_score(self, answer: str) -> float:
        """Calculate clarity score based on answer structure and quality"""
        # More sophisticated clarity metrics
        word_count = len(answer.split())
        char_count = len(answer.strip())
        
        # Check for sentence structure
        sentences = answer.split('.')
        avg_sentence_length = word_count / max(len(sentences), 1)
        
        # Check for filler words and repetition
        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally']
        filler_count = sum(answer.lower().count(filler) for filler in filler_words)
        
        # Check for vague language that should be penalized
        vague_words = ['i guess', 'i think', 'maybe', 'not sure', 'dont know', 'stuff', 'things', 'okay', 'fine']
        vague_count = sum(answer.lower().count(vague) for vague in vague_words)
        
        # Check for repetition (simple check)
        words = answer.lower().split()
        unique_words = len(set(words))
        repetition_ratio = unique_words / max(len(words), 1)
        
        # Calculate clarity score
        base_score = 0
        
        # Length scoring - more generous for detailed answers
        if word_count > 100:
            base_score += 60
        elif word_count > 50:
            base_score += 55
        elif word_count > 30:
            base_score += 50
        elif word_count > 15:
            base_score += 40
        elif word_count > 5:
            base_score += 25
        elif word_count > 1:
            base_score += 10  # Low for short answers
        else:
            base_score += 0  # Almost no score for single words
        
        # Sentence structure scoring - more generous
        if avg_sentence_length > 15:
            base_score += 25
        elif avg_sentence_length > 10:
            base_score += 20
        elif avg_sentence_length > 5:
            base_score += 15
        else:
            base_score += 10
        
        # Penalty for filler words
        filler_penalty = min(filler_count * 5, 20)
        base_score -= filler_penalty
        
        # Penalty for vague language
        vague_penalty = min(vague_count * 8, 30)
        base_score -= vague_penalty
        
        # Bonus for good vocabulary diversity
        if repetition_ratio > 0.7:
            base_score += 15
        elif repetition_ratio > 0.5:
            base_score += 10
        else:
            base_score += 5
        
        # Ensure score is within bounds
        return max(0, min(100, base_score))

    def _calculate_correctness_score(self, answer: str, canonical: str) -> float:
        """Calculate correctness score based on content quality and accuracy"""
        if not canonical:
            # If no canonical answer, score based on general quality indicators
            return self._score_general_correctness(answer)
        
        # More sophisticated correctness scoring
        canonical_terms = set(canonical.lower().split())
        answer_terms = set(answer.lower().split())
        
        # Remove stop words for better comparison
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
        canonical_terms = canonical_terms - stop_words
        answer_terms = answer_terms - stop_words
        
        common_terms = canonical_terms.intersection(answer_terms)
        
        # Calculate overlap ratio
        if len(canonical_terms) == 0:
            return self._score_general_correctness(answer)
        
        overlap_ratio = len(common_terms) / len(canonical_terms)
        
        
        # Check for vague indicators first - these override overlap scoring
        vague_indicators = ['i guess', 'i think', 'maybe', 'not sure', 'dont know', 'stuff', 'things', 'okay', 'fine', 'like', 'um', 'uh']
        vague_count = sum(1 for vague in vague_indicators if vague in answer.lower())
        
        if vague_count >= 3:  # Multiple vague indicators
            return 10.0  # Very low for very vague answers
        elif vague_count >= 1:  # Some vague indicators
            return 20.0  # Low for vague answers
        
        # If not vague, then use overlap-based scoring - more generous for good answers
        if overlap_ratio > 0.6:
            return 95.0
        elif overlap_ratio > 0.4:
            return 90.0
        elif overlap_ratio > 0.2:
            return 80.0
        elif overlap_ratio > 0.1:
            return 70.0
        else:
            if len(answer.split()) <= 2:
                return 5.0  # Very low for very short answers
            else:
                return 50.0
    
    def _score_general_correctness(self, answer: str) -> float:
        """Score correctness based on general quality indicators when no canonical answer exists"""
        word_count = len(answer.split())
        
        # Check for specific quality indicators
        quality_indicators = ['experience', 'project', 'team', 'challenge', 'solution', 'learned', 'result', 'outcome', 'success', 'improved', 'developed', 'implemented', 'managed', 'led', 'collaborated']
        indicator_count = sum(1 for indicator in quality_indicators if indicator in answer.lower())
        
        # Check for vague or weak language
        weak_indicators = ['i think', 'maybe', 'probably', 'sort of', 'kind of', 'i guess', 'not sure', 'dont know']
        weak_count = sum(1 for weak in weak_indicators if weak in answer.lower())
        
        # Base score from length - more generous for detailed answers
        if word_count > 50:
            base_score = 80
        elif word_count > 20:
            base_score = 70
        elif word_count > 5:
            base_score = 50
        elif word_count > 2:
            base_score = 20  # Low for short answers
        elif word_count > 1:
            base_score = 10  # Very low for 2 words
        else:
            base_score = 0   # No score for single words
        
        # Add points for quality indicators
        base_score += min(indicator_count * 5, 25)
        
        # Subtract points for weak language
        base_score -= min(weak_count * 8, 20)
        
        return max(0, min(100, base_score))

    def _generate_feedback(self, relevance: float, clarity: float, correctness: float, topic: str) -> str:
        """Generate dynamic feedback based on scores"""
        feedback_parts = []
        
        # Relevance feedback
        if relevance >= 85:
            feedback_parts.append("Excellent! Your answer directly addresses the question with relevant details.")
        elif relevance >= 70:
            feedback_parts.append("Good job! Your answer mostly addresses the question asked.")
        elif relevance >= 50:
            feedback_parts.append("Your answer partially addresses the question but could be more focused.")
        elif relevance >= 30:
            feedback_parts.append("Your answer has limited relevance to the specific question asked.")
        else:
            feedback_parts.append("Your answer doesn't seem to address the question directly.")
        
        # Clarity feedback
        if clarity >= 85:
            feedback_parts.append("Your response is very clear, well-structured, and easy to follow.")
        elif clarity >= 70:
            feedback_parts.append("Your response is generally clear and well-organized.")
        elif clarity >= 50:
            feedback_parts.append("Your response could be clearer and more detailed.")
        elif clarity >= 30:
            feedback_parts.append("Your response needs improvement in clarity and structure.")
        else:
            feedback_parts.append("Your response is unclear and difficult to understand.")
        
        # Correctness feedback
        if correctness >= 85:
            feedback_parts.append("Your answer demonstrates strong understanding and accuracy.")
        elif correctness >= 70:
            feedback_parts.append("Your answer shows good understanding with minor areas for improvement.")
        elif correctness >= 50:
            feedback_parts.append("Your answer shows some understanding but could be more accurate.")
        elif correctness >= 30:
            feedback_parts.append("Your answer needs more accurate information and better examples.")
        else:
            feedback_parts.append("Your answer lacks accuracy and specific examples.")
        
        return " ".join(feedback_parts)

    def _generate_suggestions(self, relevance: float, clarity: float, correctness: float) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Relevance suggestions
        if relevance < 30:
            suggestions.append("Start by directly answering the question asked, then provide supporting details.")
        elif relevance < 50:
            suggestions.append("Focus more on the specific aspects mentioned in the question.")
        elif relevance < 70:
            suggestions.append("Try to connect your examples more directly to the question topic.")
        
        # Clarity suggestions
        if clarity < 30:
            suggestions.append("Structure your answer with clear points and use complete sentences.")
        elif clarity < 50:
            suggestions.append("Provide more specific details and examples to support your points.")
        elif clarity < 70:
            suggestions.append("Consider organizing your thoughts better and reducing filler words.")
        
        # Correctness suggestions
        if correctness < 30:
            suggestions.append("Include specific examples from your experience and use concrete details.")
        elif correctness < 50:
            suggestions.append("Provide more accurate information and avoid vague statements.")
        elif correctness < 70:
            suggestions.append("Add more specific examples and measurable outcomes to strengthen your answer.")
        
        # General suggestions if all scores are good
        if not suggestions:
            suggestions.append("Great job! Continue providing detailed, relevant examples in your responses.")
            suggestions.append("Consider adding quantifiable results or specific outcomes to make your answers even stronger.")
        
        return suggestions

    def _create_fallback_evaluation(self) -> EvaluationResult:
        """Create fallback evaluation when processing fails"""
        return EvaluationResult(
            overall_score=50.0,
            relevance_score=50.0,
            clarity_score=50.0,
            correctness_score=50.0,
            feedback="Unable to provide detailed evaluation. Please try again.",
            suggestions=["Ensure your answer is clear and relevant to the question."]
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get current interview session summary"""
        return {
            "total_questions": self.current_session["total_questions"],
            "average_score": self.current_session["session_score"],
            "current_difficulty": self.current_session["current_difficulty"].value,
            "questions_asked": self.current_session["questions_asked"]
        }

    def adjust_difficulty(self, performance_score: float):
        """Adjust difficulty based on performance"""
        if performance_score >= 80:
            self.current_session["current_difficulty"] = DifficultyLevel.HARD
        elif performance_score >= 60:
            self.current_session["current_difficulty"] = DifficultyLevel.MEDIUM
        else:
            self.current_session["current_difficulty"] = DifficultyLevel.EASY

# Convenience functions
def generate_question(
    question_type: str = "hr_behavioral",
    role: str = "Software Engineer",
    difficulty: str = "medium"
) -> Question:
    """Convenience function to generate a single question"""
    processor = SimpleLLMProcessor(use_openai=False)
    return processor.generate_question(
        QuestionType(question_type),
        role,
        DifficultyLevel(difficulty)
    )

def evaluate_answer(question: str, answer: str, cleaned_answer: str = None) -> EvaluationResult:
    """Convenience function to evaluate an answer"""
    processor = SimpleLLMProcessor(use_openai=False)
    return processor.evaluate_answer(question, answer, cleaned_answer)
