"""
LLM Processor Module - Step 2
Simplified dynamic question generation and answer evaluation using LLMs
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

class LLMProcessor:
    """Main LLM processor for question generation and answer evaluation"""
    
    def __init__(self, use_openai: bool = True, openai_api_key: Optional[str] = None):
        """
        Initialize LLM processor
        
        Args:
            use_openai: Whether to use OpenAI API (True) or local models (False)
            openai_api_key: OpenAI API key (if using OpenAI)
        """
        self.use_openai = use_openai
        self.openai_client = None
        self.local_llm = None
        self.tokenizer = None
        
        # Initialize based on preference
        if use_openai:
            self._init_openai(openai_api_key)
        else:
            self._init_local_models()
        
        # Initialize prompt templates
        self._init_prompt_templates()
        
        # Initialize canonical answers database
        self._init_canonical_answers()
        
        # Interview session state
        self.current_session = None
        
        # Question generation templates
        self.question_templates = {
            QuestionType.HR_BEHAVIORAL: {
                "template": "Generate a behavioral interview question about {topic} for a {role} position. The question should assess {skill} and be {difficulty} level.",
                "topics": ["leadership", "conflict resolution", "teamwork", "problem solving", "time management", "adaptability"],
                "skills": ["leadership skills", "communication", "collaboration", "analytical thinking", "organization", "flexibility"]
            },
            QuestionType.HR_SOFT_SKILLS: {
                "template": "Create a soft skills interview question about {topic} for a {role} position. Focus on {skill} at {difficulty} level.",
                "topics": ["communication", "emotional intelligence", "work ethic", "creativity", "stress management", "cultural fit"],
                "skills": ["verbal communication", "empathy", "reliability", "innovation", "resilience", "team integration"]
            },
            QuestionType.TECHNICAL_CODING: {
                "template": "Generate a {difficulty} level coding question about {topic} for a {role} position. The question should test {skill}.",
                "topics": ["algorithms", "data structures", "problem solving", "code optimization", "debugging", "system design"],
                "skills": ["algorithmic thinking", "data manipulation", "logical reasoning", "performance analysis", "error handling", "architecture"]
            },
            QuestionType.TECHNICAL_CONCEPTS: {
                "template": "Create a {difficulty} level technical concept question about {topic} for a {role} position. Focus on {skill} understanding.",
                "topics": ["programming paradigms", "databases", "networking", "security", "cloud computing", "software engineering"],
                "skills": ["conceptual understanding", "practical application", "problem analysis", "solution design", "best practices", "industry knowledge"]
            }
        }
        
        # Enhanced evaluation criteria
        self.evaluation_criteria = {
            "relevance": "How well does the answer address the question?",
            "clarity": "Is the answer clear and well-structured?",
            "correctness": "Are the technical details and facts correct?",
            "completeness": "Does the answer cover all important aspects?",
            "examples": "Does the answer include relevant examples?",
            "confidence": "Does the candidate sound confident and knowledgeable?"
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

    def _init_prompt_templates(self):
        """Initialize prompt templates for consistent outputs"""
        
        # Question generation template (string-based for DialoGPT)
        self.question_generation_template = """You are an expert HR interviewer. Generate a {difficulty} level {question_type} question about {topic} for a {role} position.

Context: {context}

The question should:
- Assess {skill}
- Be clear and specific
- Allow for detailed responses
- Be appropriate for the difficulty level

Question:"""
        
        # Answer evaluation template (string-based for DialoGPT)
        self.evaluation_template = """You are an expert interviewer evaluating a candidate's response. Analyze the following:

Question: {question}
Answer: {answer}

Evaluation Criteria:
{criteria}

Expected Keywords: {expected_keywords}

Provide a detailed evaluation in JSON format with the following structure:
{{
    "overall_score": <0-100>,
    "scores": {{
        "relevance": <0-100>,
        "completeness": <0-100>,
        "clarity": <0-100>,
        "technical_accuracy": <0-100>,
        "examples": <0-100>,
        "confidence": <0-100>
    }},
    "feedback": "<detailed feedback>",
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "suggestions": ["<suggestion1>", "<suggestion2>"],
    "keywords_found": ["<keyword1>", "<keyword2>"],
    "missing_keywords": ["<keyword1>", "<keyword2>"],
    "confidence_level": "<high/medium/low>"
}}

Evaluation:"""

    def generate_question(
        self, 
        question_type: QuestionType, 
        role: str = "Software Engineer",
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        context: Optional[str] = None,
        previous_questions: List[str] = None
    ) -> Question:
        """
        Generate a dynamic interview question
        
        Args:
            question_type: Type of question to generate
            role: Job role/position
            difficulty: Difficulty level
            context: Additional context for question generation
            previous_questions: List of previously asked questions to avoid repetition
            
        Returns:
            Generated Question object
        """
        try:
            # Get template info
            template_info = self.question_templates.get(question_type, {})
            template = template_info.get("template", "Generate a {difficulty} level question about {topic} for a {role} position.")
            topics = template_info.get("topics", ["general"])
            skills = template_info.get("skills", ["general skills"])
            
            # Select random topic and skill
            import random
            topic = random.choice(topics)
            skill = random.choice(skills)
            
            # Build context
            full_context = context or ""
            if previous_questions:
                full_context += f"\nPreviously asked questions: {', '.join(previous_questions[-3:])}"
            
            # Generate question using LLM
            if self.use_openai and self.openai_client:
                question_text = self._generate_with_openai(
                    template, topic, role, skill, difficulty.value, full_context
                )
            else:
                question_text = self._generate_with_local(
                    template, topic, role, skill, difficulty.value, full_context
                )
            
            # Extract expected keywords based on question type and topic
            expected_keywords = self._extract_expected_keywords(question_type, topic)
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(question_text, question_type)
            
            return Question(
                question_text=question_text,
                question_type=question_type,
                difficulty=difficulty,
                context=full_context,
                expected_keywords=expected_keywords,
                follow_up_questions=follow_up_questions
            )
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            # Return fallback question
            return Question(
                question_text=f"Tell me about your experience with {question_type.value.replace('_', ' ')}.",
                question_type=question_type,
                difficulty=difficulty,
                context=context,
                expected_keywords=[],
                follow_up_questions=[]
            )

    def _generate_with_openai(self, template: str, topic: str, role: str, skill: str, difficulty: str, context: str) -> str:
        """Generate question using OpenAI API"""
        prompt = template.format(
            topic=topic, role=role, skill=skill, difficulty=difficulty
        )
        
        if context:
            prompt += f"\n\nContext: {context}"
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert HR interviewer. Generate clear, specific interview questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

    def _generate_with_local(self, template: str, topic: str, role: str, skill: str, difficulty: str, context: str) -> str:
        """Generate question using DialoGPT-Medium"""
        # Create a conversational prompt for DialoGPT
        conversation_prompt = f"Interviewer: I need to create a {difficulty} level {template.split('{')[0].strip()} question about {topic} for a {role} position. The question should assess {skill}."
        
        if context:
            conversation_prompt += f" Context: {context}"
        
        conversation_prompt += "\nInterviewer: Here's the question:"
        
        # Generate using DialoGPT-Medium
        result = self.local_llm(
            conversation_prompt,
            max_length=len(conversation_prompt.split()) + 100,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        generated_text = result[0]['generated_text']
        # Extract the question part after the prompt
        question = generated_text[len(conversation_prompt):].strip()
        
        # Clean up the question - remove any remaining conversation markers
        if "Interviewer:" in question:
            question = question.split("Interviewer:")[0].strip()
        if "Candidate:" in question:
            question = question.split("Candidate:")[0].strip()
        
        # Ensure it ends with a question mark
        if not question.endswith('?'):
            question += '?'
        
        return question

    def _extract_expected_keywords(self, question_type: QuestionType, topic: str) -> List[str]:
        """Extract expected keywords based on question type and topic"""
        keyword_mapping = {
            QuestionType.HR_BEHAVIORAL: ["experience", "situation", "task", "action", "result", "team", "leadership", "challenge"],
            QuestionType.HR_SOFT_SKILLS: ["communication", "collaboration", "adaptability", "problem-solving", "initiative"],
            QuestionType.TECHNICAL_CODING: ["algorithm", "complexity", "optimization", "data structure", "implementation"],
            QuestionType.TECHNICAL_CONCEPTS: ["architecture", "design pattern", "best practice", "scalability", "security"]
        }
        
        base_keywords = keyword_mapping.get(question_type, [])
        topic_keywords = topic.lower().split()
        
        return list(set(base_keywords + topic_keywords))

    def _generate_follow_up_questions(self, question: str, question_type: QuestionType) -> List[str]:
        """Generate follow-up questions based on the main question"""
        follow_up_templates = {
            QuestionType.HR_BEHAVIORAL: [
                "Can you give me a specific example?",
                "What was the outcome?",
                "What would you do differently?",
                "How did you handle the challenges?"
            ],
            QuestionType.TECHNICAL_CODING: [
                "Can you walk me through your approach?",
                "What's the time complexity?",
                "How would you optimize this?",
                "Can you write the code for this?"
            ],
            QuestionType.TECHNICAL_CONCEPTS: [
                "Can you explain this in more detail?",
                "What are the trade-offs?",
                "How would you implement this?",
                "What are the best practices?"
            ]
        }
        
        return follow_up_templates.get(question_type, ["Can you elaborate on that?"])

    def evaluate_answer(
        self, 
        question: str, 
        answer: str, 
        expected_keywords: List[str] = None,
        question_type: QuestionType = None
    ) -> EvaluationResult:
        """
        Evaluate candidate's answer and provide structured feedback
        
        Args:
            question: The interview question
            answer: Candidate's answer
            expected_keywords: Keywords that should be present
            question_type: Type of question for context
            
        Returns:
            EvaluationResult with detailed analysis
        """
        try:
            # Prepare evaluation data
            criteria_text = "\n".join([f"- {k}: {v}" for k, v in self.evaluation_criteria.items()])
            keywords_text = ", ".join(expected_keywords or [])
            
            # Generate evaluation using LLM
            if self.use_openai and self.openai_client:
                evaluation_json = self._evaluate_with_openai(
                    question, answer, criteria_text, keywords_text
                )
            else:
                evaluation_json = self._evaluate_with_local(
                    question, answer, criteria_text, keywords_text
                )
            
            # Parse and validate evaluation
            evaluation_result = self._parse_evaluation_result(evaluation_json)
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            # Return fallback evaluation
            return self._create_fallback_evaluation(question, answer)

    def _evaluate_with_openai(self, question: str, answer: str, criteria: str, keywords: str) -> str:
        """Evaluate answer using OpenAI API"""
        prompt = self.evaluation_template.format(
            question=question,
            answer=answer,
            criteria=criteria,
            expected_keywords=keywords
        )
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert interviewer. Provide detailed, constructive feedback in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()

    def _evaluate_with_local(self, question: str, answer: str, criteria: str, keywords: str) -> str:
        """Evaluate answer using DialoGPT-Medium"""
        # Create a conversational evaluation prompt for DialoGPT
        conversation_prompt = f"""Interviewer: I need to evaluate this interview answer. Please provide a detailed assessment.

Question: {question}
Answer: {answer}

Evaluation Criteria:
{criteria}

Expected Keywords: {keywords}

Interviewer: Please provide your evaluation in this JSON format:
{{
    "overall_score": <0-100>,
    "scores": {{
        "relevance": <0-100>,
        "completeness": <0-100>,
        "clarity": <0-100>,
        "technical_accuracy": <0-100>,
        "examples": <0-100>,
        "confidence": <0-100>
    }},
    "feedback": "<detailed feedback>",
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "suggestions": ["<suggestion1>", "<suggestion2>"],
    "keywords_found": ["<keyword1>", "<keyword2>"],
    "missing_keywords": ["<keyword1>", "<keyword2>"],
    "confidence_level": "<high/medium/low>"
}}

Interviewer: Here's my evaluation:"""
        
        result = self.local_llm(
            conversation_prompt,
            max_length=len(conversation_prompt.split()) + 300,
            num_return_sequences=1,
            temperature=0.4,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        generated_text = result[0]['generated_text']
        evaluation = generated_text[len(conversation_prompt):].strip()
        
        # Clean up the response - remove conversation markers
        if "Interviewer:" in evaluation:
            evaluation = evaluation.split("Interviewer:")[0].strip()
        if "Candidate:" in evaluation:
            evaluation = evaluation.split("Candidate:")[0].strip()
        
        # Try to extract JSON from the response
        if "{" in evaluation and "}" in evaluation:
            start = evaluation.find("{")
            end = evaluation.rfind("}") + 1
            evaluation = evaluation[start:end]
        
        return evaluation

    def _parse_evaluation_result(self, evaluation_json: str) -> EvaluationResult:
        """Parse and validate evaluation JSON result"""
        try:
            # Clean up the JSON string
            evaluation_json = evaluation_json.strip()
            if not evaluation_json.startswith("{"):
                # Try to find JSON in the response
                start = evaluation_json.find("{")
                if start != -1:
                    evaluation_json = evaluation_json[start:]
            
            # Parse JSON
            data = json.loads(evaluation_json)
            
            # Validate and extract data
            overall_score = float(data.get("overall_score", 0))
            scores = data.get("scores", {})
            feedback = data.get("feedback", "No feedback provided")
            strengths = data.get("strengths", [])
            weaknesses = data.get("weaknesses", [])
            suggestions = data.get("suggestions", [])
            keywords_found = data.get("keywords_found", [])
            missing_keywords = data.get("missing_keywords", [])
            confidence_level = data.get("confidence_level", "medium")
            
            # Ensure scores are floats
            for key in self.evaluation_criteria.keys():
                if key not in scores:
                    scores[key] = 0.0
                else:
                    scores[key] = float(scores[key])
            
            return EvaluationResult(
                overall_score=overall_score,
                scores=scores,
                feedback=feedback,
                strengths=strengths,
                weaknesses=weaknesses,
                suggestions=suggestions,
                keywords_found=keywords_found,
                missing_keywords=missing_keywords,
                confidence_level=confidence_level
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing evaluation result: {e}")
            return self._create_fallback_evaluation("", "")

    def _create_fallback_evaluation(self, question: str, answer: str) -> EvaluationResult:
        """Create a fallback evaluation when LLM fails"""
        return EvaluationResult(
            overall_score=50.0,
            scores={
                "relevance": 50.0,
                "completeness": 50.0,
                "clarity": 50.0,
                "technical_accuracy": 50.0,
                "examples": 50.0,
                "confidence": 50.0
            },
            feedback="Unable to provide detailed evaluation. Please try again.",
            strengths=["Response provided"],
            weaknesses=["Evaluation unavailable"],
            suggestions=["Try rephrasing your answer"],
            keywords_found=[],
            missing_keywords=[],
            confidence_level="low"
        )

    def generate_interview_flow(
        self, 
        role: str = "Software Engineer",
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        num_questions: int = 5
    ) -> List[Question]:
        """
        Generate a complete interview flow with multiple questions
        
        Args:
            role: Job role/position
            difficulty: Overall difficulty level
            num_questions: Number of questions to generate
            
        Returns:
            List of Question objects forming an interview flow
        """
        questions = []
        question_types = [QuestionType.HR_BEHAVIORAL, QuestionType.TECHNICAL_CODING, QuestionType.TECHNICAL_CONCEPTS]
        
        for i in range(num_questions):
            # Alternate between question types
            question_type = question_types[i % len(question_types)]
            
            # Add context from previous questions
            context = f"Interview for {role} position. Question {i+1} of {num_questions}."
            if questions:
                context += f" Previous questions covered: {', '.join([q.question_type.value for q in questions[-2:]])}"
            
            question = self.generate_question(
                question_type=question_type,
                role=role,
                difficulty=difficulty,
                context=context,
                previous_questions=[q.question_text for q in questions]
            )
            
            questions.append(question)
        
        return questions

# Convenience functions for easy usage
def generate_question(
    question_type: str = "hr_behavioral",
    role: str = "Software Engineer",
    difficulty: str = "medium"
) -> Question:
    """Convenience function to generate a single question"""
    processor = LLMProcessor()
    return processor.generate_question(
        QuestionType(question_type),
        role,
        DifficultyLevel(difficulty)
    )

def evaluate_answer(question: str, answer: str, expected_keywords: List[str] = None) -> EvaluationResult:
    """Convenience function to evaluate an answer"""
    processor = LLMProcessor()
    return processor.evaluate_answer(question, answer, expected_keywords)

def generate_interview_flow(
    role: str = "Software Engineer",
    difficulty: str = "medium",
    num_questions: int = 5
) -> List[Question]:
    """Convenience function to generate a complete interview flow"""
    processor = LLMProcessor()
    return processor.generate_interview_flow(
        role,
        DifficultyLevel(difficulty),
        num_questions
    )
