#!/usr/bin/env python3
"""
Test script for DialoGPT-Medium integration
Tests question generation and answer evaluation using DialoGPT-Medium
"""

import sys
import os
import logging
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.llm_processor_simple import SimpleLLMProcessor, QuestionType, DifficultyLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dialogpt_initialization():
    """Test DialoGPT-Medium initialization"""
    print("=" * 60)
    print("Testing DialoGPT-Medium Initialization")
    print("=" * 60)
    
    try:
        # Initialize processor with local models
        processor = SimpleLLMProcessor(use_openai=False)
        print("✅ DialoGPT-Medium initialized successfully")
        print(f"   Model loaded: {processor.local_llm is not None}")
        print(f"   Tokenizer loaded: {processor.tokenizer is not None}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize DialoGPT-Medium: {e}")
        return False

def test_question_generation():
    """Test question generation using DialoGPT-Medium"""
    print("\n" + "=" * 60)
    print("Testing Question Generation with DialoGPT-Medium")
    print("=" * 60)
    
    try:
        processor = SimpleLLMProcessor(use_openai=False)
        
        # Test different question types and difficulties
        test_cases = [
            (QuestionType.HR_BEHAVIORAL, DifficultyLevel.EASY),
            (QuestionType.HR_BEHAVIORAL, DifficultyLevel.MEDIUM),
            (QuestionType.HR_BEHAVIORAL, DifficultyLevel.HARD),
            (QuestionType.TECHNICAL, DifficultyLevel.EASY),
            (QuestionType.TECHNICAL, DifficultyLevel.MEDIUM),
            (QuestionType.TECHNICAL, DifficultyLevel.HARD),
        ]
        
        for question_type, difficulty in test_cases:
            print(f"\n--- Testing {question_type.value} - {difficulty.value} ---")
            
            question = processor.generate_question(
                question_type=question_type,
                role="Software Engineer",
                difficulty=difficulty
            )
            
            print(f"Question: {question.question_text}")
            print(f"Expected Keywords: {question.expected_keywords}")
            print(f"Type: {question.question_type.value}")
            print(f"Difficulty: {question.difficulty.value}")
            
            # Basic validation
            assert len(question.question_text) > 10, "Question too short"
            assert question.question_text.endswith('?'), "Question should end with ?"
            assert len(question.expected_keywords) > 0, "Should have expected keywords"
            
            print("✅ Question generation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        return False

def test_answer_evaluation():
    """Test answer evaluation using DialoGPT-Medium"""
    print("\n" + "=" * 60)
    print("Testing Answer Evaluation with DialoGPT-Medium")
    print("=" * 60)
    
    try:
        processor = SimpleLLMProcessor(use_openai=False)
        
        # Test cases with different quality answers
        test_cases = [
            {
                "question": "Tell me about a time when you demonstrated leadership in your previous role.",
                "answer": "I led a team of 5 developers on a critical project where we had to migrate our legacy system to microservices architecture. I organized daily standups, created a detailed project timeline, and ensured clear communication between frontend, backend, and DevOps teams. We successfully delivered the project 2 weeks ahead of schedule while maintaining 99.9% uptime.",
                "expected_score_range": (80, 95)
            },
            {
                "question": "Describe a technical project you worked on and the challenges you faced.",
                "answer": "I worked on a Python web application using Django and PostgreSQL. The main challenge was optimizing database queries for better performance. I implemented caching with Redis and used database indexing to improve response times from 2 seconds to 200ms.",
                "expected_score_range": (75, 90)
            },
            {
                "question": "Tell me about your experience with teamwork.",
                "answer": "I guess I'm okay at teamwork. I worked with some people on a project at my last job.",
                "expected_score_range": (30, 60)
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Question: {test_case['question']}")
            print(f"Answer: {test_case['answer'][:100]}...")
            
            evaluation = processor.evaluate_answer(
                question=test_case['question'],
                candidate_answer=test_case['answer']
            )
            
            print(f"Overall Score: {evaluation.overall_score:.1f}/100")
            print(f"Relevance: {evaluation.relevance_score:.1f}/100")
            print(f"Clarity: {evaluation.clarity_score:.1f}/100")
            print(f"Correctness: {evaluation.correctness_score:.1f}/100")
            print(f"Feedback: {evaluation.feedback}")
            print(f"Suggestions: {evaluation.suggestions}")
            
            # Basic validation
            assert 0 <= evaluation.overall_score <= 100, "Score should be between 0-100"
            assert len(evaluation.feedback) > 10, "Feedback should be meaningful"
            assert len(evaluation.suggestions) > 0, "Should have suggestions"
            
            print("✅ Answer evaluation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Answer evaluation failed: {e}")
        return False

def test_session_management():
    """Test session management and tracking"""
    print("\n" + "=" * 60)
    print("Testing Session Management")
    print("=" * 60)
    
    try:
        processor = SimpleLLMProcessor(use_openai=False)
        
        # Generate a few questions
        for i in range(3):
            question = processor.generate_question(
                question_type=QuestionType.HR_BEHAVIORAL,
                difficulty=DifficultyLevel.MEDIUM
            )
            
            # Evaluate a sample answer
            evaluation = processor.evaluate_answer(
                question=question.question_text,
                candidate_answer="This is a sample answer for testing purposes."
            )
        
        # Check session summary
        session_summary = processor.get_session_summary()
        print(f"Total Questions: {session_summary['total_questions']}")
        print(f"Average Score: {session_summary['average_score']:.1f}")
        print(f"Current Difficulty: {session_summary['current_difficulty']}")
        print(f"Questions Asked: {len(session_summary['questions_asked'])}")
        
        # Basic validation
        assert session_summary['total_questions'] == 3, "Should have 3 questions"
        assert session_summary['average_score'] > 0, "Should have positive average score"
        
        print("✅ Session management successful")
        return True
        
    except Exception as e:
        print(f"❌ Session management failed: {e}")
        return False

def main():
    """Run all tests"""
    print("DialoGPT-Medium Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Initialization", test_dialogpt_initialization),
        ("Question Generation", test_question_generation),
        ("Answer Evaluation", test_answer_evaluation),
        ("Session Management", test_session_management),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! DialoGPT-Medium integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
