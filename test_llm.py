"""
Test file for LLM Processor Module - Step 2
Tests dynamic question generation and answer evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.llm_processor import (
    LLMProcessor, 
    QuestionType, 
    DifficultyLevel, 
    generate_question, 
    evaluate_answer, 
    generate_interview_flow
)
import json

def test_question_generation():
    """Test dynamic question generation"""
    print("=" * 60)
    print("TESTING QUESTION GENERATION")
    print("=" * 60)
    
    processor = LLMProcessor(use_openai=False)  # Use local models for testing
    
    # Test different question types
    question_types = [
        QuestionType.HR_BEHAVIORAL,
        QuestionType.HR_SOFT_SKILLS,
        QuestionType.TECHNICAL_CODING,
        QuestionType.TECHNICAL_CONCEPTS
    ]
    
    for q_type in question_types:
        print(f"\n--- {q_type.value.upper()} QUESTION ---")
        question = processor.generate_question(
            question_type=q_type,
            role="Software Engineer",
            difficulty=DifficultyLevel.MEDIUM
        )
        
        print(f"Question: {question.question_text}")
        print(f"Type: {question.question_type.value}")
        print(f"Difficulty: {question.difficulty.value}")
        print(f"Expected Keywords: {question.expected_keywords}")
        print(f"Follow-up Questions: {question.follow_up_questions}")

def test_answer_evaluation():
    """Test answer evaluation"""
    print("\n" + "=" * 60)
    print("TESTING ANSWER EVALUATION")
    print("=" * 60)
    
    processor = LLMProcessor(use_openai=False)
    
    # Test cases
    test_cases = [
        {
            "question": "Tell me about a time you worked in a team.",
            "answer": "In my previous job at Google, I worked with a team of 5 developers to build a Python web application. I was responsible for the backend API development using FastAPI and PostgreSQL. We used agile methodology with daily standups and weekly sprints. The project was challenging because we had to integrate with multiple third-party services, but we successfully delivered it on time. I learned a lot about collaboration and communication.",
            "expected_keywords": ["team", "collaboration", "project", "communication", "experience"]
        },
        {
            "question": "Explain object-oriented programming.",
            "answer": "OOP is a programming paradigm based on objects. It has four main principles: encapsulation, inheritance, polymorphism, and abstraction. Encapsulation means bundling data and methods together. Inheritance allows classes to inherit properties from parent classes. Polymorphism lets objects of different types be treated uniformly. Abstraction hides complex implementation details.",
            "expected_keywords": ["encapsulation", "inheritance", "polymorphism", "abstraction", "objects"]
        },
        {
            "question": "How do you handle conflicts in a team?",
            "answer": "I try to listen to everyone's perspective first. Then I look for common ground and suggest compromises.",
            "expected_keywords": ["listen", "perspective", "compromise", "communication", "conflict resolution"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- TEST CASE {i} ---")
        print(f"Question: {test_case['question']}")
        print(f"Answer: {test_case['answer']}")
        
        evaluation = processor.evaluate_answer(
            question=test_case['question'],
            answer=test_case['answer'],
            expected_keywords=test_case['expected_keywords']
        )
        
        print(f"\nEVALUATION RESULTS:")
        print(f"Overall Score: {evaluation.overall_score}/100")
        print(f"Individual Scores:")
        for criterion, score in evaluation.scores.items():
            print(f"  - {criterion}: {score}/100")
        print(f"Feedback: {evaluation.feedback}")
        print(f"Strengths: {evaluation.strengths}")
        print(f"Weaknesses: {evaluation.weaknesses}")
        print(f"Suggestions: {evaluation.suggestions}")
        print(f"Keywords Found: {evaluation.keywords_found}")
        print(f"Missing Keywords: {evaluation.missing_keywords}")
        print(f"Confidence Level: {evaluation.confidence_level}")

def test_interview_flow():
    """Test complete interview flow generation"""
    print("\n" + "=" * 60)
    print("TESTING INTERVIEW FLOW GENERATION")
    print("=" * 60)
    
    processor = LLMProcessor(use_openai=False)
    
    # Generate interview flow
    interview_flow = processor.generate_interview_flow(
        role="Data Scientist",
        difficulty=DifficultyLevel.MEDIUM,
        num_questions=3
    )
    
    print(f"Generated {len(interview_flow)} questions for Data Scientist position:")
    
    for i, question in enumerate(interview_flow, 1):
        print(f"\n--- QUESTION {i} ---")
        print(f"Type: {question.question_type.value}")
        print(f"Difficulty: {question.difficulty.value}")
        print(f"Question: {question.question_text}")
        print(f"Expected Keywords: {question.expected_keywords}")
        print(f"Follow-up Questions: {question.follow_up_questions}")

def test_convenience_functions():
    """Test convenience functions"""
    print("\n" + "=" * 60)
    print("TESTING CONVENIENCE FUNCTIONS")
    print("=" * 60)
    
    # Test question generation
    print("\n--- Testing generate_question() ---")
    question = generate_question(
        question_type="technical_coding",
        role="Frontend Developer",
        difficulty="hard"
    )
    print(f"Generated Question: {question.question_text}")
    
    # Test answer evaluation
    print("\n--- Testing evaluate_answer() ---")
    evaluation = evaluate_answer(
        question="What is your greatest strength?",
        answer="I'm very detail-oriented and I always make sure my code is clean and well-documented. I also enjoy learning new technologies and I'm always looking for ways to improve my skills.",
        expected_keywords=["strength", "detail", "learning", "improvement"]
    )
    print(f"Evaluation Score: {evaluation.overall_score}/100")
    print(f"Feedback: {evaluation.feedback}")
    
    # Test interview flow generation
    print("\n--- Testing generate_interview_flow() ---")
    flow = generate_interview_flow(
        role="DevOps Engineer",
        difficulty="medium",
        num_questions=2
    )
    print(f"Generated {len(flow)} questions for DevOps Engineer")

def test_json_serialization():
    """Test JSON serialization of results"""
    print("\n" + "=" * 60)
    print("TESTING JSON SERIALIZATION")
    print("=" * 60)
    
    processor = LLMProcessor(use_openai=False)
    
    # Generate a question and evaluation
    question = processor.generate_question(QuestionType.HR_BEHAVIORAL)
    evaluation = processor.evaluate_answer(
        question.question_text,
        "I worked on a team project where we had to deliver a mobile app in 3 months. I was the lead developer and coordinated with 4 other developers. We used Scrum methodology and delivered on time.",
        question.expected_keywords
    )
    
    # Test JSON serialization
    try:
        question_dict = {
            "question_text": question.question_text,
            "question_type": question.question_type.value,
            "difficulty": question.difficulty.value,
            "expected_keywords": question.expected_keywords
        }
        
        evaluation_dict = {
            "overall_score": evaluation.overall_score,
            "scores": evaluation.scores,
            "feedback": evaluation.feedback,
            "strengths": evaluation.strengths,
            "weaknesses": evaluation.weaknesses,
            "suggestions": evaluation.suggestions
        }
        
        print("Question JSON:")
        print(json.dumps(question_dict, indent=2))
        
        print("\nEvaluation JSON:")
        print(json.dumps(evaluation_dict, indent=2))
        
        print("\n✅ JSON serialization successful!")
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")

def main():
    """Run all tests"""
    print("🤖 LLM PROCESSOR MODULE TESTS")
    print("Testing Step 2: Dynamic Question Generation & Answer Evaluation")
    
    try:
        # Run all test functions
        test_question_generation()
        test_answer_evaluation()
        test_interview_flow()
        test_convenience_functions()
        test_json_serialization()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
