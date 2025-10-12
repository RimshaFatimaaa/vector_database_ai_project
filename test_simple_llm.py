"""
Test script for simplified LLM processor
"""

from ai_modules.llm_processor_simple import SimpleLLMProcessor, QuestionType, DifficultyLevel

def test_simple_llm():
    """Test the simplified LLM processor"""
    print("Testing Simplified LLM Processor...")
    
    try:
        # Initialize processor
        processor = SimpleLLMProcessor(use_openai=False)
        print("✅ LLM Processor initialized successfully")
        
        # Test question generation
        print("\n--- Testing Question Generation ---")
        question = processor.generate_question(
            QuestionType.HR_BEHAVIORAL,
            "Software Engineer",
            DifficultyLevel.MEDIUM
        )
        print(f"Generated Question: {question.question_text}")
        print(f"Type: {question.question_type.value}")
        print(f"Difficulty: {question.difficulty.value}")
        print(f"Expected Keywords: {question.expected_keywords}")
        
        # Test answer evaluation
        print("\n--- Testing Answer Evaluation ---")
        test_answer = "I worked on a team project where we built a web application. I collaborated with 5 other developers and we used agile methodology. I was responsible for the frontend development using React and JavaScript."
        
        evaluation = processor.evaluate_answer(
            question.question_text,
            test_answer
        )
        
        print(f"Overall Score: {evaluation.overall_score:.1f}/100")
        print(f"Relevance Score: {evaluation.relevance_score:.1f}/100")
        print(f"Clarity Score: {evaluation.clarity_score:.1f}/100")
        print(f"Correctness Score: {evaluation.correctness_score:.1f}/100")
        print(f"Feedback: {evaluation.feedback}")
        print(f"Suggestions: {evaluation.suggestions}")
        
        # Test session summary
        print("\n--- Testing Session Summary ---")
        summary = processor.get_session_summary()
        print(f"Session Summary: {summary}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_llm()
