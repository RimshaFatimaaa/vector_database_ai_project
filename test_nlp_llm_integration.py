"""
Test script for NLP + LLM integration
"""

from ai_modules.nlp_processor import NLPProcessor
from ai_modules.llm_processor_simple import SimpleLLMProcessor, QuestionType, DifficultyLevel

def test_nlp_llm_integration():
    """Test the integration between NLP and LLM processors"""
    print("Testing NLP + LLM Integration...")
    
    try:
        # Initialize processors
        nlp_processor = NLPProcessor()
        llm_processor = SimpleLLMProcessor(use_openai=False)
        print("✅ Processors initialized successfully")
        
        # Test data
        test_response = "Umm I think I am good at teamwo rk, because in my last job I worked with a team of 5 people to build a Python application at Google."
        test_question = "Tell me about teamwork"
        
        # Step 1: NLP preprocessing
        print("\n--- Testing NLP Preprocessing ---")
        cleaned_data = nlp_processor.preprocess_text(test_response)
        print(f"Cleaned data keys: {list(cleaned_data.keys())}")
        print(f"No stopwords: {cleaned_data['no_stopwords']}")
        
        # Step 2: Convert to string for LLM
        cleaned_text = ' '.join(cleaned_data['no_stopwords'])
        print(f"Cleaned text string: {cleaned_text}")
        
        # Step 3: LLM evaluation
        print("\n--- Testing LLM Evaluation ---")
        evaluation = llm_processor.evaluate_answer(
            question=test_question,
            candidate_answer=test_response,
            cleaned_answer=cleaned_text
        )
        
        print(f"Overall Score: {evaluation.overall_score:.1f}/100")
        print(f"Relevance Score: {evaluation.relevance_score:.1f}/100")
        print(f"Clarity Score: {evaluation.clarity_score:.1f}/100")
        print(f"Correctness Score: {evaluation.correctness_score:.1f}/100")
        print(f"Feedback: {evaluation.feedback}")
        print(f"Suggestions: {evaluation.suggestions}")
        
        print("\n✅ Integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during integration test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nlp_llm_integration()
