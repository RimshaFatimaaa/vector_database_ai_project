#!/usr/bin/env python3
"""
Simple test script for the NLP processor
"""

from ai_modules.nlp_processor import process_interview_response

def test_nlp_processor():
    """Test the NLP processor with sample data"""
    print("Testing NLP Processor...")
    
    # Test with sample response
    test_response = "Umm I think I am good at teamwork, because in my last job I worked with a team of 5 people to build a Python application at Google."
    test_question = "Tell me about teamwork"
    
    try:
        result = process_interview_response(test_response, test_question)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print("✅ Test successful!")
        print(f"Overall score: {result['overall_score']}/3")
        print(f"Sentiment: {result['sentiment_label']}")
        print(f"Keywords: {result['keywords']}")
        print(f"Named entities: {result['named_entities']}")
        print(f"Rubric: {result['rubric']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_nlp_processor()
