"""
Test cases for LangGraph Interview Processor
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_modules.langgraph_processor import (
    LangGraphInterviewProcessor, 
    InterviewState, 
    create_interview_processor
)

class TestInterviewState(unittest.TestCase):
    """Test the InterviewState model"""
    
    def test_interview_state_creation(self):
        """Test creating an InterviewState instance"""
        state = InterviewState(
            round_type="HR",
            context="Test context",
            question="Test question"
        )
        
        self.assertEqual(state.round_type, "HR")
        self.assertEqual(state.context, "Test context")
        self.assertEqual(state.question, "Test question")
        self.assertIsNone(state.candidate_answer)
        self.assertIsNone(state.evaluation)

    def test_interview_state_defaults(self):
        """Test InterviewState with default values"""
        state = InterviewState()
        
        self.assertIsNone(state.round_type)
        self.assertIsNone(state.context)
        self.assertIsNone(state.question)
        self.assertIsNone(state.candidate_answer)
        self.assertIsNone(state.evaluation)

    def test_interview_state_model_dump(self):
        """Test InterviewState model_dump method"""
        state = InterviewState(
            round_type="Technical",
            question="What is a hash table?"
        )
        
        data = state.model_dump()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data["round_type"], "Technical")
        self.assertEqual(data["question"], "What is a hash table?")
        self.assertIsNone(data["candidate_answer"])


class TestLangGraphInterviewProcessor(unittest.TestCase):
    """Test the LangGraphInterviewProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
        self.env_patcher.start()
        
        # Create processor without OpenAI (for testing)
        self.processor = LangGraphInterviewProcessor(use_openai=False)
    
    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        processor = LangGraphInterviewProcessor(use_openai=False)
        
        self.assertFalse(processor.use_openai)
        self.assertIsNone(processor.llm)
        self.assertIsNotNone(processor.interview_graph)
    
    def test_processor_with_openai_key(self):
        """Test processor with OpenAI API key"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            processor = LangGraphInterviewProcessor(use_openai=True)
            
            self.assertTrue(processor.use_openai)
            self.assertIsNotNone(processor.llm)
    
    def test_extract_score(self):
        """Test score extraction from evaluation text"""
        # Test with valid score
        evaluation_text = "EVALUATION: Good answer\nSCORE: 8/10"
        score = self.processor._extract_score(evaluation_text)
        
        self.assertEqual(score["overall"], 8.0)
        
        # Test with no score
        evaluation_text_no_score = "EVALUATION: Good answer"
        score = self.processor._extract_score(evaluation_text_no_score)
        
        self.assertEqual(score["overall"], 5.0)
    
    def test_run_interview_round_question_only(self):
        """Test running interview round with question generation only"""
        result = self.processor.run_interview_round(
            round_type="HR",
            context="Test candidate",
            candidate_answer=None
        )
        
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["question"])
        self.assertIsNone(result["candidate_answer"])
        self.assertIsNone(result["evaluation"])
    
    def test_run_interview_round_with_evaluation(self):
        """Test running interview round with question and evaluation"""
        result = self.processor.run_interview_round(
            round_type="Technical",
            context="Backend developer",
            candidate_answer="I would use a hash map to solve this problem."
        )
        
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["question"])
        self.assertEqual(result["candidate_answer"], "I would use a hash map to solve this problem.")
        self.assertIsNotNone(result["evaluation"])
        self.assertIsNotNone(result["score"])
    
    def test_get_session_summary(self):
        """Test getting session summary"""
        summary = self.processor.get_session_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("total_interactions", summary)
        self.assertIn("average_score", summary)
        self.assertIn("memory_available", summary)
        self.assertIn("model", summary)
    
    def test_clear_memory(self):
        """Test clearing memory"""
        # This should not raise an exception
        self.processor.clear_memory()
    
    def test_get_conversation_history(self):
        """Test getting conversation history"""
        history = self.processor.get_conversation_history()
        
        self.assertIsInstance(history, list)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def test_create_interview_processor(self):
        """Test create_interview_processor function"""
        processor = create_interview_processor(use_openai=False)
        
        self.assertIsInstance(processor, LangGraphInterviewProcessor)
        self.assertFalse(processor.use_openai)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.env_patcher = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after integration tests"""
        self.env_patcher.stop()
    
    def test_full_interview_workflow(self):
        """Test complete interview workflow"""
        processor = LangGraphInterviewProcessor(use_openai=False)
        
        # Test multiple rounds
        rounds = [
            ("HR", "Software engineer with 2 years experience"),
            ("Technical", "Backend developer position"),
            ("Behavioral", "Team lead role")
        ]
        
        for round_type, context in rounds:
            result = processor.run_interview_round(
                round_type=round_type,
                context=context,
                candidate_answer=f"Sample answer for {round_type} round"
            )
            
            self.assertTrue(result["success"])
            self.assertIsNotNone(result["question"])
            self.assertIsNotNone(result["evaluation"])
            self.assertIsNotNone(result["score"])
    
    def test_error_handling(self):
        """Test error handling in workflow"""
        processor = LangGraphInterviewProcessor(use_openai=False)
        
        # Test with invalid inputs
        result = processor.run_interview_round(
            round_type=None,
            context=None,
            candidate_answer="Test answer"
        )
        
        # Should still succeed with fallback behavior
        self.assertTrue(result["success"])


class TestMockedOpenAI(unittest.TestCase):
    """Test with mocked OpenAI responses"""
    
    def setUp(self):
        """Set up mocked OpenAI tests"""
        self.env_patcher = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up mocked tests"""
        self.env_patcher.stop()
    
    @patch('ai_modules.langgraph_processor.ChatOpenAI')
    def test_openai_integration(self, mock_chat_openai):
        """Test OpenAI integration with mocked responses"""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "Tell me about a challenging project you worked on."
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        processor = LangGraphInterviewProcessor(use_openai=True)
        
        result = processor.run_interview_round(
            round_type="HR",
            context="Software engineer",
            candidate_answer="I worked on a microservices project."
        )
        
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["question"])
        self.assertIsNotNone(result["evaluation"])


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestInterviewState,
        TestLangGraphInterviewProcessor,
        TestConvenienceFunctions,
        TestIntegration,
        TestMockedOpenAI
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running LangGraph Interview Processor Tests\n")
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
