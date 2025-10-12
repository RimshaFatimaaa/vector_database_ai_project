"""
AI Modules Package
Contains all the AI processing modules for the Interview Coach application.
"""

# Import main classes and functions for easy access
from .auth import check_auth_status, init_session_state
from .auth_ui import show_auth_page, show_logout_button, show_header_logout
from .nlp_processor import process_interview_response, NLPProcessor
from .llm_processor_simple import SimpleLLMProcessor, QuestionType, DifficultyLevel
from .llm_processor import LLMProcessor
from .langchain_processor import LangChainInterviewProcessor

__all__ = [
    'check_auth_status',
    'init_session_state', 
    'show_auth_page',
    'show_logout_button',
    'show_header_logout',
    'process_interview_response',
    'NLPProcessor',
    'SimpleLLMProcessor',
    'QuestionType',
    'DifficultyLevel',
    'LLMProcessor',
    'LangChainInterviewProcessor'
]
