"""
LangChain Integration for AI Interview Coach
Based on the langchain.ipynb notebook implementation

Updated for Streamlit Cloud deployment with langchain_core imports.
"""

import os
import json
import logging
from typing import Dict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

# Import functions from the existing LLMs_test module
from notebooks.LLMs_test import generate_question, evaluate_answer

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set OpenAI API key from Streamlit secrets
try:
    import streamlit as st
    openai_key = st.secrets["OPENAI_API_KEY"]
    os.environ['OPENAI_API_KEY'] = openai_key
    logger.info("OpenAI API key loaded from Streamlit secrets")
except KeyError:
    logger.warning("OPENAI_API_KEY not found in Streamlit secrets. Please set it to use OpenAI features.")
except Exception as e:
    logger.warning(f"Failed to load OpenAI API key: {e}")


class LangChainInterviewProcessor:
    """
    LangChain-based interview processor that wraps the existing LLMs_test functions
    with LangChain chains for better conversation management and memory.
    """
    
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.question_chain = self._get_question_generation_chain()
        self.evaluation_chain = self._get_answer_evaluation_chain()
        
    def _get_question_generation_chain(self):
        """
        Creates a chain that generates interview questions using the latest ChatOpenAI.
        """
        template = """You are an expert interviewer generating {round_type} interview questions.

For HR/Behavioral questions: Generate questions that ask about past experiences, challenges, teamwork, leadership, or problem-solving. Use phrases like "Tell me about a time when...", "Describe a situation where...", "Give me an example of...".

For Technical questions: Generate questions about programming, algorithms, system design, problem-solving, or software engineering concepts.

Candidate context: {context}

Generate one clear, specific, and relevant interview question. Return only the question text, no additional explanation."""

        prompt = PromptTemplate(
            input_variables=["round_type", "context"],
            template=template
        )

        if self.use_openai:
            try:
                # Use ChatOpenAI (new API)
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
                # Use new pipe syntax
                chain = prompt | llm
                return chain
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI chain: {e}")
                self.use_openai = False
        
        # Fallback to direct function call
        return None

    def _get_answer_evaluation_chain(self):
        """
        Wraps the evaluate_answer() function into a LangChain-like callable.
        """
        def _run(question: str, candidate_answer: str) -> Dict:
            # Use the main evaluate_answer function but force OpenAI usage
            from notebooks.LLMs_test import evaluate_answer
            import os
            
            # Check if OpenAI API key is available
            if not os.environ.get('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            result = evaluate_answer(question, candidate_answer)
            return result

        # Simple chain wrapper
        class EvalChain:
            def __init__(self, memory):
                self.memory = memory

            def __call__(self, question: str, candidate_answer: str):
                result = _run(question, candidate_answer)
                # Add to memory for conversation context
                self.memory.chat_memory.add_user_message(candidate_answer)
                self.memory.chat_memory.add_ai_message(result["evaluation"]["feedback"])
                return result

        return EvalChain(self.memory)

    def generate_question_langchain(self, round_type: str, context: str = None) -> Dict:
        """
        Generate a question using LangChain chain if available, otherwise fallback to direct function.
        """
        if self.question_chain and self.use_openai:
            try:
                inputs = {"round_type": round_type, "context": context or "General candidate"}
                question_output = self.question_chain.invoke(inputs)
                generated_question = question_output.content  # get text output from ChatMessage
                
                return {
                    "question": generated_question,
                    "source": "langchain_openai",
                    "round_type": round_type,
                    "context": context
                }
            except Exception as e:
                logger.warning(f"LangChain generation failed: {e}, falling back to direct function")
        
        # Fallback to direct function call
        result = generate_question(round_type, context)
        result["round_type"] = round_type
        result["context"] = context
        return result

    def evaluate_answer_langchain(self, question: str, candidate_answer: str) -> Dict:
        """
        Evaluate an answer using LangChain chain with memory.
        """
        try:
            evaluation_result = self.evaluation_chain(question, candidate_answer)
            return evaluation_result
        except Exception as e:
            logger.error(f"LangChain evaluation failed: {e}")
            # Fallback to direct function call
            return evaluate_answer(question, candidate_answer)

    def get_conversation_history(self) -> list:
        """
        Get the conversation history from memory.
        """
        return self.memory.chat_memory.messages

    def clear_memory(self):
        """
        Clear the conversation memory.
        """
        self.memory.clear()

    def get_session_summary(self) -> Dict:
        """
        Get a summary of the current session.
        """
        messages = self.get_conversation_history()
        return {
            "total_messages": len(messages),
            "conversation_rounds": len([m for m in messages if isinstance(m, HumanMessage)]),
            "memory_available": True
        }


# Convenience functions for backward compatibility
def get_question_generation_chain():
    """
    Creates a chain that generates interview questions using the latest ChatOpenAI.
    """
    template = """You are an AI interviewer generating {round_type} interview questions.
Candidate context: {context}
Generate one clear and concise question. Return only the question text."""

    prompt = PromptTemplate(
        input_variables=["round_type", "context"],
        template=template
    )

    # Use ChatOpenAI (new API)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Use new pipe syntax
    chain = prompt | llm
    return chain


def get_answer_evaluation_chain():
    """
    Wraps the evaluate_answer() function into a LangChain-like callable.
    """
    def _run(question: str, candidate_answer: str) -> Dict:
        # Directly call the notebook function
        result = evaluate_answer(question, candidate_answer)
        return result

    # Simple chain wrapper
    class EvalChain:
        def __init__(self):
            self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        def __call__(self, question: str, candidate_answer: str):
            result = _run(question, candidate_answer)
            self.memory.chat_memory.add_user_message(candidate_answer)
            self.memory.chat_memory.add_ai_message(result["evaluation"]["feedback"])
            return result

    return EvalChain()


# Example usage function
def demo_langchain_integration():
    """
    Demo function showing how to use the LangChain integration.
    """
    print("🚀 LangChain Integration Demo\n")

    # Create processor
    processor = LangChainInterviewProcessor(use_openai=True)

    # Generate a question
    question_result = processor.generate_question_langchain(
        round_type="HR", 
        context="Candidate is a software engineer with 2 years of experience"
    )
    generated_question = question_result["question"]

    print("\n🧠 Generated Question:")
    print(generated_question)

    # Candidate answer
    candidate_ans = "I once resolved a conflict by organizing a team meeting and discussing responsibilities openly."

    # Evaluate answer
    evaluation_result = processor.evaluate_answer_langchain(generated_question, candidate_ans)
    print("\n✅ Evaluation Result:")
    print(json.dumps(evaluation_result, indent=2))

    return processor


if __name__ == "__main__":
    demo_langchain_integration()
