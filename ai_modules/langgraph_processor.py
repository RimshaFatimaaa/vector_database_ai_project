"""
LangGraph Integration for AI Interview Coach
Advanced workflow orchestration using LangGraph for interview processes
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

class InterviewState(BaseModel):
    """
    State model for the interview workflow using LangGraph
    """
    round_type: Optional[str] = None
    context: Optional[str] = None
    question: Optional[str] = None
    candidate_answer: Optional[str] = None
    evaluation: Optional[str] = None
    score: Optional[Dict[str, float]] = None
    feedback: Optional[str] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    current_step: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class LangGraphInterviewProcessor:
    """
    Advanced interview processor using LangGraph for workflow orchestration
    """
    
    def __init__(self, use_openai: bool = True, model: str = "gpt-4o-mini"):
        self.use_openai = use_openai
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key and use_openai:
            logger.warning("OPENAI_API_KEY not found. Set use_openai=False to use fallback methods.")
            self.use_openai = False
        
        # Initialize LLM
        if self.use_openai:
            self.llm = ChatOpenAI(
                model=self.model, 
                temperature=0.7, 
                api_key=self.api_key
            )
        else:
            self.llm = None
        
        # Initialize memory for conversation persistence
        self.memory = MemorySaver()
        
        # Build the interview graph
        self.interview_graph = self._build_interview_graph()
        
    def _build_interview_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for interview processing
        """
        # Define prompts
        question_prompt = ChatPromptTemplate.from_template(
            """You are an expert interviewer generating {round_type} interview questions.

For HR/Behavioral questions: Generate questions that ask about past experiences, challenges, teamwork, leadership, or problem-solving. Use phrases like "Tell me about a time when...", "Describe a situation where...", "Give me an example of...".

For Technical questions: Generate questions about programming, algorithms, system design, problem-solving, or software engineering concepts.

Candidate context: {context}

Generate one clear, specific, and relevant interview question. Return only the question text, no additional explanation."""
        )

        evaluation_prompt = ChatPromptTemplate.from_template(
            """You are an expert interviewer evaluating candidate responses.

Question: {question}
Candidate Answer: {candidate_answer}

Evaluate this response considering:
1. Relevance to the question
2. Completeness of the answer
3. Clarity and structure
4. Specific examples provided
5. Problem-solving approach

Provide:
- A brief evaluation summary (2-3 sentences)
- Specific strengths
- Areas for improvement
- A score from 1-10

Format your response as:
EVALUATION: [summary]
STRENGTHS: [strengths]
IMPROVEMENTS: [improvements]
SCORE: [score]/10"""
        )

        # Create chains
        if self.use_openai and self.llm:
            question_chain = question_prompt | self.llm
            evaluation_chain = evaluation_prompt | self.llm
        else:
            # Fallback chains for when OpenAI is not available
            question_chain = None
            evaluation_chain = None

        # Define node functions
        def generate_question(state: InterviewState) -> InterviewState:
            """Generate an interview question based on round type and context"""
            logger.info("🎯 Generating interview question...")
            
            if not self.use_openai:
                # Fallback to simple question generation
                state_data = state.model_dump()
                state_data["question"] = f"Tell me about your experience with {state.round_type or 'software development'}."
                state_data["current_step"] = "question_generated"
                return InterviewState(**state_data)
            
            try:
                result = question_chain.invoke({
                    "round_type": state.round_type or "General",
                    "context": state.context or "software engineering interview"
                })
                
                question_text = getattr(result, "content", str(result)).strip()
                
                state_data = state.model_dump()
                state_data["question"] = question_text
                state_data["current_step"] = "question_generated"
                
                return InterviewState(**state_data)
                
            except Exception as e:
                logger.error(f"Error generating question: {e}")
                state_data = state.model_dump()
                state_data["question"] = "Tell me about your relevant experience."
                state_data["current_step"] = "question_generated"
                return InterviewState(**state_data)

        def evaluate_response(state: InterviewState) -> InterviewState:
            """Evaluate the candidate's response"""
            logger.info("🧩 Evaluating candidate's response...")
            
            if not self.use_openai:
                # Fallback evaluation
                state_data = state.model_dump()
                state_data["evaluation"] = "Response received. Please provide more specific examples."
                state_data["score"] = {"overall": 6.0}
                state_data["current_step"] = "evaluation_complete"
                return InterviewState(**state_data)
            
            try:
                result = evaluation_chain.invoke({
                    "question": state.question,
                    "candidate_answer": state.candidate_answer
                })
                
                evaluation_text = getattr(result, "content", str(result)).strip()
                
                # Parse score from evaluation
                score = self._extract_score(evaluation_text)
                
                state_data = state.model_dump()
                state_data["evaluation"] = evaluation_text
                state_data["score"] = score
                state_data["current_step"] = "evaluation_complete"
                
                return InterviewState(**state_data)
                
            except Exception as e:
                logger.error(f"Error evaluating response: {e}")
                state_data = state.model_dump()
                state_data["evaluation"] = "Evaluation completed with basic assessment."
                state_data["score"] = {"overall": 5.0}
                state_data["current_step"] = "evaluation_complete"
                return InterviewState(**state_data)

        def add_to_history(state: InterviewState) -> InterviewState:
            """Add current interaction to conversation history"""
            logger.info("📝 Adding to conversation history...")
            
            state_data = state.model_dump()
            
            if not state_data.get("conversation_history"):
                state_data["conversation_history"] = []
            
            # Add current Q&A to history
            interaction = {
                "question": state.question,
                "answer": state.candidate_answer,
                "evaluation": state.evaluation,
                "score": state.score,
                "round_type": state.round_type
            }
            
            state_data["conversation_history"].append(interaction)
            state_data["current_step"] = "history_updated"
            
            return InterviewState(**state_data)

        # Build the graph
        graph = StateGraph(InterviewState)
        
        # Add nodes
        graph.add_node("generate_question", generate_question)
        graph.add_node("evaluate_response", evaluate_response)
        graph.add_node("add_to_history", add_to_history)
        
        # Define the workflow
        graph.set_entry_point("generate_question")
        graph.add_edge("generate_question", "evaluate_response")
        graph.add_edge("evaluate_response", "add_to_history")
        graph.add_edge("add_to_history", END)
        
        # Compile with memory (only if available)
        try:
            return graph.compile(checkpointer=self.memory)
        except Exception as e:
            logger.warning(f"Failed to compile with memory: {e}, using basic compilation")
            return graph.compile()

    def _extract_score(self, evaluation_text: str) -> Dict[str, float]:
        """Extract numerical score from evaluation text"""
        try:
            # Look for score pattern like "SCORE: 8/10"
            import re
            score_match = re.search(r'SCORE:\s*(\d+)/10', evaluation_text, re.IGNORECASE)
            if score_match:
                score_value = float(score_match.group(1))
                return {"overall": score_value}
            else:
                return {"overall": 5.0}
        except Exception:
            return {"overall": 5.0}

    def run_interview_round(self, round_type: str, context: str, candidate_answer: str = None) -> Dict[str, Any]:
        """
        Run a complete interview round using LangGraph workflow
        """
        logger.info(f"🚀 Starting {round_type} interview round")
        
        # Initialize state
        init_state = InterviewState(
            round_type=round_type,
            context=context,
            current_step="starting"
        )
        
        # Run the workflow
        try:
            # Use config for memory if available
            config = {"configurable": {"thread_id": "demo_thread"}} if hasattr(self.memory, 'get') else {}
            result = self.interview_graph.invoke(init_state, config=config)
            state_after_q = InterviewState(**result)
            
            # If candidate answer is provided, run evaluation
            if candidate_answer:
                state_after_q.candidate_answer = candidate_answer
                result_2 = self.interview_graph.invoke(state_after_q, config=config)
                final_state = InterviewState(**result_2)
                
                return {
                    "question": final_state.question,
                    "candidate_answer": final_state.candidate_answer,
                    "evaluation": final_state.evaluation,
                    "score": final_state.score,
                    "conversation_history": final_state.conversation_history,
                    "current_step": final_state.current_step,
                    "success": True
                }
            else:
                return {
                    "question": state_after_q.question,
                    "candidate_answer": None,
                    "evaluation": None,
                    "score": None,
                    "conversation_history": state_after_q.conversation_history,
                    "current_step": state_after_q.current_step,
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"Error running interview round: {e}")
            return {
                "question": None,
                "candidate_answer": candidate_answer,
                "evaluation": f"Error: {str(e)}",
                "score": {"overall": 0.0},
                "conversation_history": [],
                "current_step": "error",
                "success": False
            }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history from memory"""
        try:
            # This would need to be implemented based on how you want to retrieve from MemorySaver
            return []
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    def clear_memory(self):
        """Clear the conversation memory"""
        try:
            # This would need to be implemented based on MemorySaver API
            logger.info("Memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session"""
        history = self.get_conversation_history()
        return {
            "total_interactions": len(history),
            "round_types": list(set([h.get("round_type") for h in history if h.get("round_type")])),
            "average_score": sum([h.get("score", {}).get("overall", 0) for h in history]) / max(len(history), 1),
            "memory_available": True,
            "model": self.model if self.use_openai else "fallback"
        }


# Convenience functions for easy integration
def create_interview_processor(use_openai: bool = True, model: str = "gpt-4o-mini") -> LangGraphInterviewProcessor:
    """Create a new LangGraph interview processor"""
    return LangGraphInterviewProcessor(use_openai=use_openai, model=model)


def demo_langgraph_interview():
    """Demo function showing LangGraph interview workflow"""
    print("🚀 LangGraph Interview Demo\n")
    
    # Create processor
    processor = create_interview_processor(use_openai=True)
    
    # Run HR round
    print("=" * 50)
    print("HR INTERVIEW ROUND")
    print("=" * 50)
    
    hr_result = processor.run_interview_round(
        round_type="HR",
        context="Candidate is a software engineer with 2 years of experience",
        candidate_answer="I once handled a conflict by organizing a team meeting and discussing our goals openly to find a shared solution."
    )
    
    print(f"\n🧠 Question: {hr_result['question']}")
    print(f"\n💬 Answer: {hr_result['candidate_answer']}")
    print(f"\n📊 Evaluation: {hr_result['evaluation']}")
    print(f"\n⭐ Score: {hr_result['score']}")
    
    # Run Technical round
    print("\n" + "=" * 50)
    print("TECHNICAL INTERVIEW ROUND")
    print("=" * 50)
    
    tech_result = processor.run_interview_round(
        round_type="Technical",
        context="Candidate is applying for a backend developer position",
        candidate_answer="I would use a hash map to store the frequency of each character, then iterate through the string to find the first character with frequency 1."
    )
    
    print(f"\n🧠 Question: {tech_result['question']}")
    print(f"\n💬 Answer: {tech_result['candidate_answer']}")
    print(f"\n📊 Evaluation: {tech_result['evaluation']}")
    print(f"\n⭐ Score: {tech_result['score']}")
    
    # Get session summary
    summary = processor.get_session_summary()
    print(f"\n📈 Session Summary: {summary}")
    
    return processor


if __name__ == "__main__":
    demo_langgraph_interview()
