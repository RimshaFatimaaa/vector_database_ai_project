#!/usr/bin/env python3
"""
LangGraph Interview Coach Demo
Comprehensive demonstration of LangGraph-powered interview workflow
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.langgraph_processor import (
    LangGraphInterviewProcessor,
    create_interview_processor,
    InterviewState
)

def print_separator(title: str = "", char: str = "=", width: int = 60):
    """Print a formatted separator"""
    if title:
        print(f"\n{char * width}")
        print(f"{title.center(width)}")
        print(f"{char * width}")
    else:
        print(f"\n{char * width}")

def print_result(result: Dict[str, Any], show_details: bool = True):
    """Print interview result in a formatted way"""
    print(f"\n🎯 Question Generated:")
    print(f"   {result.get('question', 'N/A')}")
    
    if result.get('candidate_answer'):
        print(f"\n💬 Candidate Answer:")
        print(f"   {result['candidate_answer']}")
    
    if result.get('evaluation'):
        print(f"\n📊 Evaluation:")
        print(f"   {result['evaluation']}")
    
    if result.get('score'):
        print(f"\n⭐ Score: {result['score']}")
    
    if show_details and result.get('conversation_history'):
        print(f"\n📝 Conversation History:")
        for i, interaction in enumerate(result['conversation_history'], 1):
            print(f"   Round {i}: {interaction.get('round_type', 'Unknown')} - Score: {interaction.get('score', {}).get('overall', 'N/A')}")

def demo_basic_workflow():
    """Demonstrate basic LangGraph workflow"""
    print_separator("BASIC LANGGRAPH WORKFLOW DEMO")
    
    # Create processor
    processor = create_interview_processor(use_openai=True)
    
    print("🚀 Initializing LangGraph Interview Processor...")
    print(f"   Model: {processor.model}")
    print(f"   OpenAI Enabled: {processor.use_openai}")
    
    # Demo 1: HR Round
    print_separator("HR INTERVIEW ROUND")
    
    hr_result = processor.run_interview_round(
        round_type="HR",
        context="Candidate is a software engineer with 2 years of experience",
        candidate_answer="I once handled a conflict by organizing a team meeting and discussing our goals openly to find a shared solution."
    )
    
    print_result(hr_result)
    
    # Demo 2: Technical Round
    print_separator("TECHNICAL INTERVIEW ROUND")
    
    tech_result = processor.run_interview_round(
        round_type="Technical",
        context="Candidate is applying for a backend developer position",
        candidate_answer="I would use a hash map to store the frequency of each character, then iterate through the string to find the first character with frequency 1."
    )
    
    print_result(tech_result)
    
    return processor

def demo_multiple_rounds():
    """Demonstrate multiple interview rounds"""
    print_separator("MULTIPLE INTERVIEW ROUNDS DEMO")
    
    processor = create_interview_processor(use_openai=True)
    
    # Define different interview scenarios
    scenarios = [
        {
            "round_type": "HR",
            "context": "Senior software engineer with 5 years experience",
            "answer": "I led a team of 4 developers on a critical project. We used agile methodology and daily standups to ensure smooth communication."
        },
        {
            "round_type": "Technical",
            "context": "Full-stack developer position",
            "answer": "For a REST API, I would use Express.js with proper error handling, input validation, and rate limiting. I'd also implement JWT authentication."
        },
        {
            "round_type": "Behavioral",
            "context": "Team lead role",
            "answer": "When facing a tight deadline, I prioritize tasks based on impact and dependencies, communicate with stakeholders about trade-offs, and ensure quality isn't compromised."
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print_separator(f"ROUND {i}: {scenario['round_type'].upper()}")
        
        result = processor.run_interview_round(
            round_type=scenario["round_type"],
            context=scenario["context"],
            candidate_answer=scenario["answer"]
        )
        
        results.append(result)
        print_result(result, show_details=False)
        
        # Add delay for better demo experience
        time.sleep(1)
    
    # Show session summary
    print_separator("SESSION SUMMARY")
    summary = processor.get_session_summary()
    
    print(f"📈 Total Interactions: {summary['total_interactions']}")
    print(f"🎯 Round Types: {', '.join(summary['round_types'])}")
    print(f"⭐ Average Score: {summary['average_score']:.1f}")
    print(f"🧠 Model Used: {summary['model']}")
    print(f"💾 Memory Available: {summary['memory_available']}")
    
    return results

def demo_state_management():
    """Demonstrate state management capabilities"""
    print_separator("STATE MANAGEMENT DEMO")
    
    processor = create_interview_processor(use_openai=True)
    
    # Create initial state
    print("🔧 Creating initial interview state...")
    initial_state = InterviewState(
        round_type="Technical",
        context="Machine learning engineer position",
        metadata={"session_id": "demo_001", "candidate_level": "senior"}
    )
    
    print(f"   Round Type: {initial_state.round_type}")
    print(f"   Context: {initial_state.context}")
    print(f"   Metadata: {initial_state.metadata}")
    
    # Run workflow
    print("\n🚀 Running interview workflow...")
    result = processor.run_interview_round(
        round_type=initial_state.round_type,
        context=initial_state.context,
        candidate_answer="I would implement a neural network using TensorFlow, with proper data preprocessing, cross-validation, and hyperparameter tuning."
    )
    
    print_result(result)
    
    # Show conversation history
    print_separator("CONVERSATION HISTORY")
    history = processor.get_conversation_history()
    
    if history:
        for i, interaction in enumerate(history, 1):
            print(f"Interaction {i}:")
            print(f"  Round: {interaction.get('round_type', 'Unknown')}")
            print(f"  Question: {interaction.get('question', 'N/A')[:50]}...")
            print(f"  Score: {interaction.get('score', {}).get('overall', 'N/A')}")
            print()

def demo_error_handling():
    """Demonstrate error handling capabilities"""
    print_separator("ERROR HANDLING DEMO")
    
    # Test with invalid API key
    print("🧪 Testing with invalid OpenAI API key...")
    
    # Temporarily modify environment
    original_key = os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = 'invalid-key'
    
    try:
        processor = create_interview_processor(use_openai=True)
        
        result = processor.run_interview_round(
            round_type="HR",
            context="Test context",
            candidate_answer="Test answer"
        )
        
        print("✅ Error handling successful - fallback behavior activated")
        print_result(result)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        # Restore original key
        if original_key:
            os.environ['OPENAI_API_KEY'] = original_key
        else:
            os.environ.pop('OPENAI_API_KEY', None)
    
    # Test with fallback mode
    print("\n🧪 Testing fallback mode...")
    processor_fallback = create_interview_processor(use_openai=False)
    
    result_fallback = processor_fallback.run_interview_round(
        round_type="Technical",
        context="Backend developer",
        candidate_answer="I would use microservices architecture with proper API design."
    )
    
    print("✅ Fallback mode working correctly")
    print_result(result_fallback)

def demo_custom_workflow():
    """Demonstrate custom workflow configuration"""
    print_separator("CUSTOM WORKFLOW DEMO")
    
    # Create processor with custom model
    processor = create_interview_processor(use_openai=True, model="gpt-4o-mini")
    
    print(f"🔧 Custom Configuration:")
    print(f"   Model: {processor.model}")
    print(f"   Temperature: 0.7")
    print(f"   Memory: Enabled")
    
    # Run custom interview flow
    custom_scenarios = [
        {
            "name": "System Design",
            "round_type": "Technical",
            "context": "Senior engineer designing scalable systems",
            "answer": "I would design a distributed system using microservices, load balancers, caching layers, and database sharding for horizontal scaling."
        },
        {
            "name": "Leadership",
            "round_type": "Behavioral",
            "context": "Engineering manager role",
            "answer": "I believe in servant leadership. I focus on removing blockers, providing mentorship, and creating an environment where team members can thrive."
        }
    ]
    
    for scenario in custom_scenarios:
        print_separator(f"CUSTOM SCENARIO: {scenario['name']}")
        
        result = processor.run_interview_round(
            round_type=scenario["round_type"],
            context=scenario["context"],
            candidate_answer=scenario["answer"]
        )
        
        print_result(result, show_details=False)

def interactive_demo():
    """Interactive demo where user can input their own answers"""
    print_separator("INTERACTIVE DEMO")
    
    processor = create_interview_processor(use_openai=True)
    
    print("🎮 Interactive Interview Demo")
    print("You can provide your own answers to see how the system evaluates them.")
    print("Type 'quit' to exit the demo.\n")
    
    while True:
        print("\n" + "-" * 40)
        round_type = input("Enter round type (HR/Technical/Behavioral) or 'quit': ").strip()
        
        if round_type.lower() == 'quit':
            break
        
        if round_type not in ['HR', 'Technical', 'Behavioral']:
            print("❌ Invalid round type. Please use HR, Technical, or Behavioral.")
            continue
        
        context = input("Enter candidate context (or press Enter for default): ").strip()
        if not context:
            context = f"Candidate applying for {round_type} position"
        
        # Generate question
        print("\n🤖 Generating question...")
        question_result = processor.run_interview_round(
            round_type=round_type,
            context=context,
            candidate_answer=None
        )
        
        print(f"\n🧠 Question: {question_result['question']}")
        
        # Get user answer
        user_answer = input("\n💬 Your answer: ").strip()
        
        if not user_answer:
            print("❌ No answer provided. Skipping evaluation.")
            continue
        
        # Evaluate answer
        print("\n🤖 Evaluating your answer...")
        evaluation_result = processor.run_interview_round(
            round_type=round_type,
            context=context,
            candidate_answer=user_answer
        )
        
        print_result(evaluation_result, show_details=False)
    
    print("\n👋 Thanks for trying the interactive demo!")

def main():
    """Main demo function"""
    print("🚀 LangGraph Interview Coach - Comprehensive Demo")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("   The demo will run in fallback mode without OpenAI integration.")
        print("   To enable full functionality, set your OpenAI API key in a .env file.\n")
    
    try:
        # Run different demo sections
        demo_basic_workflow()
        demo_multiple_rounds()
        demo_state_management()
        demo_error_handling()
        demo_custom_workflow()
        
        # Ask if user wants interactive demo
        print_separator("INTERACTIVE DEMO OPTION")
        response = input("Would you like to try the interactive demo? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            interactive_demo()
        
        print_separator("DEMO COMPLETE")
        print("✅ All demos completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print("   • LangGraph workflow orchestration")
        print("   • State management and persistence")
        print("   • Error handling and fallback mechanisms")
        print("   • Multiple interview round types")
        print("   • Conversation history tracking")
        print("   • Custom workflow configuration")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo encountered an error: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
