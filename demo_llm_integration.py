"""
Demo script showing integration of LLM module with existing NLP module
Demonstrates Step 2: Dynamic Question Generation & Answer Evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.llm_processor import LLMProcessor, QuestionType, DifficultyLevel
from ai_modules.nlp_processor import NLPProcessor
import json
import time

def demo_question_generation():
    """Demonstrate dynamic question generation"""
    print("🎯 DEMO: Dynamic Question Generation")
    print("=" * 50)
    
    processor = LLMProcessor(use_openai=False)  # Use local models for demo
    
    # Generate different types of questions
    roles = ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer"]
    difficulties = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
    
    for role in roles:
        print(f"\n📋 Questions for {role}:")
        print("-" * 30)
        
        for difficulty in difficulties:
            # Generate HR question
            hr_question = processor.generate_question(
                question_type=QuestionType.HR_BEHAVIORAL,
                role=role,
                difficulty=difficulty
            )
            
            # Generate Technical question
            tech_question = processor.generate_question(
                question_type=QuestionType.TECHNICAL_CODING,
                role=role,
                difficulty=difficulty
            )
            
            print(f"\n{difficulty.value.upper()} Level:")
            print(f"  HR: {hr_question.question_text}")
            print(f"  Tech: {tech_question.question_text}")

def demo_answer_evaluation():
    """Demonstrate answer evaluation with different quality responses"""
    print("\n\n🔍 DEMO: Answer Evaluation")
    print("=" * 50)
    
    processor = LLMProcessor(use_openai=False)
    
    # Test cases with different quality answers
    test_cases = [
        {
            "question": "Tell me about a challenging project you worked on.",
            "answers": [
                {
                    "text": "I worked on a machine learning project where we had to predict customer churn. It was really hard because the data was messy and we had to clean it first. We used Python and scikit-learn. The model achieved 85% accuracy and helped the company reduce churn by 20%. I learned a lot about data preprocessing and feature engineering.",
                    "quality": "Excellent"
                },
                {
                    "text": "I had this project where we built a website. It was challenging because we had to learn new technologies. We used React and Node.js. The project was successful and the client was happy.",
                    "quality": "Good"
                },
                {
                    "text": "Umm, I worked on some projects. It was okay, I guess. We used some technologies and stuff. It turned out fine.",
                    "quality": "Poor"
                }
            ]
        },
        {
            "question": "Explain the concept of object-oriented programming.",
            "answers": [
                {
                    "text": "Object-oriented programming is a programming paradigm based on the concept of objects, which contain data and code. The four main principles are encapsulation, inheritance, polymorphism, and abstraction. Encapsulation bundles data and methods together, inheritance allows classes to inherit from parent classes, polymorphism enables objects of different types to be treated uniformly, and abstraction hides complex implementation details while exposing only necessary interfaces.",
                    "quality": "Excellent"
                },
                {
                    "text": "OOP is about objects and classes. You can create objects that have properties and methods. You can also inherit from other classes and override methods. It helps organize code better.",
                    "quality": "Good"
                },
                {
                    "text": "It's like, you know, programming with objects. I'm not really sure about the details.",
                    "quality": "Poor"
                }
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['question']} ---")
        
        for answer_data in test_case['answers']:
            print(f"\n{answer_data['quality']} Answer:")
            print(f"Text: {answer_data['text']}")
            
            # Evaluate the answer
            evaluation = processor.evaluate_answer(
                question=test_case['question'],
                answer=answer_data['text']
            )
            
            print(f"\nEvaluation:")
            print(f"  Overall Score: {evaluation.overall_score}/100")
            print(f"  Individual Scores:")
            for criterion, score in evaluation.scores.items():
                print(f"    - {criterion}: {score}/100")
            print(f"  Feedback: {evaluation.feedback}")
            print(f"  Strengths: {', '.join(evaluation.strengths)}")
            print(f"  Weaknesses: {', '.join(evaluation.weaknesses)}")
            print(f"  Suggestions: {', '.join(evaluation.suggestions)}")

def demo_nlp_llm_integration():
    """Demonstrate integration between NLP and LLM modules"""
    print("\n\n🔗 DEMO: NLP + LLM Integration")
    print("=" * 50)
    
    # Initialize both processors
    nlp_processor = NLPProcessor()
    llm_processor = LLMProcessor(use_openai=False)
    
    # Generate a question
    question = llm_processor.generate_question(
        question_type=QuestionType.HR_BEHAVIORAL,
        role="Software Engineer",
        difficulty=DifficultyLevel.MEDIUM
    )
    
    print(f"Generated Question: {question.question_text}")
    print(f"Expected Keywords: {question.expected_keywords}")
    
    # Sample answer
    answer = "In my previous role at TechCorp, I led a team of 4 developers to build a microservices architecture for our e-commerce platform. We used Docker for containerization and Kubernetes for orchestration. The project was challenging because we had to migrate from a monolithic system while maintaining zero downtime. I coordinated daily standups, managed sprints, and resolved technical conflicts. We successfully delivered the project 2 weeks ahead of schedule, resulting in 40% faster deployment times and 99.9% uptime."
    
    print(f"\nCandidate Answer: {answer}")
    
    # Process with NLP
    print("\n--- NLP Analysis ---")
    nlp_result = nlp_processor.process_response(answer, question.question_text)
    
    print(f"NLP Overall Score: {nlp_result['overall_score']}/3")
    print(f"Sentiment: {nlp_result['sentiment_label']} ({nlp_result['sentiment_score']:.2f})")
    print(f"Keywords Found: {nlp_result['keywords']}")
    print(f"Named Entities: {nlp_result['named_entities']}")
    
    # Process with LLM
    print("\n--- LLM Analysis ---")
    llm_evaluation = llm_processor.evaluate_answer(
        question=question.question_text,
        answer=answer,
        expected_keywords=question.expected_keywords
    )
    
    print(f"LLM Overall Score: {llm_evaluation.overall_score}/100")
    print(f"Individual Scores:")
    for criterion, score in llm_evaluation.scores.items():
        print(f"  - {criterion}: {score}/100")
    print(f"Feedback: {llm_evaluation.feedback}")
    
    # Combined analysis
    print("\n--- Combined Analysis ---")
    combined_score = (nlp_result['overall_score'] * 33.33) + (llm_evaluation.overall_score * 0.67)
    print(f"Combined Score: {combined_score:.1f}/100")
    
    # Check keyword overlap
    nlp_keywords = set([kw.lower() for kw in nlp_result['keywords']])
    expected_keywords = set([kw.lower() for kw in question.expected_keywords])
    keyword_overlap = len(nlp_keywords.intersection(expected_keywords))
    keyword_coverage = (keyword_overlap / len(expected_keywords)) * 100 if expected_keywords else 0
    
    print(f"Keyword Coverage: {keyword_coverage:.1f}% ({keyword_overlap}/{len(expected_keywords)} keywords found)")

def demo_interview_simulation():
    """Demonstrate a complete interview simulation"""
    print("\n\n🎭 DEMO: Complete Interview Simulation")
    print("=" * 50)
    
    processor = LLMProcessor(use_openai=False)
    nlp_processor = NLPProcessor()
    
    # Generate interview flow
    print("Generating interview flow...")
    interview_flow = processor.generate_interview_flow(
        role="Full Stack Developer",
        difficulty=DifficultyLevel.MEDIUM,
        num_questions=3
    )
    
    print(f"Generated {len(interview_flow)} questions for Full Stack Developer position")
    
    # Sample answers for demonstration
    sample_answers = [
        "I have 3 years of experience with React and Node.js. I've built several full-stack applications including an e-commerce platform and a project management tool. I'm comfortable with both frontend and backend development, and I enjoy working with modern technologies like TypeScript and GraphQL.",
        "In my previous project, I had to optimize database queries that were causing performance issues. I analyzed the slow queries using EXPLAIN, added proper indexes, and implemented query caching with Redis. This reduced page load times by 60% and improved user experience significantly.",
        "I believe in writing clean, maintainable code. I follow SOLID principles, write comprehensive tests, and use code reviews to ensure quality. I also document my code well and believe in continuous learning. I stay updated with the latest technologies through online courses and tech blogs."
    ]
    
    total_score = 0
    question_scores = []
    
    for i, question in enumerate(interview_flow):
        print(f"\n--- Question {i+1}: {question.question_type.value.upper()} ---")
        print(f"Question: {question.question_text}")
        
        if i < len(sample_answers):
            answer = sample_answers[i]
            print(f"Answer: {answer}")
            
            # Evaluate with both NLP and LLM
            nlp_result = nlp_processor.process_response(answer, question.question_text)
            llm_evaluation = processor.evaluate_answer(
                question=question.question_text,
                answer=answer,
                expected_keywords=question.expected_keywords
            )
            
            # Calculate combined score
            combined_score = (nlp_result['overall_score'] * 33.33) + (llm_evaluation.overall_score * 0.67)
            question_scores.append(combined_score)
            total_score += combined_score
            
            print(f"\nEvaluation Results:")
            print(f"  NLP Score: {nlp_result['overall_score']}/3")
            print(f"  LLM Score: {llm_evaluation.overall_score}/100")
            print(f"  Combined Score: {combined_score:.1f}/100")
            print(f"  Feedback: {llm_evaluation.feedback}")
            
            if llm_evaluation.strengths:
                print(f"  Strengths: {', '.join(llm_evaluation.strengths)}")
            if llm_evaluation.weaknesses:
                print(f"  Areas for Improvement: {', '.join(llm_evaluation.weaknesses)}")
        else:
            print("(No answer provided for this question)")
            question_scores.append(0)
    
    # Final results
    average_score = total_score / len(interview_flow)
    print(f"\n--- FINAL INTERVIEW RESULTS ---")
    print(f"Average Score: {average_score:.1f}/100")
    print(f"Individual Question Scores: {[f'{score:.1f}' for score in question_scores]}")
    
    if average_score >= 80:
        print("🎉 Excellent performance! Strong candidate.")
    elif average_score >= 60:
        print("👍 Good performance. Solid candidate with room for improvement.")
    elif average_score >= 40:
        print("⚠️  Average performance. Consider additional training or experience.")
    else:
        print("❌ Below expectations. Significant improvement needed.")

def main():
    """Run all demos"""
    print("🤖 AI-POWERED JOB INTERVIEW COACH")
    print("Step 2: LLM Module Demo")
    print("Dynamic Question Generation & Answer Evaluation")
    print("=" * 60)
    
    try:
        # Run all demo functions
        demo_question_generation()
        demo_answer_evaluation()
        demo_nlp_llm_integration()
        demo_interview_simulation()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("The LLM module is ready for integration with the main application.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
