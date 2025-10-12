#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ✅ Set your API key here
import os
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # replace with your key

import json
import logging
from typing import Dict
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

# ✅ Import functions from your notebook file (make sure LLMs_test.py exists in same dir)
from LLMs_test import generate_question, evaluate_answer


# In[2]:


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------------------------------------
# 🧠 1. Question Generation Chain
# -------------------------------------------------------
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

    # ✅ Use ChatOpenAI (new API)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # ✅ Use new pipe syntax
    chain = prompt | llm
    return chain


# -------------------------------------------------------
# 🧩 2. Answer Evaluation Chain
# -------------------------------------------------------
def get_answer_evaluation_chain():
    """
    Wraps your evaluate_answer() function into a LangChain-like callable.
    """

    def _run(question: str, candidate_answer: str) -> Dict:
        # Directly call your notebook function
        result = evaluate_answer(question, candidate_answer)
        return result

    # ✅ Simple chain wrapper
    class EvalChain:
        def __init__(self):
            self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        def __call__(self, question: str, candidate_answer: str):
            result = _run(question, candidate_answer)
            self.memory.chat_memory.add_user_message(candidate_answer)
            self.memory.chat_memory.add_ai_message(result["evaluation"]["feedback"])
            return result

    return EvalChain()


# -------------------------------------------------------
# 💬 3. Example Usage (Testing)
# -------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Step 3 — LangChain Integration Demo (Latest API Version)\n")

    # 1️⃣ Create chains
    q_chain = get_question_generation_chain()
    eval_chain = get_answer_evaluation_chain()

    # 2️⃣ Generate a question
    inputs = {"round_type": "HR", "context": "Candidate is a software engineer with 2 years of experience"}
    question_output = q_chain.invoke(inputs)
    generated_question = question_output.content  # get text output from ChatMessage

    print("\n🧠 Generated Question:")
    print(generated_question)

    # 3️⃣ Candidate answer
    candidate_ans = "I once resolved a conflict by organizing a team meeting and discussing responsibilities openly."

    # 4️⃣ Evaluate answer
    evaluation_result = eval_chain(generated_question, candidate_ans)
    print("\n✅ Evaluation Result:")
    print(json.dumps(evaluation_result, indent=2))


# In[ ]:




