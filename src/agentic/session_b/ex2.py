"""
ex2.py
Learn from past actions and feedback to improve over time.

learning (adaptive) agent, i.e., one that can reflect on past actions and feedback
 to improve its behavior over time.
#1. Define an Agent with Memory
#2. Implement Learning from Feedback


"""

"""
Goal: Demonstrate a Goal-Driven Reflective Agent
that learns from user feedback using an LLM and memory.

Features:
- Persistent memory using ConversationBufferMemory
- Feedback-based self-improvement
- OpenAI (or Azure OpenAI) model backend
"""
"""
Ex: Reflective Agent using LangChain Expression Language (LCEL)

Goal:
A goal-driven agent that answers user queries,
learns from feedback, and improves over time.

Features:
- Uses LCEL (RunnableSequence & RunnableMap)
- Maintains conversational memory
- Reflects on bad feedback using LLM
"""

"""
Ex: Reflective Agent using LangChain Expression Language (LCEL, v0.3+)

Goal:
A goal-driven agent that answers user queries,
learns from feedback, and improves over time.

Features:
- Uses LCEL composition (RunnableMap + pipe operators)
- Maintains conversational memory
- Reflects on bad feedback using an LLM
"""

"""
Reflective Agent using LangChain Expression Language (LCEL, 2025+)

Goal:
A goal-driven agent that answers GDPR-related questions,
learns from feedback, and improves over time.

Key Concepts:
- Uses LCEL pipes (`|`) for chaining
- Maintains in-memory conversation
- Reflects on user feedback to improve answers
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv
import os
import time

# ----------------------------------------------------
# 1️⃣ Setup
# ----------------------------------------------------
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_ENDPOINT"),
    temperature=0.6,
)

# Persistent conversational memory (in RAM)
memory = ConversationBufferMemory(return_messages=True)

# ----------------------------------------------------
# 2️⃣ Define prompts
# ----------------------------------------------------
qa_prompt = ChatPromptTemplate.from_template("""
You are a GDPR expert assistant.
Use the conversation so far to give a concise, accurate answer.

Chat History:
{history}

User: {question}
AI:""")

reflect_prompt = ChatPromptTemplate.from_template("""
You previously gave this answer:

"{answer}"

The user said it was not helpful.
Please reflect briefly on why and how to improve next time.
""")

# ----------------------------------------------------
# 3️⃣ Define LCEL components
# ----------------------------------------------------

# Prepare input: combine current question + chat history
prepare_input = RunnableLambda(
    lambda x: {
        "history": memory.load_memory_variables({}).get("history", ""),
        "question": x["user_input"],
    }
)

# QA reasoning: prompt → llm
qa_chain = qa_prompt | llm

# Function to save exchange into memory
def save_to_memory(data):
    memory.save_context(
        {"user": data["question"]},
        {"ai": data["response"]}
    )
    return data  # Pass downstream unchanged

save_step = RunnableLambda(save_to_memory)

# Reflection pipeline
reflect_chain = reflect_prompt | llm

# ----------------------------------------------------
# 4️⃣ Compose the main LCEL agent
# ----------------------------------------------------
agent = (
    prepare_input
    | RunnableLambda(lambda x: {
        "response": qa_chain.invoke(x).content,
        "question": x["question"]
      })
    | save_step
)

# ----------------------------------------------------
# 5️⃣ Interactive reflection loop
# ----------------------------------------------------
print("🧠 Reflective Agent (LCEL) ready! Ask GDPR-related questions. Type 'exit' to quit.\n")

while True:
    user_query = input("You: ").strip()
    if user_query.lower() == "exit":
        print("Goodbye!")
        break

    start = time.time()
    result = agent.invoke({"user_input": user_query})
    response = result["response"]

    print(f"\nAgent: {response}")
    print(f"(Response time: {time.time() - start:.2f}s)")

    feedback = input("Feedback (good/bad/exit): ").strip().lower()

    if feedback == "good":
        memory.save_context(
            {"user": "Feedback"},
            {"ai": "User found the answer helpful."}
        )

    elif feedback == "bad":
        reflection = reflect_chain.invoke({"answer": response})
        print(f"\n🪞 Reflection: {reflection.content}")
        memory.save_context(
            {"user": "Reflection"},
            {"ai": reflection.content}
        )

    elif feedback == "exit":
        break
