"""
Example 2:   RunnableSequence (pipeline) 
Goal: Demonstrate a simple Runnable Sequential Pipeline using the modern LangChain interface
"""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# load environment variables from a .env file
load_dotenv()

# --- 1️⃣ Initialize the LLM ---

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_ENDPOINT"),
    temperature=0
)
# --- 2️⃣ Define a PromptTemplate ---
prompt = PromptTemplate.from_template(
    "Write a short, engaging LinkedIn post about {topic}."
)

# --- 3️⃣ Create a RunnableSequence (pipeline) ---
chain = prompt | llm

# --- 4️⃣ Invoke the chain with input ---
response = chain.invoke({"topic": "Artificial Intelligence in Education"})

# --- 5️⃣ Print the model output ---
print(response.content)
