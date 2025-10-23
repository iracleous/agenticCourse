"""
Ex1. Simple LangChain pipeline
Goal: Demonstrate a simple Runnable Sequential Pipeline  (RunnableSequence)
using the modern LangChain interface
"""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


################################################
import time
start = time.time()
################################################

# load environment variables from a .env file
load_dotenv()

# --- 1️⃣ Initialize the LLM ---

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_AZ"),
    api_key=os.getenv("OPENAI_API_KEY_AZ"),
  #  base_url=os.getenv("OPENAI_ENDPOINT"),
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


################################################
end = time.time()
print(f"Execution time: {end - start:.4f} seconds") 
################################################