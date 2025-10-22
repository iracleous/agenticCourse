"""
Ex4. Reactive Agent with Tools using LangChain Core
Goal: Demonstrate how to use @tool to give the LLM special abilities
"""

import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.tools import tool
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI

# ----------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------
load_dotenv()

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

# ----------------------------------------------------
# 2️⃣ Define tools using @tool decorator
# ----------------------------------------------------

@tool
def get_time() -> str:
    """Returns the current local time."""
    return f"🕒 The current time is {datetime.now().strftime('%H:%M:%S')}."

@tool
def get_weather() -> str:
    """Returns fake weather info."""
    return "☀️ It’s sunny and 25°C."

@tool
def greet(name: str) -> str:
    """Greets a person by name."""
    return f"👋 Hello, {name}! How are you today?"

tools = [get_time, get_weather, greet]

# ----------------------------------------------------
# 3️⃣ Initialize LLM (local or cloud endpoint)
# ----------------------------------------------------
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_ENDPOINT,
    model=OPENAI_MODEL_NAME,
    temperature=0
)

# ----------------------------------------------------
# 4️⃣ Create a ReAct-style agent that can use tools
# ----------------------------------------------------
prompt = """
You are a helpful assistant that can use tools when needed.
If a question requires a tool, decide which one to use and explain your reasoning.
Otherwise, respond directly.
"""

agent = create_react_agent(llm, tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------------------------------------------
# 5️⃣ Run interactively
# ----------------------------------------------------
if __name__ == "__main__":
    print("🤖 Reactive Tool Agent ready (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        result = executor.invoke({"input": user_input})
        print("Agent:", result["output"], "\n")

