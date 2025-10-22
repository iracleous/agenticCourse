"""
Ex5. Goal-Driven Agent with Memory and Tools
Latest LangChain version compatible
"""

"""
Ex5. Goal-Driven Agent with Memory and Tools
Compatible with LangChain versions where `create_react_agent` requires a `prompt`.
"""

import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ----------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------
load_dotenv()
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

# ----------------------------------------------------
# 2️⃣ Initialize the LLM
# ----------------------------------------------------
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_ENDPOINT,
    model=OPENAI_MODEL_NAME,
    temperature=0.2
)

# ----------------------------------------------------
# 3️⃣ Define tools
# ----------------------------------------------------
@tool
def get_time() -> str:
    """Returns the current local time."""
    return f"🕒 The current time is {datetime.now().strftime('%H:%M:%S')}."

@tool
def get_weather(location: str) -> str:
    """Returns fake weather info for a given location."""
    return f"The weather in {location} is sunny and 25°C."

tools = [get_time, get_weather]

# ----------------------------------------------------
# 4️⃣ Add conversational memory
# ----------------------------------------------------
memory = ConversationBufferMemory(memory_key="chat_history")

# ----------------------------------------------------
# 5️⃣ Define a compatible ReAct prompt
# ----------------------------------------------------
prompt = PromptTemplate(
    input_variables=["input", "chat_history", "tools", "tool_names", "agent_scratchpad"],
    template="""
You are a helpful assistant that can use tools.

Chat History:
{chat_history}

Available Tools:
{tools}

Tool Names:
{tool_names}

User Input:
{input}

Agent Scratchpad:
{agent_scratchpad}

Respond appropriately and use tools if needed.
"""
)

# ----------------------------------------------------
# 6️⃣ Create the agent
# ----------------------------------------------------
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# ----------------------------------------------------
# 7️⃣ Run interactively
# ----------------------------------------------------
if __name__ == "__main__":
    print("🧠 Goal-Driven Agent with Memory and Tools (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        result = executor.invoke({"input": user_input})
        print("Agent:", result["output"], "\n")
