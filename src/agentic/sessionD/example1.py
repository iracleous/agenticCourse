"""
Example 1:
Goal: Create an agent that can perform ReAct-style reasoning to answer complex questions.
"""
import dotenv as dt

from langchain.agents import initialize_agent, load_tools
from langchain_openai import AzureChatOpenAI
import os
dt.load_dotenv()

temperature  = 0
max_tokens = 150
llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    model_name = os.getenv("AZURE_OPENAI_MODEL"),
    max_tokens=max_tokens, 
    temperature=temperature
)


# ReAct-style Reasoning




tools = load_tools(["llm-math"], llm=llm)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

agent.run("If a train travels at 90 km/h for 2.5 hours, how far does it go?")
