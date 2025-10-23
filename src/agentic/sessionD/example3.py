"""
Example 3:
Goal: agent loop rewritten using LangGraph
"""


# =====================================================
# Agentic AI Example – Plan, Act, Reflect Loop
# Using LangGraph + LangChain + Azure OpenAI
# =====================================================

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing import TypedDict, Literal

import os
import dotenv as dt
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

# --- 2️⃣ Simulated Tools ---

def get_weather(city):
    """Simulated API call for weather."""
    print(f"🔍 Checking weather for {city}...")
    weather_data = {"Paris": {"temp": 19, "condition": "Cloudy"}}
    w = weather_data.get(city, {"temp": "unknown", "condition": "unknown"})
    return f"The weather in {city} is {w['condition']} with {w['temp']}°C."

def send_email(recipient, subject, message):
    """Simulated email sending."""
    print(f"📧 Sending email to {recipient} with subject '{subject}'...")
    print(f"Message:\n{message}")
    return "Email sent successfully."


# --- 3️⃣ Define Prompts ---

plan_prompt = ChatPromptTemplate.from_template("""
You are an autonomous AI assistant. 
Goal: {goal}
Available tools:
- get_weather(city)
- send_email(recipient, subject, message)

Make a short numbered plan using the tools above.
Return the plan in plain text.
""")

reflect_prompt = ChatPromptTemplate.from_template("""
You executed this plan and got the following result:
{result}

Did you achieve the goal: "{goal}"?
Answer only YES or NO and a short reasoning.
""")


# --- 4️⃣ Define Graph State ---
class AgentState(TypedDict):
    goal: str
    plan: str
    result: str
    reflection: str
    status: Literal["planning", "executing", "reflecting", "done"]


# --- 5️⃣ Define Graph Nodes ---

def plan_node(state: AgentState):
    """LLM creates a plan."""
    plan = llm.invoke(plan_prompt.format(goal=state["goal"])).content
    print("\n🧠 Plan:\n", plan)
    return {**state, "plan": plan, "status": "executing"}


def execute_node(state: AgentState):
    """Execute plan by detecting tool calls."""
    plan = state["plan"]
    result = ""

    if "get_weather" in plan:
        weather_info = get_weather("Paris")
        result += weather_info + "\n"

    if "send_email" in plan:
        result += send_email("user@example.com", "Weather Report", weather_info)

    print("\n⚙️ Result:\n", result)
    return {**state, "result": result, "status": "reflecting"}


def reflect_node(state: AgentState):
    """LLM evaluates if the goal is met."""
    reflection = llm.invoke(reflect_prompt.format(goal=state["goal"], result=state["result"])).content
    print("\n💭 Reflection:\n", reflection)

    done = "YES" in reflection.upper()
    return {
        **state,
        "reflection": reflection,
        "status": "done" if done else "planning"
    }


# --- 6️⃣ Build Graph ---

graph = StateGraph(AgentState)

graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)
graph.add_node("reflect", reflect_node)

graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", "reflect")

# Loop until goal is complete
graph.add_conditional_edges(
    "reflect",
    lambda s: s["status"],
    {
        "planning": "plan",
        "done": END
    }
)

compiled_graph: CompiledStateGraph = graph.compile()


# --- 7️⃣ Run the Agent ---
initial_state: AgentState = {
    "goal": "Research current weather in Paris and send an email report to user@example.com",
    "plan": "",
    "result": "",
    "reflection": "",
    "status": "planning"
}

print("🚀 Starting Agentic Loop...\n")
compiled_graph.invoke(initial_state)
print("\n✅ Agent finished successfully!")
