import os
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Define the State
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], lambda x, y: x + y]

# Define the Tools
tool = DuckDuckGoSearchRun()
tools = [tool]

# Create the Agent

# Set up the LLM
model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_ENDPOINT"),
    temperature=0
)

# Bind the tools to the model
model_with_tools = model.bind_tools(tools)

# Define the Graph Nodes

def agent_node(state: AgentState):
    """
    Invokes the model to decide whether to respond or to call a tool.
    """
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    """
    Checks the last message for tool calls and executes them.
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        print(f"Invoking Tool: {tool_call['name']}")
        result = tool.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    return {"messages": tool_messages}

def should_continue(state: AgentState):
    """
    Checks if the agent wants to call a tool or if it has finished.
    """
    if state["messages"][-1].tool_calls:
        return "continue"
    else:
        return "end"

# Build the Graph 
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("action", tool_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
graph_builder.add_edge("action", "agent")
graph = graph_builder.compile()

# Run the Graph
inputs = {"messages": [HumanMessage(content="What is the currency of Japan?")]}

for event in graph.stream(inputs, stream_mode="values"):
    event["messages"][-1].pretty_print()
