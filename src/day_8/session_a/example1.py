import operator
import os
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode 
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from a .env file if present
load_dotenv()

# Set up the LLM
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_ENDPOINT"),
    temperature=0
)

# -------------------------------------------------------------------------
# 1. Designing Specialized "worker" Agents

# Worker 1: The Researcher Agent

class ResearcherState(TypedDict):
    topic: str
    research_data: str

def research_node(state: ResearcherState):
    topic = state['topic']
    # Simulate research by generating some facts about the topic
    research_data = f"Facts about {topic}: It is the largest planet in our solar system, a gas giant known for its Great Red Spot, a storm larger than Earth."
    return {"research_data": research_data}

researcher_graph = StateGraph(ResearcherState)
researcher_graph.add_node("research", research_node)
researcher_graph.set_entry_point("research")
researcher_graph.add_edge("research", END)
researcher_agent = researcher_graph.compile()

# Worker 2: The Writer Agent

class WriterState(TypedDict):
    research_data: str
    post: str

def write_node(state: WriterState):
    research_data = state['research_data']
    post = f"Blog Post: Jupiter, the Giant\n\nHere are some interesting facts about this incredible planet: {research_data}"
    return {"post": post}

writer_graph = StateGraph(WriterState)
writer_graph.add_node("write", write_node)
writer_graph.set_entry_point("write")
writer_graph.add_edge("write", END)
writer_agent = writer_graph.compile()

# -------------------------------------------------------------------------
# Packaging Agents as tools

@tool
def research_tool(topic: str) -> str:
    """
    A tool to perform research on a given topic.
    Returns the research data as a string.
    """
    output = researcher_agent.invoke({"topic": topic})
    return output['research_data']

@tool
def writer_tool(research_data: str) -> str:
    """
    A tool to write a post based on research data.
    Returns the final post as a string.
    """
    output = writer_agent.invoke({"research_data": research_data})
    return output['post']


# -------------------------------------------------------------------------
# The Supervisor Agent 

# the supervisor agent orchestrates the workflow
# by deciding when to call each worker tool
tools = [research_tool, writer_tool]

# Create a ToolNode directly from our list of tools.
tool_node = ToolNode(tools)

class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Bind the tools to the LLM so it knows what functions it can call
llm_with_tools = llm.bind_tools(tools)

def supervisor_node(state: SupervisorState):
    logging.info("Supervisor decides the next step..")
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}

def should_continue(state: SupervisorState):
    """Determines whether to continue or end the process."""
    if state['messages'][-1].tool_calls:
        return "continue"
    else:
        return "end"

# Define the supervisor's graph
supervisor_graph = StateGraph(SupervisorState)
supervisor_graph.add_node("supervisor", supervisor_node)
supervisor_graph.add_node("tools", tool_node)
supervisor_graph.set_entry_point("supervisor")
supervisor_graph.add_conditional_edges(
    "supervisor",
    should_continue,
    {"continue": "tools", "end": END}
)
supervisor_graph.add_edge("tools", "supervisor")

supervisor_agent = supervisor_graph.compile()

logging.info("Supervisor Agent created and ready...")

# --- Run the Supervisor ---
initial_prompt = "Your mission is to write a short post about a planet. First, you must research the planet Jupiter."
initial_messages = [HumanMessage(content=initial_prompt)]

for event in supervisor_agent.stream({"messages": initial_messages}):
    for key, value in event.items():
        logging.info(f"Node: {key}")   #, Value: {value}")

final_state = supervisor_agent.invoke({"messages": initial_messages})
logging.info("\nFinal Output:")
logging.info(final_state['messages'][-1].content)

