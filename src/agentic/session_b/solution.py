import os
from typing import List, TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

from dotenv import load_dotenv
load_dotenv()  

# ------------------------------------------------------------------------------
# Part 1: The Specialist "Worker" Agent (The Foundation)


class WorkerState(TypedDict):
    """
    The state for the worker agent, containing the question and the conversation history.
    """
    messages: Annotated[List[AnyMessage], lambda x, y: x + y]

class Worker:
    """
    A recursive agent that uses DuckDuckGo to answer a specific question.
    """
    def __init__(self):
        self.search_tool = DuckDuckGoSearchRun()
        self.model = llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_ENDPOINT"),
            temperature=0
        )

    def agent_node(self, state: WorkerState):
        response = self.model.invoke(state["messages"])
        return {"messages": [response]}

    def tool_node(self, state: WorkerState):
        tool_calls = state["messages"][-1].tool_calls
        tool_outputs = []
        for tool_call in tool_calls:
            output = self.search_tool.invoke(tool_call["args"])
            tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
        return {"messages": tool_outputs}

    def should_continue(self, state: WorkerState):
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END

def create_worker_graph():
    """Factory function to create the worker agent graph."""
    worker = Worker()
    graph = StateGraph(WorkerState)
    graph.add_node("agent_node", worker.agent_node)
    graph.add_node("tool_node", worker.tool_node)
    graph.add_conditional_edges("agent_node", worker.should_continue)
    graph.add_edge("tool_node", "agent_node")
    graph.set_entry_point("agent_node")
    return graph.compile()

# ------------------------------------------------------------------------------
# Part 2: The "Manager" Agent (The Middle Layer)

class DelegateToWorker(BaseModel):
    """The tool schema for the manager to delegate a task to the worker."""
    question: str = Field(description="A specific, answerable question for the worker to research.")
    
class ManagerState(TypedDict):
    """
    The state for the manager agent.
    """
    sub_topic: str
    worker_result: str
    messages: Annotated[List[AnyMessage], lambda x, y: x + y]

class Manager:
    """
    A linear agent that refines a sub-topic into a question for the worker.
    """
    def __init__(self, worker_graph):
        self.worker_graph = worker_graph
        self.model = llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_ENDPOINT"),
            temperature=0
        ).with_structured_output(DelegateToWorker)

    def manager_node(self, state: ManagerState):
        prompt = (
            f"You are a research manager. Your goal is to formulate a single, specific, and searchable question "
            f"for your worker to answer based on the following sub-topic. The question should be self-contained.\n\n"
            f"Sub-topic: {state['sub_topic']}"
        )
        response = self.model.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}
        
    def worker_caller_node(self, state: ManagerState):
        manager_tool_call = state["messages"][-1]
        question = manager_tool_call.question
        
        # Invoke the worker graph with the specific question
        worker_output = self.worker_graph.invoke({
            "messages": [HumanMessage(content=question)]
        })
        
        final_answer = worker_output["messages"][-1].content
        return {"worker_result": final_answer}

def create_manager_graph(worker_graph):
    """Factory function to create the manager agent graph."""
    manager = Manager(worker_graph)
    graph = StateGraph(ManagerState)
    graph.add_node("manager_node", manager.manager_node)
    graph.add_node("worker_caller_node", manager.worker_caller_node)
    graph.add_edge("manager_node", "worker_caller_node")
    graph.add_edge("worker_caller_node", END)
    graph.set_entry_point("manager_node")
    return graph.compile()

# ------------------------------------------------------------------------------
# Part 3: The "Director" Agent (The Top Layer)


class DelegateToManager(BaseModel):
    """The tool schema for the director to delegate a sub-topic to the manager."""
    sub_topic: str = Field(description="A specific, focused sub-topic for the manager to handle.")

class DirectorState(TypedDict):
    """
    The state for the director agent.
    """
    goal: str
    manager_result: str
    messages: Annotated[List[AnyMessage], lambda x, y: x + y]

class Director:
    """
    A linear agent that breaks down a goal into a sub-topic for the manager.
    """
    def __init__(self, manager_graph):
        self.manager_graph = manager_graph
        self.model = llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_ENDPOINT"),
            temperature=0
        ).with_structured_output(DelegateToManager)

    def director_node(self, state: DirectorState):
        prompt = (
            f"You are a research director. Your task is to break down a high-level goal into a more focused "
            f"sub-topic for your manager. The sub-topic should be a clear and manageable area of research.\n\n"
            f"High-level goal: {state['goal']}"
        )
        response = self.model.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}
        
    def manager_caller_node(self, state: DirectorState):
        director_tool_call = state["messages"][-1]
        sub_topic = director_tool_call.sub_topic

        # Invoke the manager graph with the sub-topic
        manager_output = self.manager_graph.invoke({"sub_topic": sub_topic})
        return {"manager_result": manager_output["worker_result"]}

def create_director_graph(manager_graph):
    """Factory function to create the director agent graph."""
    director = Director(manager_graph)
    graph = StateGraph(DirectorState)
    graph.add_node("director_node", director.director_node)
    graph.add_node("manager_caller_node", director.manager_caller_node)
    graph.add_edge("director_node", "manager_caller_node")
    graph.add_edge("manager_caller_node", END)
    graph.set_entry_point("director_node")
    return graph.compile()


# ------------------------------------------------------------------------------
# Running the Full Hierarchy

print("Instantiating the 3-level agent hierarchy...")

# Create the worker
search_worker = create_worker_graph()

# Create the manager, passing the worker to it
manager_agent = create_manager_graph(search_worker)

# Create the director, passing the manager to it
director_agent = create_director_graph(manager_agent)

# Define the high-level goal for the director to handle
goal = "Write a report on the current state of the Greek economy, focusing on its main industries and recent growth trends."
initial_state = {"goal": goal}

print(f"Starting the research process for goal: '{goal}'")

# Run the director and see the output
# The 'recursion_limit' is set to handle the potential chain of calls.
final_state = director_agent.invoke(initial_state, {"recursion_limit": 100})


print("Director's Final Result:")
print(final_state['manager_result'])