# Exercise: Building a 3-Level Hierarchical Agent with LangGraph

**Objective:** To understand and implement a **three-level** hierarchical agent system. In this exercise, you will build a "Director" agent that delegates a research area to a "Manager" agent, which in turn delegates a specific, searchable question to a specialized "Worker" agent. Follow the OOP paradigm

---

### 🧠 Core Concepts

🏢 **Hierarchical Agents**: This architecture is like a corporate structure with a **Director**, a **Manager**, and a **Worker**.
* The **Director** receives a broad, high-level goal and breaks it down into a more focused sub-task for the Manager.
* The **Manager** takes this sub-task and refines it into a specific, actionable question for the Worker.
* The **Worker** is a specialist that executes the specific question using its tools (e.g., searching the web).

**Delegation**: Each higher level's job is not to do the work itself, but to call the correct subordinate with the right instructions.

**Separation of Concerns**: By separating high-level planning (Director), task refinement (Manager), and execution (Worker), we can build more robust, modular, and scalable agent systems.

---

### 🛠️ Prerequisites

* **Python Environment**: Python 3.8+
* **Libraries**: Install the required libraries.
* **API Keys**: You will need an OpenAI API key. Set it as an environment variable.

---

### 📝 Step-by-Step Implementation

#### Part 1: The Specialist "Worker" Agent (The Foundation)

Your first task is to build the specialist. This agent's only job is to receive a question and use DuckDuckGo to find the answer. This must be a recursive agent, meaning it can use its search tool multiple times if needed.

**📋 Your Task:**
Create a function `create_worker_graph()`. Inside this function, you need to:
1.  Define the agent's state, `WorkerState`, which should contain a list of messages.
2.  Instantiate the `DuckDuckGoSearchRun` tool.
3.  Bind the tool to a `ChatOpenAI` model.
4.  Define the graph nodes:
    * `agent_node`: This node invokes the model.
    * `tool_node`: This node executes the tool calls made by the agent.
5.  Define a conditional edge function, `should_continue`, to create the recursive loop. It should route to the `tool_node` if a tool is called, or to `END` if not.
6.  Build the `StateGraph` by adding the nodes and edges. The final edge should go from the tool node back to the agent node to create the loop.
7.  Compile and return the graph.

#### Part 2: The "Manager" Agent (The Middle Layer)

Next, you will build the manager. Its graph is linear. Its process is to receive a **sub-topic from the Director**, formulate a specific question for the Worker, and then call the Worker to get the result.

**📋 Your Task:**
Create a function `create_manager_graph(worker_graph)`. Inside this function:
1.  Define the manager's state, `ManagerState`, which needs to track the overall `sub_topic`, the `worker_result`, and the manager's internal `messages`.
2.  Define how the manager delegates tasks by creating a Pydantic `BaseModel` that represents a call to the worker. This will act as the manager's "tool."
3.  Bind this `DelegateToWorker` "tool" to a new `ChatOpenAI` model for the manager.
4.  Define the manager's graph nodes:
    * `manager_node`: This node takes the `sub_topic` from the state and uses the LLM to formulate the specific question for the Worker.
    * `worker_caller_node`: This node extracts the question from the manager's decision and invokes the `worker_graph` you passed into the function. It should then store the final answer in the `worker_result` field of the state.
5.  Build the simple, linear `StateGraph` for the manager: `START` -> `manager_node` -> `worker_caller_node` -> `END`.
6.  Compile and return the graph.

#### Part 3: The "Director" Agent (The Top Layer)

Now, you will build the Director. This is the entry point to our hierarchy. Its job is to take a high-level goal, break it down into a manageable sub-topic for the Manager, and then delegate the task.

**📋 Your Task:**
Create a function `create_director_graph(manager_graph)`. Inside this function:
1.  Define the director's state, `DirectorState`, which needs to track the high-level `goal`, the `manager_result`, and its internal `messages`.
2.  Create a Pydantic `BaseModel`, `DelegateToManager`, that represents a call to the manager agent. It should take a `sub_topic` string as a field.
3.  Bind this `DelegateToManager` "tool" to a new `ChatOpenAI` model for the Director.
4.  Define the director's graph nodes:
    * `director_node`: This node takes the overall `goal` from the state and uses the LLM to formulate a more focused `sub_topic` for the Manager.
    * `manager_caller_node`: This node extracts the `sub_topic` from the Director's decision and invokes the `manager_graph`. It should then store the final answer in the `manager_result` field of the state.
5.  Build the simple, linear `StateGraph` for the Director: `START` -> `director_node` -> `manager_caller_node` -> `END`.
6.  Compile and return the graph.

#### Part 4: Running the Full Hierarchy

Finally, instantiate all three agents, connecting them in order, and run the system to test your full implementation.

**📋 Code to Run:**
```python
# First, create the worker
search_worker = create_worker_graph()

# Then, create the manager, passing the worker to it
manager_agent = create_manager_graph(search_worker)

# Finally, create the director, passing the manager to it
director_agent = create_director_graph(manager_agent)

# Define the high-level goal for the director to handle
goal = "Write a report on current European economic affairs."

initial_state = {"goal": goal}

# Run the director and see the output
final_state = director_agent.invoke(initial_state)

print("\n--- Finished ---")
print("Director's final result:", final_state['manager_result'])