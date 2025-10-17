import os
from langchain_huggingface import HuggingFaceEndpoint
from langgraph.graph import StateGraph
from typing import TypedDict

# Load environment variables (assuming HUGGINGFACEHUB_API_TOKEN is set)
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    question: str
    answer: str

def generate_answer(state: State) -> State:
    # Initialize HuggingFaceEndpoint with a hosted SLM (e.g., DistilGPT-2 or any inference model)
    llm = HuggingFaceEndpoint(
        repo_id="bert-base-uncased",  # Warm Models, 
        #Model that works with Inference API; replace with desired hosted SLM
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.7,
        max_new_tokens=100
    )
    
    try:
        # Generate answer using the LangChain wrapper
        response = llm.invoke(state["question"])
        return {"answer": response}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e) or 'No details'}"
        return {"answer": f"Error generating answer: {error_msg}"}

# Build the LangGraph workflow
graph = StateGraph(State)
graph.add_node("generate_answer", generate_answer)
graph.set_entry_point("generate_answer")
graph.set_finish_point("generate_answer")

# Compile the graph
app = graph.compile()

# Example usage
if __name__ == "__main__":
    # Test the LLM directly first
    llm = HuggingFaceEndpoint(
        repo_id="microsoft/DialoGPT-medium",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.7,
        max_new_tokens=100
    )
    try:
        test_response = llm.invoke("Hello")
        print(f"Direct LLM test: {test_response}")
    except Exception as e:
        print(f"Direct LLM error: {type(e).__name__}: {str(e) or 'No details'}")
        exit(1)
    
    initial_state = {"question": "What is the capital of France?", "answer": ""}
    result = app.invoke(initial_state)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")