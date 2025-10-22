import os
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import HttpClient
from neo4j import GraphDatabase

# ----------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------
load_dotenv()

# LiteLLM / Ollama
LITELLM_API = os.getenv("LITELLM_API")
LITELLM_MODEL = os.getenv("OPENAI_MODEL_NAME")

# Chroma
CHROMA_API = os.getenv("CHROMA_API")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ----------------------------------------------------
# 2️⃣ Initialize clients
# ----------------------------------------------------

# OpenAI-compatible client (LiteLLM)
client = OpenAI(base_url=LITELLM_API, api_key=os.getenv("OPENAI_API_KEY") )

# Chroma client
chroma_client = HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection("documents")

# Neo4j driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ----------------------------------------------------
# 3️⃣ Store data in Chroma (for retrieval)
# ----------------------------------------------------
docs = [
    "Ollama allows running large language models locally.",
    "Chroma is an open-source embedding database for vector search.",
    "Neo4j is a graph database optimized for connected data."
]

ids = ["doc1", "doc2", "doc3"]

collection.add(documents=docs, ids=ids)
print("✅ Documents added to Chroma.")

# ----------------------------------------------------
# 4️⃣ Query Chroma for most relevant context
# ----------------------------------------------------
query = "What is Chroma used for?"
results = collection.query(query_texts=[query], n_results=2)

context = " ".join(results["documents"][0])
print(f"🔍 Retrieved context:\n{context}\n")

# ----------------------------------------------------
# 5️⃣ Create a response using the local LLM (Ollama)
# ----------------------------------------------------
prompt = f"""
You are an assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""

response = client.chat.completions.create(
    model=LITELLM_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
)

answer = response.choices[0].message.content
print("💬 Model Answer:\n", answer)

# ----------------------------------------------------
# 6️⃣ Store relationships in Neo4j (optional)
# ----------------------------------------------------
cypher = """
MERGE (q:Query {text: $query})
MERGE (a:Answer {text: $answer})
MERGE (q)-[:HAS_ANSWER]->(a)
"""

with neo4j_driver.session() as session:
    session.run(cypher, parameters={"query": query, "answer": answer})
    
print("🔗 Stored query-answer relationship in Neo4j.")

