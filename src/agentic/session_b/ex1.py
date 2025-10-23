"""
ex1.py

GDPR Document Question-Answering System
#1. Preprocess the GDPR Document
#2. Create Embeddings
#3. Create a Vector Store
#4. Build a Question-Answering Chain
"""



from langchain_community.document_loaders import PyPDFLoader  # or TextLoader if you have .txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA



load_dotenv()

# Load GDPR PDF or text
loader = PyPDFLoader("docs/CELEX_32016R0679_EN_TXT.pdf")  # or TextLoader("gdpr.txt")
documents = loader.load()

# Chunk into 500-character segments with overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20
)
chunks = splitter.split_documents(documents)
# ✅ Extract just the text for embedding
texts = [chunk.page_content for chunk in chunks]

# for chunk in chunks:
#     print(chunk)

#2 Create Embeddings
###   Choose a Local Embedding Model
#sentence-transformers  paraphrase-Multilingual-MiniLM-L12-v2

# 3️⃣ Create embeddings (Hugging Face)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_vectors = HuggingFaceEmbeddings(
    model_name=embedding_model_name)

# Compute embeddings

#3. Create a Vector Store
 

# 4️⃣ Store chunks in Chroma vector DB
persist_directory = "chroma_db"
vector_store = Chroma.from_texts(
    texts=texts,  # ✅ must be list of strings
    embedding=embedding_vectors,
    collection_name="law_gdpr",
    persist_directory=persist_directory
)


#4. Build a Question-Answering Chain

 

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_ENDPOINT"),
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Ask a GDPR question
query = "What are the rights of a data subject?"


query = "What are the main principles of data processing under GDPR?"
query = "What are the conditions for lawful processing of personal data under GDPR?"
result = qa_chain.invoke({"query": query})

print(result["result"])


#5. (Optional) Use LangGraph for Multi-Step Workflows
# If your use case involves:
# Compliance routing
# Multiple legal modules
# Decision-making logic
# ...then you can use LangGraph to define a stateful graph with LLMs and retrievers. 


#6. Deploy (Optional)
# Expose as an Azure Function (API endpoint)
# Frontend via React or PowerApps
# Use Azure Cognitive Search if you need enterprise-grade search and security

 