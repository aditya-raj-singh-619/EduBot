import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

if not os.path.exists(persistent_directory):
    raise FileNotFoundError(
        f"No vector store found at {persistent_directory}. "
        f"Run the ingestion script first."
    )

# load the embedding model and the vector store
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

query = "What is the duration and eligibility for the A Level course?"

# Option A: plain similarity search
retriever = db.as_retriever(search_kwargs={"k": 4})

# Option B: restrict to a specific centre (uncomment to use)
# retriever = db.as_retriever(
#     search_kwargs={"k": 4, "filter": {"centre": "buxar"}}
# )

# Option C: similarity + score threshold (uncomment to use)
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.3},
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}\n")

if not relevant_docs:
    print("No relevant documents found. Try lowering the score threshold "
          "or check that the vector store was built correctly.")
else:
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("source", "unknown")
        centre = doc.metadata.get("centre", "unknown")
        print(f"Document {i}  |  centre: {centre}  |  source: {source}")
        print(doc.page_content)
        print("-" * 50)