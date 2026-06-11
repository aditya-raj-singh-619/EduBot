from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings 
from dotenv import load_dotenv

load_dotenv()

persistent_directory="db/chroma_db"

# load the embedding and the vectore store
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db=Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

query="which island does Spacex lease for its launches in the pecific?"  # you can ask queries here.

retriever=db.as_retriever(search_kwargs={"k":3})

"""retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":5,
        "score_threshold":0.3  # Only return chunks within the cosine similarity>=0.3
    }
)"""

relavent_docs=retriever.invoke(query)

print(f"User Query: {query}")

#display result

print("--context--")
for i, doc in enumerate(relavent_docs, 1):
    print(f"Document{i}:\n{doc.page_content}\n")
