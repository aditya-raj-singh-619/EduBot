
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
 
load_dotenv()
 
persistent_directory = "db/chroma_db"
 
if not os.path.exists(persistent_directory):
    raise FileNotFoundError(
        f"No vector store found at {persistent_directory}. "
        f"Run the ingestion script first."
    )
 
# Load embeddings and vector store
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
 
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)
 
# LLM: low temperature for factual, non-improvised answers
model = ChatOllama(model="phi3", temperature=0.1)
 
SYSTEM_PROMPT = """You are the official assistant for NIELIT (National Institute
of Electronics and Information Technology). You answer questions about NIELIT
centres, courses, fees, rules, and governance using ONLY the context provided
to you for each query.
 
Rules:
- Use only the information in the provided context. Do not use outside knowledge.
- Each context chunk is labeled with the centre it belongs to (e.g. buxar, patna,
  hq_shared). If the user's question is about a specific centre, only use chunks
  from that centre or hq_shared. If chunks from multiple centres could apply,
  say so explicitly and distinguish them in your answer instead of merging them.
- If the answer is not in the provided context, say:
  "I don't have enough information to answer that based on the available documents."
- Be concise and precise. For factual details like durations, fees, hours, or
  eligibility, quote the exact figures from the context rather than paraphrasing
  numbers.
"""
 
 
def build_context_block(docs):
    """Format retrieved docs with their centre/source metadata so the model
    can tell which centre each piece of information applies to."""
    blocks = []
    for doc in docs:
        centre = doc.metadata.get("centre", "unknown")
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        blocks.append(
            f"[centre: {centre} | source: {source}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)
 
 
def answer_query(query, k=4, centre_filter=None):
    search_kwargs = {"k": k}
    if centre_filter:
        # restrict retrieval to one centre + shared HQ docs isn't natively
        # supported by a single filter dict in Chroma, so filter post-hoc
        # below if you need OR semantics. For a single centre:
        search_kwargs["filter"] = {"centre": centre_filter}
 
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    relevant_docs = retriever.invoke(query)
 
    print(f"User Query: {query}\n")
 
    if not relevant_docs:
        print("No relevant documents retrieved.")
        return "I don't have enough information to answer that based on the available documents."
 
    print("--- Retrieved Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        centre = doc.metadata.get("centre", "unknown")
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"Document {i}  |  centre: {centre}  |  source: {source}")
        print(doc.page_content)
        print("-" * 50)
 
    context_block = build_context_block(relevant_docs)
 
    user_message = f"""Question: {query}
 
Context:
{context_block}
 
Answer the question using only the context above. If relevant chunks come
from different centres, distinguish them clearly in your answer."""
 
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
 
    result = model.invoke(messages)
    return result.content
 
 
if __name__ == "__main__":
    query = "Who is the director of nielit patna?"
    answer = answer_query(query)
 
    print("\n--- Generated Response ---")
    print(answer)
 
    # Example: restrict to one centre
    # answer = answer_query("What is the contact number?", centre_filter="buxar")