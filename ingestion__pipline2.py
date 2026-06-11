import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader  # this will helps us to read text files, ppt or docs file from particular directory
from langchain_text_splitters import RecursiveCharacterTextSplitter           # this is for chunking the documents
from langchain_ollama import OllamaEmbeddings                                 # this is for embedding models         
from langchain_chroma import Chroma                                           # this work as vector database
from dotenv import load_dotenv


load_dotenv()



def load_documents(docs_path="docs"):
    """load all text files from the dcs directory"""
    print(f"Loading documents from {docs_path}...")

    # check if directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"the directory {docs_path} does not exists. Please cretae it and add your company files.")
    
    # load all the .txt files fom the docs dir4ectory
    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents=loader.load()

    if len(documents)==0:
        raise FileNotFoundError(f"No .txt fils found in {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocuments {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print("  Content preview:", doc.page_content[:100].encode("utf-8", "ignore").decode("utf-8"))
        print(f"  metadata:{doc.metadata}")
    return documents

# chunking
def split_documents(documents, chunk_size=500, chunk_overlap=0):
    """Split documents into small size chunks without overlap"""
    print("splitting documents into chunks...")

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks=text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk{i+1}---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"content:")
            print(chunk.page_content)
            print("_" * 50)
        
        if len(chunks)>5:
            print(f"\n... and {len(chunks) -5} more chunks")

    return chunks

# creating embedding
def create_vector_store(chunks, persists_directory="db/chroma_db"):
    """create persist chroma db vector store"""
    print("Creating Embedding and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    #create ChromaDB vectore
    print("---Creating vectore database---")
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persists_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("---Finished creating vector store---")

    print(f"Vector store created and saved to {persists_directory}")
    return vectorstore


def main():
    print("Main function")
    
    # loading the file
    documents=load_documents(docs_path="docs")

    # Chunking the file
    chunks=split_documents(documents)

    # embedding and storing in the vectore database
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()