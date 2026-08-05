import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import shutil
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,       
    Docx2txtLoader,      
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# Helpers

def extract_centre_from_path(source_path, docs_root="docs"):
    """
    Infer which NIELIT centre a document belongs to from its folder.
    docs/buxar/about.txt        -> "buxar"
    docs/patna/courses.pdf      -> "patna"
    docs/_shared/nielit_about.txt -> "hq_shared"  (applies to ALL centres)
    """
    rel_path = os.path.relpath(source_path, docs_root)
    parts = rel_path.split(os.sep)
    folder = parts[0] if len(parts) > 1 else "hq_shared"
    if folder.startswith("_"):
        return "hq_shared"
    return folder.lower()


def clean_pdf_text(text):
    """
    Light cleanup for PyPDFLoader artifacts: fused words across line breaks
    (e.g. 'membership.The' -> 'membership. The'), collapsed whitespace, etc.
    Not perfect, but removes the worst of it.
    """
    # split "wordThe" -> "word The" (lowercase followed directly by uppercase)
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    # split "word.The" -> "word. The"
    text = re.sub(r'(?<=[.,;:])(?=[A-Za-z])', ' ', text)
    # collapse repeated whitespace/newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# Loading

def load_documents(docs_path="docs"):
    """Load all supported files (.txt, .md, .pdf, .docx) from docs/ recursively,
    tagging each with which NIELIT centre it belongs to."""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"the directory {docs_path} does not exist. "
            f"Please create it and add your NIELIT files."
        )

    documents = []

    # plain text and markdown files — use "**/" so subfolders (buxar/, patna/, ...) are included
    for pattern in ["*.txt", "*.md"]:
        loader = DirectoryLoader(
            path=docs_path,
            glob=f"**/{pattern}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()
        for d in docs:
            d.metadata["centre"] = extract_centre_from_path(d.metadata["source"], docs_path)
            d.metadata["filetype"] = "text"
        print(f"  Loaded {len(docs)} file(s) matching {pattern}")
        documents.extend(docs)

    # pdf files
    pdf_loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    pdf_docs = pdf_loader.load()
    for d in pdf_docs:
        d.page_content = clean_pdf_text(d.page_content)
        d.metadata["centre"] = extract_centre_from_path(d.metadata["source"], docs_path)
        d.metadata["filetype"] = "pdf"
    print(f"  Loaded {len(pdf_docs)} page(s) from .pdf files")
    documents.extend(pdf_docs)

    # docx files
    docx_loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
    )
    docx_docs = docx_loader.load()
    for d in docx_docs:
        d.metadata["centre"] = extract_centre_from_path(d.metadata["source"], docs_path)
        d.metadata["filetype"] = "docx"
    print(f"  Loaded {len(docx_docs)} file(s) matching *.docx")
    documents.extend(docx_docs)

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No supported files found in {docs_path}. Add .txt, .md, .pdf, or .docx files."
        )

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocuments {i+1}:")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Centre: {doc.metadata.get('centre')}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print("  Content preview:", doc.page_content[:100].encode("utf-8", "ignore").decode("utf-8"))
        print(f"  metadata: {doc.metadata}")

    return documents


# Chunking

def split_documents(documents, chunk_size=1200, chunk_overlap=200):
    """
    Split documents into chunks, preferring to break on markdown headers /
    paragraphs / sentences before falling back to raw characters. Larger
    chunk_size than before (1200 vs 500) so tables and numbered clause lists
    (e.g. NIELIT Rules 8.1-8.16) don't get severed mid-structure.
    """
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata.get('source')}")
            print(f"Centre: {chunk.metadata.get('centre')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("_" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


# Embedding / vector store

def create_vector_store(chunks, persist_directory="db/chroma_db", rebuild=True):
    """
    Create/persist a Chroma vector store.
    rebuild=True wipes any existing DB first so re-running this script after
    adding/editing files in docs/ doesn't create duplicate chunks.
    """
    print("Creating embeddings and storing in ChromaDB...")

    if rebuild and os.path.exists(persist_directory):
        print(f"  Removing existing vector store at {persist_directory} (rebuild=True)")
        shutil.rmtree(persist_directory)

    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )

    print("---Creating vector database---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("---Finished creating vector store---")
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Main function")

    documents = load_documents(docs_path="DATABASE")
    print(f"\nTotal documents loaded: {len(documents)}")

    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()
