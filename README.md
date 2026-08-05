# EduBot — Offline RAG-Powered Campus Query System

> Status: 🔄 In Progress | Retrieval Phase Complete

## What It Does
EduBot is an offline AI chatbot that automates admission, fee, and 
course-related queries for NIELIT Buxar using Retrieval-Augmented 
Generation (RAG) — with zero external API cost.

## Tech Stack
- LLM: Llama (via Ollama) — fully offline inference
- Vector Search: FAISS semantic embeddings
- API: FastAPI
- Data: 500+ PDF/CSV institutional documents

## Architecture
Ingestion → Preprocessing → Embedding (chromaDB) → Retrieval → LLM Response

## Results (So Far)
- ~70% better retrieval accuracy vs keyword search
- Sub-2 second target response latency
- 100% offline — no API costs

## Modules
| File | Purpose |
|------|---------|
| ingestion_pipeline.py | Document loading & preprocessing |
| _retrieval_pipeline.py | FAISS vector search |
| _generate_ans.py | LLM answer generation |
| _history.py | Conversation history management |
