# Lesson 03: RAG

This folder contains the runnable code for the third article:

- Manual context injection to show the core idea of RAG
- Embedding generation with `OpenAIEmbeddings`
- Semantic retrieval with FAISS
- A basic RAG chain built with LCEL
- Retriever tuning with top-k and MMR
- A combined RAG + memory example

## Setup

This project follows the workspace rule that secrets stay outside the repository.
Store your OpenAI-compatible configuration in:

- `~/.config/codex/openai-gateway.env`

Expected variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `OPENAI_MODEL` (optional)
- `OPENAI_EMBEDDING_MODEL` (optional)

## Install

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -r lessons/lesson-03-rag/requirements.txt
```

## Run

Run all examples:

```bash
python lessons/lesson-03-rag/main.py
```

Run a single example:

```bash
python lessons/lesson-03-rag/main.py --demo manual
python lessons/lesson-03-rag/main.py --demo embedding
python lessons/lesson-03-rag/main.py --demo retrieval
python lessons/lesson-03-rag/main.py --demo rag
python lessons/lesson-03-rag/main.py --demo tuned
python lessons/lesson-03-rag/main.py --demo rag-memory
```

## Notes

- The example stores vectors in memory with FAISS, so nothing is persisted after the script exits.
- The `rag-memory` demo uses a tiny in-memory knowledge base and in-memory chat history to mirror the article flow.
- The retrieval examples intentionally use small sample texts so the end-to-end pipeline is easy to inspect.
