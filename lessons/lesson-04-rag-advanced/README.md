# Lesson 04: Advanced RAG

This folder contains the runnable code for the fourth article:

- Chunking with `CharacterTextSplitter` and `RecursiveCharacterTextSplitter`
- Splitting real `Document` objects while preserving metadata
- Retrieval tuning with similarity threshold and MMR
- A lightweight reranking stage after vector search
- Query rewriting for history-aware retrieval
- A production-style RAG pipeline backed by persistent Chroma storage

## Setup

This project follows the workspace rule that secrets stay outside the repository.
Store your OpenAI-compatible configuration in:

- `~/.config/codex/openai-gateway.env`

Expected variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `OPENAI_MODEL` (optional)
- `OPENAI_EMBEDDING_MODEL` (optional)
- `OPENAI_EMBEDDING_BASE_URL` (optional, overrides `OPENAI_BASE_URL` for embeddings)
- `OPENAI_EMBEDDING_API_KEY` (optional, overrides `OPENAI_API_KEY` for embeddings)

## Install

```bash
source .venv/bin/activate
pip install -r lessons/lesson-04-rag-advanced/requirements.txt
```

## Run

Run all examples:

```bash
python lessons/lesson-04-rag-advanced/main.py
```

Run a single example:

```bash
python lessons/lesson-04-rag-advanced/main.py --demo split
python lessons/lesson-04-rag-advanced/main.py --demo documents
python lessons/lesson-04-rag-advanced/main.py --demo retrieval
python lessons/lesson-04-rag-advanced/main.py --demo rerank
python lessons/lesson-04-rag-advanced/main.py --demo history
python lessons/lesson-04-rag-advanced/main.py --demo pipeline
```

## Notes

- The lesson writes its local Chroma files to `lessons/lesson-04-rag-advanced/runtime/`, which is ignored by Git.
- If the remote embeddings endpoint is unavailable, the script falls back to a deterministic local hash embedding so the retrieval demos still run.
- The `history` and `pipeline` demos call the chat model because they rewrite or answer questions with an LLM.
