from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    FAISS = None


CONFIG_PATH = Path.home() / ".config" / "code" / "openai-gateway.env"
KNOWLEDGE_BASE_TEXTS = [
    "LangChain is a framework for building LLM-powered applications.",
    "RAG helps models answer questions with external knowledge.",
    "Memory manages chat history so a bot can remember prior turns.",
    "Embeddings convert text into vectors for semantic search.",
    "FAISS is a fast vector similarity library released by Meta.",
]
COMPANY_OKR_TEXT = """
Company Q3 OKRs:
1. Improve user retention by 20%.
2. Launch a new AI agent product.
3. Complete the data platform migration.
""".strip()


class LocalHashEmbeddings(Embeddings):
    """Small deterministic fallback embeddings for OpenAI-incompatible gateways."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokenize(text)

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class InMemoryRetriever:
    """Simple retriever used when FAISS is unavailable in the local environment."""

    def __init__(
        self,
        texts: list[str],
        embeddings: Embeddings,
        *,
        k: int = 4,
        search_type: str = "similarity",
    ) -> None:
        self.documents = [Document(page_content=text) for text in texts]
        self.embeddings = embeddings
        self.k = k
        self.search_type = search_type
        self.document_vectors = embeddings.embed_documents(texts)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    def invoke(self, query: str) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        scored_documents = sorted(
            (
                (self._cosine_similarity(query_vector, doc_vector), document)
                for document, doc_vector in zip(self.documents, self.document_vectors)
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        if self.search_type == "mmr":
            return [document for _, document in scored_documents[: self.k]]

        return [document for _, document in scored_documents[: self.k]]


def build_retriever(
    texts: list[str],
    embeddings: Embeddings,
    *,
    k: int = 4,
    search_type: str = "similarity",
):
    if FAISS is not None:
        try:
            vectorstore = FAISS.from_texts(texts, embedding=embeddings)
            return vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k},
            )
        except Exception:
            pass

    print(
        "Warning: FAISS is unavailable; using the built-in in-memory retriever instead.",
        file=sys.stderr,
    )
    return InMemoryRetriever(texts, embeddings, k=k, search_type=search_type)


def load_runtime_env() -> None:
    """Load runtime configuration from the user-level Codex config directory."""
    if CONFIG_PATH.exists():
        load_dotenv(CONFIG_PATH, override=False)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to ~/.config/code/openai-gateway.env."
        )


def create_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    kwargs = {
        "model": model,
        "timeout": 30,
        "max_retries": 1,
    }

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def create_embeddings() -> Embeddings:
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    kwargs = {
        "model": model,
        "timeout": 30,
        "max_retries": 1,
    }

    base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    api_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key

    remote_embeddings = OpenAIEmbeddings(**kwargs)

    try:
        remote_embeddings.embed_query("health check")
        return remote_embeddings
    except Exception as exc:
        print(
            "Warning: remote embeddings are unavailable; using local hash embeddings "
            f"instead. Reason: {exc}",
            file=sys.stderr,
        )
        return LocalHashEmbeddings()


def print_section(title: str, content: str) -> None:
    print(f"[{title}]")
    print(content)
    print()


def format_documents(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_company_retriever(embeddings: Embeddings):
    return build_retriever([COMPANY_OKR_TEXT], embeddings, k=1)


def run_manual_context_demo(llm: ChatOpenAI) -> str:
    question = "What are our company's Q3 OKRs?"
    prompt = f"""
Answer the question using the context below.

Context:
{COMPANY_OKR_TEXT}

Question: {question}
""".strip()
    response = llm.invoke(prompt)
    return response.content


def run_embedding_demo(embeddings: Embeddings) -> str:
    vector = embeddings.embed_query("What is an AI agent?")
    preview = ", ".join(f"{value:.4f}" for value in vector[:5])
    backend = type(embeddings).__name__
    return (
        f"Embedding backend: {backend}\n"
        f"Vector length: {len(vector)}\n"
        f"First 5 values: [{preview}]"
    )


def run_retrieval_demo(embeddings: Embeddings) -> str:
    retriever = build_retriever(KNOWLEDGE_BASE_TEXTS, embeddings)
    results = retriever.invoke("What is RAG?")
    return "\n".join(f"- {doc.page_content}" for doc in results)


def build_rag_chain(llm: ChatOpenAI, retriever):
    prompt = ChatPromptTemplate.from_template(
        """
Use the retrieved context to answer the question.
If the answer is not supported by the context, say "I could not find relevant information."

Context:
{context}

Question: {question}
""".strip()
    )

    return (
        {
            "context": RunnableLambda(retriever.invoke) | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )


def run_rag_demo(llm: ChatOpenAI, embeddings: Embeddings) -> str:
    retriever = build_retriever(KNOWLEDGE_BASE_TEXTS, embeddings)
    rag_chain = build_rag_chain(llm, retriever)
    response = rag_chain.invoke("What is RAG?")
    return response.content


def run_tuned_retrieval_demo(embeddings: Embeddings) -> str:
    similarity_docs = build_retriever(KNOWLEDGE_BASE_TEXTS, embeddings, k=2).invoke("What is RAG?")
    mmr_docs = build_retriever(
        KNOWLEDGE_BASE_TEXTS,
        embeddings,
        k=3,
        search_type="mmr",
    ).invoke("What is RAG?")

    similarity_text = "\n".join(f"- {doc.page_content}" for doc in similarity_docs)
    mmr_text = "\n".join(f"- {doc.page_content}" for doc in mmr_docs)
    return f"Top-2 similarity search:\n{similarity_text}\n\nTop-3 MMR search:\n{mmr_text}"


def build_rag_chain_with_memory(llm: ChatOpenAI, retriever):
    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer only from the retrieved context when possible.",
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Retrieved context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    chain = (
        {
            "context": RunnableLambda(lambda data: data["question"])
            | RunnableLambda(retriever.invoke)
            | RunnableLambda(format_documents),
            "question": RunnableLambda(lambda data: data["question"]),
            "history": RunnableLambda(lambda data: data["history"]),
        }
        | prompt
        | llm
    )

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_memory


def run_rag_memory_demo(llm: ChatOpenAI, embeddings: Embeddings) -> str:
    retriever = build_company_retriever(embeddings)
    chain_with_memory = build_rag_chain_with_memory(llm, retriever)
    config = {"configurable": {"session_id": "user_1"}}

    first = chain_with_memory.invoke(
        {"question": "What are our company's Q3 OKRs?"},
        config=config,
    )
    second = chain_with_memory.invoke(
        {"question": "Can that strategy work well for customer support?"},
        config=config,
    )
    return f"Turn 1: {first.content}\nTurn 2: {second.content}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Lesson 03 RAG examples")
    parser.add_argument(
        "--demo",
        choices=["all", "manual", "embedding", "retrieval", "rag", "tuned", "rag-memory"],
        default="all",
        help="Choose which demo to run.",
    )
    args = parser.parse_args()

    load_runtime_env()
    llm = create_llm()
    embeddings = create_embeddings()

    if args.demo in {"all", "manual"}:
        print_section("Manual Context", run_manual_context_demo(llm))

    if args.demo in {"all", "embedding"}:
        print_section("Embedding", run_embedding_demo(embeddings))

    if args.demo in {"all", "retrieval"}:
        print_section("Retrieval", run_retrieval_demo(embeddings))

    if args.demo in {"all", "rag"}:
        print_section("RAG", run_rag_demo(llm, embeddings))

    if args.demo in {"all", "tuned"}:
        print_section("Retriever Tuning", run_tuned_retrieval_demo(embeddings))

    if args.demo in {"all", "rag-memory"}:
        print_section("RAG With Memory", run_rag_memory_demo(llm, embeddings))


if __name__ == "__main__":
    main()
