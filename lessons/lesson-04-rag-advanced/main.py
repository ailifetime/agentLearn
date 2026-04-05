from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None


CONFIG_PATH = Path.home() / ".config" / "codex" / "openai-gateway.env"
LESSON_DIR = Path(__file__).resolve().parent
DATA_PATH = LESSON_DIR / "data" / "company_handbook.md"
RUNTIME_DIR = LESSON_DIR / "runtime"
COLLECTION_NAME = "lesson_04_advanced_rag"

SPLITTER_SAMPLE_TEXT = """
LangChain helps developers build LLM applications.

RAG retrieves relevant knowledge before generation.
If chunk boundaries are poor, the retriever can miss the most useful context.

Recursive splitting tries larger separators first, then falls back to smaller ones so each chunk stays coherent.
""".strip()

REFERENCE_TEXTS = [
    "RAG retrieves relevant knowledge at runtime and injects that context into the prompt.",
    "Fine-tuning updates model behavior through training, while RAG keeps the base model fixed.",
    "Chunk overlap keeps neighboring chunks connected when a sentence spans a split boundary.",
    "MMR balances query relevance with diversity so the top results are less repetitive.",
    "A reranker first reads the retrieved candidates, then reorders them using a more precise relevance signal.",
]


class LocalHashEmbeddings(Embeddings):
    """Small deterministic fallback embeddings for environments without a working endpoint."""

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


class LocalKeywordReranker:
    """Simple lexical reranker that keeps the demo runnable without extra vendor keys."""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def score(self, query: str, document: Document) -> float:
        query_tokens = self._tokens(query)
        document_tokens = self._tokens(document.page_content)
        if not query_tokens or not document_tokens:
            return 0.0

        overlap = len(query_tokens & document_tokens)
        coverage = overlap / len(query_tokens)
        density = overlap / len(document_tokens)
        return coverage * 0.8 + density * 0.2

    def rerank(self, query: str, documents: list[Document], top_n: int = 3) -> list[Document]:
        ranked = sorted(
            documents,
            key=lambda document: self.score(query, document),
            reverse=True,
        )
        return ranked[:top_n]


def load_runtime_env() -> None:
    if CONFIG_PATH.exists():
        load_dotenv(CONFIG_PATH, override=False)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to ~/.config/codex/openai-gateway.env."
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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_documents(docs: Iterable[Document]) -> str:
    lines = []
    for index, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        preview = normalize_whitespace(doc.page_content)
        lines.append(f"{index}. ({source}) {preview}")
    return "\n".join(lines)


def deduplicate_documents(docs: Iterable[Document]) -> list[Document]:
    seen: set[tuple[str, str]] = set()
    unique_docs: list[Document] = []
    for doc in docs:
        key = (doc.metadata.get("source", "unknown"), normalize_whitespace(doc.page_content))
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def filter_documents_by_similarity(
    query: str,
    documents: list[Document],
    embeddings: Embeddings,
    *,
    threshold: float,
    k: int,
) -> list[Document]:
    query_vector = embeddings.embed_query(query)
    document_vectors = embeddings.embed_documents([doc.page_content for doc in documents])
    scored_documents = sorted(
        (
            (cosine_similarity(query_vector, doc_vector), doc)
            for doc, doc_vector in zip(documents, document_vectors)
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    passed = [doc for score, doc in scored_documents if score >= threshold]
    return deduplicate_documents(passed[:k])


def load_handbook_documents() -> list[Document]:
    loader = TextLoader(str(DATA_PATH), encoding="utf-8")
    return loader.load()


def build_reference_documents() -> list[Document]:
    docs = [Document(page_content=text, metadata={"source": "reference_notes"}) for text in REFERENCE_TEXTS]
    docs.extend(load_handbook_documents())
    return docs


def build_splitter(chunk_size: int = 320, chunk_overlap: int = 50):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )


def prepare_persist_directory(name: str) -> Path:
    persist_directory = RUNTIME_DIR / name
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    return persist_directory


def build_vectorstore(
    documents: list[Document],
    embeddings: Embeddings,
    *,
    persist_directory: Path,
    collection_name: str,
):
    if Chroma is None:
        raise RuntimeError(
            "Chroma is not available. Install lesson requirements before running this demo."
        )

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )


def build_chunked_vectorstore(embeddings: Embeddings, *, store_name: str):
    raw_documents = build_reference_documents()
    chunks = build_splitter().split_documents(raw_documents)
    persist_directory = prepare_persist_directory(store_name)
    vectorstore = build_vectorstore(
        chunks,
        embeddings,
        persist_directory=persist_directory,
        collection_name=f"{COLLECTION_NAME}_{store_name}",
    )
    return vectorstore, chunks


def run_split_demo() -> str:
    basic_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=140,
        chunk_overlap=20,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=110,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    basic_chunks = basic_splitter.split_text(SPLITTER_SAMPLE_TEXT)
    recursive_chunks = recursive_splitter.split_text(SPLITTER_SAMPLE_TEXT)

    basic_text = "\n".join(
        f"- basic[{index}] ({len(chunk)} chars): {normalize_whitespace(chunk)}"
        for index, chunk in enumerate(basic_chunks)
    )
    recursive_text = "\n".join(
        f"- recursive[{index}] ({len(chunk)} chars): {normalize_whitespace(chunk)}"
        for index, chunk in enumerate(recursive_chunks)
    )
    return f"{basic_text}\n\n{recursive_text}"


def run_document_demo() -> str:
    documents = load_handbook_documents()
    chunks = build_splitter(chunk_size=260, chunk_overlap=40).split_documents(documents)
    first_chunk = chunks[0]
    return (
        f"Loaded documents: {len(documents)}\n"
        f"Chunks after splitting: {len(chunks)}\n"
        f"First chunk metadata: {first_chunk.metadata}\n"
        f"First chunk preview: {normalize_whitespace(first_chunk.page_content)}"
    )


def run_retrieval_demo(embeddings: Embeddings) -> str:
    vectorstore, chunks = build_chunked_vectorstore(embeddings, store_name="retrieval")
    query = "How many days of annual leave do employees get?"

    threshold_docs = filter_documents_by_similarity(
        query,
        chunks,
        embeddings,
        threshold=0.15,
        k=4,
    )
    mmr_docs = deduplicate_documents(vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.7},
    ).invoke(query))

    threshold_text = format_documents(threshold_docs) or "No document passed the threshold."
    mmr_text = format_documents(mmr_docs)
    return f"Similarity threshold:\n{threshold_text}\n\nMMR:\n{mmr_text}"


def run_rerank_demo(embeddings: Embeddings) -> str:
    vectorstore, _ = build_chunked_vectorstore(embeddings, store_name="rerank")
    query = "How is RAG different from fine-tuning?"
    initial_docs = deduplicate_documents(vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(query))
    reranked_docs = LocalKeywordReranker().rerank(query, initial_docs, top_n=3)
    return (
        "Initial vector search:\n"
        f"{format_documents(initial_docs)}\n\n"
        "After local reranking:\n"
        f"{format_documents(reranked_docs)}"
    )


def rewrite_question(llm: ChatOpenAI, history: str, question: str) -> str:
    rewrite_prompt = ChatPromptTemplate.from_template(
        """
Rewrite the latest question into a standalone search query.

Conversation history:
{history}

Latest question:
{question}

Standalone query:
""".strip()
    )
    chain = rewrite_prompt | llm | StrOutputParser()
    return chain.invoke({"history": history, "question": question}).strip()


def run_history_demo(llm: ChatOpenAI, embeddings: Embeddings) -> str:
    vectorstore, _ = build_chunked_vectorstore(embeddings, store_name="history")
    history = "User: What is RAG?\nAssistant: RAG retrieves external knowledge before generation."
    follow_up = "How is it different from fine-tuning?"
    rewritten = rewrite_question(llm, history, follow_up)
    docs = deduplicate_documents(vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(rewritten))
    return (
        f"Original question: {follow_up}\n"
        f"Rewritten question: {rewritten}\n\n"
        f"Retrieved documents:\n{format_documents(docs)}"
    )


def run_pipeline_demo(llm: ChatOpenAI, embeddings: Embeddings) -> str:
    vectorstore, _ = build_chunked_vectorstore(embeddings, store_name="pipeline")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7},
    )

    question = "What is the company's annual leave policy?"
    docs = deduplicate_documents(retriever.invoke(question))
    context = "\n\n".join(
        f"[Source: {Path(doc.metadata.get('source', 'unknown')).name}]\n{doc.page_content}"
        for doc in docs
    )
    prompt = ChatPromptTemplate.from_template(
        """
You are a precise knowledge-base assistant.

Rules:
- Answer only with information supported by the context.
- If the context is insufficient, say "I could not find relevant information."
- Mention the source when possible.

Context:
{context}

Question:
{question}
""".strip()
    )
    response = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return f"Question: {question}\n\nContext used:\n{format_documents(docs)}\n\nAnswer:\n{response}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Lesson 04 Advanced RAG examples")
    parser.add_argument(
        "--demo",
        choices=["all", "split", "documents", "retrieval", "rerank", "history", "pipeline"],
        default="all",
        help="Choose which demo to run.",
    )
    args = parser.parse_args()

    load_runtime_env()

    needs_embeddings = args.demo in {"all", "retrieval", "rerank", "history", "pipeline"}
    needs_llm = args.demo in {"all", "history", "pipeline"}

    embeddings = create_embeddings() if needs_embeddings else None
    llm = create_llm() if needs_llm else None

    if args.demo in {"all", "split"}:
        print_section("Chunking", run_split_demo())

    if args.demo in {"all", "documents"}:
        print_section("Document Splitting", run_document_demo())

    if args.demo in {"all", "retrieval"}:
        print_section("Retrieval Tuning", run_retrieval_demo(embeddings))

    if args.demo in {"all", "rerank"}:
        print_section("Reranking", run_rerank_demo(embeddings))

    if args.demo in {"all", "history"}:
        print_section("History-Aware Retrieval", run_history_demo(llm, embeddings))

    if args.demo in {"all", "pipeline"}:
        print_section("Production Pipeline", run_pipeline_demo(llm, embeddings))


if __name__ == "__main__":
    main()
