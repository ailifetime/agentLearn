from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI


CONFIG_PATH = Path.home() / ".config" / "codex" / "openai-gateway.env"


def load_runtime_env() -> None:
    """Load runtime configuration from the user-level Codex config directory."""
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


def print_section(title: str, content: str) -> None:
    print(f"[{title}]")
    print(content)
    print()


def run_stateless_demo(llm: ChatOpenAI) -> str:
    first = llm.invoke("My name is Jack. Reply in Chinese.")
    second = llm.invoke("What is my name? Reply in Chinese.")
    return f"Turn 1: {first.content}\nTurn 2: {second.content}"


def run_manual_history_demo(llm: ChatOpenAI) -> str:
    messages = [
        HumanMessage(content="My name is Jack. Reply in Chinese."),
        AIMessage(content="好的，我记住了。"),
        HumanMessage(content="What is my name? Reply in Chinese."),
    ]
    response = llm.invoke(messages)
    return response.content


def build_memory_chain(llm: ChatOpenAI) -> tuple[RunnableWithMessageHistory, dict[str, ChatMessageHistory]]:
    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain = RunnableWithMessageHistory(llm, get_session_history)
    return chain, store


def run_memory_demo(llm: ChatOpenAI) -> str:
    chain, _ = build_memory_chain(llm)
    config = {"configurable": {"session_id": "user_1"}}

    first = chain.invoke("My name is Jack. Reply in Chinese.", config=config)
    second = chain.invoke("What is my name? Reply in Chinese.", config=config)
    return f"Turn 1: {first.content}\nTurn 2: {second.content}"


def build_prompt_memory_chain(
    llm: ChatOpenAI,
) -> tuple[RunnableWithMessageHistory, dict[str, ChatMessageHistory]]:
    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a friendly assistant. Keep replies concise and reply in Chinese."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return chain_with_memory, store


def run_prompt_memory_demo(llm: ChatOpenAI) -> str:
    chain_with_memory, _ = build_prompt_memory_chain(llm)
    config = {"configurable": {"session_id": "user_1"}}

    first = chain_with_memory.invoke(
        {"input": "My name is Jack and I am a programmer."},
        config=config,
    )
    second = chain_with_memory.invoke(
        {"input": "What do I do for work?"},
        config=config,
    )
    return f"Turn 1: {first.content}\nTurn 2: {second.content}"


def run_multi_user_demo(llm: ChatOpenAI) -> str:
    chain_with_memory, _ = build_prompt_memory_chain(llm)
    config_a = {"configurable": {"session_id": "user_A"}}
    config_b = {"configurable": {"session_id": "user_B"}}

    chain_with_memory.invoke({"input": "My name is Alice."}, config=config_a)
    chain_with_memory.invoke({"input": "My name is Bob."}, config=config_b)

    result_a = chain_with_memory.invoke({"input": "What is my name?"}, config=config_a)
    result_b = chain_with_memory.invoke({"input": "What is my name?"}, config=config_b)
    return f"user_A: {result_a.content}\nuser_B: {result_b.content}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Lesson 02 Memory examples")
    parser.add_argument(
        "--demo",
        choices=["all", "stateless", "manual", "memory", "prompt-memory", "multi-user"],
        default="all",
        help="Choose which demo to run.",
    )
    args = parser.parse_args()

    load_runtime_env()
    llm = create_llm()

    if args.demo in {"all", "stateless"}:
        print_section("Stateless", run_stateless_demo(llm))

    if args.demo in {"all", "manual"}:
        print_section("Manual History", run_manual_history_demo(llm))

    if args.demo in {"all", "memory"}:
        print_section("RunnableWithMessageHistory", run_memory_demo(llm))

    if args.demo in {"all", "prompt-memory"}:
        print_section("Prompt With Memory", run_prompt_memory_demo(llm))

    if args.demo in {"all", "multi-user"}:
        print_section("Multi User", run_multi_user_demo(llm))


if __name__ == "__main__":
    main()
