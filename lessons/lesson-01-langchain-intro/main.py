from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
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


def run_simple_demo(llm: ChatOpenAI) -> str:
    response = llm.invoke("Explain what LangChain is in one sentence.")
    return response.content


def run_prompt_demo(llm: ChatOpenAI) -> str:
    prompt = PromptTemplate.from_template("Explain {topic} in simple language.")
    chain = prompt | llm
    result = chain.invoke({"topic": "LangChain"})
    return result.content


def run_translation_demo(llm: ChatOpenAI, text: str) -> str:
    prompt = PromptTemplate.from_template("Translate the following text into English: {text}")
    chain = prompt | llm
    result = chain.invoke({"text": text})
    return result.content


def print_section(title: str, content: str) -> None:
    print(f"[{title}]")
    print(content)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lesson 01 LangChain examples")
    parser.add_argument(
        "--demo",
        choices=["all", "simple", "prompt", "translate"],
        default="all",
        help="Choose which demo to run.",
    )
    parser.add_argument(
        "--text",
        default="I enjoy programming.",
        help="Input text for the translation demo.",
    )
    args = parser.parse_args()

    load_runtime_env()
    llm = create_llm()

    if args.demo in {"all", "simple"}:
        print_section("Simple", run_simple_demo(llm))

    if args.demo in {"all", "prompt"}:
        print_section("Prompt Template", run_prompt_demo(llm))

    if args.demo in {"all", "translate"}:
        print_section("Translate", run_translation_demo(llm, args.text))


if __name__ == "__main__":
    main()
