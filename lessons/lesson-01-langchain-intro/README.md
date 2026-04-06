# Lesson 01: LangChain Intro

This folder contains the runnable code for the first article:

- Directly call `ChatOpenAI`
- Build a chain with `PromptTemplate`
- Create a simple translation assistant

## Setup

This project follows the workspace rule that secrets stay outside the repository.
Store your OpenAI-compatible configuration in:

- `~/.config/code/openai-gateway.env`

Expected variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `OPENAI_MODEL` (optional)

## Install

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -r lessons/lesson-01-langchain-intro/requirements.txt
```

## Run

Run all examples:

```bash
python lessons/lesson-01-langchain-intro/main.py
```

If you run the file from the VS Code top-right `Run` button, make sure the workspace
interpreter is set to `.venv/bin/python`. Otherwise VS Code may use a global Python
installation and fail with `ModuleNotFoundError: No module named 'dotenv'` even though
the package is installed in the virtual environment.

Run a single example:

```bash
python lessons/lesson-01-langchain-intro/main.py --demo simple
python lessons/lesson-01-langchain-intro/main.py --demo prompt
python lessons/lesson-01-langchain-intro/main.py --demo translate --text "我喜欢编程"
```
