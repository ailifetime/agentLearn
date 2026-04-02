# Lesson 02: Memory

This folder contains the runnable code for the second article:

- Stateless chat calls
- Manual history injection
- `RunnableWithMessageHistory`
- Prompt templates with memory
- Multi-user session isolation

## Setup

This project follows the workspace rule that secrets stay outside the repository.
Store your OpenAI-compatible configuration in:

- `~/.config/codex/openai-gateway.env`

Expected variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `OPENAI_MODEL` (optional)

## Install

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -r lessons/lesson-02-memory/requirements.txt
```

## Run

Run all examples:

```bash
python lessons/lesson-02-memory/main.py
```

Run a single example:

```bash
python lessons/lesson-02-memory/main.py --demo stateless
python lessons/lesson-02-memory/main.py --demo manual
python lessons/lesson-02-memory/main.py --demo memory
python lessons/lesson-02-memory/main.py --demo prompt-memory
python lessons/lesson-02-memory/main.py --demo multi-user
```

## Notes

- The example uses in-memory storage, so conversation history disappears after the process exits.
- `session_id` is the key used to isolate each user's chat history.
