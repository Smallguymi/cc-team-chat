# CC Team Chat

A local multi-agent chat interface that lets you orchestrate a **Manager CC** and multiple **Worker CCs** — each powered by an LLM of your choice — all from a clean web UI running on your machine.
PS:其实我也不知道这B程序的细节，全是CC自己写的，我就说，啊呀我好想要一个OpenClaw啊，但是我买不起token，就剩下没几天的CC会员了，你帮我整个呗，然后他就自己捉摸了两天写出来这玩意

![screenshot placeholder]

---

## What it does

- **Manager CC** receives your messages, breaks work into subtasks, and delegates them to Worker CCs
- **Worker CCs** are autonomous specialists that each handle one atomic task and report back
- All conversations are isolated per participant — click a name in the sidebar to switch threads
- Every session is logged to `logs/session_*.txt` for later review
- Workers save a memory file (`tasks/active/`) after every exchange so they can resume after a crash

---

## Supported providers

| Provider | What you need |
|---|---|
| **Claude CLI** | Claude Code installed + Pro subscription — no extra API key |
| **Anthropic API** | An Anthropic API key |
| **OpenAI-compatible** | Any OpenAI-compatible endpoint (OpenAI, DeepSeek, Gemini, Alibaba Bailian, etc.) |

---

## Requirements

- Python 3.10+
- [Claude Code CLI](https://claude.ai/code) (optional, for CLI provider)

---

## Quick start

### Windows

```bat
start.bat
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### Manual (any platform)

```bash
pip install -r requirements.txt
python app.py
```

---

## Project structure

```
my_company/
├── app.py              # FastAPI server + WebSocket hub
├── agents.py           # ManagerAgent and WorkerAgent classes
├── llm_provider.py     # LLM backend abstraction (Anthropic, OpenAI-compat, Claude CLI)
├── CLAUDE.md           # Context loaded into Manager's system prompt each session
├── start.bat           # Windows one-click launcher
├── requirements.txt
├── static/
│   └── index.html      # Single-page chat UI
├── logs/               # Auto-created; session transcripts written here
└── tasks/
    ├── active/         # Worker task briefs + memory files (auto-created)
    └── done/           # Completed task results (auto-created)
```

---

## Configuration

Click **Settings** (top-right gear icon) in the UI to configure your provider:

- **Claude CLI** — no key needed; uses your local `claude` installation
- **Anthropic** — paste your API key and pick a model (e.g. `claude-opus-4-6`)
- **OpenAI-compatible** — paste your API key, base URL, and model name

Settings are saved in browser `localStorage`.

---

## Adding a Worker CC

Click **+ Add Worker CC** in the sidebar. Fill in:

- **Task title** — short label shown in the sidebar
- **Task brief** — full self-contained instructions (workers have no other context)
- **Permission mode**:
  - **Ask before completing** — worker pauses and shows you a summary; you click Yes / No / Tell me what to do
  - **Full permission** — worker completes automatically without asking

---

## Worker permission modes

### Ask before completing (default)
When the worker finishes its task it pauses and shows you a card with its output summary. You can:
- **Yes** — accept the result and mark the task complete
- **No** — cancel; result is discarded
- **Tell me what to do** — send a follow-up instruction directly to the worker

### Full permission
The worker runs to completion immediately and reports back to the Manager without pausing.

---

## Stopping agents

- **■ stop button** on each sidebar row — cancels that individual agent
- **⬛ Stop All** (sidebar footer) — cancels every running agent at once
- **Exit CC Team Chat** (sidebar footer) — shuts the server down completely

---

## Session logs

Every session creates a timestamped file in `logs/`, e.g.:

```
logs/session_2026-03-26_143012.txt
```

It contains the full conversation across all participants, with timestamps and sender labels.

---

## Customising the Manager

Edit `CLAUDE.md` to give the Manager permanent context about your project. This file is loaded into the Manager's system prompt at the start of every session.

Example:

```markdown
# My Project

We are building a FastAPI backend for a task management app.
Stack: Python, PostgreSQL, Redis.
Style guide: PEP 8, type hints everywhere.
```

---

## License

MIT
