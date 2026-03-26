"""
CC Agent classes — Manager and Worker.
Uses llm_provider.py so any supported LLM backend can be plugged in.
"""

from pathlib import Path
from typing import Callable, Awaitable

from llm_provider import make_provider, ToolCall

BASE_DIR  = Path(__file__).parent
CLAUDE_MD = BASE_DIR / "CLAUDE.md"
TASKS_DIR = BASE_DIR / "tasks"


def _load_claude_md() -> str:
    if CLAUDE_MD.exists():
        return CLAUDE_MD.read_text(encoding="utf-8")
    return "No CLAUDE.md found. Ask the user what project we are working on."


def _save_task_brief(task_id: str, task_title: str, brief: str):
    d = TASKS_DIR / "active"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{task_id}.md").write_text(f"# {task_title}\n\n{brief}", encoding="utf-8")


def _save_task_result(task_id: str, task_title: str, result: str):
    d = TASKS_DIR / "done"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{task_id}_result.md").write_text(
        f"# {task_title} — Result\n\n{result}", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Manager Agent
# ---------------------------------------------------------------------------

MANAGER_SYSTEM = """You are Manager CC — an orchestrator in a multi-agent team chat.

YOUR ROLE:
- Talk with the user and understand what they need
- Break work into small, atomic subtasks and delegate them to Worker CCs using the spawn_worker tool
- Review worker results and report back to the user
- Give clear feedback on worker output

RULES:
- Only spawn workers when the user explicitly asks you to do something
- Worker briefs must be 100% self-contained — workers have NO other context beyond what you write
- When a worker result arrives, summarise it for the user — do NOT auto-spawn more workers
- Do not invent tasks or assume any project context unless the user tells you
- Be concise and direct. Wait for user direction before acting.

CONTEXT (updated each session):
{claude_md}
"""

SPAWN_WORKER_TOOL = {
    "name": "spawn_worker",
    "description": (
        "Spawn a new Worker CC with a specific, atomic, self-contained task. "
        "The worker has NO context beyond what you write in task_brief."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id":    {"type": "string", "description": "Unique slug, e.g. task_001"},
            "task_title": {"type": "string", "description": "Short human-readable title"},
            "task_brief": {
                "type": "string",
                "description": (
                    "Full self-contained instructions. Include all context, "
                    "acceptance criteria, and constraints the worker needs."
                ),
            },
        },
        "required": ["task_id", "task_title", "task_brief"],
    },
}


class ManagerAgent:
    def __init__(self, broadcast_fn: Callable):
        self.history: list[dict] = []
        self.broadcast = broadcast_fn

    def _system(self) -> str:
        return MANAGER_SYSTEM.format(claude_md=_load_claude_md())

    async def process_message(
        self,
        text: str,
        provider_cfg: dict,
        spawn_worker_fn: Callable,
    ):
        self.history.append({"role": "user", "content": text})
        await self._run_turn(provider_cfg, spawn_worker_fn)

    async def receive_worker_result(
        self,
        worker_id: str,
        task_title: str,
        result: str,
    ):
        """Store worker result in history. Manager will include it as context
        on the next user-initiated turn — no automatic follow-up to prevent loops."""
        note = (
            f"[Worker {worker_id} completed: '{task_title}']\n\n"
            f"Result:\n{result[:1500]}"
        )
        self.history.append({"role": "user", "content": note})

    async def _run_turn(self, provider_cfg: dict, spawn_worker_fn: Callable):
        provider = make_provider(provider_cfg)

        while True:
            async def on_delta(text: str):
                await self.broadcast({"type": "stream_delta", "sender": "manager", "content": text})

            result = await provider.stream_turn(
                messages=self.history,
                system=self._system(),
                tools=[SPAWN_WORKER_TOOL],
                on_delta=on_delta,
            )
            await self.broadcast({"type": "stream_end", "sender": "manager"})

            if result.tool_calls:
                # Append assistant turn with tool calls
                self.history.append({
                    "role": "assistant",
                    "text": result.text,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in result.tool_calls
                    ],
                })

                # Execute each tool and collect results
                for tc in result.tool_calls:
                    if tc.name == "spawn_worker":
                        task_id    = tc.input.get("task_id", "task_unknown")
                        task_title = tc.input.get("task_title", "Untitled")
                        task_brief = tc.input.get("task_brief", "")
                        _save_task_brief(task_id, task_title, task_brief)
                        await spawn_worker_fn(task_id, task_title, task_brief, provider_cfg)
                        self.history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": f"Worker spawned for '{task_title}' (id: {task_id}).",
                        })
            else:
                # No tool calls — turn complete
                self.history.append({"role": "assistant", "content": result.text})
                break


# ---------------------------------------------------------------------------
# Worker Agent
# ---------------------------------------------------------------------------

WORKER_SYSTEM = """You are Worker CC — an autonomous specialist assigned ONE atomic task.

TASK ID: {task_id}
TASK: {task_title}

FULL BRIEF:
{task_brief}

RULES:
- Focus ONLY on this task. Do not ask for more work.
- Be thorough but concise.
- End your response with a clear "## Result" section summarising what you produced.
- If you cannot complete the task, explain exactly why.
"""


class WorkerAgent:
    def __init__(
        self,
        worker_id: str,
        task_id: str,
        task_title: str,
        task_brief: str,
        broadcast_fn: Callable,
        permission_mode: str = "ask",   # "ask" | "full"
    ):
        self.worker_id       = worker_id
        self.task_id         = task_id
        self.task_title      = task_title
        self.task_brief      = task_brief
        self.broadcast       = broadcast_fn
        self.permission_mode = permission_mode
        self._history: list[dict] = []
        self._perm_event:  asyncio.Event | None = None
        self._perm_answer: str | None           = None

    def resolve_permission(self, answer: str):
        """Called by the server when the user responds to a permission request."""
        self._perm_answer = answer
        if self._perm_event:
            self._perm_event.set()

    def _save_memory(self):
        """Write resumption file so a crashed worker can be re-briefed."""
        d = TASKS_DIR / "active"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.task_id}_memory.txt"
        lines = [
            f"# Worker Memory — {self.task_title}\n",
            f"Task ID : {self.task_id}\n",
            f"Worker  : {self.worker_id}\n\n",
            "## Original Brief\n\n",
            self.task_brief,
            "\n\n## Conversation History\n\n",
        ]
        for msg in self._history:
            role    = msg.get("role", "")
            content = msg.get("content") or msg.get("text", "")
            lines.append(f"[{role.upper()}]\n{content}\n\n")
        lines.append(
            "---\n"
            "If resuming after a crash: read the history above, then continue the task.\n"
        )
        path.write_text("".join(lines), encoding="utf-8")

    def _system(self) -> str:
        return WORKER_SYSTEM.format(
            task_id=self.task_id,
            task_title=self.task_title,
            task_brief=self.task_brief,
        )

    def _label(self) -> str:
        s = self.task_title
        return "Worker — " + (s[:25] + "…" if len(s) > 25 else s)

    async def start(self, provider_cfg: dict, on_complete: Callable):
        await self.broadcast({
            "type": "message",
            "sender": self.worker_id,
            "sender_label": self._label(),
            "content": f"Starting: *{self.task_title}*",
            "complete": True,
        })

        provider  = make_provider(provider_cfg)
        user_msg  = {"role": "user", "content": f"Please complete your task: {self.task_title}"}
        self._history.append(user_msg)
        self._save_memory()

        async def on_delta(text: str):
            await self.broadcast({"type": "stream_delta", "sender": self.worker_id, "content": text})

        result = await provider.stream_turn(
            messages=self._history,
            system=self._system(),
            tools=[],
            on_delta=on_delta,
        )
        await self.broadcast({"type": "stream_end", "sender": self.worker_id})

        self._history.append({"role": "assistant", "content": result.text})
        self._save_memory()

        if self.permission_mode == "ask":
            # Pause and ask the user before completing
            self._perm_event  = asyncio.Event()
            self._perm_answer = None
            await self.broadcast({
                "type":       "permission_request",
                "worker_id":  self.worker_id,
                "task_title": self.task_title,
                "summary":    result.text[:600],
            })
            await self._perm_event.wait()

            if self._perm_answer == "no":
                await self.broadcast({
                    "type": "message", "sender": self.worker_id,
                    "sender_label": self._label(),
                    "content": "Task cancelled by user.", "complete": True,
                })
                return
            # "yes" or any other value → complete normally

        _save_task_result(self.task_id, self.task_title, result.text)
        await on_complete(self.worker_id, self.task_title, result.text)

    async def process_message(self, text: str, provider_cfg: dict):
        provider = make_provider(provider_cfg)
        self._history.append({"role": "user", "content": text})
        self._save_memory()

        async def on_delta(chunk: str):
            await self.broadcast({"type": "stream_delta", "sender": self.worker_id, "content": chunk})

        result = await provider.stream_turn(
            messages=self._history,
            system=self._system(),
            tools=[],
            on_delta=on_delta,
        )
        await self.broadcast({"type": "stream_end", "sender": self.worker_id})

        self._history.append({"role": "assistant", "content": result.text})
        self._save_memory()
