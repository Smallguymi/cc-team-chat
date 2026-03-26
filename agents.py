"""
CC Agent classes — Manager and Worker.
Uses llm_provider.py so any supported LLM backend can be plugged in.
"""

import asyncio
import json
from pathlib import Path
from typing import Callable

from llm_provider import make_provider, ToolCall
from skills import execute as execute_skill, get_tool_schema

BASE_DIR  = Path(__file__).parent
CLAUDE_MD = BASE_DIR / "CLAUDE.md"   # overridden by set_data_dir()
TASKS_DIR = BASE_DIR / "tasks"       # overridden by set_data_dir()


def set_data_dir(data_dir: Path) -> None:
    """Point agents at a project data directory. Call before any agents are created."""
    global CLAUDE_MD, TASKS_DIR
    CLAUDE_MD = data_dir / "CLAUDE.md"
    TASKS_DIR = data_dir / "tasks"


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
# Worker Species
# Each species defines a role and the set of skills the worker is allowed to use.
# Skills map to tool names in skills.py — add new skills there, then reference here.
# ---------------------------------------------------------------------------

WORKER_SPECIES: dict[str, dict] = {
    "generalist": {
        "name": "Generalist",
        "description": "Handles any general task using knowledge alone. No external tools.",
        "role": "You are a generalist assistant. Answer thoroughly using your own knowledge.",
        "skills": [],
    },
    "researcher": {
        "name": "Researcher",
        "description": "Finds current information by searching the web.",
        "role": (
            "You are a research specialist. Use web_search to find up-to-date information. "
            "Always cite the source URLs in your result."
        ),
        "skills": ["web_search"],
    },
    "file_editor": {
        "name": "File Editor",
        "description": "Reads, writes, and organises files on the local filesystem.",
        "role": (
            "You are a file management specialist. Use read_file, write_file, and "
            "list_directory to work with the filesystem. Always confirm what you wrote."
        ),
        "skills": ["read_file", "write_file", "list_directory"],
    },
    "analyst": {
        "name": "Analyst",
        "description": "Reads files and analyses or summarises their contents.",
        "role": (
            "You are a data analyst. Use read_file and list_directory to inspect source "
            "material, then produce a clear analysis or summary."
        ),
        "skills": ["read_file", "list_directory"],
    },
}


# ---------------------------------------------------------------------------
# Manager Agent
# ---------------------------------------------------------------------------

MANAGER_SYSTEM = """You are Manager CC — an orchestrator in a multi-agent team chat.

YOUR ROLE:
- Understand what the user needs
- Decide whether to handle it directly or delegate to one or more Worker CCs
- For complex requests with distinct parts, spawn multiple workers in parallel (one per subtask)
- For simple requests, a single worker or a direct answer is fine
- After workers complete, summarise their results and report back to the user

WORKER SPECIES available to you:
{species_list}

SPAWNING RULES:
- Choose the species that best matches each subtask
- Worker briefs must be 100% self-contained — workers have NO other context beyond what you write
- Only spawn workers when the user explicitly asks you to do something
- After receiving worker results, summarise for the user — do NOT auto-spawn follow-up workers

CONTEXT (updated each session):
{claude_md}
"""

SPAWN_WORKER_TOOL = {
    "name": "spawn_worker",
    "description": (
        "Spawn a Worker CC for a specific subtask. "
        "Choose the species that best matches the work. "
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
            "species": {
                "type": "string",
                "description": "Worker species to use. Must be one of the available species IDs.",
                "enum": list(WORKER_SPECIES.keys()),
            },
        },
        "required": ["task_id", "task_title", "task_brief", "species"],
    },
}


class ManagerAgent:
    def __init__(self, broadcast_fn: Callable):
        self.history: list[dict] = []
        self.broadcast = broadcast_fn
        self._summarizing: bool = False   # True while responding to a worker result

    def _system(self) -> str:
        species_list = "\n".join(
            f"  - {sid} ({s['name']}): {s['description']}"
            + (f"  [skills: {', '.join(s['skills'])}]" if s["skills"] else "  [no tools]")
            for sid, s in WORKER_SPECIES.items()
        )
        return MANAGER_SYSTEM.format(claude_md=_load_claude_md(), species_list=species_list)

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
        provider_cfg: dict,
        spawn_worker_fn: Callable,
    ):
        """Append worker result then trigger a summary turn.
        spawn_worker tool is disabled during this turn to prevent loops."""
        note = (
            f"[Worker {worker_id} completed: '{task_title}']\n\n"
            f"Result:\n{result[:1500]}"
        )
        self.history.append({"role": "user", "content": note})
        self._summarizing = True
        try:
            await self._run_turn(provider_cfg, spawn_worker_fn)
        finally:
            self._summarizing = False

    async def _run_turn(self, provider_cfg: dict, spawn_worker_fn: Callable):
        provider = make_provider(provider_cfg)
        # Disable spawn_worker while summarising a worker result — prevents infinite loops
        tools = [] if self._summarizing else [SPAWN_WORKER_TOOL]

        while True:
            async def on_delta(text: str):
                await self.broadcast({"type": "stream_delta", "sender": "manager", "content": text})

            result = await provider.stream_turn(
                messages=self.history,
                system=self._system(),
                tools=tools,
                on_delta=on_delta,
            )
            await self.broadcast({"type": "stream_end", "sender": "manager"})

            if result.tool_calls:
                self.history.append({
                    "role": "assistant",
                    "text": result.text,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in result.tool_calls
                    ],
                })

                for tc in result.tool_calls:
                    if tc.name == "spawn_worker":
                        task_id    = tc.input.get("task_id", "task_unknown")
                        task_title = tc.input.get("task_title", "Untitled")
                        task_brief = tc.input.get("task_brief", "")
                        species    = tc.input.get("species", "generalist")
                        _save_task_brief(task_id, task_title, task_brief)
                        await spawn_worker_fn(
                            task_id, task_title, task_brief, provider_cfg,
                            species=species,
                        )
                        self.history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": f"Worker spawned for '{task_title}' (id: {task_id}, species: {species}).",
                        })
            else:
                self.history.append({"role": "assistant", "content": result.text})
                break


# ---------------------------------------------------------------------------
# Worker Agent
# ---------------------------------------------------------------------------

WORKER_SYSTEM = """You are Worker CC — an autonomous specialist assigned ONE atomic task.

SPECIES: {species_name}
ROLE: {species_role}

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
        species: str = "generalist",
    ):
        self.worker_id       = worker_id
        self.task_id         = task_id
        self.task_title      = task_title
        self.task_brief      = task_brief
        self.broadcast       = broadcast_fn
        self.permission_mode = permission_mode
        self.species         = species
        self._history: list[dict] = []
        self._perm_event:  asyncio.Event | None = None
        self._perm_answer: str | None           = None

    def resolve_permission(self, answer: str):
        """Called by the server when the user responds to a permission request."""
        self._perm_answer = answer
        if self._perm_event:
            self._perm_event.set()

    def _species_cfg(self) -> dict:
        return WORKER_SPECIES.get(self.species, WORKER_SPECIES["generalist"])

    def _get_tools(self) -> list[dict]:
        """Return tool schemas for this worker's species."""
        schemas = []
        for skill_name in self._species_cfg()["skills"]:
            schema = get_tool_schema(skill_name)
            if schema:
                schemas.append(schema)
        return schemas

    def _save_memory(self):
        """Write resumption file so a crashed worker can be re-briefed."""
        d = TASKS_DIR / "active"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.task_id}_memory.txt"
        lines = [
            f"# Worker Memory — {self.task_title}\n",
            f"Task ID : {self.task_id}\n",
            f"Worker  : {self.worker_id}\n",
            f"Species : {self.species}\n\n",
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
        cfg = self._species_cfg()
        return WORKER_SYSTEM.format(
            species_name=cfg["name"],
            species_role=cfg["role"],
            task_id=self.task_id,
            task_title=self.task_title,
            task_brief=self.task_brief,
        )

    def _label(self) -> str:
        s = self.task_title
        return "Worker — " + (s[:25] + "…" if len(s) > 25 else s)

    async def _run_tool_loop(self, provider) -> str:
        """Run LLM + tool-call loop until a final text response. Returns final text."""
        tools = self._get_tools()

        while True:
            async def on_delta(text: str):
                await self.broadcast({"type": "stream_delta", "sender": self.worker_id, "content": text})

            result = await provider.stream_turn(
                messages=self._history,
                system=self._system(),
                tools=tools,
                on_delta=on_delta,
            )
            await self.broadcast({"type": "stream_end", "sender": self.worker_id})

            if result.tool_calls:
                self._history.append({
                    "role": "assistant",
                    "text": result.text,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in result.tool_calls
                    ],
                })
                for tc in result.tool_calls:
                    # Show the tool call in the chat
                    preview = json.dumps(tc.input)
                    if len(preview) > 120:
                        preview = preview[:117] + "…"
                    await self.broadcast({
                        "type": "message",
                        "sender": self.worker_id,
                        "sender_label": self._label(),
                        "content": f"🔧 **{tc.name}** `{preview}`",
                        "complete": True,
                    })
                    tool_result = await execute_skill(tc.name, tc.input)
                    self._history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": str(tool_result),
                    })
                self._save_memory()
            else:
                self._history.append({"role": "assistant", "content": result.text})
                self._save_memory()
                return result.text

    async def start(self, provider_cfg: dict, on_complete: Callable):
        await self.broadcast({
            "type": "message",
            "sender": self.worker_id,
            "sender_label": self._label(),
            "content": f"Starting: *{self.task_title}*",
            "complete": True,
        })

        provider = make_provider(provider_cfg)
        self._history.append({"role": "user", "content": f"Please complete your task: {self.task_title}"})
        self._save_memory()

        final_text = await self._run_tool_loop(provider)

        if self.permission_mode == "ask":
            self._perm_event  = asyncio.Event()
            self._perm_answer = None
            await self.broadcast({
                "type":       "permission_request",
                "worker_id":  self.worker_id,
                "task_title": self.task_title,
                "summary":    final_text[:600],
            })
            await self._perm_event.wait()

            if self._perm_answer == "no":
                await self.broadcast({
                    "type": "message", "sender": self.worker_id,
                    "sender_label": self._label(),
                    "content": "Task cancelled by user.", "complete": True,
                })
                return

        _save_task_result(self.task_id, self.task_title, final_text)
        await on_complete(self.worker_id, self.task_title, final_text)

    async def process_message(self, text: str, provider_cfg: dict):
        provider = make_provider(provider_cfg)
        self._history.append({"role": "user", "content": text})
        self._save_memory()
        await self._run_tool_loop(provider)
