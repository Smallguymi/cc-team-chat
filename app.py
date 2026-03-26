"""
CC Team Chat — FastAPI server.

Usage:
  python app.py                         # uses ./userdata as base project dir
  python app.py --project /path/to/dir  # use any folder as base project dir

WebSocket: /ws/{room_id}
  Each room is an independent multi-agent chat with its own manager + workers.
"""

import argparse
import asyncio
import datetime
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents import ManagerAgent, WorkerAgent, WORKER_SPECIES

BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# Set by configure() before startup
DATA_DIR: Path = BASE_DIR / "userdata"

WORKER_COLORS = [
    "#2ECC71", "#E67E22", "#E74C3C", "#3498DB",
    "#1ABC9C", "#F39C12", "#8E44AD", "#16A085",
]

DEFAULT_CLAUDE_MD = (
    "# Project Context\n\n"
    "No project has been defined yet. "
    "Wait for the user to tell you what to work on.\n"
)

# ---------------------------------------------------------------------------
# Room — one independent multi-agent chat
# ---------------------------------------------------------------------------

class Room:
    def __init__(self, room_id: str, name: str, data_dir: Path):
        self.room_id  = room_id
        self.name     = name
        self.data_dir = data_dir.resolve()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        claude_md = self.data_dir / "CLAUDE.md"
        if not claude_md.exists():
            claude_md.write_text(DEFAULT_CLAUDE_MD, encoding="utf-8")

        log_dir = self.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._log_file = log_dir / f"session_{ts}.txt"
        self._stream_acc: dict[str, str] = {}

        self.connections:    list[WebSocket]         = []
        self.workers:        dict[str, WorkerAgent]  = {}
        self.worker_counter: int                     = 0
        self.worker_tasks:   dict[str, asyncio.Task] = {}
        self.manager_task:   asyncio.Task | None     = None
        self.manager = ManagerAgent(broadcast_fn=self.broadcast, data_dir=self.data_dir)

    # -- Logging -------------------------------------------------------

    def _log(self, message: dict):
        ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        typ = message.get("type", "")

        if typ == "message":
            label   = message.get("sender_label") or message.get("sender", "unknown")
            content = message.get("content", "")
            line    = f"\n[{ts}] {label}:\n{content}\n{'─'*60}\n"

        elif typ == "stream_delta":
            sender = message.get("sender", "")
            self._stream_acc[sender] = self._stream_acc.get(sender, "") + message.get("content", "")
            return

        elif typ == "stream_end":
            sender = message.get("sender", "")
            text   = self._stream_acc.pop(sender, "")
            if not text:
                return
            label = "Manager CC" if sender == "manager" else f"Worker ({sender})"
            line  = f"\n[{ts}] {label}:\n{text}\n{'─'*60}\n"

        elif typ == "worker_spawned":
            line = f"\n[{ts}] --- Worker spawned: {message.get('task_title','')} [{message.get('species','')}] ---\n"

        elif typ == "worker_complete":
            line = f"\n[{ts}] --- Worker complete: {message.get('task_title','')} ---\n"

        else:
            return

        self._log_file.open("a", encoding="utf-8").write(line)

    # -- Broadcast -------------------------------------------------------

    async def broadcast(self, message: dict):
        self._log(message)
        data = json.dumps(message)
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.remove(ws)

    # -- Spawn -----------------------------------------------------------

    async def spawn_worker(
        self,
        task_id: str,
        task_title: str,
        task_brief: str,
        provider_cfg: dict,
        permission_mode: str = "ask",
        species: str = "generalist",
    ) -> str:
        """Spawn a worker and return its worker_id."""
        self.worker_counter += 1
        worker_id    = f"worker_{self.worker_counter}"
        color        = WORKER_COLORS[(self.worker_counter - 1) % len(WORKER_COLORS)]
        label        = "Worker — " + (task_title[:25] + "…" if len(task_title) > 25 else task_title)
        species_cfg  = WORKER_SPECIES.get(species, WORKER_SPECIES["generalist"])

        worker = WorkerAgent(
            worker_id=worker_id,
            task_id=task_id,
            task_title=task_title,
            task_brief=task_brief,
            broadcast_fn=self.broadcast,
            data_dir=self.data_dir,
            permission_mode=permission_mode,
            species=species,
        )
        self.workers[worker_id] = worker

        await self.broadcast({
            "type": "worker_spawned",
            "worker_id": worker_id,
            "task_id": task_id,
            "task_title": task_title,
            "color": color,
            "label": label,
            "permission_mode": permission_mode,
            "species": species,
            "species_name": species_cfg["name"],
        })

        async def on_complete(wid: str, title: str, result: str):
            self.worker_tasks.pop(wid, None)
            await self.broadcast({"type": "worker_complete", "worker_id": wid, "task_title": title})
            await self.manager.receive_worker_result(
                wid, title, result, provider_cfg, self.spawn_worker
            )

        task = asyncio.create_task(worker.start(provider_cfg, on_complete))
        self.worker_tasks[worker_id] = task
        task.add_done_callback(lambda t: self.worker_tasks.pop(worker_id, None))
        return worker_id

    # -- Message handler -------------------------------------------------

    async def handle_message(self, msg: dict, ws: WebSocket):
        provider_cfg: dict = msg.get("provider", {})

        if msg["type"] == "user_message":
            content = msg.get("content", "").strip()
            if not content:
                return
            if not provider_cfg.get("api_key"):
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": "No API key — open Settings and configure your provider.",
                }))
                return
            await self.broadcast({
                "type": "message", "sender": "user",
                "sender_label": "You", "content": content, "complete": True,
            })
            self.manager_task = asyncio.create_task(
                self.manager.process_message(content, provider_cfg, self.spawn_worker)
            )

        elif msg["type"] == "worker_message":
            wid     = msg.get("worker_id", "")
            content = msg.get("content", "").strip()
            if wid in self.workers and content and provider_cfg.get("api_key"):
                asyncio.create_task(self.workers[wid].process_message(content, provider_cfg))

        elif msg["type"] == "spawn_worker":
            task_title      = msg.get("task_title", "Manual task")
            task_brief      = msg.get("task_brief", "No brief provided.")
            task_id         = msg.get("task_id", f"manual_{self.worker_counter + 1:03d}")
            permission_mode = msg.get("permission_mode", "ask")
            species         = msg.get("species", "generalist")
            if provider_cfg.get("api_key"):
                asyncio.create_task(
                    self.spawn_worker(task_id, task_title, task_brief, provider_cfg,
                                      permission_mode, species)
                )

        elif msg["type"] == "permission_response":
            wid    = msg.get("worker_id", "")
            answer = msg.get("answer", "yes")
            if wid in self.workers:
                self.workers[wid].resolve_permission(answer)

        elif msg["type"] == "stop_worker":
            wid = msg.get("worker_id", "")
            if wid == "manager":
                if self.manager_task and not self.manager_task.done():
                    self.manager_task.cancel()
                await self.broadcast({"type": "agent_stopped", "worker_id": "manager"})
            elif wid in self.worker_tasks:
                self.worker_tasks[wid].cancel()
                await self.broadcast({"type": "agent_stopped", "worker_id": wid})

        elif msg["type"] == "stop_all":
            if self.manager_task and not self.manager_task.done():
                self.manager_task.cancel()
            for t in list(self.worker_tasks.values()):
                t.cancel()
            self.worker_tasks.clear()
            await self.broadcast({"type": "all_stopped"})

    def cancel_all(self):
        if self.manager_task and not self.manager_task.done():
            self.manager_task.cancel()
        for t in list(self.worker_tasks.values()):
            t.cancel()


# ---------------------------------------------------------------------------
# Global rooms registry
# ---------------------------------------------------------------------------

rooms: dict[str, Room] = {}


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    rooms["default"] = Room("default", "Default", DATA_DIR)
    yield


app = FastAPI(title="CC Team Chat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def configure(data_dir: Path) -> None:
    global DATA_DIR
    DATA_DIR = data_dir.resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Base dir: {DATA_DIR}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/api/rooms")
async def list_rooms():
    return [
        {"room_id": r.room_id, "name": r.name, "path": str(r.data_dir)}
        for r in rooms.values()
    ]


@app.post("/api/rooms")
async def create_room(body: dict):
    name = (body.get("name") or "New Project").strip()
    path = (body.get("path") or "").strip()

    room_id = f"room_{int(datetime.datetime.now().timestamp() * 1000)}"

    if path:
        data_dir = Path(path)
        if not data_dir.is_absolute():
            data_dir = BASE_DIR / data_dir
    else:
        slug = re.sub(r"[^\w\-]", "_", name.lower())[:30]
        data_dir = DATA_DIR / "rooms" / slug

    room = Room(room_id, name, data_dir)
    rooms[room_id] = room
    return {"room_id": room_id, "name": name, "path": str(room.data_dir)}


@app.delete("/api/rooms/{room_id}")
async def delete_room(room_id: str):
    if room_id == "default":
        return {"error": "Cannot delete the default room"}
    room = rooms.pop(room_id, None)
    if room:
        room.cancel_all()
    return {"status": "deleted"}


@app.get("/api/species")
async def get_species():
    return {
        sid: {"name": s["name"], "description": s["description"], "skills": s["skills"]}
        for sid, s in WORKER_SPECIES.items()
    }


@app.post("/shutdown")
async def shutdown():
    async def _kill():
        await asyncio.sleep(0.5)
        os._exit(0)
    asyncio.create_task(_kill())
    return {"status": "shutting down"}


@app.websocket("/ws/{room_id}")
async def ws_endpoint(ws: WebSocket, room_id: str):
    room = rooms.get(room_id)
    if not room:
        await ws.close(code=4004)
        return

    await ws.accept()
    room.connections.append(ws)

    await ws.send_text(json.dumps({
        "type":         "init",
        "room_id":      room.room_id,
        "room_name":    room.name,
        "project_path": str(room.data_dir),
        "workers": [
            {
                "worker_id":      wid,
                "task_title":     w.task_title,
                "color":          WORKER_COLORS[i % len(WORKER_COLORS)],
                "label":          w._label(),
                "species":        w.species,
                "species_name":   WORKER_SPECIES.get(w.species, {}).get("name", ""),
                "permission_mode": w.permission_mode,
            }
            for i, (wid, w) in enumerate(room.workers.items())
        ],
    }))

    try:
        while True:
            raw = await ws.receive_text()
            await room.handle_message(json.loads(raw), ws)
    except WebSocketDisconnect:
        if ws in room.connections:
            room.connections.remove(ws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CC Team Chat")
    parser.add_argument(
        "--project", default="userdata",
        help="Path to base project data directory (default: ./userdata)",
    )
    args = parser.parse_args()

    data_dir = Path(args.project)
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir

    configure(data_dir)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
