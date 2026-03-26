"""
CC Team Chat — FastAPI server.

WebSocket message protocol (client → server):
  { "type": "user_message",  "content": "...", "provider": {...} }
  { "type": "worker_message","worker_id": "...", "content": "...", "provider": {...} }
  { "type": "spawn_worker",  "task_id":"...","task_title":"...","task_brief":"...","provider":{...} }

provider object:
  { "provider": "anthropic"|"openai_compat", "api_key": "...", "model": "...", "base_url": "..." }
"""

import asyncio
import datetime
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents import ManagerAgent, WorkerAgent, WORKER_SPECIES

app = FastAPI(title="CC Team Chat")

BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Session log
# ---------------------------------------------------------------------------

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
_session_ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
LOG_FILE     = LOG_DIR / f"session_{_session_ts}.txt"
_stream_acc: dict[str, str] = {}   # sender → accumulated stream text

def _log(message: dict):
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    typ = message.get("type", "")

    if typ == "message":
        label   = message.get("sender_label") or message.get("sender", "unknown")
        content = message.get("content", "")
        line    = f"\n[{ts}] {label}:\n{content}\n{'─'*60}\n"

    elif typ == "stream_delta":
        sender = message.get("sender", "")
        _stream_acc[sender] = _stream_acc.get(sender, "") + message.get("content", "")
        return  # don't write yet

    elif typ == "stream_end":
        sender = message.get("sender", "")
        text   = _stream_acc.pop(sender, "")
        if not text:
            return
        label  = "Manager CC" if sender == "manager" else f"Worker ({sender})"
        line   = f"\n[{ts}] {label}:\n{text}\n{'─'*60}\n"

    elif typ == "worker_spawned":
        line = f"\n[{ts}] --- Worker spawned: {message.get('task_title','')} ---\n"

    elif typ == "worker_complete":
        line = f"\n[{ts}] --- Worker complete: {message.get('task_title','')} ---\n"

    else:
        return

    LOG_FILE.open("a", encoding="utf-8").write(line)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

connections:    list[WebSocket]        = []
manager:        ManagerAgent | None    = None
workers:        dict[str, WorkerAgent] = {}
worker_counter: int                    = 0
worker_tasks:   dict[str, asyncio.Task] = {}   # worker_id → running task
manager_task:   asyncio.Task | None     = None

WORKER_COLORS = [
    "#2ECC71", "#E67E22", "#E74C3C", "#3498DB",
    "#1ABC9C", "#F39C12", "#8E44AD", "#16A085",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def broadcast(message: dict):
    _log(message)
    data = json.dumps(message)
    dead = []
    for ws in connections:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)


async def spawn_worker(
    task_id: str, task_title: str, task_brief: str, provider_cfg: dict,
    permission_mode: str = "ask",
    species: str = "generalist",
):
    global worker_counter
    worker_counter += 1
    worker_id = f"worker_{worker_counter}"
    color     = WORKER_COLORS[(worker_counter - 1) % len(WORKER_COLORS)]
    label     = "Worker — " + (task_title[:25] + "…" if len(task_title) > 25 else task_title)

    species_cfg  = WORKER_SPECIES.get(species, WORKER_SPECIES["generalist"])
    species_name = species_cfg["name"]

    worker = WorkerAgent(
        worker_id=worker_id,
        task_id=task_id,
        task_title=task_title,
        task_brief=task_brief,
        broadcast_fn=broadcast,
        permission_mode=permission_mode,
        species=species,
    )
    workers[worker_id] = worker

    await broadcast({
        "type": "worker_spawned",
        "worker_id": worker_id,
        "task_id": task_id,
        "task_title": task_title,
        "color": color,
        "label": label,
        "permission_mode": permission_mode,
        "species": species,
        "species_name": species_name,
    })

    async def on_complete(wid: str, title: str, result: str):
        worker_tasks.pop(wid, None)
        await broadcast({"type": "worker_complete", "worker_id": wid, "task_title": title})
        if manager:
            await manager.receive_worker_result(wid, title, result, provider_cfg, spawn_worker)

    task = asyncio.create_task(worker.start(provider_cfg, on_complete))
    worker_tasks[worker_id] = task
    task.add_done_callback(lambda t: worker_tasks.pop(worker_id, None))

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    global manager
    manager = ManagerAgent(broadcast_fn=broadcast)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/species")
async def get_species():
    return {
        sid: {
            "name":        s["name"],
            "description": s["description"],
            "skills":      s["skills"],
        }
        for sid, s in WORKER_SPECIES.items()
    }


@app.post("/shutdown")
async def shutdown():
    async def _kill():
        await asyncio.sleep(0.5)   # let the response reach the browser first
        os._exit(0)
    asyncio.create_task(_kill())
    return {"status": "shutting down"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connections.append(ws)

    await ws.send_text(json.dumps({
        "type": "init",
        "workers": [
            {
                "worker_id": wid,
                "task_title": w.task_title,
                "color": WORKER_COLORS[i % len(WORKER_COLORS)],
                "label": w._label(),
            }
            for i, (wid, w) in enumerate(workers.items())
        ],
    }))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            provider_cfg: dict = msg.get("provider", {})

            if msg["type"] == "user_message":
                content = msg.get("content", "").strip()
                if not content:
                    continue
                if not provider_cfg.get("api_key"):
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": "No API key — open Settings and configure your provider.",
                    }))
                    continue

                await broadcast({
                    "type": "message",
                    "sender": "user",
                    "sender_label": "You",
                    "content": content,
                    "complete": True,
                })

                if manager:
                    global manager_task
                    manager_task = asyncio.create_task(
                        manager.process_message(content, provider_cfg, spawn_worker)
                    )

            elif msg["type"] == "worker_message":
                worker_id = msg.get("worker_id", "")
                content   = msg.get("content", "").strip()
                if worker_id in workers and content and provider_cfg.get("api_key"):
                    asyncio.create_task(
                        workers[worker_id].process_message(content, provider_cfg)
                    )

            elif msg["type"] == "spawn_worker":
                task_title      = msg.get("task_title", "Manual task")
                task_brief      = msg.get("task_brief", "No brief provided.")
                task_id         = msg.get("task_id", f"manual_{worker_counter + 1:03d}")
                permission_mode = msg.get("permission_mode", "ask")
                species         = msg.get("species", "generalist")
                if provider_cfg.get("api_key"):
                    asyncio.create_task(
                        spawn_worker(task_id, task_title, task_brief, provider_cfg,
                                     permission_mode, species)
                    )

            elif msg["type"] == "permission_response":
                wid    = msg.get("worker_id", "")
                answer = msg.get("answer", "yes")
                if wid in workers:
                    workers[wid].resolve_permission(answer)

            elif msg["type"] == "stop_worker":
                wid = msg.get("worker_id", "")
                if wid == "manager":
                    if manager_task and not manager_task.done():
                        manager_task.cancel()
                    await broadcast({"type": "agent_stopped", "worker_id": "manager"})
                elif wid in worker_tasks:
                    worker_tasks[wid].cancel()
                    await broadcast({"type": "agent_stopped", "worker_id": wid})

            elif msg["type"] == "stop_all":
                if manager_task and not manager_task.done():
                    manager_task.cancel()
                for t in list(worker_tasks.values()):
                    t.cancel()
                worker_tasks.clear()
                await broadcast({"type": "all_stopped"})

    except WebSocketDisconnect:
        if ws in connections:
            connections.remove(ws)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
