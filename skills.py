"""
Skill registry for Worker CC agents.

Each skill exposes:
  - "tool"    : LLM tool schema (name / description / input_schema)
  - "handler" : async fn(inputs: dict) -> str

Add new skills by applying the @_register decorator.
"""

from pathlib import Path

SKILLS: dict = {}


def _register(name: str, description: str, properties: dict, required: list):
    def decorator(fn):
        SKILLS[name] = {
            "tool": {
                "name": name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
            "handler": fn,
        }
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Researcher skills
# ---------------------------------------------------------------------------

@_register(
    "web_search",
    "Search the web for current information. Returns top results with titles, URLs and snippets.",
    {"query": {"type": "string", "description": "Search query"}},
    ["query"],
)
async def web_search(inp: dict) -> str:
    try:
        from duckduckgo_search import DDGS
        results = list(DDGS().text(inp["query"], max_results=6))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"**{r['title']}**\n{r['href']}\n{r['body']}" for r in results
        )
    except ImportError:
        return "web_search requires: pip install duckduckgo-search"
    except Exception as e:
        return f"Search error: {e}"


# ---------------------------------------------------------------------------
# File Editor skills
# ---------------------------------------------------------------------------

@_register(
    "read_file",
    "Read the text contents of a file on disk.",
    {"path": {"type": "string", "description": "File path to read"}},
    ["path"],
)
async def read_file(inp: dict) -> str:
    try:
        return Path(inp["path"]).read_text(encoding="utf-8")
    except Exception as e:
        return f"Error: {e}"


@_register(
    "write_file",
    "Write text content to a file. Creates parent directories if needed.",
    {
        "path":    {"type": "string", "description": "File path to write"},
        "content": {"type": "string", "description": "Text content to write"},
    },
    ["path", "content"],
)
async def write_file(inp: dict) -> str:
    try:
        p = Path(inp["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(inp["content"], encoding="utf-8")
        return f"Wrote {len(inp['content'])} chars to {inp['path']}"
    except Exception as e:
        return f"Error: {e}"


@_register(
    "list_directory",
    "List files and folders in a directory.",
    {"path": {"type": "string", "description": "Directory path"}},
    ["path"],
)
async def list_directory(inp: dict) -> str:
    try:
        entries = sorted(Path(inp["path"]).iterdir(), key=lambda e: (e.is_file(), e.name))
        lines = [f"[{'FILE' if e.is_file() else ' DIR'}] {e.name}" for e in entries]
        return "\n".join(lines) if lines else "(empty)"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

async def execute(skill_name: str, inputs: dict) -> str:
    """Run a skill by name. Returns result string."""
    entry = SKILLS.get(skill_name)
    if not entry:
        return f"Unknown skill: {skill_name}"
    return await entry["handler"](inputs)


def get_tool_schema(skill_name: str) -> dict | None:
    entry = SKILLS.get(skill_name)
    return entry["tool"] if entry else None
