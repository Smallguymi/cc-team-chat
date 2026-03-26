"""
LLM Provider abstraction.

Supports:
  - Anthropic Claude (native SDK)
  - Any OpenAI-compatible endpoint:
      Alibaba Bailian, Google Gemini, OpenAI, DeepSeek, custom, etc.

All providers expose a single method:
    stream_turn(messages, system, tools, on_delta) -> StreamResult

Messages are stored in a normalized format (see below) so agents.py
never touches provider-specific types.

Normalized message format:
  {"role": "user",      "content": "..."}
  {"role": "assistant", "content": "..."}
  {"role": "assistant", "text": "...", "tool_calls": [{"id":"","name":"","input":{}}]}
  {"role": "tool",      "tool_call_id": "...", "name": "...", "content": "..."}
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Awaitable


# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class StreamResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ProviderConfig:
    provider: str    # "anthropic" | "openai_compat"
    api_key: str
    model: str
    base_url: str = ""


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider:
    def __init__(self, cfg: ProviderConfig):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=cfg.api_key)
        self.model  = cfg.model

    def _to_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        """Convert normalized messages → Anthropic API format."""
        result: list[dict] = []
        for msg in messages:
            role = msg["role"]

            if role == "user" and isinstance(msg.get("content"), str):
                result.append({"role": "user", "content": msg["content"]})

            elif role == "assistant" and "tool_calls" not in msg:
                result.append({"role": "assistant", "content": msg.get("content", "")})

            elif role == "assistant" and "tool_calls" in msg:
                blocks: list[dict] = []
                if msg.get("text"):
                    blocks.append({"type": "text", "text": msg["text"]})
                for tc in msg["tool_calls"]:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    })
                result.append({"role": "assistant", "content": blocks})

            elif role == "tool":
                # Batch consecutive tool results into one user message
                if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                    result[-1]["content"].append({
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    })
                else:
                    result.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": msg["content"],
                        }],
                    })
        return result

    async def stream_turn(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        on_delta: Callable[[str], Awaitable[None]],
        max_tokens: int = 4096,
    ) -> StreamResult:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=self._to_anthropic_messages(messages),
        )
        if tools:
            kwargs["tools"] = tools   # already in Anthropic format

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if (
                    event.type == "content_block_delta"
                    and hasattr(event.delta, "text")
                ):
                    await on_delta(event.delta.text)
            final = await stream.get_final_message()

        text = ""
        tool_calls: list[ToolCall] = []
        for block in final.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return StreamResult(text=text, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# OpenAI-compatible provider  (Bailian, Gemini, OpenAI, DeepSeek, custom…)
# ---------------------------------------------------------------------------

class OpenAICompatProvider:
    def __init__(self, cfg: ProviderConfig):
        from openai import AsyncOpenAI
        kw: dict = {"api_key": cfg.api_key}
        if cfg.base_url:
            kw["base_url"] = cfg.base_url
        self.client = AsyncOpenAI(**kw)
        self.model  = cfg.model

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Anthropic tool format → OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],   # JSON Schema is compatible
                },
            }
            for t in tools
        ]

    def _to_openai_messages(self, system: str, messages: list[dict]) -> list[dict]:
        """Convert normalized messages → OpenAI API format."""
        result: list[dict] = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg["role"]

            if role == "user":
                result.append({"role": "user", "content": msg.get("content", "")})

            elif role == "assistant" and "tool_calls" not in msg:
                result.append({"role": "assistant", "content": msg.get("content", "")})

            elif role == "assistant" and "tool_calls" in msg:
                result.append({
                    "role": "assistant",
                    "content": msg.get("text", ""),
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"]),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ],
                })

            elif role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"],
                })
        return result

    async def stream_turn(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        on_delta: Callable[[str], Awaitable[None]],
        max_tokens: int = 4096,
    ) -> StreamResult:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=self._to_openai_messages(system, messages),
            stream=True,
        )
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        full_text = ""
        tc_acc: dict[int, dict] = {}   # index → {id, name, arguments}

        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content
                await on_delta(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_acc:
                        tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tc_acc[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tc_acc[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tc_acc[idx]["arguments"] += tc.function.arguments

        tool_calls: list[ToolCall] = []
        for idx in sorted(tc_acc):
            tc = tc_acc[idx]
            try:
                inp = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                inp = {}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], input=inp))

        return StreamResult(text=full_text, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Claude CLI provider  (uses `claude -p` — no API key needed)
# ---------------------------------------------------------------------------

class ClaudeCLIProvider:
    """Shells out to the local `claude` CLI (Claude Code Pro subscription)."""

    def __init__(self, cfg: ProviderConfig):
        self.model = cfg.model  # passed as --model flag if set

    def _build_prompt(self, system: str, messages: list[dict]) -> str:
        """Flatten system + last N messages into a single prompt string."""
        parts: list[str] = []
        if system:
            parts.append(f"[System Context]\n{system}")
        # Keep last 10 messages to stay well within CLI argument limits
        for msg in messages[-10:]:
            role = msg["role"]
            if role == "user":
                parts.append(f"[User]\n{msg.get('content', '')}")
            elif role == "assistant":
                text = msg.get("content") or msg.get("text", "")
                parts.append(f"[Assistant]\n{text}")
            # tool roles skipped — not supported in CLI mode
        return "\n\n---\n\n".join(parts)

    async def stream_turn(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        on_delta: Callable[[str], Awaitable[None]],
        max_tokens: int = 4096,
    ) -> StreamResult:
        prompt = self._build_prompt(system, messages)

        # On Windows, `claude` is a .cmd file — must run via cmd.exe
        if sys.platform == "win32":
            cmd = ["cmd", "/c", "claude", "-p", prompt]
        else:
            cmd = ["claude", "-p", prompt]
        if self.model:
            cmd += ["--model", self.model]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        full_text = ""
        while True:
            chunk = await proc.stdout.read(512)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            full_text += text
            await on_delta(text)

        await proc.wait()

        if proc.returncode != 0:
            err = (await proc.stderr.read()).decode("utf-8", errors="replace")
            raise RuntimeError(f"claude CLI exited {proc.returncode}: {err[:300]}")

        return StreamResult(text=full_text.strip(), tool_calls=[])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_provider(config: dict):
    """
    Build a provider from a dict sent by the frontend, e.g.:
      {
        "provider": "anthropic",
        "api_key": "sk-ant-...",
        "model":   "claude-opus-4-6",
        "base_url": ""
      }
    """
    cfg = ProviderConfig(
        provider = config.get("provider", "anthropic"),
        api_key  = config.get("api_key", ""),
        model    = config.get("model", "claude-opus-4-6"),
        base_url = config.get("base_url", ""),
    )
    if cfg.provider == "anthropic":
        return AnthropicProvider(cfg)
    if cfg.provider == "claude_cli":
        return ClaudeCLIProvider(cfg)
    return OpenAICompatProvider(cfg)
