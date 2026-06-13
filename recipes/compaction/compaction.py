# Copyright 2026 Jeff Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Layered compaction middleware for Genkit coding-agent harnesses.

Structural layers (cheap, lossless where files stay on disk):
  - offload large tool outputs to session artifacts
  - clip bulky tool-call arguments in older turns
  - truncate oversized tool responses outside the keep window

Summarization layer (when context budget is still exceeded):
  - append evicted transcript to a conversation log artifact
  - replace the evicted prefix with an LLM-written summary + log pointer

Pair with official ``Filesystem`` and ``Artifacts`` from ``genkit.plugins.middleware``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from genkit._core._action import ActionKind
from genkit._core._model import Message, ModelRequest, text_from_content
from genkit._core._typing import (
    Artifact,
    Part,
    Role,
    TextPart,
    ToolRequestPart,
    ToolResponsePart,
)
from genkit.middleware import (
    BaseMiddleware,
    GenerateHookParams,
    GenerateMiddlewareContext,
    MultipartToolResponse,
    ToolHookParams,
)

_ARTIFACT_SOURCE = 'compaction-middleware'
_SUMMARY_SOURCE = 'compaction-summary'
_CHARS_PER_TOKEN = 4
_DEFAULT_BULK_INPUT_KEYS = frozenset(
    {'content', 'old_string', 'new_string', 'command', 'code', 'patch', 'diff'}
)
_TOOLS_EXCLUDED_FROM_OFFLOAD = frozenset({'read_file', 'write_file', 'edit_file', 'list_files'})

_SUMMARY_PROMPT = """\
<instructions>
Extract the context needed to continue this coding-agent session without the full
message history. Structure the result with these sections (use "None" when empty):

## SESSION INTENT
## SUMMARY
## ARTIFACTS
## NEXT STEPS
</instructions>

Messages to summarize:
{messages}

Respond ONLY with the structured summary.
"""


class Summarizer(Protocol):
    """Async callable that turns evicted messages into a summary string."""

    async def __call__(self, messages: list[Message], *, ctx: GenerateMiddlewareContext) -> str: ...


class CompactionConfig(BaseModel):
    """Knobs for coding-harness context compaction."""

    max_tool_output_chars: int = Field(
        default=1500,
        description='Inline cap for a single tool result outside the keep window.',
    )
    max_tool_input_chars: int = Field(
        default=400,
        description='Per-field cap when trimming tool arguments in older messages.',
    )
    keep_recent_messages: int = Field(
        default=6,
        description='Fallback keep window (message count) when token budget is unset.',
    )
    max_context_tokens: int | None = Field(
        default=200_000,
        description='Approximate model context size used for fraction triggers.',
    )
    trigger_fraction: float = Field(
        default=0.85,
        description='Summarize when estimated tokens reach this fraction of max_context_tokens.',
    )
    keep_fraction: float = Field(
        default=0.10,
        description='Fraction of max_context_tokens to preserve verbatim at the tail.',
    )
    preview_chars: int = Field(
        default=120,
        description='Characters kept at the start of a truncated inline field.',
    )
    preview_head_lines: int = Field(
        default=5,
        description='Lines shown from the start of an offloaded tool result.',
    )
    preview_tail_lines: int = Field(
        default=5,
        description='Lines shown from the end of an offloaded tool result.',
    )
    offload_large_outputs: bool = Field(
        default=True,
        description='When a session is available, store full tool output in an artifact.',
    )
    offload_tool_threshold_chars: int = Field(
        default=80_000,
        description='Tool results larger than this (~20k tokens) are offloaded immediately.',
    )
    enable_summarization: bool = Field(
        default=True,
        description='When true and a summarizer/model is configured, run LLM compaction at the trigger.',
    )
    summary_model: str | None = Field(
        default=None,
        description='Registry model name for summarization (use a cheap/fast model).',
    )
    trim_summary_input_tokens: int = Field(
        default=4000,
        description='Max estimated tokens from evicted messages sent to the summary model.',
    )
    conversation_log_prefix: str = Field(
        default='conversation-history',
        description='Artifact name prefix for appended conversation logs.',
    )
    bulk_input_keys: list[str] = Field(
        default_factory=lambda: sorted(_DEFAULT_BULK_INPUT_KEYS),
        description='Tool input dict keys likely to carry file bodies or patches.',
    )
    truncation_suffix: str = Field(
        default='…[truncated]',
        description='Appended after clipped tool arguments and outputs.',
    )


def _as_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _message_text(msg: Message) -> str:
    return text_from_content(msg.content)


def _approx_tokens(messages: Sequence[Message]) -> int:
    return sum(len(_message_text(m)) for m in messages) // _CHARS_PER_TOKEN


def _truncate_text(text: str, max_chars: int, preview_chars: int, suffix: str) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:preview_chars].rstrip()
    return f'{head}{suffix} ({len(text):,} chars total)'


def _head_tail_preview(
    text: str,
    *,
    head_lines: int,
    tail_lines: int,
) -> str:
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return '\n'.join(line[:1000] for line in lines)
    head = '\n'.join(line[:1000] for line in lines[:head_lines])
    tail = '\n'.join(line[:1000] for line in lines[-tail_lines:])
    omitted = len(lines) - head_lines - tail_lines
    return f'{head}\n... [{omitted} lines truncated] ...\n{tail}'


def _truncate_tool_input(
    tool_input: Any,
    *,
    max_chars: int,
    preview_chars: int,
    suffix: str,
    bulk_keys: frozenset[str],
) -> Any:
    if not isinstance(tool_input, dict):
        return tool_input
    out = deepcopy(tool_input)
    for key in bulk_keys:
        if key not in out:
            continue
        raw = out[key]
        if not isinstance(raw, str):
            continue
        out[key] = _truncate_text(raw, max_chars, preview_chars, suffix)
    return out


def _determine_cutoff_index(messages: list[Message], cfg: CompactionConfig) -> int:
    """Index where the keep window starts. Messages before this may be compacted."""
    if not messages:
        return 0

    max_tokens = cfg.max_context_tokens
    if max_tokens is not None and max_tokens > 0:
        target = max(1, int(max_tokens * cfg.keep_fraction))
        kept = 0
        for i in range(len(messages) - 1, -1, -1):
            kept += max(1, len(_message_text(messages[i])) // _CHARS_PER_TOKEN)
            if kept >= target:
                return i
        return 0

    keep = cfg.keep_recent_messages
    if len(messages) <= keep:
        return len(messages)
    return len(messages) - keep


def _should_summarize(messages: list[Message], cfg: CompactionConfig) -> bool:
    if not cfg.enable_summarization:
        return False
    max_tokens = cfg.max_context_tokens
    if max_tokens is None or max_tokens <= 0:
        return False
    threshold = int(max_tokens * cfg.trigger_fraction)
    return _approx_tokens(messages) >= threshold


def _format_transcript(messages: list[Message]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
        body = _message_text(msg).strip()
        if not body:
            continue
        lines.append(f'{role}: {body}')
    return '\n\n'.join(lines)


def _compact_prefix(
    messages: list[Message],
    cutoff: int,
    cfg: CompactionConfig,
) -> list[Message]:
    """Clip bulky fields in messages before the keep window."""
    if cutoff <= 0 or cutoff >= len(messages):
        return list(messages)

    bulk = frozenset(cfg.bulk_input_keys)
    compacted: list[Message] = []
    for idx, msg in enumerate(messages):
        if idx >= cutoff:
            compacted.append(msg)
            continue

        new_parts: list[Part] = []
        for part in msg.content:
            root = part.root
            if isinstance(root, ToolRequestPart):
                tr = root.tool_request
                new_input = _truncate_tool_input(
                    tr.input,
                    max_chars=cfg.max_tool_input_chars,
                    preview_chars=cfg.preview_chars,
                    suffix=cfg.truncation_suffix,
                    bulk_keys=bulk,
                )
                new_parts.append(
                    Part(
                        root=ToolRequestPart(
                            tool_request=tr.model_copy(update={'input': new_input}),
                            metadata=root.metadata,
                        )
                    )
                )
                continue

            if isinstance(root, ToolResponsePart):
                tr = root.tool_response
                output_text = _as_text(tr.output)
                if len(output_text) > cfg.max_tool_output_chars:
                    output_text = _truncate_text(
                        output_text,
                        cfg.max_tool_output_chars,
                        cfg.preview_chars,
                        cfg.truncation_suffix,
                    )
                    new_parts.append(
                        Part(
                            root=ToolResponsePart(
                                tool_response=tr.model_copy(update={'output': output_text}),
                                metadata=root.metadata,
                            )
                        )
                    )
                    continue

            if isinstance(root, TextPart) and isinstance(root.text, str):
                if len(root.text) > cfg.max_tool_output_chars:
                    new_parts.append(
                        Part(
                            root=TextPart(
                                text=_truncate_text(
                                    root.text,
                                    cfg.max_tool_output_chars,
                                    cfg.preview_chars,
                                    cfg.truncation_suffix,
                                ),
                                metadata=root.metadata,
                            )
                        )
                    )
                    continue

            new_parts.append(part)

        compacted.append(Message(role=msg.role, content=new_parts, metadata=msg.metadata))

    return compacted


def _build_summary_message(summary: str, log_path: str | None) -> Message:
    if log_path:
        text = (
            'You are in the middle of a conversation that has been summarized.\n\n'
            f'The full conversation history has been saved to artifact "{log_path}" '
            'should you need to refer back to it for details.\n\n'
            f'<summary>\n{summary}\n</summary>'
        )
    else:
        text = f'Here is a summary of the conversation to date:\n\n{summary}'

    return Message(
        role=Role.USER,
        content=[Part(text=text)],
        metadata={_SUMMARY_SOURCE: True},
    )


async def _session_log_name(ctx: GenerateMiddlewareContext, cfg: CompactionConfig) -> str:
    session_id = 'session'
    if ctx.session is not None:
        state = await ctx.session.state()
        if state.session_id:
            session_id = state.session_id
    return f'{cfg.conversation_log_prefix}/{session_id}.md'


async def _read_artifact_text(ctx: GenerateMiddlewareContext, name: str) -> str:
    if ctx.session is None:
        return ''
    for art in await ctx.session.get_artifacts():
        if art.name == name:
            return _as_text(_extract_artifact_text(art))
    return ''


def _extract_artifact_text(artifact: Artifact) -> str:
    parts: list[str] = []
    for part in artifact.parts:
        root = part.root
        if isinstance(root, TextPart) and root.text:
            parts.append(root.text)
    return '\n'.join(parts)


async def _append_conversation_log(
    ctx: GenerateMiddlewareContext,
    log_name: str,
    evicted: list[Message],
) -> bool:
    if ctx.session is None:
        return False

    timestamp = datetime.now(UTC).isoformat()
    section = f'## Compacted at {timestamp}\n\n{_format_transcript(evicted)}\n\n'
    existing = await _read_artifact_text(ctx, log_name)
    await ctx.session.add_artifacts(
        Artifact(
            name=log_name,
            parts=[Part(text=existing + section)],
            metadata={'source': _ARTIFACT_SOURCE, 'kind': 'conversation-log'},
        )
    )
    return True


def _trim_messages_for_summary(messages: list[Message], max_tokens: int) -> list[Message]:
    """Keep the tail of evicted messages that fits the summary model budget."""
    if not messages:
        return messages

    budget_chars = max_tokens * _CHARS_PER_TOKEN
    total = 0
    start = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        total += len(_message_text(messages[i]))
        if total > budget_chars:
            break
        start = i
    return messages[start:]


class Compaction(BaseMiddleware[CompactionConfig]):
    """Layered compaction: structural clipping, tool offload, optional LLM summarization."""

    def __init__(
        self,
        *,
        summarizer: Summarizer | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self._summarizer = summarizer

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[Any]],
    ) -> Any:
        cfg = self.config
        msgs = list(params.options.messages)
        cutoff = _determine_cutoff_index(msgs, cfg)
        msgs = _compact_prefix(msgs, cutoff, cfg)

        if _should_summarize(msgs, cfg):
            cutoff = _determine_cutoff_index(msgs, cfg)
            if cutoff > 0:
                msgs = await self._summarize_and_replace(msgs, cutoff, ctx)

        new_options = params.options.model_copy(update={'messages': msgs})
        new_request = params.request.model_copy(update={'messages': msgs})
        new_params = params.model_copy(update={'options': new_options, 'request': new_request})
        return await next_fn(new_params, ctx)

    async def _summarize_and_replace(
        self,
        messages: list[Message],
        cutoff: int,
        ctx: GenerateMiddlewareContext,
    ) -> list[Message]:
        evicted = [m for m in messages[:cutoff] if not (m.metadata or {}).get(_SUMMARY_SOURCE)]
        kept = messages[cutoff:]
        if not evicted:
            return messages

        log_name = await _session_log_name(ctx, self.config)
        log_ok = await _append_conversation_log(ctx, log_name, evicted)
        log_path = log_name if log_ok else None

        summary = await self._create_summary(evicted, ctx)
        return [_build_summary_message(summary, log_path), *kept]

    async def _create_summary(self, messages: list[Message], ctx: GenerateMiddlewareContext) -> str:
        trimmed = _trim_messages_for_summary(messages, self.config.trim_summary_input_tokens)
        transcript = _format_transcript(trimmed)

        if self._summarizer is not None:
            return await self._summarizer(trimmed, ctx=ctx)

        model_name = self.config.summary_model
        if not model_name:
            return (
                '## SESSION INTENT\n(automatic summary unavailable — no summary_model configured)\n\n'
                f'## SUMMARY\n{_truncate_text(transcript, 2000, 500, self.config.truncation_suffix)}'
            )

        action = await ctx.registry.resolve_action(ActionKind.MODEL, model_name)
        if action is None:
            return (
                f'## SUMMARY\nSummary model {model_name!r} not found in registry.\n\n'
                f'{_truncate_text(transcript, 2000, 500, self.config.truncation_suffix)}'
            )

        prompt = _SUMMARY_PROMPT.format(messages=transcript)
        result = await action.run(
            ModelRequest(
                messages=[Message(role=Role.USER, content=[Part(text=prompt)])],
            )
        )
        response = result.response
        if getattr(response, 'message', None) is not None:
            return _message_text(response.message).strip()
        return _as_text(response).strip()

    async def wrap_tool(
        self,
        params: ToolHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
    ) -> MultipartToolResponse:
        result = await next_fn(params, ctx)
        cfg = self.config
        tool_name = params.tool.name
        if tool_name in _TOOLS_EXCLUDED_FROM_OFFLOAD:
            return result

        text = _as_text(result.output)
        if len(text) <= cfg.offload_tool_threshold_chars:
            return result

        tool_name = params.tool.name
        ref = params.tool_request_part.tool_request.ref or uuid.uuid4().hex[:8]
        artifact_name = f'tool-output/{tool_name}/{ref}.txt'

        if cfg.offload_large_outputs and ctx.session is not None:
            await ctx.session.add_artifacts(
                Artifact(
                    name=artifact_name,
                    parts=[Part(text=text)],
                    metadata={'source': _ARTIFACT_SOURCE, 'tool': tool_name, 'ref': ref},
                )
            )
            preview = _head_tail_preview(
                text,
                head_lines=cfg.preview_head_lines,
                tail_lines=cfg.preview_tail_lines,
            )
            compact = (
                f'Tool result too large; full output saved to artifact "{artifact_name}". '
                f'Use read_artifact to retrieve (paginate if needed).\n\n'
                f'Preview:\n{preview}'
            )
            return MultipartToolResponse(
                output=compact,
                content=result.content,
                metadata=result.metadata,
            )

        compact = _truncate_text(
            text,
            cfg.max_tool_output_chars,
            cfg.preview_chars,
            cfg.truncation_suffix,
        )
        return MultipartToolResponse(
            output=compact,
            content=result.content,
            metadata=result.metadata,
        )
