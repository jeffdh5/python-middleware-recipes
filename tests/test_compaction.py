# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import genkit._core._typing as genkit_typing

if not hasattr(genkit_typing, 'Artifact'):
    pytest.skip('requires genkit with session Artifact support', allow_module_level=True)

from genkit import ModelRequest, ModelResponse
from genkit._core._model import GenerateActionOptions, Message
from genkit._core._registry import Registry
from genkit._core._typing import (
    Part,
    Role,
    ToolRequest,
    ToolRequestPart,
)
from genkit.middleware import GenerateHookParams, GenerateMiddlewareContext, MultipartToolResponse, ToolHookParams

from recipes.compaction.compaction import (
    Compaction,
    _compact_prefix,
    _determine_cutoff_index,
    _head_tail_preview,
    _should_summarize,
    _truncate_tool_input,
)
from tests.conftest import MockSession


def _make_params(messages: list[Message]) -> GenerateHookParams:
    opts = GenerateActionOptions(messages=messages)
    return GenerateHookParams(
        options=opts,
        request=ModelRequest(messages=list(messages)),
        iteration=1,
    )


def test_truncate_tool_input_content_field() -> None:
    big = 'x' * 5000
    out = _truncate_tool_input(
        {'file_path': 'a.py', 'content': big},
        max_chars=400,
        preview_chars=80,
        suffix='…[truncated]',
        bulk_keys=frozenset({'content'}),
    )
    assert isinstance(out, dict)
    assert out['file_path'] == 'a.py'
    assert len(out['content']) < len(big)
    assert '…[truncated]' in out['content']


def test_compact_prefix_clips_old_write_file_args() -> None:
    big = 'x' * 5000
    old = Message(
        role=Role.MODEL,
        content=[
            Part(
                root=ToolRequestPart(
                    tool_request=ToolRequest(name='write_file', input={'file_path': 'a.py', 'content': big}),
                )
            )
        ],
    )
    recent = Message(role=Role.USER, content=[Part(text='latest task')])
    cfg = Compaction(
        max_context_tokens=None,
        keep_recent_messages=1,
        max_tool_input_chars=400,
        preview_chars=80,
    ).config
    cutoff = _determine_cutoff_index([old, recent], cfg)
    compacted = _compact_prefix([old, recent], cutoff, cfg)
    clipped = compacted[0].content[0].root.tool_request.input['content']
    assert len(clipped) < len(big)
    assert '…[truncated]' in clipped


def test_head_tail_preview() -> None:
    text = '\n'.join(f'line {i}' for i in range(20))
    preview = _head_tail_preview(text, head_lines=2, tail_lines=2)
    assert 'line 0' in preview
    assert 'line 19' in preview
    assert 'lines truncated' in preview
    assert 'line 10' not in preview


def test_should_summarize_at_fraction_threshold() -> None:
    cfg = Compaction(max_context_tokens=1000, trigger_fraction=0.5).config
    fat = Message(role=Role.USER, content=[Part(text='word ' * 3000)])
    assert _should_summarize([fat], cfg)


@pytest.mark.asyncio
async def test_wrap_tool_offloads_large_output_to_artifact(ctx: GenerateMiddlewareContext) -> None:
    mw = Compaction(offload_tool_threshold_chars=200, preview_head_lines=2, preview_tail_lines=2)
    session = MockSession(session_id='test-session')
    ctx.session = session

    async def next_fn(_params, _ctx):
        return MultipartToolResponse(output='line\n' * 800)

    from genkit._ai._tools import define_tool

    scratch = Registry()

    async def noop() -> str:
        return ''

    tool = define_tool(scratch, noop, name='execute').action()
    tr = ToolRequest(name='execute', input={'command': 'npm test'}, ref='abc')
    params = ToolHookParams(tool_request_part=ToolRequestPart(tool_request=tr), tool=tool)

    result = await mw.wrap_tool(params, ctx, next_fn)
    assert 'read_artifact' in result.output
    assert 'lines truncated' in result.output
    arts = await session.get_artifacts()
    assert len(arts) == 1
    assert arts[0].name.startswith('tool-output/execute/')


@pytest.mark.asyncio
async def test_summarization_replaces_prefix_and_logs_transcript(ctx: GenerateMiddlewareContext) -> None:
    async def fake_summarizer(messages, *, ctx):  # noqa: ANN001, ARG001
        return '## SESSION INTENT\nFix auth test\n## SUMMARY\nRead auth.py and patched login.'

    mw = Compaction(
        summarizer=fake_summarizer,
        max_context_tokens=500,
        trigger_fraction=0.3,
        keep_fraction=0.2,
        keep_recent_messages=1,
    )
    session = MockSession(session_id='sess-1')
    ctx.session = session

    old = Message(role=Role.USER, content=[Part(text='fix auth ' * 200)])
    mid = Message(role=Role.MODEL, content=[Part(text='reading files ' * 200)])
    recent = Message(role=Role.USER, content=[Part(text='latest')])

    captured: dict[str, list[Message]] = {}

    async def next_fn(params, _ctx):
        captured['messages'] = list(params.options.messages)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part(text='ok')]))

    await mw.wrap_generate(_make_params([old, mid, recent]), ctx, next_fn)

    msgs = captured['messages']
    assert msgs[0].metadata.get('compaction-summary') is True
    assert 'conversation-history/sess-1.md' in msgs[0].content[0].root.text
    assert 'Fix auth test' in msgs[0].content[0].root.text
    assert msgs[-1].content[0].root.text == 'latest'
    assert len(msgs) == 3  # summary + mid + recent (keep window retained mid)

    arts = await session.get_artifacts()
    assert len(arts) == 1
    assert arts[0].name == 'conversation-history/sess-1.md'
    assert 'fix auth' in arts[0].parts[0].root.text


@pytest.mark.asyncio
async def test_read_file_excluded_from_tool_offload(ctx: GenerateMiddlewareContext) -> None:
    mw = Compaction(offload_tool_threshold_chars=10)
    huge = 'x' * 5000

    async def next_fn(_params, _ctx):
        return MultipartToolResponse(output=huge)

    from genkit._ai._tools import define_tool

    scratch = Registry()

    async def read_file_tool() -> str:
        return huge

    tool = define_tool(scratch, read_file_tool, name='read_file').action()
    tr = ToolRequest(name='read_file', input={'file_path': 'a.py'}, ref='r1')
    params = ToolHookParams(tool_request_part=ToolRequestPart(tool_request=tr), tool=tool)

    result = await mw.wrap_tool(params, ctx, next_fn)
    assert result.output == huge
