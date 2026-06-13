# Compaction

Middleware for coding agents that are running out of context. Works in layers — cheap structural clipping first, LLM summarization only when the budget is still tight.

Pair with official [`Filesystem`](https://github.com/firebase/genkit/tree/main/py/plugins/middleware) and [`Artifacts`](https://github.com/firebase/genkit/tree/main/py/plugins/middleware) from `genkit.plugins.middleware`.

## What it does

**Before each model call (`wrap_generate`):**

1. **Clip old tool traffic** — outside the keep window, shorten bulky `write_file` / `edit_file` arguments and truncate oversized tool responses.
2. **Summarize (optional)** — when estimated usage crosses `trigger_fraction` of `max_context_tokens`, append the evicted turns to a conversation-log artifact, then replace them with a short handoff note. The log path is embedded in the handoff so the agent can `read_artifact` the verbatim transcript later.

**After each tool call (`wrap_tool`):**

- Tool results above `offload_tool_threshold_chars` go to a session artifact with a head/tail sample inline. `read_file` / `write_file` / `edit_file` / `list_files` are excluded (they are already bounded or paginated).

## Quick start

```python
from recipes.compaction import Compaction
from genkit.plugins.middleware import Artifacts, Filesystem

await ai.generate(
    prompt='Fix the failing test in auth.py',
    use=[
        Filesystem(root_dir='./workspace'),
        Artifacts(),
        Compaction(
            max_context_tokens=200_000,
            trigger_fraction=0.85,
            keep_fraction=0.10,
            summary_model='googleai/gemini-flash-latest',
        ),
    ],
)
```

Tell your favorite coding agent to look at [`compaction.py`](compaction.py) and copy it into your app, then:

```python
from compaction import Compaction
```

## Custom summarizer

Pass a callable instead of `summary_model`:

```python
async def handoff(messages, *, ctx):
    ...

Compaction(summarizer=handoff, ...)
```

## Key config

| Field | Default | Role |
|-------|---------|------|
| `max_context_tokens` | `200_000` | Budget for fraction triggers |
| `trigger_fraction` | `0.85` | When to run summarization |
| `keep_fraction` | `0.10` | Recent tail kept verbatim |
| `keep_recent_messages` | `6` | Fallback keep window (message count) when token budget is unset |
| `offload_tool_threshold_chars` | `80_000` | Immediate tool-result offload (~20k tokens) |
| `summary_model` | `None` | Registry model for handoff generation |
| `conversation_log_prefix` | `conversation-history` | Artifact prefix for archived transcripts |

## Tests

From the repo root:

```bash
uv run pytest tests/test_compaction.py -q
```
