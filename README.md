# Python middleware recipes

Copy-paste middleware recipes for [Genkit](https://github.com/firebase/genkit) Python coding agents. Not part of the official `genkit-plugin-middleware` package.

## Recipes

### Compaction (`recipes/compaction/`)

Layered context compaction inspired by [Deep Agents](https://github.com/langchain-ai/deepagents) / Open SWE:

1. **Structural** (cheap, no LLM): clip old `write_file` / `edit_file` arguments and truncate oversized tool outputs outside the keep window
2. **Tool offload**: large tool results → session artifact + head/tail preview pointer (`read_artifact` to recover)
3. **Summarization** (optional): at ~85% of context budget, append evicted transcript to a conversation log artifact, replace the prefix with an LLM summary + log pointer

```python
from recipes.compaction import Compaction
from genkit.plugins.middleware import Artifacts, Filesystem

async def summarize(messages, *, ctx):
    # optional custom summarizer; otherwise set summary_model=
    ...

await ai.generate(
    prompt='Fix the failing test in auth.py',
    use=[
        Filesystem(root_dir='./workspace'),
        Artifacts(),
        Compaction(
            max_context_tokens=200_000,
            trigger_fraction=0.85,
            keep_fraction=0.10,
            summary_model='googleai/gemini-2.0-flash',  # cheap model for summaries
        ),
    ],
)
```

Or copy `recipes/compaction/compaction.py` into your project and import locally.

## Setup

Requires Genkit Python with session artifacts (install from [firebase/genkit](https://github.com/firebase/genkit) `py/` tree — not all features are in the PyPI 0.7 build yet):

```bash
pip install genkit genkit-plugin-middleware genkit-plugin-google-genai websockets
```

For development against a local Genkit checkout:

```bash
uv add --editable ../genkit-middleware/py/packages/genkit
uv add --editable ../genkit-middleware/py/plugins/middleware
```

## Tests

```bash
uv sync
uv run pytest -q
```

## License

Apache-2.0
