# Python middleware recipes

Copy-paste middleware recipes for [Genkit](https://github.com/firebase/genkit) Python coding agents. Not part of the official `genkit-plugin-middleware` package.

## Recipes

| Recipe | What it does |
|--------|----------------|
| [Compaction](recipes/compaction/) | Keeps long coding-agent runs inside the context window — clips old tool payloads, offloads fat tool output to artifacts, and optionally summarizes evicted history with a recoverable log. |

Tell your favorite coding agent to open the recipe you want under `recipes/` and copy the `.py` file into your app. Each recipe has its own README with setup and config.

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
