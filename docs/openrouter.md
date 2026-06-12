# OpenRouter configuration (Agent RPG Simulator)

This repository ships **`OpenRouterBackend`** (`src/agent_rpg/backends/openrouter.py`), a dedicated client for [OpenRouter](https://openrouter.ai/) chat completions (`POST /chat/completions`). It is the recommended path for OpenRouter API keys and model slugs.

An alternative is **`HuggingFaceInferenceBackend`** with `base_url="https://openrouter.ai/api/v1"` and your OpenRouter key as `token` — same OpenAI-compatible wire format, but without OpenRouter attribution headers (`HTTP-Referer`, `X-Title`).

## Endpoint and credentials

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | Required bearer token ([create a key](https://openrouter.ai/settings/keys)) |
| `OPENROUTER_BASE_URL` | Optional; default `https://openrouter.ai/api/v1` |
| `OPENROUTER_HTTP_REFERER` | Optional; sent as `HTTP-Referer` for [app attribution](https://openrouter.ai/docs/app-attribution) |
| `OPENROUTER_APP_TITLE` | Optional; sent as `X-Title` |

See [`envs/.env.example`](../envs/.env.example).

## CLI

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
agent-rpg run path/to/scenario.yaml --backend openrouter --output runs
```

`--backend openrouter` constructs `OpenRouterBackend()` and passes it as both the default backend and `openrouter_backend` to `SimulationEngine.run()` (`src/agent_rpg/cli.py`).

Agents with YAML `backend: auto` or `backend: hf_inference` use the default backend (OpenRouter when `--backend openrouter`). Agents with `backend: openrouter` require `openrouter_backend=OpenRouterBackend()` in library code (`src/agent_rpg/engine.py`, `_backend_for_agent`).

## Library usage

```python
import os
from agent_rpg import SimulationEngine, load_scenario
from agent_rpg.backends.openrouter import OpenRouterBackend

scenario = load_scenario("examples/scenarios/minimal.yaml")
backend = OpenRouterBackend(api_key=os.environ["OPENROUTER_API_KEY"])
out = SimulationEngine(scenario).run(
    backend,
    openrouter_backend=backend,
    output_dir="runs",
)
```

## Model identifiers

Per-agent `model_id` values in YAML are passed unchanged as the `model` field. Use **OpenRouter model slugs** (e.g. `meta-llama/llama-3.2-3b-instruct:free`), not Hugging Face Hub ids. Stock scenarios under `examples/scenarios/` use Hub-style ids and need retargeting for OpenRouter runs.

## Generation parameters wired by this repository

| Source | Parameter | Where applied |
|--------|-----------|---------------|
| `AgentConfig` | `model_id` | Every agent turn: `SimulationEngine.run` → `generate(..., model_id=...)` |
| `AgentConfig` | `max_new_tokens` | Mapped to `max_tokens` in `OpenRouterBackend.generate` |
| `AgentConfig` | `temperature` | Passed through to the API body |
| `AgentConfig` | `top_p` | Included when not `None` |
| `orchestration.reactive_router_model_id` | Router `model_id` | Reactive turn order: `max_new_tokens=64`, `temperature=0.2` (`engine.py`, `_speaker_order`) |
| `SimulationEngine.run(..., llm_extras=...)` | Extra kwargs | Merged into every `generate(...)` call (`frequency_penalty`, `presence_penalty`, `seed`, `stop`, `response_format`, `user`, `stream`, `chunk_callback`) |

### Streaming

When `llm_extras` contains `stream=True` (and optionally `chunk_callback`), `OpenRouterBackend` reads SSE `data:` lines, accumulates `delta.content`, and returns the full string. Non-dict SSE payloads and malformed `delta`/`message` shapes are skipped. If the stream yields no text content, or only `error` objects, `generate` raises `RuntimeError` (non-stream empty/malformed responses already raise).

### Non-stream error handling

- HTTP errors → `RuntimeError` with status and body prefix
- Invalid JSON or non-object top-level payload → `RuntimeError`
- Empty `choices` or non-dict choice entries → `RuntimeError`
- Non-dict `message` → treated as empty content (no crash)

Unit tests: `tests/test_openrouter_backend.py`.

## Alternative: Hugging Face client + OpenRouter base URL

```python
from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend

backend = HuggingFaceInferenceBackend(
    token=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
```

This path does not set OpenRouter attribution headers. Streaming uses `hf_inference.py` aggregation logic.

## Official OpenRouter documentation (curated)

| Topic | URL |
|--------|-----|
| Documentation index | https://openrouter.ai/docs/llms.txt |
| Quickstart | https://openrouter.ai/docs/quickstart |
| API reference overview | https://openrouter.ai/docs/api/reference/overview |
| Request parameters | https://openrouter.ai/docs/api/reference/parameters |
| Authentication | https://openrouter.ai/docs/api/reference/authentication |
| Streaming | https://openrouter.ai/docs/api/reference/streaming |
| Errors and debugging | https://openrouter.ai/docs/api/reference/errors-and-debugging |
| Rate limits | https://openrouter.ai/docs/api/reference/limits |
| Models overview | https://openrouter.ai/docs/guides/overview/models |
| FAQ (free usage / rate limits) | https://openrouter.ai/docs/faq |
| Free model variant (`:free`) | https://openrouter.ai/docs/guides/routing/model-variants/free |
| Free models router | https://openrouter.ai/docs/guides/routing/routers/free-router |
| App attribution | https://openrouter.ai/docs/app-attribution |

Interactive request builder: https://openrouter.ai/request-builder

## Suggested starter pool of free-tier models (speculative)

**Not validated by this project's CI or maintainers.** Confirm each slug on [Models](https://openrouter.ai/models) before relying on it.

This simulator expects chat models that follow a system prompt and return JSON-shaped text (`thought` / `say` / `directed_at`) and, for reactive turn order, compact router JSON (`next_agent_id`). Prefer instruction-tuned chat models; append `:free` where supported.

Illustrative pool (rotate via `assign_models_to_agents`; swap if a slug 404s or returns policy errors):

- `meta-llama/llama-3.2-3b-instruct:free`
- `google/gemma-2-9b-it:free`
- `mistralai/mistral-7b-instruct:free`
- `microsoft/phi-3-mini-128k-instruct:free`
- `qwen/qwen-2-7b-instruct:free`

For `turn_order: reactive`, pick a `reactive_router_model_id` reliable at short JSON (often the smallest instruct model in the pool).
