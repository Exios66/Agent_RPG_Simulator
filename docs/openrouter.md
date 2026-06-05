# OpenRouter configuration (Agent RPG Simulator)

Remote chat completions use **`OpenRouterBackend`** in [`src/agent_rpg/backends/openrouter.py`](../src/agent_rpg/backends/openrouter.py): OpenAI-compatible `POST {base_url}/chat/completions` via `urllib.request` (no extra OpenRouter SDK dependency).

Agents can also use **`HuggingFaceInferenceBackend`** with `base_url="https://openrouter.ai/api/v1"` and an OpenRouter key as `token` (see [Alternative: HF client](#alternative-hf-inferenceclient-via-base_url) below). The dedicated backend is preferred when you need OpenRouter attribution headers (`HTTP-Referer`, `X-Title`) or CLI `--backend openrouter`.

## Credentials and base URL

| Variable | Role |
|----------|------|
| `OPENROUTER_API_KEY` | Required for `OpenRouterBackend` when `api_key` is not passed to the constructor |
| `OPENROUTER_BASE_URL` | Optional; default `https://openrouter.ai/api/v1` |
| `OPENROUTER_HTTP_REFERER` | Optional; sent as `HTTP-Referer` |
| `OPENROUTER_APP_TITLE` | Optional; sent as `X-Title` |

See [`envs/.env.example`](../envs/.env.example) and the README [OpenRouter API key](../README.md#openrouter-api-key) section.

## CLI

[`src/agent_rpg/cli.py`](../src/agent_rpg/cli.py) accepts `--backend openrouter`:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
agent-rpg run path/to/scenario.yaml --backend openrouter --output runs
```

- The CLI constructs one `OpenRouterBackend()` and passes it as both the default backend and `openrouter_backend` to `SimulationEngine.run()`.
- Per-agent YAML `backend: auto` or `hf_inference` use that default (same pattern as HF).
- Per-agent `backend: openrouter` requires `openrouter_backend` on `SimulationEngine.run()`; the CLI sets this when `--backend openrouter` is used.

Stock scenarios under `examples/scenarios/` use **Hugging Face Hub** `model_id` values. For OpenRouter you must use **OpenRouter model slugs** in YAML (or assign them with `assign_models_to_agents` from [`src/agent_rpg/multi_model.py`](../src/agent_rpg/multi_model.py)).

## Library usage

```python
import os
from agent_rpg import SimulationEngine, load_scenario
from agent_rpg.backends.openrouter import OpenRouterBackend

scenario = load_scenario("examples/scenarios/minimal.yaml")
backend = OpenRouterBackend()  # reads OPENROUTER_API_KEY
out = SimulationEngine(scenario).run(
    backend,
    openrouter_backend=backend,
    output_dir="runs",
)
```

For mixed backends (some agents on OpenRouter, some on HF or local), pass the appropriate `openrouter_backend` / `local_backend` and set each agent's `backend` in YAML; see `_backend_for_agent` in [`src/agent_rpg/engine.py`](../src/agent_rpg/engine.py).

## Generation parameters

| Source | Parameter | Applied in |
|--------|-----------|------------|
| `AgentConfig.model_id` | OpenRouter `model` | Every agent `generate(..., model_id=...)` |
| `AgentConfig.max_new_tokens` | `max_tokens` | `OpenRouterBackend.generate` |
| `AgentConfig.temperature`, `top_p` | Same names in request body | `OpenRouterBackend.generate` |
| `orchestration.reactive_router_model_id` | Router model | `_speaker_order`: `max_new_tokens=64`, `temperature=0.2` |
| `SimulationEngine.run(..., llm_extras=...)` | Extra kwargs | Merged into each `generate` (after `stream` / `chunk_callback` handling) |

Passthrough body keys when present in `llm_extras` / kwargs: `frequency_penalty`, `presence_penalty`, `seed`, `stop`, `response_format`, `user`.

Streaming: pass `stream=True` and optionally `chunk_callback` in `llm_extras`; streamed deltas are aggregated into the returned string.

## Error handling (non-stream and stream)

Implemented in `OpenRouterBackend.generate` and `_read_sse_stream`:

- HTTP errors → `RuntimeError` with status and body prefix.
- Non-stream: invalid JSON, top-level JSON that is not an object, empty `choices`, non-dict choice entries, or non-dict `message` → `RuntimeError` or safe empty content (see tests in [`tests/test_openrouter_backend.py`](../tests/test_openrouter_backend.py)).
- Stream: malformed SSE lines and non-object JSON payloads are skipped; non-dict entries inside `choices` are skipped.

## Alternative: HF `InferenceClient` via `base_url`

[`HuggingFaceInferenceBackend`](../src/agent_rpg/backends/hf_inference.py) accepts `base_url` and `token`. You can point it at OpenRouter:

```python
HuggingFaceInferenceBackend(
    token=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
```

That path does **not** set OpenRouter's optional `HTTP-Referer` / `X-Title` headers. Use `OpenRouterBackend` if you need them.

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
| FAQ (incl. free usage) | https://openrouter.ai/docs/faq |
| Free model variant (`:free`) | https://openrouter.ai/docs/guides/routing/model-variants/free |
| Free models router | https://openrouter.ai/docs/guides/routing/routers/free-router |
| App attribution | https://openrouter.ai/docs/app-attribution |

Interactive request builder: https://openrouter.ai/request-builder

## Suggested starter pool of free-tier models (speculative)

**Not validated by this project's CI.** Confirm slugs and tiers on [openrouter.ai/models](https://openrouter.ai/models) before production use.

The simulator expects chat models that can return JSON-shaped text for agents (`thought` / `say` / `directed_at`) and, for `turn_order: reactive`, compact router JSON (`next_agent_id`). Prefer instruction-tuned chat models; `:free` suffixes target free variants where OpenRouter offers them.

Illustrative pool (rotate via `assign_models_to_agents`; replace any slug that 404s or errors):

- `meta-llama/llama-3.2-3b-instruct:free`
- `google/gemma-2-9b-it:free`
- `mistralai/mistral-7b-instruct:free`
- `microsoft/phi-3-mini-128k-instruct:free`
- `qwen/qwen-2-7b-instruct:free`

For `turn_order: reactive`, set `reactive_router_model_id` to a model that reliably emits short JSON; lighter models are often sufficient for routing only.
