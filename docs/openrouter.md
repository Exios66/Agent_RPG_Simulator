# OpenRouter configuration (Agent RPG Simulator)

This repository integrates OpenRouter in two **independent** ways:

1. **`OpenRouterBackend`** (`src/agent_rpg/backends/openrouter.py`) — stdlib `urllib` POSTs to OpenRouter’s OpenAI-compatible **`/chat/completions`** endpoint. This is what the CLI uses when you pass `--backend openrouter` (`src/agent_rpg/cli.py`).
2. **`HuggingFaceInferenceBackend`** with a custom `base_url` (`src/agent_rpg/backends/hf_inference.py`) — `huggingface_hub.InferenceClient` pointed at `https://openrouter.ai/api/v1`. Use this only if you explicitly want the HF client stack while still hitting OpenRouter.

The sections below describe (1) first, then the HF-based alternative, then shared simulation wiring.

---

## `OpenRouterBackend`: credentials and HTTP behavior

### Environment and constructor

| Setting | Source |
|---------|--------|
| API key | Constructor `api_key=...`, else `OPENROUTER_API_KEY` (`OpenRouterBackend.__init__`). |
| Base URL | Constructor `base_url=...`, else `OPENROUTER_BASE_URL`, else default `https://openrouter.ai/api/v1`. |
| Optional ranking headers | `OPENROUTER_HTTP_REFERER` → request header `HTTP-Referer`; `OPENROUTER_APP_TITLE` → `X-Title` (same as in code). |

If no API key is available, `generate` raises **`ValueError`** with a message mentioning `OPENROUTER_API_KEY` (`src/agent_rpg/backends/openrouter.py`).

### Request shape

`generate` builds a JSON body with `model`, `messages`, `max_tokens` (from `max_new_tokens`), `temperature`, optional `top_p`, optional `stream`, and passthrough keys: `frequency_penalty`, `presence_penalty`, `seed`, `stop`, `response_format`, `user` when present in `kwargs`.

### Non-stream responses: validation and errors

After a successful HTTP response, the body is read as UTF-8 (with replacement for invalid bytes), then parsed with `json.loads`. The following cases raise **`RuntimeError`** with an explicit message (they do **not** crash the interpreter with `JSONDecodeError` or `AttributeError` on unexpected types):

| Condition | Implementation reference |
|-----------|---------------------------|
| Body is not valid JSON | `json.JSONDecodeError` wrapped in `RuntimeError` (“invalid JSON”). |
| JSON root is not an object (e.g. `[]`) | `RuntimeError` (“not an object” / type name in message). |
| `choices` missing, empty, or first element not an object | `RuntimeError` (“no choices” / “invalid choice”). |

Coverage for the above is in `tests/test_openrouter_backend.py` (invalid JSON, array root, malformed `choices`).

### Streaming (`stream=True`)

The streaming path iterates SSE `data:` lines, `json.loads` each chunk, skips non-dict choices and decode errors, aggregates `delta.content` strings, and supports optional `chunk_callback` (`OpenRouterBackend._read_sse_stream`).

### HTTP / network errors

`HTTPError` and `URLError` from `urlopen` are wrapped in **`RuntimeError`** with status or reason text (`OpenRouterBackend.generate`).

---

## Wiring into `SimulationEngine` and CLI

- **Per-agent backend** is `AgentConfig.backend` (`src/agent_rpg/schemas/agent.py`). Values include `openrouter`, `hf_inference`, `transformers_local`, `auto`.
- **`_backend_for_agent`** (`src/agent_rpg/engine.py`): agents with `backend: openrouter` **require** `SimulationEngine.run(..., openrouter_backend=...)`. If it is omitted, **`ValueError`** is raised.
- **CLI** `agent-rpg run --backend openrouter`: constructs `OpenRouterBackend()` and passes it as both the default backend and `openrouter_backend`, so agents on `auto` / `hf_inference` also use OpenRouter for that run (`src/agent_rpg/cli.py`).

### Generation parameters (simulation → `OpenRouterBackend`)

These are enforced by `SimulationEngine.run` and the backend, independent of provider docs:

| Source | Parameter | Where applied |
|--------|-----------|----------------|
| `AgentConfig` | `model_id` | Passed as `model_id=` into every agent `generate` (`src/agent_rpg/engine.py`). |
| `AgentConfig` | `max_new_tokens`, `temperature`, `top_p` | Passed through to `generate`. |
| `orchestration.reactive_router_model_id` | Router model / caps | Reactive router uses `max_new_tokens=64`, `temperature=0.2` in `_speaker_order` (`src/agent_rpg/engine.py`). |
| `SimulationEngine.run(..., llm_extras=...)` | Extra kwargs | Spread into every `generate` call (`**llm_kw` in `src/agent_rpg/engine.py`). Backends may read or strip keys such as `stream` and `chunk_callback` in their own `generate` implementations. |

---

## Alternative: `HuggingFaceInferenceBackend` + OpenRouter base URL

`InferenceClient` accepts `base_url`. Point it at OpenRouter and pass the OpenRouter key as `token` (the constructor falls back to **`HF_TOKEN`** if `token` is omitted, which is usually wrong for OpenRouter unless you intentionally reuse that variable):

```python
import os
from agent_rpg import SimulationEngine, load_scenario
from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend

scenario = load_scenario("examples/scenarios/minimal.yaml")
backend = HuggingFaceInferenceBackend(
    token=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
out = SimulationEngine(scenario).run(backend, output_dir="runs")
```

OpenRouter’s optional attribution headers (`HTTP-Referer`, title) are **not** set by `HuggingFaceInferenceBackend`; only `OpenRouterBackend` reads `OPENROUTER_HTTP_REFERER` / `OPENROUTER_APP_TITLE` from the environment.

### Model identifiers (both backends)

Per-agent `model_id` in YAML is passed through as the remote **`model`** / Hub model argument. For OpenRouter, use **OpenRouter slugs** (for example `openai/gpt-4o-mini`), not Hugging Face Hub repo ids, when the HTTP target is OpenRouter.

---

## CLI summary

| Mode | Behavior |
|------|----------|
| `--backend hf` (default) | `HuggingFaceInferenceBackend(token=os.environ.get("HF_TOKEN"))` — **no** `base_url`; not OpenRouter. |
| `--backend openrouter` | `OpenRouterBackend()`; requires `OPENROUTER_API_KEY` (or `api_key` if you call from code). |
| `--backend local` | `TransformersLocalBackend` |

There is **no** separate flag for OpenRouter `base_url`; override via env `OPENROUTER_BASE_URL` or construct `OpenRouterBackend` in Python.

---

## Official OpenRouter documentation (curated)

Links below are from OpenRouter’s documentation index (`https://openrouter.ai/docs/llms.txt`). Use them for provider behavior, quotas, and request fields beyond what this repo sets explicitly.

| Topic | URL |
|--------|-----|
| Documentation index / LLM-oriented index | https://openrouter.ai/docs/llms.txt |
| Quickstart (API URL, auth, minimal request) | https://openrouter.ai/docs/quickstart |
| API reference overview | https://openrouter.ai/docs/api/reference/overview |
| Request parameters (temperature, max tokens, top_p, provider routing, etc.) | https://openrouter.ai/docs/api/reference/parameters |
| Authentication | https://openrouter.ai/docs/api/reference/authentication |
| Streaming | https://openrouter.ai/docs/api/reference/streaming |
| Errors and debugging | https://openrouter.ai/docs/api/reference/errors-and-debugging |
| Rate limits | https://openrouter.ai/docs/api/reference/limits |
| Models overview | https://openrouter.ai/docs/guides/overview/models |
| FAQ (incl. free usage / rate limits) | https://openrouter.ai/docs/faq |
| Free model variant (`:free`) | https://openrouter.ai/docs/guides/routing/model-variants/free |
| Free models router | https://openrouter.ai/docs/guides/routing/routers/free-router |
| Model fallbacks | https://openrouter.ai/docs/guides/routing/model-fallbacks |
| Provider routing | https://openrouter.ai/docs/guides/routing/provider-selection |
| Structured outputs | https://openrouter.ai/docs/guides/features/structured-outputs |
| App attribution (optional headers) | https://openrouter.ai/docs/app-attribution |
| Frameworks: OpenAI SDK + OpenRouter | https://openrouter.ai/docs/guides/community/openai-sdk |
| Python client SDK | https://openrouter.ai/docs/client-sdks/python/overview |

Interactive request builder (useful for probing payloads): https://openrouter.ai/request-builder

---

## Suggested starter pool of free-tier models (speculative)

**Not validated by this project’s CI or maintainers.** OpenRouter’s catalog and free-tier availability change frequently. Confirm each slug, pricing tier, and context limits on [Models](https://openrouter.ai/models) and in the [free variant](https://openrouter.ai/docs/guides/routing/model-variants/free) / [FAQ](https://openrouter.ai/docs/faq) docs before relying on them.

This simulator expects **chat** models that can follow a system prompt and return **JSON-shaped text** for agents (`thought` / `say` / `directed_at`) and, for reactive turn order, compact JSON for the router (`next_agent_id`). Prefer **instruction-tuned** chat models. Appending OpenRouter’s `:free` suffix (where supported) targets free variants.

Illustrative pool (rotate or assign via `assign_models_to_agents`; deduplicate and swap if a slug 404s or returns policy errors):

- `meta-llama/llama-3.2-3b-instruct:free` — small instruct model; reasonable latency for many short turns.
- `google/gemma-2-9b-it:free` — mid-size instruct; stronger adherence than very small models on structured-ish replies.
- `mistralai/mistral-7b-instruct:free` — common baseline instruct checkpoint on aggregators.
- `microsoft/phi-3-mini-128k-instruct:free` — compact instruct model with long context (useful if transcripts grow; still cap rounds/memory in YAML).
- `qwen/qwen-2-7b-instruct:free` — general instruct workhorse in many hosted catalogs.

For **`turn_order: reactive`**, pick a `reactive_router_model_id` that is **reliable at short JSON** (often the smallest instruct model in the pool is enough). Heavier models can remain on speaking agents only.

Re-check the live model list regularly; free routers and `:free` slots are operational constraints on OpenRouter’s side, not guarantees in this repository.
