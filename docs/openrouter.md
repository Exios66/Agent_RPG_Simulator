# OpenRouter configuration (Agent RPG Simulator)

This project does not ship an OpenRouter-specific backend class. Remote chat runs through `HuggingFaceInferenceBackend` (`src/agent_rpg/backends/hf_inference.py`), which wraps `huggingface_hub.InferenceClient.chat_completion`. That client accepts a custom `base_url`, so you can point it at OpenRouter’s OpenAI-compatible endpoint while keeping the same simulation code paths as Hugging Face Inference.

## Endpoint and credentials

- **Base URL (from OpenRouter):** `https://openrouter.ai/api/v1` — see [OpenRouter Quickstart](https://openrouter.ai/docs/quickstart).
- **Authentication:** OpenRouter expects `Authorization: Bearer <OPENROUTER_API_KEY>`. In this repo, pass your OpenRouter key as the backend `token` argument (the constructor also falls back to `HF_TOKEN` if `token` is omitted, which is usually *not* what you want for OpenRouter).

Construct the backend explicitly, for example:

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

OpenRouter documents optional attribution headers (`HTTP-Referer`, `X-OpenRouter-Title`) for app rankings — see [App attribution](https://openrouter.ai/docs/app-attribution). This backend only forwards `token` and `base_url` into `InferenceClient`; it does not set those headers. If you need them, use a small wrapper backend or extend `HuggingFaceInferenceBackend` locally.

## Model identifiers

Per-agent `model_id` values in YAML (or values assigned via `assign_models_to_agents` in `src/agent_rpg/multi_model.py`) are passed through unchanged as the `model` argument to `chat_completion`. Use **OpenRouter model slugs** (for example `openai/gpt-4o-mini`), not Hugging Face Hub ids, when `base_url` targets OpenRouter.

## Generation parameters wired by this repository

These are enforced in code, independent of provider:

| Source | Parameter | Where it is applied |
|--------|-----------|---------------------|
| `AgentConfig` | `model_id` | Every agent turn: `SimulationEngine.run` → `generate(..., model_id=...)` (`src/agent_rpg/engine.py`). |
| `AgentConfig` | `max_new_tokens` | Mapped to `max_tokens` in `HuggingFaceInferenceBackend.generate` (`src/agent_rpg/backends/hf_inference.py`). |
| `AgentConfig` | `temperature` | Passed through to `chat_completion`. |
| `AgentConfig` | `top_p` | Included only if not `None`. |
| `orchestration.reactive_router_model_id` | Router `model_id` | When `turn_order` is `reactive`, router call uses `max_new_tokens=64` and `temperature=0.2` (`src/agent_rpg/engine.py`, `_speaker_order`). |
| `SimulationEngine.run(..., llm_extras=...)` | Extra kwargs | Merged into every `generate(...)` invocation (agent lines and router). Values become part of the kwargs dict consumed by `HuggingFaceInferenceBackend.generate` (after `stream` / `chunk_callback` handling). |

Streaming: if `llm_extras` contains `stream=True` and optionally `chunk_callback`, `hf_inference.py` aggregates streamed deltas into a single string return value (same as the Hugging Face path).

## CLI

`agent-rpg run` (`src/agent_rpg/cli.py`) always builds `HuggingFaceInferenceBackend(token=os.environ.get("HF_TOKEN"))` with **no** `base_url`. There is **no** built-in flag for OpenRouter; use the library pattern above or a thin script.

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
