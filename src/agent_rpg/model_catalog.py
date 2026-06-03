"""Curated Hugging Face instruct chat models for multi-agent simulations.

`SMALL_INSTRUCT_MODELS` lists **open-weights / low-friction** checkpoints: no Meta Llama
gating (except where noted), no paid Hub access beyond normal Inference rules. You still
need a `HF_TOKEN` for remote Inference; for fully offline runs use `TransformersLocalBackend`
with the same Hub ids (models are downloaded once from the Hub).

**HF Inference router** (``InferenceClient.chat_completion``) only exposes a subset of Hub
repos. Smaller Qwen checkpoints (1.5B / 3B) often return ``model_not_supported``; use
``DEFAULT_HF_INFERENCE_MODEL_ID`` or another id in ``HF_ROUTER_INSTRUCT_MODEL_IDS``.

For **OpenRouter** (`OpenRouterBackend`, ``backend: openrouter`` on agents), set each
agent's ``model_id`` to an OpenRouter model slug from https://openrouter.ai/models .
"""

from __future__ import annotations

from typing import TypedDict


class ModelEntry(TypedDict):
    id: str
    label: str
    approx_params: str


# Local / YAML default — small, fast download.
DEFAULT_INSTRUCT_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"

# HF Inference router default (chat_completion); 1.5B is usually *not* routed.
DEFAULT_HF_INFERENCE_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"

# Repo ids that worked on the public HF router in project smoke tests (your account may differ).
HF_ROUTER_INSTRUCT_MODEL_IDS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    }
)

# Open / widely usable instruct models. Order: HF-router-friendly first, then local-friendly.
SMALL_INSTRUCT_MODELS: list[ModelEntry] = [
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen2.5 7B Instruct (HF Inference default)",
        "approx_params": "7B",
    },
    {
        "id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "label": "Meta Llama 3 8B Instruct (HF router, gated)",
        "approx_params": "8B",
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5 1.5B Instruct (local default)",
        "approx_params": "1.5B",
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "label": "Qwen2.5 3B Instruct (local)",
        "approx_params": "3B",
    },
    {
        "id": "HuggingFaceH4/zephyr-7b-beta",
        "label": "Zephyr 7B beta (local / router varies)",
        "approx_params": "7B",
    },
]


def default_model_id_for_execution(execution_mode: str) -> str:
    """Pick a catalog default from notebook Execution toggle text or backend name."""
    key = execution_mode.strip().lower()
    if key in ("hf inference api", "hf", "hf_inference", "huggingface"):
        return DEFAULT_HF_INFERENCE_MODEL_ID
    return DEFAULT_INSTRUCT_MODEL_ID


def hf_router_model_entries() -> list[ModelEntry]:
    return [m for m in SMALL_INSTRUCT_MODELS if m["id"] in HF_ROUTER_INSTRUCT_MODEL_IDS]


def model_ids_for_widgets() -> list[tuple[str, str]]:
    """Return (display_label, repo_id) for ipywidgets Dropdown options."""
    return [(m["label"], m["id"]) for m in SMALL_INSTRUCT_MODELS]


def labels_by_id() -> dict[str, str]:
    return {m["id"]: m["label"] for m in SMALL_INSTRUCT_MODELS}


def label_to_id_map() -> dict[str, str]:
    return {m["label"]: m["id"] for m in SMALL_INSTRUCT_MODELS}


def warn_if_not_hf_router(model_ids: list[str], *, execution_mode: str) -> list[str]:
    """Return model ids that are likely to fail on the HF Inference router."""
    if execution_mode.strip().lower() not in ("hf inference api", "hf", "hf_inference"):
        return []
    return [mid for mid in model_ids if mid not in HF_ROUTER_INSTRUCT_MODEL_IDS]
