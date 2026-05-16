"""Curated Hugging Face instruct chat models for multi-agent simulations.

`SMALL_INSTRUCT_MODELS` lists **open-weights / low-friction** checkpoints: no Meta Llama
gating, no paid Hub access beyond normal Inference rules. You still need a `HF_TOKEN`
for remote Inference; for fully offline runs use `TransformersLocalBackend` with the
same Hub ids (models are downloaded once from the Hub).

For **OpenRouter** (`OpenRouterBackend`, ``backend: openrouter`` on agents), set each
agent's ``model_id`` to an OpenRouter model slug from https://openrouter.ai/models (many
listings expose a ``:free`` variant; availability and quotas are defined by OpenRouter).
"""

from __future__ import annotations

from typing import TypedDict


class ModelEntry(TypedDict):
    id: str
    label: str
    approx_params: str


# Default for new agents / YAML omissions — Qwen2.5 1.5B is usually routed on HF Inference;
# SmolLM2 and other small checkpoints are often absent (`model_not_supported`).
DEFAULT_INSTRUCT_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"

# Open / widely usable instruct models. HF **Inference Providers** only expose a subset of Hub
# repos; if you see `model_not_supported`, pick another id here, browse
# https://huggingface.co/inference/models , or use `TransformersLocalBackend` locally.
SMALL_INSTRUCT_MODELS: list[ModelEntry] = [
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5 1.5B Instruct (default)",
        "approx_params": "1.5B",
    },
    {
        "id": "HuggingFaceH4/zephyr-7b-beta",
        "label": "Zephyr 7B beta",
        "approx_params": "7B",
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "label": "Qwen2.5 3B Instruct",
        "approx_params": "3B",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen2.5 7B Instruct",
        "approx_params": "7B",
    },
]


def model_ids_for_widgets() -> list[tuple[str, str]]:
    """Return (display_label, repo_id) for ipywidgets Dropdown options."""
    return [(m["label"], m["id"]) for m in SMALL_INSTRUCT_MODELS]


def labels_by_id() -> dict[str, str]:
    return {m["id"]: m["label"] for m in SMALL_INSTRUCT_MODELS}
