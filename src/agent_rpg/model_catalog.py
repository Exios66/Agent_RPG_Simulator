"""Curated Hugging Face instruct chat models (small / mid-size) for multi-agent simulations."""

from __future__ import annotations

from typing import TypedDict


class ModelEntry(TypedDict):
    id: str
    label: str
    approx_params: str


# Default for new agents / YAML omissions (Llama 3.1 8B Instruct).
DEFAULT_INSTRUCT_MODEL_ID: str = "meta-llama/Llama-3.1-8B-Instruct"

# User-selectable pool: instruct-tuned, relatively modest size; availability depends on HF plan.
SMALL_INSTRUCT_MODELS: list[ModelEntry] = [
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "label": "Llama 3.1 8B Instruct (default)",
        "approx_params": "8B",
    },
    {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "label": "Mistral 7B Instruct v0.3",
        "approx_params": "7B",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen2.5 7B Instruct",
        "approx_params": "7B",
    },
    {
        "id": "google/gemma-2-2b-it",
        "label": "Gemma 2 2B IT",
        "approx_params": "2B",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "label": "SmolLM2 1.7B Instruct",
        "approx_params": "1.7B",
    },
    {
        "id": "HuggingFaceH4/zephyr-7b-beta",
        "label": "Zephyr 7B beta",
        "approx_params": "7B",
    },
]


def model_ids_for_widgets() -> list[tuple[str, str]]:
    """Return (display_label, repo_id) for ipywidgets Dropdown options."""
    return [(m["label"], m["id"]) for m in SMALL_INSTRUCT_MODELS]


def labels_by_id() -> dict[str, str]:
    return {m["id"]: m["label"] for m in SMALL_INSTRUCT_MODELS}
