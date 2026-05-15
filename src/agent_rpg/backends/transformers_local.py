from __future__ import annotations

import importlib.util
from typing import Any

from agent_rpg.backends.base import LLMBackend


def _accelerate_installed() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def _fallback_single_device(torch_mod: Any, device_map: str | dict[str, Any] | None) -> Any:
    """Map ``device_map`` widget strings to one device when ``accelerate`` is not available."""
    if isinstance(device_map, dict):
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    s = str(device_map or "auto").strip()
    if not s or s.lower() == "auto":
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        mps = getattr(getattr(torch_mod, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return torch_mod.device("mps")
        return torch_mod.device("cpu")
    return torch_mod.device(s)


def _load_causal_lm(
    model_id: str,
    *,
    device_map: str | dict[str, Any],
    torch: Any,
    AutoModelForCausalLM: Any,
) -> Any:
    """Load weights; ``device_map=\"auto\"`` maps to one device to avoid Accelerate disk offload."""
    import os

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    use_accel = _accelerate_installed()
    allow_hf_auto_offload = os.environ.get("AGENT_RPG_INFERENCE_OFFLOAD", "").lower() in ("1", "true", "yes")

    if isinstance(device_map, dict):
        if not use_accel:
            m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
            return m.to(_fallback_single_device(torch, "cuda" if torch.cuda.is_available() else "cpu"))
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    req = str(device_map or "auto").strip()
    if req.lower() != "auto":
        if use_accel:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=req,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
        return m.to(torch.device(req))

    if allow_hf_auto_offload and use_accel:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    if torch.cuda.is_available():
        if use_accel:
            idx = torch.cuda.current_device()
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": f"cuda:{idx}"},
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
        return m.to(torch.device("cuda"))

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        m = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        return m.to(torch.device("mps"))

    m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
    return m.to(torch.device("cpu"))


class TransformersLocalBackend:
    """Local inference via transformers pipeline (optional dependency).

    Each call builds a fresh ``GenerationConfig`` so we never inherit the Hub default
    ``max_length=20`` (transformers' sentinel) alongside ``max_new_tokens``, which otherwise
    triggers warnings and odd length behavior.

    **Device placement:** ``device_map="auto"`` (the notebook default) is resolved to a
    **single** CUDA / MPS / CPU device so Hugging Face does **not** use Accelerate disk
    offload (which pulls weights from disk every layer and is extremely slow). For true
    ``device_map="auto"`` offload on huge models, set env ``AGENT_RPG_INFERENCE_OFFLOAD=1``.
    """

    def __init__(self, device_map: str | dict[str, Any] | None = None) -> None:
        self._device_map = device_map or "auto"
        self._pipelines: dict[str, Any] = {}

    def _get_pipeline(self, model_id: str) -> Any:
        if model_id in self._pipelines:
            return self._pipelines[model_id]
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as e:
            raise ImportError(
                "Install optional deps: pip install 'agent-rpg[local]'"
            ) from e

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = _load_causal_lm(model_id, device_map=self._device_map, torch=torch, AutoModelForCausalLM=AutoModelForCausalLM)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            trust_remote_code=True,
        )
        self._pipelines[model_id] = pipe
        return pipe

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model_id: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> str:
        kwargs.pop("stream", None)
        kwargs.pop("chunk_callback", None)
        _ = kwargs  # HF backend may pass extras; local pipeline uses generation_config only.
        pipe = self._get_pipeline(model_id)
        tok = pipe.tokenizer
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        from transformers import GenerationConfig

        eos_id = tok.eos_token_id
        if eos_id is None:
            eos_id = getattr(getattr(pipe.model, "config", None), "eos_token_id", None)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_cfg.temperature = temperature
        if top_p is not None:
            gen_cfg.top_p = top_p

        out = pipe(prompt, generation_config=gen_cfg, return_full_text=False)
        if not out:
            return ""
        generated = out[0].get("generated_text", "")
        if generated.startswith(prompt):
            return generated[len(prompt) :].strip()
        return generated.strip()
