"""TransformersLocalBackend accepts HF-style extras (e.g. stream) without passing them to the pipeline."""

from __future__ import annotations

from types import SimpleNamespace

from transformers import GenerationConfig

from agent_rpg.backends.transformers_local import TransformersLocalBackend


def test_generate_strips_stream_and_chunk_callback() -> None:
    class _Tok:
        pad_token_id = 1
        eos_token_id = 2

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "PROMPT"

    class _Pipe:
        tokenizer = _Tok()
        model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))
        generation_config = GenerationConfig(max_new_tokens=256, do_sample=True, temperature=0.7)
        last_call: dict = {}

        def __call__(self, prompt, **kwargs):
            _Pipe.last_call = kwargs
            assert "max_new_tokens" not in kwargs
            assert "generation_config" in kwargs
            gc = kwargs["generation_config"]
            assert gc.max_new_tokens == 256
            assert gc.max_length is None
            assert kwargs.get("return_full_text") is False
            assert "stream" not in kwargs
            assert "chunk_callback" not in kwargs
            return [{"generated_text": prompt + "ANS"}]

    b = TransformersLocalBackend()
    b._get_pipeline = lambda _mid: _Pipe()  # type: ignore[method-assign]

    out = b.generate(
        [{"role": "user", "content": "hi"}],
        model_id="dummy/dummy",
        stream=True,
        chunk_callback=lambda _t: None,
    )
    assert "ANS" in out
