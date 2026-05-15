# Template for a new persona

Copy this folder pattern under `personas/<your_slug>/`:

- `manifest.yaml` — machine-readable bundle (see any sibling folder).
- `motivations.md`, `lore.md`, `methodology.md`, `relationships.md`, `voice.md`, `constraints.md`, `prompts.md`
- `injection_block.md` — single paste target for LLM system prompts

Then add your slug to `CATALOG.yaml` and re-run `tools/build_library.py` if you extend the generator;
manual additions work without the script.
