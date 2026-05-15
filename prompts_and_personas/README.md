# Prompts & personas library

Authoring pack at repo root: **character psychology, voice, methodology, and injectable prompts**
aligned with procedural archetypes in `agent_rpg.random_scenario` (`_ARCHETYPES`).

## Layout

- `CATALOG.yaml` — index of all personas by `slug` and canonical `archetype` string.
- `personas/<slug>/` — one folder per persona:
  - `manifest.yaml` — structured bundle (motivations, lore, methodology, constraints, relationships, voice, prompts).
  - `motivations.md`, `lore.md`, `methodology.md`, `relationships.md`, `voice.md`, `constraints.md`, `prompts.md` — human-editable chapters.
  - `injection_block.md` — **single file to paste** into an agent system prompt or external tool.
- `_template/` — schema example for new personas.

## Wiring into Agent RPG

- **Quick paste:** append `injection_block.md` to `AgentConfig.system_prompt` (and clear `prompt_template_id` if you want raw text only), *or*
- **Jinja variables:** keep `prompt_template_id: default` and add keys to `prompt_variables` in YAML, then extend `src/agent_rpg/templates/agents/default.jinja2` to reference them (e.g. `{ persona_core }`).
- **Archetype string:** set `archetype:` in scenario YAML to match `manifest.yaml` → `archetype` so procedural/random scenarios line up with this library.

## Regeneration

Persona files are committed so the repo is usable without running tools. To rebuild from the embedded dataset:

```bash
python prompts_and_personas/tools/build_library.py
```

There are **12** core personas under `personas/`.
