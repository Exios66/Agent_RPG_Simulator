#!/usr/bin/env python3
"""Regenerate derived markdown, injection blocks, and CATALOG from personas/*/manifest.yaml.

Source of truth: each folder under ``personas/<slug>/manifest.yaml``. This script does **not**
edit manifests (avoid YAML churn); it only refreshes the companion ``*.md`` files and
``injection_block.md`` for paste-ready prompts.

Run from repo root:
  python prompts_and_personas/tools/build_library.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = ROOT / "personas"


def _write_markdown(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", ""]
    for h, body in sections:
        lines.append(f"## {h}")
        lines.append("")
        lines.append(textwrap.dedent(body).strip())
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _injection_block(p: dict) -> str:
    a = p["archetype"]
    lines = [
        f"<!-- Persona pack: {a} — paste below into agent system prompt or prompt_variables -->",
        "",
        f"## Persona core: {p['title']}",
        "",
        "### Motivations",
        f"- **Primary:** {p['motivations']['primary']}",
        f"- **Secondary:** {p['motivations']['secondary']}",
        f"- **Fear:** {p['motivations']['fear']}",
        f"- **Surface goal:** {p['motivations']['surface_goal']}",
        "",
        "### Lore (public / private)",
        f"- **Public:** {p['lore']['public_face']}",
        f"- **Private:** {p['lore']['private_truth']}",
        "- **Hooks:** " + "; ".join(p["lore"]["hooks"]),
        "",
        "### Methodology",
        f"- **Decision style:** {p['methodology']['decision_style']}",
        "- **Debate / pressure tactics:**",
        *[f"  - {t}" for t in p["methodology"]["debate_tactics"]],
        f"- **Information habits:** {p['methodology']['information_habits']}",
        "",
        "### Constraints & blind spots",
        "- **Will not:** " + "; ".join(p["constraints"]["will_not"]),
        f"- **Blind spots:** {p['constraints']['blind_spots']}",
        "",
        "### Relationships",
        f"- **Default stance:** {p['relationships']['default_stance']}",
        f"- **When trusts:** {p['relationships']['when_trusts']}",
        f"- **When suspects:** {p['relationships']['when_suspects']}",
        "",
        "### Voice",
        f"- **Diction:** {p['voice']['diction']}",
        "- **Patterns:** " + "; ".join(p["voice"]["patterns"]),
        "- **Avoid:** " + "; ".join(p["voice"]["avoid"]),
        "",
        "### Model directives (obey in JSON `say` / `thought`)",
        *[f"- {d}" for d in p["prompts"]["system_directives"]],
        "",
        "### Scene seeds (GM / author)",
        *[f"- {s}" for s in p["prompts"]["scene_seeds"]],
        "",
    ]
    return "\n".join(lines)


def _emit_derived(d: Path, p: dict) -> None:
    title = p["title"]
    _write_markdown(
        d / "motivations.md",
        f"Motivations — {title}",
        [
            ("Primary drive", p["motivations"]["primary"]),
            ("Secondary drive", p["motivations"]["secondary"]),
            ("Core fear", p["motivations"]["fear"]),
            ("What they say they want", p["motivations"]["surface_goal"]),
        ],
    )
    _write_markdown(
        d / "lore.md",
        f"Lore — {title}",
        [
            ("Public reputation", p["lore"]["public_face"]),
            ("Private truth", p["lore"]["private_truth"]),
            ("Story hooks", "\n".join(f"- {h}" for h in p["lore"]["hooks"])),
        ],
    )
    _write_markdown(
        d / "methodology.md",
        f"Methodology — {title}",
        [
            ("Decision style", p["methodology"]["decision_style"]),
            ("Debate & pressure tactics", "\n".join(f"- {t}" for t in p["methodology"]["debate_tactics"])),
            ("Information discipline", p["methodology"]["information_habits"]),
        ],
    )
    _write_markdown(
        d / "relationships.md",
        f"Relationships — {title}",
        [
            ("Default stance", p["relationships"]["default_stance"]),
            ("When they trust", p["relationships"]["when_trusts"]),
            ("When they suspect", p["relationships"]["when_suspects"]),
        ],
    )
    _write_markdown(
        d / "voice.md",
        f"Voice & delivery — {title}",
        [
            ("Diction", p["voice"]["diction"]),
            ("Speech patterns", "\n".join(f"- {x}" for x in p["voice"]["patterns"])),
            ("Avoid", "\n".join(f"- {x}" for x in p["voice"]["avoid"])),
        ],
    )
    _write_markdown(
        d / "constraints.md",
        f"Constraints — {title}",
        [
            ("Lines they will not cross", "\n".join(f"- {x}" for x in p["constraints"]["will_not"])),
            ("Blind spots", p["constraints"]["blind_spots"]),
        ],
    )
    _write_markdown(
        d / "prompts.md",
        f"Prompts — {title}",
        [
            ("Model directives", "\n".join(f"- {x}" for x in p["prompts"]["system_directives"])),
            ("Scene seeds", "\n".join(f"- {x}" for x in p["prompts"]["scene_seeds"])),
        ],
    )
    (d / "injection_block.md").write_text(_injection_block(p), encoding="utf-8")


def _ensure_template_stub() -> None:
    tmpl = ROOT / "_template"
    tmpl.mkdir(parents=True, exist_ok=True)
    example_path = tmpl / "manifest.example.yaml"
    if example_path.is_file():
        return
    example = {
        "slug": "example_new_persona",
        "archetype": "example archetype label",
        "title": "Example Title",
        "tags": ["tag_one", "tag_two"],
        "motivations": {"primary": "", "secondary": "", "fear": "", "surface_goal": ""},
        "lore": {"public_face": "", "private_truth": "", "hooks": []},
        "methodology": {"decision_style": "", "debate_tactics": [], "information_habits": ""},
        "constraints": {"will_not": [], "blind_spots": ""},
        "relationships": {"default_stance": "", "when_trusts": "", "when_suspects": ""},
        "voice": {"diction": "", "patterns": [], "avoid": []},
        "prompts": {"system_directives": [], "scene_seeds": []},
        "integration": {"yaml_agent_fields": {}, "system_prompt_append": ""},
    }
    example_path.write_text(
        yaml.dump(example, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    if not PERSONAS_DIR.is_dir():
        raise SystemExit(f"Missing personas directory: {PERSONAS_DIR}")

    catalog: list[dict[str, str]] = []
    n = 0
    for d in sorted(PERSONAS_DIR.iterdir(), key=lambda x: x.name):
        if not d.is_dir() or d.name.startswith("_") or d.name.startswith("."):
            continue
        mf = d / "manifest.yaml"
        if not mf.is_file():
            print("skip (no manifest.yaml):", d.name)
            continue
        p = yaml.safe_load(mf.read_text(encoding="utf-8"))
        if not isinstance(p, dict):
            print("skip (invalid yaml root):", d.name)
            continue
        for key in ("archetype", "title", "motivations", "lore", "methodology", "constraints", "relationships", "voice", "prompts"):
            if key not in p:
                raise SystemExit(f"{mf}: missing required key {key!r}")
        slug = str(p.get("slug", d.name))
        catalog.append({"slug": slug, "archetype": p["archetype"], "title": p["title"]})
        _emit_derived(d, p)
        n += 1

    (ROOT / "CATALOG.yaml").write_text(
        yaml.dump({"personas": catalog}, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _ensure_template_stub()
    print(f"Regenerated derived files for {n} personas under {PERSONAS_DIR}")


if __name__ == "__main__":
    main()
