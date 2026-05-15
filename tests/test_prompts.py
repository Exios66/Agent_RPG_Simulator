from agent_rpg.loader import load_scenario
from agent_rpg.prompts import build_agent_system_prompt


def test_system_prompt_contains_relationships():
    s = load_scenario("examples/scenarios/minimal.yaml")
    alice = s.agents[0]
    text = build_agent_system_prompt(
        s,
        alice,
        active_events_text="(none)",
        round_index=0,
    )
    assert "study partner" in text
    assert "bob" in text.lower() or "Bob" in text


def test_template_render_default():
    s = load_scenario("examples/scenarios/tavern.yaml")
    mara = s.agents[0]
    mara.system_prompt = None
    mara.prompt_template_id = "default"
    out = build_agent_system_prompt(s, mara, active_events_text="- [x] y", round_index=1)
    assert "Mara Keen" in out
    assert "Silver Tankard" in out or "coastal" in out.lower()
