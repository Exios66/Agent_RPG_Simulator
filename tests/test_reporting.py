from agent_rpg.reporting.builder import ReportBuilder
from agent_rpg.reporting.metrics import gini
from agent_rpg.schemas.events import SimulationEvent


def test_gini_equal():
    assert gini([1.0, 1.0, 1.0]) == 0.0


def test_report_builder_decisions():
    events = [
        SimulationEvent(
            run_id="r",
            round=0,
            agent_id="a",
            event_type="message",
            payload={"text": "DECISION: we leave at dawn", "directed_at": None},
        ),
        SimulationEvent(
            run_id="r",
            round=0,
            agent_id="b",
            event_type="message",
            payload={"text": "short", "directed_at": "a"},
        ),
    ]
    rb = ReportBuilder(events)
    d = rb.to_dict()
    assert d["social_dynamics"]["decisions"]
    assert "a<->b" in d["social_dynamics"]["reciprocity_min_counts"]
