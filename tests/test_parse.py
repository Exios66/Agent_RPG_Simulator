from agent_rpg.parse import parse_agent_json_response, parse_router_response


def test_parse_json_plain():
    p = parse_agent_json_response('{"thought":"t","say":"hello","directed_at":null}')
    assert p.thought == "t"
    assert p.say == "hello"
    assert p.directed_at is None
    assert p.parse_error is None


def test_parse_fenced():
    raw = '```json\n{"thought":"","say":"hi","directed_at":"bob"}\n```'
    p = parse_agent_json_response(raw)
    assert p.say == "hi"
    assert p.directed_at == "bob"


def test_parse_invalid_fallback():
    p = parse_agent_json_response("not json")
    assert p.parse_error == "invalid_json"


def test_router_parse():
    assert parse_router_response('{"next_agent_id":"a"}', {"a", "b"}) == "a"
    assert parse_router_response('{"next_agent_id":"z"}', {"a", "b"}) is None


def test_parse_message_fallback_field():
    p = parse_agent_json_response('{"thought":"t","message":"via message key"}')
    assert p.say == "via message key"
    assert p.parse_error is None


def test_parse_non_object_json():
    p = parse_agent_json_response("[1, 2, 3]")
    assert p.parse_error == "not_object"
    assert p.say == "[1, 2, 3]"


def test_router_parse_next_alias():
    assert parse_router_response('{"next":"b"}', {"a", "b"}) == "b"
