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


def test_parse_fence_without_json_label():
    raw = '```\n{"thought":"","say":"from fence","directed_at":null}\n```'
    p = parse_agent_json_response(raw)
    assert p.say == "from fence"
    assert p.parse_error is None


def test_parse_not_object():
    p = parse_agent_json_response("[1, 2]")
    assert p.parse_error == "not_object"


def test_parse_empty_say():
    p = parse_agent_json_response('{"thought":"x","say":"","directed_at":null}')
    assert p.parse_error == "empty_say"


def test_router_invalid_json_returns_none():
    assert parse_router_response("not json", {"a"}) is None


def test_router_non_object_returns_none():
    assert parse_router_response("[1]", {"a"}) is None


def test_parse_message_key_fallback():
    p = parse_agent_json_response('{"thought":"","message":"via message key","directed_at":null}')
    assert p.say == "via message key"
    assert p.parse_error is None


def test_parse_directed_at_null_string():
    p = parse_agent_json_response('{"thought":"","say":"hi","directed_at":"null"}')
    assert p.directed_at is None


def test_router_next_alias():
    assert parse_router_response('{"next":"bob"}', {"alice", "bob"}) == "bob"


def test_router_fenced_json():
    raw = '```json\n{"next_agent_id":"a"}\n```'
    assert parse_router_response(raw, {"a", "b"}) == "a"
