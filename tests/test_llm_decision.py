import pytest

from src.llm_decision import build_decision_prompt, parse_decision, get_llm_decision


FEATURES = {
    'RSI_14': 62.5, 'RSI_70': 55.0, 'ATR_ratio': 1.3, 'bb_width': 0.08,
    'bb_position': 0.7, 'close_to_sma20': 0.02, 'close_to_sma50': 0.05,
    'vol_regime': 1.1, 'roc_10': 0.03, 'roc_20': 0.06,
}


def test_build_decision_prompt_includes_direction_and_features():
    prompt = build_decision_prompt('BTC/USDT', 1, 65000.123, 800.5, FEATURES)
    assert 'BTC/USDT' in prompt
    assert 'LONG' in prompt
    assert '65000.1' in prompt
    assert 'RSI_14: 62.5' in prompt


def test_build_decision_prompt_short_direction():
    prompt = build_decision_prompt('ETH/USDT', -1, 3000.0, 50.0, FEATURES)
    assert 'SHORT' in prompt


def test_build_decision_prompt_skips_missing_features():
    features = dict(FEATURES)
    features['RSI_14'] = None
    prompt = build_decision_prompt('SOL/USDT', 1, 150.0, 3.0, features)
    assert 'RSI_14' not in prompt
    assert 'RSI_70' in prompt


def test_parse_decision_valid_json():
    text = '{"decision": "approve", "confidence": 0.8, "reason": "trend confirms breakout"}'
    result = parse_decision(text)
    assert result == {'decision': 'approve', 'confidence': 0.8, 'reason': 'trend confirms breakout'}


def test_parse_decision_strips_markdown_fence():
    text = '```json\n{"decision": "reject", "confidence": 0.3, "reason": "conflicting RSI"}\n```'
    result = parse_decision(text)
    assert result['decision'] == 'reject'
    assert result['confidence'] == 0.3


def test_parse_decision_fails_closed_on_garbage():
    result = parse_decision("I'm not sure, maybe you should buy?")
    assert result['decision'] == 'reject'
    assert result['confidence'] == 0.0


def test_parse_decision_fails_closed_on_invalid_decision_value():
    result = parse_decision('{"decision": "maybe", "confidence": 0.9, "reason": "unclear"}')
    assert result['decision'] == 'reject'


def test_parse_decision_clamps_confidence_to_unit_range():
    result = parse_decision('{"decision": "approve", "confidence": 5, "reason": "x"}')
    assert result['confidence'] == 1.0
    result = parse_decision('{"decision": "approve", "confidence": -2, "reason": "x"}')
    assert result['confidence'] == 0.0


class _FakeTextBlock:
    def __init__(self, text):
        self.type = 'text'
        self.text = text


class _FakeThinkingBlock:
    def __init__(self, thinking):
        self.type = 'thinking'
        self.thinking = thinking


class _FakeResponse:
    def __init__(self, content, stop_reason='end_turn'):
        self.content = content
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self.messages = _FakeMessages(response=response, exc=exc)


def _text_response(text):
    return _FakeResponse([_FakeTextBlock(text)])


def test_get_llm_decision_parses_successful_response():
    client = _FakeClient(response=_text_response(
        '{"decision": "approve", "confidence": 0.9, "reason": "clean setup"}'))
    result = get_llm_decision(client, 'claude-sonnet-5', 'BTC/USDT', 1, 65000.0, 800.0, FEATURES)
    assert result['decision'] == 'approve'
    assert result['confidence'] == 0.9


def test_get_llm_decision_fails_closed_on_api_error():
    client = _FakeClient(exc=RuntimeError("connection reset"))
    result = get_llm_decision(client, 'claude-sonnet-5', 'BTC/USDT', 1, 65000.0, 800.0, FEATURES)
    assert result['decision'] == 'reject'
    assert 'connection reset' in result['reason']
    assert 'RuntimeError' in result['reason']


def test_get_llm_decision_fails_closed_when_content_is_none():
    """Regression test: a response with content=None (seen in practice from a
    non-Anthropic model routed through a gateway) must never be iterated
    directly -- that raised TypeError and got mislabeled as an "API error"
    even though the request itself succeeded.
    """
    client = _FakeClient(response=_FakeResponse(None, stop_reason='end_turn'))
    result = get_llm_decision(client, 'claude-sonnet-5', 'BTC/USDT', 1, 65000.0, 800.0, FEATURES)
    assert result['decision'] == 'reject'
    assert 'no text in response' in result['reason']
    assert 'API error' not in result['reason']


def test_get_llm_decision_fails_closed_when_only_thinking_block_present():
    """Regression test: a thinking-capable model that spends its whole
    max_tokens budget on internal reasoning and never emits the JSON
    answer must be reported distinctly (with stop_reason) rather than as
    an unparseable response with no useful diagnostic.
    """
    client = _FakeClient(response=_FakeResponse(
        [_FakeThinkingBlock('reasoning about the trade...')], stop_reason='max_tokens'))
    result = get_llm_decision(client, 'claude-sonnet-5', 'BTC/USDT', 1, 65000.0, 800.0, FEATURES)
    assert result['decision'] == 'reject'
    assert 'max_tokens' in result['reason']
    assert 'thinking' in result['reason']


def test_get_llm_decision_includes_raw_text_snippet_when_unparseable():
    client = _FakeClient(response=_text_response("sure, let me think about this trade..."))
    result = get_llm_decision(client, 'claude-sonnet-5', 'BTC/USDT', 1, 65000.0, 800.0, FEATURES)
    assert result['decision'] == 'reject'
    assert 'let me think about this trade' in result['reason']


def test_get_llm_decision_passes_max_tokens_through():
    client = _FakeClient(response=_text_response(
        '{"decision": "approve", "confidence": 0.5, "reason": "x"}'))
    get_llm_decision(client, 'claude-sonnet-5', 'BTC/USDT', 1, 65000.0, 800.0, FEATURES,
                     max_tokens=2048)
    assert client.messages.last_kwargs['max_tokens'] == 2048


def test_get_llm_decision_defaults_max_tokens_generously():
    client = _FakeClient(response=_text_response(
        '{"decision": "approve", "confidence": 0.5, "reason": "x"}'))
    get_llm_decision(client, 'claude-sonnet-5', 'BTC/USDT', 1, 65000.0, 800.0, FEATURES)
    assert client.messages.last_kwargs['max_tokens'] >= 1024
