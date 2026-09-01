"""LLM-gated trade confirmation: ask Claude whether a rule-based
`vol_breakout` candidate (`src.labeling.volatility_breakout_entries`) is
worth taking, given a structured snapshot of the same indicators
`scripts/train_ml.create_features` already computes.

Why this exists, and what it is NOT: the rest of this repo's ethos is
statistical rigor (README, docs/RESEARCH_LOG.md) -- deterministic
backtests, significance testing, deflated p-values. An LLM gate breaks
that: it is not deterministic, cannot be scored with `src/significance.py`
the way a fixed rule or a fitted classifier can, and costs real money per
call. It is offered here as an ADDITIONAL, PAPER-ONLY confirmation layer
to test empirically against the live shadow paper-trader
(`scripts/paper_test_llm.py`), not a replacement for the statistical
validation the rest of this repo relies on, and not a live-execution bot --
no order is ever sent.

All network/API-client code is isolated in `get_llm_decision`; prompt
construction and response parsing are pure functions so they're testable
without hitting the real API.
"""
import json
import re

DECISION_SYSTEM_PROMPT = (
    "You are a risk-averse trade-approval gate for a systematic crypto "
    "trading bot. A rule-based volatility-breakout signal has already "
    "fired on a closed candle; your only job is to APPROVE or REJECT "
    "taking it, based on the indicator snapshot given. Be skeptical by "
    "default -- most candidate trades should be rejected unless the "
    "snapshot shows clearly favorable, non-conflicting conditions for "
    "this specific direction. Respond with ONLY a JSON object of the form "
    '{"decision": "approve"|"reject", "confidence": <0-1 float>, '
    '"reason": "<one sentence>"}. No other text, no markdown fences.'
)

# Subset of scripts/train_ml.create_features columns relevant to a
# breakout entry: trend, momentum, volatility regime and mean-reversion
# context. Kept as an explicit whitelist rather than dumping every
# feature column so the prompt stays small and each field is one the
# model can plausibly reason about.
FEATURE_KEYS = (
    'RSI_14', 'RSI_70', 'ATR_ratio', 'bb_width', 'bb_position',
    'close_to_sma20', 'close_to_sma50', 'vol_regime', 'roc_10', 'roc_20',
)


def build_decision_prompt(asset: str, side: int, signal_price: float, atr: float,
                           features: dict) -> str:
    direction = "LONG" if side > 0 else "SHORT"
    lines = [
        f"Asset: {asset}",
        f"Signal: volatility-breakout {direction} at close {signal_price:.6g}",
        f"ATR(14, price units): {atr:.6g}",
    ]
    for key in FEATURE_KEYS:
        value = features.get(key)
        if value is not None:
            lines.append(f"{key}: {value:.6g}")
    lines.append(
        "Approve only if the indicators support this specific direction; "
        "reject on conflicting, ambiguous, or missing evidence."
    )
    return "\n".join(lines)


def parse_decision(text: str) -> dict:
    """Parse the model's reply into {'decision', 'confidence', 'reason'}.
    Fails CLOSED (defaults to reject) on anything that doesn't parse
    cleanly -- an approval gate should never fail open.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match is not None:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            obj = None
    else:
        obj = None

    if obj is None:
        return {'decision': 'reject', 'confidence': 0.0, 'reason': 'unparseable response'}

    decision = str(obj.get('decision', '')).strip().lower()
    if decision not in ('approve', 'reject'):
        decision = 'reject'

    try:
        confidence = float(obj.get('confidence', 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    reason = str(obj.get('reason', ''))[:500]
    return {'decision': decision, 'confidence': confidence, 'reason': reason}


def get_llm_decision(client, model: str, asset: str, side: int, signal_price: float,
                      atr: float, features: dict) -> dict:
    """Call the Anthropic Messages API to approve/reject one candidate
    trade. `client` is an `anthropic.Anthropic`-shaped object (injected so
    tests can pass a fake) exposing `.messages.create(...)`. Any API
    failure is treated the same as an unparseable response -- reject,
    never approve on an error.
    """
    prompt = build_decision_prompt(asset, side, signal_price, atr, features)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=200,
            system=DECISION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content
                        if getattr(block, 'type', None) == 'text')
    except Exception as exc:
        return {'decision': 'reject', 'confidence': 0.0, 'reason': f'API error: {exc}'}
    return parse_decision(text)
