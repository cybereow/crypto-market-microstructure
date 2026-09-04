"""LLM-gated trade confirmation: ask Claude whether a rule-based primary
signal's candidate trade (any signal in `src/labeling.py` -- vol_breakout,
funding_reversion, etc.) is worth taking, given a structured snapshot of
the same indicators `scripts/train_ml.create_features` already computes.

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
    "trading bot. A rule-based signal has already fired on a closed "
    "candle; your only job is to APPROVE or REJECT taking it, based on "
    "the signal description and indicator snapshot given. Reason "
    "explicitly about whether the available evidence supports THIS "
    "SPECIFIC direction and THIS SPECIFIC signal's own stated premise "
    "(e.g. a reversion signal needs evidence of exhaustion/crowding, a "
    "breakout signal needs evidence of genuine momentum) -- do not apply "
    "a generic breakout or trend checklist to a signal that isn't one. "
    "Weigh confirming and conflicting evidence against each other rather "
    "than pattern-matching a single favorable-looking number, and let "
    "your confidence reflect how one-sided that balance actually is (a "
    "close call is NOT a 0.9). Be skeptical by default -- most candidate "
    "trades should be rejected unless the snapshot shows clearly "
    "favorable, non-conflicting conditions. Respond with ONLY a JSON "
    'object of the form {"decision": "approve"|"reject", "confidence": '
    '<0-1 float>, "reason": "<one sentence citing the specific numbers '
    'that drove the call>"}. No other text, no markdown fences.'
)

# Human-readable description of each src/labeling.py signal's own premise,
# so the prompt states what evidence would actually support (or refute)
# THIS signal rather than a generic "breakout" framing that only fits
# some of them. Falls back to the raw signal key if not listed here.
SIGNAL_DESCRIPTIONS = {
    'breakout': 'Donchian breakout (trend continuation)',
    'reversion': 'RSI mean-reversion (fading an oversold/overbought bounce)',
    'vol_breakout': 'volatility-squeeze breakout (trend continuation out of a low-volatility regime)',
    'trend_pullback': 'trend-pullback (buying a dip / selling a rip within an established trend)',
    'range_fade': 'range-fade (mean-reversion at a stable, non-expanding range edge)',
    'obi_momentum': 'order-book-imbalance momentum (following resting bid/ask pressure)',
    'funding_reversion': ('funding-rate-extreme mean-reversion (fading crowded, over-leveraged '
                          'long/short perpetual-futures positioning)'),
    'obv_divergence': 'volume-divergence fade (a price breakout on-balance volume does not confirm)',
    'btc_lead_lag': "cross-asset BTC lead-lag (this altcoin catching up to BTC's own recent move)",
}

# Subset of scripts/train_ml.create_features columns relevant across the
# signals above: trend, momentum, volatility regime, mean-reversion
# context, and (when the candidate came from a funding-based signal and
# the caller joined a 'funding_rate' column before feature-building) the
# funding-derived columns create_features adds automatically. Kept as an
# explicit whitelist rather than dumping every feature column so the
# prompt stays small and each field is one the model can plausibly
# reason about; a key simply doesn't appear in the prompt when absent
# (e.g. funding_rate for a non-funding signal), rather than erroring.
FEATURE_KEYS = (
    'RSI_14', 'RSI_70', 'ATR_ratio', 'bb_width', 'bb_position',
    'close_to_sma20', 'close_to_sma50', 'vol_regime', 'roc_10', 'roc_20',
    'funding_rate', 'funding_rate_diff', 'funding_sma_5',
)


def build_decision_prompt(asset: str, side: int, signal_price: float, atr: float,
                           features: dict, signal_name: str = 'vol_breakout') -> str:
    direction = "LONG" if side > 0 else "SHORT"
    description = SIGNAL_DESCRIPTIONS.get(signal_name, signal_name)
    lines = [
        f"Asset: {asset}",
        f"Signal type: {description}",
        f"Candidate: {direction} at close {signal_price:.6g}",
        f"ATR(14, price units): {atr:.6g}",
    ]
    for key in FEATURE_KEYS:
        value = features.get(key)
        if value is not None:
            lines.append(f"{key}: {value:.6g}")
    lines.append(
        "Approve only if the evidence supports THIS signal's own premise in THIS "
        "specific direction; reject on conflicting, ambiguous, or missing evidence."
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
                      atr: float, features: dict, max_tokens: int = 1024,
                      signal_name: str = 'vol_breakout') -> dict:
    """Call the Messages API (works against the real Anthropic API, or any
    Anthropic-Messages-API-compatible gateway/proxy set via base_url) to
    approve/reject one candidate trade. `client` is injected so tests can
    pass a fake.

    Fails closed (reject) on any problem, and reports which kind
    separately rather than lumping them all into one label -- a request
    exception (bad model name, auth, rate limit) looks nothing like a
    response that came back successfully but with no text (observed in
    practice: a thinking-capable model burning its entire `max_tokens`
    budget on internal reasoning and never emitting the JSON answer), and
    conflating the two makes a real bug indistinguishable from a normal
    API error. `max_tokens` defaults generously for the same reason -- a
    tight budget can be consumed entirely by a model's own reasoning
    before it reaches the answer.
    """
    prompt = build_decision_prompt(asset, side, signal_price, atr, features, signal_name=signal_name)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=DECISION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        return {'decision': 'reject', 'confidence': 0.0,
                'reason': f'API error ({type(exc).__name__}): {exc}'}

    # `content` can legitimately be empty/None on some gateways even
    # without raising -- never iterate it directly inside the try above,
    # or a local bug here gets misreported as an "API error".
    content = getattr(response, 'content', None) or []
    text = "".join(getattr(block, 'text', '') for block in content
                   if getattr(block, 'type', None) == 'text')

    if not text:
        block_types = [getattr(block, 'type', '?') for block in content]
        return {'decision': 'reject', 'confidence': 0.0,
                'reason': f'no text in response (blocks={block_types}, '
                          f"stop_reason={getattr(response, 'stop_reason', None)}) -- "
                          f'try a higher max_tokens if a thinking block used up the budget'}

    decision = parse_decision(text)
    if decision['reason'] == 'unparseable response':
        decision['reason'] = f'unparseable response: {text[:200]!r}'
    return decision
