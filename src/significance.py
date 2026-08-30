"""Statistical significance for trade-selection results.

The problem this addresses: a selective gate that admits 85 trades out of
2778 candidates and reports 62.4% win rate looks impressive, but so would
many random 85-trade subsets drawn from a 48.8%-base pool, purely by
chance. With samples this small, "the filter works" and "the filter got
lucky" produce similar-looking tables, and only an explicit null model
separates them.

Two complementary tests are provided:

1. `binomial_ci` / `binomial_pvalue` — is the win rate above the geometric
   breakeven at all? Cheap, but assumes independent trades and ignores
   that we chose this configuration after looking at several.

2. `permutation_test` — the honest one. Holds the *number* of selected
   trades fixed and reshuffles which trades get selected, so the null
   distribution answers exactly the right question: "among all ways to
   pick n trades from this candidate pool, how unusual is our pick?"
   This automatically accounts for base rate, pool size, and selectivity.

A third helper, `deflated_pvalue`, applies a multiple-testing correction
for the number of configurations that were tried before reporting one.
Skipping that step is how backtest results become fiction: try twenty
variants at p<0.05 and one will pass on noise alone.
"""
from __future__ import annotations

import numpy as np

try:  # SciPy is already a transitive dependency via scikit-learn.
    from scipy import stats as _stats
except ImportError:  # pragma: no cover - keeps the module importable
    _stats = None


def binomial_ci(n_wins: int, n_trades: int, alpha: float = 0.05) -> tuple:
    """Clopper-Pearson (exact) confidence interval for a win rate.

    Exact rather than normal-approximation because at n<100 the normal
    approximation is optimistically narrow, which is the wrong direction
    to be wrong in when deciding whether to risk money.
    """
    if n_trades <= 0:
        return (float('nan'), float('nan'))
    if _stats is None:  # pragma: no cover
        p = n_wins / n_trades
        return (p, p)

    lo = 0.0 if n_wins == 0 else _stats.beta.ppf(alpha / 2, n_wins, n_trades - n_wins + 1)
    hi = 1.0 if n_wins == n_trades else _stats.beta.ppf(1 - alpha / 2, n_wins + 1, n_trades - n_wins)
    return (float(lo), float(hi))


def binomial_pvalue(n_wins: int, n_trades: int, p_null: float) -> float:
    """One-sided p-value for win rate > p_null."""
    if n_trades <= 0 or _stats is None:
        return float('nan')
    return float(_stats.binomtest(n_wins, n_trades, p_null,
                                  alternative='greater').pvalue)


def breakeven_win_rate(payoff_ratio: float, cost_per_trade: float = 0.0,
                       avg_risk: float = 1.0) -> float:
    """Win rate at which a fixed-payoff strategy breaks even.

    From the barrier geometry: with reward `b` per win and 1 per loss,
    expectancy is zero when p*b = (1-p), i.e. p = 1/(1+b). Costs shift
    the requirement upward, which is why a 1:1 setup needs materially
    more than 50% to be worth trading.
    """
    if payoff_ratio <= 0:
        return float('nan')
    base = 1.0 / (1.0 + payoff_ratio)
    if cost_per_trade and avg_risk:
        drag = cost_per_trade / (avg_risk * (1.0 + payoff_ratio))
        return float(min(1.0, base + drag))
    return float(base)


def permutation_test(labels, selected_mask, n_iter: int = 10000,
                     statistic: str = 'win_rate', returns=None,
                     random_state: int = 42) -> dict:
    """Is this selection better than a random selection of the same size?

    labels        : 0/1 outcome for every candidate trade in the pool
    selected_mask : bool, which candidates the gate actually took
    statistic     : 'win_rate' or 'mean_return' (needs `returns`)

    The null hypothesis is deliberately *not* "win rate equals the base
    rate". It is "the gate's choice of which trades to take carries no
    information", which is tested by keeping the count fixed and
    permuting the selection. Reported p is the fraction of random
    selections that did at least as well.
    """
    labels = np.asarray(labels)
    selected_mask = np.asarray(selected_mask, dtype=bool)

    if labels.shape[0] != selected_mask.shape[0]:
        raise ValueError("labels and selected_mask must be the same length")

    n_selected = int(selected_mask.sum())
    n_pool = labels.shape[0]

    if n_selected == 0 or n_selected == n_pool:
        return {'statistic': float('nan'), 'null_mean': float('nan'),
                'p_value': float('nan'), 'n_selected': n_selected,
                'n_pool': n_pool, 'reason': 'degenerate selection'}

    if statistic == 'win_rate':
        values = labels.astype(float)
    elif statistic == 'mean_return':
        if returns is None:
            raise ValueError("statistic='mean_return' requires `returns`")
        values = np.asarray(returns, dtype=float)
    else:
        raise ValueError(f"unknown statistic: {statistic}")

    observed = float(values[selected_mask].mean())

    rng = np.random.default_rng(random_state)
    null_stats = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        # Partial shuffle is enough and much cheaper than a full permutation.
        idx = rng.choice(n_pool, size=n_selected, replace=False)
        null_stats[i] = values[idx].mean()

    # +1 in numerator and denominator: the observed value is itself one
    # valid arrangement, which keeps p strictly positive and unbiased.
    p_value = float((np.sum(null_stats >= observed) + 1) / (n_iter + 1))

    return {
        'statistic': observed,
        'null_mean': float(null_stats.mean()),
        'null_std': float(null_stats.std()),
        'null_p95': float(np.percentile(null_stats, 95)),
        'p_value': p_value,
        'n_selected': n_selected,
        'n_pool': n_pool,
        'reason': '',
    }


def deflated_pvalue(p_value: float, n_configurations_tried: int) -> float:
    """Sidak correction for having tried several configurations.

    If you test N independent variants, the chance at least one clears
    p<0.05 by luck is 1-(1-0.05)^N, which is 64% at N=20. This maps a
    single-test p to the probability that the *best of N* would look at
    least this good under the null. It is conservative when the variants
    are correlated (they usually are), but a conservative correction is
    the right default for money-at-risk decisions.
    """
    if not np.isfinite(p_value) or n_configurations_tried < 1:
        return float('nan')
    return float(1.0 - (1.0 - p_value) ** n_configurations_tried)


def describe_result(n_wins: int, n_trades: int, payoff_ratio: float,
                    cost_per_trade: float = 0.0,
                    n_configurations_tried: int = 1) -> dict:
    """Bundle the verdict for one configuration into a printable dict."""
    if n_trades <= 0:
        return {'n_trades': 0, 'win_rate': float('nan'), 'verdict': 'no trades'}

    wr = n_wins / n_trades
    be = breakeven_win_rate(payoff_ratio, cost_per_trade)
    lo, hi = binomial_ci(n_wins, n_trades)
    p = binomial_pvalue(n_wins, n_trades, be)
    p_def = deflated_pvalue(p, n_configurations_tried)

    if not np.isfinite(p_def):
        verdict = 'unknown'
    elif lo > be and p_def < 0.05:
        verdict = 'significant'
    elif p_def < 0.05:
        verdict = 'marginal (CI touches breakeven)'
    else:
        verdict = 'NOT significant'

    return {
        'n_trades': n_trades, 'win_rate': wr, 'breakeven_wr': be,
        'ci_low': lo, 'ci_high': hi, 'p_value': p,
        'p_deflated': p_def, 'verdict': verdict,
    }
