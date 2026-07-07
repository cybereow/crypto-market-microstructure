"""
Pre-entry planning calculator -- run this BEFORE opening a real position.
Only computes numbers from real historical data; places no orders.

Usage:
    python scripts/plan_entry.py --capital 7000 --leverage 2 --symbol BTCUSDT
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.funding import FundingArb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=7000.0)
    ap.add_argument("--leverage", type=float, default=2.0)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--recent-days", type=int, default=90)
    args = ap.parse_args()

    strat = FundingArb(args.symbol, leverage=args.leverage)
    plan = strat.plan_entry(args.capital, args.leverage, args.recent_days)

    print(f"\n{'='*64}\nPRE-ENTRY PLAN: {plan['symbol']}  capital=${plan['capital']:,.0f}  "
          f"leverage={plan['leverage']}x\n{'='*64}")
    print(f"\nReference price: ${plan['reference_price']:,.2f}  ({plan['reference_price_date']})")
    print("NOTE: most recent archived daily close, not a live tick -- check the live price before entry.")

    s = plan["sizing"]
    print(f"\n--- Position sizing ---")
    print(f"  Notional per leg      : ${s['notional_per_leg']:,.2f}")
    print(f"  Spot leg (buy)        : ${s['spot_leg_cost']:,.2f}")
    print(f"  Perp leg margin       : ${s['perp_leg_margin']:,.2f}  (short, {plan['leverage']}x)")
    print(f"  Unused capital        : ${s['unused_capital']:,.2f}  (keep as buffer)")

    liq = plan["liquidation"]
    print(f"  Liquidation buffer    : +{liq['buffer_pct']}% adverse move")
    print(f"  Approx liquidation px : ${liq['approx_price']:,.2f}  <-- set a price alert at this level")

    for label, inc in plan["income"].items():
        print(f"\n--- Expected income ({label}, rate={inc['rate_pct_per_8h']}%/8h) ---")
        print(f"  Gross monthly         : ${inc['gross_monthly']:,.2f}")
        print(f"  Gross annual          : ${inc['gross_annual']:,.2f}  ({inc['gross_annual_pct_of_capital']}% of capital)")
        print(f"  One-time entry+exit fee: ${plan['entry_exit_fee']:,.2f}")

    print(f"\n--- Historical safety check ---")
    for window, mv in plan["safety_check"].items():
        flag = "UNSAFE (would have liquidated)" if mv["unsafe"] else "ok historically"
        print(f"  {window:>2}d worst upside move = {mv['worst_pct']:>6.1f}%  (p99={mv['p99_pct']}%)  "
              f"vs {liq['buffer_pct']}% buffer -> {flag}")

    print(f"\n--- Before you open anything ---")
    print(f"  [ ] Confirm ${s['spot_leg_cost']:,.0f} ready for the SPOT buy leg")
    print(f"  [ ] Confirm ${s['perp_leg_margin']:,.0f} margin ready for the SHORT PERP leg")
    print(f"  [ ] Set a price alert at ${liq['approx_price']:,.2f}")
    print(f"  [ ] Weekly check: price vs alert level, funding rate still positive on net")
    if plan["unsafe_historically"]:
        print(f"  [!] This leverage has historically been breached within a monthly gap in monitoring --")
        print(f"      do not skip more than ~1 week without checking.")
    print(f"  [ ] Know your exit plan: close both legs simultaneously if price nears liquidation")


if __name__ == "__main__":
    main()
