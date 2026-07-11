"""
Executes a detected triangular-arbitrage opportunity as 3 real market
orders on Tabdeal.

DRY-RUN BY DEFAULT. Pass live=True (and have TABDEAL_API_KEY /
TABDEAL_API_SECRET set) to place real orders -- this places REAL money at
REAL risk. If leg 2 or 3 fails after leg 1 succeeds, the position is left
unhedged; this module does not attempt automatic unwinding, it only
reports the failure loudly so a human can react.
"""
from __future__ import annotations

import math


def round_step(quantity: float, step_size: float) -> float:
    """Round DOWN to the nearest step_size. Uses log10 instead of string
    inspection of the float -- str(0.00000001) is '1e-08' in Python, which
    silently broke a naive "count digits after the decimal point" approach
    (every quantity below step_size rounded to 0.0)."""
    if step_size <= 0:
        return quantity
    # tiny epsilon guards against floor() clipping a value that should
    # land exactly on a step boundary due to float imprecision
    steps = math.floor(quantity / step_size + 1e-9)
    decimals = max(0, -int(math.floor(math.log10(step_size))))
    return round(steps * step_size, decimals)


def get_lot_size(symbol_info: dict) -> tuple[float, float]:
    """Prefer MARKET_LOT_SIZE (market orders) over LOT_SIZE if both exist."""
    filters = {f["filterType"]: f for f in symbol_info.get("filters", [])}
    f = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
    return float(f.get("minQty", 0)), float(f.get("stepSize", 0.00000001))


class TriangularExecutor:
    def __init__(self, symbols_info: dict[str, dict], live: bool = False):
        """symbols_info: {symbol: exchangeInfo entry} for quantity rounding."""
        self.symbols_info = symbols_info
        self.live = live
        self.client = None
        if live:
            from ..exchanges.tabdeal_trading import TabdealTradingClient
            self.client = TabdealTradingClient()

    def _order(self, symbol: str, side: str, quantity: float) -> dict:
        info = self.symbols_info.get(symbol)
        if info:
            min_qty, step = get_lot_size(info)
            quantity = round_step(quantity, step)
            if quantity < min_qty:
                return {"status": "REJECTED_LOCAL", "reason": f"qty {quantity} < minQty {min_qty}", "symbol": symbol}

        if not self.live:
            print(f"    [DRY-RUN] {side} MARKET {symbol} qty={quantity}")
            return {"status": "DRY_RUN", "symbol": symbol, "side": side,
                     "executedQty": quantity, "cummulativeQuoteQty": None}

        result = self.client.place_market_order(symbol, side, f"{quantity:.8f}")
        print(f"    [LIVE] {side} MARKET {symbol} qty={quantity} -> {result.get('status')}")
        return result

    def execute(self, base_asset: str, direction: str, capital_irt: float,
                x_irt_ask: float, x_usdt_bid: float, x_usdt_ask: float,
                usdt_irt_ask: float) -> dict:
        legs = []
        try:
            if direction == "irt_to_x_to_usdt_to_irt":
                qty_x = capital_irt / x_irt_ask
                r1 = self._order(f"{base_asset}IRT", "BUY", qty_x)
                legs.append(r1)
                if r1.get("status") == "REJECTED_LOCAL":
                    return {"error": "leg 1 rejected before sending", "legs": legs}
                qty_x_filled = float(r1.get("executedQty") or qty_x)

                r2 = self._order(f"{base_asset}USDT", "SELL", qty_x_filled)
                legs.append(r2)
                usdt_amount = float(r2.get("cummulativeQuoteQty") or qty_x_filled * x_usdt_bid)

                r3 = self._order("USDTIRT", "SELL", usdt_amount)
                legs.append(r3)

            elif direction == "irt_to_usdt_to_x_to_irt":
                qty_usdt = capital_irt / usdt_irt_ask
                r1 = self._order("USDTIRT", "BUY", qty_usdt)
                legs.append(r1)
                if r1.get("status") == "REJECTED_LOCAL":
                    return {"error": "leg 1 rejected before sending", "legs": legs}
                usdt_filled = float(r1.get("executedQty") or qty_usdt)

                qty_x = usdt_filled / x_usdt_ask
                r2 = self._order(f"{base_asset}USDT", "BUY", qty_x)
                legs.append(r2)
                qty_x_filled = float(r2.get("executedQty") or qty_x)

                r3 = self._order(f"{base_asset}IRT", "SELL", qty_x_filled)
                legs.append(r3)
            else:
                return {"error": f"unknown direction {direction}"}

        except Exception as e:
            return {
                "error": str(e),
                "legs": legs,
                "UNHEDGED_RISK": 0 < len(legs) < 3,
            }

        return {"legs": legs, "success": True, "live": self.live}
