"""Strategy-side US whole-share execution using the shared push cache."""

from dataclasses import replace
from datetime import datetime, timezone
import math

from investing_algorithm_framework.domain import (
    OperationalException,
    SignalSide,
)
from investing_algorithm_framework.services.executors import LimitOrderExecutor
from . import sdk, us_symbol
from ..data_providers.longbridge import is_us_regular_session


def is_tradable_quote(quote, now):
    """Use the same status/age gate for risk checks and order submission."""
    age = (now - quote.timestamp.astimezone(timezone.utc)).total_seconds()
    return quote.trade_status == sdk().TradeStatus.Normal and 0 <= age <= 120


class LongbridgeLimitExecutor(LimitOrderExecutor):
    """Quote-priced limits, optionally capped by ATR-based signal risk."""

    def __init__(self, client, min_entry_shares=1):
        if type(min_entry_shares) is not int or min_entry_shares < 1:
            raise ValueError("min_entry_shares must be a positive integer")
        self.client = client
        self.min_entry_shares = min_entry_shares

    def decline(self, intent, reason):
        """Record a skipped order's concrete reason for health reports."""
        self.client.__dict__.setdefault("execution_issues", {})[
            us_symbol(intent.full_symbol)
        ] = reason
        return None

    def cap_entry_amount(self, intent, context, metadata, amount, depth):
        """组合执行器在同一盘口和实际风险距离上进一步限制股数。"""
        return amount

    def execute(self, intent, context, metadata):
        now = datetime.now(timezone.utc)
        buying = intent.side == SignalSide.OPEN_LONG
        if not is_us_regular_session(self.client, now, buying):
            return self.decline(
                intent,
                (
                    "outside_entry_window"
                    if buying
                    else "outside_regular_session"
                ),
            )
        if intent.side not in (SignalSide.OPEN_LONG, SignalSide.CLOSE_LONG):
            raise OperationalException(
                "Longbridge strategy execution is long-only"
            )
        symbol = us_symbol(intent.full_symbol)
        quote = self.client.get_quote(symbol)
        if not is_tradable_quote(quote, now):
            return self.decline(intent, "quote_stale_or_suspended")
        depth = self.client.get_depth(symbol)
        book = depth.asks if buying else depth.bids
        if not book or book[0].price is None or book[0].price <= 0:
            return self.decline(intent, "missing_order_book")
        price = float(book[0].price)
        if not math.isfinite(price):
            return self.decline(intent, "invalid_order_book_price")
        amount = math.floor(intent.amount)
        metadata = dict(metadata)
        if buying:
            # Chandelier 提供绝对保护价；按本次买价计算风险距离，跳空不能沿用旧距离。
            initial_stop = metadata.get("initial_stop_price")
            if initial_stop is not None:
                if (
                    isinstance(initial_stop, bool)
                    or not isinstance(initial_stop, (int, float))
                    or not math.isfinite(initial_stop)
                    or initial_stop <= 0
                ):
                    raise OperationalException(
                        "Invalid initial protective stop"
                    )
                if initial_stop >= price:
                    return self.decline(
                        intent, "price_at_or_below_initial_stop"
                    )
                metadata["stop_distance"] = price - initial_stop
            portfolio = context.get_portfolio()
            amount = min(
                amount,
                math.floor(intent.quote_amount / price),
                math.floor(portfolio.unallocated * 0.98 / price),
            )
            distance = metadata.get("stop_distance")
            if distance is not None:
                risk = metadata.get("risk_fraction", 0)
                if (
                    not math.isfinite(distance)
                    or not 0 < distance < price
                    or not math.isfinite(risk)
                    or not 0 < risk <= 1
                ):
                    raise OperationalException("Invalid ATR risk metadata")
                amount = min(
                    amount,
                    math.floor(
                        portfolio.get_net_size() * risk / distance,
                    ),
                )
                # The distance stays on the buy order for every fill/restart.
                metadata["stop_distance"] = distance
            amount = self.cap_entry_amount(
                intent, context, metadata, amount, depth
            )
            if amount is None:
                return None
        if amount <= 0:
            return self.decline(intent, "cash_or_risk_below_one_share")
        # Apply the entry floor after every cap; never increase risk to fit it.
        if buying and amount < self.min_entry_shares:
            return self.decline(intent, "below_minimum_entry_shares")
        if not buying and (
            metadata.get("risk_exit") or metadata.get("market_exit")
        ):
            return context.create_market_order(
                target_symbol=intent.symbol,
                order_side="SELL",
                amount=amount,
                execute=True,
                validate=True,
                sync=True,
                metadata=metadata,
            )
        return super().execute(
            replace(
                intent, amount=amount, price=price, quote_amount=amount * price
            ),
            context,
            metadata,
        )
