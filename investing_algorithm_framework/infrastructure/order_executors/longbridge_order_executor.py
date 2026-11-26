"""Cash US equity orders through Longbridge's official SDK."""

from decimal import Decimal
import json
import os
from uuid import NAMESPACE_URL, uuid5

from investing_algorithm_framework.domain import (
    OperationalException,
    Order,
    OrderExecutor,
    OrderSide,
    OrderStatus,
    OrderType,
)
from investing_algorithm_framework.infrastructure.longbridge import (
    sdk,
    submission_marker,
    us_symbol,
)


def update_order(order, detail):
    """Keep non-terminal broker states OPEN, including pending cancellation."""
    api = sdk()
    statuses = [
        (api.OrderStatus.Filled, OrderStatus.CLOSED.value),
        (api.OrderStatus.Canceled, OrderStatus.CANCELED.value),
        (api.OrderStatus.PartialWithdrawal, OrderStatus.CANCELED.value),
        (api.OrderStatus.Expired, OrderStatus.EXPIRED.value),
        (api.OrderStatus.Rejected, OrderStatus.REJECTED.value),
    ]
    if detail.status == api.OrderStatus.Unknown:
        raise OperationalException(
            "Longbridge returned an unknown order status"
        )
    data = order.to_dict()
    data["metadata"] = dict(order.metadata or {})
    order = Order.from_dict(data)
    previous_filled = Decimal(str(order.filled or 0))
    filled = detail.executed_quantity
    if filled < previous_filled:
        raise OperationalException("Longbridge fill quantity moved backwards")
    if filled > previous_filled:
        if detail.executed_price is None:
            raise OperationalException(
                "Longbridge fill has no execution price"
            )
        notional = filled * detail.executed_price
        previous_notional = Decimal(
            order.metadata.get("longbridge_executed_notional", "0")
        )
        # IAF books each fill delta; the broker reports a cumulative VWAP.
        order.price = float(
            (notional - previous_notional) / (filled - previous_filled)
        )
        order.metadata["longbridge_executed_notional"] = str(notional)
    order.external_id = detail.order_id
    order.status = next(
        (status for external, status in statuses if detail.status == external),
        OrderStatus.OPEN.value,
    )
    order.filled = float(detail.executed_quantity)
    order.remaining = float(detail.quantity - detail.executed_quantity)
    charges = detail.charge_detail
    if charges is not None:
        if charges.currency != order.trading_symbol:
            raise OperationalException("Unexpected Longbridge fee currency")
        order.order_fee = float(charges.total_amount)
        order.order_fee_currency = charges.currency
    order.updated_at = detail.updated_at or order.updated_at
    return order


class LongbridgeOrderExecutor(OrderExecutor):
    def __init__(self, client, priority=1):
        super().__init__(priority=priority)
        self.client = client

    def supports_market(self, market):
        return market.upper() == "LONGBRIDGE"

    def execute_order(self, portfolio, order, market_credential):
        if not self.client.trading_enabled:
            raise OperationalException("Longbridge trading is disabled")
        if portfolio.trading_symbol != "USD":
            raise OperationalException("Phase one requires a USD portfolio")
        symbol = us_symbol(order.get_symbol())
        side = OrderSide.from_value(order.order_side)
        kind = OrderType.from_value(order.order_type)
        if side not in (OrderSide.BUY, OrderSide.SELL):
            raise OperationalException("Phase one supports long-only BUY/SELL")
        if kind not in (OrderType.LIMIT, OrderType.MARKET):
            raise OperationalException(
                "Phase one supports LIMIT/MARKET orders"
            )
        quantity = Decimal(str(order.amount))
        if (
            not quantity.is_finite()
            or quantity <= 0
            or quantity != quantity.to_integral_value()
        ):
            raise OperationalException(
                "Longbridge requires positive whole shares"
            )
        api = sdk()
        kwargs = {}
        if kind == OrderType.LIMIT:
            price = Decimal(str(order.price))
            if not price.is_finite() or price <= 0:
                raise OperationalException(
                    "A positive limit price is required"
                )
            kwargs["submitted_price"] = price
        if side == OrderSide.SELL:
            positions = self.client.trade.stock_positions([symbol])
            available = sum(
                p.available_quantity
                for channel in positions.channels
                for p in channel.positions
                if p.symbol == symbol
            )
            if available < quantity:
                raise OperationalException("Insufficient sellable shares")
        marker = submission_marker(self.config, symbol)
        request_id = str(
            uuid5(
                NAMESPACE_URL,
                f"iaf:{portfolio.id}:{order.id}:"
                f"{order.created_at.isoformat()}",
            )
        )
        try:
            handle = marker.open("x")
        except FileExistsError as exc:
            raise OperationalException(
                f"Unreconciled Longbridge submission: {marker}. "
                "Check broker orders before recovering this marker."
            ) from exc
        record = {
            "symbol": symbol,
            "request_id": request_id,
            "side": side.value,
            "quantity": str(quantity),
            "created_at": order.created_at.isoformat(),
            "local_order_id": order.id,
            "portfolio_id": portfolio.id,
            "broker_environment": "paper" if self.client.paper else "live",
        }
        with handle:
            json.dump(record, handle)
            handle.flush()
            os.fsync(handle.fileno())
        response = self.client.trade.submit_order(
            symbol=symbol,
            order_type=(
                api.OrderType.LO
                if kind == OrderType.LIMIT
                else api.OrderType.MO
            ),
            side=(
                api.OrderSide.Buy
                if side == OrderSide.BUY
                else api.OrderSide.Sell
            ),
            submitted_quantity=quantity,
            time_in_force=api.TimeInForceType.Day,
            outside_rth=api.OutsideRTH.RTHOnly,
            remark="IAF strategy",
            client_request_id=request_id,
            **kwargs,
        )
        # Persist the acceptance before any further network calls. A failed
        # immediate detail lookup must not turn an accepted order into a retry.
        order.external_id = response.order_id
        order.status = OrderStatus.OPEN.value
        order.filled = 0
        order.remaining = order.amount
        record["order_id"] = response.order_id
        with marker.open("w") as handle:
            json.dump(record, handle)
            handle.flush()
            os.fsync(handle.fileno())
        return order

    def cancel_order(self, portfolio, order, market_credential):
        if not self.client.trading_enabled:
            raise OperationalException("Longbridge trading is disabled")
        self.client.trade.cancel_order(str(order.external_id))
        # Cancel API acknowledges a request; only order_detail confirms it.
        return update_order(
            order, self.client.trade.order_detail(str(order.external_id))
        )
