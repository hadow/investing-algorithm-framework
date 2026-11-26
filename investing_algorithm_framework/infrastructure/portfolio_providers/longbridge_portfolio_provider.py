from investing_algorithm_framework.domain import PortfolioProvider, Position
from investing_algorithm_framework.infrastructure.longbridge import (
    submission_marker, us_symbol,
)
from investing_algorithm_framework.infrastructure.order_executors\
    .longbridge_order_executor import update_order


class LongbridgePortfolioProvider(PortfolioProvider):
    def __init__(self, client, priority=1):
        super().__init__(priority=priority)
        self.client = client

    def supports_market(self, market):
        return market.upper() == "LONGBRIDGE"

    def get_order(self, portfolio, order, market_credential):
        if not order.external_id:
            return None
        # Propagate auth/network errors so reconciliation cannot treat them
        # as a missing or canceled order.
        # SQLite's INTEGER external_id column reloads numeric broker IDs as
        # ints; the SDK requires the original string at its API boundary.
        result = update_order(
            order, self.client.trade.order_detail(str(order.external_id))
        )
        # A persisted local order now identifies this broker submission.
        if self.config:
            submission_marker(self.config, order.target_symbol).unlink(
                missing_ok=True
            )
        return result

    def get_position(self, portfolio, symbol, market_credential):
        if symbol == portfolio.trading_symbol:
            balances = self.client.trade.account_balance()
            cash = sum(
                info.available_cash for balance in balances
                for info in balance.cash_infos if info.currency == symbol
            )
            return Position(symbol=symbol, amount=float(cash), cost=0,
                            portfolio_id=portfolio.id)
        security = us_symbol(symbol)
        response = self.client.trade.stock_positions([security])
        positions = [
            p for channel in response.channels for p in channel.positions
            if p.symbol == security
        ]
        if not positions:
            return None
        return Position(
            symbol=symbol,
            amount=float(sum(p.quantity for p in positions)),
            cost=float(sum(p.quantity * p.cost_price for p in positions)),
            portfolio_id=portfolio.id,
        )
