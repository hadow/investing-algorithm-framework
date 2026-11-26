"""共享账户的持仓、待成交买单风险及新仓额度限制。"""

import logging
import math

from . import us_symbol
from .execution import LongbridgeLimitExecutor
from .runtime import WORKING

logger = logging.getLogger(__name__)


def portfolio_risk(context):
    """返回已占用的标的集合，以及持仓和待成交买单的初始保护距离风险总额。

    从原买单 metadata 恢复固定 stop_distance；同一股票若有多笔成交，取其中
    最大距离作保守估计。持仓数量与成交可用数量取较大值，确保已经为卖单预留、
    但尚未卖出的股票仍计入风险；另加有效买单未成交余量乘以各自退出距离。
    同一股票的分批成交只在持仓与待成交余量间转移，不重复占用标的名额。
    无法解释的持仓、缺失退出距离或非法风险值直接抛错，供执行器暂停新开仓。
    此处衡量的是预算占用，滑点、跳空和进程离线后的实际亏损不由该值限定。
    """
    occupied, risk = set(), 0.0
    distances, available = {}, {}
    for trade in context.get_open_trades(market="LONGBRIDGE"):
        symbol = us_symbol(trade.target_symbol)
        occupied.add(symbol)
        distance = trade.buy_order.metadata.get("stop_distance")
        if not distance or not math.isfinite(distance) or distance <= 0:
            raise ValueError("Open trade has no valid persisted stop distance")
        distances[symbol] = max(distance, distances.get(symbol, 0))
        available[symbol] = available.get(symbol, 0) + float(
            trade.available_amount
        )
    positions = {
        us_symbol(p.symbol): float(p.amount)
        for p in context.get_positions(market="LONGBRIDGE")
        if p.symbol != "USD" and p.amount != 0
    }
    if not set(positions).issubset(occupied) or any(
        q < 0 for q in positions.values()
    ):
        raise ValueError("Position has no corresponding managed long trade")
    # 挂出卖单只预留数量，实际成交前持仓仍承受风险，不能提前释放组合额度。
    for symbol, distance in distances.items():
        risk += max(positions.get(symbol, 0), available[symbol]) * distance
    for order in context.get_orders(market="LONGBRIDGE"):
        if order.status not in WORKING or order.order_side != "BUY":
            continue
        occupied.add(us_symbol(order.target_symbol))
        distance = order.metadata.get("stop_distance")
        if not distance or not math.isfinite(distance) or distance <= 0:
            raise ValueError(
                "Working buy has no valid persisted stop distance"
            )
        risk += float(order.remaining) * distance
    if not math.isfinite(risk) or risk < 0:
        raise ValueError("Invalid portfolio stop risk")
    return occupied | set(positions), risk


class PortfolioLimitExecutor(LongbridgeLimitExecutor):
    """实际买价确定后，合并所有策略的持仓和挂单限制组合风险。"""

    def __init__(
        self,
        client,
        max_positions=4,
        total_risk_fraction=0.03,
        max_spread_fraction=0.0015,
        min_entry_shares=1,
    ):
        super().__init__(client, min_entry_shares)
        if type(max_positions) is not int or max_positions < 1:
            raise ValueError("max_positions must be a positive integer")
        if not all(
            math.isfinite(v) and 0 < v <= 1
            for v in (total_risk_fraction, max_spread_fraction)
        ):
            raise ValueError("Invalid portfolio risk or spread limit")
        self.max_positions = max_positions
        self.total_risk_fraction = total_risk_fraction
        self.max_spread_fraction = max_spread_fraction

    def cap_entry_amount(self, intent, context, metadata, amount, depth):
        try:
            occupied, used_risk = portfolio_risk(context)
            if (
                len(occupied) >= self.max_positions
                or us_symbol(intent.symbol) in occupied
            ):
                return self.decline(
                    intent, "portfolio_slots_or_existing_position"
                )
            if not depth.bids or not depth.asks:
                return self.decline(intent, "missing_order_book")
            bid, ask = float(depth.bids[0].price), float(depth.asks[0].price)
            if (
                not all(math.isfinite(p) and p > 0 for p in (bid, ask))
                or ask < bid
            ):
                return self.decline(intent, "invalid_order_book_price")
            if (ask - bid) / ((ask + bid) / 2) > self.max_spread_fraction:
                return self.decline(intent, "spread_above_limit")
            distance = metadata["stop_distance"]
            if not math.isfinite(distance) or distance <= 0:
                return self.decline(intent, "invalid_stop_distance")
            headroom = (
                context.get_portfolio().get_net_size()
                * self.total_risk_fraction
                - used_risk
            )
            # 父类已按同一份盘口把绝对保护价转成距离；不能先用旧距离估算总风险。
            capacity = math.floor(max(0.0, headroom) / distance)
            if capacity < 1:
                return self.decline(intent, "portfolio_risk_below_one_share")
            return min(amount, capacity)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("New entry paused for %s: %s", intent.symbol, exc)
            return self.decline(intent, str(exc))
