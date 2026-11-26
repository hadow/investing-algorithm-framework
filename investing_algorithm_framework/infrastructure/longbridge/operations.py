"""运行诊断与只读预检；所有券商调用均为查询，文件输出用于本机观察。"""

from datetime import datetime, timezone
import math
from pathlib import Path

import pandas as pd

from investing_algorithm_framework import (
    LongbridgeOHLCVDataProvider,
    LongbridgePortfolioProvider,
    LongbridgeTickerDataProvider,
)
from ..data_providers.longbridge import (
    candle_frame,
    is_us_regular_session,
    latest_completed_us_bar,
)
from investing_algorithm_framework.infrastructure.longbridge import sdk
from .runtime import WORKING, atomic_json
from .ledger import read_peer_ledgers


class IsolatedPortfolioProvider(LongbridgePortfolioProvider):
    """隔离单笔订单查询失败；现金等账户级错误仍按底层规则抛出。"""

    def get_order(self, portfolio, order, market_credential):
        """失败时保留原委托和提交标记，记录阻断原因；成功对账后自动清除。

        None 仅让订单服务跳过本次更新，不能解释为撤单或成交。RuntimeControls
        读取同一错误表，暂停新仓与该订单的撤单／卖出，其他股票继续同步和退出。
        """
        issues = self.client.__dict__.setdefault("order_issues", {})
        key = str(order.id)
        try:
            result = super().get_order(portfolio, order, market_credential)
            if result is not None:
                issues.pop(key, None)
            return result
        except Exception as exc:
            issues[key] = f"{order.target_symbol}: {exc}"
            return None


class IsolatedOHLCVProvider(LongbridgeOHLCVDataProvider):
    """单只股票行情失败时返回空帧，让其他股票的退出仍能继续。"""

    def get_data(self, *args, **kwargs):
        """记录具体数据源错误；成功读取后清除，空帧不会被当作可交易行情。

        框架统一获取所有策略数据，单一源抛错会阻断整轮。隔离只应用于账户入口，
        研究和底层通用 provider 仍保留原有抛错行为，健康报告显式展示这里的失败。
        """
        issues = self.client.__dict__.setdefault("data_issues", {})
        key = f"{self.symbol}:{self.time_frame}"
        try:
            result = super().get_data(*args, **kwargs)
            issues.pop(key, None)
            return result
        except Exception as exc:
            issues[key] = str(exc)
            frame = candle_frame([])
            if self.pandas:
                return pd.DataFrame(
                    columns=["Open", "High", "Low", "Close", "Volume"],
                    index=pd.DatetimeIndex([], tz="UTC"),
                )
            return frame


class IsolatedTickerProvider(LongbridgeTickerDataProvider):
    """报价异常隔离到对应股票，禁止以无效价格执行委托。"""

    def get_data(self, *args, **kwargs):
        """失败时保留明确的空盘口和错误原因，其他标的仍使用各自的有效报价。"""
        issues = self.client.__dict__.setdefault("data_issues", {})
        key = f"{self.symbol}:ticker"
        try:
            result = super().get_data(*args, **kwargs)
            issues.pop(key, None)
            return result
        except Exception as exc:
            issues[key] = str(exc)
            return {
                "symbol": self.symbol,
                "market": "LONGBRIDGE",
                "datetime": datetime.now(timezone.utc),
                "last": None,
                "bid": None,
                "ask": None,
                "volume": 0,
            }


def preflight(
    client,
    app,
    symbols,
    capital,
    selection_issue=None,
    time_frame="1h",
    minimum_bars=250,
    peer_directories=(),
    heartbeat_required=True,
):
    """只读核对券商环境、现金、持仓、有效委托、行情和本地提交记录。

    关闭交易权限后执行，不调用 App.run。先通过只读订单回报同步本地成交，
    再与券商总持仓比较；账户中的外部持仓或未知挂单要求人工对账，不自动接管。
    盘外旧报价不视为故障，盘中则必须在120秒内；主周期仍须匹配交易日历。
    返回 ready/issues/checks，问题用于暂停新仓，不能关闭已有仓位的退出链路。
    """
    if client.trading_enabled:
        raise ValueError("Preflight requires trading_enabled=False")
    issues = []
    now = datetime.now(timezone.utc)
    balances = client.trade.account_balance()
    cash = float(
        sum(
            info.available_cash
            for balance in balances
            for info in balance.cash_infos
            if info.currency == "USD"
        )
    )
    broker_positions = {
        p.symbol: float(p.quantity)
        for channel in client.trade.stock_positions().channels
        for p in channel.positions
        if p.quantity != 0
    }
    broker_orders = client.trade.today_orders()
    app.initialize_config()
    app.initialize_storage()
    # 先注册持仓所需行情，再同步成交和计算净值；预检本身不会执行策略。
    app.initialize_data_sources(app.get_algorithm().data_sources)
    app.initialize_services()
    app.initialize_portfolios()
    orders = app.container.order_service().get_all()
    positions = app.container.position_service().get_all()
    local_positions = {
        p.symbol: float(p.amount)
        for p in positions
        if p.symbol != "USD" and p.amount != 0
    }
    peer_positions, peer_orders, peer_cash, peer_issues = read_peer_ledgers(
        peer_directories,
        app.config["RESOURCE_DIRECTORY"],
        client.paper,
    )
    expected_positions = dict(peer_positions)
    for symbol, quantity in local_positions.items():
        expected_positions[symbol] = (
            expected_positions.get(symbol, 0) + quantity
        )
    if broker_positions != expected_positions:
        issues.append(
            "Broker/local positions differ; reconcile before new entries"
        )
    portfolios = app.container.portfolio_service().get_all()
    required_cash = (
        min(capital, portfolios[0].unallocated) if portfolios else capital
    )
    if not math.isfinite(cash) or cash < required_cash + peer_cash:
        issues.append(
            "Broker USD available cash does not cover the local cash budget"
        )
    api = sdk()
    terminal = (
        api.OrderStatus.Filled,
        api.OrderStatus.Canceled,
        api.OrderStatus.PartialWithdrawal,
        api.OrderStatus.Expired,
        api.OrderStatus.Rejected,
    )
    issues.extend(peer_issues)
    known = {
        str(order.external_id) for order in orders if order.external_id
    } | peer_orders
    if any(
        order.status not in terminal and str(order.order_id) not in known
        for order in broker_orders
    ):
        issues.append(
            "Unknown working broker orders; reconcile before new entries"
        )
    if any(
        order.status in WORKING and not order.external_id for order in orders
    ):
        issues.append(
            "Local submission without broker order ID needs reconciliation"
        )
    directory = Path(app.config["RESOURCE_DIRECTORY"])
    if list(directory.glob("longbridge_submissions/*.json")):
        issues.append("Unreconciled submission marker remains")
    entry_blockers = list(issues)
    issues.extend(client.__dict__.get("order_issues", {}).values())
    if selection_issue:
        issues.append(selection_issue)
    checks = {}
    regular = is_us_regular_session(client, now, False)
    for symbol in symbols:
        try:
            candles = client.get_candlesticks(symbol, time_frame, count=300)
            heartbeat = (
                client.get_candlesticks(symbol, "1m", count=2)
                if heartbeat_required
                else []
            )
            quote = client.get_quote(symbol)
            depth = client.get_depth(symbol)
            expected = latest_completed_us_bar(client, time_frame, now)
            last = (
                candles[-1].timestamp.astimezone(timezone.utc)
                if candles
                else None
            )
            age = (
                now - quote.timestamp.astimezone(timezone.utc)
            ).total_seconds()
            book = bool(
                depth.bids
                and depth.asks
                and depth.bids[0].price
                and depth.asks[0].price
                and 0 < depth.bids[0].price <= depth.asks[0].price
            )
            checks[symbol] = {
                "completed_bars": len(candles),
                "heartbeat_bars": len(heartbeat),
                "last_bar": last.isoformat() if last else None,
                "expected_bar": expected.isoformat(),
                "order_book": book,
                "quote_age_seconds": round(age, 1),
            }
            if (
                len(candles) < minimum_bars
                or (heartbeat_required and len(heartbeat) < 1)
                or last != expected
                or not book
            ):
                issues.append(
                    f"{symbol}: incomplete/stale candles or missing order book"
                )
            if regular and (
                not 0 <= age <= 120
                or quote.trade_status != api.TradeStatus.Normal
            ):
                issues.append(f"{symbol}: quote stale or trading suspended")
        except Exception as exc:
            issues.append(f"{symbol}: {exc}")
    return {
        "checked_at": now.isoformat(),
        "ready": not issues,
        "broker_environment": "paper" if client.paper else "live",
        "trading_enabled": False,
        "issues": issues,
        "checks": checks,
        "entry_blockers": entry_blockers,
        "usd_available_cash": cash,
        "broker_positions": broker_positions,
        "strategy_positions": local_positions,
        "peer_strategy_positions": peer_positions,
        "strategy_cash_budget": required_cash,
        "peer_cash_budgets": peer_cash,
        "broker_orders_today": len(broker_orders),
    }


def runtime_report(client, app, directory, report=None, *, risk_reader=None, details=None):
    """把框架执行痕迹、有效委托和持仓写入本机健康文件，委托异常标记 degraded。

    框架会捕获部分执行器异常，因此 completed 不能单独代表成功。汇总每只策略
    的水位、阻断原因及 error/declined 痕迹，并列出实际挂单数量供运行者核对。
    调用方通过 details 补充候选和订阅信息，与基础状态合并后一次性发布 health.json。
    """
    strategies = app.get_algorithm().strategies
    states, issues = {}, dict(client.__dict__.get("data_issues", {}))
    if client.__dict__.get("entry_blockers"):
        issues["account"] = client.entry_blockers
    if client.__dict__.get("order_issues"):
        issues["order_sync"] = dict(client.order_issues)
    for strategy in strategies:
        state = dict(strategy.runtime.status) if strategy.runtime else {}
        traces = getattr(strategy, "traces", [])
        state["execution_events"] = [
            {"event": tag, "detail": str(payload)}
            for tag, payload in traces
            if tag.startswith("emit_orders.")
        ]
        if state.get("error") or state.get("reason") == "needs_reconciliation":
            issues[strategy.symbols[0]] = state.get("error", state["reason"])
        if state.get("reason") in (
            "stale_or_unconfirmed_bar",
            "insufficient_bars",
        ):
            issues[strategy.symbols[0]] = state["reason"]
        if state.get("risk_check") == "quote_stale_or_suspended":
            issues[strategy.symbols[0]] = "ATR check deferred: invalid quote"
        for tag, payload in traces:
            if tag.endswith("error"):
                issues[strategy.symbols[0]] = str(payload)
            if (
                tag == "emit_orders.declined"
                and payload.side.value == "close_long"
            ):
                reason = client.__dict__.get("execution_issues", {}).get(
                    strategy.symbols[0]
                )
                if reason != "outside_regular_session":
                    issues[strategy.symbols[0]] = (
                        f"Exit deferred: {reason or 'executor declined'}"
                    )
        states[strategy.symbols[0]] = state
    orders = app.container.order_service().get_all()
    result = {
        **(details or {}),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "degraded"
            if issues or (report and report.get("status") == "failed")
            else "completed"
        ),
        "broker_environment": "paper" if client.paper else "live",
        "trading_enabled": client.trading_enabled,
        "entries_paused": (Path(directory) / "pause_entries").exists(),
        "entry_blockers": client.__dict__.get("entry_blockers", []),
        "execution_reasons": client.__dict__.get("execution_issues", {}),
        "issues": issues,
        "strategies": states,
        "working_orders": [
            {
                "id": order.id,
                "broker_id": order.external_id,
                "symbol": order.target_symbol,
                "side": order.order_side,
                "status": order.status,
                "filled": order.filled,
                "remaining": order.remaining,
            }
            for order in orders
            if order.status in WORKING
        ],
        "orders_total": len(orders),
        "positions": {
            p.symbol: p.amount
            for p in app.container.position_service().get_all()
            if p.symbol != "USD" and p.amount != 0
        },
    }
    contexts = [
        strategy.context
        for strategy in strategies
        if strategy.context is not None
    ]
    if contexts and risk_reader is not None:
        try:
            occupied, used = risk_reader(contexts[0])
            result.update(
                risk_used_usd=used, occupied_symbols=sorted(occupied)
            )
        except (ValueError, KeyError, TypeError) as exc:
            result["issues"]["risk"] = str(exc)
            result["status"] = "degraded"
    atomic_json(Path(directory) / "health.json", result)
    return result
