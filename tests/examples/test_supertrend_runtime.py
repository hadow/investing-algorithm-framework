"""Account controls: cancellation races, restarts and once-per-bar decisions."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from investing_algorithm_framework import (
    LongbridgeClient,
    LongbridgeOAuthCredential,
    Signal,
    SignalSide,
)
from investing_algorithm_framework.infrastructure.data_providers.longbridge import (
    is_us_regular_session,
    latest_completed_us_bar,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/algorithm_examples/strategies_showcase/27_supertrend_rsi"
NOW = datetime(2026, 11, 9, 16, 0, tzinfo=timezone.utc)
BAR = datetime(2026, 11, 9, 14, 30, tzinfo=timezone.utc)


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return importlib.import_module("investing_algorithm_framework.infrastructure.longbridge.runtime")


@pytest.fixture
def setup_runtime(runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "latest_completed_us_bar", lambda *a: BAR)
    monkeypatch.setattr(runtime, "is_us_regular_session", lambda *a: True)
    api = pytest.importorskip("longbridge.openapi")
    client = NS(trading_enabled=True, get_quote=Mock(return_value=NS(
        timestamp=NOW, trade_status=api.TradeStatus.Normal
    )))
    control = runtime.RuntimeControls(client, tmp_path, clock=lambda: NOW)
    strategy = NS(
        strategy_id="test-SPY",
        symbols=["SPY.US"],
        symbol="SPY.US/USD",
        market="LONGBRIDGE",
        data_key="ohlcv",
        atr_period=10,
        rsi_period=14,
        time_frame="1h",
        _generate_uncontrolled_signals=Mock(return_value=[]),
    )
    strategy._generate_core_signals = strategy._generate_uncontrolled_signals
    strategy.generate_risk_signals = lambda context: [
        signal
        for signal in strategy._generate_uncontrolled_signals.return_value
        if signal.metadata.get("risk_exit")
    ]
    context = Mock()
    position = NS(amount=4)
    orders = []
    context.get_orders.side_effect = lambda **kwargs: list(orders)
    context.get_position.return_value = position
    data = {"ohlcv": pd.DataFrame(index=pd.date_range(end=BAR, periods=20, freq="h"))}
    return control, strategy, context, position, orders, data


def order(side="BUY", kind="LIMIT", **kwargs):
    return NS(
        id=1,
        external_id="broker-1",
        status="OPEN",
        order_side=side,
        order_type=kind,
        created_at=NOW - timedelta(seconds=10),
        metadata={},
        **kwargs,
    )


def test_idle_ticks_query_orders_once_and_keep_checking_risk(setup_runtime):
    control, strategy, context, position, orders, data = setup_runtime
    position.amount = 0
    strategy.generate_risk_signals = Mock(return_value=[])
    for _ in range(3):
        assert list(control.generate(strategy, context, data)) == []
    assert context.get_orders.call_count == 3
    assert strategy.generate_risk_signals.call_count == 3
    strategy._generate_core_signals.assert_called_once()


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_exit_waits_for_cancel_and_survives_rebound_restart(
    setup_runtime, runtime, tmp_path, side
):
    control, strategy, context, position, orders, data = setup_runtime
    orders.append(order(side))
    strategy._generate_uncontrolled_signals.return_value = [
        Signal("SPY.US", SignalSide.CLOSE_LONG, metadata={"risk_exit": True})
    ]
    assert list(control.generate(strategy, context, data)) == []
    assert control.status["reason"] == "waiting_cancel_confirmation"
    context.order_service.cancel_order.assert_called_once()
    assert json.loads(control.path.read_text())["exit"]["metadata"]["risk_exit"]
    # A cancel acknowledgement is not terminal. During the race, another buy
    # fill increases the held quantity; the next evaluation must use this value.
    position.amount = 5 if side == "BUY" else 3
    orders.clear()
    strategy._generate_uncontrolled_signals.return_value = []  # price rebounded
    restarted = runtime.RuntimeControls(control.client, tmp_path, clock=lambda: NOW)
    signals = list(restarted.generate(strategy, context, data))
    assert len(signals) == 1 and signals[0].metadata["risk_exit"]
    assert context.get_position.return_value.amount == position.amount
    # A known market exit already in flight prevents duplicate sells.
    pending = order("SELL", "MARKET")
    pending.metadata = signals[0].metadata
    orders.append(pending)
    assert list(restarted.generate(strategy, context, data)) == []
    assert restarted.status["reason"] == "exit_order_working"
    orders.clear()
    position.amount = 0
    assert list(restarted.generate(strategy, context, data)) == []
    assert json.loads(restarted.path.read_text())["exit"] is None
    assert list(restarted.generate(strategy, context, data)) == []


def test_timeout_sell_waits_for_terminal_then_falls_back_to_market(setup_runtime):
    control, strategy, context, position, orders, data = setup_runtime
    pending = order("SELL")
    pending.created_at = NOW - timedelta(seconds=121)
    orders.append(pending)
    assert list(control.generate(strategy, context, data)) == []
    assert control.status["reason"] == "waiting_cancel_confirmation"
    position.amount = 2  # partial sell before cancellation
    orders.clear()
    signals = list(control.generate(strategy, context, data))
    assert len(signals) == 1 and signals[0].metadata["market_exit"]


def test_timeout_buy_cancels_remainder_without_liquidating_fill(setup_runtime):
    control, strategy, context, position, orders, data = setup_runtime
    pending = order()
    pending.created_at = NOW - timedelta(seconds=121)
    orders.append(pending)
    context.order_service.cancel_order.side_effect = lambda o: orders.clear()
    assert list(control.generate(strategy, context, data)) == []
    context.order_service.cancel_order.assert_called_once()
    assert position.amount == 4
    assert json.loads(control.path.read_text())["exit"] is None


def test_same_bar_entry_exit_not_replayed_after_fill_or_restart(
    setup_runtime, runtime, tmp_path
):
    control, strategy, context, position, orders, data = setup_runtime
    position.amount = 0
    strategy._generate_uncontrolled_signals.return_value = [
        Signal(
            "SPY.US", SignalSide.CLOSE_LONG, metadata={"signal_bar": BAR.isoformat()}
        ),
        Signal(
            "SPY.US", SignalSide.OPEN_LONG, metadata={"signal_bar": BAR.isoformat()}
        ),
    ]
    assert [s.side for s in control.generate(strategy, context, data)] == [
        SignalSide.OPEN_LONG
    ]
    position.amount = 4
    restarted = runtime.RuntimeControls(control.client, tmp_path, clock=lambda: NOW)
    assert list(restarted.generate(strategy, context, data)) == []
    assert restarted.status["reason"] == "bar_already_processed"


def test_stale_bars_block_entries_but_not_latched_exit(setup_runtime):
    control, strategy, context, position, orders, data = setup_runtime
    data["ohlcv"].index -= timedelta(days=4)
    strategy._generate_uncontrolled_signals.return_value = [
        Signal("SPY.US", SignalSide.OPEN_LONG)
    ]
    position.amount = 0
    assert list(control.generate(strategy, context, data)) == []
    assert control.status["reason"] == "stale_or_unconfirmed_bar"
    position.amount = 4
    strategy._generate_uncontrolled_signals.return_value = [
        Signal("SPY.US", SignalSide.CLOSE_LONG, metadata={"risk_exit": True})
    ]
    assert len(list(control.generate(strategy, context, data))) == 1


def test_pause_entries_keeps_exit_chain(setup_runtime, tmp_path):
    control, strategy, context, position, orders, data = setup_runtime
    (tmp_path / "pause_entries").touch()
    position.amount = 0
    strategy._generate_uncontrolled_signals.return_value = [
        Signal("SPY.US", SignalSide.OPEN_LONG)
    ]
    assert list(control.generate(strategy, context, data)) == []
    assert control.status["reason"] == "entries_paused"
    position.amount = 4
    strategy._generate_uncontrolled_signals.return_value = [
        Signal("SPY.US", SignalSide.CLOSE_LONG, metadata={"risk_exit": True})
    ]
    assert len(list(control.generate(strategy, context, data))) == 1


def test_unknown_submission_never_cancels_or_retries(setup_runtime):
    control, strategy, context, position, orders, data = setup_runtime
    pending = order()
    pending.status, pending.external_id = "CREATED", None
    orders.append(pending)
    strategy._generate_uncontrolled_signals.return_value = [
        Signal("SPY.US", SignalSide.CLOSE_LONG, metadata={"risk_exit": True})
    ]
    assert list(control.generate(strategy, context, data)) == []
    assert control.status["reason"] == "needs_reconciliation"
    context.order_service.cancel_order.assert_not_called()


def test_numeric_order_id_matches_accepted_marker(setup_runtime, runtime, tmp_path):
    control, strategy, context, position, orders, data = setup_runtime
    pending = order()
    pending.external_id = 1304467974119231488
    runtime.atomic_json(
        tmp_path / "longbridge_submissions/SPY.US.json",
        {
            "order_id": str(pending.external_id),
        },
    )
    assert not control._unknown_submission(strategy, [pending])


def test_numeric_order_id_preflight_recognizes_working_broker_order(
    runtime, monkeypatch, tmp_path
):
    api = pytest.importorskip("longbridge.openapi")
    operations = importlib.import_module("operations")
    broker_id = "1304467974119231488"
    client = NS(paper=True, trading_enabled=False, trade=Mock())
    client.trade.account_balance.return_value = [
        NS(cash_infos=[NS(currency="USD", available_cash=Decimal("2000"))])
    ]
    client.trade.stock_positions.return_value = NS(channels=[])
    client.trade.today_orders.return_value = [
        NS(order_id=broker_id, status=api.OrderStatus.New)
    ]
    app = Mock()
    app.config = {"RESOURCE_DIRECTORY": str(tmp_path)}
    app.container.order_service.return_value.get_all.return_value = [
        NS(external_id=int(broker_id), status="OPEN")
    ]
    app.container.position_service.return_value.get_all.return_value = []
    app.container.portfolio_service.return_value.get_all.return_value = [
        NS(unallocated=2000)
    ]
    monkeypatch.setattr("investing_algorithm_framework.infrastructure.longbridge.operations.is_us_regular_session", lambda *args: False)
    report = operations.preflight(client, app, [], 2000)
    assert report["ready"], report["issues"]


def test_latched_exit_does_not_need_core_candles(setup_runtime):
    control, strategy, context, position, orders, data = setup_runtime
    strategy._generate_uncontrolled_signals.return_value = [
        Signal("SPY.US", SignalSide.CLOSE_LONG, metadata={"risk_exit": True})
    ]
    assert len(list(control.generate(strategy, context, data))) == 1
    strategy._generate_uncontrolled_signals.return_value = []
    strategy._generate_core_signals.side_effect = RuntimeError("main OHLCV broken")
    data.clear()
    assert len(list(control.generate(strategy, context, data))) == 1


@pytest.mark.parametrize("invalid", ["stale", "future", "suspended", "outside_rth"])
def test_invalid_quote_cannot_latch_atr_exit(runtime, monkeypatch, tmp_path, invalid):
    api = pytest.importorskip("longbridge.openapi")
    module = importlib.import_module("supertrend_rsi")
    strategy = module.SupertrendRSIStrategy(stop_atr_multiple=2)
    client = NS(trading_enabled=True, get_quote=Mock())
    quote = NS(timestamp=NOW, trade_status=api.TradeStatus.Normal)
    client.get_quote.return_value = quote
    if invalid == "stale":
        quote.timestamp -= timedelta(minutes=10)
    elif invalid == "future":
        quote.timestamp += timedelta(seconds=1)
    elif invalid == "suspended":
        quote.trade_status = api.TradeStatus.Halted
    regular = invalid != "outside_rth"
    monkeypatch.setattr(runtime, "is_us_regular_session", lambda *a: regular)
    control = runtime.RuntimeControls(client, tmp_path, clock=lambda: NOW)
    context = Mock()
    context.get_open_trades.return_value = [
        NS(id=1, strategy_id=strategy.strategy_id, open_price=100,
           buy_order=NS(metadata={"stop_distance": 5}))
    ]
    context.get_orders.return_value = []
    context.get_position.return_value = NS(amount=12)
    context.get_latest_price.return_value = 94
    assert list(control.generate(strategy, context, {})) == []
    assert not control.path.exists()  # A rejected risk quote cannot create an exit.
    # A fresh rebound must not replay a stop inferred from the bad quote.
    regular = True
    quote.timestamp, quote.trade_status = NOW, api.TradeStatus.Normal
    context.get_latest_price.return_value = 100
    assert list(control.generate(strategy, context, {})) == []
    context.get_latest_price.return_value = 94
    assert len(list(control.generate(strategy, context, {}))) == 1
    # A genuine stop remains committed even if prices later rebound.
    context.get_latest_price.return_value = 100
    assert len(list(control.generate(strategy, context, {}))) == 1


def test_calendar_full_exit_window_entry_buffers_weekend_and_half_day():
    client = LongbridgeClient(LongbridgeOAuthCredential("test"))
    client.quote = Mock()
    client.quote.trading_days.return_value = NS(
        trading_days=[date(2026, 11, 6), date(2026, 11, 9), date(2026, 11, 27)],
        half_trading_days=[date(2026, 11, 27)],
    )

    def stamp(day, hour, minute):
        return datetime(2026, 11, day, hour, minute, tzinfo=timezone.utc)

    assert is_us_regular_session(client, stamp(9, 14, 31), False)
    assert not is_us_regular_session(client, stamp(9, 14, 31))
    assert is_us_regular_session(client, stamp(9, 20, 59), False)
    assert not is_us_regular_session(client, stamp(9, 20, 59))
    assert not is_us_regular_session(client, stamp(9, 21, 0), False)
    assert is_us_regular_session(client, stamp(27, 17, 59), False)
    assert not is_us_regular_session(client, stamp(27, 18, 0), False)
    assert latest_completed_us_bar(client, "1h", stamp(9, 15, 29)) == stamp(6, 20, 30)
    assert latest_completed_us_bar(client, "1h", stamp(8, 18, 0)) == stamp(6, 20, 30)
    assert latest_completed_us_bar(client, "1h", stamp(9, 15, 30)) == stamp(9, 14, 30)
    assert latest_completed_us_bar(client, "1h", stamp(27, 18, 0)) == stamp(27, 17, 30)


@pytest.mark.parametrize("restart", [False, True])
def test_broker_cycle_partial_buy_cancel_race_and_restarted_exit(
    runtime, monkeypatch, tmp_path, restart
):
    api = pytest.importorskip("longbridge.openapi")
    runner = importlib.import_module("longbridge_app")
    reference = pd.read_csv(ROOT / "tests/resources/supertrend_rsi_pine.csv")
    index = int(np.flatnonzero(reference.entry_long.to_numpy())[0])
    frame = reference.iloc[: index + 1].copy()
    frame.index = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(hours=1),
        periods=len(frame),
        freq="h",
    )
    monkeypatch.setattr(
        runtime, "latest_completed_us_bar", lambda *a: frame.index[-1].to_pydatetime()
    )
    monkeypatch.setattr(runtime, "is_us_regular_session", lambda *a: True)
    monkeypatch.setattr(
        "investing_algorithm_framework.infrastructure.longbridge.execution."
        "is_us_regular_session",
        lambda *a: True,
    )
    client = LongbridgeClient(LongbridgeOAuthCredential("test"), trading_enabled=True)
    client.quote, client.trade = Mock(), Mock()
    client.quote.quote.return_value = [
        NS(
            timestamp=datetime.now(timezone.utc),
            last_done=Decimal("100"),
            volume=10,
            trade_status=api.TradeStatus.Normal,
        )
    ]
    client.quote.depth.return_value = NS(
        bids=[NS(price=Decimal("99.99"))], asks=[NS(price=Decimal("100"))]
    )
    client.quote.subscribe_candlesticks.return_value = [
        NS(
            timestamp=t,
            open=r.Open,
            high=r.High,
            low=r.Low,
            close=r.Close,
            volume=r.Volume,
        )
        for t, r in frame.iterrows()
    ]
    client.trade.account_balance.return_value = [
        NS(cash_infos=[NS(currency="USD", available_cash=Decimal("20000"))])
    ]
    client.trade.stock_positions.return_value = NS(channels=[])
    client.trade.submit_order.return_value = NS(order_id="buy-1")
    app = runner.build_app(client, tmp_path, capital=20000)
    app.run(number_of_iterations=1)
    buy = app.container.order_service().get_all()[0]
    assert buy.amount >= 4
    total = Decimal(str(buy.amount))
    detail = NS(
        order_id="buy-1",
        status=api.OrderStatus.PartialFilled,
        quantity=total,
        executed_quantity=Decimal("2"),
        executed_price=Decimal("100"),
        updated_at=datetime.now(timezone.utc),
        charge_detail=None,
    )
    client.trade.order_detail.return_value = detail
    app.container.order_service().check_pending_orders()
    distance = buy.metadata["stop_distance"]
    client.quote.set_on_quote.call_args.args[0](
        "SPY.US",
        NS(
            timestamp=datetime.now(timezone.utc),
            last_done=Decimal(str(99 - distance)),
            volume=20,
            trade_status=api.TradeStatus.Normal,
        ),
    )
    client.quote.set_on_depth.call_args.args[0]("SPY.US", NS())
    client.quote.realtime_depth.return_value = NS(
        bids=[NS(price=Decimal(str(99 - distance)))], asks=[NS(price=Decimal("100"))]
    )
    detail.status = api.OrderStatus.PendingCancel
    if restart:
        app = runner.build_app(client, tmp_path, capital=20000)
    app.run(number_of_iterations=1)
    client.trade.cancel_order.assert_called_once_with("buy-1")
    assert client.trade.submit_order.call_count == 1
    # A third share fills during cancellation, then the market rebounds.
    detail.status, detail.executed_quantity = (
        api.OrderStatus.PartialWithdrawal,
        Decimal("3"),
    )
    client.quote.realtime_depth.return_value = NS(
        bids=[NS(price=Decimal("101"))], asks=[NS(price=Decimal("101.01"))]
    )
    client.trade.stock_positions.return_value = NS(
        channels=[NS(positions=[NS(symbol="SPY.US", available_quantity=Decimal("3"))])]
    )
    client.trade.submit_order.return_value = NS(order_id="sell-1")
    if restart:
        app = runner.build_app(client, tmp_path, capital=20000)
    app.run(number_of_iterations=1)
    request = client.trade.submit_order.call_args.kwargs
    assert client.trade.submit_order.call_count == 2
    assert request["order_type"] == api.OrderType.MO
    assert request["side"] == api.OrderSide.Sell
    assert request["submitted_quantity"] == Decimal("3")
    assert app.container.position_service().find({"symbol": "SPY.US"}).amount == 3
    client.close()


def test_preflight_reads_paper_account_and_detects_orphan_marker(
    runtime, monkeypatch, tmp_path
):
    api = pytest.importorskip("longbridge.openapi")
    operations = importlib.import_module("operations")
    runner = importlib.import_module("longbridge_app")
    client = LongbridgeClient(
        LongbridgeOAuthCredential("test"), paper=True, trading_enabled=False
    )
    client.quote, client.trade = Mock(), Mock()
    client.trade.account_balance.return_value = [
        NS(cash_infos=[NS(currency="USD", available_cash=Decimal("2000"))])
    ]
    client.trade.stock_positions.return_value = NS(channels=[])
    client.trade.today_orders.return_value = []
    candles = [
        NS(
            timestamp=BAR - timedelta(hours=299 - i),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1000,
        )
        for i in range(300)
    ]
    client.get_candlesticks = Mock(return_value=candles)
    client.get_quote = Mock(
        return_value=NS(
            timestamp=datetime.now(timezone.utc), trade_status=api.TradeStatus.Normal
        )
    )
    client.get_depth = Mock(return_value=NS(bids=[NS(price=99)], asks=[NS(price=100)]))
    monkeypatch.setattr("investing_algorithm_framework.infrastructure.longbridge.operations.is_us_regular_session", lambda *a: False)
    monkeypatch.setattr("investing_algorithm_framework.infrastructure.longbridge.operations.latest_completed_us_bar", lambda *a: BAR)
    app = runner.build_app(
        client, tmp_path, capital=2000, order_budget=1000, min_entry_shares=10
    )
    result = operations.preflight(client, app, ["SPY.US"], 2000)
    assert result["ready"] and result["broker_environment"] == "paper"
    assert result["trading_enabled"] is False
    client.trade.submit_order.assert_not_called()
    client.trade.cancel_order.assert_not_called()
    marker = tmp_path / "longbridge_submissions" / "SPY.US.json"
    runtime.atomic_json(marker, {"request_id": "unknown"})
    result = operations.preflight(client, app, ["SPY.US"], 2000)
    assert (
        not result["ready"]
        and "Unreconciled submission marker remains" in result["entry_blockers"]
    )
    assert marker.exists()
    client.trade.submit_order.assert_not_called()
    client.trade.cancel_order.assert_not_called()
    client.trading_enabled = True
    with pytest.raises(ValueError):
        operations.preflight(client, app, [], 2000)


def test_account_directory_cannot_switch_paper_live(runtime, tmp_path):
    runner = importlib.import_module("longbridge_app")
    client = LongbridgeClient(LongbridgeOAuthCredential("test"), paper=True)
    runner.configure_app(client, tmp_path)
    client.paper = False
    with pytest.raises(ValueError, match="account mode"):
        runner.configure_app(client, tmp_path)


def test_health_report_exposes_swallowed_order_error(runtime, tmp_path):
    operations = importlib.import_module("operations")
    runner = importlib.import_module("longbridge_app")
    client = LongbridgeClient(LongbridgeOAuthCredential("test"), paper=True)
    app = runner.build_app(client, tmp_path)
    app.initialize_config()
    app.initialize_storage()
    strategy = app.get_algorithm().strategies[0]
    strategy.traces = [("emit_orders.error", "request acknowledgement unknown")]
    result = operations.runtime_report(client, app, tmp_path)
    assert result["status"] == "degraded"
    assert "acknowledgement unknown" in result["issues"]["SPY.US"]
    assert json.loads((tmp_path / "health.json").read_text())["status"] == "degraded"


def test_one_symbol_quote_failure_does_not_abort_other_data(runtime):
    operations = importlib.import_module("operations")
    client = Mock()

    def candles(symbol, *args, **kwargs):
        if symbol == "AAA.US/USD":
            raise RuntimeError("AAA quote entitlement missing")
        return [NS(timestamp=BAR, open=100, high=101, low=99, close=100, volume=1)]

    client.get_candlesticks.side_effect = candles
    broken = operations.IsolatedOHLCVProvider(
        client, symbol="AAA.US/USD", time_frame="1h", pandas=True
    )
    healthy = operations.IsolatedOHLCVProvider(
        client, symbol="BBB.US/USD", time_frame="1h", pandas=True
    )
    assert broken.get_data().empty
    assert len(healthy.get_data()) == 1
    assert any("AAA" in message for message in client.data_issues.values())


def test_order_timeout_does_not_block_other_fills_and_recovers(
    runtime, tmp_path
):
    api = pytest.importorskip("longbridge.openapi")
    runner = importlib.import_module("longbridge_app")
    operations = importlib.import_module("operations")
    client = LongbridgeClient(LongbridgeOAuthCredential("test"), trading_enabled=True)
    client.trade = Mock()
    client.trade.account_balance.return_value = [NS(cash_infos=[
        NS(currency="USD", available_cash=Decimal("25000"))
    ])]
    client.trade.submit_order.side_effect = [
        NS(order_id="a-order"), NS(order_id="b-order")
    ]
    app = runner.configure_app(client, tmp_path)
    app.initialize_config()
    app.initialize_storage()
    app.initialize_services()
    app.initialize_portfolios()
    for symbol in ("AAA.US", "BBB.US"):
        app.context.create_limit_order(
            target_symbol=symbol, order_side="BUY", amount=10, price=100,
            execute=True, validate=True, sync=True, metadata={"stop_distance": 5}
        )
    failing = True

    def detail(order_id):
        if order_id == "a-order" and failing:
            raise TimeoutError("AAA order timeout")
        return NS(
            order_id=order_id, status=api.OrderStatus.Filled,
            executed_quantity=Decimal("10"), quantity=Decimal("10"),
            executed_price=Decimal("100"), updated_at=NOW, charge_detail=None,
        )

    client.trade.order_detail.side_effect = detail
    app.initialize_portfolios()
    assert client.trade.order_detail.call_count == 2
    positions = app.container.position_service()
    assert positions.find({"symbol": "BBB.US"}).amount == 10
    assert positions.find({"symbol": "AAA.US"}).amount == 0
    orders = app.container.order_service().get_all()
    bad = next(o for o in orders if o.target_symbol == "AAA.US")
    assert client.order_issues == {str(bad.id): "AAA.US: AAA order timeout"}
    report = operations.runtime_report(client, app, tmp_path)
    assert report["status"] == "degraded"
    assert report["issues"]["order_sync"] == client.order_issues
    # Only an actual successful query may release the reconciliation block.
    failing = False
    app.initialize_portfolios()
    assert positions.find({"symbol": "AAA.US"}).amount == 10
    assert client.order_issues == {}
    assert operations.runtime_report(client, app, tmp_path)["status"] == "completed"


def test_failed_order_sync_blocks_entries_but_keeps_healthy_exit(setup_runtime):
    control, strategy, context, position, orders, data = setup_runtime
    control.client.order_issues = {"99": "AAA.US: timed out"}
    position.amount = 0
    strategy._generate_core_signals.return_value = [
        Signal("SPY.US", SignalSide.OPEN_LONG)
    ]
    assert list(control.generate(strategy, context, data)) == []
    assert control.status["reason"] == "entries_paused"
    position.amount = 4
    strategy._generate_core_signals.return_value = [
        Signal("SPY.US", SignalSide.CLOSE_LONG, metadata={"risk_exit": True})
    ]
    assert len(list(control.generate(strategy, context, data))) == 1


def test_affected_order_waits_for_sync_before_cancel_or_sell(setup_runtime):
    control, strategy, context, position, orders, data = setup_runtime
    pending = order()
    orders.append(pending)
    control.client.order_issues = {str(pending.id): "SPY.US: timed out"}
    strategy._generate_core_signals.return_value = [
        Signal("SPY.US", SignalSide.CLOSE_LONG, metadata={"risk_exit": True})
    ]
    assert list(control.generate(strategy, context, data)) == []
    assert control.status["reason"] == "order_sync_pending"
    assert json.loads(control.path.read_text())["exit"] is not None
    context.order_service.cancel_order.assert_not_called()
