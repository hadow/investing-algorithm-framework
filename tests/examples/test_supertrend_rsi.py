"""Pine signal parity, risk execution, and IAF runs with a fake broker."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from investing_algorithm_framework import Signal, SignalSide
from investing_algorithm_framework.infrastructure.longbridge.execution import (
    LongbridgeLimitExecutor,
)
from investing_algorithm_framework.services.strategy_phases import SizedIntent

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/algorithm_examples/strategies_showcase/27_supertrend_rsi"
REFERENCE = ROOT / "tests/resources/supertrend_rsi_pine.csv"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return importlib.import_module("supertrend_rsi")


@pytest.fixture
def reference():
    return pd.read_csv(REFERENCE, parse_dates=["Datetime"]).set_index("Datetime")


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"supertrend_rsi_{name}",
        EXAMPLE / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pine_reference_indicators_and_signals(module, reference):
    actual = module.prepare_indicators(reference)
    for column in ("ATR", "RSI", "Supertrend", "Direction"):
        valid = reference[column].notna()
        np.testing.assert_allclose(
            actual.loc[valid, column], reference.loc[valid, column], atol=1e-8
        )
    for column in ("entry_long", "exit_long"):
        np.testing.assert_array_equal(actual[column], reference[column].astype(bool))


def test_screener_atr_matches_trading_indicator_at_each_warmup_boundary(module, reference):
    prices = reference[["High", "Low", "Close"]]
    for count in (0, 9, 10, 30, 300):
        data = prices.iloc[:count]
        sizing = module.prepare_atr(data)
        trading = module.prepare_indicators(data)
        pd.testing.assert_series_equal(sizing.ATR, trading.ATR, check_exact=True)
        assert list(sizing.columns) == ["High", "Low", "Close", "ATR"]
        assert sizing.index.equals(data.index)


def test_no_future_data_and_inputs_are_unchanged(module, reference):
    original = reference.copy()
    full = module.prepare_indicators(reference)
    for length in (30, 100, 230, 400):
        partial = module.prepare_indicators(reference.iloc[:length])
        pd.testing.assert_frame_equal(full.iloc[:length], partial)
    pd.testing.assert_frame_equal(reference, original)


def test_rsi_uses_wilder_smoothing_after_sma_seed(module):
    values = np.array([np.nan, 1.0, 2.0, 3.0, 0.0, 4.0])
    np.testing.assert_allclose(
        module.rma(values, 3),
        [np.nan, np.nan, np.nan, 2.0, 4 / 3, 20 / 9],
        equal_nan=True,
    )


def test_rma_nonfinite_samples_do_not_advance_smoothing(module):
    np.testing.assert_allclose(
        module.rma([np.nan, 1, np.inf, 2, -np.inf, 3, np.nan, 4], 3),
        [np.nan, np.nan, np.nan, np.nan, np.nan, 2, np.nan, 8 / 3],
        equal_nan=True,
    )


def test_original_signal_series_and_latest_signal_agree(module, reference):
    strategy = module.SupertrendRSIStrategy()
    series = list(strategy.generate_signal_series({strategy.data_key: reference}))
    assert len(series) == 2
    for side, column in (
        (SignalSide.OPEN_LONG, "entry_long"),
        (SignalSide.CLOSE_LONG, "exit_long"),
    ):
        np.testing.assert_array_equal(
            next(s.series for s in series if s.side == side),
            reference[column].astype(bool),
        )
    index = np.flatnonzero(reference.entry_long.to_numpy())[0]
    signals = list(
        strategy.generate_signals(
            None,
            {strategy.data_key: reference.iloc[: index + 1]},
        )
    )
    assert any(s.side == SignalSide.OPEN_LONG for s in signals)
    assert all(
        s.side not in (SignalSide.OPEN_SHORT, SignalSide.CLOSE_SHORT) for s in signals
    )


@pytest.mark.parametrize(
    "closes,expected",
    [
        ([100, 99, 99.9, 100.9], True),
        ([100, 99, 100, 101], False),  # previous RSI exactly 50
        ([100, 99, 99, 99.5], False),  # current RSI exactly 50
    ],
)
def test_rsi_crossing_is_strict(module, closes, expected):
    frame = pd.DataFrame(
        {
            "High": np.array(closes) + 0.1,
            "Low": np.array(closes) - 0.1,
            "Close": closes,
        },
        index=pd.date_range("2024-01-01", periods=4, freq="h"),
    )
    row = module.prepare_indicators(frame, atr_period=2, factor=0.1, rsi_period=2).iloc[
        -1
    ]
    assert row.Direction == -1
    assert bool(row.entry_long) == expected


@pytest.mark.parametrize(
    "params",
    [
        {"atr_period": 0},
        {"atr_period": 2.5},
        {"rsi_period": 0},
        {"factor": float("nan")},
        {"factor": -1},
        {"rsi_exit": 50},
        {"warmup_window": 3},
        {"warmup_window": 1000},
        {"stop_atr_multiple": 0},
        {"risk_fraction": 0},
        {"risk_fraction": float("nan")},
        {"order_budget": -10},
    ],
)
def test_invalid_parameters_fail_before_account_access(module, params):
    with pytest.raises(ValueError):
        module.SupertrendRSIStrategy(**params)


def test_us_symbol_normalization_keeps_data_and_signal_keys_consistent(module):
    strategy = module.SupertrendRSIStrategy(symbol="spy.us", market="longbridge")
    assert strategy.symbol == "SPY.US/USD"
    assert strategy.data_key == "SPY.US/USD-1h-ohlcv"
    assert strategy.symbols == ["SPY.US"]


def test_bad_market_data_fails_closed(module, reference):
    for bad in (reference.iloc[::-1], pd.concat([reference.iloc[:1]] * 2)):
        with pytest.raises(ValueError):
            module.prepare_indicators(bad)
    bad = reference.copy()
    bad.iloc[-1, bad.columns.get_loc("Close")] = np.nan
    with pytest.raises(ValueError):
        module.prepare_indicators(bad)


def test_warmup_does_not_emit_signals(module, reference):
    strategy = module.SupertrendRSIStrategy()
    assert (
        list(
            strategy.generate_signals(
                None,
                {strategy.data_key: reference.iloc[:10]},
            )
        )
        == []
    )


def test_order_history_blocks_same_bar_reentry_after_restart(module, reference):
    strategy = module.SupertrendRSIStrategy()
    index = np.flatnonzero(reference.entry_long.to_numpy())[0]
    context = Mock()
    context.get_orders.return_value = [
        NS(
            strategy_id=strategy.strategy_id,
            metadata={"signal_bar": reference.index[index].isoformat()},
        )
    ]
    assert not any(
        s.side == SignalSide.OPEN_LONG
        for s in strategy.generate_signals(
            context,
            {strategy.data_key: reference.iloc[: index + 1]},
        )
    )


@pytest.mark.parametrize("bid,triggered", [
    (96, True), (97, True), (98, False),
    (0, False), (-1, False), (None, False), (np.nan, False), (np.inf, False),
])
def test_risk_recovers_stop_distance_from_filled_order(module, bid, triggered):
    strategy = module.SupertrendRSIStrategy(stop_atr_multiple=2)
    context = Mock()
    context.get_open_trades.return_value = [
        NS(
            id=1,
            strategy_id=strategy.strategy_id,
            open_price=101,
            buy_order=NS(metadata={"stop_distance": 4}),
        )
    ]
    context.get_latest_price.return_value = bid
    signals = list(strategy.generate_signals(context, {}))
    assert len(signals) == int(triggered)
    if triggered:
        assert signals[0].side == SignalSide.CLOSE_LONG
        assert signals[0].metadata["risk_exit"] is True
    assert strategy.schedule.step().total_seconds() == 60


def test_protected_variant_is_not_mislabeled_as_vector_baseline(module):
    with pytest.raises(ValueError, match="original"):
        list(
            module.SupertrendRSIStrategy(stop_atr_multiple=2).generate_signal_series({})
        )


@pytest.fixture
def execution(monkeypatch):
    api = pytest.importorskip("longbridge.openapi")
    monkeypatch.setattr(
        "investing_algorithm_framework.infrastructure.longbridge.execution."
        "is_us_regular_session",
        lambda *args: True,
    )
    client = Mock()
    client.get_quote.return_value = NS(
        timestamp=datetime.now(timezone.utc),
        trade_status=api.TradeStatus.Normal,
    )
    client.get_depth.return_value = NS(
        asks=[NS(price=Decimal("100"))],
        bids=[NS(price=Decimal("99"))],
    )
    context = Mock()
    context.get_portfolio.return_value = NS(
        unallocated=2000,
        get_net_size=lambda: 2000,
    )
    return LongbridgeLimitExecutor(client), client, context


def intent(side=SignalSide.OPEN_LONG):
    return SizedIntent(
        signal=Signal("SPY.US", side),
        amount=10,
        price=100,
        quote_amount=1000,
        full_symbol="SPY.US/USD",
        order_reason="test",
    )


def test_atr_risk_budget_caps_whole_shares_and_persists_stop(execution):
    executor, client, context = execution
    metadata = {"stop_distance": 4, "risk_fraction": 0.005}
    executor.execute(intent(), context, metadata)
    request = context.create_limit_order.call_args.kwargs
    assert request["amount"] == 2  # 2000 * 0.5% / $4 = 2.5 -> 2 shares.
    assert request["metadata"]["stop_distance"] == 4
    assert metadata == {"stop_distance": 4, "risk_fraction": 0.005}


def test_atr_exit_routes_to_market_sell_and_stale_quotes_are_blocked(
    execution,
):
    executor, client, context = execution
    executor.execute(intent(SignalSide.CLOSE_LONG), context, {"risk_exit": True})
    assert context.create_market_order.call_args.kwargs["order_side"] == "SELL"
    context.reset_mock()
    client.get_quote.return_value.timestamp -= timedelta(minutes=3)
    executor.execute(intent(), context, {})
    context.create_limit_order.assert_not_called()


def test_full_broker_cycle_and_restart(module, reference, monkeypatch, tmp_path):
    api = pytest.importorskip("longbridge.openapi")
    from investing_algorithm_framework import (
        LongbridgeClient,
        LongbridgeOAuthCredential,
    )

    runner = load_script("longbridge_app")
    monkeypatch.setattr(
        "investing_algorithm_framework.infrastructure.longbridge.execution."
        "is_us_regular_session",
        lambda *args: True,
    )
    # Translate the synthetic timestamps to recent closed hourly bars.
    index = int(np.flatnonzero(reference.entry_long.to_numpy())[0])
    frame = reference.iloc[: index + 1].copy()
    frame.index = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(hours=1),
        periods=len(frame),
        freq="h",
    )
    monkeypatch.setattr(
        "investing_algorithm_framework.infrastructure.longbridge.runtime.latest_completed_us_bar", lambda *a: frame.index[-1].to_pydatetime()
    )
    monkeypatch.setattr("investing_algorithm_framework.infrastructure.longbridge.runtime.is_us_regular_session", lambda *a: True)
    client = LongbridgeClient(LongbridgeOAuthCredential("test"), trading_enabled=True)
    client.quote = Mock()
    client.quote.quote.return_value = [
        NS(
            timestamp=datetime.now(timezone.utc),
            last_done=Decimal("100"),
            volume=10,
            trade_status=api.TradeStatus.Normal,
        )
    ]
    client.quote.depth.return_value = NS(
        bids=[NS(price=Decimal("99"))],
        asks=[NS(price=Decimal("100"))],
    )
    client.quote.subscribe_candlesticks.return_value = [
        NS(
            timestamp=stamp,
            open=row.Open,
            high=row.High,
            low=row.Low,
            close=row.Close,
            volume=row.Volume,
        )
        for stamp, row in frame.iterrows()
    ]
    client.trade = Mock()
    client.trade.account_balance.return_value = [
        NS(
            cash_infos=[
                NS(
                    currency="USD",
                    available_cash=Decimal("25000"),
                )
            ]
        )
    ]
    client.trade.stock_positions.return_value = NS(channels=[])
    client.trade.submit_order.return_value = NS(order_id="buy-1")
    app = runner.build_app(client, tmp_path)
    app.run(number_of_iterations=1)
    client.trade.submit_order.assert_called_once()
    service = app.container.order_service()
    buy = service.get_all()[0]
    assert buy.metadata["stop_distance"] > 0
    quantity = Decimal(str(buy.amount))
    client.trade.order_detail.return_value = NS(
        order_id="buy-1",
        status=api.OrderStatus.Filled,
        quantity=quantity,
        executed_quantity=quantity,
        executed_price=Decimal("100"),
        updated_at=datetime.now(timezone.utc),
        charge_detail=None,
    )
    service.check_pending_orders()
    # Existing filled entries survive app reconstruction and block duplicates.
    app = runner.build_app(client, tmp_path)
    app.run(number_of_iterations=1)
    client.trade.submit_order.assert_called_once()
    trades = app.container.trade_service().get_all()
    assert len(trades) == 1
    distance = trades[0].buy_order.metadata["stop_distance"]
    callback = client.quote.set_on_quote.call_args.args[0]
    callback(
        "SPY.US",
        NS(
            timestamp=datetime.now(timezone.utc) + timedelta(milliseconds=1),
            last_done=Decimal(str(99 - distance)),
            volume=20,
            trade_status=api.TradeStatus.Normal,
        ),
    )
    client.quote.set_on_depth.call_args.args[0]("SPY.US", NS())
    client.quote.realtime_depth.return_value = NS(
        bids=[NS(price=Decimal(str(99 - distance)))],
        asks=[NS(price=Decimal(str(100 - distance)))],
    )
    client.trade.stock_positions.return_value = NS(
        channels=[
            NS(
                positions=[
                    NS(
                        symbol="SPY.US",
                        available_quantity=quantity,
                    )
                ]
            )
        ]
    )
    client.trade.submit_order.return_value = NS(order_id="sell-1")
    app = runner.build_app(client, tmp_path)
    app.run(number_of_iterations=1)
    assert client.trade.submit_order.call_count == 2
    request = client.trade.submit_order.call_args.kwargs
    assert request["order_type"] == api.OrderType.MO
    assert request["side"] == api.OrderSide.Sell
    client.trade.order_detail.return_value = NS(
        order_id="sell-1",
        status=api.OrderStatus.Filled,
        quantity=quantity,
        executed_quantity=quantity,
        executed_price=Decimal(str(99 - distance)),
        updated_at=datetime.now(timezone.utc),
        charge_detail=None,
    )
    app.container.order_service().check_pending_orders()
    position = app.container.position_service().find({"symbol": "SPY.US"})
    assert position.amount == 0
    assert all(t.status == "CLOSED" for t in app.container.trade_service().get_all())
    client.close()


def test_csv_research_runs_without_a_broker(module, tmp_path, caplog):
    research = load_script("backtest")
    result, _ = research.run_research(
        REFERENCE, warmup_window=100, resource_directory=tmp_path
    )
    assert len(result.get_all_backtest_runs()) == 1
    assert result.get_all_backtest_metrics()
    assert not any("has no attribute 'longbridge'" in r.message for r in caplog.records)
