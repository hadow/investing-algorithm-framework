"""Longbridge broker contracts with an in-memory SDK transport."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace as NS
from unittest.mock import Mock

import polars as pl
import pytest

api = pytest.importorskip("longbridge.openapi")

from investing_algorithm_framework import (  # noqa: E402
    DataSource, DataType, LongbridgeClient, LongbridgeOAuthCredential,
    LongbridgeOHLCVDataProvider, LongbridgeOrderExecutor,
    LongbridgePortfolioProvider, Order, OperationalException,
    RESOURCE_DIRECTORY,
)
from investing_algorithm_framework.domain import (  # noqa: E402
    ImproperlyConfigured,
)
from investing_algorithm_framework.infrastructure.data_providers.longbridge \
    import is_us_regular_session  # noqa: E402
from investing_algorithm_framework.infrastructure.longbridge import (  # noqa
    us_symbol,
)
from investing_algorithm_framework.infrastructure.order_executors\
    .longbridge_order_executor import update_order  # noqa: E402


def make_order(**kwargs):
    data = dict(order_type="LIMIT", order_side="BUY", amount=10, price=100,
                target_symbol="SPY.US", trading_symbol="USD", id=7,
                external_id="broker-1", filled=0, remaining=10)
    data.update(kwargs)
    return Order(**data)


def detail(status=None, filled="0", average=None, fees=None):
    return NS(order_id="broker-1", status=status or api.OrderStatus.New,
              executed_quantity=Decimal(filled), quantity=Decimal("10"),
              executed_price=Decimal(average) if average else None,
              updated_at=datetime.now(timezone.utc), charge_detail=fees)


@pytest.fixture
def client(tmp_path):
    client = LongbridgeClient(LongbridgeOAuthCredential("test-client"),
                              trading_enabled=True)
    client.quote = Mock()
    client.quote.subscribe_candlesticks.return_value = []
    client.trade = Mock()
    client.trade.submit_order.return_value = NS(order_id="broker-1")
    client.resource_directory = tmp_path
    return client


def make_executor(client):
    executor = LongbridgeOrderExecutor(client)
    executor.config = {RESOURCE_DIRECTORY: str(client.resource_directory)}
    return executor


@pytest.fixture
def portfolio():
    return NS(id=1, market="LONGBRIDGE", trading_symbol="USD")


@pytest.mark.parametrize("transport", ["ws", "http"])
def test_oauth_uses_sdk_cache_and_explicit_paper_mode(monkeypatch, transport):
    oauth = Mock()
    builder = Mock()
    builder.build.return_value = oauth
    factory = Mock(return_value=builder)
    config = Mock()
    monkeypatch.setattr(api, "OAuthBuilder", factory)
    monkeypatch.setattr(api, "Config", config)
    callback = Mock()
    client = LongbridgeClient(LongbridgeOAuthCredential(" test-id "),
                              on_open_url=callback, quote_transport=transport)
    assert client.config is client.config
    factory.assert_called_once_with("test-id")
    builder.build.assert_called_once_with(callback)
    config.from_oauth.assert_called_once_with(
        oauth, enable_papertrading=True, enable_print_quote_packages=False,
        push_candlestick_mode=api.PushCandlestickMode.Realtime,
        quote_transport=(api.QuoteTransport.Http if transport == "http"
                         else api.QuoteTransport.WebSocket),
    )
    assert not client.trading_enabled


def test_http_client_cannot_start_streams_or_create_trade_context(monkeypatch):
    quote, trade = Mock(), Mock()
    monkeypatch.setattr(api, "QuoteContext", quote)
    monkeypatch.setattr(api, "TradeContext", trade)
    client = LongbridgeClient(quote_transport="http")
    for read in (
        lambda: client.get_quote("SPY.US"),
        lambda: client.get_depth("SPY.US"),
        lambda: client.get_candlesticks("SPY.US", "1h"),
    ):
        with pytest.raises(ImproperlyConfigured, match="HTTP-only"):
            read()
    client.close()
    quote.assert_not_called()
    trade.assert_not_called()


def test_invalid_quote_transport_fails_before_authorization():
    with pytest.raises(ImproperlyConfigured, match="quote_transport"):
        LongbridgeClient(quote_transport="invalid")


def test_trade_requests_share_pacing_across_methods_and_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from investing_algorithm_framework.infrastructure import longbridge

    clock = NS(now=0.0)
    monkeypatch.setattr(longbridge, "time", NS(
        monotonic=lambda: clock.now,
        sleep=lambda delay: setattr(clock, "now", clock.now + delay),
    ))
    calls = []
    context = Mock()
    for name in ("account_balance", "order_detail", "submit_order", "cancel_order"):
        getattr(context, name).side_effect = lambda *a, **k: calls.append(clock.now)
    factory = Mock(return_value=context)
    monkeypatch.setattr(api, "TradeContext", factory)
    client = LongbridgeClient()
    client.config = object()
    trade = client.trade  # 初始化后并发使用同一个上下文。
    methods = [trade.account_balance, trade.order_detail, trade.submit_order, trade.cancel_order]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda i: methods[i % 4](), range(62)))
    factory.assert_called_once()
    assert client.trade is trade
    assert len(calls) == 62
    assert all(b - a >= 1.009 for a, b in zip(calls, calls[1:]))
    assert all(calls[i + 30] - calls[i] > 30 for i in range(len(calls) - 30))


def test_failed_trade_request_consumes_interval_without_retry(monkeypatch):
    from investing_algorithm_framework.infrastructure import longbridge

    clock = NS(now=0.0)
    monkeypatch.setattr(longbridge, "time", NS(
        monotonic=lambda: clock.now,
        sleep=lambda delay: setattr(clock, "now", clock.now + delay),
    ))
    context = Mock()
    context.submit_order.side_effect = TimeoutError("uncertain")
    monkeypatch.setattr(api, "TradeContext", Mock(return_value=context))
    client = LongbridgeClient()
    client.config = object()
    with pytest.raises(TimeoutError, match="uncertain"):
        client.trade.submit_order(symbol="SPY.US")
    client.trade.order_detail("broker-1")
    context.submit_order.assert_called_once_with(symbol="SPY.US")
    context.order_detail.assert_called_once_with("broker-1")
    assert clock.now == pytest.approx(1.01)


def test_oauth_missing_client_fails(monkeypatch):
    monkeypatch.delenv("LONGBRIDGE_OAUTH_CLIENT_ID", raising=False)
    with pytest.raises(ImproperlyConfigured):
        LongbridgeOAuthCredential().initialize()


@pytest.mark.parametrize("symbol", ["BTC/USD", "700.HK/HKD", "SPY.US/EUR"])
def test_symbols_reject_unsupported_markets_and_currency(symbol):
    with pytest.raises(ImproperlyConfigured):
        us_symbol(symbol)


def test_limit_order_and_no_detail_lookup_after_acceptance(client, portfolio):
    order = make_order()
    executor = make_executor(client)
    result = executor.execute_order(portfolio, order, None)
    assert result.external_id == "broker-1"
    assert result.status == "OPEN" and result.filled == 0
    request = client.trade.submit_order.call_args.kwargs
    assert request["symbol"] == "SPY.US"
    assert request["submitted_quantity"] == Decimal("10")
    assert request["submitted_price"] == Decimal("100")
    assert request["outside_rth"] == api.OutsideRTH.RTHOnly
    assert request["client_request_id"]
    with pytest.raises(OperationalException, match="Unreconciled"):
        make_executor(client).execute_order(portfolio, order, None)
    client.trade.submit_order.assert_called_once()
    client.trade.order_detail.assert_not_called()


@pytest.mark.parametrize("change", [dict(amount=0.5), dict(amount=-1),
                                    dict(order_side="SHORT"),
                                    dict(order_type="STOP", stop_price=95),
                                    dict(price=float("nan"))])
def test_invalid_orders_never_reach_broker(client, portfolio, change):
    with pytest.raises(OperationalException):
        make_executor(client).execute_order(
            portfolio, make_order(**change), None,
        )
    client.trade.submit_order.assert_not_called()


def test_sell_cannot_open_short(client, portfolio):
    client.trade.stock_positions.return_value = NS(channels=[])
    with pytest.raises(OperationalException, match="sellable"):
        make_executor(client).execute_order(
            portfolio, make_order(order_side="SELL"), None,
        )
    client.trade.submit_order.assert_not_called()


def test_readonly_blocks_submit_and_cancel(client, portfolio):
    client.trading_enabled = False
    executor = make_executor(client)
    for method in (executor.execute_order, executor.cancel_order):
        with pytest.raises(OperationalException, match="disabled"):
            method(portfolio, make_order(), None)
    client.trade.submit_order.assert_not_called()
    client.trade.cancel_order.assert_not_called()


def test_timeout_blocks_resubmission_after_restart(client, portfolio):
    client.trade.submit_order.side_effect = TimeoutError("uncertain")
    with pytest.raises(TimeoutError):
        make_executor(client).execute_order(portfolio, make_order(), None)
    client.trade.submit_order.side_effect = None
    with pytest.raises(OperationalException, match="Unreconciled"):
        make_executor(client).execute_order(portfolio, make_order(), None)
    client.trade.submit_order.assert_called_once()


def test_persisted_order_reconciliation_clears_submission_marker(
    client, portfolio,
):
    executor = make_executor(client)
    order = executor.execute_order(portfolio, make_order(), None)
    client.trade.order_detail.return_value = detail()
    provider = LongbridgePortfolioProvider(client)
    provider.config = executor.config
    provider.get_order(portfolio, order, None)
    assert not list((client.resource_directory /
                     "longbridge_submissions").glob("*.json"))


@pytest.mark.parametrize("broker,expected", [
    (api.OrderStatus.Filled, "CLOSED"),
    (api.OrderStatus.PartialFilled, "OPEN"),
    (api.OrderStatus.PendingCancel, "OPEN"),
    (api.OrderStatus.PartialWithdrawal, "CANCELED"),
    (api.OrderStatus.Canceled, "CANCELED"),
    (api.OrderStatus.Expired, "EXPIRED"),
    (api.OrderStatus.Rejected, "REJECTED"),
])
def test_order_states(broker, expected):
    assert update_order(make_order(), detail(broker)).status == expected


def test_partial_fills_use_incremental_notional_and_persist_through_restart():
    original = make_order()
    partial = update_order(original, detail(api.OrderStatus.PartialFilled,
                                            "4", "100"))
    # Do not mutate the repository's prior snapshot.
    assert original.filled == 0
    restored = Order.from_dict(partial.to_dict())
    final = update_order(restored, detail(api.OrderStatus.Filled, "10", "106"))
    assert final.price == 110  # 4*100 + 6*110 = 10*106
    assert final.filled == 10 and final.remaining == 0
    assert final.metadata["longbridge_executed_notional"] == "1060"
    assert update_order(final, detail(
        api.OrderStatus.Filled, "10", "106")).price == 110


def test_fee_and_pending_cancel(client, portfolio):
    fees = NS(currency="USD", total_amount=Decimal("1.25"))
    client.trade.order_detail.return_value = detail(
        api.OrderStatus.PendingCancel, "2", "101", fees,
    )
    order = make_executor(client).cancel_order(
        portfolio, make_order(), None,
    )
    assert order.status == "OPEN" and order.filled == 2
    assert order.order_fee == 1.25


def test_numeric_order_id_cancel_uses_sdk_string_ids(client, portfolio):
    broker_id = "1304467974119231488"
    order = make_order(external_id=int(broker_id))
    response = detail(api.OrderStatus.PendingCancel)
    response.order_id = broker_id
    client.trade.order_detail.return_value = response
    make_executor(client).cancel_order(portfolio, order, None)
    client.trade.cancel_order.assert_called_once_with(broker_id)
    client.trade.order_detail.assert_called_once_with(broker_id)


def test_network_failure_is_not_a_missing_order(client, portfolio):
    client.trade.order_detail.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        LongbridgePortfolioProvider(client).get_order(
            portfolio, make_order(), None,
        )


def test_cash_uses_available_currency_cash_not_margin(client, portfolio):
    client.trade.account_balance.return_value = [NS(cash_infos=[
        NS(currency="USD", available_cash=Decimal("200")),
        NS(currency="HKD", available_cash=Decimal("99999")),
    ], total_cash=Decimal("999999"))]
    position = LongbridgePortfolioProvider(client).get_position(
        portfolio, "USD", None,
    )
    assert position.amount == 200


def test_daily_candles_exclude_current_exchange_day_and_copy_client(client):
    end = datetime(2026, 10, 30, 15, tzinfo=timezone.utc)
    previous = end.replace(day=29, hour=13, minute=30)

    def candle(stamp):
        return NS(timestamp=stamp, open=100, high=105, low=95,
                  close=101, volume=7)
    client.quote.history_candlesticks_by_date.return_value = [
        candle(end), candle(previous), candle(previous),
    ]
    source = DataSource(symbol="SPY.US/USD", time_frame="1d",
                        market="LONGBRIDGE", data_type=DataType.OHLCV,
                        warmup_window=60, pandas=True)
    provider = LongbridgeOHLCVDataProvider(client).copy(source)
    assert provider.client is client and provider.pandas
    result = provider.get_data(
        start_date=end - timedelta(days=2), end_date=end,
    )
    assert len(result) == 1
    assert result.index[0].to_pydatetime() == previous
    assert provider._create_start_date(end) < end - timedelta(days=120)


def test_calendar_handles_holidays_half_days_and_dst(client):
    day = datetime(2026, 11, 27, tzinfo=timezone.utc).date()
    client.quote.trading_days.return_value = NS(
        trading_days=[], half_trading_days=[day],
    )
    assert is_us_regular_session(client, datetime(
        2026, 11, 27, 15, tzinfo=timezone.utc))
    assert not is_us_regular_session(
        client, datetime(2026, 11, 27, 18, tzinfo=timezone.utc))
    client.quote.trading_days.return_value.half_trading_days = []
    assert not is_us_regular_session(
        client, datetime(2026, 11, 27, 15, tzinfo=timezone.utc))


def test_empty_daily_result_has_utc_schema(client):
    client.quote.history_candlesticks_by_date.return_value = []
    provider = LongbridgeOHLCVDataProvider(client)
    frame = provider._download_ohlcv(
        "SPY.US/USD", "1d", datetime(2026, 10, 1, tzinfo=timezone.utc),
        datetime(2026, 10, 30, tzinfo=timezone.utc),
    )
    assert frame.is_empty() and frame.schema["Datetime"] == pl.Datetime(
        "us", "UTC")
