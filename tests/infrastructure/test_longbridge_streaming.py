"""Push cache contracts using the installed SDK enums and fake transports."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

api = pytest.importorskip("longbridge.openapi")

from investing_algorithm_framework import (  # noqa: E402
    DataSource, DataType, LongbridgeClient, LongbridgeOHLCVDataProvider,
    LongbridgeTickerDataProvider,
)
from investing_algorithm_framework.domain import (  # noqa: E402
    ImproperlyConfigured, OperationalException,
)
from investing_algorithm_framework.infrastructure.longbridge import (  # noqa
    NEW_YORK, TIMEFRAME_PERIODS, candle_completed, timeframe_period,
)


def candle(stamp, close=100):
    return NS(timestamp=stamp, open=Decimal("100"), high=Decimal("105"),
              low=Decimal("95"), close=Decimal(str(close)), volume=100)


@pytest.fixture
def client(clock):
    client = LongbridgeClient()
    client.quote = Mock()
    day = clock.current.date()
    client.quote.trading_days.return_value = NS(
        trading_days=[day - timedelta(days=1), day], half_trading_days=[],
    )
    client.quote.history_candlesticks_by_date.return_value = []
    client.quote.quote.return_value = [NS(
        timestamp=datetime.now(timezone.utc), last_done=Decimal("100"),
        volume=10, trade_status=api.TradeStatus.Normal,
    )]
    client.quote.depth.return_value = NS(
        bids=[NS(price=Decimal("99"))], asks=[NS(price=Decimal("101"))],
    )
    client.quote.subscribe_candlesticks.return_value = []
    return client


def callback(client, kind):
    return getattr(client.quote, f"set_on_{kind}").call_args.args[0]


@pytest.fixture
def clock(monkeypatch):
    from investing_algorithm_framework.infrastructure import longbridge

    class Clock(datetime):
        current = datetime(2026, 11, 20, 15, 59, 59, tzinfo=NEW_YORK)
        elapsed = 0.0

        @classmethod
        def now(cls, tz):
            return cls.current.astimezone(tz)

        @classmethod
        def advance(cls, seconds):
            cls.current += timedelta(seconds=seconds)
            cls.elapsed += seconds

    monkeypatch.setattr(longbridge, "datetime", Clock)
    monkeypatch.setattr(__name__ + ".datetime", Clock)
    monkeypatch.setattr(longbridge, "time", NS(
        monotonic=lambda: Clock.elapsed, sleep=Clock.advance,
    ))
    return Clock


@pytest.mark.parametrize("month,day,close_hour,period,start_hour", [
    (11, 20, 16, "1h", 15), (11, 27, 13, "1h", 12),
    (6, 17, 16, "4h", 13),
])
@pytest.mark.parametrize("restart_at_close", [False, True])
def test_short_final_candle_follows_exchange_close(
    client, clock, month, day, close_hour, period, start_hour, restart_at_close,
):
    close = datetime(2026, month, day, close_hour, tzinfo=NEW_YORK)
    stamp = close.replace(hour=start_hour, minute=30)
    duration = timedelta(hours=int(period[:-1]))
    old = candle(stamp - duration)
    forming, final = candle(stamp, 99), candle(stamp, 103)
    client.quote.trading_days.return_value = NS(
        trading_days=[] if close_hour == 13 else [close.date()],
        half_trading_days=[close.date()] if close_hour == 13 else [],
    )
    history = client.quote.history_candlesticks_by_date
    history.return_value = [old, final, candle(close + duration)]
    if restart_at_close:
        clock.current = close
        client.quote.subscribe_candlesticks.return_value = [old, final]
    else:
        clock.current = close - timedelta(seconds=1)
        client.quote.subscribe_candlesticks.return_value = [old, forming]
        assert client.get_candlesticks("WMT.US", period) == [old]
        callback(client, "candlestick")("WMT.US", NS(
            period=timeframe_period(period)[1],
            candlestick=candle(stamp.astimezone().replace(tzinfo=None), 98),
            is_confirmed=False,
        ))
        history.assert_not_called()
        clock.advance(19)

    assert client.get_candlesticks("WMT.US", period) == [old, final]
    if restart_at_close:
        history.assert_not_called()
    else:
        history.assert_called_once_with(
            "WMT.US", timeframe_period(period)[1], api.AdjustType.NoAdjust,
            close.date(), close.date(), api.TradeSessions.Intraday,
        )
    callback(client, "candlestick")("WMT.US", NS(
        period=timeframe_period(period)[1], candlestick=forming,
        is_confirmed=False,
    ))
    clock.advance(120)
    assert client.get_candlesticks("WMT.US", period) == [old, final]
    assert history.call_count == int(not restart_at_close)
    client.quote.subscribe_candlesticks.assert_called_once()


@pytest.mark.parametrize("first_response", ["empty", "error"])
def test_missing_confirmation_retries_after_cooldown_without_promoting_local_bar(
    client, clock, first_response,
):
    day = clock.current.date()
    old = candle(clock.current.replace(hour=14, minute=30, second=0))
    final = candle(old.timestamp + timedelta(hours=1), 104)
    client.quote.trading_days.return_value = NS(
        trading_days=[day], half_trading_days=[],
    )
    client.quote.subscribe_candlesticks.return_value = [
        old, candle(final.timestamp, 97),
    ]
    history = client.quote.history_candlesticks_by_date
    history.side_effect = [
        [old] if first_response == "empty" else RuntimeError("unavailable"),
        [old, final],
    ]
    assert client.get_candlesticks("WMT.US", "1h") == [old]
    clock.advance(20)
    assert client.get_candlesticks("WMT.US", "1h") == [old]
    clock.advance(59)
    assert client.get_candlesticks("WMT.US", "1h") == [old]
    assert history.call_count == 1
    clock.advance(1)
    assert client.get_candlesticks("WMT.US", "1h") == [old, final]
    assert history.call_count == 2


def test_missing_pushes_across_symbols_share_backfill_rate_limit(client, clock):
    day = clock.current.date()
    client.quote.trading_days.return_value = NS(
        trading_days=[day], half_trading_days=[],
    )
    old = candle(clock.current.replace(hour=14, minute=30, second=0))
    final = candle(old.timestamp + timedelta(hours=1))
    client.quote.subscribe_candlesticks.return_value = [old]
    calls = []

    def history(*args):
        calls.append(clock.elapsed)
        return [final]

    client.quote.history_candlesticks_by_date.side_effect = history
    symbols = ("WMT.US", "JNJ.US", "SO.US")
    for symbol in symbols:
        assert client.get_candlesticks(symbol, "1h") == [old]
    clock.advance(20)
    for symbol in symbols:
        assert client.get_candlesticks(symbol, "1h") == [old, final]
    assert len(calls) == 3
    assert all(b - a >= 0.55 - 1e-9 for a, b in zip(calls, calls[1:]))


def test_confirmed_push_wins_race_with_history_and_fills_older_gap(client, clock):
    day = clock.current.date()
    client.quote.trading_days.return_value = NS(
        trading_days=[day], half_trading_days=[],
    )
    old = candle(clock.current.replace(hour=13, minute=30, second=0))
    missing = candle(old.timestamp + timedelta(hours=1))
    final = candle(old.timestamp + timedelta(hours=2), 106)
    client.quote.subscribe_candlesticks.return_value = [old]
    client.subscribe_candlesticks("WMT.US", "1h")
    clock.advance(20)

    def history(*args):
        callback(client, "candlestick")("WMT.US", NS(
            period=api.Period.Min_60, candlestick=final, is_confirmed=True,
        ))
        return [missing, candle(final.timestamp, 102)]

    client.quote.history_candlesticks_by_date.side_effect = history
    assert client.get_candlesticks("WMT.US", "1h") == [old, missing, final]


def test_in_session_confirmation_grace_keeps_history_off_the_normal_path(
    client, clock,
):
    clock.current = clock.current.replace(hour=12, minute=29, second=59)
    old = candle(clock.current.replace(hour=10, minute=30, second=0))
    next_bar = candle(old.timestamp + timedelta(hours=1))
    client.quote.subscribe_candlesticks.return_value = [old, next_bar]
    assert client.get_candlesticks("WMT.US", "1h") == [old]
    clock.advance(5)
    assert client.get_candlesticks("WMT.US", "1h") == [old]
    callback(client, "candlestick")("WMT.US", NS(
        period=api.Period.Min_60, candlestick=next_bar, is_confirmed=True,
    ))
    clock.advance(10)
    assert client.get_candlesticks("WMT.US", "1h") == [old, next_bar]
    client.quote.history_candlesticks_by_date.assert_not_called()


def test_broker_throttle_pauses_other_symbols_and_weekend_retries_friday(
    client, clock,
):
    friday = clock.current.date()
    old = candle(clock.current.replace(hour=14, minute=30, second=0))
    final = candle(old.timestamp + timedelta(hours=1))
    client.quote.subscribe_candlesticks.return_value = [old]
    history = client.quote.history_candlesticks_by_date
    history.side_effect = [RuntimeError("request rate limit"), [final]]
    clock.advance(24 * 3600)
    assert client.get_candlesticks("WMT.US", "1h") == [old]
    assert client.get_candlesticks("JNJ.US", "1h") == [old]
    assert history.call_count == 1
    clock.advance(60)
    assert client.get_candlesticks("JNJ.US", "1h") == [old, final]
    assert history.call_args.args[3:5] == (friday, friday)


def test_subscription_that_spans_close_needs_fresh_tail_data(client, clock):
    old = candle(clock.current.replace(hour=14, minute=30, second=0))
    forming = candle(old.timestamp + timedelta(hours=1), 99)

    def slow_subscription(*args):
        clock.advance(20)
        return [old, forming]

    client.quote.subscribe_candlesticks.side_effect = slow_subscription
    assert client.get_candlesticks("WMT.US", "1h") == [old]
    client.quote.history_candlesticks_by_date.assert_called_once()


def test_selection_and_history_include_short_bar_only_at_early_close(
    client, tmp_path,
):
    from investing_algorithm_framework.infrastructure.longbridge.selection import (
        SelectionSource, session_window,
    )

    close = datetime(2026, 11, 27, 13, tzinfo=NEW_YORK)
    day = close.date()
    client.quote.trading_days.return_value = NS(
        trading_days=[day + timedelta(days=n) for n in (-2, 3, 4)],
        half_trading_days=[day],
    )
    old = candle(close - timedelta(minutes=90))
    tail = candle(close - timedelta(minutes=30))
    client.quote.history_candlesticks_by_offset.return_value = [old, tail]
    client.quote.history_candlesticks_by_date.return_value = [old, tail]
    source = SelectionSource(client, tmp_path)
    provider = LongbridgeOHLCVDataProvider(client)
    assert session_window(client, close)[0] == day
    for as_of, expected in ((close - timedelta(seconds=1), [old]), (close, [old, tail])):
        assert source.signal_candlesticks("WMT.US", "1h", as_of) == expected
        frame = provider._download_ohlcv("WMT.US", "1h", old.timestamp, as_of)
        assert frame["Datetime"].to_list() == [c.timestamp for c in expected]


def test_quotes_push_replace_snapshot_without_network_polling(client):
    original = client.get_quote("SPY.US/USD")
    event = NS(timestamp=original.timestamp + timedelta(seconds=1),
               last_done=Decimal("102"), volume=20,
               trade_status=api.TradeStatus.Normal)
    callback(client, "quote")("SPY.US", event)
    assert client.get_quote("SPY.US").last_done == 102
    callback(client, "quote")("SPY.US", original)
    assert client.get_quote("SPY.US").last_done == 102
    client.quote.subscribe.assert_called_once()
    client.quote.quote.assert_called_once()
    client.quote.depth.assert_called_once()
    client.quote.set_on_candlestick.assert_called_once()


def test_depth_uses_sdk_merged_book_after_incremental_push(client):
    assert client.get_depth("SPY.US").bids[0].price == 99
    callback(client, "depth")("SPY.US", NS(asks=[], bids=[]))
    # An empty merged book must stay empty, rather than resurrect seed prices.
    client.quote.realtime_depth.return_value = NS(asks=[], bids=[])
    assert client.get_depth("SPY.US").bids == []
    client.quote.depth.assert_called_once()
    client.quote.realtime_depth.assert_called_once_with("SPY.US")


def test_ticker_reads_latest_pushed_quote_and_book(client):
    provider = LongbridgeTickerDataProvider(
        client, symbol="SPY.US/USD", market="LONGBRIDGE",
    )
    assert provider.get_data()["last"] == 100
    original = client.get_quote("SPY.US")
    event = NS(timestamp=original.timestamp + timedelta(seconds=1),
               last_done=Decimal("110"), volume=30)
    callback(client, "quote")("SPY.US", event)
    assert provider.get_data()["last"] == 110
    client.quote.quote.assert_called_once()


def test_confirmed_and_forming_candles_are_separate_and_update_in_place(
    client,
):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    old, forming = candle(now - timedelta(minutes=10)), candle(now, 101)
    client.quote.subscribe_candlesticks.return_value = [old, forming]
    assert client.get_candlesticks("SPY.US", "5m") == [old]
    assert len(client.get_candlesticks(
        "SPY.US", "5m", include_incomplete=True,
    )) == 2
    update = candle(now, 105)
    push = callback(client, "candlestick")
    push("SPY.US", NS(period=api.Period.Min_5, candlestick=update,
                      is_confirmed=False))
    assert client.get_candlesticks("SPY.US", "5m") == [old]
    push("SPY.US", NS(period=api.Period.Min_5, candlestick=update,
                      is_confirmed=True))
    push("SPY.US", NS(period=api.Period.Min_5, candlestick=forming,
                      is_confirmed=False))
    result = client.get_candlesticks("SPY.US", "5m")
    assert len(result) == 2 and result[-1].close == 105
    client.quote.subscribe_candlesticks.assert_called_once_with(
        "SPY.US", api.Period.Min_5, api.TradeSessions.Intraday,
    )


def test_symbol_and_period_subscriptions_are_independent(client):
    now = datetime.now(timezone.utc)
    for symbol, tf in [("SPY.US", "1m"), ("SPY.US", "5m"),
                       ("SPY.US", "1d"), ("QQQ.US", "1m")]:
        client.subscribe_candlesticks(symbol, tf)
    push = callback(client, "candlestick")
    push("SPY.US", NS(period=api.Period.Min_1,
                      candlestick=candle(now, 1), is_confirmed=True))
    push("SPY.US", NS(period=api.Period.Min_5,
                      candlestick=candle(now, 5), is_confirmed=True))
    assert client.get_candlesticks("SPY.US", "1m")[-1].close == 1
    assert client.get_candlesticks("SPY.US", "5m")[-1].close == 5
    assert client.get_candlesticks("QQQ.US", "1m") == []
    assert client.quote.subscribe_candlesticks.call_count == 4


def test_retired_symbols_are_released_while_exposure_stays_subscribed(client):
    for symbol in ("SPY.US", "QQQ.US"):
        client.get_quote(symbol)
        client.get_candlesticks(symbol, "1h")
    client.retain_subscriptions(["QQQ.US"])
    assert client._quote_symbols == {"QQQ.US"}
    assert client._candle_subscriptions == {("QQQ.US", "1h")}
    assert ("SPY.US", "1h") not in client._candles
    client.quote.unsubscribe_candlesticks.assert_called_once_with(
        "SPY.US", api.Period.Min_60,
    )
    client.get_quote("QQQ.US")
    assert client.quote.subscribe.call_count == 2


def test_quote_cache_compares_naive_sdk_and_aware_timestamps(client):
    initial = client.get_quote("SPY.US")
    stamp = initial.timestamp + timedelta(seconds=1)
    event = NS(timestamp=stamp.astimezone().replace(tzinfo=None),
               last_done=Decimal("111"), volume=30)
    callback(client, "quote")("SPY.US", event)
    assert client.get_quote("SPY.US").last_done == 111


def test_bootstrap_does_not_overwrite_push_received_during_subscribe(client):
    now = datetime.now(timezone.utc)
    latest = candle(now, 110)

    def initial(*args):
        callback(client, "candlestick")("SPY.US", NS(
            period=api.Period.Min_1, candlestick=latest, is_confirmed=True,
        ))
        return [candle(now, 100)]

    client.quote.subscribe_candlesticks.side_effect = initial
    assert client.get_candlesticks("SPY.US", "1m") == [latest]


def test_readers_and_pushes_share_one_subscription_safely(client):
    client.get_candlesticks("SPY.US", "1m")
    push = callback(client, "candlestick")
    start = datetime.now(timezone.utc) - timedelta(minutes=10)

    def update(index):
        push("SPY.US", NS(period=api.Period.Min_1,
                          candlestick=candle(start + timedelta(minutes=index)),
                          is_confirmed=True))
        return client.get_candlesticks("SPY.US", "1m")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(update, range(30)))
    assert len(client.get_candlesticks("SPY.US", "1m")) == 30
    client.quote.subscribe_candlesticks.assert_called_once()


def test_live_cache_is_bounded(client):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    client.quote.subscribe_candlesticks.return_value = [
        candle(now - timedelta(minutes=i)) for i in range(1200)
    ]
    result = client.get_candlesticks("SPY.US", "1m", count=999)
    assert len(result) == 999
    assert len(client._candles[("SPY.US", "1m")]) == 1000


@pytest.mark.parametrize("value", [0, -1, 1000, 1.5])
def test_invalid_warmup_does_not_silently_truncate(client, value):
    with pytest.raises(ImproperlyConfigured):
        client.get_candlesticks("SPY.US", "1m", count=value)
    client.quote.subscribe_candlesticks.assert_not_called()


@pytest.mark.parametrize("name", TIMEFRAME_PERIODS)
def test_sdk_period_mapping(name):
    assert timeframe_period(name)[0] == name


def test_candle_boundaries_resolve_each_intraday_period_once(monkeypatch):
    from investing_algorithm_framework.infrastructure import longbridge

    resolve = Mock(wraps=longbridge.TimeFrame.from_value)
    monkeypatch.setattr(longbridge.TimeFrame, "from_value", resolve)
    now = datetime(2026, 11, 9, 17, tzinfo=timezone.utc)
    for minutes in range(-500, 500):
        sample = candle(now + timedelta(minutes=minutes))
        assert candle_completed(sample, "1h", now) == (minutes <= -60)
    assert resolve.call_count <= 1


def test_unsupported_period_fails_before_subscribing(client):
    with pytest.raises(ImproperlyConfigured, match="Unsupported"):
        client.get_candlesticks("SPY.US", "4m")
    client.quote.subscribe_candlesticks.assert_not_called()
    with pytest.raises(OperationalException, match="time_frame"):
        LongbridgeOHLCVDataProvider(client).get_data()


def test_history_is_unadjusted_and_paginates_intraday_by_exchange_date(client):
    client.quote.history_candlesticks_by_date.return_value = []
    provider = LongbridgeOHLCVDataProvider(client)
    provider._download_ohlcv(
        "SPY.US/USD", "5m", datetime(2026, 10, 29, 14, tzinfo=timezone.utc),
        datetime(2026, 10, 30, 16, tzinfo=timezone.utc),
    )
    calls = client.quote.history_candlesticks_by_date.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.args[1] == api.Period.Min_5
        assert call.args[2] == api.AdjustType.NoAdjust
        assert call.args[3] == call.args[4]


def test_subscribe_failure_can_be_retried(client):
    client.quote.subscribe_candlesticks.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError):
        client.get_candlesticks("SPY.US", "1m")
    client.quote.subscribe_candlesticks.side_effect = None
    client.get_candlesticks("SPY.US", "1m")
    assert client.quote.subscribe_candlesticks.call_count == 2


def test_close_cleans_all_periods_and_quotes_and_can_restart(client):
    client.get_quote("SPY.US")
    client.subscribe_candlesticks("SPY.US", "1m")
    client.subscribe_candlesticks("SPY.US", "5m")
    push = callback(client, "candlestick")
    client.close()
    client.close()
    assert client.quote.unsubscribe_candlesticks.call_count == 2
    client.quote.unsubscribe.assert_called_once()
    push("SPY.US", NS(period=api.Period.Min_1,
                      candlestick=candle(datetime.now(timezone.utc)),
                      is_confirmed=True))
    assert client._candles == {} and client._quotes == {}
    client.get_candlesticks("SPY.US", "1m")
    assert client.quote.subscribe_candlesticks.call_count == 3


def test_close_retries_throttled_unsubscribe(client, monkeypatch):
    pause = Mock()
    monkeypatch.setattr(
        "investing_algorithm_framework.infrastructure.longbridge.time.sleep",
        pause,
    )
    client.get_candlesticks("SPY.US", "1h")
    client.quote.unsubscribe_candlesticks.side_effect = [
        RuntimeError("code=301606 request rate limit"), None,
    ]
    client.close()
    assert client.quote.unsubscribe_candlesticks.call_count == 2
    assert client._candle_subscriptions == set()
    pause.assert_called_once()


def test_persistent_throttle_is_bounded_and_keeps_cleanup_retryable(
    client, monkeypatch,
):
    pause = Mock()
    monkeypatch.setattr(
        "investing_algorithm_framework.infrastructure.longbridge.time.sleep",
        pause,
    )
    client.get_candlesticks("SPY.US", "1h")
    client.quote.unsubscribe_candlesticks.side_effect = RuntimeError(
        "request rate limit"
    )
    with pytest.raises(RuntimeError):
        client.close()
    assert client.quote.unsubscribe_candlesticks.call_count == 4
    assert ("SPY.US", "1h") in client._candle_subscriptions
    assert pause.call_count == 3


def test_close_attempts_remaining_subscriptions_on_error(client):
    client.get_quote("SPY.US")
    client.subscribe_candlesticks("SPY.US", "1m")
    client.quote.unsubscribe_candlesticks.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError):
        client.close()
    client.quote.unsubscribe.assert_called_once()
    client.quote.unsubscribe_candlesticks.side_effect = None
    client.close()
    assert not client._candle_subscriptions


def test_provider_clones_share_pushes_preserve_options_and_return_own_frames(
    client,
):
    source = DataSource(symbol="SPY.US/USD", time_frame="5m",
                        market="LONGBRIDGE", data_type=DataType.OHLCV,
                        warmup_window=60, pandas=True)
    template = LongbridgeOHLCVDataProvider(client, include_incomplete=True)
    first, second = template.copy(source), template.copy(source)
    stamp = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    client.quote.subscribe_candlesticks.return_value = [candle(stamp)]
    frame = first.get_data()
    frame.iloc[0, frame.columns.get_loc("Close")] = -1
    assert second.get_data().iloc[0]["Close"] == 100
    callback(client, "candlestick")("SPY.US", NS(
        period=api.Period.Min_5, candlestick=candle(stamp, 105),
        is_confirmed=True,
    ))
    assert first.get_data().iloc[0]["Close"] == 105
    client.quote.subscribe_candlesticks.assert_called_once()


@pytest.mark.parametrize("order", [(), (0, 1), (1, 0), (1, 0, 0)])
def test_pandas_live_candles_match_polars_schema_and_inclusive_cutoff(order):
    import pandas as pd
    import polars as pl
    from investing_algorithm_framework.domain import convert_polars_to_pandas
    from investing_algorithm_framework.infrastructure.data_providers.longbridge import (
        candle_frame,
    )

    first = datetime(2026, 11, 20, 15, 30, 0, 123456, tzinfo=NEW_YORK)
    last = first + timedelta(minutes=1)
    candles = [candle(first.astimezone().replace(tzinfo=None)), candle(last, 105)]
    rows = [candles[i] for i in order]
    broker = Mock()
    broker.get_candlesticks.return_value = rows
    provider = LongbridgeOHLCVDataProvider(
        broker, symbol="WMT.US/USD", time_frame="1m", pandas=True,
    )
    canonical = candle_frame(rows)
    for cutoff in (None, first - timedelta(microseconds=1), first, last):
        expected = canonical
        if cutoff is not None:
            expected = expected.filter(
                pl.col("Datetime") <= cutoff.astimezone(timezone.utc)
            )
        pd.testing.assert_frame_equal(
            provider.get_data(date=cutoff), convert_polars_to_pandas(expected),
            check_exact=True,
        )


@pytest.mark.parametrize("tf,stamp,now,expected", [
    ("5m", "2026-10-30T13:30:00+00:00", "2026-10-30T13:34:59+00:00", False),
    ("5m", "2026-10-30T13:30:00+00:00", "2026-10-30T13:35:00+00:00", True),
    ("1d", "2026-10-30T13:30:00+00:00", "2026-10-30T20:00:00+00:00", False),
    ("1d", "2026-10-30T13:30:00+00:00", "2026-10-31T04:00:00+00:00", True),
    ("1W", "2026-10-26T13:30:00+00:00", "2026-11-02T05:00:00+00:00", True),
    ("1M", "2026-10-01T13:30:00+00:00", "2026-11-01T04:00:00+00:00", True),
    ("1Y", "2026-01-02T14:30:00+00:00", "2027-01-01T05:00:00+00:00", True),
])
def test_bootstrap_period_boundaries_respect_new_york_dst(tf, stamp, now,
                                                          expected):
    assert candle_completed(candle(datetime.fromisoformat(stamp)), tf,
                            datetime.fromisoformat(now)) == expected
