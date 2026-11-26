"""Daily stock selection, shared portfolio risk and retained-position lifecycle."""

from dataclasses import asdict, replace
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

from investing_algorithm_framework import Signal, SignalSide
from investing_algorithm_framework.infrastructure.longbridge import NEW_YORK
from investing_algorithm_framework.services.strategy_phases import SizedIntent

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/algorithm_examples/strategies_showcase/27_supertrend_rsi"
NOW = datetime(2026, 11, 3, 14, tzinfo=timezone.utc)
AS_OF = date(2026, 11, 2)


@pytest.fixture
def selection(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return importlib.import_module("screener")


@pytest.fixture
def portfolio(selection):
    return importlib.import_module("portfolio")


@pytest.fixture
def runner(selection):
    return importlib.import_module("portfolio_app")


def daily(slope=0.1):
    close = 100 + np.arange(300) * slope
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(300, 1_000_000),
            "DollarVolume": np.full(300, 100_000_000),
        },
        index=pd.bdate_range(end="2026-11-02", periods=300, tz=NEW_YORK)
        + pd.Timedelta(hours=9, minutes=30),
    )


def snapshot(selection, symbols, now=NOW):
    today = now.astimezone(NEW_YORK).date()
    return {
        "schema_version": 1,
        "selection_model": selection.SELECTION_MODEL,
        "base_pool": list(dict.fromkeys(symbols)),
        "generated_at": (now - timedelta(seconds=1)).isoformat(),
        "trading_date": today.isoformat(),
        "as_of_date": (today - timedelta(days=1)).isoformat(),
        "rules": asdict(selection.SelectionRules()),
        "selected": [{"symbol": s} for s in symbols],
    }


def test_daily_filters_rank_strength_and_preserve_input(selection):
    stronger, weaker = daily(0.2), daily(0.1)
    original = stronger.copy()
    low_volume = daily(0.3)
    low_volume["DollarVolume"] = 1_000_000
    histories = {
        "AAA.US": weaker,
        "BBB.US": stronger,
        "CCC.US": low_volume,
        "DDD.US": daily(-0.1),
        "EEE.US": daily(0.4),
    }
    accepted, rejected = selection.rank_candidates(
        histories, daily(0.05), AS_OF, selection.SelectionRules(), {"EEE.US"}
    )
    assert [r["symbol"] for r in accepted] == ["BBB.US", "AAA.US"]
    reasons = {r["symbol"]: r["reasons"] for r in rejected}
    assert "insufficient_dollar_volume" in reasons["CCC.US"]
    assert "daily_trend_not_up" in reasons["DDD.US"]
    assert "earnings_within_two_sessions" in reasons["EEE.US"]
    pd.testing.assert_frame_equal(stronger, original)


def test_future_daily_bars_cannot_change_today_selection(selection):
    frame = daily(0.2)
    expected = selection.rank_candidates(
        {"AAA.US": frame}, daily(0.05), AS_OF, selection.SelectionRules()
    )
    future = frame.iloc[-1:].copy()
    future.index += pd.Timedelta(days=1)
    future["Close"] = 1
    actual = selection.rank_candidates(
        {"AAA.US": pd.concat([frame, future])},
        daily(0.05),
        AS_OF,
        selection.SelectionRules(),
    )
    assert actual == expected


def test_risk_adjusted_score_prefers_steady_trend_over_larger_noisy_return(selection):
    steady, volatile = daily(), daily()
    time = np.arange(300)
    for frame, close in (
        (steady, 100 * np.exp(0.0015 * time)),
        (volatile, 100 * np.exp(0.003 * time + 0.03 * np.sin(2 * np.pi * time / 5))),
    ):
        frame["Close"] = frame["Open"] = close
        frame["High"], frame["Low"] = close + 1, close - 1
    ranked, rejected = selection.rank_candidates(
        {"VOL.US": volatile, "STEADY.US": steady}, daily(0.05), AS_OF,
        selection.SelectionRules(),
    )
    assert rejected == []
    assert [row["symbol"] for row in ranked] == ["STEADY.US", "VOL.US"]
    assert ranked[1]["relative_return_60"] > ranked[0]["relative_return_60"]
    assert ranked[0]["selection_score"] > ranked[1]["selection_score"]


@pytest.mark.parametrize("case,reason", [
    ("weakening", "recent_relative_strength_not_positive"),
    ("extended", "extended_above_ema20"),
])
def test_recent_weakness_and_abrupt_extension_do_not_enter_pool(selection, case, reason):
    frame = daily(0.2 if case == "weakening" else 0.1)
    if case == "weakening":
        frame.loc[frame.index[-20:], "Close"] -= np.linspace(0, 3, 20)
    else:
        frame.loc[frame.index[-1], "Close"] *= 1.1
    for column, offset in (("Open", 0), ("High", 1), ("Low", -1)):
        frame[column] = frame.Close + offset
    ranked, rejected = selection.rank_candidates(
        {"AAA.US": frame}, daily(0.05), AS_OF, selection.SelectionRules()
    )
    assert ranked == []
    assert rejected[0]["relative_return_60"] > 0
    assert reason in rejected[0]["reasons"]


def test_correlations_ignore_future_bars_and_use_aligned_returns(selection):
    histories = {"AAA.US": daily(0.2), "BBB.US": daily(0.2)}
    ranked = [{"symbol": symbol} for symbol in histories]
    expected = selection.candidate_correlations(histories, ranked, AS_OF)
    assert expected.loc["AAA.US", "BBB.US"] == pytest.approx(1)
    future = histories["AAA.US"].iloc[-1:].copy()
    future.index += pd.Timedelta(days=1)
    future["Close"] = 1
    histories["AAA.US"] = pd.concat([histories["AAA.US"], future])
    pd.testing.assert_frame_equal(
        expected, selection.candidate_correlations(histories, ranked, AS_OF)
    )


@pytest.mark.parametrize("field,value", [
    ("max_correlation", 0), ("max_correlation", 1.01),
    ("max_correlation", float("nan")), ("max_extension_atr", 0),
    ("max_extension_atr", float("inf")),
])
def test_selection_quality_parameters_are_validated(selection, field, value):
    with pytest.raises(ValueError):
        selection.SelectionRules(**{field: value})


def test_previous_universe_only_manages_retained_positions(selection, runner, tmp_path, monkeypatch):
    from investing_algorithm_framework import LongbridgeClient, LongbridgeOAuthCredential

    old = snapshot(selection, ["AAA.US"])
    old.update(selection_model=selection.LEGACY_SELECTION_MODEL)
    old["rules"].update(base_size=200, top_n=60)
    path = tmp_path / "universe.json"
    selection.atomic_json(path, old)
    with pytest.raises(ValueError, match="formal C universe"):
        selection.load_snapshot(path, NOW)
    client = LongbridgeClient(LongbridgeOAuthCredential("test"))
    monkeypatch.setattr(runner, "retained_symbols", lambda *args: {"HELD.US"})
    app, tracked, selected = runner.build_portfolio_app(client, tmp_path / "account", old)
    assert selected == [] and tracked == ["HELD.US"]
    assert app.get_algorithm().strategies[0].entry_date is None


@pytest.mark.parametrize("bad", ["short", "stale", "duplicate", "nan", "date_gap"])
def test_bad_history_is_rejected_per_symbol(selection, bad):
    frame = daily(0.2)
    if bad == "short":
        frame = frame.iloc[-200:]
    elif bad == "stale":
        frame = frame.iloc[:-1]
    elif bad == "duplicate":
        frame = pd.concat([frame, frame.iloc[-1:]])
    elif bad == "nan":
        frame.iloc[-1, frame.columns.get_loc("DollarVolume")] = np.nan
    else:
        frame = frame.drop(frame.index[-20])
    accepted, rejected = selection.rank_candidates(
        {"AAA.US": frame}, daily(0.05), AS_OF, selection.SelectionRules()
    )
    assert accepted == [] and len(rejected) == 1


@pytest.mark.parametrize("day", [date(2026, 3, 8), date(2026, 11, 1)])
def test_daily_cutoff_uses_new_york_midnight_across_dst(selection, day):
    midnight = pd.Timestamp(day + timedelta(days=1), tz=NEW_YORK)
    frame = pd.DataFrame(
        {"Close": [100., 101., 102.], "Volume": [1000.] * 3,
         "DollarVolume": [100_000.] * 3},
        index=pd.DatetimeIndex([
            midnight - pd.Timedelta(nanoseconds=1), midnight,
            midnight + pd.Timedelta(nanoseconds=1),
        ]),
    )
    expected = frame.iloc[:1].tz_convert("UTC")
    pd.testing.assert_frame_equal(selection.normalize_daily(frame, day), expected)


def test_calendar_uses_us_date_and_rejects_incomplete_responses(selection):
    event = {"symbol": "AAA.US", "type": "report", "ext": {"local_date": "2026-11-03"}}
    payload = {"list": [{"date": "2026-11-04", "infos": [event]}], "next_date": ""}
    assert selection.earnings_symbols(payload, {date(2026, 11, 3)}) == {"AAA.US"}
    assert selection.earnings_symbols(payload, {date(2026, 11, 4)}) == set()
    disclosed = {"list": [{"infos": [{**event, "type": "financial"}]}]}
    assert not selection.earnings_symbols(disclosed, {date(2026, 11, 3)})
    assert selection.earnings_symbols(
        disclosed, {date(2026, 11, 3)}, event_types=("report", "financial")
    ) == {"AAA.US"}
    with pytest.raises(ValueError):
        selection.earnings_symbols({**payload, "next_date": "2026-11-05"}, set())
    with pytest.raises(ValueError):
        selection.earnings_symbols({}, set())


def test_snapshot_rejects_stale_duplicate_and_future_files(selection, tmp_path):
    path = tmp_path / "universe.json"
    selection.atomic_json(path, snapshot(selection, ["AAA.US"]))
    assert selection.load_snapshot(path, NOW)["selected"][0]["symbol"] == "AAA.US"
    with pytest.raises(ValueError):
        selection.load_snapshot(path, NOW + timedelta(days=1))
    selection.atomic_json(path, snapshot(selection, ["AAA.US", "AAA.US"]))
    with pytest.raises(ValueError):
        selection.load_snapshot(path, NOW)
    value = snapshot(selection, [])
    value["generated_at"] = (NOW + timedelta(seconds=1)).isoformat()
    selection.atomic_json(path, value)
    with pytest.raises(ValueError):
        selection.load_snapshot(path, NOW)


def test_failed_atomic_write_preserves_previous_universe(selection, tmp_path):
    path = tmp_path / "universe.json"
    selection.atomic_json(path, {"old": True})
    with pytest.raises(ValueError):
        selection.atomic_json(path, {"value": float("nan")})
    assert path.read_text().strip() == '{\n  "old": true\n}'


def test_simultaneous_snapshot_writers_publish_complete_json(selection, tmp_path, monkeypatch):
    import os
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)
    fsync = os.fsync

    def sync_together(fd):
        fsync(fd)
        barrier.wait(timeout=5)

    monkeypatch.setattr(os, "fsync", sync_together)
    path = tmp_path / "universe.json"
    snapshots = [{"selected": [name] * 100} for name in ("AAA.US", "BBB.US")]
    with ThreadPoolExecutor(max_workers=2) as writers:
        list(writers.map(lambda s: selection.atomic_json(path, s), snapshots))
    assert json.loads(path.read_text()) in snapshots
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("base_size", [100, 150, 200])
def test_expanded_base_pool_reuses_cache_for_smaller_scans(
    selection, tmp_path, base_size
):
    symbols = [f"S{i}.US" for i in range(base_size)]
    path = tmp_path / "base/2026-11-03.json"
    selection.atomic_json(path, {"stocks": [{"symbol": s} for s in symbols[:50]]})
    cli = Mock(return_value={"stocks": [{"symbol": s} for s in symbols]})
    source = selection.SelectionSource(None, tmp_path, cli)
    assert source.base_pool(date(2026, 11, 3), base_size) == symbols
    assert source.base_pool(date(2026, 11, 3), 50) == symbols[:50]
    cli.assert_called_once_with(
        ["constituent", ".SPX.US", "--sort", "turnover", "--limit", str(base_size)]
    )
    assert selection.SelectionRules(base_size=100).top_n == 100
    with pytest.raises(ValueError):
        selection.SelectionRules(base_size=201)


def test_default_pool_and_shared_executor_limits(selection, runner, tmp_path):
    from investing_algorithm_framework import (
        LongbridgeClient,
        LongbridgeOAuthCredential,
    )

    rules = selection.SelectionRules()
    assert (rules.base_size, rules.top_n) == (None, 100)
    client = LongbridgeClient(LongbridgeOAuthCredential("test"), trading_enabled=False)
    app, tracked, selected = runner.build_portfolio_app(
        client, tmp_path, snapshot(selection, ["AAA.US"])
    )
    assert tracked == selected == ["AAA.US"]
    executor = app.get_algorithm().strategies[0].executor
    assert executor.max_positions == 5
    assert executor.total_risk_fraction == 0.025
    assert executor.min_entry_shares == 1
    assert (rules.capital, rules.order_budget) == (25000, 4500)


@pytest.mark.parametrize("legacy_floor", [None, 10])
def test_retired_floor_is_ignored_in_dated_snapshots(
    selection, runner, tmp_path, legacy_floor
):
    from investing_algorithm_framework import (
        LongbridgeClient,
        LongbridgeOAuthCredential,
    )

    prior = snapshot(selection, ["AAA.US"])
    if legacy_floor is not None:
        prior["rules"]["min_entry_shares"] = legacy_floor
    path = tmp_path / "universe.json"
    selection.atomic_json(path, prior)
    assert "min_entry_shares" not in selection.load_snapshot(path, NOW)["rules"]
    client = LongbridgeClient(LongbridgeOAuthCredential("test"), trading_enabled=False)
    app, tracked, selected = runner.build_portfolio_app(
        client, tmp_path / "account", prior, min_entry_shares=10
    )
    assert selected == tracked == ["AAA.US"]
    assert app.get_algorithm().strategies[0].executor.min_entry_shares == 1


@pytest.mark.parametrize("budget,shares", [(100, 0), (150, 1), (1000, 2)])
def test_screener_uses_affordable_whole_shares(
    selection, monkeypatch, budget, shares
):
    monkeypatch.setattr(
        selection,
        "session_window",
        lambda *a: (date(2026, 11, 3), AS_OF, [date(2026, 11, 3)]),
    )
    source = Mock()
    source.base_pool.return_value = ["AAA.US"]
    source.blocked_earnings.return_value = set()
    source.daily.side_effect = lambda symbol, *a: daily(
        0.05 if symbol == "SPY.US" else 0.1
    )
    source.signal_candlesticks.return_value = [
        NS(
            timestamp=t,
            open=r.Open,
            high=r.High,
            low=r.Low,
            close=r.Close,
            volume=r.Volume,
        )
        for t, r in daily(0.1).iloc[-100:].iterrows()
    ]
    rules = selection.SelectionRules(
        base_size=1, top_n=1, capital=2000, order_budget=budget, min_entry_shares=10
    )
    result = selection.screen(source, rules, NOW)
    assert len(result["selected"]) == int(shares > 0)
    if shares:
        assert result["selected"][0]["estimated_shares"] == shares
    else:
        assert result["rejected"][0]["reasons"] == [
            "risk_or_budget_below_one_share"
        ]


def test_incomplete_base100_response_keeps_previous_cache(selection, tmp_path):
    payload = {"stocks": [{"symbol": f"S{i}.US"} for i in range(50)]}
    path = tmp_path / "base/2026-11-03.json"
    selection.atomic_json(path, payload)
    cli = Mock(return_value=payload)
    source = selection.SelectionSource(None, tmp_path, cli)
    with pytest.raises(ValueError, match="fewer symbols"):
        source.base_pool(date(2026, 11, 3), 100)
    assert json.loads(path.read_text()) == payload


def test_full_screen_skips_unaffordable_stronger_stock(selection, monkeypatch):
    pytest.importorskip("longbridge.openapi")
    monkeypatch.setattr(
        selection,
        "session_window",
        lambda *a: (date(2026, 11, 3), AS_OF, [date(2026, 11, d) for d in (3, 4, 5)]),
    )
    source = Mock()
    source.base_pool.return_value = ["AAA.US", "BBB.US"]
    source.blocked_earnings.return_value = set()
    source.daily.side_effect = lambda s, *a: daily(
        {"SPY.US": 0.05, "AAA.US": 0.1, "BBB.US": 0.2}[s]
    )

    def candles(symbol, *a, **k):
        frame = daily(0.1).iloc[-100:].copy()
        if symbol == "BBB.US":
            frame[["Open", "High", "Low", "Close"]] *= 100
        return [
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

    source.signal_candlesticks.side_effect = candles
    result = selection.screen(
        source, selection.SelectionRules(base_size=2, top_n=1), NOW
    )
    assert [r["symbol"] for r in result["selected"]] == ["AAA.US"]
    assert any(r["symbol"] == "BBB.US" for r in result["rejected"])


@pytest.mark.parametrize("all_failed", [False, True])
def test_hourly_data_failures_are_reported_and_cannot_publish_false_empty_pool(
    selection, monkeypatch, all_failed
):
    monkeypatch.setattr(
        selection, "session_window", lambda *a: (date(2026, 11, 3), AS_OF, [])
    )
    source = Mock()
    source.base_pool.return_value = ["AAA.US", "BBB.US"]
    source.blocked_earnings.return_value = set()
    source.daily.side_effect = lambda s, *a: daily(
        {"SPY.US": 0.05, "AAA.US": 0.1, "BBB.US": 0.2}[s]
    )

    def candles(symbol, *a, **k):
        if all_failed or symbol == "BBB.US":
            raise TimeoutError("hourly upstream timeout")
        return [
            NS(timestamp=t, open=r.Open, high=r.High, low=r.Low,
               close=r.Close, volume=r.Volume)
            for t, r in daily(0.1).iloc[-100:].iterrows()
        ]

    source.signal_candlesticks.side_effect = candles
    rules = selection.SelectionRules(base_size=2, top_n=1)
    if all_failed:
        with pytest.raises(ValueError, match="No usable signal histories"):
            selection.screen(source, rules, NOW)
    else:
        result = selection.screen(source, rules, NOW)
        assert [row["symbol"] for row in result["selected"]] == ["AAA.US"]
        assert result["errors"] == {"BBB.US": "signal_data: hourly upstream timeout"}


@pytest.mark.parametrize("same_direction", [True, False])
def test_candidate_correlation_keeps_diversification_and_budget_rules(
    selection, monkeypatch, same_direction
):
    monkeypatch.setattr(
        selection, "session_window", lambda *a: (date(2026, 11, 3), AS_OF, [])
    )
    histories = {"AAA.US": daily(0.2), "BBB.US": daily(0.2), "CCC.US": daily(0.1)}
    if not same_direction:
        time = np.arange(300)
        for symbol, sign in (("AAA.US", 1), ("BBB.US", -1)):
            close = 100 * np.exp(0.0015 * time + sign * 0.005 * np.sin(time))
            frame = histories[symbol]
            frame["Close"] = frame["Open"] = close
            frame["High"], frame["Low"] = close + 1, close - 1
    close = histories["CCC.US"].Close + 0.05 * np.sin(np.arange(300))
    histories["CCC.US"]["Close"] = histories["CCC.US"]["Open"] = close
    histories["CCC.US"]["High"], histories["CCC.US"]["Low"] = close + 1, close - 1
    source = Mock()
    source.base_pool.return_value = list(histories)
    source.blocked_earnings.return_value = set()
    source.daily.side_effect = lambda symbol, *a: (
        daily(0.05) if symbol == "SPY.US" else histories[symbol]
    )
    source.signal_candlesticks.return_value = [
        NS(timestamp=t, open=r.Open, high=r.High, low=r.Low, close=r.Close, volume=r.Volume)
        for t, r in daily(0.1).iloc[-100:].iterrows()
    ]
    result = selection.screen(source, selection.SelectionRules(base_size=3, top_n=2), NOW)
    assert result["selection_model"] == selection.LEGACY_SELECTION_MODEL
    assert all(row["estimated_shares"] >= 10 for row in result["selected"])
    if not same_direction:
        assert {row["symbol"] for row in result["selected"]} == {"AAA.US", "BBB.US"}
        correlations = selection.candidate_correlations(
            histories, result["selected"], AS_OF
        )
        assert correlations.loc["AAA.US", "BBB.US"] < -0.9
        return
    assert [row["symbol"] for row in result["selected"]] == ["AAA.US", "CCC.US"]
    duplicate = next(row for row in result["rejected"] if row["symbol"] == "BBB.US")
    assert duplicate["reasons"] == ["high_correlation_with_selected"]
    assert duplicate["correlated_with"] == "AAA.US"
    assert duplicate["correlation"] == pytest.approx(1)
    assert [call.args[0] for call in source.signal_candlesticks.call_args_list] == [
        "AAA.US", "CCC.US"
    ]
    research = selection.screen(
        source, selection.SelectionRules(base_size=3, top_n=2), NOW,
        candidate_correlation=False,
    )
    assert research["selection_model"] == "research_order_correlation"
    assert [r["symbol"] for r in research["selected"]] == ["AAA.US", "BBB.US"]


def intent(symbol="AAA.US", side=SignalSide.OPEN_LONG):
    return SizedIntent(
        Signal(symbol, side),
        amount=10,
        price=100,
        quote_amount=1000,
        full_symbol=f"{symbol}/USD",
        order_reason="test",
    )


@pytest.fixture
def execution(portfolio, monkeypatch):
    api = pytest.importorskip("longbridge.openapi")
    monkeypatch.setattr(
        "investing_algorithm_framework.infrastructure.longbridge.execution."
        "is_us_regular_session",
        lambda *a: True,
    )
    client, context = Mock(), Mock()
    client.get_quote.return_value = NS(
        timestamp=datetime.now(timezone.utc), trade_status=api.TradeStatus.Normal
    )
    client.get_depth.return_value = NS(
        bids=[NS(price=Decimal("99.99"))], asks=[NS(price=Decimal("100"))]
    )
    context.get_portfolio.return_value = NS(unallocated=2000, get_net_size=lambda: 2000)
    context.get_open_trades.return_value = []
    context.get_positions.return_value = []
    context.get_orders.return_value = []
    return (
        portfolio.PortfolioExecutor(client, total_risk_fraction=0.01),
        client,
        context,
    )


def test_five_aapl_sized_entries_fit_new_cash_and_risk_budgets(execution, portfolio):
    _, client, context = execution
    executor = portfolio.PortfolioExecutor(client)
    capital = NS(unallocated=25000, get_net_size=lambda: 25000)
    context.get_portfolio.return_value = capital
    client.get_depth.return_value = NS(
        bids=[NS(price=Decimal("350.62"))], asks=[NS(price=Decimal("350.63"))]
    )
    working = []
    context.get_orders.side_effect = lambda **kwargs: working

    def create_limit(**request):
        capital.unallocated -= request["amount"] * request["price"]
        order = NS(
            target_symbol=request["target_symbol"],
            status="OPEN",
            order_side="BUY",
            remaining=request["amount"],
            metadata=request["metadata"],
        )
        working.append(order)
        return order

    context.create_limit_order.side_effect = create_limit
    metadata = {"stop_distance": 6.863840683708149, "risk_fraction": 0.005}
    for symbol in ("AAA.US", "BBB.US", "CCC.US", "DDD.US", "EEE.US"):
        proposal = replace(
            intent(symbol), amount=4500 / 350.63, price=350.63, quote_amount=4500
        )
        assert executor.execute(proposal, context, metadata) is not None
    assert [o.remaining for o in working] == [12] * 5
    assert portfolio.portfolio_risk(context)[1] == pytest.approx(411.8304410224889)
    assert capital.unallocated > 2500
    assert (
        executor.execute(
            replace(intent("FFF.US"), amount=4500 / 350.63, quote_amount=4500),
            context,
            metadata,
        )
        is None
    )
    assert len(working) == 5


@pytest.mark.parametrize(
    "price,distance,cash,expected",
    [(451, 5, 25000, 9), (350.63, 13, 25000, 9), (350.63, 6.86, 3000, 8),
     (350.63, 6.86, 400, 1), (350.63, 6.86, 100, 0)]
)
def test_new_entries_use_actual_budget_after_all_caps(
    execution, portfolio, price, distance, cash, expected
):
    _, client, context = execution
    executor = portfolio.PortfolioExecutor(client)
    context.get_portfolio.return_value = NS(
        unallocated=cash, get_net_size=lambda: 25000
    )
    client.get_depth.return_value = NS(
        bids=[NS(price=Decimal(str(price)) - Decimal("0.01"))],
        asks=[NS(price=Decimal(str(price)))],
    )
    proposal = replace(intent(), amount=4500 / price, price=price, quote_amount=4500)
    executor.execute(proposal, context, {"stop_distance": distance, "risk_fraction": 0.005})
    if expected:
        assert context.create_limit_order.call_args.kwargs["amount"] == expected
    else:
        context.create_limit_order.assert_not_called()


def test_minimum_entry_size_does_not_block_one_share_exit(execution, portfolio):
    _, client, context = execution
    executor = portfolio.PortfolioExecutor(client)
    executor.execute(
        replace(intent(side=SignalSide.CLOSE_LONG), amount=1),
        context,
        {"risk_exit": True},
    )
    assert context.create_market_order.call_args.kwargs["amount"] == 1


def test_pending_buys_reserve_total_risk_before_they_fill(execution):
    executor, _, context = execution
    context.get_orders.return_value = [
        NS(
            target_symbol="BBB.US",
            status="OPEN",
            order_side="BUY",
            remaining=4,
            metadata={"stop_distance": 4},
        )
    ]
    executor.execute(intent(), context, {"stop_distance": 4, "risk_fraction": 0.005})
    assert context.create_limit_order.call_args.kwargs["amount"] == 1


def test_sell_reservations_do_not_release_risk_before_fill(portfolio):
    context = Mock()
    context.get_open_trades.return_value = [
        NS(
            target_symbol="AAA.US",
            available_amount=0,
            buy_order=NS(metadata={"stop_distance": 4}),
        )
    ]
    context.get_positions.return_value = [NS(symbol="AAA.US", amount=5)]
    context.get_orders.return_value = [
        NS(target_symbol="AAA.US", status="OPEN", order_side="SELL", remaining=5)
    ]
    assert portfolio.portfolio_risk(context) == ({"AAA.US"}, 20)


def test_max_slots_and_spread_block_entries_but_allow_risk_exit(execution):
    executor, client, context = execution
    context.get_orders.return_value = [
        NS(
            target_symbol=f"{s}.US",
            status="OPEN",
            order_side="BUY",
            remaining=1,
            metadata={"stop_distance": 1},
        )
        for s in ("B", "C", "D", "E", "F")
    ]
    executor.execute(intent(), context, {"stop_distance": 4, "risk_fraction": 0.005})
    context.create_limit_order.assert_not_called()
    executor.execute(intent(side=SignalSide.CLOSE_LONG), context, {"risk_exit": True})
    context.create_market_order.assert_called_once()
    context.reset_mock()
    context.get_orders.return_value = []
    client.get_depth.return_value.bids[0].price = Decimal("98")
    executor.execute(intent(), context, {"stop_distance": 4, "risk_fraction": 0.005})
    context.create_limit_order.assert_not_called()


def test_fifth_symbol_is_allowed_within_existing_risk_budget(execution):
    executor, _, context = execution
    context.get_orders.return_value = [
        NS(
            target_symbol=f"{s}.US",
            status="OPEN",
            order_side="BUY",
            remaining=1,
            metadata={"stop_distance": 1},
        )
        for s in ("B", "C", "D", "E")
    ]
    executor.execute(intent(), context, {"stop_distance": 4, "risk_fraction": 0.005})
    context.create_limit_order.assert_called_once()


def test_missing_stop_metadata_blocks_new_risk(execution):
    executor, _, context = execution
    context.get_orders.return_value = [
        NS(
            target_symbol="BBB.US",
            status="OPEN",
            order_side="BUY",
            remaining=1,
            metadata={},
        )
    ]
    executor.execute(intent(), context, {"stop_distance": 4, "risk_fraction": 0.005})
    context.create_limit_order.assert_not_called()


def test_runner_lock_prevents_two_writers(runner, tmp_path):
    with runner.single_runner(tmp_path):
        with pytest.raises(RuntimeError):
            with runner.single_runner(tmp_path):
                pass


@pytest.mark.parametrize("iterations", [0, 1])
def test_account_loop_records_failure_and_retries_only_in_watch_mode(tmp_path, monkeypatch, iterations):
    from investing_algorithm_framework.infrastructure.longbridge import runner

    tick = Mock(side_effect=[ValueError("feed unavailable"), KeyboardInterrupt()])
    sleep = Mock()
    monkeypatch.setattr(runner.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt if iterations == 0 else ValueError):
        runner.run_iterations(tick, iterations, tmp_path, "paper")
    health = json.loads((tmp_path / "health.json").read_text())
    assert (health["status"], health["broker_environment"], health["error"]) == (
        "failed", "paper", "feed unavailable",
    )
    if iterations == 0:
        sleep.assert_called_once_with(60)
        assert tick.call_count == 2
    else:
        sleep.assert_not_called()
        assert tick.call_count == 1


def test_continuous_runner_reuses_app_and_checks_risk_every_tick(
    selection, runner, monkeypatch, tmp_path,
):
    import sys
    from investing_algorithm_framework.infrastructure.longbridge import operations, runner as shared_runner
    from investing_algorithm_framework.infrastructure.longbridge import runtime
    from investing_algorithm_framework import LongbridgeClient, LongbridgeOAuthCredential

    api = pytest.importorskip("longbridge.openapi")
    now = datetime.now(timezone.utc)
    bar = now - timedelta(hours=1)
    client = LongbridgeClient(LongbridgeOAuthCredential("offline-performance"))
    client.quote, client.trade = Mock(), Mock()
    client.quote.quote.return_value = [NS(
        timestamp=now, last_done=Decimal("100"), volume=100,
        trade_status=api.TradeStatus.Normal,
    )]
    client.quote.depth.return_value = NS(
        bids=[NS(price=Decimal("99.99"))], asks=[NS(price=Decimal("100"))],
    )
    client.quote.subscribe_candlesticks.return_value = [
        NS(timestamp=stamp, open=100., high=102., low=98., close=100., volume=1000)
        for stamp in pd.date_range(end=bar, periods=300, freq="h")
    ]
    client.trade.account_balance.return_value = [NS(cash_infos=[
        NS(currency="USD", available_cash=Decimal("25000")),
    ])]
    client.trade.stock_positions.return_value = NS(channels=[])
    path = tmp_path / "universe.json"
    selection.atomic_json(path, snapshot(selection, ["AAA.US"], now))
    monkeypatch.setattr(runner, "LongbridgeClient", Mock(return_value=client))
    monkeypatch.setattr(runner, "load_dotenv", Mock())
    monkeypatch.setattr(runner, "preflight", Mock(return_value={
        "ready": True, "entry_blockers": ["offline performance test"],
    }))
    monkeypatch.setattr(runtime, "latest_completed_us_bar", lambda *a: bar)
    monkeypatch.setattr(runtime, "is_us_regular_session", lambda *a: True)
    monkeypatch.setattr(shared_runner.time, "sleep", Mock())
    monkeypatch.setattr(sys, "argv", [
        "portfolio_app.py", "--mode", "paper", "--iterations", "2",
        "--directory", str(tmp_path / "account"), "--universe", str(path),
    ])
    configure = Mock(wraps=runner.configure_app)
    monkeypatch.setattr(runner, "configure_app", configure)
    original = runtime.RuntimeControls.generate
    ticks = []
    execute = runner.run_once
    requests = []
    writes = Mock(wraps=operations.atomic_json)
    monkeypatch.setattr(operations, "atomic_json", writes)
    monkeypatch.setattr(runner, "atomic_json", writes)

    def run_once(*args, **kwargs):
        if requests:
            refreshed = json.loads(path.read_text())
            refreshed["rejected"] = [{"symbol": "OTHER.US", "reasons": ["price_below_minimum"]}]
            selection.atomic_json(path, refreshed)
        requests.append((args, kwargs))
        return execute(*args, **kwargs)

    def generate(control, *args, **kwargs):
        ticks.append(1)
        yield from original(control, *args, **kwargs)

    monkeypatch.setattr(runtime.RuntimeControls, "generate", generate)
    monkeypatch.setattr(runner, "run_once", run_once)
    runner.main()
    assert len(ticks) == 2
    assert configure.call_count == 2  # 一次预检，一次账户构建。
    health = [call.args[1] for call in writes.call_args_list if call.args[0].name == "health.json"]
    assert [row["app_reused"] for row in health] == [False, True]
    assert all(row["selected"] == row["managed"] == ["AAA.US"] for row in health)
    client.trade.submit_order.assert_not_called()
    args, kwargs = requests[-1]
    assert kwargs["app_cache"]
    binding = tmp_path / "account/account.json"
    changed = json.loads(binding.read_text())
    changed["capital"] += 1
    selection.atomic_json(binding, changed)
    with pytest.raises(ValueError, match="account mode or capital"):
        execute(*args, **kwargs)
    assert kwargs["app_cache"] == {}


@pytest.mark.parametrize("missing_universe", [False, True, "failed_refresh"])
@pytest.mark.parametrize("reuse_app", [False, True])
def test_shared_app_retains_dropped_positions_through_restart_and_exit(
    selection, runner, portfolio, monkeypatch, tmp_path, missing_universe, reuse_app
):
    app_cache = {} if reuse_app else None
    api = pytest.importorskip("longbridge.openapi")
    from investing_algorithm_framework import (
        LongbridgeClient,
        LongbridgeOAuthCredential,
    )

    monkeypatch.setattr(
        "investing_algorithm_framework.infrastructure.longbridge.execution."
        "is_us_regular_session",
        lambda *a: True,
    )
    now = datetime.now(timezone.utc)
    reference = pd.read_csv(ROOT / "tests/resources/supertrend_rsi_pine.csv")
    index = int(np.flatnonzero(reference.entry_long.to_numpy())[0])
    frame = reference.iloc[: index + 1].copy()
    frame.index = pd.date_range(
        end=now - timedelta(hours=1), periods=len(frame), freq="h"
    )
    monkeypatch.setattr(
        "investing_algorithm_framework.infrastructure.longbridge.runtime.latest_completed_us_bar", lambda *a: frame.index[-1].to_pydatetime()
    )
    monkeypatch.setattr("investing_algorithm_framework.infrastructure.longbridge.runtime.is_us_regular_session", lambda *a: True)
    client = LongbridgeClient(LongbridgeOAuthCredential("test"), trading_enabled=True)
    client.quote, client.trade = Mock(), Mock()
    client.quote.quote.return_value = [
        NS(
            timestamp=now,
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
        NS(cash_infos=[NS(currency="USD", available_cash=Decimal("25000"))])
    ]
    client.trade.stock_positions.return_value = NS(channels=[])
    client.trade.submit_order.return_value = NS(order_id="buy-1")
    path = tmp_path / "universe.json"
    selection.atomic_json(path, snapshot(selection, ["BBB.US", "AAA.US"], now))
    state = runner.run_once(client, tmp_path / "account", path,
                            max_positions=1, app_cache=app_cache)
    assert state["selected"] == ["BBB.US", "AAA.US"]
    client.trade.submit_order.assert_called_once()
    assert client.trade.submit_order.call_args.kwargs["symbol"] == "BBB.US"
    app, retained_pending, _ = runner.build_portfolio_app(client, tmp_path / "account")
    assert retained_pending == ["BBB.US"]
    app.initialize_services()
    buy = app.container.order_service().get_all()[0]
    quantity = Decimal(str(buy.amount))
    client.trade.order_detail.return_value = NS(
        order_id="buy-1",
        status=api.OrderStatus.Filled,
        quantity=quantity,
        executed_quantity=quantity,
        executed_price=Decimal("100"),
        updated_at=now,
        charge_detail=None,
    )
    app.initialize_portfolios()
    app.container.order_service().check_pending_orders()
    selection.atomic_json(path, snapshot(selection, [], datetime.now(timezone.utc)))
    app, tracked, selected = runner.build_portfolio_app(
        client, tmp_path / "account", selection.load_snapshot(path)
    )
    assert tracked == ["BBB.US"] and selected == []
    if missing_universe == "failed_refresh":
        selection.atomic_json(
            path.with_suffix(".error.json"), {"error": "calendar unavailable"}
        )
    elif missing_universe:
        path.unlink()
    distance = buy.metadata["stop_distance"]
    client.quote.set_on_quote.call_args.args[0](
        "BBB.US",
        NS(
            timestamp=datetime.now(timezone.utc),
            last_done=Decimal(str(99 - distance)),
            volume=20,
            trade_status=api.TradeStatus.Normal,
        ),
    )
    client.quote.set_on_depth.call_args.args[0]("BBB.US", NS())
    client.quote.realtime_depth.return_value = NS(
        bids=[NS(price=Decimal(str(99 - distance)))],
        asks=[NS(price=Decimal(str(99.01 - distance)))],
    )
    client.trade.stock_positions.return_value = NS(
        channels=[NS(positions=[NS(symbol="BBB.US", available_quantity=quantity)])]
    )
    client.trade.submit_order.return_value = NS(order_id="sell-1")
    state = runner.run_once(client, tmp_path / "account", path,
                            max_positions=1, app_cache=app_cache)
    assert state["managed"] == ["BBB.US"] and state["selected"] == []
    assert bool(state["selection_issue"]) == bool(missing_universe)
    request = client.trade.submit_order.call_args.kwargs
    assert (
        request["order_type"] == api.OrderType.MO
        and request["side"] == api.OrderSide.Sell
    )
    app, _, _ = runner.build_portfolio_app(client, tmp_path / "account")
    app.initialize_services()
    client.trade.order_detail.return_value = NS(
        order_id="sell-1",
        status=api.OrderStatus.Filled,
        quantity=quantity,
        executed_quantity=quantity,
        executed_price=Decimal(str(99 - distance)),
        updated_at=datetime.now(timezone.utc),
        charge_detail=None,
    )
    app.initialize_portfolios()
    app.container.order_service().check_pending_orders()
    state = runner.run_once(client, tmp_path / "account", path,
                            max_positions=1, app_cache=app_cache)
    assert state["managed"] == []
    assert len(app.container.portfolio_service().get_all()) == 1
    client.close()


def test_partial_buy_fill_and_pending_remainder_share_one_risk_budget(portfolio):
    context = Mock()
    context.get_open_trades.return_value = [
        NS(
            target_symbol="AAA.US",
            available_amount=3,
            buy_order=NS(metadata={"stop_distance": 4}),
        )
    ]
    context.get_positions.return_value = [NS(symbol="AAA.US", amount=3)]
    order = NS(
        target_symbol="AAA.US",
        status="OPEN",
        order_side="BUY",
        remaining=2,
        metadata={"stop_distance": 4},
    )
    context.get_orders.return_value = [order]
    assert portfolio.portfolio_risk(context) == ({"AAA.US"}, 20)
    order.status = "CANCELED"
    assert portfolio.portfolio_risk(context) == ({"AAA.US"}, 12)


def test_naive_sdk_quote_timestamp_can_execute(execution):
    executor, client, context = execution
    client.get_quote.return_value.timestamp = (
        datetime.now().astimezone().replace(tzinfo=None)
    )
    executor.execute(intent(), context, {"stop_distance": 4, "risk_fraction": 0.005})
    context.create_limit_order.assert_called_once()


def test_daily_source_aligns_naive_local_sdk_timestamps(
    selection, tmp_path, monkeypatch
):
    pytest.importorskip("longbridge.openapi")
    monkeypatch.setattr(selection.time, "sleep", lambda *a: None)
    frame = daily(0.1)
    candles = [
        NS(
            timestamp=t.to_pydatetime().astimezone().replace(tzinfo=None),
            open=r.Open,
            high=r.High,
            low=r.Low,
            close=r.Close,
            volume=r.Volume,
            turnover=Decimal("123456789"),
        )
        for t, r in frame.iterrows()
    ]
    client = Mock()
    client.quote.history_candlesticks_by_offset.return_value = candles
    source = selection.SelectionSource(client, tmp_path)
    result = source.daily("AAA.US", AS_OF, 300)
    assert len(result) == 300 and result.DollarVolume.eq(123456789).all()
    source.daily("AAA.US", AS_OF, 300)
    assert client.quote.history_candlesticks_by_offset.call_count == 2


def test_history_rate_limit_retries_but_other_errors_fail(
    selection, tmp_path, monkeypatch
):
    sleeps = Mock()
    monkeypatch.setattr(selection.time, "sleep", sleeps)
    client = Mock()
    client.quote.history_candlesticks_by_offset.side_effect = [
        RuntimeError("request rate limit"),
        ["complete"],
    ]
    source = selection.SelectionSource(client, tmp_path)
    assert source.history("AAA.US") == ["complete"]
    client.quote.history_candlesticks_by_offset.side_effect = RuntimeError(
        "permission denied"
    )
    with pytest.raises(RuntimeError, match="permission denied"):
        source.history("AAA.US")


def test_cli_forces_http_transport_and_preserves_environment(selection, monkeypatch):
    monkeypatch.setenv("LONGBRIDGE_QUOTE_TRANSPORT", "ws")
    monkeypatch.setenv("SELECTION_TEST_ENV", "preserved")
    execute = Mock(return_value=NS(stdout='{"stocks": []}'))
    monkeypatch.setattr(selection.subprocess, "run", execute)
    assert selection.cli_json(["constituent", ".SPX.US"]) == {"stocks": []}
    args, kwargs = execute.call_args
    assert args[0] == ["longbridge", "constituent", ".SPX.US", "--format", "json"]
    assert kwargs["env"]["LONGBRIDGE_QUOTE_TRANSPORT"] == "http"
    assert kwargs["env"]["SELECTION_TEST_ENV"] == "preserved"


def test_earnings_calendar_retries_only_failed_batch(selection, monkeypatch, tmp_path):
    symbols = [f"S{i}.US" for i in range(10)] + ["SOLV.US"]
    dates = [date(2026, 11, n) for n in (24, 25, 27)]
    payload = {"list": [{"infos": [{"type": "report", "symbol": "SOLV.US",
                                    "ext": {"local_date": "2026-11-25"}}]}]}
    execute = Mock(side_effect=[
        NS(stdout='{"list": []}'),
        selection.subprocess.CalledProcessError(1, ["longbridge"], stderr=(
            "Error: error sending request for url (https://openapi.longbridge.cn/v1/quote/"
            "finance_calendar): client error (Connect)"
        )),
        NS(stdout=json.dumps(payload)),
    ])
    sleep = Mock()
    monkeypatch.setattr(selection.subprocess, "run", execute)
    monkeypatch.setattr(selection.time, "sleep", sleep)
    monkeypatch.setenv("LONGBRIDGE_REGION", "global")
    source = selection.SelectionSource(None, tmp_path)

    assert source.blocked_earnings(symbols, dates) == {"SOLV.US"}
    calls = execute.call_args_list
    assert len(calls) == 3 and calls[1] == calls[2]
    assert "SOLV.US" not in calls[0].args[0] and "SOLV.US" in calls[1].args[0]
    assert all(c.kwargs["env"]["LONGBRIDGE_REGION"] == "global" for c in calls)
    sleep.assert_called_once_with(1)


@pytest.mark.parametrize(("failure", "attempts", "detail"), [
    ("client error (Connect)", 3, "client error (Connect)"),
    ("HTTP 429 Too Many Requests", 3, "HTTP 429"),
    ("permission denied (HTTP 403)", 1, "permission denied"),
    ("timeout", 3, "90"),
])
def test_earnings_calendar_propagates_batch_diagnostics(
    selection, monkeypatch, tmp_path, failure, attempts, detail,
):
    error = (
        selection.subprocess.TimeoutExpired(["longbridge"], 90)
        if failure == "timeout" else
        selection.subprocess.CalledProcessError(1, ["longbridge"], stderr=failure)
    )
    request = Mock(side_effect=error)
    sleep = Mock()
    monkeypatch.setattr(selection.time, "sleep", sleep)
    source = selection.SelectionSource(None, tmp_path, run_cli=request)
    with pytest.raises(RuntimeError) as caught:
        source.blocked_earnings(["SOLV.US"], [date(2026, 11, 24), date(2026, 11, 27)])
    message = str(caught.value)
    assert all(part in message for part in ["2026-11-24", "2026-11-27", "SOLV.US", detail])
    assert request.call_count == attempts and sleep.call_count == attempts - 1


def test_watch_retains_snapshot_without_streams_or_trade_context(
    selection, monkeypatch, tmp_path,
):
    import sys
    from investing_algorithm_framework import LongbridgeClient

    now = datetime.now(timezone.utc)
    today = now.astimezone(NEW_YORK).date()
    client = LongbridgeClient(quote_transport="http")
    client.quote = Mock()
    client.quote.trading_days.return_value = NS(trading_days=[
        today + timedelta(days=i) for i in (-1, 0, 1, 2)
    ], half_trading_days=[])
    factory = Mock(return_value=client)
    monkeypatch.setattr(selection, "LongbridgeClient", factory)
    monkeypatch.setattr(selection, "load_dotenv", Mock())
    result = snapshot(selection, ["AAA.US"], now)
    result.update(errors={}, selection_model=selection.SELECTION_MODEL)
    scan = Mock(return_value=result)
    monkeypatch.setattr(selection, "screen", scan)
    monkeypatch.setattr(sys, "argv", [
        "screener.py", "--watch", "--directory", str(tmp_path),
    ])
    waits = []

    def stop_after_two_iterations(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(selection.time, "sleep", stop_after_two_iterations)
    with pytest.raises(KeyboardInterrupt):
        selection.main()
    assert factory.call_args.kwargs == {"paper": False, "quote_transport": "http"}
    assert json.loads((tmp_path / "universe.json").read_text()) == result
    scan.assert_called_once()
    assert client.quote.trading_days.call_count == 2
    client.quote.subscribe.assert_not_called()
    client.quote.subscribe_candlesticks.assert_not_called()
    client.quote.set_on_quote.assert_not_called()
    assert "trade" not in client.__dict__


def test_combined_membership_uses_both_dated_sources_and_reuses_its_own_cache(
    selection, monkeypatch, tmp_path
):
    import io
    universe = importlib.import_module("supertrend_universe")
    day = date(2026, 11, 3)
    large = [f"L{i:03d}.US" for i in range(499)]
    mid = [f"M{i:03d}" for i in range(398)] + ["MOG A", "L000"]
    holdings = ('Fund Holdings as of,"Nov 02, 2026"\nTicker,Name,Asset Class\n'
                + "".join(f"{s},Issuer,Equity\n" for s in mid)
                + "USD,Cash,Cash\nFUT,Future,Futures\n").encode()
    opened = Mock(side_effect=lambda *a, **k: io.BytesIO(holdings))
    monkeypatch.setattr(universe, "urlopen", opened)
    quote = {"total": 499, "stocks": [{"symbol": s} for s in large]}
    cli = Mock(return_value=quote)
    source = selection.SelectionSource(None, tmp_path, cli)
    selection.atomic_json(tmp_path / f"base/{day}.json", {"stocks": quote["stocks"][:200]})
    members, provenance = universe.combined_membership(source, day, AS_OF)
    assert members[:499] == large and len(members) == len(set(members)) == 898
    assert "MOG.A.US" in members and "FUT.US" not in members and "USD.US" not in members
    assert (provenance["sp500_count"], provenance["ijh_equity_count"], provenance["overlap_count"]) == (499, 400, 1)
    assert provenance["ijh_as_of"] == "2026-11-02"
    assert universe.combined_membership(source, day, AS_OF) == (members, provenance)
    cli.assert_called_once_with(universe.SP500_QUERY)
    assert opened.call_count == 1
    assert len(json.loads((tmp_path / f"base/{day}.json").read_text())["stocks"]) == 200


def test_incomplete_or_stale_membership_cannot_publish_a_partial_c_universe(
    selection, monkeypatch, tmp_path
):
    import io
    universe = importlib.import_module("supertrend_universe")
    day = date(2026, 11, 3)
    stocks = [{"symbol": f"L{i}.US"} for i in range(499)]
    cli = Mock(return_value={"total": 500, "stocks": stocks})
    source = selection.SelectionSource(None, tmp_path, cli)
    with pytest.raises(ValueError, match="Incomplete Longbridge"):
        universe.combined_membership(source, day, AS_OF)
    cli.return_value["total"] = 499
    holdings = b'Fund Holdings as of,"Oct 01, 2026"\nTicker,Name,Asset Class\nA,Company,Equity\n'
    monkeypatch.setattr(universe, "urlopen", lambda *a, **k: io.BytesIO(holdings))
    with pytest.raises(ValueError, match="Stale or future"):
        universe.combined_membership(source, day, AS_OF)
    assert not (tmp_path / f"base_c/{day}.json").exists()


def test_formal_screen_ranks_the_union_and_publishes_up_to_one_hundred(
    selection, monkeypatch, tmp_path
):
    day = NOW.astimezone(NEW_YORK).date()
    large = [f"L{i}.US" for i in range(80)]
    mid = [f"M{i}.US" for i in range(30)]
    symbols = large + mid
    index = {s: i for i, s in enumerate(symbols)}
    monkeypatch.setattr(selection, "session_window", lambda *args: (day, AS_OF, [day]))
    monkeypatch.setattr(selection, "combined_membership", lambda *args: (
        symbols, {"sp500_count": 80, "ijh_equity_count": 30, "overlap_count": 0},
    ))
    source = Mock()
    source.blocked_earnings.return_value = {mid[-1]}
    source.daily.side_effect = lambda s, *a: daily(0.05 if s == "SPY.US" else 0.1 + index[s] / 1000)
    source.signal_candlesticks.return_value = [
        NS(timestamp=t, open=r.Open, high=r.High, low=r.Low, close=r.Close, volume=r.Volume)
        for t, r in daily().iloc[-100:].iterrows()
    ]
    result = selection.screen(source, selection.SelectionRules(max_correlation=1), NOW)
    names = [r["symbol"] for r in result["selected"]]
    assert len(names) == 100 and mid[-1] not in names
    assert set(large) & set(names) and set(mid) & set(names)
    assert [r["rank"] for r in result["selected"]] == list(range(1, 101))
    assert result["selection_model"] == selection.SELECTION_MODEL and result["base_pool"] == symbols
    source.blocked_earnings.assert_called_once_with(symbols, [day])
    source.base_pool.assert_not_called()
    path = tmp_path / "universe.json"
    selection.atomic_json(path, result)
    assert selection.load_snapshot(path, NOW)["selected"] == result["selected"]
    with pytest.raises(ValueError):
        selection.SelectionRules(top_n=101)
