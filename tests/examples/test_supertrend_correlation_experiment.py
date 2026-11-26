"""Regression tests for paired selection timing, reservations and ambiguous archived eligibility."""

from dataclasses import replace
import importlib
from pathlib import Path

import pytest


EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/algorithm_examples/strategies_showcase/27_supertrend_rsi"
)


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return importlib.import_module("correlation_replay")


def test_correlation_uses_pending_and_carried_positions(replay):
    b = replay.Account("order", 1, False)
    matrix = {"B": {"A": 0.851}, "C": {"A": 0.85}}
    b.entry("A", 100, 2, matrix, 0, "open")
    b.entry("B", 100, 2, matrix, 0, "open")
    assert (
        b.decisions[-1]["reason"] == "correlation" and b.decisions[-1]["peer_pending"]
    )
    b.settle({"A": 100}, 0, "open")
    b.entry("B", 100, 2, matrix, 1, "next")
    assert (
        b.decisions[-1]["reason"] == "correlation"
        and not b.decisions[-1]["peer_pending"]
    )
    b.entry("C", 100, 2, matrix, 1, "next")
    assert "C" in b.pending
    b.entry("D", 100, 2, matrix, 1, "next")
    assert b.decisions[-1]["reason"] == "unknown_correlation"
    b.sell("A", 99, "next", "ATR")
    b.entry("B", 100, 2, {"B": {"C": -0.95}}, 1, "next")
    assert "B" in b.pending


def test_risk_reservation_transfers_on_fill_and_returns_on_expiry(replay):
    b = replay.Account("candidate", 1, True)
    b.entry("A", 50, 2, {}, 0, "open")
    qty, risk = b.pending["A"].shares, b.risk()
    assert qty == 62 and risk == 124
    b.settle({"A": 51}, 1, "t1")
    assert b.risk() == risk and not b.held
    b.settle({"A": 50}, 2, "t2")
    assert b.risk() == 0 and b.cash == replay.CAPITAL
    b.entry("A", 50, 2, {}, 3, "t3")
    b.settle({"A": 49}, 4, "t4")
    assert b.risk() == risk and b.held["A"].price < 50
    b.sell("A", 51, "t5", "hourly_exit")
    b.mark({})
    assert b.risk() == 0
    assert b.equity == pytest.approx(replay.CAPITAL + b.realized)


def test_five_orders_share_risk_before_any_fill(replay):
    b = replay.Account("candidate", 1, False)
    for s in "ABCDEF":
        b.entry(s, 50, 2, {}, 0, "open")
    assert len(b.pending) == 5 and b.risk() == 620
    assert b.peak_risk_pct <= 2.5 and b.decisions[-1]["reason"] == "positions_full"


def test_unknown_earnings_can_displace_a_known_lower_ranked_signal(replay, monkeypatch):
    prep = importlib.import_module("correlation_prepare")
    monkeypatch.setattr(prep, "RULES", replace(replay.RULES, top_n=2))
    monkeypatch.setattr(prep, "calendar_estimate", lambda day: set())
    info = {
        "production": {"d": {"selected": ["A", "B"]}},
        "known_reports": {},
        "daily": {
            "d": {
                "ranked": ["A", "H", "B"],
                "base_observed": "prior",
                "missing_daily": [],
            }
        },
    }
    data = prep.selections(
        "d",
        info,
        {"A": 10, "B": 11, "H": 12},
        {"H": {"A": 0.90}, "B": {"A": 0.2}},
        {"now": {"H": {"buy": True}, "B": {"buy": True}}},
    )
    assert data["A"] == ["A", "B"] and data["B"] == ["A", "H"]
    assert data["B_projections"] == [["B"], ["H"]]


def test_rank_can_only_displace_a_new_reservation_in_the_same_minute(replay):
    research = importlib.import_module("universe_study")
    day = "2026-11-18"
    first, later = "2026-11-18T15:30:00+00:00", "2026-11-18T16:30:00+00:00"
    symbols = list("ABCDEFG")
    opens = dict.fromkeys(symbols, 1500)
    closes = {**opens, "F": 1600}
    signal = {"buy": True, "sell": False, "distance": 12}
    tape = {
        day: [
            (0, first, opens, opens, dict.fromkeys(symbols[:6], signal)),
            (1, later, opens, closes, {"G": signal}),
        ]
    }
    prep = {"correlations": {day: {}}}

    def play(priority):
        return research.attribute(
            replay.run(
                prep,
                tape,
                {},
                {day: symbols},
                "candidate",
                cost=0,
                min_shares=1,
                audit_slots=True,
                first_at=priority,
            )
        )

    baseline, switched = play({}), play({first: "F"})
    blocked = [
        r for r in baseline["entry_decisions"] if r["reason"] == "positions_full"
    ]
    assert blocked[0]["same_minute_occupants"] == list("ABCDE")
    assert blocked[1]["same_minute_occupants"] == []
    assert all(r["qty"] == 3 for r in baseline["fills"])
    assert {r["symbol"] for r in switched["fills"]} == set("ABCDF")
    assert switched["metrics"]["net_usd"] - baseline["metrics"][
        "net_usd"
    ] == pytest.approx(300)
    assert switched["metrics"]["peak_occupied"] == 5


def test_liquidity_priority_reorders_only_the_selected_names(replay):
    research = importlib.import_module("universe_study")
    rows = [
        {"symbol": s, "rank": i, "dollar_volume_20": liquidity}
        for i, (s, liquidity) in enumerate([("A", 10), ("B", 50), ("C", 50)], 1)
    ]
    assert research.ordered(rows, "liquidity") == ["B", "C", "A"]
    assert research.ordered(rows, "rank") == ["A", "B", "C"]
