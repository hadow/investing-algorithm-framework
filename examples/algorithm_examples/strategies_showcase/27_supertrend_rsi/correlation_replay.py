"""Two correlation timings in independent, identically funded research portfolios."""

from collections import Counter
from dataclasses import dataclass
from itertools import product
import math

import pandas as pd

from correlation_history import CAL, DAYS, ROOT, RULES, bars_path, read
from screener import atomic_json
from settings import (
    CAPITAL,
    MAX_POSITIONS,
    ARCHIVED_MIN_ENTRY_SHARES,
    ORDER_BUDGET,
    RISK_FRACTION,
    TOTAL_RISK_FRACTION,
)


@dataclass
class Lot:
    symbol: str
    shares: int
    price: float
    distance: float
    time: int
    stamp: str


class Account:
    def __init__(self, gate, cost, delayed, min_shares=ARCHIVED_MIN_ENTRY_SHARES, audit_slots=False):
        self.audit_slots = audit_slots
        self.minutes = self.full_minutes = 0
        self.exposure_sum = self.spread_cost = 0.0
        self.min_shares = min_shares
        self.gate = gate
        self.fee = 0.0001 * cost
        self.half_spread = 0.0002 * cost
        self.delayed = delayed
        self.cash = float(CAPITAL)
        self.realized = self.receivable = self.dividends = self.cost = self.turnover = (
            0.0
        )
        self.held, self.pending, self.payments = {}, {}, []
        self.exit_latches, self.missing_while_held = set(), []
        self.fills, self.decisions, self.curve = [], [], []
        self.peak = self.equity = float(CAPITAL)
        self.dd = self.peak_risk_pct = self.peak_slots = 0

    def risk(self):
        return sum(
            lot.shares * lot.distance
            for book in (self.held, self.pending)
            for lot in book.values()
        )

    def net_size(self):
        return CAPITAL + self.realized + self.dividends - self.receivable

    def entry(self, sym, midpoint, distance, matrix, minute, stamp):
        occupied = {**self.held, **self.pending}
        ask = midpoint * (1 + self.half_spread)
        event = {"symbol": sym, "time": stamp}
        self.decisions.append(event)
        if sym in occupied:
            event["reason"] = "already_held_or_pending"
            return
        if len(occupied) >= MAX_POSITIONS:
            event["reason"] = "positions_full"
            if self.audit_slots:
                event["occupants"] = {
                    s: {"entry": p.stamp, "pending": s in self.pending}
                    for s, p in occupied.items()
                }
            return
        reserved = sum(
            v.shares * v.price * (1 + self.fee) for v in self.pending.values()
        )
        net = self.net_size()
        if not 0 < distance < ask:
            raise ValueError(f"Invalid stop distance: {sym} {stamp}")
        shares = math.floor(
            min(
                ORDER_BUDGET / ask,
                (self.cash - reserved) * 0.98 / ask,
                net * RISK_FRACTION / distance,
                (net * TOTAL_RISK_FRACTION - self.risk()) / distance,
            )
        )
        if shares < self.min_shares:
            event["reason"] = "budget_or_risk"
            return
        event["compared_to"] = {
            peer: matrix.get(sym, {}).get(peer) for peer in occupied
        }
        event["pending_peers"] = list(self.pending)
        for peer, value in event["compared_to"].items():
            if self.gate == "order" and (
                value is None or value > RULES.max_correlation
            ):
                event.update(
                    reason="unknown_correlation" if value is None else "correlation",
                    peer=peer,
                    peer_pending=peer in self.pending,
                    correlation=value,
                )
                return
        self.pending[sym] = Lot(sym, shares, ask, distance, minute, stamp)
        event.update(reason="order_reserved", shares=shares)
        risk_pct = 100 * self.risk() / net
        if risk_pct > 100 * TOTAL_RISK_FRACTION + 1e-8:
            raise AssertionError(
                "Combined positions and pending orders exceed the risk budget"
            )
        self.peak_risk_pct = max(self.peak_risk_pct, risk_pct)
        self.peak_slots = max(self.peak_slots, len(occupied) + 1)

    def settle(self, opens, minute, stamp):
        for sym, order in list(self.pending.items()):
            age = minute - order.time
            if self.delayed and age == 0:
                continue
            if age >= 2:
                self.decisions.append(
                    {"time": stamp, "symbol": sym, "reason": "limit_expired"}
                )
                del self.pending[sym]
                continue
            if sym not in opens:
                continue
            ask = opens[sym] * (1 + self.half_spread)
            if ask > order.price:
                continue
            gross = order.shares * ask
            self.cash -= gross * (1 + self.fee)
            if self.cash < -1e-7:
                raise AssertionError("Negative cash")
            self.cost += gross * self.fee
            self.spread_cost += gross * self.half_spread / (1 + self.half_spread)
            self.turnover += gross
            self.held[sym] = Lot(sym, order.shares, ask, order.distance, minute, stamp)
            del self.pending[sym]
            self.fills.append(
                {
                    "time": stamp,
                    "side": "buy",
                    "symbol": sym,
                    "qty": order.shares,
                    "price": ask,
                    "stop": ask - order.distance,
                }
            )

    def sell(self, sym, bid, stamp, reason):
        lot = self.held.pop(sym)
        gross = lot.shares * bid
        pnl = gross * (1 - self.fee) - lot.shares * lot.price * (1 + self.fee)
        self.cash += gross * (1 - self.fee)
        self.realized += pnl
        self.cost += gross * self.fee
        self.spread_cost += gross * self.half_spread / (1 - self.half_spread)
        self.turnover += gross
        self.fills.append(
            {
                "time": stamp,
                "side": "sell",
                "symbol": sym,
                "qty": lot.shares,
                "price": bid,
                "reason": reason,
                "entry": lot.stamp,
                "net_pnl": pnl,
            }
        )

    def corporate_actions(self, day, actions):
        for sym, lot in self.held.items():
            event = actions.get(sym, {}).get(day, {})
            if event.get("unsupported") or event.get("split", 1) != 1:
                raise ValueError(
                    f"Carried position requires split or distribution reconciliation: {sym} {day}"
                )
            for payday, amount in event.get("payments", []):
                value = lot.shares * amount
                self.receivable += value
                self.dividends += value
                self.payments.append((payday, value))
        for pay in self.payments.copy():
            if pay[0] and pay[0] <= day:
                self.cash += pay[1]
                self.receivable -= pay[1]
                self.payments.remove(pay)

    def mark(self, closes):
        self.equity = (
            self.cash
            + self.receivable
            + sum(p.shares * closes[s] for s, p in self.held.items())
        )
        if self.audit_slots:
            self.minutes += 1
            self.full_minutes += len(self.held) + len(self.pending) == MAX_POSITIONS
            self.exposure_sum += (self.equity - self.cash - self.receivable) / self.equity
        self.peak = max(self.peak, self.equity)
        self.dd = max(self.dd, 100 * (1 - self.equity / self.peak))

    def result(self):
        counts = Counter(r["reason"] for r in self.decisions)
        buys = sum(r["side"] == "buy" for r in self.fills)
        return {
            "metrics": {
                **({
                    "average_invested_pct": self.exposure_sum * 100 / self.minutes,
                    "full_minutes": self.full_minutes,
                    "session_minutes": self.minutes,
                    "modeled_spread_cost_usd": self.spread_cost,
                } if self.audit_slots and self.minutes else {}),
                "net_pct": 100 * (self.equity / CAPITAL - 1),
                "net_usd": self.equity - CAPITAL,
                "minute_max_drawdown_pct": self.dd,
                "entries": buys,
                "exits": len(self.fills) - buys,
                "ending_open": len(self.held),
                "peak_occupied": self.peak_slots,
                "peak_initial_risk_pct": self.peak_risk_pct,
                "ending_cash": self.cash,
                "realized_trade_pnl": self.realized,
                "dividend_accrual": self.dividends,
                "commission_usd": self.cost,
                "turnover_pct": self.turnover * 100 / CAPITAL,
                "decisions": dict(counts),
            },
            "daily": self.curve,
            "fills": self.fills,
            "entry_decisions": self.decisions,
            "missing_quote_while_held": self.missing_while_held,
        }


def market_tape(prep, *, days=DAYS, minute_rows=None, actions=None):
    symbols = prep["minute_symbols"]
    frames, quality = {}, []
    timestamps = {
        day: pd.date_range(
            CAL.session_open(day), CAL.session_close(day), freq="min", inclusive="left"
        )
        for day in days
    }
    index = timestamps[days[0]].append([timestamps[d] for d in days[1:]])
    for sym in symbols:
        rows = []
        if minute_rows is None:
            for start, end in ((DAYS[0], DAYS[1]), (DAYS[2], DAYS[3])):
                rows.extend(read(bars_path("1m", sym, start, end))["response"])
        else:
            for day in days:
                rows.extend(minute_rows(sym, day))
        by_time = {r["time"]: r for r in rows}
        if len(by_time) != len(rows):
            raise ValueError(f"Duplicate minute: {sym}")
        for day in days:
            retry = bars_path("1m", sym, day, day)
            if retry.is_file():
                for r in read(retry)["response"]:
                    if float(r["open"]) > 0 and float(r["close"]) > 0:
                        old = by_time.get(r["time"])
                        if old and min(float(old["open"]), float(old["close"])) > 0:
                            if (float(r["open"]), float(r["close"])) != (
                                float(old["open"]),
                                float(old["close"]),
                            ):
                                quality.append(
                                    {
                                        "symbol": sym,
                                        "minute": r["time"],
                                        "reason": "valid_original_retained",
                                        "original": [old["open"], old["close"]],
                                        "single_day": [r["open"], r["close"]],
                                    }
                                )
                        else:
                            by_time[r["time"]] = r
                            quality.append(
                                {
                                    "symbol": sym,
                                    "minute": r["time"],
                                    "reason": "recovered_from_single_day",
                                }
                            )
        frame = pd.DataFrame(list(by_time.values()))
        frame.index = pd.to_datetime(frame.pop("time"), utc=True).dt.as_unit("ns")
        if len(frame.index.difference(index)) or frame.index.has_duplicates:
            raise ValueError(f"Unexpected minute timestamps: {sym}")
        prices = frame[["open", "close"]].astype(float).reindex(index)
        valid = (prices > 0).all(axis=1) & (prices < float("inf")).all(axis=1)
        for stamp in prices.index[~valid]:
            quality.append(
                {
                    "symbol": sym,
                    "minute": stamp.isoformat(),
                    "reason": "invalid_reported_price"
                    if stamp in frame.index
                    else "missing_bar",
                }
            )
        prices.loc[~valid] = float("nan")
        # A stale mark can value an existing holding. It must never create a quote for a fill.
        prices["close"] = prices["close"].ffill()
        frames[sym] = prices.to_numpy()
    tape = {}
    if actions is None:
        actions = read(ROOT / "inputs/actions.json")
    offset = 0
    for i, day in enumerate(days):
        ticks = []
        for j, ts in enumerate(timestamps[day]):
            stamp = ts.isoformat()
            group = prep["signals"][day].get(stamp, {})
            if not j and i:
                group = prep["signals"][days[i - 1]].get(
                    CAL.session_close(days[i - 1]).isoformat(), {}
                )
            opens = {
                s: v[offset, 0]
                for s, v in frames.items()
                if math.isfinite(v[offset, 0])
            }
            closes = {s: v[offset, 1] for s, v in frames.items()}
            ticks.append((offset, stamp, opens, closes, group))
            offset += 1
        tape[day] = ticks
    return tape, actions, quality


def run(prep, tape, actions, pools, gate, cost=1, delayed=False, *,
        min_shares=ARCHIVED_MIN_ENTRY_SHARES, audit_slots=False, first_at=None):
    book = Account(gate, cost, delayed, min_shares=min_shares, audit_slots=audit_slots)
    for day in sorted(pools):
        if book.pending:
            raise AssertionError("Carried a day order across sessions")
        book.corporate_actions(day, actions)
        ranks = {s: i for i, s in enumerate(pools[day])}
        for minute, stamp, opens, closes, events in tape[day]:
            book.settle(opens, minute, stamp)
            just_exited = set()
            for sym, pos in list(book.held.items()):
                if events.get(sym, {}).get("sell"):
                    book.exit_latches.add(sym)
                if sym not in opens:
                    book.missing_while_held.append({"time": stamp, "symbol": sym})
                    continue
                bid = opens[sym] * (1 - book.half_spread)
                stop = bid <= pos.price - pos.distance
                if stop or sym in book.exit_latches:
                    book.sell(sym, bid, stamp, "ATR" if stop else "hourly_exit")
                    book.exit_latches.discard(sym)
                    just_exited.add(sym)
            preferred = first_at.get(stamp) if first_at else None
            for sym in sorted(
                (
                    s
                    for s, e in events.items()
                    if e["buy"] and s in ranks and s not in just_exited
                ),
                key=lambda s: (s != preferred, ranks[s]),
            ):
                if sym not in opens:
                    book.decisions.append(
                        {"time": stamp, "symbol": sym, "reason": "missing_quote"}
                    )
                    continue
                book.entry(
                    sym,
                    opens[sym],
                    events[sym]["distance"],
                    prep["correlations"][day],
                    minute,
                    stamp,
                )
            if not delayed:
                book.settle(opens, minute, stamp)
            book.mark(closes)
        book.curve.append(
            {
                "date": day,
                "equity": book.equity,
                "cash": book.cash,
                "holdings": {s: p.shares for s, p in book.held.items()},
            }
        )
    return book.result()


def main():
    prep = read(ROOT / "inputs/prepared.json")
    tape, actions, quality = market_tape(prep)
    pools_a = {d: prep["lists"][d]["A"] for d in DAYS}
    pools_b = {d: prep["lists"][d]["B"] for d in DAYS}
    results = {}
    for label, cost, delayed in (
        ("base", 1, False),
        ("double_cost", 2, False),
        ("delayed_buy", 1, True),
    ):
        results[label] = {
            "A": run(prep, tape, actions, pools_a, "candidate", cost, delayed),
            "B": run(prep, tape, actions, pools_b, "order", cost, delayed),
        }
    alternatives = [prep["lists"][d]["B_projections"] for d in DAYS]
    possible = []
    for combo in product(*alternatives):
        outcome = run(prep, tape, actions, dict(zip(DAYS, combo)), "order")
        possible.append(
            {
                "daily_signal_lists": dict(zip(DAYS, combo)),
                "metrics": outcome["metrics"],
            }
        )
    a, b = results["base"]["A"], results["base"]["B"]
    a_orders = {(x["time"], x["symbol"]) for x in a["fills"] if x["side"] == "buy"}
    b_orders = {(x["time"], x["symbol"]) for x in b["fills"] if x["side"] == "buy"}
    new, displaced = sorted(b_orders - a_orders), sorted(a_orders - b_orders)
    signals = [
        (d, s, t)
        for d in DAYS
        for t, group in prep["signals"][d].items()
        for s, e in group.items()
        if e["buy"]
    ]
    diffs = {
        "signals_in_B_outside_A": [
            (t, s) for d, s, t in signals if s in pools_b[d] and s not in pools_a[d]
        ],
        "signals_in_A_outside_B": [
            (t, s) for d, s, t in signals if s in pools_a[d] and s not in pools_b[d]
        ],
        "B_only_buys": new,
        "A_only_buys": displaced,
    }
    ranges = {
        k: [
            min(v["metrics"][k] for v in possible),
            max(v["metrics"][k] for v in possible),
        ]
        for k in (
            "net_pct",
            "minute_max_drawdown_pct",
            "entries",
            "exits",
            "ending_open",
        )
    }
    report = {
        "dates": DAYS,
        "selection": prep["lists"],
        "minute_symbols": prep["minute_symbols"],
        "minute_data_audit": quality,
        "comparison": diffs,
        "runs": results,
        "earnings_sensitivity": {
            "distinct_signal_list_paths": len(possible),
            "base_cost_B_ranges": ranges,
            "paths": possible,
        },
        "model": {
            "capital": CAPITAL,
            "order_budget": ORDER_BUDGET,
            "min_shares": ARCHIVED_MIN_ENTRY_SHARES,
            "max_slots": MAX_POSITIONS,
            "trade_risk": RISK_FRACTION,
            "total_risk": TOTAL_RISK_FRACTION,
            "threshold": RULES.max_correlation,
            "half_spread_bps": 2,
            "fee_bps_per_side": 1,
            "execution": "Minute-open bid/ask proxies, exits first, reserve every ranked entry before batch fills. Delayed-buy sensitivity samples next-minute ask and expires unfilled orders at 120 seconds.",
            "valuation": "Minute-close mark, end positions open. Dividend receivables included at ex date. Married entry/exit RSI signals follow existing exit priority when held.",
        },
    }
    atomic_json(ROOT / "comparison.json", report)
    for scenario, pair in results.items():
        print(
            scenario,
            {
                k: {
                    m: round(v["metrics"][m], 4)
                    for m in (
                        "net_pct",
                        "minute_max_drawdown_pct",
                        "entries",
                        "exits",
                        "ending_open",
                    )
                }
                for k, v in pair.items()
            },
        )
    print("Signal differences", diffs)
    print("Earnings sensitivity", len(possible), "distinct signal paths", ranges)


if __name__ == "__main__":
    main()
