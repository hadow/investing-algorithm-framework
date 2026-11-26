"""Reconstruct ranked lists and hourly signals without using intraday prices to select stocks."""

from datetime import date
from itertools import product
import math

import pandas as pd

from correlation_history import (
    CAL,
    DAYS,
    HOUR_START,
    ROOT,
    RULES,
    bars_path,
    load_daily,
    previous,
    read,
)
from investing_algorithm_framework.infrastructure.longbridge.selection import (
    candidate_correlations,
)
from settings import ARCHIVED_MIN_ENTRY_SHARES
from screener import atomic_json
from supertrend_rsi import prepare_indicators


def hourly_rows(symbol):
    rows = read(bars_path("1h", symbol, HOUR_START, DAYS[-1]))["response"]
    frame = pd.DataFrame(rows).rename(columns=str.capitalize)
    frame.index = pd.to_datetime(frame.pop("Time"), utc=True).dt.as_unit("ns")
    return frame.astype(float)


def hourly_inputs(symbols, *, days=DAYS, load_frame=hourly_rows):
    sessions = CAL.sessions_in_range(HOUR_START, days[-1])
    starts = pd.DatetimeIndex(
        [
            t
            for d in sessions
            for t in pd.date_range(
                CAL.session_open(d), CAL.session_close(d), freq="h", inclusive="left"
            )
        ]
    )
    sizing = {d: {} for d in days}
    signals = {d: {} for d in days}
    for sym in symbols:
        frame = load_frame(sym)
        if (
            not starts.equals(frame.index)
            or (frame[["High", "Low", "Close", "Open"]] <= 0).any().any()
        ):
            raise ValueError(f"Incomplete hourly history: {sym}")
        for day in days:
            opening = CAL.session_open(day)
            close = CAL.session_close(day)
            first = starts.get_loc(opening)
            if first < 300:
                raise ValueError("Insufficient warm-up")
            size = prepare_indicators(frame.iloc[first - 300 : first]).iloc[-1]
            sizing[day][sym] = math.floor(
                min(
                    RULES.order_budget / size.Close,
                    RULES.capital * 0.98 / size.Close,
                    RULES.capital * RULES.risk_fraction / (2 * size.ATR),
                )
            )
            for i in range(first, starts.searchsorted(close)):
                row = prepare_indicators(frame.iloc[i - 299 : i + 1]).iloc[-1]
                when = min(starts[i] + pd.Timedelta(hours=1), close)
                buy = bool(row.entry_long) and opening + pd.Timedelta(
                    minutes=5
                ) <= when < close - pd.Timedelta(minutes=5)
                if buy or row.exit_long:
                    stamp = when.isoformat()
                    signals[day].setdefault(stamp, {})[sym] = {
                        "buy": buy,
                        "sell": bool(row.exit_long),
                        "distance": float(2 * row.ATR),
                    }
    return sizing, signals


def correlation_matrix(day, symbols):
    as_of = previous(day)
    histories, unavailable = {}, []
    for sym in symbols:
        try:
            histories[sym] = load_daily(as_of, sym)
        except FileNotFoundError:
            # A retained position can be absent from today's archived 200. Leave its
            # estimate unknown so an order depending on it cannot obtain permission.
            unavailable.append(sym)
    keys = [{"symbol": s} for s in histories]
    matrix = candidate_correlations(histories, keys, as_of)
    return {
        s: {t: float(v) for t, v in matrix[s].dropna().items()} for s in matrix
    }, unavailable


def calendar_estimate(day):
    dates = set(
        CAL.sessions[CAL.sessions.get_loc(day) : CAL.sessions.get_loc(day) + 3].date
    )
    blocked = set()
    for p in (ROOT / "inputs/earnings").glob("*.json"):
        for group in read(p)["response"]["list"]:
            for event in group["infos"]:
                if event.get("type") not in {"report", "financial"}:
                    continue
                local = event.get("ext", {}).get("local_date")
                dt = (
                    date.fromisoformat(local)
                    if local
                    else pd.Timestamp(int(event["datetime"]), unit="s", tz="UTC")
                    .tz_convert("America/New_York")
                    .date()
                )
                if dt in dates:
                    blocked.add(event["symbol"])
    return blocked


def selections(day, observed, sizing, corr, signals):
    public = observed["production"][day]["selected"]
    ranks = observed["daily"][day]["ranked"]
    viable = [s for s in ranks if sizing[s] >= ARCHIVED_MIN_ENTRY_SHARES]
    if [s for s in viable if s in public] != public:
        raise ValueError(
            f"Archived order or sizing mismatch for {day}: {set(public) - set(viable)}"
        )
    known = observed["known_reports"].get(day, {}).get("earnings_pass", {})
    blocked = calendar_estimate(day)
    eligible, unknown, past_public = [], [], []
    for sym in viable:
        if len(past_public) == RULES.top_n:
            break
        high = any(corr[sym][peer] > RULES.max_correlation for peer in past_public)
        if sym in public:
            if high or known.get(sym) is False:
                raise ValueError(
                    f"Archived selection disagrees with dated inputs: {day} {sym}"
                )
            eligible.append(sym)
            past_public.append(sym)
        elif high and known.get(sym, True):
            # The original gate would reject this name before its hourly budget check.
            eligible.append(sym)
            if sym not in known:
                unknown.append(sym)
        elif sym in known and known[sym] and not high:
            raise ValueError(f"Unexpected omission in archived baseline: {day} {sym}")
    buying = {s for group in signals.values() for s, e in group.items() if e["buy"]}
    projections = set()
    for switches in product((False, True), repeat=len(unknown)):
        excluded = {s for s, keep in zip(unknown, switches) if not keep}
        pool = [s for s in eligible if s not in excluded][: RULES.top_n]
        projections.add(tuple(s for s in pool if s in buying))
    late = [s for s in eligible if s not in unknown or s not in blocked][: RULES.top_n]
    return {
        "A": public,
        "B": late,
        "B_unknown_earnings": unknown,
        "estimated_blocked_unknown": sorted(set(unknown) & blocked),
        "B_projections": [list(v) for v in sorted(projections)],
        "upper_candidate_union": sorted(set(eligible)),
        "base_observed": observed["daily"][day]["base_observed"],
        "missing_daily": observed["daily"][day]["missing_daily"],
    }


def prepare():
    observed = read(ROOT / "inputs/observed.json")
    syms = sorted({s for d in observed["daily"].values() for s in d["ranked"]})
    sizing, signals = hourly_inputs(syms)
    lists, corr, unavailable = {}, {}, {}
    minute_symbols = set()
    for day in DAYS:
        corr[day], unavailable[day] = correlation_matrix(day, syms)
        lists[day] = selections(day, observed, sizing[day], corr[day], signals[day])
        potential = set(lists[day]["A"])
        for group in lists[day]["B_projections"]:
            potential.update(group)
        minute_symbols.update(
            s
            for group in signals[day].values()
            for s, e in group.items()
            if e["buy"] and s in potential
        )
    # For the order-time test only symbols that can actually trade need minute marks and pair estimates.
    matrices = {
        d: {
            s: {t: corr[d].get(s, {}).get(t) for t in sorted(minute_symbols)}
            for s in sorted(minute_symbols)
        }
        for d in DAYS
    }
    atomic_json(
        ROOT / "inputs/prepared.json",
        {
            "dates": DAYS,
            "lists": lists,
            "correlations": matrices,
            "missing_dated_correlations": {
                d: sorted(set(unavailable[d]) & minute_symbols) for d in DAYS
            },
            "signals": signals,
            "minute_symbols": sorted(minute_symbols),
        },
    )
    print("Minute symbols", len(minute_symbols))
    for d, v in lists.items():
        print(
            d,
            len(v["A"]),
            len(v["B"]),
            "earnings unknown",
            v["B_unknown_earnings"],
            "distinct orderable projections",
            len(v["B_projections"]),
        )


if __name__ == "__main__":
    prepare()
