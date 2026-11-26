"""Historical, read-only inputs for the four archived 200-stock sessions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import time

import exchange_calendars as xc
import pandas as pd

from investing_algorithm_framework.infrastructure.longbridge.selection import (
    cli_json,
    rank_candidates,
)
from screener import SelectionRules, atomic_json


HERE = Path(__file__).resolve().parent
SELECTION = HERE / "resources/selection"
ROOT = HERE / "resources/research/correlation-timing"
DAYS = ("2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18")
HOUR_START = "2026-08-24"
CAL = xc.get_calendar("XNYS", start="2026-08-01", end="2026-11-30")
RULES = SelectionRules(base_size=200, top_n=60)


def read(path):
    return json.loads(path.read_text())


def previous(day):
    return CAL.previous_session(day).date()


def load_daily(as_of, symbol):
    path = SELECTION / "daily" / str(as_of) / f"{symbol}.csv"
    frame = pd.read_csv(path, index_col="Datetime")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def ranked_days():
    days = {}
    for day in DAYS:
        as_of = previous(day)
        base = SELECTION / "base" / f"{day}.json"
        observed = datetime.fromtimestamp(base.stat().st_mtime, timezone.utc)
        if observed >= CAL.session_open(day):
            raise ValueError(f"Late base universe for {day}")
        symbols = [r["symbol"] for r in read(base)["stocks"]]
        if len(set(symbols)) != RULES.base_size or len(symbols) != RULES.base_size:
            raise ValueError(f"Incomplete base: {day}")
        histories = {
            s: load_daily(as_of, s)
            for s in symbols
            if (SELECTION / "daily" / str(as_of) / f"{s}.csv").exists()
        }
        ranked, _ = rank_candidates(
            histories, load_daily(as_of, "SPY.US"), as_of, RULES
        )
        days[day] = {
            "ranked": [r["symbol"] for r in ranked],
            "base_observed": observed.isoformat(),
            "missing_daily": sorted(set(symbols) - histories.keys()),
        }
    return days


def archive():
    path = ROOT / "inputs/observed.json"
    if path.exists():
        return read(path)
    observed = {}
    for log in sorted(SELECTION.glob("screener*.log")):
        contents = log.read_text()
        for match in re.finditer(r"^\{", contents, re.M):
            record, _ = json.JSONDecoder().raw_decode(contents[match.start() :])
            day = record["trading_date"]
            if (
                day in DAYS
                and record.get("selection_model") == "risk_adjusted_momentum_v2"
            ):
                item = {k: record[k] for k in ("selected", "data_errors")}
                if day in observed and observed[day] != item:
                    raise ValueError(f"Conflicting production lists for {day}")
                observed[day] = item
    if set(observed) != set(DAYS):
        raise ValueError("Missing contemporaneous ranked production list")
    exact = {}
    for p in (
        HERE / "resources/research/selection-v2-2026-11-13/optimized-universe.json",
        SELECTION / "universe.json",
    ):
        snap = read(p)
        if snap["trading_date"] not in DAYS or datetime.fromisoformat(
            snap["generated_at"]
        ) >= CAL.session_open(snap["trading_date"]):
            raise ValueError("Expected historical pre-open selection snapshot")
        # Daily filters, including the earnings gate, run before rank truncation.
        known = {r["symbol"]: True for r in snap["selected"]}
        for row in snap["rejected"]:
            known[row["symbol"]] = "earnings_within_two_sessions" not in row["reasons"]
        exact[snap["trading_date"]] = {
            "earnings_pass": known,
            "generated_at": snap["generated_at"],
            "file": str(p.relative_to(HERE)),
            "sha256": sha256(p.read_bytes()).hexdigest(),
        }
    result = {"production": observed, "known_reports": exact, "daily": ranked_days()}
    atomic_json(path, result)
    return result


def fetch(path, args):
    if path.exists():
        cached = read(path)
        if cached["arguments"] != args:
            raise ValueError(
                f"Historical request differs from the archived request: {path}"
            )
        return cached["response"]
    for attempt in range(3):
        try:
            response = cli_json(args)
            break
        except subprocess.CalledProcessError as exc:
            error = exc.stderr.strip()
            if attempt == 2 or not any(
                s in error.lower()
                for s in ("301606", "429", "rate", "timeout", "502", "503")
            ):
                raise RuntimeError(f"{args[:4]}: {error}") from exc
            time.sleep(2**attempt)
    atomic_json(
        path,
        {
            "arguments": args,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "response": response,
        },
    )
    return response


def bars_path(period, symbol, start, end):
    return ROOT / "inputs" / period / f"{symbol}_{start}_{end}.json"


def download_candles(job):
    symbol, period, start, end = job
    fetch(
        bars_path(period, symbol, start, end),
        [
            "kline",
            "history",
            symbol,
            "--period",
            period,
            "--start",
            start,
            "--end",
            end,
            "--session",
            "intraday",
            "--adjust",
            "none",
        ],
    )


def history():
    info = archive()
    candidates = sorted({s for d in info["daily"].values() for s in d["ranked"]})
    with ThreadPoolExecutor(max_workers=3) as pool:
        for _ in pool.map(
            download_candles, [(s, "1h", HOUR_START, DAYS[-1]) for s in candidates]
        ):
            pass
    for offset in range(0, len(candidates), 10):
        names = candidates[offset : offset + 10]
        args = [
            "finance-calendar",
            "report",
            "--market",
            "US",
            "--start",
            "2026-11-12",
            "--end",
            "2026-11-21",
            "--count",
            "1000",
        ]
        for name in names:
            args.extend(["--symbol", name])
        path = ROOT / "inputs/earnings" / ("_".join(names) + ".json")
        result = fetch(path, args)
        if result.get("next_date") or not isinstance(result.get("list"), list):
            raise ValueError(f"Incomplete calendar at {path}")
    print(
        f"Downloaded hourly data for {len(candidates)} daily-ranked stocks and earnings calendars."
    )


def corporate_events(sym, rows, days):
    """把除息与支付日归入指定窗口；遇到无法还原的公司行动则停止回放。"""
    dates = {
        r["id"]: str(datetime.strptime(r["date"], "%Y%m%d").date())
        for r in rows
        if r["date_type"] == "Payment Date"
    }
    events = {}
    for row in rows:
        if row["date_type"] != "Ex Date":
            continue
        day = str(datetime.strptime(row["date"], "%Y%m%d").date())
        if day not in days:
            continue
        amount = re.search(
            r"(?:dividend|distribution)\s+(?:of\s+)?(?:USD\s*)?([\d.]+)\s*USD\b",
            row["act_desc"],
            re.I,
        )
        if row["action"] != "DividendExDate" or not amount:
            raise ValueError(
                f"Review the corporate action for {sym} {day}: {row['act_desc']}"
            )
        events.setdefault(day, {"payments": []})["payments"].append(
            [dates.get(row["id"]), float(amount[1])]
        )
    return events


def minute_history():
    prep = read(ROOT / "inputs/prepared.json")
    symbols = prep["minute_symbols"]
    jobs = [
        (s, "1m", start, end)
        for s in symbols
        for start, end in ((DAYS[0], DAYS[1]), (DAYS[2], DAYS[3]))
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for _ in pool.map(download_candles, jobs):
            pass
    for sym, period, start, end in jobs:
        rows = read(bars_path(period, sym, start, end))["response"]
        good = {
            pd.Timestamp(r["time"])
            for r in rows
            if float(r["open"]) > 0 and float(r["close"]) > 0
        }
        for day in CAL.sessions_in_range(start, end):
            expected = pd.date_range(
                CAL.session_open(day),
                CAL.session_close(day),
                freq="min",
                inclusive="left",
            )
            if not set(expected).issubset(good):
                single_day = str(day.date())
                download_candles((sym, period, single_day, single_day))
    # Preserve the existing Longbridge ex-date records or obtain the same source for new names.
    actions = {}
    for sym in symbols:
        p = (
            HERE.parent
            / "30_supertrend_frama_chandelier/resources/market/actions"
            / f"{sym}.json"
        )
        cached = read(p) if p.exists() else {}
        if cached.get("schema") == 3 and cached.get("asof", "") >= DAYS[-1]:
            actions[sym] = {d: e for d, e in cached["events"].items() if d in DAYS}
            continue
        rows = fetch(
            ROOT / "inputs/corporate" / f"{sym}.json",
            ["corp-action", sym, "--all", "--lang", "en"],
        )["items"]
        actions[sym] = corporate_events(sym, rows, DAYS)
    atomic_json(ROOT / "inputs/actions.json", actions)
    print(f"Downloaded four regular sessions of minute data for {len(symbols)} stocks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("history", "minutes"))
    args = parser.parse_args()
    (history if args.task == "history" else minute_history)()
