"""Freeze both research candidate lists from one pre-open, quote-only observation."""

from datetime import datetime, time, timezone
import fcntl
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from investing_algorithm_framework import LongbridgeClient, LongbridgeOAuthCredential
from investing_algorithm_framework.infrastructure.longbridge import NEW_YORK
from screener import (
    SelectionRules,
    SelectionSource,
    atomic_json,
    screen,
    session_window,
)


ROOT = Path(__file__).parent / "resources/research/correlation-timing"


def capture():
    load_dotenv(Path(__file__).with_name(".env"))
    now = datetime.now(timezone.utc)
    client = LongbridgeClient(LongbridgeOAuthCredential(), quote_transport="http")
    try:
        target, _, upcoming = session_window(client, now)
        opening = datetime.combine(target, time(9, 30), NEW_YORK)
        if now.astimezone(NEW_YORK).date() != target or now >= opening:
            raise ValueError("Freeze the lists before the target regular session opens")
        directory = ROOT / "forward" / str(target)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "capture.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = directory / "frozen.json"
            if path.exists():
                print(path)
                return
            source = SelectionSource(client, directory)
            rules = SelectionRules(base_size=200, top_n=60)
            symbols = source.base_pool(target, rules.base_size)
            blocked = source.blocked_earnings(symbols, upcoming)
            source.blocked_earnings = lambda names, dates: blocked
            source.signal_candlesticks = lru_cache(maxsize=None)(
                source.signal_candlesticks
            )
            early = screen(source, rules, now)
            late = screen(source, rules, now, candidate_correlation=False)
            finished = datetime.now(timezone.utc)
            if finished >= opening:
                raise ValueError("Capture did not finish before the regular session")
            atomic_json(
                path,
                {
                    "capture_completed_utc": finished.isoformat(),
                    "earnings_blocked": sorted(blocked),
                    "calendar_window": [str(d) for d in upcoming],
                    "candidate_gate": early,
                    "order_gate_candidates": late,
                    "order_gate_max_correlation": rules.max_correlation,
                },
            )
            print(
                f"{path}  candidates {len(early['selected'])}/{len(late['selected'])}",
                flush=True,
            )
    finally:
        client.close()


if __name__ == "__main__":
    capture()
