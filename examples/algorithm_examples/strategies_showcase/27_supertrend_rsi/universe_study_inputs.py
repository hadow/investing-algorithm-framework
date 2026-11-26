"""11月18日扩池研究的只读数据。收集阶段按请求缓存，重放阶段仅读取归档。"""

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from hashlib import sha256
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import exchange_calendars as xc
import pandas as pd
from correlation_history import DAYS, HERE, bars_path, fetch, read
from screener import atomic_json
from supertrend_universe import ISSUER_CSV, SP500_QUERY, issuer_holdings, sp500_members

from investing_algorithm_framework.infrastructure.longbridge.selection import (
    earnings_symbols,
)

STUDY = HERE / "resources/research/universe-slots"
DAY = DAYS[-1]
AS_OF = "2026-11-17"
DAILY_START = "2025-09-10"  # 与当时 SPY 归档的 300 根日线起点一致。
MIDCAP = HERE / "research_inputs/midcap-2026-11-18.json"
LARGE = (
    HERE.parent
    / "28_chandelier_flip_radar/resources/selection/snapshots/20261118T100247Z.json"
)


def history(symbol, period, start, end, *, adjust="none", download=False):
    """同一证券优先使用当时账户研究的原始归档，新增历史数据保存在独立目录。"""
    old = bars_path(period, symbol, start, end)
    if adjust == "none" and old.is_file():
        return read(old)["response"]
    directory = period if adjust == "none" else f"{period}-{adjust}"
    path = STUDY / "inputs" / directory / f"{symbol}_{start}_{end}.json"
    args = [
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
        adjust,
    ]
    if download:
        return fetch(path, args)
    return read(path)["response"]


def midcap_names():
    data = read(MIDCAP)
    names = data["symbols"]
    if data["trading_date"] != DAY or len(names) != 400 or len(set(names)) != 400:
        raise ValueError("Invalid dated mid-cap membership")
    return names


def mid_daily(symbol, download=False):
    """取前复权 OHLCV 及未复权成交额，与正式选股器的 CSV 口径对齐。"""
    frames = []
    for adjust in ("none", "forward"):
        rows = history(
            symbol, "day", DAILY_START, AS_OF, adjust=adjust, download=download
        )
        if not rows:
            raise ValueError(f"Empty {adjust} daily history: {symbol}")
        frame = pd.DataFrame(rows)
        frame.index = pd.to_datetime(frame.pop("time"), utc=True).dt.as_unit("ns")
        frames.append(frame.astype(float))
    raw, adjusted = frames
    daily = adjusted[["open", "high", "low", "close", "volume"]].rename(
        columns=str.capitalize
    )
    daily["DollarVolume"] = raw.turnover.reindex(daily.index)
    return daily.tail(300)


def mid_earnings(download=False, *, include_disclosures=True):
    """补采当周财报日历，保留原始响应，以便核对事后修订可能造成的影响。"""
    dates = [date(2026, 11, d) for d in (18, 19, 20)]
    kinds = ("report", "financial") if include_disclosures else ("report",)
    return earnings_for(
        midcap_names(), dates, STUDY / "inputs/earnings", download, event_types=kinds
    )


def earnings_for(names, dates, directory, download, *, event_types=("report",)):
    """逐批记录原始日历，分页或截断响应中止采样，避免把未知财报误判成可买。"""
    blocked = set()
    for n in range(0, len(names), 10):
        batch = names[n : n + 10]
        path = directory / ("_".join(batch) + ".json")
        args = [
            "finance-calendar",
            "report",
            "--market",
            "US",
            "--start",
            str(dates[0]),
            "--end",
            str(dates[-1]),
            "--count",
            "1000",
        ]
        for symbol in batch:
            args.extend(["--symbol", symbol])
        payload = fetch(path, args) if download else read(path)["response"]
        if sum(len(g["infos"]) for g in payload.get("list", [])) >= 1000:
            raise ValueError(f"Incomplete earnings batch: {batch}")
        blocked.update(earnings_symbols(payload, set(dates), event_types=event_types))
    return blocked


def collect_daily():
    """并发上限三，只补归档中缺失的输入。网络异常可用同一命令继续。"""

    def collect(symbol):
        try:
            mid_daily(symbol, download=True)
        except ValueError as exc:
            if not str(exc).startswith("Empty "):
                raise
            print(exc, flush=True)

    with ThreadPoolExecutor(max_workers=3) as workers:
        for n, _ in enumerate(workers.map(collect, midcap_names()), 1):
            if n % 50 == 0:
                print(f"Historical daily coverage checked: {n}/400", flush=True)
    print(
        "Retrospective earnings exclusions:",
        sorted(mid_earnings(download=True)),
        flush=True,
    )


def freeze():
    """盘前冻结会随时间消失的成分、成交额顺序与财报信息，供后续样本外比较。"""
    ny = ZoneInfo("America/New_York")
    now = datetime.now(timezone.utc)
    target = now.astimezone(ny).date()
    directory = STUDY / "forward" / str(target)
    cal = xc.get_calendar("XNYS")
    if not cal.is_session(target):
        raise ValueError("Today is not a US market session")
    opening = cal.session_open(target)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "frozen-inputs.json"
    with (directory / "capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            print(path)
            return
        if datetime.now(timezone.utc) >= opening:
            raise ValueError("A new capture must predate the open")
        large = sp500_members(fetch(directory / "constituents.json", SP500_QUERY))
        with urlopen(ISSUER_CSV, timeout=25) as response:
            content = response.read()
        holdings_date, mid = issuer_holdings(content, target)
        names = list(dict.fromkeys(large + mid))
        dates = [s.date() for s in cal.sessions_window(target, 3)]
        blocked = earnings_for(sorted(names), dates, directory / "earnings", True)
        completed = datetime.now(timezone.utc)
        if completed >= opening:
            raise ValueError("Capture finished after the open")
        (directory / "ijh.csv").write_bytes(content)
        atomic_json(
            path,
            {
                "trading_date": str(target),
                "completed_utc": completed.isoformat(),
                "base_A": large[:200],
                "base_B": large,
                "base_C": names,
                "midcap_proxy": "IJH Equity",
                "holdings_date": str(holdings_date),
                "holdings_url": ISSUER_CSV,
                "holdings_sha256": sha256(content).hexdigest(),
                "earnings_window": list(map(str, dates)),
                "earnings_blocked": sorted(blocked),
            },
        )
        print(path, len(large), len(mid), len(blocked), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task", choices=["historical", "freeze"], default="historical", nargs="?"
    )
    if parser.parse_args().task == "freeze":
        freeze()
    else:
        collect_daily()
