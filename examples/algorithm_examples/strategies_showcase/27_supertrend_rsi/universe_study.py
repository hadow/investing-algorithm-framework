"""用相同的小时信号和五仓账本比较扩池与同小时买单优先级。可用 --fetch 补行情。"""

import argparse
from collections import Counter
from datetime import datetime
from functools import lru_cache
from types import SimpleNamespace

import pandas as pd
from correlation_history import (
    CAL,
    DAYS,
    HERE,
    HOUR_START,
    ROOT,
    RULES,
    SELECTION,
    bars_path,
    corporate_events,
    fetch,
    load_daily,
    previous,
    read,
)
from correlation_prepare import calendar_estimate, hourly_inputs
from correlation_replay import market_tape, run
from screener import atomic_json, screen_universe
from settings import MAX_POSITIONS, TOTAL_RISK_FRACTION
from universe_study_inputs import (
    DAY,
    LARGE,
    STUDY,
    history,
    mid_daily,
    mid_earnings,
    midcap_names,
)


class Archive:
    """向正式选股器提供当时可知的日线和盘前小时线，所有新增读取均使用独立目录。"""

    def __init__(self, download):
        self.download = download
        self.mid = set(midcap_names())
        self.read_errors = []
        self.minute_repairs = []
        self.hourly = lru_cache(maxsize=200)(self._read_hourly)

    def daily(self, symbol, as_of, count):
        path = SELECTION / "daily" / str(as_of) / f"{symbol}.csv"
        if path.is_file():
            return load_daily(as_of, symbol).tail(count)
        if symbol in self.mid and str(as_of) == str(previous(DAY)):
            try:
                return mid_daily(symbol)
            except FileNotFoundError:
                self.read_errors.append(f"Missing mid-cap input: {symbol}")
                raise
        raise ValueError("missing_archived_daily")

    def _read_hourly(self, symbol):
        try:
            rows = history(symbol, "1h", HOUR_START, DAY, download=self.download)
        except Exception as exc:
            self.read_errors.append(f"Hourly request failed: {symbol} {exc}")
            raise
        if not rows:
            raise ValueError(f"Empty hourly history: {symbol}")
        frame = pd.DataFrame(rows).rename(columns=str.capitalize)
        frame.index = pd.to_datetime(frame.pop("Time"), utc=True).dt.as_unit("ns")
        return frame.astype(float)

    def signal_candlesticks(self, symbol, time_frame, now, count):
        frame = self.hourly(symbol)
        # now 固定为开盘前一秒；任何当日 K 线都不参与选股或预算估算。
        prefix = frame.loc[frame.index < now].tail(count)
        return [
            SimpleNamespace(
                timestamp=r.Index,
                open=r.Open,
                high=r.High,
                low=r.Low,
                close=r.Close,
                volume=r.Volume,
            )
            for r in prefix.itertuples()
        ]

    def minutes(self, symbol, day):
        pair = DAYS[:2] if day in DAYS[:2] else DAYS[2:]
        old = bars_path("1m", symbol, *pair)
        if old.is_file():
            return [row for row in read(old)["response"] if row["time"][:10] == day]
        rows = history(symbol, "1m", day, day, download=self.download)

        def usable(row):
            return float(row["open"]) > 0 and float(row["close"]) > 0

        expected = CAL.session_minutes(day)
        valid = {pd.Timestamp(row["time"]) for row in rows if usable(row)}
        missing = set(expected) - valid
        if missing:
            # 只用另一查询区间补原本不可成交的分钟，保留首次取得的有效价格。
            indexed = {row["time"]: row for row in rows}
            for row in history(symbol, "1m", *pair, download=self.download):
                if pd.Timestamp(row["time"]) in missing and usable(row):
                    indexed[row["time"]] = row
                    self.minute_repairs.append({"symbol": symbol, "time": row["time"]})
            rows = list(indexed.values())
        return rows

    def actions(self, symbols):
        events = read(ROOT / "inputs/actions.json")
        for s in set(symbols) - events.keys():
            path = (
                HERE.parent
                / f"30_supertrend_frama_chandelier/resources/market/actions/{s}.json"
            )
            cached = read(path) if path.exists() else {}
            if cached.get("schema") == 3 and cached.get("asof", "") >= DAY:
                events[s] = {d: v for d, v in cached["events"].items() if d in DAYS}
                continue
            path = STUDY / f"inputs/corporate/{s}.json"
            data = (
                fetch(path, ["corp-action", s, "--all", "--lang", "en"])
                if self.download
                else read(path)["response"]
            )
            events[s] = corporate_events(s, data["items"], DAYS)
        return events


def select(source, day, symbols, blocked):
    """沿用线上先排名、去相关、检查小时 ATR 可买股数的顺序，只提取研究字段。"""
    cutoff = CAL.session_open(day) - pd.Timedelta(seconds=1)
    snap = screen_universe(
        source, RULES, cutoff, cutoff.date(), previous(day), symbols, blocked
    )
    if source.read_errors:
        raise RuntimeError(source.read_errors[0])
    return {
        "base_size": len(symbols),
        "daily_eligible": snap["daily_eligible_count"],
        "selected": [
            {
                k: r[k]
                for k in (
                    "symbol",
                    "rank",
                    "selection_score",
                    "estimated_shares",
                    "stop_distance_estimate",
                    "dollar_volume_20",
                )
            }
            for r in snap["selected"]
        ],
        "errors": snap["errors"],
        "rejections": dict(Counter(r["reasons"][0] for r in snap["rejected"])),
    }


def prepare_pools(source, include_mid):
    observed = read(ROOT / "inputs/observed.json")
    broad = read(LARGE)
    if broad["trading_date"] != DAY or datetime.fromisoformat(
        broad["generated_at"]
    ) >= CAL.session_open(DAY):
        raise ValueError("Broad membership must predate the studied session")
    broad_names = broad["base_pool"]
    output = {"A": {}, "B": {}}
    if include_mid:
        output["C"] = {}
    for day in DAYS:
        base = [r["symbol"] for r in read(SELECTION / f"base/{day}.json")["stocks"]]
        if len(base) != RULES.base_size or len(set(base)) != len(base):
            raise ValueError(f"Incomplete original base on {day}")
        # 优先使用当日留下的财报标记和已发布入选记录；其余仅能用事后日历重建。
        known = dict(observed["known_reports"].get(day, {}).get("earnings_pass", {}))
        known.update(dict.fromkeys(observed["production"][day]["selected"], True))
        blocked = {s for s in calendar_estimate(day) if s not in known}
        blocked.update(s for s, passed in known.items() if not passed)
        output["A"][day] = select(source, day, base, blocked)
        if day != DAY:
            continue
        if (
            broad_names[: len(base)] != base
            or len(set(broad_names)) != 500
            or not set(broad["data_exclusions"]) <= known.keys()
        ):
            raise ValueError("Broad base or archived earnings coverage does not match")
        # 当天全池快照也归档了财报排除，前200只的标记以该策略自身归档为准。
        outside = {
            r["symbol"]
            for r in broad["rejected"]
            if r["symbol"] not in known
            and "earnings_within_two_sessions" in r["reasons"]
        }
        blocked.update(outside)
        output["B"][day] = select(source, day, broad_names, blocked)
        if include_mid:
            names = list(dict.fromkeys(broad_names + midcap_names()))
            reports = mid_earnings(include_disclosures=False)
            estimated = mid_earnings()
            main = select(source, day, names, blocked | estimated)
            sensitivity = select(source, day, names, blocked | reports)
            main["midcap_earnings_from_disclosures"] = sorted(estimated - reports)
            main["report_only_daily_eligible"] = sensitivity["daily_eligible"]
            output["C"][day] = main
            if sensitivity["selected"] != main["selected"]:
                output["C_report_only"] = {day: sensitivity}
    return output


def ordered(rows, policy):
    """只在选中集合内安排同一根小时线的买单，流动性相同时回到盘前动量名次。"""
    if policy == "liquidity":
        rows = sorted(rows, key=lambda r: (-r["dollar_volume_20"], r["rank"]))
    return [r["symbol"] for r in rows]


def attribute(result):
    """区分同小时新单竞争与早先仓位占满，并保留每次被挡信号的竞争者。"""
    counts = Counter()
    for event in result["entry_decisions"]:
        if event["reason"] == "positions_full":
            same = [
                s for s, p in event["occupants"].items() if p["entry"] == event["time"]
            ]
            event["same_minute_occupants"] = same
            counts["with_same_minute_order" if same else "all_slots_preoccupied"] += 1
    result["metrics"]["full_reasons"] = dict(counts)
    return result


def study_tape(source, groups):
    """给每一组使用同一个信号时间轴，旧仓掉出新名单后仍保留行情及退出。"""
    # 前三日共用 A，四日账本保留继承仓位；同日空仓对照只使用 18 日可用资金。
    saved = read(ROOT / "inputs/prepared.json")["signals"]
    historic = set(read(ROOT / "inputs/observed.json")["daily"][DAY]["ranked"])
    # 旧信号表按每日所有原基础池中日线合格的股票计算，覆盖可能在中途掉池的旧持仓。
    seen = {p.name.split("_")[0] for p in (ROOT / "inputs/1h").glob("*.json")}
    if not {s for d in groups["A"].values() for s in d} <= seen or not historic <= seen:
        raise ValueError("Missing original hourly signal coverage")
    on_last = {s for dates in groups.values() for s in dates[DAY]}
    _, extra = hourly_inputs(
        sorted(on_last - seen), days=(DAY,), load_frame=source.hourly
    )
    signals = {d: {t: dict(rows) for t, rows in saved[d].items()} for d in DAYS}
    for stamp, rows in extra[DAY].items():
        signals[DAY].setdefault(stamp, {}).update(rows)
    # 分钟行情只影响有入场机会的标的及其后续管理，所有候选都已计算小时信号。
    symbols = set()
    for day in DAYS:
        eligible = set(groups["A"][day]) if day != DAY else on_last
        symbols.update(
            s
            for rows in signals[day].values()
            for s, sig in rows.items()
            if s in eligible and sig["buy"]
        )
    prep = {
        "signals": signals,
        "minute_symbols": sorted(symbols),
        "correlations": {d: {} for d in DAYS},
    }
    tape, actions, quality = market_tape(
        prep, minute_rows=source.minutes, actions=source.actions(symbols)
    )
    return prep, tape, actions, quality


def reserved_at(result, stamp):
    return {
        r["symbol"]
        for r in result["entry_decisions"]
        if r["time"] == stamp and r["reason"] == "order_reserved"
    }


def study(download=False, include_mid=True):
    """冻结选股输入后比较各组独立账本，并逐次重跑同小时争夺最后仓位的信号。"""
    source = Archive(download)
    pools = prepare_pools(source, include_mid)
    groups = {
        g: {d: ordered(v["selected"], "rank") for d, v in dates.items()}
        for g, dates in pools.items()
    }
    prep, tape, actions, quality = study_tape(source, groups)
    accounts = {}
    for group in pools:
        for policy in ("rank", "liquidity"):
            last = ordered(pools[group][DAY]["selected"], policy)
            common_prior = {d: groups["A"][d] for d in DAYS[:-1]}
            scenarios = {"same_day": {DAY: last}, "carry": {**common_prior, DAY: last}}
            if group == "A":
                scenarios["four_day"] = {
                    d: ordered(pools["A"][d]["selected"], policy) for d in DAYS
                }
            for scenario, active in scenarios.items():
                # 开仓日 09:30 仅承接前收盘退出信号；新开仓信号仍遵守正式时间窗口。
                for model, cost, delay in (
                    ("base", 1, False),
                    ("double_cost", 2, False),
                    ("one_minute_delay", 1, True),
                ):
                    key = f"{scenario}/{group}/{policy}/{model}"
                    accounts[key] = attribute(
                        run(
                            prep,
                            tape,
                            actions,
                            active,
                            "candidate",
                            cost,
                            delay,
                            min_shares=1,
                            audit_slots=True,
                        )
                    )
    # 逐次将同一时点被挡的股票提到首位，每次都从原起点重跑，量化入替对账户的影响。
    switches = []
    for group, dates in groups.items():
        for scenario in ("same_day", "carry"):
            key = f"{scenario}/{group}/rank/base"
            baseline = accounts[key]
            active = {DAY: dates[DAY]}
            if scenario == "carry":
                active = {**{d: groups["A"][d] for d in DAYS[:-1]}, **active}
            for event in baseline["entry_decisions"]:
                stamp, sym = event["time"], event["symbol"]
                if stamp[:10] != DAY or not event.get("same_minute_occupants"):
                    continue
                alternate = run(
                    prep,
                    tape,
                    actions,
                    active,
                    "candidate",
                    min_shares=1,
                    first_at={stamp: sym},
                )
                accepted = reserved_at(alternate, stamp)
                switches.append(
                    {
                        "account": key,
                        "time": stamp,
                        "promoted": sym,
                        "reserved": sym in accepted,
                        "displaced": sorted(reserved_at(baseline, stamp) - accepted),
                        "ending_equity_delta_usd": alternate["metrics"]["net_usd"]
                        - baseline["metrics"]["net_usd"],
                    }
                )
    result = {
        "dates": DAYS,
        "comparison_day": DAY,
        "portfolio_rules": {
            "capital_usd": RULES.capital,
            "per_order_usd": RULES.order_budget,
            "positive_whole_shares": True,
            "max_positions": MAX_POSITIONS,
            "risk_fraction": RULES.risk_fraction,
            "total_risk_fraction": TOTAL_RISK_FRACTION,
            "top_n": RULES.top_n,
            "stop_atr_multiple": RULES.stop_atr_multiple,
            "candidate_max_correlation": RULES.max_correlation,
        },
        "pools": pools,
        "accounts": accounts,
        "slot_replacements": switches,
        "minute_quality": quality,
        "minute_symbols": prep["minute_symbols"],
        "recovered_minutes": source.minute_repairs,
        "midcap_source": "research_inputs/midcap-2026-11-18.json"
        if include_mid
        else None,
    }
    out = STUDY / ("abc.json" if include_mid else "ab.json")
    atomic_json(out, result)
    print(out, flush=True)
    for key, ledger in accounts.items():
        if key.endswith("/base"):
            m = ledger["metrics"]
            print(
                key,
                m["entries"],
                f"{m['net_pct']:+.3f}%",
                m["full_reasons"],
                flush=True,
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true", help="仅补采独立研究目录中缺少的行情"
    )
    parser.add_argument(
        "--large-only", action="store_true", help="先跑 A/B，无需中盘股输入"
    )
    args = parser.parse_args()
    study(args.fetch, not args.large_only)
