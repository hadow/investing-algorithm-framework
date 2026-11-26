"""C 方案的当日基础池。长桥提供标普成分，基金发行人提供中盘股票代理名单。"""

import csv
import fcntl
import io
import json
import re
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from urllib.request import urlopen

from investing_algorithm_framework.domain.exceptions import ImproperlyConfigured
from investing_algorithm_framework.infrastructure.longbridge import us_symbol
from investing_algorithm_framework.infrastructure.longbridge.runtime import atomic_json

ISSUER_CSV = "https://www.ishares.com/us/products/239763/ishares-core-s-p-mid-cap-etf/latest-holdings.csv"
SP500_QUERY = ["constituent", ".SPX.US", "--limit", "550", "--sort", "turnover"]


def sp500_members(payload):
    """核对券商报告的总数，正常接纳 499 等实际成分数量，拒绝分页截断和重复。"""
    rows = payload["stocks"]
    symbols = [us_symbol(row["symbol"]) for row in rows]
    if (
        len(symbols) != int(payload["total"])
        or len(symbols) != len(set(symbols))
        or len(symbols) < 450
    ):
        raise ValueError("Incomplete Longbridge S&P 500 constituents")
    return symbols


def issuer_holdings(content, target):
    """解析 IJH 的完整股票明细，统一类别股代码，只接受目标交易日之前的近期持仓。"""
    rows = csv.reader(io.StringIO(content.decode("utf-8-sig")))
    holdings_date = None
    for row in rows:
        if row[:1] == ["Fund Holdings as of"] and len(row) > 1:
            # 日期格式固定为英文月份，避免依赖服务器的 LC_TIME。
            month, day, year = row[1].replace(",", "").split()
            months = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            holdings_date = date(int(year), months.index(month) + 1, int(day))
        if row[:2] == ["Ticker", "Name"]:
            header = row
            break
    else:
        raise ValueError("Missing IJH holdings table")
    if (
        holdings_date is None
        or not target - timedelta(days=7) <= holdings_date < target
    ):
        raise ValueError("Stale or future IJH holdings date")
    ticker = header.index("Ticker")
    asset = header.index("Asset Class")
    symbols = set()
    for row in rows:
        if len(row) <= asset or row[asset] != "Equity":
            continue
        name = row[ticker].strip().upper().replace(" ", ".")
        if not re.fullmatch(r"[A-Z0-9]+(?:[.-][A-Z0-9]+)*", name):
            raise ValueError(f"Invalid ticker in IJH equity holdings: {name!r}")
        symbols.add(us_symbol(name + ".US"))
    if len(symbols) < 350:
        raise ValueError("Incomplete IJH equity holdings")
    return holdings_date, sorted(symbols)


def checked_membership(record, target, as_of):
    """读写使用相同的日期、数量及重复代码检查，缺一组时整轮选股失败。"""
    large, mid = record["sp500"], record["ijh_equities"]
    holdings_date = date.fromisoformat(record["ijh_as_of"])
    if (
        record["trading_date"] != str(target)
        or not target - timedelta(days=7) <= holdings_date <= as_of
    ):
        raise ValueError("C universe membership dates do not match the session")
    if (
        not isinstance(large, list)
        or len(large) < 450
        or len(set(large)) != len(large)
        or len(large) != record["sp500_total"]
        or not isinstance(mid, list)
        or len(mid) < 350
        or len(set(mid)) != len(mid)
    ):
        raise ValueError("Incomplete C universe membership")
    if any(not isinstance(s, str) or us_symbol(s) != s for s in large + mid):
        raise ValueError("Invalid symbol in C universe membership")
    symbols = list(dict.fromkeys(large + mid))
    return symbols, {
        "sp500_count": len(large),
        "ijh_equity_count": len(mid),
        "overlap_count": len(large) + len(mid) - len(symbols),
        "ijh_as_of": record["ijh_as_of"],
        "observed_at": record["observed_at"],
        "ijh_sha256": record["ijh_sha256"],
    }


def combined_membership(source, target, as_of):
    """原子固定当日并集。缺失或损坏的缓存可用两个完整来源重建。"""
    path = source.base_directory / "base_c" / f"{target}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # 手动刷新与定时选股可能同时启动，同一交易日只固定一次完整的双方成分。
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            try:
                return checked_membership(json.loads(path.read_text()), target, as_of)
            except (ValueError, KeyError, TypeError, ImproperlyConfigured):
                pass
        large = sp500_members(source.run_cli(SP500_QUERY))
        with urlopen(ISSUER_CSV, timeout=25) as response:
            content = response.read()
        holdings_date, mid = issuer_holdings(content, target)
        record = {
            "trading_date": str(target),
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "sp500": large,
            "sp500_total": len(large),
            "ijh_equities": mid,
            "ijh_as_of": str(holdings_date),
            "ijh_url": ISSUER_CSV,
            "ijh_sha256": sha256(content).hexdigest(),
        }
        result = checked_membership(record, target, as_of)
        atomic_json(path, record)
        return result
