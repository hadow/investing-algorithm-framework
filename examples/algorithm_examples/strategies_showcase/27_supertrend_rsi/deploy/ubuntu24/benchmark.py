"""离线模拟候选账户循环与日线选股，测量本机资源，不连接券商或提交订单。"""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

# 在导入数值库前限制线程；与部署服务使用相同设置。
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "POLARS_MAX_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import psutil

EXAMPLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXAMPLE))


def main():
    """记录导入、跨轮账户应用的RSS及耗时；全部外部传输使用内存桩。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--workload", choices=("portfolio", "selection"), default="portfolio")
    parser.add_argument("--candidates", type=int, help="Defaults to current SelectionRules.top_n")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.02):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.monotonic()
    import numpy as np
    import pandas as pd
    from longbridge import openapi as api
    from investing_algorithm_framework import LongbridgeClient, LongbridgeOAuthCredential
    from investing_algorithm_framework.infrastructure import longbridge
    from investing_algorithm_framework.infrastructure.longbridge import NEW_YORK, runtime
    import portfolio_app as runner
    import screener
    from screener import SelectionRules, SELECTION_MODEL

    candidates = args.candidates if args.candidates is not None else SelectionRules.top_n
    if not 1 <= candidates <= SelectionRules.top_n:
        parser.error("candidates must be between 1 and the formal candidate limit")
    base_size = 900  # 合成 500 只大盘股和 400 只中盘股，用于测量完整并集。
    base = [f"B{i:03d}.US" for i in range(base_size)]

    imported_rss = process.memory_info().rss
    now = datetime.now(timezone.utc)
    local_day = now.astimezone(NEW_YORK).date()
    market_now = now.replace(second=0, microsecond=0) - timedelta(seconds=15)

    class MarketClock(datetime):
        @classmethod
        def now(cls, tz):
            return market_now.astimezone(tz)

    client = LongbridgeClient(LongbridgeOAuthCredential("offline"), trading_enabled=False)
    client.quote, client.trade = Mock(), Mock()
    days = [local_day - timedelta(days=i) for i in range(16)]
    client.quote.trading_days.return_value = NS(
        trading_days=[day for day in days if day.weekday() < 5],
        half_trading_days=[],
    )
    last_bars = {
        minutes: longbridge.latest_completed_us_bar(client, name, market_now)
        for minutes, name in ((1, "1m"), (60, "1h"))
    }
    client.quote.history_candlesticks_by_date.return_value = []
    client.entry_blockers = ["offline resource benchmark"]
    client.quote.quote.return_value = [NS(
        timestamp=now, last_done=Decimal("100"), volume=100,
        trade_status=api.TradeStatus.Normal,
    )]
    client.quote.depth.return_value = NS(
        bids=[NS(price=Decimal("99.99"))], asks=[NS(price=Decimal("100"))]
    )

    def initial(symbol, period, *args):
        minutes = 1 if period == api.Period.Min_1 else 60
        last_bar = last_bars[minutes]
        return [NS(
            timestamp=last_bar - timedelta(minutes=minutes * (999 - i)),
            open=100 + np.sin(i / 15), high=102 + np.sin(i / 15),
            low=98 + np.sin(i / 15), close=100 + np.sin(i / 15), volume=1000,
        ) for i in range(1000)]

    client.quote.subscribe_candlesticks.side_effect = initial
    client.trade.account_balance.return_value = [NS(cash_infos=[
        NS(currency="USD", available_cash=Decimal("25000"))
    ])]
    client.trade.stock_positions.return_value = NS(channels=[])
    rss, elapsed = [], []
    app_cache, app_reuses = {}, 0
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            snapshot = {
                "schema_version": 1, "selection_model": SELECTION_MODEL,
                "generated_at": now.isoformat(),
                "trading_date": local_day.isoformat(),
                "as_of_date": (local_day - timedelta(days=1)).isoformat(),
                "rules": asdict(SelectionRules(top_n=candidates)), "base_pool": base,
                "selected": [{"symbol": f"B{i:03d}.US"} for i in range(candidates)],
            }
            # 同一行情时刻重复跑账户轮次，交易日历、订阅和成交均使用离线数据。
            universe = path / "universe.json"
            runner.atomic_json(universe, snapshot)
            if args.workload == "portfolio":
                with patch.object(longbridge, "datetime", MarketClock), \
                        patch.object(runtime, "datetime", MarketClock):
                    for _ in range(args.iterations):
                        tick = time.monotonic()
                        result = runner.run_once(
                            client, path / "account", universe, app_cache=app_cache,
                        )
                        assert len(result["strategies"]) == candidates
                        if elapsed:
                            assert all(
                                state["reason"] == "bar_already_processed"
                                for state in result["strategies"].values()
                            )
                        app_reuses += result["app_reused"]
                        elapsed.append(time.monotonic() - tick)
                        rss.append(round(process.memory_info().rss / 2**20, 1))
            else:
                dates = pd.bdate_range(end=local_day - timedelta(days=1), periods=300, tz=NEW_YORK)
                dates += pd.Timedelta(hours=9, minutes=30)
                histories = {}
                for i in range(base_size + 1):
                    close = 100 + np.arange(300) * (0.05 if i == base_size else 0.12 + i / 2000)
                    histories["SPY.US" if i == base_size else f"B{i:03d}.US"] = pd.DataFrame({
                        "Open": close, "High": close + 1, "Low": close - 1,
                        "Close": close, "Volume": 1_000_000, "DollarVolume": 100_000_000,
                    }, index=dates)
                source = Mock()
                source.client = client
                source.blocked_earnings.return_value = set()
                source.daily.side_effect = lambda symbol, *a: histories[symbol]
                source.signal_candlesticks.side_effect = lambda symbol, *a, **k: initial(
                    symbol, api.Period.Min_60
                )[-300:]
                # 压测允许相关候选，确保读取足额候选的小时缓存；不是收益实验。
                rules = SelectionRules(top_n=candidates, max_correlation=1)
                with patch.object(screener, "session_window", return_value=(
                    local_day, dates[-1].date(), [local_day]
                )), patch.object(screener, "combined_membership", return_value=(
                    base, {"sp500_count": 500, "ijh_equity_count": 400}
                )):
                    for _ in range(args.iterations):
                        tick = time.monotonic()
                        result = screener.screen(source, rules, now)
                        assert len(result["selected"]) == candidates, {
                            "selected_count": len(result["selected"]),
                            "errors": result["errors"],
                            "rejected_reasons": [row["reasons"] for row in result["rejected"][:5]],
                        }
                        elapsed.append(time.monotonic() - tick)
                        rss.append(round(process.memory_info().rss / 2**20, 1))
            assert not client.trade.submit_order.called
            assert not client.trade.cancel_order.called
    finally:
        client.close()
        stopped.set()
        sampler.join()
    print(json.dumps({
        "platform": platform.platform(), "architecture": platform.machine(),
        "python": platform.python_version(), "logical_cpus": psutil.cpu_count(),
        "workload": args.workload,
        "candidate_count": candidates, "base_size": base_size,
        "physical_memory_mib": round(psutil.virtual_memory().total / 2**20),
        "import_rss_mib": round(imported_rss / 2**20, 1),
        "peak_rss_mib": round(max(peak[0] / 2**20, max(rss)), 1), "iteration_rss_mib": rss,
        "iteration_seconds": [round(value, 3) for value in elapsed],
        "mean_steady_iteration_seconds": (
            round(sum(elapsed[1:]) / len(elapsed[1:]), 3) if len(elapsed) > 1 else None
        ),
        "max_iteration_seconds": round(max(elapsed), 3),
        "mean_iteration_seconds": round(sum(elapsed) / len(elapsed), 3),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "broker_order_mutations": 0,
        "quote_subscriptions": client.quote.subscribe.call_count,
        "candle_subscriptions": client.quote.subscribe_candlesticks.call_count,
        "history_backfills": client.quote.history_candlesticks_by_date.call_count,
        "app_reuses": app_reuses,
        "limitations": "Synthetic workload; excludes real SDK network buffers, CLI, OS, open positions and long-duration load",
    }, indent=2))


if __name__ == "__main__":
    main()
