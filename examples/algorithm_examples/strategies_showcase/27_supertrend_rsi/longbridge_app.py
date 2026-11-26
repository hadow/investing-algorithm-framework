"""Supertrend + RSI 的 Longbridge 单标的入口，支持只读检查、模拟及实盘运行。

configure_app 注册多个入口共用的账户和行情适配器；build_app 再挂入一只股票
的策略实例。多股票选股池通过 portfolio_app 复用公共初始化，而非启动本入口。
行情由共享 SDK 客户端持续订阅，信号只读取已完成 K 线，账户版附加入场 ATR 风控。
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from dotenv import load_dotenv

from investing_algorithm_framework import (
    LongbridgeClient,
    LongbridgeOAuthCredential,
    LongbridgeOHLCVDataProvider,
    LongbridgeTickerDataProvider,
)
from investing_algorithm_framework.infrastructure.longbridge import us_symbol
from investing_algorithm_framework.infrastructure.longbridge.runner import (
    configure_app as configure_longbridge_app, run_iterations, single_runner,
)
from investing_algorithm_framework.infrastructure.longbridge.execution import (
    LongbridgeLimitExecutor,
)
from supertrend_rsi import SupertrendRSIStrategy
from settings import CAPITAL, ORDER_BUDGET, RISK_FRACTION
from runtime import RuntimeControls, atomic_json
from operations import (
    preflight,
    runtime_report,
)


def configure_app(client, resource_directory, capital=CAPITAL):
    """沿用示例原入口，实际账户组装由两个策略共用的基础层负责。"""
    return configure_longbridge_app(client, resource_directory, capital)


def build_app(
    client,
    resource_directory,
    *,
    symbol="SPY.US/USD",
    time_frame="1h",
    capital=CAPITAL,
    order_budget=ORDER_BUDGET,
    atr_period=10,
    factor=3.0,
    rsi_period=14,
    rsi_exit=72,
    warmup_window=300,
    stop_atr_multiple=2.0,
    risk_fraction=RISK_FRACTION,
    min_entry_shares=None,  # 兼容旧调用，已停用。
    limit_ttl=120,
):
    """构建单标的账户应用，把信号逻辑接到持续行情与整股执行链路。

    先校验美股代码及单笔预算，再复用 configure_app，并注册一份指定周期的
    SupertrendRSIStrategy。默认采用入场2×ATR距离、0.5%单笔风险比例，
    LongbridgeLimitExecutor 根据实时盘口、可用现金及风险距离决定整股数量。
    返回可运行的 App，调用方可用模拟传输验证完整流程；本函数本身不下单。
    多标的名额与组合风险上限由 portfolio_app 的专用执行器负责。
    """
    us_symbol(symbol)
    if not 0 < order_budget <= capital * 0.98:
        raise ValueError("order_budget must be <=98% of capital")
    app = configure_app(client, resource_directory, capital)
    app.add_strategy(
        SupertrendRSIStrategy(
            symbol=symbol,
            time_frame=time_frame,
            atr_period=atr_period,
            factor=factor,
            rsi_period=rsi_period,
            rsi_exit=rsi_exit,
            warmup_window=warmup_window,
            stop_atr_multiple=stop_atr_multiple,
            risk_fraction=risk_fraction,
            order_budget=order_budget,
            executor=LongbridgeLimitExecutor(client),
            runtime=RuntimeControls(client, resource_directory, limit_ttl),
        )
    )
    return app


def main():
    """解析单标的运行参数，先校验配置，再按模式选择检查或账户事件循环。

    读取本目录 .env 的 OAuth 配置，默认 check：只读账户、报价及已完成 K 线，
    输出指标和原始信号；可导出 CSV 供 backtest.py 离线研究，交易权限保持关闭。
    paper/live 显式启用交易，并用模式、标的、周期隔离资源目录；iterations
    默认只运行一轮，0交给框架持续调度。paper 依赖模拟账户授权，并非把实盘
    token 自动转为模拟账户。所有路径最终释放共享订阅，避免遗留行情资源。
    """
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "paper", "live"], default="check")
    parser.add_argument(
        "--account",
        choices=["paper", "live"],
        help="Account for check; defaults to paper",
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--symbol", default="SPY.US/USD")
    parser.add_argument(
        "--time-frame", choices=["1m", "5m", "15m", "30m", "1h"], default="1h"
    )
    parser.add_argument("--capital", type=float, default=CAPITAL)
    parser.add_argument("--order-budget", type=float, default=ORDER_BUDGET)
    # 兼容已有启动命令。旧的测试参数已停用，不参与股数计算。
    parser.add_argument("--min-entry-shares", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--atr-period", type=int, default=10)
    parser.add_argument("--factor", type=float, default=3.0)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-exit", type=float, default=72)
    parser.add_argument("--warmup-window", type=int, default=300)
    parser.add_argument("--stop-atr-multiple", type=float, default=2.0)
    parser.add_argument("--risk-percent", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=1, help="0 runs continuously")
    parser.add_argument("--limit-ttl", type=float, default=120)
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Export checked OHLCV for offline research (check only)",
    )
    args = parser.parse_args()
    parameters = {
        k: v
        for k, v in vars(args).items()
        if k
        not in (
            "mode",
            "account",
            "directory",
            "iterations",
            "risk_percent",
            "min_entry_shares",
            "export_csv",
        )
    }
    parameters["risk_fraction"] = args.risk_percent / 100
    # 配置错误应在 OAuth 授权或读取账户之前暴露，避免无效参数触发外部请求。
    try:
        if args.iterations < 0:
            raise ValueError("iterations must be nonnegative")
        if args.export_csv and args.mode != "check":
            raise ValueError("--export-csv requires --mode check")
        if not math.isfinite(args.capital) or args.capital <= 0:
            raise ValueError("capital must be positive and finite")
        if not 0 < args.order_budget <= args.capital * 0.98:
            raise ValueError("order_budget must be <=98% of capital")
        strategy = SupertrendRSIStrategy(
            **{
                k: v
                for k, v in parameters.items()
                if k not in ("capital", "limit_ttl")
            }
        )
        us_symbol(args.symbol)
        RuntimeControls(None, ".", args.limit_ttl)
        if args.mode != "check" and args.account and args.account != args.mode:
            raise ValueError("--account must match the execution mode")
    except ValueError as exc:
        parser.error(str(exc))
    account = (args.account or "paper") if args.mode == "check" else args.mode
    directory = args.directory or (
        Path(__file__).parent
        / "resources"
        / account
        / us_symbol(args.symbol)
        / args.time_frame
    )
    client = LongbridgeClient(
        LongbridgeOAuthCredential(),
        paper=account == "paper",
        trading_enabled=False,
    )
    try:
        with single_runner(directory):
            app = build_app(client, directory, **parameters)
            checks = preflight(
                client,
                app,
                [us_symbol(args.symbol)],
                args.capital,
                time_frame=args.time_frame,
                minimum_bars=max(args.atr_period + 1, args.rsi_period + 2),
            )
            atomic_json(directory / "preflight.json", checks)
            if args.mode != "check":
                print(json.dumps(checks, indent=2), flush=True)
                client.entry_blockers = checks["entry_blockers"]
                client.trading_enabled = True

                def tick():
                    client.__dict__["execution_issues"] = {}
                    current = build_app(client, directory, **parameters)
                    report = current.run(number_of_iterations=1)
                    return runtime_report(client, current, directory, report)

                run_iterations(tick, args.iterations, directory, account)
                return
            ticker = LongbridgeTickerDataProvider(
                client,
                symbol=args.symbol,
                market="LONGBRIDGE",
            ).get_data()
            candles = LongbridgeOHLCVDataProvider(
                client,
                symbol=args.symbol,
                time_frame=args.time_frame,
                warmup_window=args.warmup_window,
                pandas=True,
            ).get_data(date=datetime.now(timezone.utc))
            if len(candles) < max(args.atr_period + 1, args.rsi_period + 2):
                raise RuntimeError("Insufficient completed candles")
            if args.export_csv:
                args.export_csv.parent.mkdir(parents=True, exist_ok=True)
                candles.to_csv(args.export_csv, index_label="Datetime")
            data = {strategy.data_key: candles}
            row = strategy.prepare(data).iloc[-1]
            print(
                json.dumps(
                    {
                        **checks,
                        "symbol": args.symbol,
                        "time_frame": args.time_frame,
                        "completed_bars": len(candles),
                        "bar": row.name.isoformat(),
                        "atr": float(row.ATR),
                        "rsi": float(row.RSI),
                        "supertrend": float(row.Supertrend),
                        "direction": int(row.Direction),
                        "signals": [
                            s.side.value for s in strategy.generate_signals(None, data)
                        ],
                        "order_book": bool(ticker["bid"] and ticker["ask"]),
                        "trading_enabled": False,
                    },
                    indent=2,
                )
            )
            return 0 if checks["ready"] else 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
