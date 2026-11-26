"""读取每日候选池，在同一份持久化 Longbridge 资金账户中运行多个策略实例。

选股由独立 screener 进程完成，本入口每轮合并候选股、已有持仓及未完成委托，
挂载单标的信号策略和共享组合执行器。持续模式每分钟重新读取资格及持仓状态，
期间 SDK 连接和行情缓存保持复用；候选池失效时继续维护已有风险的退出链路。
"""

import argparse
from datetime import date
from functools import partial
import json
from pathlib import Path

from dotenv import load_dotenv

from investing_algorithm_framework import LongbridgeClient, LongbridgeOAuthCredential
from investing_algorithm_framework.infrastructure.longbridge.runner import (
    retained_symbols, run_iterations, single_runner,
)
from longbridge_app import configure_app
from portfolio import PortfolioExecutor, SelectedStrategy
from screener import DIRECTORY, SELECTION_MODEL, SelectionRules, load_snapshot
from runtime import RuntimeControls, atomic_json
from operations import preflight, runtime_report
from settings import (
    CAPITAL,
    ORDER_BUDGET,
    MAX_POSITIONS,
    TOTAL_RISK_FRACTION,
)
from investing_algorithm_framework.infrastructure.longbridge import us_symbol


def build_portfolio_app(
    client,
    directory,
    snapshot=None,
    *,
    capital=CAPITAL,
    order_budget=ORDER_BUDGET,
    max_positions=MAX_POSITIONS,
    total_risk_fraction=TOTAL_RISK_FRACTION,
    min_entry_shares=None,  # 旧调用参数，股数由预算与风险决定。
    limit_ttl=120,
):
    """把当日开仓资格与本地风险敞口合并，构建共享账户的多策略 App。

    初始化存储后恢复持仓、委托和提交标记，为掉池敞口注册退出策略。候选的
    预算或信号设置不匹配时清空开仓资格；所有标的复用组合执行器，按排名注册。
    仅管理退出的股票使用 entry_date=None，稳定策略 ID 用于恢复成交及去重。
    返回 App、管理列表及开仓候选，调用方在这些配置稳定时可复用应用。
    """
    rules = SelectionRules(capital=capital, order_budget=order_budget)
    app = configure_app(client, directory, capital)
    app.initialize_config()
    app.initialize_storage()
    retained = retained_symbols(client, app, directory)
    selected = []
    entry_date = None
    if snapshot and snapshot.get("selection_model") == SELECTION_MODEL:
        matching_rules = all(
            snapshot["rules"].get(key) == getattr(rules, key)
            for key in ("base_size", "capital", "order_budget", "risk_fraction",
                        "stop_atr_multiple", "time_frame")
        )
        if matching_rules:
            selected = [us_symbol(row["symbol"]) for row in snapshot["selected"]]
            entry_date = date.fromisoformat(snapshot["trading_date"])
    executor = PortfolioExecutor(client, max_positions, total_risk_fraction)
    # 候选按排名先竞争额度；掉池保留项只生成退出，不因重新注册而恢复开仓资格。
    tracked = list(dict.fromkeys(selected + sorted(retained)))
    eligible = set(selected)
    for symbol in tracked:
        app.add_strategy(
            SelectedStrategy(
                symbol=symbol,
                time_frame=rules.time_frame,
                stop_atr_multiple=rules.stop_atr_multiple,
                risk_fraction=rules.risk_fraction,
                order_budget=order_budget,
                executor=executor,
                entry_date=entry_date if symbol in eligible else None,
                runtime=RuntimeControls(client, directory, limit_ttl),
            )
        )
    return app, tracked, selected


def _app_cache_key(client, directory, parameters):
    """账户环境、参数或身份文件变化后重新走构建校验，不能沿用旧应用绕过检查。"""
    binding = Path(directory) / "account.json"
    return (
        id(client), client.paper, str(Path(directory).resolve()),
        tuple((name, type(value), value) for name, value in sorted(parameters.items())),
        binding.read_bytes() if binding.exists() else None,
    )


def run_once(client, directory, universe_path, *, app_cache=None, **parameters):
    """执行一轮“读取资格 → 恢复账户 → 处理信号 → 清理订阅”，返回状态摘要。

    股票池缺失、损坏、过期或带失败标记时，将 snapshot 置空并记录原因；随后
    仍从数据库恢复持仓及委托。构建结果有 tracked 才运行一轮框架事件循环，
    由框架同步成交、检查持仓、生成信号并经过组合执行器处理订单。
    结束后只保留本轮管理标的的订阅，后续轮次可释放已经归零且不再入池的股票。
    持续入口缓存一个 App。交易日、选股规则、候选顺序、账户参数及管理标的一致时复用。
    每轮仍重读风险敞口并运行完整事件循环，行情及成交不使用跨轮状态快照。
    先清空旧缓存、仅成功后恢复，异常后的下一轮重新构建，避免复用半初始化服务。
    未被本函数处理的运行错误向上传播，由持续模式主循环决定是否重试。
    """
    cached = dict(app_cache or {})
    if app_cache is not None:
        app_cache.clear()
    client.__dict__["execution_issues"] = {}
    issue = None
    try:
        snapshot = load_snapshot(universe_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        snapshot, issue = None, str(exc)
    # 诊断字段会随重筛变化，只有日期、规则和有序候选参与策略注册。
    selection_key = None if snapshot is None else (
        snapshot["trading_date"], snapshot["selection_model"], snapshot["rules"],
        tuple(row["symbol"] for row in snapshot["selected"]),
    )
    reused = False
    if (
        cached.get("key") == _app_cache_key(client, directory, parameters)
        and cached.get("selection_key") == selection_key
    ):
        app, selected = cached["app"], list(cached["selected"])
        retained = retained_symbols(client, app, directory)
        tracked = list(dict.fromkeys(selected + sorted(retained)))
        reused = tracked == cached["tracked"]
    if not reused:
        app, tracked, selected = build_portfolio_app(
            client, directory, snapshot, **parameters
        )
    if snapshot and snapshot["selected"] and not selected:
        issue = "Snapshot budgets or signal settings differ from this runner"
    report = app.run(number_of_iterations=1) if tracked else None
    result = runtime_report(
        client, app, directory, report, details={
            "selected": selected, "managed": tracked, "selection_issue": issue,
            "app_reused": reused,
        },
    )
    client.retain_subscriptions(tracked)
    if report and report["status"] == "failed":
        raise RuntimeError("Portfolio iteration failed; see health.json")
    if app_cache is not None:
        app_cache.update(
            key=_app_cache_key(client, directory, parameters), selection_key=selection_key,
            app=app, tracked=list(tracked), selected=list(selected),
        )
    return result


def main():
    """共享账户命令入口，划分只读诊断、单轮执行和持续运行的职责。

    在建立 OAuth 客户端前校验资金、标的数量和组合风险设置。check 只验证
    股票池、账户授权及需管理标的的行情，交易权限关闭；paper/live 才进入
    持有目录锁的执行循环。默认目录按模式隔离，iterations=0 表示每60秒运行
    一轮，其他非负值用于限定轮数。持续模式保留单轮异常后的重试机会，有限
    轮数时异常直接抛出；最终统一释放 SDK 行情订阅。
    """
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "paper", "live"), default="check")
    parser.add_argument(
        "--account",
        choices=("paper", "live"),
        help="Account for check; defaults to paper",
    )
    parser.add_argument("--universe", type=Path, default=DIRECTORY / "universe.json")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--peer-directory", type=Path, action="append", default=[])
    parser.add_argument("--capital", type=float, default=CAPITAL)
    parser.add_argument("--order-budget", type=float, default=ORDER_BUDGET)
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    parser.add_argument(
        "--total-risk-percent", type=float, default=TOTAL_RISK_FRACTION * 100
    )
    # 兼容已有启动命令。旧的测试参数已停用，不参与股数计算。
    parser.add_argument("--min-entry-shares", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--iterations", type=int, default=1, help="0 runs continuously")
    parser.add_argument(
        "--limit-ttl", type=float, default=120, help="Limit order lifetime in seconds"
    )
    args = parser.parse_args()
    parameters = {
        "capital": args.capital,
        "order_budget": args.order_budget,
        "max_positions": args.max_positions,
        "total_risk_fraction": args.total_risk_percent / 100,
        "limit_ttl": args.limit_ttl,
    }
    try:
        SelectionRules(
            capital=args.capital,
            order_budget=args.order_budget,
        )
        PortfolioExecutor(
            None,
            args.max_positions,
            args.total_risk_percent / 100,
        )
        if args.iterations < 0:
            raise ValueError("iterations must be nonnegative")
        if args.mode != "check" and args.account and args.account != args.mode:
            raise ValueError("--account must match the execution mode")
        RuntimeControls(None, ".", args.limit_ttl)
    except ValueError as exc:
        parser.error(str(exc))
    account = (args.account or "paper") if args.mode == "check" else args.mode
    directory = (
        args.directory or Path(__file__).parent / f"resources/portfolio/{account}"
    )
    client = LongbridgeClient(
        LongbridgeOAuthCredential(),
        paper=account == "paper",
        trading_enabled=False,
    )
    try:
        with single_runner(directory):
            issue = None
            try:
                snapshot = load_snapshot(args.universe)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                snapshot, issue = None, str(exc)
            app, tracked, selected = build_portfolio_app(
                client, directory, snapshot, **parameters
            )
            if snapshot and snapshot["selected"] and not selected:
                issue = "Snapshot budgets or signal settings differ from this runner"
            checks = preflight(
                client, app, tracked, args.capital, issue,
                peer_directories=args.peer_directory,
            )
            checks.update(
                selected=selected, managed=tracked, max_positions=args.max_positions
            )
            checks.update(
                capital=args.capital,
                order_budget=args.order_budget,
                total_risk_fraction=args.total_risk_percent / 100,
            )
            atomic_json(directory / "preflight.json", checks)
            print(json.dumps(checks, indent=2), flush=True)
            if args.mode == "check":
                return 0 if checks["ready"] else 2
            client.entry_blockers = checks["entry_blockers"]
            client.trading_enabled = True
            tick = partial(run_once, client, directory, args.universe, app_cache={}, **parameters)
            run_iterations(tick, args.iterations, directory, account)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
