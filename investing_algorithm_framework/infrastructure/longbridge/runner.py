"""多个 Longbridge 策略共用的应用组装与运行目录锁。"""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import time

from . import us_symbol
from .runtime import WORKING, atomic_json


def configure_app(client, resource_directory, capital):
    """绑定账户环境和资金，复用行情连接、成交同步及异常隔离。"""
    from investing_algorithm_framework import (
        LongbridgeOrderExecutor,
        PortfolioConfiguration,
        RESOURCE_DIRECTORY,
        create_app,
    )
    from .operations import (
        IsolatedPortfolioProvider,
        IsolatedOHLCVProvider,
        IsolatedTickerProvider,
    )

    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be positive and finite")
    binding = Path(resource_directory) / "account.json"
    identity = {
        "broker_environment": "paper" if client.paper else "live",
        "capital": capital,
    }
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError(
            "Resource directory account mode or capital "
            "differs from account.json"
        )
    atomic_json(binding, identity)
    app = create_app(config={RESOURCE_DIRECTORY: str(resource_directory)})
    app.add_market_credential(client.credential)
    app.add_order_executor(LongbridgeOrderExecutor(client))
    app.add_portfolio_provider(IsolatedPortfolioProvider(client))
    app.add_data_provider(IsolatedOHLCVProvider(client))
    app.add_data_provider(IsolatedTickerProvider(client))
    app.add_portfolio_configuration(
        PortfolioConfiguration(
            market="LONGBRIDGE",
            trading_symbol="USD",
            initial_balance=capital,
            identifier="LONGBRIDGE",
        )
    )
    return app


@contextmanager
def single_runner(directory):
    """一个运行目录仅允许一个写入进程；系统锁在退出时自动释放。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runner.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "A portfolio runner already owns this directory"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def retained_symbols(client, app, directory):
    """每轮重读持仓、有效委托和提交标记，并清除已经没有敞口的旧退出状态。

    应用复用不缓存账户状态；新出现的掉池敞口必须加入管理，归零后及时释放
    订阅。退出状态清理与重新构建时保持一致，不能因应用复用多保留一轮旧锁定。
    """
    retained = {
        us_symbol(p.symbol)
        for p in app.container.position_service().get_all()
        if p.symbol != "USD" and p.amount != 0
    }
    retained.update(
        us_symbol(o.target_symbol)
        for o in app.container.order_service().get_all()
        if o.status in WORKING
    )
    retained.update(
        path.stem
        for path in Path(directory).glob("longbridge_submissions/*.json")
    )
    for path in Path(directory).glob("runtime/*.json"):
        try:
            state = json.loads(path.read_text())
            if (
                state.get("exit")
                and us_symbol(state["symbol"]) not in retained
            ):
                state["exit"] = None
                atomic_json(path, state)
        except (ValueError, KeyError, TypeError) as exc:
            client.__dict__.setdefault("data_issues", {})[
                f"runtime:{path.name}"
            ] = str(exc)
    return retained


def run_iterations(tick, iterations, directory, account):
    """运行账户回调并输出状态，固定轮数在失败时抛错，常驻进程记录错误后再试。

    传入零表示持续运行。只在仍有下一轮时等待 60 秒；终端中断会退出并由入口释放连接和锁。
    """
    completed = 0
    while iterations == 0 or completed < iterations:
        try:
            print(json.dumps(tick(), indent=2), flush=True)
        except Exception as exc:
            atomic_json(Path(directory) / "health.json", {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed", "error": str(exc), "broker_environment": account,
            })
            if iterations:
                raise
            print(f"Iteration failed; retrying in 60 seconds: {exc}", flush=True)
        completed += 1
        if iterations == 0 or completed < iterations:
            time.sleep(60)
