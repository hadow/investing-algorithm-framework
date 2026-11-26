"""Supertrend + RSI 的独立日频选股任务，只读取 Longbridge 数据并生成候选池。

流程为：确定美股交易日 → 固定基础池 → 日线趋势及风险调整动量排序 → 去除高相关候选 → 小时 ATR
整股可执行性估算 → 发布 universe.json。CLI 提供成分股和财报日历，SDK 提供
交易日及历史行情，全部通过 HTTP 获取，不建立实时订阅或交易上下文；
portfolio_app.py 独占实时行情连接，消费快照、管理持仓和下单。本进程的网络等待
与账户分钟风控相互独立，刷新失败通过错误标记通知账户入口暂停新开仓。
"""

import argparse
from dataclasses import InitVar, asdict, dataclass
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import time

from dotenv import load_dotenv
import numpy as np

from investing_algorithm_framework import LongbridgeClient, LongbridgeOAuthCredential
from investing_algorithm_framework.infrastructure.data_providers.longbridge import (
    candle_frame,
)
from investing_algorithm_framework.infrastructure.longbridge import (
    NEW_YORK,
    us_symbol,
)
from supertrend_rsi import prepare_atr
from investing_algorithm_framework.infrastructure.longbridge.selection import (
    SelectionSource, candidate_correlations, rank_candidates, session_window,
    normalize_daily as normalize_daily,
    earnings_symbols as earnings_symbols,
    cli_json as cli_json,
    subprocess as subprocess,
)

from investing_algorithm_framework.infrastructure.longbridge.runtime import atomic_json
from settings import CAPITAL, ORDER_BUDGET, RISK_FRACTION
from supertrend_universe import combined_membership

DIRECTORY = Path(__file__).parent / "resources/selection"
SELECTION_MODEL = "risk_adjusted_momentum_sp500_ijh_v1"
LEGACY_SELECTION_MODEL = "risk_adjusted_momentum_v2"


@dataclass(frozen=True)
class SelectionRules:
    """不可变的选股规则及预算口径，会完整写入快照供追溯和账户入口核对。

    正式基础池使用完整 C 并集。top_n 控制候选上限，显式 base_size 仅用于旧基础池研究。
    日线过滤流动性、趋势和多周期相对强弱，
    capital、order_budget、risk_fraction 及 ATR 倍数共同估算可负担的整股数。
    max_extension_atr 限制日线偏离，max_correlation 限制候选之间的正相关性。
    选股通过代表有资格等待信号，不代表已经触发小时买入或获得了账户资金。
    """

    base_size: int | None = None  # 正式方案取两组成分的完整并集，按日决定实际数量。
    top_n: int = 100
    min_price: float = 10.0
    min_dollar_volume: float = 50_000_000.0
    history_bars: int = 300
    capital: float = CAPITAL
    order_budget: float = ORDER_BUDGET
    risk_fraction: float = RISK_FRACTION
    min_entry_shares: InitVar[int | None] = None  # 兼容旧调用，不保存到规则或快照。
    stop_atr_multiple: float = 2.0
    time_frame: str = "1h"
    max_extension_atr: float = 4.0
    max_correlation: float = 0.85

    def __post_init__(self, _retired_minimum):
        """在规则创建时验证参数，阻止无效配置进入网络扫描和股票池发布流程。

        数量必须为整数，候选数量不能大于基础池，历史长度须覆盖日线研究需求；
        价格、金额、风险和 ATR 倍数须有效且为正。预算保留现金余量，周期限定为
        第一版支持的分钟／小时周期。账户入口也复用本校验，统一两端参数口径。
        """
        if any(
            type(v) is not int for v in (self.top_n, self.history_bars)
        ):
            raise ValueError("Pool sizes and history_bars must be integers")
        if self.top_n < 1 or (self.base_size is None and self.top_n > 100):
            raise ValueError("Require 1 <= top_n <= 100 for the formal C universe")
        if self.base_size is not None and (
            type(self.base_size) is not int or not self.top_n <= self.base_size <= 200
        ):
            raise ValueError("Require top_n <= base_size <= 200 for a limited S&P scan")
        if not 250 <= self.history_bars <= 999:
            raise ValueError("history_bars must be between 250 and 999")
        values = (
            self.min_price,
            self.min_dollar_volume,
            self.capital,
            self.order_budget,
            self.risk_fraction,
            self.stop_atr_multiple,
            self.max_extension_atr,
            self.max_correlation,
        )
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Selection prices, budgets and risk must be positive")
        if self.order_budget > self.capital * 0.98 or self.risk_fraction > 1:
            raise ValueError("Invalid order budget or risk fraction")
        if self.time_frame not in ("1m", "5m", "15m", "30m", "1h"):
            raise ValueError("Unsupported signal timeframe")
        if self.max_correlation > 1:
            raise ValueError("max_correlation must be in (0, 1]")


def screen(source, rules=SelectionRules(), now=None, progress=None, *, candidate_correlation=True):
    """编排一轮选股并返回可发布快照，本函数不写股票池文件也不提交订单。

    先固定交易日、基础池、SPY 基准和财报暂停名单，再收集日线并调用纯筛选器。
    单只日线失败记录到 errors，基准或财报等公共依赖失败则中止整轮。
    按评分依次考察，先与已入选且通过预算校验的候选比较60日收益率相关性；
    正相关超过上限时跳过并继续下一名，不为凑足候选数量放宽约束。通过相关性后
    再读取已完成主周期 K 线，用入场 ATR 距离、现金余量、
    单笔预算和风险预算估算整股数；不足一股则继续考察下一名，最多保留 top_n。
    小时数据在此仅用于可执行性估算，不要求已经出现买入信号；真正入场由
    账户策略等待。返回 selected、rejected、errors 及规则／日期，供发布和审计；
    小时数据错误也记入 errors；若所有日线合格者的小时数据均不可用，中止刷新，
    不以行情服务故障产生的空池覆盖旧快照。预算不足一股仍属正常筛选结果。
    progress 只用于通知读取进度，now 可注入以验证时间截断逻辑。
    只读实验可将 candidate_correlation 设为 False，把相关性留到开仓时检查。
    """
    now = now or datetime.now(timezone.utc)
    target, as_of, upcoming = session_window(source.client, now)
    membership = None
    if rules.base_size is None:
        symbols, membership = combined_membership(source, target, as_of)
    else:
        symbols = source.base_pool(target, rules.base_size)
    blocked = source.blocked_earnings(symbols, upcoming)
    result = screen_universe(
        source, rules, now, target, as_of, symbols, blocked, progress,
        candidate_correlation=candidate_correlation,
    )
    if membership is not None:
        # 报价故障使整组成分缺失时，不能发布一个实际只覆盖单一来源的 C 池。
        failed = {s for s, reason in result["errors"].items() if reason.startswith("daily_data:")}
        large_count = membership["sp500_count"]
        for group in (symbols[:large_count], symbols[large_count:]):
            if group and failed.issuperset(group):
                raise ValueError("Daily histories unavailable for an entire C universe group")
        result["base_sources"] = membership
        if candidate_correlation:
            result["selection_model"] = SELECTION_MODEL
    return result


def screen_universe(
    source, rules, now, target, as_of, symbols, blocked, progress=None, *,
    candidate_correlation=True,
):
    """对给定时点的成分和财报名单执行同一套选股规则，供盘前选股与历史回放复用。"""
    benchmark = source.daily("SPY.US", as_of, rules.history_bars)
    histories, errors = {}, {}
    for symbol in symbols:
        try:
            histories[symbol] = source.daily(symbol, as_of, rules.history_bars)
        except Exception as exc:
            errors[symbol] = f"daily_data: {exc}"
        if progress:
            progress(symbol)
    ranked, rejected = rank_candidates(histories, benchmark, as_of, rules, blocked)
    if not histories:
        raise ValueError("No usable daily histories; prior snapshot retained")
    correlations = candidate_correlations(histories, ranked, as_of) if candidate_correlation else None
    selected, evaluated = [], 0
    for rank_index, row in enumerate(ranked):
        symbol = row["symbol"]
        if selected and correlations is not None:
            peers = correlations.loc[symbol, [item["symbol"] for item in selected]]
            if not np.isfinite(peers.to_numpy()).all():
                errors[symbol] = "correlation_data_unavailable"
                rejected.append({**row, "reasons": ["correlation_data_unavailable"]})
                continue
            peer, correlation = peers.idxmax(), float(peers.max())
            if correlation > rules.max_correlation:
                rejected.append(
                    {
                        **row, "reasons": ["high_correlation_with_selected"],
                        "correlated_with": peer, "correlation": correlation,
                    }
                )
                continue
        try:
            candles = source.signal_candlesticks(
                symbol,
                rules.time_frame,
                now,
                count=300,
            )
            indicator = prepare_atr(candle_frame(candles, pandas=True))
            if len(indicator) < 30:
                raise ValueError("insufficient_completed_signal_bars")
            latest = indicator.iloc[-1]
            distance = float(latest.ATR) * rules.stop_atr_multiple
            price = float(latest.Close)
            if not math.isfinite(distance) or not 0 < distance < price:
                raise ValueError("invalid_stop_distance")
            shares = math.floor(
                min(
                    rules.order_budget / price,
                    rules.capital * 0.98 / price,
                    rules.capital * rules.risk_fraction / distance,
                )
            )
        except Exception as exc:
            errors[symbol] = f"signal_data: {exc}"
            rejected.append({**row, "reasons": [str(exc)]})
            continue
        evaluated += 1
        if shares < 1:
            rejected.append(
                {**row, "reasons": ["risk_or_budget_below_one_share"]}
            )
            continue
        # 开盘前只能估算股数，下单时还会按实时现金、盘口和组合风险再次裁剪。
        selected.append(
            {
                **row,
                "rank": len(selected) + 1,
                "estimated_shares": shares,
                "stop_distance_estimate": distance,
                "signal_bar": latest.name.isoformat(),
            }
        )
        if len(selected) == rules.top_n:
            rejected.extend(
                {**rest, "reasons": ["outside_top_n"]}
                for rest in ranked[rank_index + 1 :]
            )
            break
    if ranked and not evaluated:
        raise ValueError("No usable signal histories; prior snapshot retained")
    return {
        "schema_version": 1,
        "selection_model": LEGACY_SELECTION_MODEL if candidate_correlation else "research_order_correlation",
        "generated_at": now.isoformat(),
        "trading_date": target.isoformat(),
        "as_of_date": as_of.isoformat(),
        "benchmark": "SPY.US",
        "rules": asdict(rules),
        "base_pool": symbols,
        "daily_eligible_count": len(ranked),
        "selected": selected,
        "rejected": rejected,
        "errors": errors,
    }


def load_snapshot(path, now=None):
    """校验账户入口即将消费的股票池，返回当前美东日期可用的完整快照。

    存在刷新失败标记时，即使旧快照仍属当日也拒绝新开仓；随后验证版本、规则、
    带时区的生成时间、数据截止日和交易日，以及候选数量与重复代码。
    文件缺失、损坏、过期或校验不通过时抛错，由 portfolio_app.run_once 将
    本轮候选置空，并独立从持仓数据库恢复退出管理。此函数不负责平仓或撤单。
    """
    now = now or datetime.now(timezone.utc)
    if Path(path).with_suffix(".error.json").exists():
        raise ValueError("Selection refresh failed; new entries are paused")
    payload = json.loads(Path(path).read_text())
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported universe schema")
    if payload.get("selection_model") != SELECTION_MODEL or payload["rules"].get("base_size") is not None:
        raise ValueError("Refresh the formal C universe before opening new positions")
    target = date.fromisoformat(payload["trading_date"])
    as_of = date.fromisoformat(payload["as_of_date"])
    generated = datetime.fromisoformat(payload["generated_at"])
    # 旧候选已通过当时更严格的股数检查。读取时忽略退役字段，下一次选股重新排名。
    payload["rules"].pop("min_entry_shares", None)
    SelectionRules(**payload["rules"])
    if generated.tzinfo is None or generated > now or as_of >= target:
        raise ValueError("Invalid universe timestamps")
    if target != now.astimezone(NEW_YORK).date():
        raise ValueError("Universe is not for the current US date")
    symbols = [us_symbol(r["symbol"]) for r in payload["selected"]]
    if len(symbols) > payload["rules"]["top_n"] or len(set(symbols)) != len(symbols):
        raise ValueError("Invalid selected symbol list")
    base = {us_symbol(symbol) for symbol in payload["base_pool"]}
    if len(base) != len(payload["base_pool"]) or not set(symbols) <= base:
        raise ValueError("Selected symbols must belong to the unique C universe")
    return payload


def main():
    """只读选股命令入口，负责参数、定期刷新、快照发布和失败状态通知。

    默认执行一次；--watch 每60秒检查美东日期及目标交易日，跨日重新筛选，
    同日规则一致时可复用快照。需要刷新时，--force 或已有失败标记会禁止复用
    结果快照，重新执行筛选流程，
    日线与基础池的按日缓存仍可复用。周末可能生成下个交易日的预览快照。
    成功后原子发布结果并解除失败标记。失败保留旧快照、写入错误标记，
    持续模式等待5分钟再尝试。选股进程只读 HTTP 行情。
    所有行情和 CLI 请求均服务于选股，账户执行由独立的 portfolio_app 负责。
    """
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--watch", action="store_true", help="Refresh every US date")
    parser.add_argument("--force", action="store_true", help="Refresh today's snapshot")
    parser.add_argument("--base-size", type=int, help="Research only: limit to S&P names by turnover")
    parser.add_argument("--top-n", type=int, default=SelectionRules.top_n)
    parser.add_argument("--capital", type=float, default=CAPITAL)
    parser.add_argument("--order-budget", type=float, default=ORDER_BUDGET)
    # 兼容已有启动命令。旧的测试参数已停用，不参与股数计算。
    parser.add_argument("--min-entry-shares", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-extension-atr", type=float, default=SelectionRules.max_extension_atr,
        help="Maximum daily (close - EMA20) / ATR14; default: 4",
    )
    parser.add_argument(
        "--max-correlation", type=float, default=SelectionRules.max_correlation,
        help="Maximum 60-day return correlation between candidates; default: 0.85",
    )
    args = parser.parse_args()
    try:
        rules = SelectionRules(
            base_size=args.base_size,
            top_n=args.top_n,
            capital=args.capital,
            order_budget=args.order_budget,
            max_extension_atr=args.max_extension_atr,
            max_correlation=args.max_correlation,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if rules.base_size is not None and args.directory.resolve() == DIRECTORY.resolve():
        parser.error("Use a separate --directory for a limited S&P research scan")
    expected_model = SELECTION_MODEL if rules.base_size is None else LEGACY_SELECTION_MODEL
    path = args.directory / "universe.json"
    client = LongbridgeClient(
        LongbridgeOAuthCredential(), paper=False, quote_transport="http"
    )
    source = SelectionSource(client, args.directory)
    last_day = None
    retry_at = 0.0
    try:
        while True:
            try:
                current = datetime.now(timezone.utc)
                target, _, _ = session_window(client, current)
                day_key = (target, current.astimezone(NEW_YORK).date())
                if day_key != last_day and time.monotonic() >= retry_at:
                    cached = None
                    if (
                        path.exists()
                        and not args.force
                        and not path.with_suffix(".error.json").exists()
                    ):
                        try:
                            cached = json.loads(path.read_text())
                            created_date = (
                                datetime.fromisoformat(cached["generated_at"])
                                .astimezone(NEW_YORK)
                                .date()
                            )
                            if (
                                cached["trading_date"] != target.isoformat()
                                or cached.get("selection_model") != expected_model
                                or cached["rules"] != asdict(rules)
                                or created_date != day_key[1]
                            ):
                                cached = None
                        except (OSError, ValueError, KeyError, TypeError):
                            cached = None
                    result = cached or screen(
                        source,
                        rules,
                        progress=lambda s: print(f"Checked {s}", flush=True),
                    )
                    if cached is None:
                        atomic_json(path, result)
                    path.with_suffix(".error.json").unlink(missing_ok=True)
                    last_day = day_key
                    print(
                        json.dumps(
                            {
                                "snapshot": str(path),
                                "trading_date": result["trading_date"],
                                "selection_model": result["selection_model"],
                                "selected": [r["symbol"] for r in result["selected"]],
                                "data_errors": result["errors"],
                                "trading_enabled": False,
                            },
                            indent=2,
                        ),
                        flush=True,
                    )
                if not args.watch:
                    break
            except Exception as exc:
                atomic_json(
                    path.with_suffix(".error.json"),
                    {
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "error": str(exc),
                    },
                )
                if not args.watch:
                    raise
                print(
                    f"Selection refresh failed; prior snapshot retained: {exc}",
                    flush=True,
                )
                retry_at = time.monotonic() + 300
            if args.watch:
                time.sleep(60)
    finally:
        client.close()


if __name__ == "__main__":
    main()
