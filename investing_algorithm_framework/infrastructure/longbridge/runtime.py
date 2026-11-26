"""账户运行控制：持久化信号去重、退出意图和撤单确认，不改变 Pine 指标公式。"""

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from investing_algorithm_framework import Signal, SignalSide
from ..data_providers.longbridge import (
    is_us_regular_session,
    latest_completed_us_bar,
)
from investing_algorithm_framework.infrastructure.longbridge import NEW_YORK
from investing_algorithm_framework.infrastructure.longbridge.execution import (
    is_tradable_quote,
)

WORKING = frozenset(("CREATED", "OPEN"))


def atomic_json(path, value):
    """先完整写入并同步同目录临时文件，再原子替换，供退出状态和健康报告共用。

    退出意图必须先于券商撤单落盘；进程重启后读取的是完整旧版或新版。
    禁止 NaN，失败时保留旧文件，调用方必须停止该标的本轮执行并报告错误。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def utc(stamp):
    """将框架持久化时间转为 UTC；SQLite 无时区字段按框架的 UTC 约定解释。

    仅用于本地订单及状态，SDK 的无时区报价时间仍须按机器本地时区转换。
    """
    return (
        stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
    )


class RuntimeControls:
    """每个策略共享资源目录，状态按稳定策略 ID 隔离，并由运行器目录锁保护。"""

    def __init__(self, client, directory, limit_ttl=120, clock=None):
        """注入券商、持久化目录、限价单寿命和可测试时钟，不访问网络或下单。"""
        if not math.isfinite(limit_ttl) or limit_ttl <= 0:
            raise ValueError("limit_ttl must be positive and finite")
        self.client = client
        self.directory = Path(directory)
        self.limit_ttl = limit_ttl
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.status = {}

    def _load(self, strategy):
        """恢复水位与未完成退出；损坏文件直接报错，避免重置后重复开仓。"""
        self.path = self.directory / "runtime" / f"{strategy.strategy_id}.json"
        state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "schema_version": 1,
                "last_bar": None,
                "exit": None,
            }
        )
        if (
            state.get("schema_version") != 1
            or not {"last_bar", "exit"} <= state.keys()
        ):
            raise ValueError(f"Invalid runtime state: {self.path}")
        if state["last_bar"] is not None:
            utc(datetime.fromisoformat(state["last_bar"]))
        if state["exit"] is not None and not isinstance(state["exit"], dict):
            raise ValueError(f"Invalid exit state: {self.path}")
        state["symbol"] = strategy.symbols[0]
        return state

    def _orders(self, strategy, context):
        """每次从数据库读取有效委托，撤单与竞态成交后不复用旧对象。"""
        return [
            order
            for order in context.get_orders(
                target_symbol=strategy.symbols[0], market=strategy.market
            )
            if order.status in WORKING
        ]

    def _quantity(self, strategy, context):
        """读取最新总持仓，包含尚被卖单预留的数量，不能用 trade.remaining 替代。"""
        position = context.get_position(
            strategy.symbols[0], market=strategy.market
        )
        return float(position.amount) if position is not None else 0.0

    def _latch(self, state, metadata, source):
        """先持久化退出承诺；撤单等待期间即使价格反弹或进程重启也继续退出。"""
        if state["exit"] is None:
            state["exit"] = {
                "exit_id": str(uuid4()),
                "source": source,
                "metadata": dict(metadata),
            }
        elif metadata.get("risk_exit") or metadata.get("market_exit"):
            state["exit"]["metadata"].update(metadata)
        atomic_json(self.path, state)

    def _unknown_submission(self, strategy, orders):
        """识别无券商编号的本地委托及孤立提交标记，禁止猜测是否已成交并重试。

        已有持久化外部编号的同笔标记由框架订单同步清除；其余交由账户对账。
        """
        if any(not order.external_id for order in orders):
            return True
        path = (
            self.directory
            / "longbridge_submissions"
            / f"{strategy.symbols[0]}.json"
        )
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        return not any(
            str(order.external_id) == str(record.get("order_id"))
            for order in orders
        )

    def _cancel(self, context, order):
        """只对已知 OPEN 委托申请撤单；确认状态及分笔成交由订单服务同步。

        SDK cancel 返回仅表示受理，因此调用后仍需重读数据库；本轮出现异常时
        保留退出意图，下轮先同步状态再决定，不生成可能重复的卖单。
        """
        if not self.client.trading_enabled:
            raise RuntimeError("Runtime cancellation requires trading mode")
        if order.id in self._cancelled_this_tick:
            return
        if order.status == "OPEN" and order.external_id:
            self._cancelled_this_tick.add(order.id)
            context.order_service.cancel_order(order)

    def _maintain_limits(self, strategy, context, state, now, orders):
        """买单超时撤余量并保留已成交仓位；卖单超时锁定市价退出后申请撤余量。

        委托寿命自本地创建起计时，不对 MARKET 单套用撤单重发规则。
        所有后续委托都必须等待原单终态，避免超卖或订单回报不明时重复提交。
        使用本轮刚读取的有效委托；调用方在可能撤单后重新查询状态。
        """
        for order in orders:
            if order.order_type != "LIMIT" or order.status != "OPEN":
                continue
            if (now - utc(order.created_at)).total_seconds() < self.limit_ttl:
                continue
            if order.order_side == "SELL":
                self._latch(state, {"market_exit": True}, "limit_exit_timeout")
            self._cancel(context, order)

    def _exit_signal(self, strategy, context, state):
        """等待冲突单终态，再按同步后的持仓交给框架定量；同一退出单只提交一次。

        正在工作的本次退出单继续等待回报。其他买卖挂单先撤，待撤状态仍然阻挡
        卖出；撤单期间的新成交会先入账，随后 SizePositions 读取最新持仓。
        全部归零且无挂单才清除锁定，退出完成这一轮不会重新开仓。
        """
        latch = state["exit"]
        orders = self._orders(strategy, context)
        if self._unknown_submission(strategy, orders):
            self.status["reason"] = "needs_reconciliation"
            return
        for order in orders:
            same_exit = order.metadata.get("exit_id") == latch["exit_id"]
            if same_exit and order.order_side == "SELL":
                urgent = latch["metadata"].get("risk_exit") or latch[
                    "metadata"
                ].get("market_exit")
                if not urgent or order.order_type == "MARKET":
                    self.status["reason"] = "exit_order_working"
                    return
            self._cancel(context, order)
        if self._orders(strategy, context):
            self.status["reason"] = "waiting_cancel_confirmation"
            return
        if self._quantity(strategy, context) <= 0:
            state["exit"] = None
            atomic_json(self.path, state)
            self.status["reason"] = "exit_complete"
            return
        self.status["reason"] = "exit_pending"
        yield Signal(
            strategy.symbols[0],
            SignalSide.CLOSE_LONG,
            source=latch["source"],
            metadata={**latch["metadata"], "exit_id": latch["exit_id"]},
        )

    def generate(self, strategy, context, data):
        """按分钟维护风险，按已确认主周期 K 线消费一次普通决策，并记录阻断原因。

        水位在输出普通意图前落盘：开仓失败、无额度或无盘口均跳过本根，避免
        同根反复试单；退出另有持久化承诺，会持续到持仓归零。空仓时同根同时
        命中原版入场／离场只尝试入场，下一分钟不会使用旧离场信号平掉新仓。
        行情必须匹配交易日历预期的最新完成 K 线，旧数据不能触发新仓；ATR
        退出和已有退出承诺不依赖主周期更新。新 ATR 退出仅接受常规时段、正常
        交易且120秒内的报价。订单同步失败时暂停新仓，受影响股票保留退出承诺
        并等待对账，其他股票继续退出。pause_entries 文件只暂停开仓。
        """
        now = self.clock()
        self._cancelled_this_tick = set()
        state = self._load(strategy)
        self.status = {"last_bar": state["last_bar"], "reason": "no_signal"}
        for signal in strategy.generate_risk_signals(context):
            if signal.metadata.get("risk_exit"):
                # 必须在承诺退出前校验；执行器拒单不能撤回已落盘的错误止损。
                if not is_us_regular_session(self.client, now, False):
                    self.status["risk_check"] = "outside_regular_session"
                    continue
                quote = self.client.get_quote(strategy.symbols[0])
                if not is_tradable_quote(quote, now):
                    self.status["risk_check"] = "quote_stale_or_suspended"
                    continue
                frame = data.get(strategy.data_key)
                if frame is not None and len(frame):
                    stamp = frame.index[-1]
                    if state["last_bar"] is None or utc(stamp) > utc(
                        datetime.fromisoformat(state["last_bar"])
                    ):
                        state["last_bar"] = stamp.isoformat()
                self._latch(state, signal.metadata, signal.source)
        orders = self._orders(strategy, context)
        order_issues = self.client.__dict__.get("order_issues", {})
        if any(str(order.id) in order_issues for order in orders):
            self.status["reason"] = "order_sync_pending"
            return
        self._maintain_limits(strategy, context, state, now, orders)
        if state["exit"] is not None:
            yield from self._exit_signal(strategy, context, state)
            return
        # 空列表不会触发撤单或成交同步；有委托时继续重读，保留撤单竞态保护。
        if orders:
            orders = self._orders(strategy, context)
        if self._unknown_submission(strategy, orders):
            self.status["reason"] = "needs_reconciliation"
            return
        frame = data.get(strategy.data_key)
        # 不同策略声明各自的预热要求，账户控制不依赖某一种指标。
        minimum_bars = getattr(strategy, "minimum_bars", None)
        if minimum_bars is None:
            minimum_bars = max(
                strategy.atr_period + 1, strategy.rsi_period + 2
            )
        if frame is None or len(frame) < minimum_bars:
            self.status["reason"] = "insufficient_bars"
            return
        stamp = frame.index[-1].to_pydatetime()
        expected = latest_completed_us_bar(
            self.client, strategy.time_frame, now
        )
        self.status.update(
            last_bar=stamp.isoformat(), expected_bar=expected.isoformat()
        )
        if utc(stamp) != expected:
            self.status["reason"] = "stale_or_unconfirmed_bar"
            return
        if state["last_bar"] and utc(stamp) <= utc(
            datetime.fromisoformat(state["last_bar"])
        ):
            self.status["reason"] = "bar_already_processed"
            return
        cold_start = state["last_bar"] is None
        raw = list(strategy._generate_core_signals(context, data))
        state["last_bar"] = stamp.isoformat()
        atomic_json(self.path, state)
        held = self._quantity(strategy, context) > 0
        exits = [
            signal for signal in raw if signal.side == SignalSide.CLOSE_LONG
        ]
        if exits and (
            held or any(order.order_side == "BUY" for order in orders)
        ):
            self._latch(state, exits[0].metadata, exits[0].source)
            yield from self._exit_signal(strategy, context, state)
            return
        if (
            (self.directory / "pause_entries").exists()
            or self.client.__dict__.get("entry_blockers")
            or order_issues
        ):
            self.status["reason"] = "entries_paused"
            return
        if (
            stamp.astimezone(NEW_YORK).date()
            != now.astimezone(NEW_YORK).date()
        ):
            self.status["reason"] = "awaiting_current_session_bar"
            return
        if not is_us_regular_session(self.client, now):
            self.status["reason"] = "outside_entry_window"
            return
        # 新加入的标的先建立水位；重启后也只能消费刚收盘的信号，防止追买旧翻转。
        if cold_start and getattr(strategy, "skip_initial_entry", False):
            self.status["reason"] = "cold_start_observation"
            return
        max_delay = getattr(strategy, "entry_max_delay_seconds", None)
        if (
            max_delay is not None
            and latest_completed_us_bar(
                self.client,
                strategy.time_frame,
                now - timedelta(seconds=max_delay),
            )
            == expected
        ):
            self.status["reason"] = "entry_signal_expired"
            return
        for signal in raw:
            if signal.side == SignalSide.OPEN_LONG and not held and not orders:
                self.status["reason"] = "entry_signal"
                yield signal
