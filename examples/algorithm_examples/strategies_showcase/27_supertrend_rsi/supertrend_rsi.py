# SPDX-License-Identifier: MPL-2.0
# Signal rules adapted from Alorse (c) alorse, Supertrend + RSI.pine.
# Upstream: Alorse/pinescript-strategies
# Commit: 8fb3837ddcd137f1450c3e82d654ff3df24d54d5
"""Supertrend + RSI 的指标计算与信号核心，移植自 Alorse 的 Pine v5 策略。

本模块负责把已完成 K 线转为只做多信号，供原版 CSV 研究和账户运行共同使用。
选股资格由 portfolio.SelectedStrategy 限制，资金和订单执行由各自执行器负责。
账户版可额外启用基于入场 ATR 的退出；原版回测保留上游信号作为比较基线。
"""

import math

import numpy as np
import pandas as pd
from investing_algorithm_framework.domain.utils.wilder import rma, wilder_atr
from investing_algorithm_framework.infrastructure.longbridge import us_symbol

from investing_algorithm_framework import (
    ConflictPolicy,
    DataSource,
    DataType,
    PositionSize,
    Schedule,
    Signal,
    SignalSide,
    TimeFrame,
    TimeUnit,
    TradingStrategy,
    signal_series_from_column,
)

UPSTREAM_COMMIT = "8fb3837ddcd137f1450c3e82d654ff3df24d54d5"


def prepare_atr(frame, atr_period=10):
    """校验已完成的 K 线，返回带 Wilder ATR 的副本，供选股和交易共用。

    时间统一为 UTC。首根使用当根振幅，后续真实波幅纳入与前收盘价的跳空。
    选股用最新 ATR 估算止损距离和可买股数，交易用同一序列构建趋势轨道。
    """
    frame = frame.copy()
    if "Datetime" in frame:
        frame = frame.set_index("Datetime")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("OHLCV requires a DatetimeIndex or Datetime column")
    frame.index = pd.to_datetime(frame.index, utc=True)
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("OHLCV timestamps must be unique and increasing")
    prices = frame[["High", "Low", "Close"]].to_numpy(dtype=float)
    if (
        not np.isfinite(prices).all()
        or (prices <= 0).any()
        or (prices[:, 0] < prices[:, 1]).any()
    ):
        raise ValueError("OHLCV prices must be finite, positive and High >= Low")
    high, low, close = prices.T
    frame["ATR"] = wilder_atr(high, low, close, atr_period)
    return frame


def prepare_indicators(frame, atr_period=10, factor=3.0, rsi_period=14, rsi_exit=72.0):
    """将 OHLCV 转为指标及原版开平仓标记，统一向量与事件两条信号路径。

    调用方提供已完成的 K 线，prepare_atr 负责校验价格、时间顺序并计算 ATR。
    Supertrend 继承历史有效轨道，以收盘突破决定是否翻转，-1 为上涨、+1 为下跌。
    开多要求上涨且 Wilder RSI 从严格小于 50 变为严格大于 50。平多为 RSI 超过
    rsi_exit 或方向翻空。返回原始数据加指标和布尔标记，不依赖持仓或未来数据。
    """
    frame = prepare_atr(frame, atr_period)
    high, low, close = (
        frame[name].to_numpy(dtype=float) for name in ("High", "Low", "Close")
    )
    atr = frame["ATR"].to_numpy()
    count = len(frame)
    upper = (high + low) / 2 + factor * atr
    lower = (high + low) / 2 - factor * atr
    direction = np.ones(count, dtype=int)  # 沿用 Pine 符号：+1 下跌，-1 上涨。
    supertrend = np.full(count, np.nan)
    for index in range(count):
        if math.isnan(atr[index]):
            continue
        if index and math.isfinite(atr[index - 1]):
            # 历史轨道约束使上涨下轨逐步收紧、下跌上轨逐步收紧，突破后再重置。
            if not (
                lower[index] > lower[index - 1] or close[index - 1] < lower[index - 1]
            ):
                lower[index] = lower[index - 1]
            if not (
                upper[index] < upper[index - 1] or close[index - 1] > upper[index - 1]
            ):
                upper[index] = upper[index - 1]
            if supertrend[index - 1] == upper[index - 1]:
                direction[index] = -1 if close[index] > upper[index] else 1
            else:
                direction[index] = 1 if close[index] < lower[index] else -1
        supertrend[index] = lower[index] if direction[index] < 0 else upper[index]
    changes = np.diff(close, prepend=np.nan)
    gains = rma(np.maximum(changes, 0), rsi_period)
    losses = rma(np.maximum(-changes, 0), rsi_period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + gains / losses)
    # 对齐当前 Pine 参考实现：平均跌幅为零时 RSI 为 100，平价输入也走此分支。
    rsi = np.where(losses == 0, 100, rsi)
    frame["Supertrend"] = supertrend
    frame["Direction"] = direction
    frame["RSI"] = rsi
    frame["entry_long"] = (
        (frame.Direction < 0) & (frame.RSI.shift(1) < 50) & (frame.RSI > 50)
    )
    frame["exit_long"] = (frame.RSI > rsi_exit) | (frame.Direction.diff() > 0)
    return frame


class SupertrendRSIStrategy(TradingStrategy):
    """把同一套指标规则适配为框架策略，并在账户运行时附加成交驱动的风控。"""

    algorithm_id = "alorse-supertrend-rsi"

    def __init__(
        self,
        symbol="SPY.US/USD",
        time_frame="1h",
        market="LONGBRIDGE",
        atr_period=10,
        factor=3.0,
        rsi_period=14,
        rsi_exit=72.0,
        warmup_window=300,
        stop_atr_multiple=None,
        risk_fraction=0.005,
        order_budget=None,
        schedule=None,
        executor=None,
        runtime=None,
    ):
        """组装数据源、仓位规则和调度，明确原版研究与账户运行的边界。

        symbol 统一为券商标的加 USD 计价格式；指标参数和预热长度在访问账户前
        校验。主 OHLCV 决定信号周期，注入 executor 时才额外注册实时报价源。
        stop_atr_multiple 为空时保留上游默认只做多基线，并按信号周期调度；
        启用 ATR 退出后改为每分钟检查，必要时加 1m 心跳，但信号仍取主周期。
        order_budget 指定单笔名义预算；未指定时沿用上游 20% 组合资金比例。
        稳定的标的／周期策略 ID 用于重启后识别成交归属及同根买单去重。
        本类只生成意图，实际整股数量、现金约束和券商下单交给框架及执行器。
        """
        market = market.upper()
        if market == "LONGBRIDGE":
            symbol = f"{us_symbol(symbol)}/USD"
        for name, period in (("atr_period", atr_period), ("rsi_period", rsi_period)):
            if type(period) is not int or period < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("factor must be finite and positive")
        if not math.isfinite(rsi_exit) or not 50 < rsi_exit <= 100:
            raise ValueError("rsi_exit must be greater than 50 and at most 100")
        if (
            type(warmup_window) is not int
            or not max(atr_period + 1, rsi_period + 2) <= warmup_window <= 999
        ):
            raise ValueError("warmup_window must cover indicators and be <=999")
        if stop_atr_multiple is not None and (
            not math.isfinite(stop_atr_multiple) or stop_atr_multiple <= 0
        ):
            raise ValueError("stop_atr_multiple must be finite and positive")
        if not math.isfinite(risk_fraction) or not 0 < risk_fraction <= 1:
            raise ValueError("risk_fraction must be in (0, 1]")
        if order_budget is not None and (
            not math.isfinite(order_budget) or order_budget <= 0
        ):
            raise ValueError("order_budget must be finite and positive")
        self.symbol = symbol
        self.time_frame = TimeFrame.from_value(time_frame).value
        self.market = market
        self.atr_period, self.factor = atr_period, factor
        self.rsi_period, self.rsi_exit = rsi_period, rsi_exit
        self.stop_atr_multiple = stop_atr_multiple
        self.risk_fraction = risk_fraction
        self.runtime = runtime
        self.data_key = f"{symbol}-{self.time_frame}-ohlcv"
        sources = [
            DataSource(
                identifier=self.data_key,
                data_type=DataType.OHLCV,
                symbol=symbol,
                market=market,
                time_frame=self.time_frame,
                warmup_window=warmup_window,
                pandas=True,
            )
        ]
        if executor is not None:
            sources.append(
                DataSource(
                    identifier=f"{symbol}-ticker",
                    symbol=symbol,
                    market=market,
                    data_type=DataType.TICKER,
                )
            )
        minutes = TimeFrame.from_value(time_frame).amount_of_minutes
        # 风险检查需要分钟调度心跳；新增 1m 数据源不改变主信号所用的 K 线周期。
        if stop_atr_multiple is not None and self.time_frame != "1m":
            sources.append(
                DataSource(
                    identifier=f"{symbol}-risk-1m",
                    data_type=DataType.OHLCV,
                    symbol=symbol,
                    market=market,
                    time_frame="1m",
                    warmup_window=2,
                    pandas=True,
                )
            )
        size = (
            PositionSize(fixed_amount=order_budget)
            if order_budget
            else (PositionSize(percentage_of_portfolio=20))
        )
        super().__init__(
            algorithm_id=self.algorithm_id,
            strategy_id=f"supertrend-rsi-{symbol.replace('/', '-')}-{self.time_frame}",
            schedule=schedule
            or Schedule.every(
                1 if stop_atr_multiple is not None else minutes,
                TimeUnit.MINUTE,
            ),
            symbols=[symbol.split("/")[0]],
            data_sources=sources,
            position_sizes=[size],
            conflict_policy=ConflictPolicy.long_only(),
            executor=executor,
        )
        self.set_parameters(
            {
                "symbol": symbol,
                "time_frame": self.time_frame,
                "market": market,
                "atr_period": atr_period,
                "factor": factor,
                "rsi_period": rsi_period,
                "rsi_exit": rsi_exit,
                "warmup_window": warmup_window,
                "stop_atr_multiple": stop_atr_multiple,
                "risk_fraction": risk_fraction,
                "upstream_commit": UPSTREAM_COMMIT,
            }
        )

    def prepare(self, data):
        """从框架数据字典取出主周期 K 线，按本策略参数调用统一指标函数。

        data_key 明确区分同一标的的主周期、实时报价和风控心跳数据，防止把
        1m 心跳误当成 1h 信号输入。两种信号生成方法都经过这里，避免公式分叉。
        """
        return prepare_indicators(
            data[self.data_key],
            self.atr_period,
            self.factor,
            self.rsi_period,
            self.rsi_exit,
        )

    def generate_signal_series(self, data):
        """为向量回测生成与历史索引对齐的开多、平多 SignalSeries。

        这里不读取账户状态，由回测引擎处理仓位与成交。启用额外 ATR 退出时
        主动拒绝该入口，因为成交价、盘口和分钟检查无法由这两列原版信号表达，
        避免把原版向量回测的收益误归到受保护的账户版本。
        """
        if self.stop_atr_multiple is not None:
            raise ValueError("Vector research uses the original signal variant")
        frame = self.prepare(data)
        for column, side in (
            ("entry_long", SignalSide.OPEN_LONG),
            ("exit_long", SignalSide.CLOSE_LONG),
        ):
            yield signal_series_from_column(
                frame,
                column,
                symbol=self.symbols[0],
                side=side,
                source="supertrend_rsi",
            )

    def generate_signals(self, context, data):
        """账户入口启用持久化运行控制；原版只读研究继续直接输出指标信号。

        RuntimeControls 将分钟风险检查与主周期单次决策分开，并在生成平仓意图
        前撤销冲突挂单、确认终态及同步持仓。未注入 runtime 的独立研究不受
        账户时钟、交易日历或本地状态影响，保持 Pine 公式和向量基线一致。
        """
        if self.runtime is not None and context is not None:
            yield from self.runtime.generate(self, context, data)
        else:
            yield from self._generate_uncontrolled_signals(context, data)

    def run_strategy(self, context, data):
        """隔离账户运行中单只股票的错误，供其他股票继续处理退出并记录失败原因。

        原版研究仍直接抛错。账户错误写入 runtime.status，运行器据此报告降级，
        不能只凭框架 completed 状态判定所有委托成功；退出文件不会因异常清除。
        """
        if self.runtime is None:
            return super().run_strategy(context, data)
        try:
            return super().run_strategy(context, data)
        except Exception as exc:
            self.runtime.status = {"reason": "strategy_error", "error": str(exc)}
            self.traces = [("runtime.error", str(exc))]

    def _generate_uncontrolled_signals(self, context, data):
        """生成本轮账户信号：优先风险退出，再检查最新已完成 K 线的原版规则。

        有 context 且启用 ATR 风控时，先从归属本策略的原买单恢复固定退出距离，
        用当前买一价与各笔实际成交价比较；任一笔触及阈值即发出标的平多信号，
        携带 risk_exit 让执行器选择市价退出，并结束本轮信号生成。
        普通信号附带 K 线时间、指标值和上游版本，方便持久化与追溯。同根 K 线
        已存在本策略买单时不再开仓，重启后也依据历史订单保持该约束。
        新买单把信号 K 线的 ATR 距离冻结在元数据中，并携带风险比例；成交后
        用实际成交价减去该距离形成退出阈值，后续 K 线不会把这段距离改成跟踪值。
        context 为空时可只读观察原始信号；最终持仓和未完成委托检查由框架执行。
        """
        risk_signals = list(self.generate_risk_signals(context))
        if risk_signals:
            yield from risk_signals
            return
        yield from self._generate_core_signals(context, data)

    def generate_risk_signals(self, context):
        """仅用成交记录和当前买一价检查固定 ATR 退出，不读取主周期指标。

        与普通信号拆开后，小时数据缺失、错误或尚未更新都不会阻断分钟风险检查，
        已锁定的退出也可继续撤单及卖出。任一归属本策略的成交触及阈值即平标的。
        """
        target = self.symbols[0]
        if self.stop_atr_multiple is not None and context is not None:
            trades = [
                t
                for t in context.get_open_trades(target_symbol=target)
                if t.strategy_id == self.strategy_id
            ]
            if trades:
                bid = context.get_latest_price(self.symbol, market=self.market)
                # 无效盘口不能变成不可撤回的止损承诺；运行控制另校验报价时效与时段。
                if bid is None or not math.isfinite(bid) or bid <= 0:
                    return
                for trade in trades:
                    distance = trade.buy_order.metadata.get("stop_distance")
                    if (
                        distance
                        and bid is not None
                        and (bid <= trade.open_price - distance)
                    ):
                        yield Signal(
                            target,
                            SignalSide.CLOSE_LONG,
                            source="supertrend_rsi_atr_stop",
                            metadata={"risk_exit": True, "trigger_trade_id": trade.id},
                        )
                        return

    def _generate_core_signals(self, context, data):
        """从主周期生成原版开平信号，冻结新仓 ATR 距离并保留历史买单去重。

        不负责行情时效与信号水位；账户 RuntimeControls 确认本根尚未消费后
        才调用，独立研究则通过未控制入口调用，保证两端使用同一套 Pine 公式。
        """
        target = self.symbols[0]
        candles = data.get(self.data_key)
        if candles is None or len(candles) < max(
            self.atr_period + 1, self.rsi_period + 2
        ):
            return
        frame = self.prepare(data)
        if len(frame) < max(self.atr_period + 1, self.rsi_period + 2):
            return
        row = frame.iloc[-1]
        metadata = {
            "signal_bar": frame.index[-1].isoformat(),
            "upstream_commit": UPSTREAM_COMMIT,
            "rsi": float(row.RSI),
            "atr": float(row.ATR),
        }
        if row.exit_long:
            yield Signal(
                target,
                SignalSide.CLOSE_LONG,
                source="supertrend_rsi",
                metadata=metadata,
            )
        if row.entry_long:
            if context is not None:
                # 去重查询持久化历史，而非进程内标记，防止重启或止损后同根再买。
                orders = context.get_orders(
                    target_symbol=target, market=self.market, order_side="BUY"
                )
                if any(
                    o.strategy_id == self.strategy_id
                    and o.metadata.get("signal_bar") == metadata["signal_bar"]
                    for o in orders
                ):
                    return
            if self.stop_atr_multiple is not None:
                metadata.update(
                    stop_distance=float(row.ATR) * self.stop_atr_multiple,
                    risk_fraction=self.risk_fraction,
                )
            yield Signal(
                target, SignalSide.OPEN_LONG, source="supertrend_rsi", metadata=metadata
            )
