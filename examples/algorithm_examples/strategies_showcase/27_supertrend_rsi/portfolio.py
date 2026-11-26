"""原 Supertrend 的候选资格与共享账户约束；保留已有调用接口和默认参数。"""

from datetime import datetime, timezone

from investing_algorithm_framework import SignalSide
from investing_algorithm_framework.infrastructure.longbridge import NEW_YORK
from investing_algorithm_framework.infrastructure.longbridge.portfolio import (
    PortfolioLimitExecutor, portfolio_risk,
)
from investing_algorithm_framework.infrastructure.longbridge.runtime import WORKING
from supertrend_rsi import SupertrendRSIStrategy
from settings import MAX_POSITIONS, TOTAL_RISK_FRACTION


class PortfolioExecutor(PortfolioLimitExecutor):
    """旧入口保留原默认资金口径，执行链复用共同实现。"""

    def __init__(
        self, client, max_positions=MAX_POSITIONS,
        total_risk_fraction=TOTAL_RISK_FRACTION,
        max_spread_fraction=0.0015, min_entry_shares=None,  # 兼容旧调用，已停用。
    ):
        super().__init__(client, max_positions, total_risk_fraction,
                         max_spread_fraction)


class SelectedStrategy(SupertrendRSIStrategy):
    """保留原版信号算法，仅在开仓端增加选股日期资格，退出始终交由原策略判断。"""

    def __init__(self, *args, entry_date=None, **kwargs):
        """用 entry_date 记录本实例可开新仓的美东日期，其余参数由原策略组装。

        组合入口为当日候选股传入日期，为仅因持仓或委托而保留的股票传 None。
        继承父类的策略 ID 和订单元数据规则，可在股票池变化后继续识别历史成交。
        """
        super().__init__(*args, **kwargs)
        self.entry_date = entry_date

    def generate_signals(self, context, data):
        """过滤失去当日资格的开多信号，继续传递父类生成的所有退出信号。

        日期按美东计算，避免北京时间换日导致交易资格错位。即使股票已经掉池，
        或同一实例运行跨过了资格日期，普通平仓与 ATR 风险退出仍继续评估。
        该方法不强制因掉池而卖出，也不自行取消原有委托。
        """
        today = datetime.now(timezone.utc).astimezone(NEW_YORK).date()
        for signal in super().generate_signals(context, data):
            if signal.side != SignalSide.OPEN_LONG or self.entry_date == today:
                yield signal
