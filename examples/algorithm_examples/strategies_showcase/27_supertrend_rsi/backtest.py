"""使用本地 OHLCV CSV 研究原版只做多 Supertrend + RSI，不连接券商账户。

本入口提供固定标的的向量回测基线，保留上游信号与20%资金比例，计入设置的
手续费和滑点。每日动态选股、真实盘口成交和额外 ATR 退出由其他链路验证，
不能把此处基线的指标直接归为多标的账户版本的收益。
"""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import pandas as pd

from investing_algorithm_framework import (
    BacktestDateRange,
    BacktestEngine,
    BacktestRunConfiguration,
    BacktestWindow,
    ExecutionConfig,
    PandasOHLCVDataProvider,
    RESOURCE_DIRECTORY,
    Study,
    TradingCost,
    Universe,
    create_app,
)
from supertrend_rsi import SupertrendRSIStrategy
from investing_algorithm_framework.infrastructure.longbridge import us_symbol


def run_research(
    csv_path,
    *,
    symbol="SPY.US/USD",
    time_frame="1h",
    capital=2000,
    warmup_window=300,
    fee_percentage=0.01,
    slippage_percentage=0.02,
    start=None,
    end=None,
    resource_directory=None,
):
    """建立并执行一个本地 CSV 研究任务，返回 Backtest 结果和研究名称。

    先规范标的、UTC 时间及指标输入，默认保留 warmup_window 行作为预热，
    用其后区间研究；指定 start/end 时仍须满足数据覆盖和预热边界。
    框架要求研究起止为整点，因此起点向上、终点向下收敛。其数据源预热长度
    按自然时间计算，这里扩展到 CSV 可用前缀，减轻周末／休市造成的有效样本
    缩短；首个不完整区间可能仍被裁掉，比较递归指标时须对齐实际输入切片。
    通过 Pandas provider 和向量引擎运行原版信号，禁用缺失 K 线填充及检查点
    复用，以免造出休市价格或误用旧结果。费用与滑点是研究假设，成交模型也
    属于框架回测模型；返回对象可继续读取逐笔交易和统计指标。
    """
    symbol = f"{us_symbol(symbol)}/USD"
    frame = pd.read_csv(csv_path)
    frame["Datetime"] = pd.to_datetime(frame.Datetime, utc=True)
    frame = frame.sort_values("Datetime").set_index("Datetime")
    strategy = SupertrendRSIStrategy(
        symbol=symbol,
        time_frame=time_frame,
        warmup_window=warmup_window,
    )
    strategy.prepare({strategy.data_key: frame})  # 在启动回测前发现指标输入问题。
    if len(frame) <= warmup_window + 1:
        raise ValueError("CSV needs warmup history plus at least two test bars")
    start = (
        pd.to_datetime(start, utc=True) if start else frame.index[warmup_window]
    ).ceil("h")
    end = (pd.to_datetime(end, utc=True) if end else frame.index[-1]).floor("h")
    if not frame.index[warmup_window] <= start < end <= frame.index[-1]:
        raise ValueError("Test range must follow the CSV warmup and fit its data")
    if (
        not all(
            math.isfinite(v)
            for v in (
                capital,
                fee_percentage,
                slippage_percentage,
            )
        )
        or capital <= 0
        or fee_percentage < 0
        or slippage_percentage < 0
    ):
        raise ValueError("Capital must be positive; costs must be nonnegative")
    directory = resource_directory or Path(__file__).parent / "resources/research"
    # 框架按自然时间回溯预热；扩展到 CSV 前缀，避免周末休市挤掉有效指标样本。
    # shortcut: 整点边界可能舍弃首个非完整区间，比较递归指标前须对齐实际切片。
    minutes = strategy.data_sources[0].time_frame.amount_of_minutes
    intervals = int((start - frame.index[0]).total_seconds() / (minutes * 60))
    strategy.data_sources = [
        replace(source, warmup_window=intervals, window_size=None)
        for source in strategy.data_sources
    ]
    app = create_app(config={RESOURCE_DIRECTORY: str(directory)})
    app.add_data_provider(
        PandasOHLCVDataProvider(
            frame,
            symbol=symbol,
            market="LONGBRIDGE",
            time_frame=time_frame,
            warmup_window=intervals,
            pandas=True,
        )
    )
    study = Study(
        name="supertrend_rsi_original",
        initial_capital=capital,
        universe=Universe(
            market="LONGBRIDGE", trading_symbol="USD", symbols=[symbol.split("/")[0]]
        ),
        engines=[BacktestEngine.VECTOR],
        backtest_windows=[
            BacktestWindow(
                train_range=BacktestDateRange(
                    start_date=start.to_pydatetime(),
                    end_date=end.to_pydatetime(),
                )
            )
        ],
        execution_config=ExecutionConfig(
            trading_costs=[
                TradingCost(
                    fee_percentage=fee_percentage,
                    slippage_percentage=slippage_percentage,
                )
            ]
        ),
        metadata={"variant": "original_long_only", "csv": str(csv_path)},
    )
    results = app.run_backtest(
        strategy=strategy,
        study=study,
        run_configuration=BacktestRunConfiguration(
            use_checkpoints=False,
            show_progress=False,
            n_workers=1,
            fill_missing_data=False,
            dynamic_position_sizing=True,
        ),
    )
    return next(results.iter_backtests()), study.name


def main():
    """把 CSV 路径及研究参数转交 run_research，并输出精简 JSON 指标。

    参数包含周期、初始资金、预热、费用与可选样本起止时间，便于固定输入复现
    不同研究区间。输出明确标记 original_long_only，只列交易数量、净收益、
    回撤和夏普等摘要，详细成交仍保存在返回结果及框架研究产物中。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--symbol", default="SPY.US/USD")
    parser.add_argument("--time-frame", default="1h")
    parser.add_argument("--capital", type=float, default=2000)
    parser.add_argument("--warmup-window", type=int, default=300)
    parser.add_argument("--fee-percentage", type=float, default=0.01)
    parser.add_argument("--slippage-percentage", type=float, default=0.02)
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = vars(parser.parse_args())
    csv_path = args.pop("csv")
    result, name = run_research(csv_path, **args)
    fields = (
        "number_of_trades",
        "number_of_trades_closed",
        "total_net_gain",
        "total_net_gain_percentage",
        "max_drawdown",
        "sharpe_ratio",
    )
    print(
        json.dumps(
            {
                "study": name,
                "variant": "original_long_only",
                "metrics": [
                    {field: getattr(m, field) for field in fields}
                    for m in result.get_all_backtest_metrics()
                ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
