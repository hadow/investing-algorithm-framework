# SPDX-License-Identifier: MPL-2.0
"""日线选股和信号 K 线共用的 Wilder 指标，不依赖券商或交易日历。"""

import math

import numpy as np


def rma(values, period):
    """计算 ATR 和 RSI 共用的 Wilder 递归平滑序列。

    先收集 period 个有效样本，以它们的算术平均值初始化；之后按
    (上次平滑值 × (period - 1) + 当前值) / period 递推。采用同一初始化规则，
    是移植 Pine 指标时对齐数值和穿越时点的关键，普通滚动均值不能直接替代。
    非有限输入所在位置及预热不足的位置保留 NaN，有效输入才推进平滑状态。
    返回与输入等长的数组，调用方据此保持所有指标与 K 线索引逐根对齐。
    """
    result = np.full(len(values), np.nan)
    samples = []
    previous = np.nan
    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        if math.isnan(previous):
            samples.append(value)
            if len(samples) < period:
                continue
            previous = sum(samples) / period
        else:
            previous = (previous * (period - 1) + value) / period
        result[index] = previous
    return result


def wilder_atr(high, low, close, period):
    """在已按时间排序的价格上合并日内振幅与隔根跳空，再按 Wilder 方法平滑。"""
    previous_close = np.roll(close, 1)
    if len(close):
        previous_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - previous_close),
            abs(low - previous_close),
        ),
    )
    return rma(tr, period)
