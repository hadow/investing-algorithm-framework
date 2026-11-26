"""独立 HTTP 选股的数据源、交易日和日线资格计算；不订阅实时行情或下单。"""

from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from ...domain.utils.wilder import rma as rma, wilder_atr
from ..data_providers.longbridge import candle_frame
from . import (
    NEW_YORK,
    _quote_request,
    candle_completed,
    sdk,
    timeframe_period,
    us_symbol,
    us_trading_dates,
)
from .runtime import atomic_json


def cli_json(arguments):
    """执行只读 Longbridge CLI 请求，并把标准输出解析为 Python 数据。

    成分股和财报信息由 CLI 提供，调用方仅传子命令及业务参数，这里统一添加
    JSON 输出格式。使用参数列表而非 shell 拼接，并限制90秒超时；命令失败、
    超时和响应解析失败向上传播，由选股主循环保留旧快照并标记刷新失败。
    """
    result = subprocess.run(
        ["longbridge", *arguments, "--format", "json"],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
        env={**os.environ, "LONGBRIDGE_QUOTE_TRANSPORT": "http"},
    )
    return json.loads(result.stdout)


def session_window(client, now):
    """按美股交易日历返回目标交易日、日线截止日和财报暂停窗口。

    先把 now 转为美东日期，再查询真实交易日，避免把周末或节假日当作开市日。
    target 为今天或之后的第一个交易日；previous 取今天以前且早于 target 的
    最近交易日，使日线筛选使用完整交易日数据。future 含 target 及其后两个
    交易日，供财报过滤使用。交易日历缺失或覆盖不足时中止本轮扫描。
    """
    today = now.astimezone(NEW_YORK).date()
    response = _quote_request(
        client.quote.trading_days,
        sdk().Market.US,
        today - timedelta(days=14),
        today + timedelta(days=14),
    )
    sessions = sorted(us_trading_dates(response))
    target = next(d for d in sessions if d >= today)
    previous = max(d for d in sessions if d < target and d < today)
    future = [d for d in sessions if d >= target][:3]
    if len(future) != 3:
        raise ValueError("Insufficient US trading calendar")
    return target, previous, future


def normalize_daily(frame, as_of):
    """复制并校验单个日线表，确保筛选只使用指定截止日以前的有效样本。

    接受 Datetime 列或时间索引，统一转 UTC；截断时再按美东日期比较，避免
    把当天未完成日线或未来数据带入 EMA 和涨幅计算。时间须唯一递增，末根须
    覆盖 as_of，Close、Volume、DollarVolume 须有限且满足价格／数量约束。
    返回经过截断的表；不补造休市日或缺失价格，过期数据交由上层记录为不可用。
    """
    frame = frame.copy()
    if "Datetime" in frame:
        frame = frame.set_index("Datetime")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Daily candles require timestamps")
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(
            "Daily candles must have unique increasing timestamps"
        )
    # 比较下一美东午夜，避免逐根创建 Python date，同时正确处理冬夏令时边界。
    cutoff = pd.Timestamp(as_of + timedelta(days=1), tz=NEW_YORK)
    frame = frame.loc[frame.index < cutoff]
    if frame.empty or frame.index[-1].astimezone(NEW_YORK).date() != as_of:
        raise ValueError("Daily candles are stale or empty")
    prices = frame[["Close", "Volume", "DollarVolume"]].to_numpy(float)
    if (
        not np.isfinite(prices).all()
        or (prices[:, 0] <= 0).any()
        or (prices[:, 1:] < 0).any()
    ):
        raise ValueError("Invalid daily prices or volumes")
    return frame


def rank_candidates(
    histories,
    benchmark,
    as_of,
    rules,
    blocked=(),
    *,
    require_relative_strength=True,
):
    """用历史日线完成纯数据筛选与排序，返回通过列表和带原因的剔除列表。

    benchmark 为 SPY，histories 为标的到日线表的映射；两者都须有至少250根
    完整日线，最近61根日期须对齐，才能公平比较60个交易日的涨幅。
    保留价格、流动性、EMA50趋势和财报门槛，并要求20日及60日均跑赢 SPY。
    收盘偏离EMA20超过 max_extension_atr 倍日ATR14时跳过，避免追逐异常拉升。
    用两个周期超额收益的均值除以60日年化波动率排序；分母至少10%，防止
    近乎不变的价格将得分放大。该评分是候选排序工具，不代表预期收益或夏普比率。
    基准不可用则整轮失败，单只数据异常则仅剔除该股。本函数不访问账户、不下单，
    也不决定最终 top_n；后续 screen 还会检查小时 ATR 下的整股可执行性。
    """
    # Chandelier 将相对强弱仅用于排序；旧策略仍保留两个周期的硬门槛。
    benchmark = normalize_daily(benchmark, as_of)
    if len(benchmark) < 250:
        raise ValueError(
            "Benchmark requires at least 250 completed daily bars"
        )
    benchmark_return = float(
        benchmark.Close.iloc[-1] / benchmark.Close.iloc[-61] - 1
    )
    benchmark_return_20 = float(
        benchmark.Close.iloc[-1] / benchmark.Close.iloc[-21] - 1
    )
    accepted, rejected = [], []
    for symbol, history in histories.items():
        symbol = us_symbol(symbol)
        try:
            frame = normalize_daily(history, as_of)
            if len(frame) < 250:
                raise ValueError("less_than_250_daily_bars")
            if not frame.index[-61:].equals(benchmark.index[-61:]):
                raise ValueError("benchmark_dates_do_not_match")
            close = float(frame.Close.iloc[-1])
            # 成交额沿用原始 turnover；复权价格乘原始成交量会改变流动性口径。
            dollar_volume = float(frame.DollarVolume.iloc[-20:].mean())
            ema = frame.Close.ewm(span=50, adjust=False, min_periods=50).mean()
            relative = float(
                close / frame.Close.iloc[-61] - 1 - benchmark_return
            )
            relative_20 = float(
                close / frame.Close.iloc[-21] - 1 - benchmark_return_20
            )
            prices = frame[["High", "Low", "Close"]].to_numpy(float)
            if (
                not np.isfinite(prices).all()
                or (prices <= 0).any()
                or (prices[:, 0] < prices[:, 1]).any()
            ):
                raise ValueError("invalid_daily_ohlc")
            high, low, closes = prices.T
            atr = float(wilder_atr(high, low, closes, 14)[-1])
            volatility = float(
                frame.Close.pct_change(fill_method=None).iloc[-60:].std()
                * np.sqrt(252)
            )
            if (
                not math.isfinite(atr)
                or atr <= 0
                or not math.isfinite(volatility)
                or volatility <= 0
            ):
                raise ValueError("invalid_daily_atr_or_volatility")
            ema20 = float(
                frame.Close.ewm(span=20, adjust=False, min_periods=20)
                .mean()
                .iloc[-1]
            )
            extension = (close - ema20) / atr
            score = (relative_20 + relative) / 2 / max(volatility, 0.10)
            reasons = []
            if close < rules.min_price:
                reasons.append("price_below_minimum")
            if (
                not math.isfinite(dollar_volume)
                or dollar_volume < rules.min_dollar_volume
            ):
                reasons.append("insufficient_dollar_volume")
            if not (close > ema.iloc[-1] and ema.iloc[-1] > ema.iloc[-11]):
                reasons.append("daily_trend_not_up")
            if require_relative_strength and relative <= 0:
                reasons.append("not_stronger_than_spy")
            if require_relative_strength and relative_20 <= 0:
                reasons.append("recent_relative_strength_not_positive")
            if extension > rules.max_extension_atr:
                reasons.append("extended_above_ema20")
            if symbol in blocked:
                reasons.append("earnings_within_two_sessions")
            row = {
                "symbol": symbol,
                "close_adjusted": close,
                "dollar_volume_20": dollar_volume,
                "ema50": float(ema.iloc[-1]),
                "relative_return_60": relative,
                "relative_return_20": relative_20,
                "volatility_60": volatility,
                "daily_atr_14": atr,
                "ema20_extension_atr": extension,
                "selection_score": score,
            }
            if reasons:
                rejected.append({**row, "reasons": reasons})
            else:
                accepted.append(row)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"symbol": symbol, "reasons": [str(exc)]})
    accepted.sort(
        key=lambda r: (
            -r["selection_score"],
            -r["relative_return_60"],
            r["symbol"],
        )
    )
    return accepted, rejected


def candidate_correlations(histories, ranked, as_of):
    """计算合格股票最近60个交易日收益率的相关矩阵，不补齐缺失日期或未来数据。

    日线排名已验证最近61根日期与基准一致；这里再次按截止日截断，使用恰好
    60个收益样本。常数或不足样本导致的未知相关性由 screen 保守剔除。
    """
    returns = {
        row["symbol"]: normalize_daily(histories[row["symbol"]], as_of)
        .Close.pct_change(fill_method=None)
        .iloc[-60:]
        for row in ranked
    }
    return pd.DataFrame(returns).corr(min_periods=60)


def earnings_symbols(payload, dates, *, event_types=("report",)):
    """从一批 CLI 财报事件中提取落在暂停窗口内的美股代码。

    dates 是目标交易日及未来两个交易日。优先读取事件的美东 local_date，
    缺失时才从时间戳换算；响应分组日期可能属于查看者时区，不能直接用于判断。
    响应结构不符或 next_date 表示未取完时抛错，避免把不完整日历当作没有财报。
    盘前扫描读取 report（日程）。历史研究也可纳入 financial（已披露业绩）重建当时的窗口。
    """
    if not isinstance(payload, dict) or not isinstance(
        payload.get("list"), list
    ):
        raise ValueError("Unrecognized Longbridge earnings response")
    if payload.get("next_date"):
        raise ValueError(
            "Earnings response is truncated; do not assume no events"
        )
    blocked = set()
    for group in payload["list"]:
        for event in group["infos"]:
            if event.get("type") not in event_types:
                continue
            symbol = us_symbol(event["symbol"])
            # 分组日期可能已转成查看者时区；财报限制须按事件对应的美东日期判断。
            local = event.get("ext", {}).get("local_date")
            if local:
                event_date = date.fromisoformat(local)
            else:
                event_date = (
                    datetime.fromtimestamp(
                        int(event["datetime"]), timezone.utc
                    )
                    .astimezone(NEW_YORK)
                    .date()
                )
            if event_date in dates:
                blocked.add(symbol)
    return blocked


class SelectionSource:
    """封装选股所需的 CLI、SDK 和按日缓存，使筛选计算与外部数据获取分开。"""

    def __init__(
        self, client, directory, run_cli=cli_json, *, base_directory=None
    ):
        """注入共享只读行情客户端、缓存根目录和可替换的 CLI 请求函数。

        run_cli 可在测试中替换为固定响应，验证筛选流程时无需连接外部服务。
        记录历史请求启动时间，用于后续限速；构造本身不立即发起行情请求。
        基础池目录可以独立保存，使共享日线缓存不改变另一策略的基础池。
        """
        self.client, self.directory, self.run_cli = (
            client,
            Path(directory),
            run_cli,
        )
        self.base_directory = Path(base_directory or directory)
        self._last_history_request = -math.inf

    def history(self, *arguments):
        """对幂等历史行情请求限速，并仅对明确的请求限流做有界重试。

        请求启动间隔至少0.35秒，发生限流时按1、2、4秒退避，合计最多尝试4次。
        权限、标的等其他错误立即抛出，避免重试掩盖真实问题；调用者决定是剔除
        单只股票还是让本轮扫描失败。本封装不用于订单提交。
        """

        def request():
            delay = 0.35 - (time.monotonic() - self._last_history_request)
            if delay > 0:
                time.sleep(delay)
            self._last_history_request = time.monotonic()
            return self.client.quote.history_candlesticks_by_offset(*arguments)

        return _quote_request(request)

    def base_pool(self, target, size):
        """获取并固定目标交易日的基础池，返回去重、规范化后的美股代码列表。

        首次读取 Longbridge 的标普500成分股，按成交额取前 size 只并保存响应；
        同一目标日后续复用快照；扩大 size 时重新请求，并在验证完整后才替换
        旧缓存。数据为空、重复去重后不足请求数量时抛错，保留原缓存供原规模复用。
        这里保存的是获取时的成分信息，不能用来倒填历史股票池做无偏回测。
        """
        path = self.base_directory / f"base/{target}.json"
        payload = (
            json.loads(path.read_text()) if path.exists() else {"stocks": []}
        )
        symbols = list(
            dict.fromkeys(us_symbol(s["symbol"]) for s in payload["stocks"])
        )
        if len(symbols) < size:
            payload = self.run_cli(
                [
                    "constituent",
                    ".SPX.US",
                    "--sort",
                    "turnover",
                    "--limit",
                    str(size),
                ]
            )
            if (
                not isinstance(payload.get("stocks"), list)
                or not payload["stocks"]
            ):
                raise ValueError("Longbridge returned no S&P 500 constituents")
            symbols = list(
                dict.fromkeys(
                    us_symbol(s["symbol"]) for s in payload["stocks"]
                )
            )
            if len(symbols) < size:
                raise ValueError("Base pool has fewer symbols than requested")
            atomic_json(path, payload)
        return symbols[:size]

    def signal_candlesticks(self, symbol, time_frame, now, count=300):
        """通过 HTTP 历史接口读取信号周期，仅返回扫描时点以前已收盘的 K 线。

        使用未复权的常规交易时段数据，口径与账户实时策略一致。多取一根用于
        剔除正在形成的末根，按统一周期边界过滤未来或未完成数据，再截取所需
        长度；本入口不调用 get_candlesticks，避免隐式建立实时订阅。
        """
        name, period = timeframe_period(time_frame)
        api = sdk()
        candles = self.history(
            us_symbol(symbol),
            period,
            api.AdjustType.NoAdjust,
            False,
            count + 1,
            now.astimezone(NEW_YORK),
            api.TradeSessions.Intraday,
        )
        return [
            c for c in candles
            if candle_completed(c, name, now, client=self.client)
        ][-count:]

    def daily(self, symbol, as_of, count):
        """取得截至 as_of 的日线研究输入，区分价格复权与成交额的计算口径。

        先查按截止日和标的保存的 CSV；首次按 count 请求前复权与未复权日线。
        趋势和相对涨幅使用前复权 OHLCV，DollarVolume 直接取未复权 turnover；
        两套数据先统一 UTC 再按时间索引对齐，缺失或无效值由 normalize_daily
        拒绝。校验成功后缓存供同日重筛复用，实际交易仍使用未复权实时行情。
        SDK 可能返回无时区的本地时间，因此转换方式须与 candle_frame 保持一致。
        """
        path = self.directory / f"daily/{as_of}/{symbol}.csv"
        if path.exists():
            return pd.read_csv(path, parse_dates=["Datetime"]).set_index(
                "Datetime"
            )
        api = sdk()
        cutoff = datetime.combine(as_of, datetime.max.time(), NEW_YORK)
        adjusted = self.history(
            symbol,
            api.Period.Day,
            api.AdjustType.ForwardAdjust,
            False,
            count,
            cutoff,
            api.TradeSessions.Intraday,
        )
        raw = self.history(
            symbol,
            api.Period.Day,
            api.AdjustType.NoAdjust,
            False,
            count,
            cutoff,
            api.TradeSessions.Intraday,
        )
        frame = candle_frame(adjusted).to_pandas().set_index("Datetime")
        turnover = (
            pd.Series(
                [float(c.turnover) for c in raw],
                index=pd.to_datetime(
                    [c.timestamp.astimezone(timezone.utc) for c in raw],
                    utc=True,
                ),
            )
            .groupby(level=0)
            .last()
        )
        frame["DollarVolume"] = turnover.reindex(frame.index)
        frame = normalize_daily(frame, as_of)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index_label="Datetime")
        return frame

    def blocked_earnings(self, symbols, dates):
        """分批获取候选股财报事件，合并需要暂停新开仓的标的集合。

        CLI 单次最多传10只股票，因此按批查询同一日期窗口，并由 earnings_symbols
        统一处理事件时区和响应完整性。网络抖动只重试失败的批次。重试耗尽、
        权限错误或触及事件数量上限时向上传播，
        避免仅凭部分财报数据发布新的开仓资格；已有持仓退出由账户进程继续负责。
        """
        blocked = set()
        sessions = set(dates)
        for offset in range(0, len(symbols), 10):
            payload = self._earnings_batch(symbols[offset:offset + 10], dates[0], dates[-1])
            if sum(len(g["infos"]) for g in payload.get("list", [])) >= 1000:
                raise ValueError("Earnings event limit reached")
            blocked.update(earnings_symbols(payload, sessions))
        return blocked

    def _earnings_batch(self, symbols, start, end):
        """只读财报查询遇连接失败、超时或限流时退避 1、2 秒，最多请求三次。

        CLI 的退出码 1 同时用于网络和业务错误，必须读取 stderr 才能区分。
        失败时将窗口、股票和具体原因传给刷新日志，保持财报过滤完整。
        """
        arguments = [
            "finance-calendar", "report", "--market", "US", "--start", start.isoformat(),
            "--end", end.isoformat(), "--count", "1000",
        ]
        for symbol in symbols:
            arguments.extend(["--symbol", symbol])
        for attempt in range(3):
            try:
                return self.run_cli(arguments)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                if isinstance(exc, subprocess.TimeoutExpired):
                    detail = f"CLI timed out after {exc.timeout}s"
                    retryable = True
                else:
                    detail = (exc.stderr or "").strip() or f"CLI exited with code {exc.returncode}"
                    message = detail.lower()
                    retryable = any(marker in message for marker in (
                        "client error (connect)", "connection reset", "timed out",
                        "too many requests", "rate limit", "301606", "http 429",
                        "bad gateway", "service unavailable", "gateway timeout",
                        "http 502", "http 503", "http 504",
                    ))
                if not retryable or attempt == 2:
                    raise RuntimeError(
                        f"Longbridge finance-calendar {start}..{end} "
                        f"[{', '.join(symbols)}] (attempt {attempt + 1}): {detail}"
                    ) from exc
                time.sleep(2**attempt)
