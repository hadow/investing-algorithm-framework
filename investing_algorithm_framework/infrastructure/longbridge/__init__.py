"""供行情与交易适配器共用的 Longbridge OAuth 连接及实时行情缓存。

凭据仅保存 OAuth client ID，令牌缓存与刷新交给 SDK 管理；行情和交易上下文
在首次使用时创建。报价、盘口及各周期 K 线持续订阅，策略读取共享缓存，避免
每轮重复订阅和拉取快照。独立选股进程使用 HTTP 拉取，避免占用第二条行情
长连接；交易上下文统一限速，订单提交及交易开关校验仍由订单适配器负责。
"""

import importlib
import logging
import os
from pathlib import Path
import re
import time
import webbrowser
from functools import cached_property, lru_cache
from datetime import datetime, timedelta, timezone
from threading import RLock
from zoneinfo import ZoneInfo

from investing_algorithm_framework.domain import (
    ImproperlyConfigured,
    MarketCredential,
    OperationalException,
    RESOURCE_DIRECTORY,
    TimeFrame,
)

# 框架周期字符串映射到 SDK 枚举名称；使用名称可将可选 SDK 的导入延迟到调用时。
TIMEFRAME_PERIODS = {
    "1m": "Min_1",
    "2m": "Min_2",
    "3m": "Min_3",
    "5m": "Min_5",
    "10m": "Min_10",
    "15m": "Min_15",
    "20m": "Min_20",
    "30m": "Min_30",
    "1h": "Min_60",
    "2h": "Min_120",
    "4h": "Min_240",
    "1d": "Day",
    "1W": "Week",
    "1M": "Month",
    "1Y": "Year",
}
# 美股周期边界按交易所本地时间解释，由时区库处理冬夏令时切换。
NEW_YORK = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)

# 历史 K 线接口每 30 秒最多 60 次。错开客户端补拉，并给实时确认留出传输时间。
_CONFIRMATION_GRACE = timedelta(seconds=10)
_BACKFILL_INTERVAL = 0.55
_BACKFILL_COOLDOWN = 60


def _quote_rate_limited(exc):
    message = str(exc).lower()
    return getattr(exc, "code", None) in (429, 301606) or any(
        text in message for text in ("request rate limit", "too many requests")
    )


def _quote_request(method, *args):
    """对行情读取、订阅及退订的限流错误执行有次数上限的退避重试。

    HTTP 429 与 SDK 限流错误最多调用4次，重试前分别等待1、2、4秒；
    其他错误及最后一次失败直接抛出。
    订单请求不使用本函数，避免券商已收单但响应丢失时重复提交。
    """
    for attempt in range(4):
        try:
            return method(*args)
        except Exception as exc:
            if not _quote_rate_limited(exc) or attempt == 3:
                raise
            time.sleep(2**attempt)


def timeframe_period(time_frame):
    """统一框架周期表示，返回标准周期字符串及对应的 SDK Period 枚举。

    行情订阅、历史读取和退订共用映射，确保相同周期使用一致的缓存键与参数。
    不支持的周期在调用 SDK 行情接口前报配置错误。
    """
    value = TimeFrame.from_value(time_frame).value
    if value not in TIMEFRAME_PERIODS:
        raise ImproperlyConfigured(
            f"Unsupported Longbridge timeframe: {value}"
        )
    return value, getattr(sdk().Period, TIMEFRAME_PERIODS[value])


@lru_cache(maxsize=32)
def _intraday_duration(time_frame):
    """缓存固定周期的时长，避免初始化成千上万根 K 线时反复解析同一枚举。"""
    return timedelta(
        minutes=TimeFrame.from_value(time_frame).amount_of_minutes
    )


def us_calendar(client, start, end):
    """同一行情连接的所有标的共享按日期范围缓存的交易日历。"""
    cache = client.__dict__.setdefault("_us_calendar_cache", {})
    key = (start, end)
    if key not in cache:
        if len(cache) > 16:
            cache.clear()
        cache[key] = _quote_request(
            client.quote.trading_days, sdk().Market.US, start, end,
        )
    return cache[key]


def us_trading_dates(calendar):
    """半日市可能只出现在 half_trading_days 中。"""
    return set(calendar.trading_days).union(calendar.half_trading_days)


def us_session_close(calendar, day):
    """返回交易日实际的美东常规收市时刻，休市日返回 None。"""
    if day in calendar.half_trading_days:
        hour = 13
    elif day in calendar.trading_days:
        hour = 16
    else:
        return None
    return datetime(day.year, day.month, day.day, hour, tzinfo=NEW_YORK)


def latest_completed_us_bar(client, time_frame, now=None):
    """从最近的有效交易日计算已结束的最后一根日内 K 线起点，统一返回 UTC。"""
    now = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    duration = _intraday_duration(time_frame)
    if duration >= timedelta(days=1):
        raise ValueError("Runtime freshness requires an intraday timeframe")
    calendar = us_calendar(client, now.date() - timedelta(days=14), now.date())
    for day in sorted(us_trading_dates(calendar), reverse=True):
        close = us_session_close(calendar, day)
        start = close.replace(hour=9, minute=30)
        elapsed = min(now, close) - start
        bars, remainder = divmod(elapsed, duration)
        if now >= close and remainder:
            bars += 1  # 收盘时不足完整周期的尾根也已结束。
        if bars > 0:
            return (start + (bars - 1) * duration).astimezone(timezone.utc)
    raise OperationalException("No completed US session in exchange calendar")


def candle_completed(candle, time_frame, now, *, client=None):
    """按美东周期边界保守判断初始化快照或历史 K 线是否已经结束。

    日内快照按周期长度判断，只在当前末根可能因收市截短时读取共享交易日历。
    日、周、月、年线等待下一周期边界。实时推送的确认由 SDK 提供。
    """
    stamp = candle.timestamp.astimezone(NEW_YORK)
    now = now.astimezone(NEW_YORK)
    if time_frame.endswith(("m", "h")):
        if stamp + _intraday_duration(time_frame) <= now:
            return True
        if client is None or stamp.date() != now.date() or not (
            stamp.hour < 13 <= now.hour or stamp.hour < 16 <= now.hour
        ):
            return False
        return stamp <= latest_completed_us_bar(client, time_frame, now)
    day = stamp.replace(hour=0, minute=0, second=0, microsecond=0)
    if time_frame == "1d":
        end = day + timedelta(days=1)
    elif time_frame == "1W":
        end = day + timedelta(days=7 - day.weekday())
    elif time_frame == "1M":
        end = day.replace(
            year=day.year + (day.month == 12), month=day.month % 12 + 1, day=1
        )
    else:
        end = day.replace(year=day.year + 1, month=1, day=1)
    return end <= now


def sdk():
    """按需导入可选 SDK，未安装时给出包含 longbridge 扩展依赖的安装提示。

    将依赖检查放在使用入口，使框架的其他券商及离线功能可独立导入。
    """
    try:
        return importlib.import_module("longbridge.openapi")
    except ImportError as exc:
        raise ImproperlyConfigured(
            "Install investing-algorithm-framework[longbridge] first"
        ) from exc


def us_symbol(symbol):
    """把框架的 SPY.US/USD 或券商的 SPY.US 统一为大写券商代码。

    第一阶段限定美股／ETF及 USD 计价，在共享边界拒绝其他市场和货币，防止
    行情、委托及本地提交标记使用不同代码或误路由到未实现的市场。
    """
    symbol = symbol.upper()
    security, separator, currency = symbol.partition("/")
    if not re.fullmatch(r"[A-Z0-9.-]+\.US", security) or (
        separator and currency != "USD"
    ):
        raise ImproperlyConfigured(
            "Phase one supports US stocks/ETFs in USD, e.g. SPY.US/USD"
        )
    return security


def open_authorization(url):
    """作为 SDK 的 OAuth 授权回调，打印授权地址并尝试用默认浏览器打开。

    打印地址方便无桌面环境手动访问；后续授权结果与令牌管理仍由 SDK 处理。
    """
    print(f"Longbridge OAuth: open this URL to authorize:\n{url}", flush=True)
    webbrowser.open(url)


def submission_marker(config, symbol):
    """返回按标的隔离的订单提交标记路径，并确保其持久化目录存在。

    本函数只准备路径；订单执行器负责独占创建标记、记录提交结果，账户同步
    在识别对应券商订单后清理标记，以保护重启和响应不明时的重复提交边界。
    """
    if not config or not config.get(RESOURCE_DIRECTORY):
        raise ImproperlyConfigured(
            "A persistent RESOURCE_DIRECTORY is required for Longbridge orders"
        )
    directory = Path(config[RESOURCE_DIRECTORY]) / "longbridge_submissions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{us_symbol(symbol)}.json"


class LongbridgeOAuthCredential(MarketCredential):
    """框架的 Longbridge OAuth 凭据入口，仅携带 client ID；令牌由 SDK 管理。"""

    def __init__(self, client_id=None):
        """登记券商市场与可选 client ID，创建对象时不触发授权或网络请求。"""
        super().__init__(market="LONGBRIDGE")
        self.client_id = client_id

    def initialize(self):
        """优先使用显式 client ID，否则读取环境变量，去除首尾空白后校验。

        由客户端创建 SDK 配置时调用，缺失配置会在授权流程启动前报错。
        """
        self.client_id = (
            self.client_id or os.getenv("LONGBRIDGE_OAUTH_CLIENT_ID", "")
        ).strip()
        if not self.client_id:
            raise ImproperlyConfigured(
                "LONGBRIDGE_OAUTH_CLIENT_ID is required"
            )

    def __repr__(self):
        """提供只含市场名称的日志表示，避免在日志中展开凭据内容。"""
        return "LongbridgeOAuthCredential(market=LONGBRIDGE)"


class _RateLimitedTradeContext:
    """串行调度同一客户端的交易请求，不重试任何调用。

    SDK 不自动限制交易请求。统一保持请求起点至少相隔1.01秒，满足任意30秒
    最多30次及最小20毫秒间隔，并留出时间边界余量。查询、下单、撤单共用锁，
    失败请求也占用间隔；提交结果不明时仍由现有持久化标记阻止重复下单。
    本限制只覆盖本进程，独立选股进程不创建交易上下文。
    """

    def __init__(self, context):
        """包装唯一的 SDK 交易上下文，所有适配器通过此对象共享请求节奏。"""
        self._context = context
        self._lock = RLock()
        self._last_request = None

    def __getattr__(self, name):
        """保留 SDK 属性和方法接口，在实际调用前等待额度，不吞掉券商异常。"""
        value = getattr(self._context, name)
        if not callable(value):
            return value

        def request(*args, **kwargs):
            with self._lock:
                if self._last_request is not None:
                    delay = 1.01 - (time.monotonic() - self._last_request)
                    if delay > 0:
                        time.sleep(delay)
                self._last_request = time.monotonic()
                return value(*args, **kwargs)

        return request


class LongbridgeClient:
    """共享延迟创建的 SDK 上下文，并管理行情订阅、推送回调和缓存。

    默认关闭交易开关；paper=True 通过 SDK 配置请求模拟交易环境，仍需对应
    账户授权。行情和交易适配器在同一进程内复用本实例；独立选股客户端必须
    选择 quote_transport="http" 并仅调用支持 HTTP 的历史行情及交易日历接口。
    """

    def __init__(
        self,
        credential=None,
        *,
        paper=True,
        trading_enabled=False,
        quote_transport="ws",
        on_open_url=open_authorization,
    ):
        """注入凭据、账户环境和授权回调，初始化锁及缓存，不立即连接券商。

        订阅锁串行化订阅／退订，并允许 get_* 嵌套调用订阅入口；缓存锁只保护
        推送与读取的共享数据。SDK 回调仅使用缓存锁，避免网络调用等待回调时死锁。
        """
        if quote_transport not in ("ws", "http"):
            raise ImproperlyConfigured(
                "quote_transport must be 'ws' or 'http'"
            )
        self.credential = credential or LongbridgeOAuthCredential()
        self.paper = paper
        self.trading_enabled = trading_enabled
        self.quote_transport = quote_transport
        self.on_open_url = on_open_url
        # 网络操作可持有订阅锁，但必须先释放回调所需的缓存锁。
        self._subscription_lock = RLock()
        self._cache_lock = RLock()
        self._callbacks_set = False
        # 显式 Quote/Depth 订阅与初始完整盘口；推送到达后改读 SDK 合并盘口。
        self._quote_symbols = set()
        self._quotes = {}
        self._depth_seed = {}
        self._depth_seen = set()
        # K 线按 (标的, 周期) 隔离；每根保存 (SDK K线对象, 是否已确认)。
        self._candle_subscriptions = set()
        self._candles = {}
        self._backfill_retry_at = {}
        self._next_backfill_at = 0.0
        self._backfill_paused_until = 0.0

    @cached_property
    def config(self):
        """首次访问时完成凭据校验和 OAuth 构建，并缓存共用的 SDK 配置。

        SDK 负责令牌缓存与刷新；启用实时 K 线推送模式，使形成中和已确认事件
        均能进入缓存，再由读取端决定是否包含未完成 K 线。
        """
        self.credential.initialize()
        api = sdk()
        self.oauth = api.OAuthBuilder(self.credential.client_id).build(
            self.on_open_url
        )
        return api.Config.from_oauth(
            self.oauth,
            enable_papertrading=self.paper,
            enable_print_quote_packages=False,
            push_candlestick_mode=api.PushCandlestickMode.Realtime,
            quote_transport=(
                api.QuoteTransport.Http
                if self.quote_transport == "http"
                else api.QuoteTransport.WebSocket
            ),
        )

    @cached_property
    def quote(self):
        """首次使用行情时创建 QuoteContext，同一客户端的所有标的和周期共用。"""
        return sdk().QuoteContext(self.config)

    @cached_property
    def trade(self):
        """首次访问时创建并包装 TradeContext，使所有适配器共用同一交易限速器。"""
        return _RateLimitedTradeContext(sdk().TradeContext(self.config))

    def _ensure_callbacks(self):
        """在订阅前统一注册三类推送回调；由持有订阅锁的入口保证仅注册一次。

        回调只更新共享缓存，策略逻辑和订单操作由框架运行线程处理。
        """
        if self.quote_transport == "http":
            raise ImproperlyConfigured(
                "HTTP-only Longbridge clients cannot subscribe; "
                "use history APIs"
            )
        if not self._callbacks_set:
            self.quote.set_on_quote(self._on_quote)
            self.quote.set_on_depth(self._on_depth)
            self.quote.set_on_candlestick(self._on_candlestick)
            self._callbacks_set = True

    def _on_quote(self, symbol, event):
        """接收报价推送，仅更新仍在订阅且时间不早于缓存的事件。

        退订后的迟到推送直接忽略；比较前统一到 UTC，兼容 SDK 与本地快照的
        时间表示，防止乱序事件或初始化旧快照覆盖新报价。
        """
        with self._cache_lock:
            if symbol not in self._quote_symbols:
                return
            previous = self._quotes.get(symbol)
            if previous is None or (
                event.timestamp.astimezone(timezone.utc)
                >= previous.timestamp.astimezone(timezone.utc)
            ):
                self._quotes[symbol] = event

    def _on_depth(self, symbol, event):
        """记录该标的已收到盘口增量，后续读取转向 SDK 的完整合并盘口。

        事件本身可能只含变动档位，不能直接当作完整买卖盘替换初始快照。
        """
        with self._cache_lock:
            if symbol in self._quote_symbols:
                self._depth_seen.add(symbol)

    def _on_candlestick(self, symbol, event):
        """把推送周期映射回框架周期，按 K 线起点更新对应标的的周期缓存。

        同一根已确认 K 线不允许被迟到的未确认事件回退；每个缓存最多1000根，
        超限删除最早时间，避免持续运行时无限增长。无对应缓存的事件直接忽略。
        """
        name = next(
            (
                name
                for name, attr in TIMEFRAME_PERIODS.items()
                if event.period == getattr(sdk().Period, attr)
            ),
            None,
        )
        key = (symbol, name)
        with self._cache_lock:
            candles = self._candles.get(key)
            if candles is None:
                return
            stamp = event.candlestick.timestamp.astimezone(timezone.utc)
            previous = candles.get(stamp)
            if previous is not None and previous[1] and not event.is_confirmed:
                return
            candles[stamp] = (event.candlestick, event.is_confirmed)
            if len(candles) > 1000:
                del candles[min(candles)]

    def subscribe_quotes(self, symbol):
        """幂等订阅报价与盘口，并在缺少推送时各读取一次初始快照。

        先登记标的再请求 SDK，让订阅过程中到达的推送也能入缓存；失败时回滚
        登记与相关缓存，保留下一次重试机会。报价快照经同一回调处理乱序比较。
        """
        symbol = us_symbol(symbol)
        api = sdk()
        with self._subscription_lock:
            self._ensure_callbacks()
            if symbol not in self._quote_symbols:
                with self._cache_lock:
                    self._quote_symbols.add(symbol)
                try:
                    _quote_request(
                        self.quote.subscribe,
                        [symbol],
                        [api.SubType.Quote, api.SubType.Depth],
                    )
                except Exception:
                    with self._cache_lock:
                        self._quote_symbols.discard(symbol)
                        self._quotes.pop(symbol, None)
                        self._depth_seen.discard(symbol)
                    raise
            # 盘外可能没有首条推送，初始快照让启动读取不必等待下一笔行情。
            with self._cache_lock:
                needs_quote = symbol not in self._quotes
            if needs_quote:
                self._on_quote(
                    symbol, _quote_request(self.quote.quote, [symbol])[0]
                )
            if symbol not in self._depth_seed:
                self._depth_seed[symbol] = _quote_request(
                    self.quote.depth, symbol
                )

    def get_quote(self, symbol):
        """确保订阅存在后返回最新缓存报价，重复读取不再主动轮询报价快照。

        返回 SDK 对象供调用方只读使用；交易状态与报价新鲜度由策略和执行器校验。
        """
        symbol = us_symbol(symbol)
        with self._subscription_lock:
            self.subscribe_quotes(symbol)
            with self._cache_lock:
                return self._quotes[symbol]

    def get_depth(self, symbol):
        """返回首次推送前的盘口快照，或 SDK 根据增量维护的最新完整盘口。

        收到增量后即以 realtime_depth 为准，即使其返回空盘也不退回旧快照，
        避免已经撤销的档位重新成为可交易价格。
        """
        symbol = us_symbol(symbol)
        with self._subscription_lock:
            self.subscribe_quotes(symbol)
            with self._cache_lock:
                seen = symbol in self._depth_seen
            # SDK 本地存储负责合并增量档位，应用层不自行拼接盘口。
            if seen:
                return self.quote.realtime_depth(symbol)
            return self._depth_seed[symbol]

    def subscribe_candlesticks(self, symbol, time_frame):
        """按 (标的, 周期) 幂等订阅常规交易时段 K 线，并合并初始历史快照。

        先创建缓存接收订阅过程中的推送，按请求发起时的交易日历标记日内快照。
        已到达的推送优先于初始快照；订阅失败则删除临时缓存，允许后续重试。
        """
        symbol = us_symbol(symbol)
        name, period = timeframe_period(time_frame)
        key = (symbol, name)
        with self._subscription_lock:
            self._ensure_callbacks()
            if key in self._candle_subscriptions:
                return
            with self._cache_lock:
                self._candles[key] = {}
            # 用请求开始时间判断初始快照，跨收盘返回的请求仍可能携带收盘前的值。
            as_of = datetime.now(timezone.utc)
            try:
                initial = _quote_request(
                    self.quote.subscribe_candlesticks,
                    symbol,
                    period,
                    sdk().TradeSessions.Intraday,
                )
            except Exception:
                with self._cache_lock:
                    self._candles.pop(key, None)
                raise
            cutoff = (
                self._expected_candle(name, as_of)
                if initial and name.endswith(("m", "h")) else None
            )
            with self._cache_lock:
                candles = self._candles[key]
                for candle in initial:
                    # 订阅期间收到的推送优先，初始快照不能覆盖同一根的新值。
                    stamp = candle.timestamp.astimezone(timezone.utc)
                    confirmed = (
                        stamp <= cutoff if cutoff is not None
                        else candle_completed(candle, name, as_of)
                    )
                    candles.setdefault(stamp, (candle, confirmed))
                for stamp in sorted(candles)[:-1000]:
                    del candles[stamp]
            self._candle_subscriptions.add(key)

    def _expected_candle(self, name, as_of):
        """日历暂不可用时保留已有确认行情，冷却期后再尝试恢复。"""
        if time.monotonic() < self._backfill_paused_until:
            return None
        try:
            return latest_completed_us_bar(self, name, as_of)
        except Exception as exc:
            self._backfill_paused_until = time.monotonic() + _BACKFILL_COOLDOWN
            logger.warning("Longbridge candle calendar unavailable: %s", exc)
            return None

    def _backfill_candles(self, key, period):
        """给迟到推送留出宽限，再按缺失 K 线的交易日从同一个 QuoteContext 补拉。

        读线程持有订阅锁，限速等待和网络请求都在缓存锁外。空响应、请求失败
        或更晚的形成中 K 线都不能确认本地旧值。优先保留并发到达的已确认推送。
        """
        if time.monotonic() < self._backfill_retry_at.get(key, 0):
            return
        symbol, name = key
        expected = self._expected_candle(
            name, datetime.now(timezone.utc) - _CONFIRMATION_GRACE,
        )
        if expected is None:
            return
        with self._cache_lock:
            candles = self._candles[key]
            target = expected
            if candles.get(expected, (None, False))[1]:
                target = max(
                    (stamp for stamp, (_, confirmed) in candles.items()
                     if not confirmed and stamp < expected),
                    default=None,
                )
                if target is None:
                    return

        delay = self._next_backfill_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        with self._cache_lock:
            if candles.get(target, (None, False))[1]:
                return
        started = time.monotonic()
        self._next_backfill_at = started + _BACKFILL_INTERVAL
        self._backfill_retry_at[key] = started + _BACKFILL_COOLDOWN
        day = target.astimezone(NEW_YORK).date()
        try:
            history = self.quote.history_candlesticks_by_date(
                symbol, period, sdk().AdjustType.NoAdjust, day, day,
                sdk().TradeSessions.Intraday,
            )
        except Exception as exc:
            if _quote_rate_limited(exc):
                self._backfill_paused_until = (
                    time.monotonic() + _BACKFILL_COOLDOWN
                )
            logger.warning(
                "Longbridge candle backfill %s %s %s: %s", symbol, name, day, exc,
            )
            return

        restored = 0
        with self._cache_lock:
            for candle in history:
                stamp = candle.timestamp.astimezone(timezone.utc)
                if stamp > expected or stamp.astimezone(NEW_YORK).date() != day:
                    continue
                if not candles.get(stamp, (None, False))[1]:
                    candles[stamp] = (candle, True)
                    restored += 1
            for stamp in sorted(candles)[:-1000]:
                del candles[stamp]
        if restored:
            logger.info(
                "Longbridge restored %d %s candles for %s on %s",
                restored, name, symbol, day,
            )

    def get_candlesticks(
        self, symbol, time_frame, count=60, *, include_incomplete=False
    ):
        """确保持续订阅后，按时间升序返回最近 count 根缓存 K 线。

        默认只返回已确认的 K 线，include_incomplete 可包含形成中的最新一根。
        日内缺确认时限速拉取实际成交 K 线。count 限定1至999，有效数据不足时
        返回实际数量。
        """
        if not isinstance(count, int) or not 1 <= count <= 999:
            raise ImproperlyConfigured("Live candle count must be 1..999")
        symbol = us_symbol(symbol)
        name, period = timeframe_period(time_frame)
        with self._subscription_lock:
            self.subscribe_candlesticks(symbol, name)
            if name.endswith(("m", "h")):
                self._backfill_candles((symbol, name), period)
            with self._cache_lock:
                entries = self._candles[(symbol, name)]
                ordered = sorted(entries.items())
                return [
                    candle
                    for _, (candle, confirmed) in ordered
                    if include_incomplete or confirmed
                ][-count:]

    def retain_subscriptions(self, symbols):
        """保留指定标的，释放其余标的的报价、盘口及所有周期 K 线订阅。

        调用方应传入候选股、已有持仓和有效委托的并集，避免停止持仓风控行情。
        每次 SDK 退订成功后才清理对应本地记录，失败记录保留以供后续重试。
        """
        retained = {us_symbol(s) for s in symbols}
        with self._subscription_lock:
            removed = self._quote_symbols - retained
            if removed:
                _quote_request(
                    self.quote.unsubscribe,
                    sorted(removed),
                    [sdk().SubType.Quote, sdk().SubType.Depth],
                )
                with self._cache_lock:
                    self._quote_symbols.difference_update(removed)
                    for symbol in removed:
                        self._quotes.pop(symbol, None)
                        self._depth_seed.pop(symbol, None)
                        self._depth_seen.discard(symbol)
            for symbol, name in list(self._candle_subscriptions):
                if symbol in retained:
                    continue
                _quote_request(
                    self.quote.unsubscribe_candlesticks,
                    symbol,
                    timeframe_period(name)[1],
                )
                self._candle_subscriptions.remove((symbol, name))
                self._backfill_retry_at.pop((symbol, name), None)
                with self._cache_lock:
                    self._candles.pop((symbol, name), None)

    def close(self):
        """尽力释放所有共享订阅，保留 SDK 上下文与 OAuth，客户端可再次订阅。

        单项失败仍继续清理其他项，最后抛出首个错误；失败项的本地记录保留，
        再次调用可继续清理。重复调用已成功清理的客户端不会再次退订这些项目。
        """
        with self._subscription_lock:
            errors = []
            # 先释放显式报价，最后一项 K 线退订可一并释放 SDK 的隐式 Quote/Trade 订阅。
            for symbol in list(self._quote_symbols):
                try:
                    _quote_request(
                        self.quote.unsubscribe,
                        [symbol],
                        [sdk().SubType.Quote, sdk().SubType.Depth],
                    )
                    with self._cache_lock:
                        self._quote_symbols.remove(symbol)
                        self._quotes.pop(symbol, None)
                        self._depth_seed.pop(symbol, None)
                        self._depth_seen.discard(symbol)
                except Exception as exc:
                    errors.append(exc)
            for symbol, name in list(self._candle_subscriptions):
                try:
                    _quote_request(
                        self.quote.unsubscribe_candlesticks,
                        symbol,
                        timeframe_period(name)[1],
                    )
                    self._candle_subscriptions.remove((symbol, name))
                    self._backfill_retry_at.pop((symbol, name), None)
                    with self._cache_lock:
                        self._candles.pop((symbol, name), None)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
