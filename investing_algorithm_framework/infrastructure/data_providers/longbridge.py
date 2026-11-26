"""US candle and quote subscriptions from a shared OAuth SDK connection."""

from datetime import datetime, time, timedelta, timezone

import pandas as pd
import polars as pl

from investing_algorithm_framework.domain import (
    DataProvider,
    DataType,
    OperationalException,
    TimeFrame,
    convert_polars_to_pandas,
)
from investing_algorithm_framework.infrastructure.data_providers\
    .ohlcv_base import OHLCVDataProviderBase
from investing_algorithm_framework.infrastructure.longbridge import (
    NEW_YORK,
    TIMEFRAME_PERIODS,
    candle_completed,
    latest_completed_us_bar as latest_completed_us_bar,
    sdk,
    timeframe_period,
    us_calendar,
    us_session_close,
    us_symbol,
)


_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def candle_frame(candles, *, pandas=False):
    """将 SDK K 线整理为按 UTC 时间升序排列、每个时间戳仅一行的行情表。"""
    timestamps, values = [], []
    for candle in candles:
        timestamps.append(candle.timestamp.astimezone(timezone.utc))
        values.append((
            float(candle.open), float(candle.high), float(candle.low),
            float(candle.close), float(candle.volume),
        ))
    if pandas:
        index = pd.DatetimeIndex(
            timestamps, dtype="datetime64[us, UTC]", name="Datetime"
        )
        if index.is_unique:
            frame = pd.DataFrame(values, index=index, columns=_OHLCV_COLUMNS,
                                 dtype=float)
            return frame if index.is_monotonic_increasing else frame.sort_index()

    # 重复时间戳沿用 Polars 行情路径的去重规则，两种格式获得相同的处理。
    frame = pl.DataFrame(
        values, schema={name: pl.Float64 for name in _OHLCV_COLUMNS},
        orient="row",
    ).insert_column(
        0, pl.Series("Datetime", timestamps, dtype=pl.Datetime("us", "UTC"))
    )
    frame = frame.unique(subset=["Datetime"]).sort("Datetime")
    return convert_polars_to_pandas(frame) if pandas else frame


class LongbridgeOHLCVDataProvider(OHLCVDataProviderBase):
    market_name = "LONGBRIDGE"
    data_provider_identifier = "longbridge_ohlcv"
    timeframe_map = TIMEFRAME_PERIODS

    def __init__(self, client=None, include_incomplete=False, **kwargs):
        super().__init__(**kwargs)
        self.client = client
        self.include_incomplete = include_incomplete

    def get_data(self, date=None, start_date=None, end_date=None, save=False):
        if self.symbol is None or self.time_frame is None:
            raise OperationalException("Symbol and time_frame are required")
        if start_date is not None or end_date is not None:
            return super().get_data(date, start_date, end_date, save)
        candles = self.client.get_candlesticks(
            self.symbol,
            self.time_frame,
            count=self.window_size or 60,
            include_incomplete=self.include_incomplete,
        )
        aware_cutoff = isinstance(date, datetime) and date.utcoffset() is not None
        if aware_cutoff:
            date = date.astimezone(timezone.utc)
        if self.pandas and not save and (date is None or aware_cutoff):
            frame = candle_frame(candles, pandas=True)
            return frame if date is None else frame.loc[:date]

        frame = candle_frame(candles)
        if date is not None:
            frame = frame.filter(pl.col("Datetime") <= date)
        if save:
            end = date or datetime.now(timezone.utc)
            storage = self.get_storage_directory()
            if storage is None:
                raise OperationalException("Storage directory is not set")
            self._save_data_to_storage(
                self.symbol,
                self.time_frame,
                self._create_start_date(end),
                end,
                frame,
                storage,
            )
        return convert_polars_to_pandas(frame) if self.pandas else frame

    def _validate_symbol(self, data_source):
        us_symbol(data_source.symbol)
        return True

    def copy(self, data_source):
        provider = super().copy(data_source)
        provider.client = self.client
        provider.include_incomplete = self.include_incomplete
        return provider

    def _create_start_date(self, end_date):
        minutes = self.time_frame.amount_of_minutes
        return end_date - timedelta(
            minutes=(self.window_size or 60) * minutes * 3,
            days=10,
        )

    def _storage_file_suffix(self):
        # Subscription bars are unadjusted; avoid mixing old forward-adjusted
        # CSV caches with the same symbol/timeframe.
        return "longbridge_no_adjust"

    def _download_ohlcv(self, symbol, time_frame, start_date, end_date):
        api = sdk()
        security = us_symbol(symbol)
        name, period = timeframe_period(time_frame)
        start = start_date.astimezone(NEW_YORK).date()
        end = end_date.astimezone(NEW_YORK).date()
        result = []
        intraday = TimeFrame.from_value(name).amount_of_minutes < 1440
        while start <= end:
            # A US regular session has <=390 one-minute bars; stay below
            # the history endpoint's per-request limit for intraday data.
            chunk_end = (
                start
                if intraday
                else min(
                    start + timedelta(days=300),
                    end,
                )
            )
            candles = self.client.quote.history_candlesticks_by_date(
                security,
                period,
                api.AdjustType.NoAdjust,
                start,
                chunk_end,
            )
            for candle in candles:
                stamp = candle.timestamp.astimezone(timezone.utc)
                if start_date <= stamp <= end_date and candle_completed(
                    candle, name, end_date, client=self.client
                ):
                    result.append(candle)
            start = chunk_end + timedelta(days=1)
        return candle_frame(result)


class LongbridgeTickerDataProvider(DataProvider):
    data_type = DataType.TICKER
    data_provider_identifier = "longbridge_ticker"

    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def has_data(self, data_source, start_date=None, end_date=None):
        if data_source.market != "LONGBRIDGE" or not DataType.TICKER.equals(
            data_source.data_type
        ):
            return False
        us_symbol(data_source.symbol)
        return True

    def copy(self, data_source):
        return type(self)(
            self.client,
            symbol=data_source.symbol,
            market=data_source.market,
            config=self.config,
        )

    def get_data(self, date=None, start_date=None, end_date=None, save=False):
        security = us_symbol(self.symbol)
        quote = self.client.get_quote(security)
        depth = self.client.get_depth(security)
        bid = depth.bids[0].price if depth.bids else None
        ask = depth.asks[0].price if depth.asks else None
        return {
            "symbol": self.symbol,
            "market": "LONGBRIDGE",
            "datetime": quote.timestamp,
            "last": float(quote.last_done),
            "bid": float(bid) if bid else None,
            "ask": float(ask) if ask else None,
            "volume": quote.volume,
        }

    def prepare_backtest_data(self, *args, **kwargs):
        pass

    def get_backtest_data(self, *args, **kwargs):
        return None

    def get_number_of_data_points(self, start_date, end_date):
        return 1

    def get_missing_data_dates(self, start_date, end_date):
        return []

    def get_data_source_file_path(self):
        return None


def is_us_regular_session(client, now=None, for_entry=True):
    now = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    day = now.date()
    calendar = us_calendar(client, day, day)
    close = us_session_close(calendar, day)
    if close is None:
        return False
    if not for_entry:
        return time(9, 30) <= now.time() < close.time()
    # Entry buffers must not prevent reducing an existing position.
    return (
        time(9, 35)
        <= now.time()
        < (close - timedelta(minutes=5)).time()
    )
