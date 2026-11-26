"""使用真实 SDK 和本地 HTTP 服务，验证选股接口不会回退到行情 WebSocket。"""

from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import Mock

import pytest

api = pytest.importorskip("longbridge.openapi")

from investing_algorithm_framework import LongbridgeClient  # noqa: E402

EXAMPLE = (Path(__file__).resolve().parents[2]
           / "examples/algorithm_examples/strategies_showcase/27_supertrend_rsi")


def _serve_http(port_writer, log_path, timestamps, throttle_first):
    # 同步 SDK 调用可能持有 GIL，HTTP 服务放到独立进程避免阻塞响应线程。
    throttled = False

    class Handler(BaseHTTPRequestHandler):
        def record(self, body):
            with open(log_path, "a") as stream:
                stream.write(json.dumps([self.path, body]) + "\n")

        def do_POST(self):
            nonlocal throttled
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.record(body)
            if throttle_first and not throttled:
                throttled = True
                self.reply(429, {"code": 429, "message": "Too Many Requests"})
                return
            if self.path == "/quote/markets/trading-days":
                data = {"trade_day": ["20261102", "20261103", "20261104", "20261105"],
                        "half_trade_day": []}
            elif self.path == "/quote/history-candlesticks":
                data = {"symbol": "SPY.US", "candlesticks": [
                    {"timestamp": str(stamp), "open": "100",
                     "high": "102", "low": "99", "close": "101",
                     "volume": "1000", "turnover": "101000", "trade_session": 0}
                    for stamp in timestamps
                ]}
            else:
                self.send_error(404)
                return
            self.reply(200, {"code": 0, "message": "success", "data": data})

        def reply(self, status, body):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.record("unexpected WebSocket")
            self.send_error(400)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port_writer.send(server.server_port)
    port_writer.close()
    server.serve_forever()


@pytest.mark.parametrize("throttle_first", [False, True])
def test_screener_calendar_and_history_use_http_without_websocket(
    monkeypatch, tmp_path, throttle_first,
):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    selection = importlib.import_module("screener")
    now = datetime(2026, 11, 3, 16, 15, tzinfo=timezone.utc)
    stamps = [
        datetime(2026, 11, 2, 19, 30, tzinfo=timezone.utc),
        datetime(2026, 11, 3, 14, 30, tzinfo=timezone.utc),
        datetime(2026, 11, 3, 15, 30, tzinfo=timezone.utc),  # 尚未收盘
        datetime(2026, 11, 3, 16, 30, tzinfo=timezone.utc),  # 未来数据
    ]
    context = get_context("spawn")
    port_reader, port_writer = context.Pipe(duplex=False)
    log_path = tmp_path / "requests.jsonl"
    server = context.Process(
        target=_serve_http,
        args=(port_writer, str(log_path), [int(s.timestamp()) for s in stamps], throttle_first),
        daemon=True,
    )
    server.start()
    port_writer.close()
    client = LongbridgeClient(quote_transport="http")
    trade_factory = Mock(side_effect=AssertionError("Screener must not create TradeContext"))
    monkeypatch.setattr(api, "TradeContext", trade_factory)
    try:
        assert port_reader.poll(10), "Local HTTP server did not start"
        endpoint = f"http://127.0.0.1:{port_reader.recv()}"
        # 仅使用虚构凭据访问本地服务，WS 地址同样指向本地以捕获意外回退。
        client.config = api.Config.from_apikey(
            "test-key", "test-secret", "test-token", http_url=endpoint,
            quote_ws_url=endpoint.replace("http://", "ws://") + "/unexpected-ws",
            quote_transport=api.QuoteTransport.Http, enable_print_quote_packages=False,
        )
        assert selection.session_window(client, now) == (
            date(2026, 11, 3), date(2026, 11, 2),
            [date(2026, 11, 3), date(2026, 11, 4), date(2026, 11, 5)],
        )
        source = selection.SelectionSource(client, tmp_path)
        candles = source.signal_candlesticks("SPY.US/USD", "1h", now, count=2)
        assert [c.timestamp.astimezone(timezone.utc) for c in candles] == stamps[:2]
        # 日线复权与未复权请求使用同一 HTTP 历史接口，不触发实时订阅。
        for adjustment in (api.AdjustType.ForwardAdjust, api.AdjustType.NoAdjust):
            assert len(source.history(
                "SPY.US", api.Period.Day, adjustment, False, 300,
                now, api.TradeSessions.Intraday,
            )) == 4
        requests = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert [path for path, _ in requests] == [
            *["/quote/markets/trading-days"] * (2 if throttle_first else 1),
            *["/quote/history-candlesticks"] * 3,
        ]
        assert not client._candle_subscriptions and not client._quote_symbols
        trade_factory.assert_not_called()
    finally:
        client.close()
        server.terminate()
        server.join(timeout=5)
        port_reader.close()
