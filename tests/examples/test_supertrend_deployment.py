"""部署授权工具只显示授权链接，服务启动检查不发起交互登录或交易。"""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "examples/algorithm_examples/strategies_showcase/27_supertrend_rsi/deploy/ubuntu24"


@pytest.fixture
def authorization(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("supertrend_authorize", DEPLOY / "authorize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = tmp_path / "paper.env"
    env.write_text("LONGBRIDGE_OAUTH_CLIENT_ID=test-client\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LONGBRIDGE_OAUTH_CLIENT_ID", "old-client")
    monkeypatch.setattr(sys, "argv", ["authorize.py", "--env-file", str(env)])
    return module, tmp_path


def test_missing_cache_blocks_service_without_starting_oauth(authorization, monkeypatch):
    api = pytest.importorskip("longbridge.openapi")
    module, _ = authorization
    builder = Mock()
    monkeypatch.setattr(api, "OAuthBuilder", builder)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--check-cache"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    builder.assert_not_called()


def test_cache_check_uses_service_home_without_printing_tokens(authorization, monkeypatch, capsys):
    api = pytest.importorskip("longbridge.openapi")
    module, home = authorization
    cache = home / ".longbridge/openapi/tokens/test-client"
    cache.parent.mkdir(parents=True)
    cache.write_text("private-token-must-not-be-printed")
    builder = Mock()
    monkeypatch.setattr(api, "OAuthBuilder", builder)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--check-cache"])
    assert module.main() == 0
    builder.assert_not_called()
    assert "private-token" not in capsys.readouterr().out


def test_manual_oauth_prints_ssh_url_without_opening_trade_context(
    authorization, monkeypatch, capsys
):
    api = pytest.importorskip("longbridge.openapi")
    module, _ = authorization
    builder = Mock()
    builder.return_value.build.side_effect = lambda callback: callback(
        "https://example.invalid/authorize"
    )
    trade = Mock()
    monkeypatch.setattr(api, "OAuthBuilder", builder)
    monkeypatch.setattr(api, "TradeContext", trade)
    assert module.main() == 0
    builder.assert_called_once_with("test-client")
    trade.assert_not_called()
    output = capsys.readouterr().out
    assert "60355:127.0.0.1:60355" in output
    assert "https://example.invalid/authorize" in output


def test_deployment_preflight_keeps_check_mode_and_config_out_of_command_line(
    monkeypatch, tmp_path
):
    spec = importlib.util.spec_from_file_location("supertrend_preflight", DEPLOY / "preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = tmp_path / "paper.env"
    env.write_text("LONGBRIDGE_OAUTH_CLIENT_ID=private-client\n")
    monkeypatch.setenv("LONGBRIDGE_OAUTH_CLIENT_ID", "old-client")
    monkeypatch.setattr(sys, "argv", [
        "preflight.py", "--env-file", str(env), "--state-root", str(tmp_path),
    ])
    execute = Mock(return_value=Mock(returncode=2))
    monkeypatch.setattr(module.subprocess, "run", execute)
    assert module.main() == 2
    request = execute.call_args.args[0]
    assert request[request.index("--mode") + 1] == "check"
    assert request[request.index("--account") + 1] == "paper"
    assert not any("private-client" in arg for arg in request)
    assert request[request.index("--directory") + 1] == str(tmp_path / "paper/account")
