"""通过 SSH 转发完成 SDK OAuth 授权；服务启动时只检查缓存是否存在。"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def main():
    """使用服务用户的原生SDK缓存，不输出令牌，也不连接交易上下文。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--check-cache", action="store_true")
    args = parser.parse_args()
    if not args.env_file.is_file():
        parser.error(f"Environment file not found: {args.env_file}")
    load_dotenv(args.env_file, override=True)
    client_id = os.environ.get("LONGBRIDGE_OAUTH_CLIENT_ID", "").strip()
    if not client_id or Path(client_id).name != client_id or client_id in (".", ".."):
        parser.error("A valid LONGBRIDGE_OAUTH_CLIENT_ID is required")
    cache = Path.home() / ".longbridge/openapi/tokens" / client_id
    if args.check_cache:
        if not cache.is_file() or cache.stat().st_size == 0:
            parser.error("SDK OAuth cache missing; run authorize.py as the service user first")
        print("SDK OAuth cache found; the SDK handles token expiry and refresh.")
        return 0

    from longbridge.openapi import OAuthBuilder

    def show_url(url):
        """只向当前交互终端显示授权地址，本地浏览器回调经SSH隧道送回VPS。"""
        print("Keep the SSH tunnel open: -L 60355:127.0.0.1:60355", flush=True)
        print(f"Open this URL in your local browser:\n{url}", flush=True)

    # 缓存是服务用户的长期授权资料，后续重启复用；不复制或打印访问令牌。
    previous_umask = os.umask(0o077)
    try:
        OAuthBuilder(client_id).build(show_url)
    finally:
        os.umask(previous_umask)
    print(f"SDK OAuth ready. Cache directory: {cache.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
