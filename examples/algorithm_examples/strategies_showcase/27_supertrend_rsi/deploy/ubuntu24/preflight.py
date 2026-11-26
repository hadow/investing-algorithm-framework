"""读取部署环境后执行固定的只读账户预检，避免用shell展开配置文件。"""

import argparse
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv


def main():
    """环境变量在父进程加载，子进程只接收路径与check参数，不传令牌。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", choices=("paper", "live"), default="paper")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--state-root", type=Path, default=Path("/var/lib/iaf-supertrend"))
    args = parser.parse_args()
    env = args.env_file or Path(f"/etc/iaf-supertrend/{args.account}.env")
    if not env.is_file():
        parser.error(f"Environment file not found: {env}")
    load_dotenv(env, override=True)
    example = Path(__file__).resolve().parents[2]
    state = args.state_root.expanduser().resolve() / args.account
    return subprocess.run(
        [
            sys.executable, str(example / "portfolio_app.py"),
            "--mode", "check", "--account", args.account,
            "--directory", str(state / "account"),
            "--universe", str(state / "selection/universe.json"),
        ],
        cwd=example.parents[3],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
