#!/usr/bin/env bash
# 安装运行环境和systemd模板；不启动交易，不改写已有账户与OAuth缓存。
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    printf '请用 sudo bash install.sh 运行。\n' >&2
    exit 1
fi
if [[ ! -f /etc/os-release ]]; then
    printf '此安装器仅支持 Ubuntu 24.04。\n' >&2
    exit 1
fi
. /etc/os-release
if [[ ${ID} != ubuntu || ${VERSION_ID} != 24.04 ]]; then
    printf '此安装器仅支持 Ubuntu 24.04。\n' >&2
    exit 1
fi
case "$(uname -m)" in
    x86_64|aarch64) ;;
    *) printf '需要 x86_64 或 ARM64 系统。\n' >&2; exit 1 ;;
esac

task_deploy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
task_repo_dir=$(cd -- "${task_deploy_dir}/../../../../../.." && pwd -P)
if [[ ! -f ${task_repo_dir}/pyproject.toml ]]; then
    printf '找不到工程根目录，请保留完整目录结构。\n' >&2
    exit 1
fi
for task_unit in iaf-supertrend@paper.service iaf-supertrend@live.service \
    iaf-selection@paper.service iaf-selection@live.service \
    iaf-selection@paper.timer iaf-selection@live.timer; do
    if systemctl is-active --quiet "${task_unit}"; then
        printf '请先停止 %s，再更新环境。\n' "${task_unit}" >&2
        exit 1
    fi
done

apt-get update
apt-get install -y python3-venv ca-certificates curl rsync tzdata
if ! id iaf >/dev/null 2>&1; then
    useradd --system --create-home --home-dir /home/iaf --user-group --shell /usr/sbin/nologin iaf
fi
task_service_home=$(getent passwd iaf | cut -d: -f6)
if [[ ! -d ${task_service_home} ]]; then
    printf 'iaf用户的主目录不存在，请先修正用户配置。\n' >&2
    exit 1
fi
if ! runuser -u iaf -- test -r "${task_repo_dir}/pyproject.toml"; then
    printf 'iaf用户无法读取代码，请按文档放到可读的 /opt 目录。\n' >&2
    exit 1
fi
install -d -o root -g iaf -m 0750 /etc/iaf-supertrend
install -d -o iaf -g iaf -m 0700 /var/lib/iaf-supertrend
install -d -o iaf -g iaf -m 0700 /var/lib/iaf-supertrend/paper/account
install -d -o iaf -g iaf -m 0700 /var/lib/iaf-supertrend/paper/selection
if [[ ! -f /etc/iaf-supertrend/paper.env ]]; then
    install -o root -g iaf -m 0640 "${task_deploy_dir}/environment.example" /etc/iaf-supertrend/paper.env
fi

python3 -m venv "${task_repo_dir}/.venv"
"${task_repo_dir}/.venv/bin/python" -m pip install --no-cache-dir --upgrade pip
# 依赖只接受wheel，避免1GB机器编译Rust/C扩展；工程本身单独以纯Python方式安装。
"${task_repo_dir}/.venv/bin/python" -m pip install --no-cache-dir --only-binary=:all: \
    -c "${task_deploy_dir}/constraints.txt" -r "${task_deploy_dir}/requirements.txt"
"${task_repo_dir}/.venv/bin/python" -m pip install --no-cache-dir --no-deps -e "${task_repo_dir}"

if [[ ! -x /usr/local/bin/longbridge ]]; then
    task_cli_installer=$(mktemp)
    trap 'rm -f "${task_cli_installer}"' EXIT
    curl -fsSL https://open.longbridge.com/longbridge/longbridge-terminal/install -o "${task_cli_installer}"
    sh "${task_cli_installer}"
    rm -f "${task_cli_installer}"
    trap - EXIT
fi
if [[ ! -x /usr/local/bin/longbridge ]]; then
    printf '请把Longbridge CLI安装到 /usr/local/bin/longbridge，供服务用户使用。\n' >&2
    exit 1
fi

python3 - "${task_repo_dir}" "${task_deploy_dir}" <<'PY'
from pathlib import Path
import sys
project, deployment = sys.argv[1:]
# systemd双引号和百分号使用自己的转义语法，避免路径被当作说明符或参数拆分。
escaped = project.replace("\\", "\\\\").replace('"', '\\"').replace('%', '%%')
for name in ('iaf-supertrend@.service', 'iaf-selection@.service', 'iaf-selection@.timer'):
    content = (Path(deployment) / name).read_text().replace('@PROJECT_ROOT@', escaped)
    (Path('/etc/systemd/system') / name).write_text(content)
PY
install -d -m 0755 /etc/systemd/journald.conf.d
install -m 0644 "${task_deploy_dir}/journald.conf" /etc/systemd/journald.conf.d/iaf-supertrend.conf
systemctl daemon-reload
systemctl restart systemd-journald
printf '安装完成，交易服务和选股定时器均未自动启用。\n'
printf '请先迁移完整账户目录、完成SDK和CLI授权，再执行只读预检。\n'
printf '授权用户：iaf，主目录：%s\n' "${task_service_home}"
printf '请阅读 %s/README.md 的后续命令。\n' "${task_deploy_dir}"
