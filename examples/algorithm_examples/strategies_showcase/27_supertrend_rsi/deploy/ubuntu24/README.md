# Ubuntu 24 小内存 VPS 部署

适用于独占的1核、1GB内存、50GB磁盘VPS。使用Ubuntu 24.04自带Python 3.12、
原生venv和systemd；账户进程常驻，选股仅在美东开盘前启动，减少两个Python进程
同时占用内存的时间。安装器不自动启动交易，也不创建或重置账户账本。

连接分工：选股进程通过HTTP读取交易日历、日线及已收盘小时线，不建立实时
行情订阅或交易上下文；账户进程独占行情WebSocket，并使用统一限速的交易
上下文。因此两个服务可以同时运行。盘前timer用于减少内存重叠，不是用于
规避连接限制；旧版选股仍会建立订阅，部署前必须将两端代码一起更新。
依据：[长桥连接与调用限制](https://open.longbridge.com/zh-CN/docs#频率限制)。

当前正式基础池合并长桥返回的标普成分和 IJH 股票持仓，通常约 900 只。最多 100 只候选、5 个持仓。部署使用
25,000 USD管理资金和4,500 USD单笔预算。按风险和可用现金取整，一股即可开仓。1GB适合先做模拟账户
验证；长期实盘需要看目标机器的内存峰值、健康状态和网络情况，2GB会有更多余量。

## 资源参考

2026-11-16本机macOS x86_64、Python 3.12离线测量如下。数值库使用单线程，
主机有4个逻辑CPU，未做单核隔离；这些数据不是Ubuntu VPS实测。

| 工作负载 | 进程RSS峰值 | 耗时 |
| --- | --- | --- |
| 30只候选，账户循环10轮 | 约272 MiB | 平均2.4秒/轮 |
| 60只候选，账户循环10轮 | 约310 MiB | 平均3.2秒/轮，最慢7.4秒 |
| 200只基础池，读取足额60只小时缓存 | 约251 MiB | 约5秒计算 |

账户后几轮RSS稳定在约306 MiB。测试使用临时空仓账户及模拟传输，不包含真实SDK
网络缓冲、CLI、操作系统和长期运行开销。选股允许相关候选以加载足额缓存，仅用于
资源测试。到VPS后运行同一工具复核，不应把swap当作运行内存的等价替代。

2026-11-17在同机复测200/60/5配置，各运行5轮，性能调整前后结果如下：

| 指标 | 调整前 | 调整后 |
| --- | --- | --- |
| 账户首轮耗时 | 5.75秒 | 4.57秒 |
| 账户后续4轮平均耗时 | 2.46秒 | 1.89秒（减少约23%） |
| 300点RMA函数微基准 | 0.864毫秒 | 0.143毫秒（输出完全一致） |

运行器复用相同配置和管理标的的App，空闲股票的本地订单查询从每轮3次降至1次，
固定K线周期时长也只解析一次；有委托、撤单或成交时仍重读状态。另做20轮账户
复测，复用19次，首轮之后平均1.92秒，RSS峰值约314MiB，最后5轮约312MiB。
基准会核对策略数量及同根K线消费状态，回归测试另行核对App复用后仍逐轮执行
风险检查；性能调整不跳过分钟风控。

选股整轮模拟计算单次为2.26秒→2.46秒，尚未观察到稳定的整轮提速；不能把
指标函数微基准的提升直接套用到完整选股。初次基准的启动耗时波动较大，优先
比较同一组复测的稳定轮次。完整本机记录保存在
`resources/research/performance-2026-11-17/report.json`。
这些都是离线空仓负载，未计真实行情缓冲、交易请求限速等待、CLI、网络及
长时间持仓运行；仍未做Ubuntu单核VPS实测。使用下文benchmark命令在目标机复核。

2026-11-21 在同一台 Mac 上用相同的模拟数据对照本次改动。60 只候选的空仓账户
运行 8 轮，首轮后的 7 轮平均从 2.216 秒降至 0.611 秒，进程 RSS 峰值为 308 MiB。
两次均建立 60 次报价订阅、120 次 K 线订阅，历史补拉均为 0 次。200 只选 60 只
的选股计算各运行 5 轮，平均从 2.360 秒降至 1.990 秒，候选排序、预估股数及淘汰原因逐项一致。

50GB磁盘可容纳代码、venv、数据和2GB swap。安装器把journal日志限制为200MB、
最多14天；SQLite报告与日线缓存仍会增长，定期用`du`检查和做完整账本备份。
安装器修改的是本机journald总预算，适合这台只运行策略的VPS。

## 部署已提交版本

在本机检出包含 27 号策略的版本，将 `ubuntu@VPS` 换成实际 SSH 地址。`git archive HEAD` 只传输该提交中的源码和公开配置模板。OAuth 和现有账户账本分别按后文授权或迁移。

```bash
git rev-parse --short HEAD
git cat-file -e HEAD:examples/algorithm_examples/strategies_showcase/27_supertrend_rsi/screener.py
git archive HEAD | ssh ubuntu@VPS 'mkdir -p ~/iaf-upload && tar -x -C ~/iaf-upload'
ssh -t -L 60355:127.0.0.1:60355 ubuntu@VPS
```

在这条 SSH 连接的 VPS 终端执行安装。

```bash
sudo mkdir -p /opt/investing-algorithm-framework
sudo rsync -a --chown=root:root --chmod=D755,F644 ~/iaf-upload/ /opt/investing-algorithm-framework/
sudo bash /opt/investing-algorithm-framework/examples/algorithm_examples/strategies_showcase/27_supertrend_rsi/deploy/ubuntu24/install.sh
```

安装器创建专用用户`iaf`，只安装运行依赖，使用锁定版本并要求依赖提供wheel，
避免在1GB机器编译原生扩展；工程自身以editable方式安装。Linux x86_64及ARM64
的Python 3.12依赖已验证可解析到二进制包。依赖清单来自当前`poetry.lock`的main组，
更新框架锁文件后需同步更新这两个清单；不安装Jupyter和测试依赖。

代码归管理员管理；账户与选股数据在`/var/lib/iaf-supertrend/paper/`，OAuth缓存
归`iaf`用户管理。安装器保留已有`paper.env`及账户数据；服务运行期间会拒绝更新。

## 检查内存和时钟

```bash
free -h
swapon --show
df -h /var/lib
timedatectl status
```

确认NTP正常，报价时效与交易日历依赖准确时钟。服务使用UTC环境，交易时段仍由
程序按America/New_York计算。若机器没有swap，且`/swapfile`不存在，可配置2GB：

```bash
(
    set -e
    if [ -e /swapfile ] || [ -n "$(swapon --noheadings --show=NAME)" ]; then
        printf '已有swap或同名文件，保留原配置。\n'
    else
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        printf '/swapfile none swap sw 0 0\n' | sudo tee -a /etc/fstab
    fi
)
```

已有swap时不要重复执行上述创建命令。老款x86虚拟CPU若没有AVX2，按
[Polars官方说明](https://docs.pola.rs/user-guide/installation/)安装兼容运行时，
再执行资源测试：

```bash
sudo /opt/investing-algorithm-framework/.venv/bin/python -m pip install --only-binary=:all: 'polars[rtcompat]==1.40.1'
```

资源测试不需要OAuth，不连接券商，不操作真实账户：

```bash
sudo -H -u iaf /opt/investing-algorithm-framework/.venv/bin/python /opt/investing-algorithm-framework/examples/algorithm_examples/strategies_showcase/27_supertrend_rsi/deploy/ubuntu24/benchmark.py --iterations 30
sudo -H -u iaf /opt/investing-algorithm-framework/.venv/bin/python /opt/investing-algorithm-framework/examples/algorithm_examples/strategies_showcase/27_supertrend_rsi/deploy/ubuntu24/benchmark.py --workload selection --iterations 1
```

## 迁移账户账本

选美股休市时切换。先停旧机器的交易运行器，并确认进程管理器不会自动重新拉起。
只有本机目录锁不能防止两台机器同时管理一个账户。已有持仓或挂单时必须复制
完整账户目录，包括`databases/`、`runtime/`、`longbridge_submissions/`、`account.json`。
不能用空目录替代旧账本。新授权在VPS上完成，不需要把本机令牌夹在代码包中上传。

在本机另一个终端执行，先确认旧交易运行器已经停止：

```bash
rsync -az examples/algorithm_examples/strategies_showcase/27_supertrend_rsi/resources/portfolio/paper/ ubuntu@VPS:~/iaf-account-paper/
```

在VPS终端执行：

```bash
sudo rsync -a --chown=iaf:iaf ~/iaf-account-paper/ /var/lib/iaf-supertrend/paper/account/
sudo chmod -R go-rwx /var/lib/iaf-supertrend/paper/account
```

没有持仓和挂单的全新模拟账户可以由程序建立新账本。资金设置必须与迁移的
`account.json`一致；不要通过删除数据库绕过资金或持仓不一致的预检错误。

## 首次 OAuth 授权

SDK与选股CLI是两条授权链路，初次部署分别完成一次。后续正常过期由SDK刷新；
授权撤销或refresh token失效仍需重新授权。

**Python SDK：**保持最初的SSH隧道连接，检查配置，再运行授权工具：

```bash
sudoedit /etc/iaf-supertrend/paper.env
sudo -H -u iaf /opt/investing-algorithm-framework/.venv/bin/python /opt/investing-algorithm-framework/examples/algorithm_examples/strategies_showcase/27_supertrend_rsi/deploy/ubuntu24/authorize.py --env-file /etc/iaf-supertrend/paper.env
```

把终端打印的网址复制到自己电脑的浏览器完成授权。浏览器访问本机
`http://localhost:60355/callback`时，SSH会转发到VPS的回调服务。VPS不需要桌面，
防火墙不需要开放60355公网端口；client ID的注册地址需包含这个回调地址。

令牌存放于服务用户的`~/.longbridge/openapi/tokens/<client_id>`，默认是
`/home/iaf/.longbridge/openapi/tokens/<client_id>`。始终使用`sudo -H -u iaf`授权，
避免把凭据写进root或SSH用户的缓存。依据：[SDK OAuth文档](https://open.longbridge.com/docs/getting-started)。

**CLI：**使用设备码流程，在自己电脑或手机浏览器输入设备码即可，不需要隧道：

```bash
sudo -H -u iaf /usr/local/bin/longbridge auth login
sudo -H -u iaf /usr/local/bin/longbridge check
```

CLI可能使用独立的client ID；SDK授权成功不代表CLI已登录。依据：
[CLI授权文档](https://open.longbridge.com/docs/cli/install)。
服务启动前的`--check-cache`只检查SDK缓存文件是否存在，不代表访问权限有效；
下一步的只读预检用于验证真实账户和行情。
若SDK授权已撤销而缓存仍存在，先停服务，将当前client ID对应的缓存文件改名
备份，再运行授权工具；CLI的logout不能替代清理独立的SDK客户端缓存。

## 选股 预检和启动

先执行一次选股，再做模拟账户只读预检：

预检同样会建立实时行情连接，应在账户服务尚未启动时执行；已有账户服务
运行时不要另开预检进程。升级中的已有持仓继续按下文维护流程迁移和恢复。

```bash
sudo systemctl start iaf-selection@paper.service
sudo -H -u iaf /opt/investing-algorithm-framework/.venv/bin/python /opt/investing-algorithm-framework/examples/algorithm_examples/strategies_showcase/27_supertrend_rsi/deploy/ubuntu24/preflight.py --account paper
```

预检要求退出码0、`ready: true`，并核对账户为paper、交易开关为false。已有持仓和
券商委托应与账本一致。上述操作不会提交或撤销订单。预检通过后才启用交易服务：

```bash
sudo systemctl enable --now iaf-selection@paper.timer
sudo systemctl enable --now iaf-supertrend@paper.service
```

`paper`会向券商模拟账户真实提交订单。选股在美东工作日 07:30 首次执行，08:30 和 08:50 补查，同日成功后复用快照，
后两次可以重试失败刷新；假日由程序内的交易日历处理。成功快照同日复用，
新规则或错误标记会触发重筛。重启后timer补跑一次；当天没有有效池时暂停新仓，
已有本地持仓继续退出。固定美东时区可随冬夏令时切换，参见
[systemd时间表达式](https://www.freedesktop.org/software/systemd/man/latest/systemd.time.html)。

实盘需要独立`live.env`、对应client ID及`/var/lib/iaf-supertrend/live/`账本，
再以`@live`实例运行同一模板；保持只读预检先于启动。不要仅更改目录中的
`account.json`来把paper账本当作live账本。
同一长桥用户不要同时运行`@paper`和`@live`账户服务，切换前先停止另一实例；
不同OAuth client ID不能视为独立的行情连接额度。

## 观察资源与维护

```bash
sudo systemctl status iaf-supertrend@paper.service
sudo systemctl show iaf-supertrend@paper.service -p MemoryCurrent -p MemoryPeak -p NRestarts -p TasksCurrent
sudo journalctl -u iaf-supertrend@paper.service -n 60 --no-pager
sudo journalctl -u iaf-selection@paper.service -n 40 --no-pager
sudo systemctl list-timers 'iaf-selection@*'
sudo cat /var/lib/iaf-supertrend/paper/account/health.json
sudo du -sh /var/lib/iaf-supertrend
```

服务使用单线程数值库、低优先级选股和内存限制：账户`MemoryHigh=384M`、
`MemoryMax=640M`，选股256M/384M。High会施加回收压力，Max达到后可能触发OOM杀进程，
限制不是预留内存或稳定性保证；参见[systemd资源控制](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html)。

模拟账户先观察数个完整交易日：账户循环和健康文件应按分钟持续更新，无OOM或
反复重启，选股期间仍有内存余量，swap不持续增长。若不满足，优先升级2GB；
也可在独立目录运行较小规模的资源测试，正式账户只消费 C 方案的股票池。
客户端ATR退出依赖进程、网络和券商服务，systemd重启不能消除停机期间的风险。

升级时先停止交易服务和选股timer，再备份完整账户目录，更新代码、运行安装器及
只读预检后恢复。不要删除旧账本、提交标记或OAuth缓存；保留可回退的代码版本。
