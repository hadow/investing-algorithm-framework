# Supertrend + RSI · Longbridge 美股组合

27 号策略在盘前合并长桥标普 500 成分与 IJH 股票持仓，最多选出 100 只候选。盘中用已确认的小时线判断 Supertrend 与 RSI 信号，每分钟检查已有仓位的风险，最多同时持有 5 只股票。选股、信号、仓位公式和研究证据见[策略设计](DESIGN.zh-CN.md)。

| 入口 | 用途 |
| --- | --- |
| `screener.py` | 通过 HTTP 获取成分、财报及历史 K 线，发布 `resources/selection/universe.json`。默认运行一次。 |
| `portfolio_app.py` | 正式组合入口。读取当日快照，使用一个共享账户管理候选、已有持仓和未完成委托。 |
| `longbridge_app.py` | 单标的检查、行情导出或独立交易。使用自己的资源目录。 |
| `backtest.py` | 使用本地 CSV 研究原版信号，输出回测结果。 |

## 本地安装与授权

以下命令从工程根目录执行。示例使用 Python 3.12。把 `.env.example` 复制为策略目录的 `.env`，填写对应账户的 `LONGBRIDGE_OAUTH_CLIENT_ID`。SDK 首次运行通过浏览器授权，此后复用本机令牌缓存。选股使用的 Longbridge CLI 也需要完成登录。[长桥 CLI 安装与登录](https://open.longbridge.com/docs/cli/install)

```bash
strategy=examples/algorithm_examples/strategies_showcase/27_supertrend_rsi
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[longbridge]'
test -f "$strategy/.env" || cp "$strategy/.env.example" "$strategy/.env"
longbridge auth login
longbridge check
```

如果财报查询报 `openapi.longbridge.cn` 的 `client error (Connect)`，先在原命令前加 `LONGBRIDGE_REGION=global` 验证。同一查询成功后，可在本策略的 `.env` 中设置该变量并重启选股进程。

## 生成股票池，检查账户

```bash
.venv/bin/python "$strategy/screener.py"
.venv/bin/python "$strategy/portfolio_app.py" --mode check --account paper
```

检查读取授权、USD 可用现金、券商与本地持仓、有效委托、主周期 K 线、1m 心跳及盘口，并记录 `resources/portfolio/paper/preflight.json`。退出码 `0` 表示就绪，`2` 表示有待处理的问题。检查模式可以更新本地对账记录，券商委托操作关闭。

`--force` 可以强制重新选股。`--watch` 按美东日期刷新并在失败后等待五分钟重试。Ubuntu 的 systemd timer 在美东 07:30 先运行一次，08:30 和 08:50 再检查，成功快照可复用。1 核、1 GB 机器的依赖、资源限制、SSH OAuth 和账本迁移见[Ubuntu 24 部署说明](deploy/ubuntu24/README.md)。

## 运行组合

```bash
.venv/bin/python "$strategy/portfolio_app.py" --mode paper --iterations 0
```

`paper` 会向已授权的模拟账户提交订单。正式账户使用 `--mode live`、对应的 OAuth 授权和独立 `resources/portfolio/live/` 账本。`--iterations` 默认一轮，`0` 表示持续运行。macOS 可在上述命令前加 `caffeinate -i` 防止空闲休眠。

默认管理 25,000 USD，单笔名义上限 4,500 USD，单笔和组合计划风险分别为净值的 0.5% 与 2.5%。选股和账户入口的 `--capital`、`--order-budget` 需要一致。组合入口另外支持 `--max-positions` 和 `--total-risk-percent`。已有账本会校验绑定的账户模式和初始资金。旧启动命令里的 `--min-entry-shares` 为兼容参数，当前版本按可负担整股数交易。

| 出现的状态 | 含义与检查位置 |
| --- | --- |
| 财报查询或整轮刷新失败 | `resources/selection/universe.error.json` 记录失败时间和原因。修复后重新选股，完整成功的扫描会清除标记。单只行情异常另列于快照 `errors`。 |
| `stale_or_unconfirmed_bar` | 当前主周期 K 线尚未与交易日历期望的已完成 K 线对齐。检查订阅、半日市收盘时间和补拉结果。 |
| 有候选但没有买单 | 查看 `health.json` 的 `strategies`、`execution_reasons`、`entry_blockers`，区分无信号、价差、资金、名额及对账问题。 |
| 订单状态或提交回执不明 | 核对 `working_orders`、`longbridge_submissions/`、本地订单与券商委托或成交。恢复时沿用原账本。共用券商账户时可用 `--peer-directory` 指定其他策略的账本，仅供核对。 |

每个账户资源目录只允许运行一个写入进程。`health.json` 记录最近一轮候选、管理标的、有效委托、持仓、风险及策略状态。需要暂时停止新开仓、继续维护退出时，可以使用账户目录内的 `pause_entries` 文件。

```bash
account="$strategy/resources/portfolio/paper"
touch "$account/pause_entries"
# 恢复新开仓资格，下一根符合条件的信号才能入场。
rm -f "$account/pause_entries"
```

## 离线研究与验证

```bash
.venv/bin/python "$strategy/longbridge_app.py" --mode check --account paper --symbol SPY.US/USD --time-frame 1h --warmup-window 999 --export-csv "$strategy/resources/SPY.US-1h.csv"
.venv/bin/python "$strategy/backtest.py" --csv "$strategy/resources/SPY.US-1h.csv" --warmup-window 300
```

CSV 使用 UTC `Datetime,Open,High,Low,Close,Volume`。向量回测默认手续费 0.01%、滑点 0.02%，支持 `--start`、`--end` 和成本参数。2026-11-03 的一次 SPY 小时样本在 2,000 USD 初始资金下记录了 12 笔交易，11 笔已平仓，净变动为 −18.87 USD。该样本用于验证原版研究链路。正式组合的收益评估还要覆盖盘前选股、名额竞争及逐笔执行。

```bash
.venv/bin/python -m pip install pytest 'exchange-calendars==4.13.3'
.venv/bin/python -m pytest -q tests/examples/test_supertrend_{rsi,selection,runtime,deployment,correlation_experiment}.py tests/infrastructure/test_longbridge{,_http,_streaming}.py
```

`exchange-calendars` 用于历史组合实验。历史报告中的 `resources/research/`、逐日快照、行情导出与账户账本都由本地采集，历史实验需要对应日期的归档。仓库包含可直接运行的合成回归数据。实验结论及输入口径见[相关性检查时点](correlation_experiment.zh-CN.md)和[基础池与仓位分配](universe_slots.zh-CN.md)。

`supertrend_rsi.py` 保留 Alorse 的作者署名和 MPL-2.0 许可，移植依据固定提交 `8fb3837ddcd137f1450c3e82d654ff3df24d54d5`。[上游 Pine 源码](https://github.com/Alorse/pinescript-strategies/blob/8fb3837ddcd137f1450c3e82d654ff3df24d54d5/strategies/trend/Supertrend%20%2B%20RSI.pine) · [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/)
