"""旧示例继续导出共享诊断，并附加本组合的风险汇总。"""

from investing_algorithm_framework.infrastructure.longbridge.operations import (
    IsolatedPortfolioProvider, IsolatedOHLCVProvider, IsolatedTickerProvider,
    preflight, runtime_report as _runtime_report,
)


def runtime_report(client, app, directory, report=None, *, details=None):
    from portfolio import portfolio_risk

    return _runtime_report(
        client, app, directory, report, risk_reader=portfolio_risk, details=details,
    )


__all__ = [
    "IsolatedPortfolioProvider", "IsolatedOHLCVProvider",
    "IsolatedTickerProvider", "preflight", "runtime_report",
]
