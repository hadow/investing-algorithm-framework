"""只读识别同一券商账户中的其他策略账本，不接管其持仓或消耗本策略风险额度。"""

from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3


def read_peer_ledgers(directories, own_directory, paper):
    positions, order_ids, cash, issues = {}, set(), 0.0, []
    seen = {Path(own_directory).resolve()}
    for directory in directories:
        directory = Path(directory).resolve()
        if directory in seen:
            raise ValueError("Strategy ledger directories must be distinct")
        seen.add(directory)
        binding = json.loads((directory / "account.json").read_text())
        if binding["broker_environment"] != ("paper" if paper else "live"):
            raise ValueError(
                "Peer ledger belongs to a different account environment"
            )
        if binding.get("ledger_format") == "exposure-json-v1":
            book = json.loads((directory / "book.json").read_text())
            config = book["config"]
            if (
                config["environment"] != binding["broker_environment"]
                or config["limits"]["capital"] != binding["capital"]
            ):
                raise ValueError("Peer book differs from its account binding")
            exposure = book["exposure"]
            for symbol, amount in exposure["positions"].items():
                quantity = float(amount)
                if not math.isfinite(quantity) or quantity < 0:
                    raise ValueError("Peer ledger has an invalid long quantity")
                if quantity:
                    positions[symbol] = positions.get(symbol, 0) + quantity
            unallocated = float(exposure["cash"])
            if not math.isfinite(unallocated) or unallocated < 0:
                raise ValueError("Peer ledger has invalid cash")
            cash += unallocated
            order_ids.update(str(key) for key in exposure["order_ids"])
            if exposure["unknown"]:
                issues.append("Peer ledger has a submission without broker ID")
            continue
        # 明确读取既有 IAF 数据库；不能为未找到的同伴账本新建空库掩盖持仓。
        database = directory / "databases/prod-database.sqlite3"
        with closing(
            sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        ) as connection:
            connection.execute("BEGIN")
            for symbol, amount in connection.execute(
                "SELECT symbol, amount FROM positions"
            ):
                quantity = float(amount)
                if not math.isfinite(quantity) or quantity < 0:
                    raise ValueError(
                        "Peer ledger has an invalid long quantity"
                    )
                if symbol != "USD" and quantity != 0:
                    positions[symbol] = positions.get(symbol, 0) + quantity
            for external_id, status in connection.execute(
                "SELECT external_id, status FROM orders"
            ):
                if external_id:
                    order_ids.add(str(external_id))
                elif status in ("CREATED", "OPEN"):
                    issues.append(
                        "Peer ledger has a submission without broker ID"
                    )
            for row in connection.execute(
                "SELECT unallocated FROM portfolios"
            ):
                unallocated = float(row[0])
                if not math.isfinite(unallocated) or unallocated < 0:
                    raise ValueError("Peer ledger has invalid cash")
                cash += unallocated
        if list(directory.glob("longbridge_submissions/*.json")):
            issues.append("Peer ledger has an unreconciled submission marker")
    return positions, order_ids, cash, issues
