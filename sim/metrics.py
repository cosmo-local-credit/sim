from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import pandas as pd

@dataclass
class MetricsStore:
    network_rows: List[Dict[str, Any]] = field(default_factory=list)
    pool_rows: List[Dict[str, Any]] = field(default_factory=list)

    def add_network(self, row: Dict[str, Any]) -> None:
        self.network_rows.append(row)

    def add_pool_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.pool_rows.extend(rows)

    def network_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.network_rows)

    def pool_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.pool_rows)
