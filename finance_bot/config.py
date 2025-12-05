from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BotConfig:
    """Configuration for the live scanner and Discord bot."""

    discord_token: str
    announcement_channel_ids: List[int]
    categories: List[str] = field(
        default_factory=lambda: ["sp500", "nasdaq100", "dow", "most_active"]
    )
    history_days: int = 180
    lookback_regime: int = 60
    lookback_buy: int = 20
    lookback_boom: int = 5
    risk_free_rate: float = 0.02
    cache_dir: str = ".cache"
    yfinance_options: Dict[str, bool] = field(
        default_factory=lambda: {"progress": False, "threads": True}
    )
