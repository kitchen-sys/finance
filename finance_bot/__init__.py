"""Finance bot package providing OT-driven modeling and live Discord scanning."""

from .model import OTFinanceModel
from .scanner import YFinanceScanner
from .discord_bot import create_bot
from .config import BotConfig

__all__ = [
    "OTFinanceModel",
    "YFinanceScanner",
    "create_bot",
    "BotConfig",
]
