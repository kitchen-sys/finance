"""Finance bot package providing OT-driven modeling and live Discord scanning."""

from .model import OTFinanceModel
from .alpha import CrossSectionalSignalLayer, CrossSectionalMLForecaster
from .scanner import YFinanceScanner
from .discord_bot import create_bot
from .config import BotConfig

__all__ = [
    "OTFinanceModel",
    "CrossSectionalSignalLayer",
    "CrossSectionalMLForecaster",
    "YFinanceScanner",
    "create_bot",
    "BotConfig",
]
