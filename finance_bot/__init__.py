"""Finance bot package providing OT-driven modeling and live Discord scanning."""

from .model import OTFinanceModel
from .alpha import CrossSectionalSignalLayer, CrossSectionalMLForecaster
from .risk import FactorRiskModel
from .scanner import YFinanceScanner
from .discord_bot import create_bot
from .config import BotConfig

__all__ = [
    "OTFinanceModel",
    "CrossSectionalSignalLayer",
    "CrossSectionalMLForecaster",
    "FactorRiskModel",
    "YFinanceScanner",
    "create_bot",
    "BotConfig",
]
