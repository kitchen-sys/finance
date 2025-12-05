from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from finance_bot.config import BotConfig
from finance_bot.discord_bot import create_bot
from finance_bot.scanner import YFinanceScanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OT-powered finance scanner Discord bot")
    parser.add_argument("--token", help="Discord bot token (overrides ENV)")
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        help="Discord channel IDs to receive periodic scans",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Market universes to scan (sp500, nasdaq100, dow, most_active)",
    )
    parser.add_argument("--history", type=int, default=180, help="History window in days")
    parser.add_argument("--lookback-regime", type=int, default=60, help="Lookback for OT regime detection")
    parser.add_argument("--lookback-buy", type=int, default=20, help="Lookback window for buy scoring")
    parser.add_argument("--lookback-boom", type=int, default=5, help="Lookback window for boom scoring")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    token = args.token or os.getenv("DISCORD_TOKEN")
    channels = args.channels or os.getenv("DISCORD_CHANNELS", "").split(",")
    channel_ids = [int(ch) for ch in channels if str(ch).strip()]
    categories = args.categories or os.getenv("SCAN_CATEGORIES", "").split(",")
    categories = [cat for cat in categories if cat]

    resolved_categories = categories or BotConfig.__dataclass_fields__["categories"].default_factory()

    config = BotConfig(
        discord_token=token or "",
        announcement_channel_ids=channel_ids,
        categories=resolved_categories,
        history_days=args.history,
        lookback_regime=args.lookback_regime,
        lookback_buy=args.lookback_buy,
        lookback_boom=args.lookback_boom,
    )
    scanner = YFinanceScanner(config)
    bot = create_bot(config, scanner)
    bot.run(config.discord_token)


if __name__ == "__main__":
    main()
