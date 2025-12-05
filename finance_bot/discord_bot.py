from __future__ import annotations

import asyncio
from typing import Callable, Optional

import discord
from discord import app_commands
from discord.ext import commands

from .config import BotConfig
from .scanner import YFinanceScanner


def create_bot(config: BotConfig, scanner: Optional[YFinanceScanner] = None) -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)
    scanner = scanner or YFinanceScanner(config)

    async def send_scan(channel: discord.abc.Messageable) -> None:
        results = await scanner.scan_all()
        await channel.send(scanner.format_results(results))

    @bot.event
    async def on_ready() -> None:
        for guild in bot.guilds:
            await bot.tree.sync(guild=guild)
        print(f"Connected as {bot.user}")

    @bot.tree.command(description="Run a live OT-powered market scan")
    async def scan(interaction: discord.Interaction) -> None:  # type: ignore[valid-type]
        await interaction.response.defer()
        results = await scanner.scan_all()
        await interaction.followup.send(scanner.format_results(results))

    @bot.command(name="scan")
    async def legacy_scan(ctx: commands.Context) -> None:
        await ctx.send("Running scan...")
        await send_scan(ctx.channel)

    async def periodic_scan() -> None:
        await bot.wait_until_ready()
        while not bot.is_closed():
            for channel_id in config.announcement_channel_ids:
                channel = bot.get_channel(channel_id)
                if channel:
                    await send_scan(channel)
            await asyncio.sleep(60 * 30)

    bot.loop.create_task(periodic_scan())
    return bot
