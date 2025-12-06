from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .config import BotConfig
from .model import OTFinanceModel
from .alpha_vantage_provider import AlphaVantageProvider


@dataclass
class ScanResult:
    category: str
    buy_candidates: List[Tuple[str, float]]
    boom_candidates: List[Tuple[str, float]]


class YFinanceScanner:
    """Live scanner built on yfinance data with OT-aware scoring."""

    def __init__(self, config: BotConfig, av_provider: Optional[AlphaVantageProvider] = None):
        self.config = config
        self.model = OTFinanceModel(
            lookback_regime=config.lookback_regime, risk_free_rate=config.risk_free_rate
        )
        self.av_provider = av_provider

    @lru_cache(maxsize=8)
    def _universe(self, category: str) -> List[str]:
        category = category.lower()
        if category == "sp500":
            tickers = yf.tickers_sp500()
        elif category == "nasdaq100":
            tickers = yf.tickers_nasdaq()
        elif category == "dow":
            tickers = yf.tickers_dow()
        elif category == "most_active":
            tickers = [t.symbol for t in yf.get_day_movers("most_actives")]  # type: ignore[attr-defined]
        else:
            tickers = yf.tickers_sp500()
        return tickers[:50]

    def _download_history(self, tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            hist = yf.download(
                ticker,
                period=f"{self.config.history_days}d",
                interval="1d",
                auto_adjust=True,
                progress=self.config.yfinance_options.get("progress", False),
                threads=self.config.yfinance_options.get("threads", True),
            )
            if not hist.empty:
                data[ticker] = hist
        return data

    def _score_buy(self, history: pd.DataFrame, ticker: Optional[str] = None) -> float:
        closes = history["Close"].dropna()
        if len(closes) < self.config.lookback_buy:
            return -np.inf
        recent = closes.tail(self.config.lookback_buy)
        ma = closes.rolling(self.config.lookback_buy).mean().iloc[-1]
        momentum = recent.pct_change().mean()
        volatility = recent.pct_change().std()
        base_score = float(momentum - volatility + (recent.iloc[-1] - ma) / ma)

        # Enhance with Alpha Vantage technical indicators
        if self.av_provider and ticker and self.config.use_alpha_vantage_technicals:
            av_score = self._get_av_technical_score(ticker)
            # Weighted combination: 70% base, 30% Alpha Vantage
            return 0.7 * base_score + 0.3 * av_score

        return base_score

    def _score_boom(self, history: pd.DataFrame, ticker: Optional[str] = None) -> float:
        closes = history["Close"].dropna()
        if len(closes) < self.config.lookback_boom + 5:
            return -np.inf
        recent = closes.tail(self.config.lookback_boom)
        breakout = (recent.iloc[-1] - recent.min()) / recent.min()
        volume_surge = history["Volume"].tail(self.config.lookback_boom).mean() / (
            history["Volume"].tail(self.config.lookback_boom * 4).mean()
        )
        base_score = float(breakout * volume_surge)

        # Enhance with Alpha Vantage ADX for trend strength
        if self.av_provider and ticker and self.config.use_alpha_vantage_technicals:
            trend_strength = self._get_av_trend_strength(ticker)
            # Boost score if strong trend detected
            return base_score * (1 + 0.5 * trend_strength)

        return base_score

    def _get_av_technical_score(self, ticker: str) -> float:
        """Get composite technical score from Alpha Vantage indicators."""
        try:
            indicators = self.av_provider.get_technical_indicators(
                ticker, indicators=['RSI', 'MACD', 'STOCH']
            )
            score, _ = self.av_provider.compute_technical_score(indicators)
            return score
        except Exception as e:
            print(f"Error getting AV technical score for {ticker}: {e}")
            return 0.0

    def _get_av_trend_strength(self, ticker: str) -> float:
        """Get trend strength from Alpha Vantage ADX indicator."""
        try:
            indicators = self.av_provider.get_technical_indicators(
                ticker, indicators=['ADX']
            )
            if 'ADX' in indicators and not indicators['ADX'].empty:
                adx = indicators['ADX'].iloc[-1]['ADX']
                # ADX > 25 indicates strong trend
                # Normalize to [0, 1] range
                return min((adx - 25) / 50, 1.0) if adx > 25 else 0.0
            return 0.0
        except Exception as e:
            print(f"Error getting AV trend strength for {ticker}: {e}")
            return 0.0

    def _prepare_scan(self, category: str) -> ScanResult:
        tickers = self._universe(category)
        history = self._download_history(tickers)
        scored_buy: List[Tuple[str, float]] = []
        scored_boom: List[Tuple[str, float]] = []
        for symbol, hist in history.items():
            buy_score = self._score_buy(hist, ticker=symbol)
            boom_score = self._score_boom(hist, ticker=symbol)
            scored_buy.append((symbol, buy_score))
            scored_boom.append((symbol, boom_score))
        scored_buy.sort(key=lambda item: item[1], reverse=True)
        scored_boom.sort(key=lambda item: item[1], reverse=True)
        return ScanResult(category, scored_buy[:5], scored_boom[:5])

    async def scan_all(self) -> List[ScanResult]:
        loop = asyncio.get_event_loop()
        results = await asyncio.gather(
            *[loop.run_in_executor(None, self._prepare_scan, category) for category in self.config.categories]
        )
        return list(results)

    def format_results(self, results: List[ScanResult]) -> str:
        lines: List[str] = []
        for result in results:
            lines.append(f"**{result.category.upper()}**")
            lines.append("Top 5 Buy Candidates:")
            for symbol, score in result.buy_candidates:
                lines.append(f"• {symbol}: score {score:.4f}")
            lines.append("Top 5 Boom Candidates:")
            for symbol, score in result.boom_candidates:
                lines.append(f"• {symbol}: score {score:.4f}")
            lines.append("")
        return "\n".join(lines)
