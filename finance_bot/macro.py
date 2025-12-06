"""Macro economic indicators for regime detection enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class MacroRegimeSignal:
    """Signals from macro indicators for regime classification."""

    growth_regime: str  # 'expansion', 'contraction', 'neutral'
    volatility_regime: str  # 'high', 'low', 'normal'
    risk_regime: str  # 'risk_on', 'risk_off', 'neutral'
    macro_score: float  # Composite score [-1, 1]


class MacroIndicatorLayer:
    """
    Macro economic indicators to enhance regime detection.

    Note: Alpha Vantage provides limited macro data on free tier.
    This implementation provides a framework that can be extended
    with premium data or alternative sources.
    """

    def __init__(self, use_vix_proxy: bool = True):
        """
        Initialize macro indicator layer.

        Args:
            use_vix_proxy: Use VIX as volatility regime indicator
        """
        self.use_vix_proxy = use_vix_proxy

    def compute_market_regime(
        self,
        spy_prices: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> MacroRegimeSignal:
        """
        Compute macro regime signals from market data.

        Args:
            spy_prices: S&P 500 (SPY) price data
            vix_data: VIX volatility index (optional)

        Returns:
            MacroRegimeSignal with regime classifications
        """
        # Growth regime: based on SPY trend
        growth_regime = self._classify_growth_regime(spy_prices)

        # Volatility regime: based on VIX or realized vol
        if vix_data is not None:
            volatility_regime = self._classify_volatility_from_vix(vix_data)
        else:
            volatility_regime = self._classify_volatility_from_returns(spy_prices)

        # Risk regime: combination of trend and volatility
        risk_regime = self._classify_risk_regime(growth_regime, volatility_regime)

        # Composite score
        macro_score = self._compute_composite_score(
            growth_regime, volatility_regime, risk_regime
        )

        return MacroRegimeSignal(
            growth_regime=growth_regime,
            volatility_regime=volatility_regime,
            risk_regime=risk_regime,
            macro_score=macro_score
        )

    def _classify_growth_regime(self, spy_prices: pd.DataFrame) -> str:
        """Classify growth regime based on trend."""
        if len(spy_prices) < 50:
            return 'neutral'

        # Use SMA crossover
        sma_20 = spy_prices.rolling(20).mean()
        sma_50 = spy_prices.rolling(50).mean()

        latest_20 = sma_20.iloc[-1].values[0] if hasattr(sma_20.iloc[-1], 'values') else sma_20.iloc[-1]
        latest_50 = sma_50.iloc[-1].values[0] if hasattr(sma_50.iloc[-1], 'values') else sma_50.iloc[-1]

        if latest_20 > latest_50 * 1.02:
            return 'expansion'
        elif latest_20 < latest_50 * 0.98:
            return 'contraction'
        else:
            return 'neutral'

    def _classify_volatility_from_vix(self, vix_data: pd.DataFrame) -> str:
        """Classify volatility regime from VIX."""
        latest_vix = vix_data.iloc[-1].values[0] if hasattr(vix_data.iloc[-1], 'values') else vix_data.iloc[-1]

        if latest_vix > 30:
            return 'high'
        elif latest_vix < 15:
            return 'low'
        else:
            return 'normal'

    def _classify_volatility_from_returns(self, prices: pd.DataFrame) -> str:
        """Classify volatility regime from realized volatility."""
        returns = prices.pct_change()
        vol_20 = returns.rolling(20).std().iloc[-1]

        vol_value = vol_20.values[0] if hasattr(vol_20, 'values') else vol_20

        # Annualized volatility
        ann_vol = vol_value * np.sqrt(252)

        if ann_vol > 0.25:
            return 'high'
        elif ann_vol < 0.12:
            return 'low'
        else:
            return 'normal'

    def _classify_risk_regime(self, growth: str, volatility: str) -> str:
        """Classify risk regime from growth and volatility."""
        if growth == 'expansion' and volatility == 'low':
            return 'risk_on'
        elif growth == 'contraction' or volatility == 'high':
            return 'risk_off'
        else:
            return 'neutral'

    def _compute_composite_score(
        self, growth: str, volatility: str, risk: str
    ) -> float:
        """Compute composite macro score."""
        score = 0.0

        # Growth component
        if growth == 'expansion':
            score += 0.5
        elif growth == 'contraction':
            score -= 0.5

        # Volatility component (low vol is good)
        if volatility == 'low':
            score += 0.3
        elif volatility == 'high':
            score -= 0.3

        # Risk component
        if risk == 'risk_on':
            score += 0.2
        elif risk == 'risk_off':
            score -= 0.2

        return np.clip(score, -1, 1)

    def compute_sector_rotation_signal(
        self,
        sector_prices: Dict[str, pd.DataFrame],
        lookback: int = 20
    ) -> Dict[str, float]:
        """
        Compute sector rotation signals based on relative performance.

        Args:
            sector_prices: Dictionary mapping sector name to price DataFrame
            lookback: Lookback period for momentum calculation

        Returns:
            Dictionary mapping sector to momentum score
        """
        sector_momentum = {}

        for sector, prices in sector_prices.items():
            if len(prices) < lookback:
                sector_momentum[sector] = 0.0
                continue

            # Compute momentum
            momentum = (prices.iloc[-1] / prices.iloc[-lookback] - 1).values[0]
            sector_momentum[sector] = momentum

        # Normalize to z-scores
        if sector_momentum:
            values = list(sector_momentum.values())
            mean = np.mean(values)
            std = np.std(values)

            if std > 0:
                sector_momentum = {
                    k: (v - mean) / std
                    for k, v in sector_momentum.items()
                }

        return sector_momentum


class MarketBreadthIndicator:
    """Market breadth indicators for regime detection."""

    def compute_advance_decline_ratio(
        self, prices: pd.DataFrame, threshold: float = 0.0
    ) -> pd.Series:
        """
        Compute advance/decline ratio.

        Args:
            prices: DataFrame with multiple stock prices
            threshold: Minimum return to count as advance (default 0%)

        Returns:
            Series with advance/decline ratio per date
        """
        returns = prices.pct_change()

        advances = (returns > threshold).sum(axis=1)
        declines = (returns < -threshold).sum(axis=1)

        ratio = advances / (declines + 1e-8)
        return ratio

    def compute_new_highs_lows(
        self, prices: pd.DataFrame, lookback: int = 252
    ) -> Dict[str, pd.Series]:
        """
        Compute new highs and new lows indicators.

        Args:
            prices: DataFrame with multiple stock prices
            lookback: Period for defining new highs/lows

        Returns:
            Dictionary with 'new_highs' and 'new_lows' series
        """
        rolling_max = prices.rolling(lookback).max()
        rolling_min = prices.rolling(lookback).min()

        new_highs = (prices == rolling_max).sum(axis=1)
        new_lows = (prices == rolling_min).sum(axis=1)

        return {
            'new_highs': new_highs,
            'new_lows': new_lows,
            'nh_nl_ratio': new_highs / (new_lows + 1e-8)
        }
