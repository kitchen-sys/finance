"""Alpha Vantage data provider for enhanced technical and fundamental data."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.fundamentaldata import FundamentalData


class AlphaVantageProvider:
    """Provides technical indicators and fundamental data from Alpha Vantage."""

    def __init__(self, api_key: str, max_calls_per_minute: int = 5):
        """
        Initialize Alpha Vantage provider.

        Args:
            api_key: Alpha Vantage API key
            max_calls_per_minute: Rate limiting (free tier = 5 calls/min, 500/day)
        """
        self.api_key = api_key
        self.max_calls_per_minute = max_calls_per_minute
        self.last_call_time = 0.0
        self.call_interval = 60.0 / max_calls_per_minute

        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.ti = TechIndicators(key=api_key, output_format='pandas')
        self.fd = FundamentalData(key=api_key, output_format='pandas')

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.call_interval:
            time.sleep(self.call_interval - elapsed)
        self.last_call_time = time.time()

    def get_technical_indicators(
        self,
        symbol: str,
        indicators: Optional[list] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch technical indicators for a symbol.

        Args:
            symbol: Stock ticker
            indicators: List of indicators to fetch. Default: ['RSI', 'MACD', 'BBANDS', 'ADX']

        Returns:
            Dictionary mapping indicator name to DataFrame
        """
        if indicators is None:
            indicators = ['RSI', 'MACD', 'BBANDS', 'ADX']

        results = {}

        for indicator in indicators:
            try:
                self._rate_limit()

                if indicator == 'RSI':
                    data, _ = self.ti.get_rsi(symbol=symbol, interval='daily', time_period=14)
                    results['RSI'] = data

                elif indicator == 'MACD':
                    data, _ = self.ti.get_macd(symbol=symbol, interval='daily')
                    results['MACD'] = data

                elif indicator == 'BBANDS':
                    data, _ = self.ti.get_bbands(symbol=symbol, interval='daily', time_period=20)
                    results['BBANDS'] = data

                elif indicator == 'ADX':
                    data, _ = self.ti.get_adx(symbol=symbol, interval='daily', time_period=14)
                    results['ADX'] = data

                elif indicator == 'ATR':
                    data, _ = self.ti.get_atr(symbol=symbol, interval='daily', time_period=14)
                    results['ATR'] = data

                elif indicator == 'STOCH':
                    data, _ = self.ti.get_stoch(symbol=symbol, interval='daily')
                    results['STOCH'] = data

            except Exception as e:
                print(f"Error fetching {indicator} for {symbol}: {e}")

        return results

    def get_fundamentals(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch fundamental data for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with 'overview', 'income_statement', 'balance_sheet'
        """
        results = {}

        try:
            self._rate_limit()
            overview, _ = self.fd.get_company_overview(symbol=symbol)
            results['overview'] = overview
        except Exception as e:
            print(f"Error fetching overview for {symbol}: {e}")

        try:
            self._rate_limit()
            income, _ = self.fd.get_income_statement_annual(symbol=symbol)
            results['income_statement'] = income
        except Exception as e:
            print(f"Error fetching income statement for {symbol}: {e}")

        try:
            self._rate_limit()
            balance, _ = self.fd.get_balance_sheet_annual(symbol=symbol)
            results['balance_sheet'] = balance
        except Exception as e:
            print(f"Error fetching balance sheet for {symbol}: {e}")

        return results

    def get_earnings_calendar(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch earnings calendar.

        Args:
            symbol: Optional stock ticker (if None, fetches all upcoming)

        Returns:
            DataFrame with earnings dates
        """
        try:
            self._rate_limit()
            if symbol:
                data, _ = self.fd.get_earnings_calendar(symbol=symbol)
            else:
                # This endpoint requires premium tier
                print("Full earnings calendar requires premium API tier")
                return pd.DataFrame()
            return data
        except Exception as e:
            print(f"Error fetching earnings calendar: {e}")
            return pd.DataFrame()

    def extract_quality_metrics(self, fundamentals: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extract quality metrics from fundamental data.

        Args:
            fundamentals: Output from get_fundamentals()

        Returns:
            Dictionary with quality metrics (ROE, ROA, profit_margin, etc.)
        """
        metrics = {}

        if 'overview' in fundamentals and not fundamentals['overview'].empty:
            overview = fundamentals['overview']

            # Financial ratios
            metrics['pe_ratio'] = self._safe_float(overview.get('PERatio', [None])[0])
            metrics['peg_ratio'] = self._safe_float(overview.get('PEGRatio', [None])[0])
            metrics['price_to_book'] = self._safe_float(overview.get('PriceToBookRatio', [None])[0])
            metrics['roe'] = self._safe_float(overview.get('ReturnOnEquityTTM', [None])[0])
            metrics['roa'] = self._safe_float(overview.get('ReturnOnAssetsTTM', [None])[0])
            metrics['profit_margin'] = self._safe_float(overview.get('ProfitMargin', [None])[0])
            metrics['operating_margin'] = self._safe_float(overview.get('OperatingMarginTTM', [None])[0])
            metrics['debt_to_equity'] = self._safe_float(overview.get('DebtToEquity', [None])[0])
            metrics['current_ratio'] = self._safe_float(overview.get('CurrentRatio', [None])[0])
            metrics['quick_ratio'] = self._safe_float(overview.get('QuickRatio', [None])[0])

        return metrics

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def compute_technical_score(
        self,
        technical_indicators: Dict[str, pd.DataFrame],
        lookback_days: int = 1
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute aggregate technical score from multiple indicators.

        Args:
            technical_indicators: Output from get_technical_indicators()
            lookback_days: Number of days to look back (1 = most recent)

        Returns:
            Tuple of (aggregate_score, individual_scores_dict)
        """
        scores = {}

        # RSI: oversold (< 30) = bullish, overbought (> 70) = bearish
        if 'RSI' in technical_indicators:
            rsi_df = technical_indicators['RSI']
            if len(rsi_df) >= lookback_days:
                rsi = rsi_df.iloc[-lookback_days]['RSI']
                if rsi < 30:
                    scores['rsi'] = 1.0  # Oversold, bullish
                elif rsi > 70:
                    scores['rsi'] = -1.0  # Overbought, bearish
                else:
                    scores['rsi'] = (50 - rsi) / 20  # Normalized

        # MACD: positive histogram = bullish
        if 'MACD' in technical_indicators:
            macd_df = technical_indicators['MACD']
            if len(macd_df) >= lookback_days:
                macd_hist = macd_df.iloc[-lookback_days]['MACD_Hist']
                scores['macd'] = np.tanh(macd_hist)  # Normalized to [-1, 1]

        # Bollinger Bands: near lower band = bullish, near upper = bearish
        if 'BBANDS' in technical_indicators:
            bbands_df = technical_indicators['BBANDS']
            if len(bbands_df) >= lookback_days:
                row = bbands_df.iloc[-lookback_days]
                upper = row['Real Upper Band']
                lower = row['Real Lower Band']
                middle = row['Real Middle Band']

                # Assume current price is near middle (we'd need price data for exact)
                # Use band width as volatility proxy
                band_width = (upper - lower) / middle
                scores['bbands_width'] = -band_width  # Wider bands = higher vol = lower score

        # ADX: > 25 indicates strong trend
        if 'ADX' in technical_indicators:
            adx_df = technical_indicators['ADX']
            if len(adx_df) >= lookback_days:
                adx = adx_df.iloc[-lookback_days]['ADX']
                scores['adx'] = (adx - 25) / 25 if adx > 25 else 0  # Strong trend bonus

        # Stochastic: < 20 oversold, > 80 overbought
        if 'STOCH' in technical_indicators:
            stoch_df = technical_indicators['STOCH']
            if len(stoch_df) >= lookback_days:
                slow_k = stoch_df.iloc[-lookback_days]['SlowK']
                if slow_k < 20:
                    scores['stoch'] = 1.0
                elif slow_k > 80:
                    scores['stoch'] = -1.0
                else:
                    scores['stoch'] = (50 - slow_k) / 30

        # Aggregate: equal weighted average
        if scores:
            aggregate = np.mean(list(scores.values()))
        else:
            aggregate = 0.0

        return aggregate, scores
