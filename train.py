#!/usr/bin/env python3
"""
Terminal Stock Trainer
Train OT Finance Model on top tech and AI stocks
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from finance_bot.model import OTFinanceModel
from finance_bot.alpha_vantage_provider import AlphaVantageProvider


# Top 20 Tech Stocks (by market cap and sector dominance)
TOP_TECH_STOCKS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "META",  # Meta
    "TSLA",  # Tesla
    "NVDA",  # NVIDIA
    "AVGO",  # Broadcom
    "ORCL",  # Oracle
    "ADBE",  # Adobe
    "CRM",   # Salesforce
    "CSCO",  # Cisco
    "INTC",  # Intel
    "AMD",   # AMD
    "QCOM",  # Qualcomm
    "TXN",   # Texas Instruments
    "INTU",  # Intuit
    "IBM",   # IBM
    "NOW",   # ServiceNow
    "SHOP",  # Shopify
]

# Top 20 AI Stocks (AI-focused companies and heavy AI investors)
TOP_AI_STOCKS = [
    "NVDA",  # NVIDIA - AI chips
    "MSFT",  # Microsoft - OpenAI, Azure AI
    "GOOGL", # Alphabet - DeepMind, Bard
    "META",  # Meta - LLaMA, AI Research
    "AMZN",  # Amazon - AWS AI, Alexa
    "TSLA",  # Tesla - Autopilot, Dojo
    "AMD",   # AMD - AI chips
    "ORCL",  # Oracle - AI infrastructure
    "AVGO",  # Broadcom - AI networking
    "QCOM",  # Qualcomm - AI mobile chips
    "ADBE",  # Adobe - Firefly AI
    "CRM",   # Salesforce - Einstein AI
    "PLTR",  # Palantir - AI platforms
    "SNOW",  # Snowflake - AI data
    "NOW",   # ServiceNow - AI workflows
    "PATH",  # UiPath - AI automation
    "AI",    # C3.ai - Enterprise AI
    "BBAI",  # BigBear.ai - AI solutions
    "SOUN",  # SoundHound - Voice AI
    "AMBA",  # Ambarella - AI vision chips
]


class StockTrainer:
    """Train OT Finance Model on stock universes."""

    def __init__(
        self,
        lookback_regime: int = 60,
        history_days: int = 365,
        av_provider: Optional[AlphaVantageProvider] = None,
        output_dir: str = "./training_results"
    ):
        self.model = OTFinanceModel(lookback_regime=lookback_regime)
        self.history_days = history_days
        self.av_provider = av_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def download_stock_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Download historical data for list of tickers."""
        print(f"\n📊 Downloading data for {len(tickers)} stocks...")
        data = {}
        failed = []

        for i, ticker in enumerate(tickers, 1):
            try:
                print(f"  [{i}/{len(tickers)}] Fetching {ticker}...", end=" ")
                hist = yf.download(
                    ticker,
                    period=f"{self.history_days}d",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )
                if not hist.empty and len(hist) > self.model.lookback_regime * 2:
                    data[ticker] = hist
                    print("✓")
                else:
                    print("✗ (insufficient data)")
                    failed.append(ticker)
            except Exception as e:
                print(f"✗ ({str(e)[:30]})")
                failed.append(ticker)

        print(f"\n✓ Successfully downloaded {len(data)}/{len(tickers)} stocks")
        if failed:
            print(f"✗ Failed: {', '.join(failed)}")

        return data

    def train_stock(self, ticker: str, data: pd.DataFrame) -> Dict:
        """Train model on single stock and extract insights."""
        closes = data["Close"]
        returns = closes.pct_change().dropna()

        results = {
            "ticker": ticker,
            "data_points": len(closes),
            "date_range": {
                "start": str(closes.index[0].date()),
                "end": str(closes.index[-1].date()),
            },
        }

        # Regime detection
        try:
            regime = self.model.detect_regime(closes)
            results["regime"] = {
                "regime_change": regime.regime_change,
                "distance": float(regime.distance),
                "current_state": regime.current_state,
                "threshold": float(regime.threshold),
            }
        except Exception as e:
            results["regime"] = {"error": str(e)}

        # ARIMA forecasting
        try:
            if len(closes) > 30:
                arima_result = self.model.forecast_arima(closes, steps=5)
                results["arima_forecast"] = {
                    "forecast_5d": float(arima_result.forecast),
                    "stderr": float(arima_result.stderr),
                    "conf_int_lower": float(arima_result.conf_int[0]),
                    "conf_int_upper": float(arima_result.conf_int[1]),
                }
        except Exception as e:
            results["arima_forecast"] = {"error": str(e)}

        # Volatility forecasting
        try:
            if len(returns) > 50:
                vol = self.model.forecast_volatility(returns)
                results["volatility"] = {
                    "current": float(vol.iloc[-1]),
                    "mean": float(vol.mean()),
                    "max": float(vol.max()),
                }
        except Exception as e:
            results["volatility"] = {"error": str(e)}

        # Risk metrics
        try:
            var_result = self.model.value_at_risk(returns)
            sharpe = self.model.sharpe_ratio(returns)
            results["risk_metrics"] = {
                "var_95": float(var_result.var),
                "cvar_95": float(var_result.cvar),
                "sharpe_ratio": float(sharpe),
            }
        except Exception as e:
            results["risk_metrics"] = {"error": str(e)}

        # Stationarity test
        try:
            is_stationary = self.model.is_stationary(returns)
            results["stationarity"] = {
                "is_stationary": is_stationary,
            }
        except Exception as e:
            results["stationarity"] = {"error": str(e)}

        # Current price and momentum
        results["current_metrics"] = {
            "price": float(closes.iloc[-1]),
            "return_1d": float(returns.iloc[-1]),
            "return_5d": float(closes.pct_change(5).iloc[-1]),
            "return_20d": float(closes.pct_change(20).iloc[-1]),
            "return_60d": float(closes.pct_change(60).iloc[-1]),
        }

        return results

    def train_universe(self, tickers: List[str], universe_name: str) -> List[Dict]:
        """Train model on entire universe of stocks."""
        print(f"\n{'='*60}")
        print(f"Training on {universe_name} Universe")
        print(f"{'='*60}")

        # Download data
        data = self.download_stock_data(tickers)

        # Train on each stock
        print(f"\n🧠 Training model on each stock...")
        results = []

        for i, (ticker, stock_data) in enumerate(data.items(), 1):
            print(f"  [{i}/{len(data)}] Training {ticker}...", end=" ")
            try:
                result = self.train_stock(ticker, stock_data)
                results.append(result)
                print("✓")
            except Exception as e:
                print(f"✗ ({str(e)[:30]})")
                results.append({
                    "ticker": ticker,
                    "error": str(e),
                })

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{universe_name}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        # Print summary
        self._print_summary(results, universe_name)

        return results

    def _print_summary(self, results: List[Dict], universe_name: str):
        """Print training summary statistics."""
        print(f"\n{'='*60}")
        print(f"{universe_name} Training Summary")
        print(f"{'='*60}")

        successful = [r for r in results if "error" not in r]
        print(f"\n✓ Successfully trained: {len(successful)}/{len(results)} stocks")

        if not successful:
            return

        # Regime changes detected
        regime_changes = [r for r in successful if r.get("regime", {}).get("regime_change")]
        print(f"\n⚠️  Regime changes detected: {len(regime_changes)} stocks")
        if regime_changes:
            for r in regime_changes[:5]:  # Show top 5
                print(f"    • {r['ticker']}: {r['regime']['current_state']}")

        # Top performers (60-day return)
        with_returns = [r for r in successful if "current_metrics" in r]
        sorted_by_return = sorted(
            with_returns,
            key=lambda x: x["current_metrics"].get("return_60d", -999),
            reverse=True
        )

        print(f"\n📈 Top 5 performers (60-day return):")
        for r in sorted_by_return[:5]:
            ret = r["current_metrics"]["return_60d"] * 100
            print(f"    • {r['ticker']}: {ret:+.2f}%")

        print(f"\n📉 Bottom 5 performers (60-day return):")
        for r in sorted_by_return[-5:]:
            ret = r["current_metrics"]["return_60d"] * 100
            print(f"    • {r['ticker']}: {ret:+.2f}%")

        # Risk metrics
        with_sharpe = [r for r in successful if "risk_metrics" in r and "sharpe_ratio" in r["risk_metrics"]]
        if with_sharpe:
            sorted_by_sharpe = sorted(
                with_sharpe,
                key=lambda x: x["risk_metrics"]["sharpe_ratio"],
                reverse=True
            )

            print(f"\n⭐ Top 5 by Sharpe Ratio:")
            for r in sorted_by_sharpe[:5]:
                sharpe = r["risk_metrics"]["sharpe_ratio"]
                print(f"    • {r['ticker']}: {sharpe:.3f}")

        # High volatility stocks
        with_vol = [r for r in successful if "volatility" in r and "current" in r["volatility"]]
        if with_vol:
            sorted_by_vol = sorted(
                with_vol,
                key=lambda x: x["volatility"]["current"],
                reverse=True
            )

            print(f"\n💥 Most volatile (current):")
            for r in sorted_by_vol[:5]:
                vol = r["volatility"]["current"] * 100
                print(f"    • {r['ticker']}: {vol:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Train OT Finance Model on Tech and AI stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--universe",
        choices=["tech", "ai", "both"],
        default="both",
        help="Stock universe to train on (default: both)",
    )

    parser.add_argument(
        "--history",
        type=int,
        default=365,
        help="Historical data period in days (default: 365)",
    )

    parser.add_argument(
        "--lookback-regime",
        type=int,
        default=60,
        help="Lookback period for regime detection (default: 60)",
    )

    parser.add_argument(
        "--output-dir",
        default="./training_results",
        help="Output directory for results (default: ./training_results)",
    )

    parser.add_argument(
        "--av-key",
        help="Alpha Vantage API key (overrides ALPHA_VANTAGE_KEY env)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize Alpha Vantage provider if available
    av_provider = None
    av_key = args.av_key or os.getenv("ALPHA_VANTAGE_KEY")
    if av_key:
        av_provider = AlphaVantageProvider(av_key)
        print("✓ Alpha Vantage integration enabled")

    # Initialize trainer
    trainer = StockTrainer(
        lookback_regime=args.lookback_regime,
        history_days=args.history,
        av_provider=av_provider,
        output_dir=args.output_dir,
    )

    print("\n" + "="*60)
    print("🚀 Stock Trainer - OT Finance Model")
    print("="*60)
    print(f"History period: {args.history} days")
    print(f"Regime lookback: {args.lookback_regime} days")
    print(f"Output directory: {args.output_dir}")

    # Train on selected universes
    if args.universe in ["tech", "both"]:
        trainer.train_universe(TOP_TECH_STOCKS, "tech")

    if args.universe in ["ai", "both"]:
        trainer.train_universe(TOP_AI_STOCKS, "ai")

    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nTo view results, check the JSON files in the output directory.")


if __name__ == "__main__":
    main()
