#!/usr/bin/env python3
"""
Stock Training Database with AI Agent
Comprehensive database for tracking training runs on 20 tech stocks using Alpha Vantage data
Includes RAG-powered AI agent for intelligent insights
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from finance_bot.model import OTFinanceModel
from finance_bot.alpha_vantage_provider import AlphaVantageProvider


# Top 20 Tech Stocks for tracking
TECH_STOCKS_20 = [
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


@dataclass
class TrainingRun:
    """Represents a complete training run."""
    run_id: Optional[int] = None
    timestamp: Optional[str] = None
    universe_name: str = "tech_20"
    lookback_regime: int = 60
    history_days: int = 365
    status: str = "pending"  # pending, running, completed, failed
    num_stocks_trained: int = 0
    num_stocks_failed: int = 0
    alpha_vantage_enabled: bool = False


@dataclass
class StockTrainingResult:
    """Results from training on a single stock."""
    result_id: Optional[int] = None
    run_id: Optional[int] = None
    ticker: str = ""
    training_timestamp: Optional[str] = None
    data_points: int = 0
    date_start: Optional[str] = None
    date_end: Optional[str] = None

    # Regime metrics
    regime_change: Optional[bool] = None
    regime_distance: Optional[float] = None
    regime_state: Optional[str] = None
    regime_threshold: Optional[float] = None

    # Risk metrics
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    sharpe_ratio: Optional[float] = None

    # Volatility metrics
    current_vol: Optional[float] = None
    mean_vol: Optional[float] = None
    max_vol: Optional[float] = None

    # Forecast metrics
    forecast_5d: Optional[float] = None
    forecast_stderr: Optional[float] = None
    forecast_conf_lower: Optional[float] = None
    forecast_conf_upper: Optional[float] = None

    # Current metrics
    price: Optional[float] = None
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_20d: Optional[float] = None
    return_60d: Optional[float] = None

    # Alpha Vantage metrics
    av_rsi: Optional[float] = None
    av_technical_score: Optional[float] = None
    av_pe_ratio: Optional[float] = None
    av_roe: Optional[float] = None
    av_profit_margin: Optional[float] = None

    # Errors
    error_message: Optional[str] = None


class StockTrainingDatabase:
    """SQLite database for tracking stock training runs."""

    def __init__(self, db_path: str = "./stock_training.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database schema."""
        cursor = self.conn.cursor()

        # Training runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                universe_name TEXT NOT NULL,
                lookback_regime INTEGER NOT NULL,
                history_days INTEGER NOT NULL,
                status TEXT NOT NULL,
                num_stocks_trained INTEGER DEFAULT 0,
                num_stocks_failed INTEGER DEFAULT 0,
                alpha_vantage_enabled BOOLEAN DEFAULT 0
            )
        """)

        # Stock training results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_training_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                training_timestamp TEXT NOT NULL,
                data_points INTEGER,
                date_start TEXT,
                date_end TEXT,

                -- Regime metrics
                regime_change BOOLEAN,
                regime_distance REAL,
                regime_state TEXT,
                regime_threshold REAL,

                -- Risk metrics
                var_95 REAL,
                cvar_95 REAL,
                sharpe_ratio REAL,

                -- Volatility metrics
                current_vol REAL,
                mean_vol REAL,
                max_vol REAL,

                -- Forecast metrics
                forecast_5d REAL,
                forecast_stderr REAL,
                forecast_conf_lower REAL,
                forecast_conf_upper REAL,

                -- Current metrics
                price REAL,
                return_1d REAL,
                return_5d REAL,
                return_20d REAL,
                return_60d REAL,

                -- Alpha Vantage metrics
                av_rsi REAL,
                av_technical_score REAL,
                av_pe_ratio REAL,
                av_roe REAL,
                av_profit_margin REAL,

                -- Errors
                error_message TEXT,

                FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
            )
        """)

        # Create indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_results_ticker
            ON stock_training_results(ticker)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_results_run_id
            ON stock_training_results(run_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_runs_timestamp
            ON training_runs(timestamp)
        """)

        self.conn.commit()

    def create_training_run(self, run: TrainingRun) -> int:
        """Create a new training run and return its ID."""
        run.timestamp = datetime.now().isoformat()
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO training_runs
            (timestamp, universe_name, lookback_regime, history_days, status,
             num_stocks_trained, num_stocks_failed, alpha_vantage_enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run.timestamp, run.universe_name, run.lookback_regime,
            run.history_days, run.status, run.num_stocks_trained,
            run.num_stocks_failed, run.alpha_vantage_enabled
        ))

        self.conn.commit()
        return cursor.lastrowid

    def update_training_run_status(
        self,
        run_id: int,
        status: str,
        num_trained: Optional[int] = None,
        num_failed: Optional[int] = None
    ):
        """Update training run status and counts."""
        cursor = self.conn.cursor()

        if num_trained is not None and num_failed is not None:
            cursor.execute("""
                UPDATE training_runs
                SET status = ?, num_stocks_trained = ?, num_stocks_failed = ?
                WHERE run_id = ?
            """, (status, num_trained, num_failed, run_id))
        else:
            cursor.execute("""
                UPDATE training_runs SET status = ? WHERE run_id = ?
            """, (status, run_id))

        self.conn.commit()

    def save_training_result(self, result: StockTrainingResult) -> int:
        """Save training result for a single stock."""
        result.training_timestamp = datetime.now().isoformat()
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO stock_training_results (
                run_id, ticker, training_timestamp, data_points,
                date_start, date_end,
                regime_change, regime_distance, regime_state, regime_threshold,
                var_95, cvar_95, sharpe_ratio,
                current_vol, mean_vol, max_vol,
                forecast_5d, forecast_stderr, forecast_conf_lower, forecast_conf_upper,
                price, return_1d, return_5d, return_20d, return_60d,
                av_rsi, av_technical_score, av_pe_ratio, av_roe, av_profit_margin,
                error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.run_id, result.ticker, result.training_timestamp,
            result.data_points, result.date_start, result.date_end,
            result.regime_change, result.regime_distance, result.regime_state,
            result.regime_threshold, result.var_95, result.cvar_95,
            result.sharpe_ratio, result.current_vol, result.mean_vol,
            result.max_vol, result.forecast_5d, result.forecast_stderr,
            result.forecast_conf_lower, result.forecast_conf_upper,
            result.price, result.return_1d, result.return_5d,
            result.return_20d, result.return_60d, result.av_rsi,
            result.av_technical_score, result.av_pe_ratio, result.av_roe,
            result.av_profit_margin, result.error_message
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_latest_run(self) -> Optional[Dict]:
        """Get the most recent training run."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM training_runs
            ORDER BY timestamp DESC LIMIT 1
        """)
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_run_results(self, run_id: int) -> List[Dict]:
        """Get all training results for a specific run."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM stock_training_results
            WHERE run_id = ?
            ORDER BY ticker
        """, (run_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_stock_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get training history for a specific stock."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM stock_training_results
            WHERE ticker = ?
            ORDER BY training_timestamp DESC
            LIMIT ?
        """, (ticker, limit))
        return [dict(row) for row in cursor.fetchall()]

    def get_top_performers(self, metric: str = "return_60d", limit: int = 10) -> List[Dict]:
        """Get top performing stocks by a specific metric."""
        cursor = self.conn.cursor()

        # Get latest run
        latest_run = self.get_latest_run()
        if not latest_run:
            return []

        cursor.execute(f"""
            SELECT * FROM stock_training_results
            WHERE run_id = ? AND {metric} IS NOT NULL
            ORDER BY {metric} DESC
            LIMIT ?
        """, (latest_run['run_id'], limit))
        return [dict(row) for row in cursor.fetchall()]

    def get_regime_changes(self) -> List[Dict]:
        """Get all stocks currently showing regime changes."""
        latest_run = self.get_latest_run()
        if not latest_run:
            return []

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM stock_training_results
            WHERE run_id = ? AND regime_change = 1
            ORDER BY regime_distance DESC
        """, (latest_run['run_id'],))
        return [dict(row) for row in cursor.fetchall()]

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute arbitrary SQL query (for AI agent)."""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection."""
        self.conn.close()


class StockTrainer:
    """Enhanced stock trainer with database persistence and Alpha Vantage integration."""

    def __init__(
        self,
        db: StockTrainingDatabase,
        av_provider: Optional[AlphaVantageProvider] = None,
        lookback_regime: int = 60,
        history_days: int = 365
    ):
        self.db = db
        self.av_provider = av_provider
        self.model = OTFinanceModel(lookback_regime=lookback_regime)
        self.lookback_regime = lookback_regime
        self.history_days = history_days

    def download_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download historical data for a single ticker."""
        try:
            hist = yf.download(
                ticker,
                period=f"{self.history_days}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if not hist.empty and len(hist) > self.lookback_regime * 2:
                return hist
            return None
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            return None

    def get_alpha_vantage_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch Alpha Vantage technical and fundamental data."""
        av_data = {}

        if not self.av_provider:
            return av_data

        try:
            # Get technical indicators
            tech_indicators = self.av_provider.get_technical_indicators(
                ticker,
                indicators=['RSI', 'MACD', 'ADX']
            )

            # Compute technical score
            if tech_indicators:
                tech_score, individual_scores = self.av_provider.compute_technical_score(
                    tech_indicators
                )
                av_data['technical_score'] = tech_score

                # Get latest RSI value
                if 'RSI' in tech_indicators and not tech_indicators['RSI'].empty:
                    av_data['rsi'] = float(tech_indicators['RSI'].iloc[-1]['RSI'])

            # Get fundamentals
            fundamentals = self.av_provider.get_fundamentals(ticker)
            if fundamentals:
                quality_metrics = self.av_provider.extract_quality_metrics(fundamentals)
                av_data['pe_ratio'] = quality_metrics.get('pe_ratio')
                av_data['roe'] = quality_metrics.get('roe')
                av_data['profit_margin'] = quality_metrics.get('profit_margin')

        except Exception as e:
            print(f"  Alpha Vantage error for {ticker}: {e}")

        return av_data

    def train_stock(self, ticker: str, run_id: int) -> StockTrainingResult:
        """Train model on single stock and return result."""
        result = StockTrainingResult(run_id=run_id, ticker=ticker)

        try:
            # Download price data
            data = self.download_stock_data(ticker)
            if data is None or data.empty:
                result.error_message = "Failed to download data or insufficient data"
                return result

            closes = data["Close"]
            returns = closes.pct_change().dropna()

            result.data_points = len(closes)
            result.date_start = str(closes.index[0].date())
            result.date_end = str(closes.index[-1].date())

            # Regime detection
            try:
                regime = self.model.detect_regime(closes)
                result.regime_change = regime.regime_change
                result.regime_distance = float(regime.distance)
                result.regime_state = regime.current_state
                result.regime_threshold = float(regime.threshold)
            except Exception as e:
                print(f"  Regime detection error: {e}")

            # ARIMA forecasting
            try:
                if len(closes) > 30:
                    arima_result = self.model.forecast_arima(closes, steps=5)
                    result.forecast_5d = float(arima_result.forecast)
                    result.forecast_stderr = float(arima_result.stderr)
                    result.forecast_conf_lower = float(arima_result.conf_int[0])
                    result.forecast_conf_upper = float(arima_result.conf_int[1])
            except Exception as e:
                print(f"  ARIMA error: {e}")

            # Volatility forecasting
            try:
                if len(returns) > 50:
                    vol = self.model.forecast_volatility(returns)
                    result.current_vol = float(vol.iloc[-1])
                    result.mean_vol = float(vol.mean())
                    result.max_vol = float(vol.max())
            except Exception as e:
                print(f"  Volatility error: {e}")

            # Risk metrics
            try:
                var_result = self.model.value_at_risk(returns)
                sharpe = self.model.sharpe_ratio(returns)
                result.var_95 = float(var_result.var)
                result.cvar_95 = float(var_result.cvar)
                result.sharpe_ratio = float(sharpe)
            except Exception as e:
                print(f"  Risk metrics error: {e}")

            # Current metrics
            result.price = float(closes.iloc[-1])
            result.return_1d = float(returns.iloc[-1])
            result.return_5d = float(closes.pct_change(5).iloc[-1])
            result.return_20d = float(closes.pct_change(20).iloc[-1])
            result.return_60d = float(closes.pct_change(60).iloc[-1])

            # Alpha Vantage data
            if self.av_provider:
                av_data = self.get_alpha_vantage_data(ticker)
                result.av_rsi = av_data.get('rsi')
                result.av_technical_score = av_data.get('technical_score')
                result.av_pe_ratio = av_data.get('pe_ratio')
                result.av_roe = av_data.get('roe')
                result.av_profit_margin = av_data.get('profit_margin')

        except Exception as e:
            result.error_message = str(e)

        return result

    def train_all_stocks(
        self,
        tickers: List[str] = TECH_STOCKS_20,
        universe_name: str = "tech_20"
    ) -> int:
        """Train on all stocks and save to database."""
        print(f"\n{'='*70}")
        print(f"🚀 Stock Training Run: {universe_name}")
        print(f"{'='*70}")
        print(f"Stocks: {len(tickers)}")
        print(f"History: {self.history_days} days")
        print(f"Regime lookback: {self.lookback_regime} days")
        print(f"Alpha Vantage: {'Enabled' if self.av_provider else 'Disabled'}")
        print(f"{'='*70}\n")

        # Create training run
        run = TrainingRun(
            universe_name=universe_name,
            lookback_regime=self.lookback_regime,
            history_days=self.history_days,
            status="running",
            alpha_vantage_enabled=self.av_provider is not None
        )
        run_id = self.db.create_training_run(run)

        print(f"📊 Training Run ID: {run_id}\n")

        # Train on each stock
        num_trained = 0
        num_failed = 0

        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Training {ticker}...", end=" ")

            result = self.train_stock(ticker, run_id)
            self.db.save_training_result(result)

            if result.error_message:
                print(f"✗ ({result.error_message[:40]})")
                num_failed += 1
            else:
                print("✓")
                num_trained += 1

        # Update run status
        self.db.update_training_run_status(
            run_id,
            "completed",
            num_trained,
            num_failed
        )

        print(f"\n{'='*70}")
        print(f"✅ Training Complete!")
        print(f"{'='*70}")
        print(f"Successful: {num_trained}/{len(tickers)}")
        print(f"Failed: {num_failed}/{len(tickers)}")
        print(f"{'='*70}\n")

        # Print summary
        self._print_summary(run_id)

        return run_id

    def _print_summary(self, run_id: int):
        """Print training summary."""
        results = self.db.get_run_results(run_id)

        if not results:
            return

        # Filter successful results
        successful = [r for r in results if not r['error_message']]

        print("\n📈 Top 5 Performers (60-day return):")
        top_performers = sorted(
            [r for r in successful if r['return_60d'] is not None],
            key=lambda x: x['return_60d'],
            reverse=True
        )[:5]

        for r in top_performers:
            ret = r['return_60d'] * 100
            print(f"  • {r['ticker']:6s}: {ret:+7.2f}%")

        print("\n⭐ Top 5 by Sharpe Ratio:")
        top_sharpe = sorted(
            [r for r in successful if r['sharpe_ratio'] is not None],
            key=lambda x: x['sharpe_ratio'],
            reverse=True
        )[:5]

        for r in top_sharpe:
            print(f"  • {r['ticker']:6s}: {r['sharpe_ratio']:7.3f}")

        print("\n⚠️  Regime Changes Detected:")
        regime_changes = [r for r in successful if r['regime_change']]
        if regime_changes:
            for r in regime_changes[:5]:
                print(f"  • {r['ticker']:6s}: {r['regime_state']} (distance: {r['regime_distance']:.4f})")
        else:
            print("  None")


class AIInsightAgent:
    """
    RAG-powered AI agent for intelligent insights on stock training data.
    Provides natural language interface to query and analyze training results.
    """

    def __init__(
        self,
        db: StockTrainingDatabase,
        api_key: Optional[str] = None
    ):
        self.db = db
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []

    def _build_context(self, query: str) -> str:
        """Build context from database for RAG."""
        context_parts = []

        # Get latest training run
        latest_run = self.db.get_latest_run()
        if latest_run:
            context_parts.append(f"Latest Training Run (ID: {latest_run['run_id']}):")
            context_parts.append(f"  Timestamp: {latest_run['timestamp']}")
            context_parts.append(f"  Universe: {latest_run['universe_name']}")
            context_parts.append(f"  Stocks trained: {latest_run['num_stocks_trained']}")
            context_parts.append(f"  Alpha Vantage: {'Yes' if latest_run['alpha_vantage_enabled'] else 'No'}")
            context_parts.append("")

            # Get results for this run
            results = self.db.get_run_results(latest_run['run_id'])

            # Add summary statistics
            if results:
                successful = [r for r in results if not r['error_message']]

                context_parts.append("Summary Statistics:")
                context_parts.append(f"  Total stocks: {len(results)}")
                context_parts.append(f"  Successful: {len(successful)}")
                context_parts.append("")

                # Top performers
                top_perf = sorted(
                    [r for r in successful if r['return_60d'] is not None],
                    key=lambda x: x['return_60d'],
                    reverse=True
                )[:10]

                if top_perf:
                    context_parts.append("Top 10 Performers (60-day return):")
                    for r in top_perf:
                        ret = r['return_60d'] * 100
                        sharpe = r['sharpe_ratio'] or 0
                        regime = "⚠️ SHIFT" if r['regime_change'] else "stable"
                        context_parts.append(
                            f"  {r['ticker']:6s}: {ret:+7.2f}% | "
                            f"Sharpe: {sharpe:6.3f} | "
                            f"Regime: {regime}"
                        )
                    context_parts.append("")

                # Regime changes
                regime_changes = [r for r in successful if r['regime_change']]
                if regime_changes:
                    context_parts.append(f"Regime Changes: {len(regime_changes)} stocks")
                    for r in regime_changes[:5]:
                        context_parts.append(
                            f"  {r['ticker']:6s}: {r['regime_state']} "
                            f"(distance: {r['regime_distance']:.4f})"
                        )
                    context_parts.append("")

                # Volatility leaders
                high_vol = sorted(
                    [r for r in successful if r['current_vol'] is not None],
                    key=lambda x: x['current_vol'],
                    reverse=True
                )[:5]

                if high_vol:
                    context_parts.append("Most Volatile:")
                    for r in high_vol:
                        vol = r['current_vol'] * 100
                        context_parts.append(f"  {r['ticker']:6s}: {vol:6.2f}%")
                    context_parts.append("")

        return "\n".join(context_parts)

    def query(self, user_query: str) -> str:
        """
        Query the AI agent with natural language.
        Uses RAG to provide context-aware insights.
        """
        # Build context from database
        context = self._build_context(user_query)

        # Build prompt
        system_prompt = """You are an expert financial analyst AI assistant with access to stock training data.
You have been trained on optimal transport theory, regime detection, risk metrics, and portfolio optimization.

Your role is to:
1. Analyze stock training results and provide actionable insights
2. Identify patterns, anomalies, and opportunities in the data
3. Explain complex financial concepts in clear, concise language
4. Make data-driven recommendations based on the training results

Use the provided context to answer questions accurately. If you need specific data not in the context,
say so and suggest what additional queries might help.

Be direct, insightful, and focus on what matters to traders and investors."""

        user_prompt = f"""Context from latest training run:

{context}

User Question: {user_query}

Provide a clear, insightful answer based on the data above."""

        # For now, return a structured response based on the context
        # In production, this would call the Anthropic API
        response = self._generate_response(context, user_query)

        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def _generate_response(self, context: str, query: str) -> str:
        """
        Generate response based on context and query.
        This is a simple rule-based system. In production, use Anthropic Claude API.
        """
        query_lower = query.lower()

        # Extract data from context
        latest_run = self.db.get_latest_run()
        if not latest_run:
            return "No training runs found in database. Please run a training session first."

        results = self.db.get_run_results(latest_run['run_id'])
        successful = [r for r in results if not r['error_message']]

        # Handle different query types
        if any(word in query_lower for word in ['top', 'best', 'performer', 'winner']):
            return self._analyze_top_performers(successful)

        elif any(word in query_lower for word in ['regime', 'shift', 'change']):
            return self._analyze_regime_changes(successful)

        elif any(word in query_lower for word in ['risk', 'volatile', 'volatility']):
            return self._analyze_risk_metrics(successful)

        elif any(word in query_lower for word in ['sharpe', 'ratio']):
            return self._analyze_sharpe_ratios(successful)

        elif any(word in query_lower for word in ['summary', 'overview', 'general']):
            return self._generate_summary(latest_run, successful)

        elif any(word in query_lower for word in ['recommend', 'buy', 'invest', 'trade']):
            return self._generate_recommendations(successful)

        else:
            return self._generate_summary(latest_run, successful)

    def _analyze_top_performers(self, results: List[Dict]) -> str:
        """Analyze top performing stocks."""
        top_perf = sorted(
            [r for r in results if r['return_60d'] is not None],
            key=lambda x: x['return_60d'],
            reverse=True
        )[:5]

        if not top_perf:
            return "No performance data available."

        response = "📈 Top 5 Performers (60-day return):\n\n"

        for i, r in enumerate(top_perf, 1):
            ret = r['return_60d'] * 100
            sharpe = r['sharpe_ratio'] or 0
            regime = "experiencing regime shift" if r['regime_change'] else "stable"

            response += f"{i}. {r['ticker']}: {ret:+.2f}%\n"
            response += f"   Sharpe Ratio: {sharpe:.3f}\n"
            response += f"   Regime: {regime}\n"

            if r['av_technical_score'] is not None:
                response += f"   Technical Score: {r['av_technical_score']:.3f}\n"

            response += "\n"

        # Add insight
        avg_return = np.mean([r['return_60d'] for r in top_perf]) * 100
        response += f"Average return of top 5: {avg_return:+.2f}%\n"
        response += "\n💡 Insight: "

        if avg_return > 20:
            response += "Strong momentum in top performers. Consider these for continuation plays."
        elif avg_return > 10:
            response += "Solid performance from leaders. Watch for profit-taking signals."
        else:
            response += "Moderate gains in top tier. Market may be ranging."

        return response

    def _analyze_regime_changes(self, results: List[Dict]) -> str:
        """Analyze regime changes."""
        regime_changes = [r for r in results if r['regime_change']]

        response = f"⚠️  Regime Change Analysis:\n\n"
        response += f"Stocks showing regime shifts: {len(regime_changes)}/{len(results)}\n\n"

        if not regime_changes:
            response += "No regime changes detected. Market conditions appear stable.\n"
            return response

        # Sort by distance (strength of regime change)
        regime_changes.sort(key=lambda x: x['regime_distance'], reverse=True)

        response += "Top 5 strongest regime shifts:\n\n"
        for i, r in enumerate(regime_changes[:5], 1):
            ret = r['return_60d'] * 100 if r['return_60d'] else 0
            response += f"{i}. {r['ticker']}: {r['regime_state']}\n"
            response += f"   Distance: {r['regime_distance']:.4f}\n"
            response += f"   60-day return: {ret:+.2f}%\n"
            response += "\n"

        # Add insight
        response += "💡 Insight: Regime changes indicate shifting market dynamics. "
        response += "These stocks may be entering new volatility or trend patterns. "
        response += "Adjust position sizing and risk management accordingly."

        return response

    def _analyze_risk_metrics(self, results: List[Dict]) -> str:
        """Analyze risk and volatility."""
        with_vol = [r for r in results if r['current_vol'] is not None]

        if not with_vol:
            return "No volatility data available."

        response = "📊 Risk & Volatility Analysis:\n\n"

        # Sort by volatility
        high_vol = sorted(with_vol, key=lambda x: x['current_vol'], reverse=True)[:5]
        low_vol = sorted(with_vol, key=lambda x: x['current_vol'])[:5]

        response += "Highest Volatility:\n"
        for r in high_vol:
            vol = r['current_vol'] * 100
            sharpe = r['sharpe_ratio'] or 0
            response += f"  {r['ticker']:6s}: {vol:6.2f}% (Sharpe: {sharpe:.3f})\n"

        response += "\nLowest Volatility:\n"
        for r in low_vol:
            vol = r['current_vol'] * 100
            sharpe = r['sharpe_ratio'] or 0
            response += f"  {r['ticker']:6s}: {vol:6.2f}% (Sharpe: {sharpe:.3f})\n"

        # Calculate average
        avg_vol = np.mean([r['current_vol'] for r in with_vol]) * 100
        response += f"\nAverage volatility: {avg_vol:.2f}%\n"

        response += "\n💡 Insight: "
        if avg_vol > 3:
            response += "High volatility environment. Use wider stops and smaller positions."
        elif avg_vol > 2:
            response += "Moderate volatility. Normal position sizing appropriate."
        else:
            response += "Low volatility. May indicate consolidation or range-bound markets."

        return response

    def _analyze_sharpe_ratios(self, results: List[Dict]) -> str:
        """Analyze Sharpe ratios."""
        with_sharpe = [r for r in results if r['sharpe_ratio'] is not None]

        if not with_sharpe:
            return "No Sharpe ratio data available."

        response = "⭐ Sharpe Ratio Analysis (Risk-Adjusted Returns):\n\n"

        # Sort by Sharpe
        top_sharpe = sorted(with_sharpe, key=lambda x: x['sharpe_ratio'], reverse=True)[:8]

        response += "Top 8 by Sharpe Ratio:\n\n"
        for i, r in enumerate(top_sharpe, 1):
            sharpe = r['sharpe_ratio']
            ret = r['return_60d'] * 100 if r['return_60d'] else 0
            vol = r['current_vol'] * 100 if r['current_vol'] else 0

            response += f"{i}. {r['ticker']:6s}: {sharpe:7.3f}\n"
            response += f"   Return: {ret:+.2f}% | Volatility: {vol:.2f}%\n"
            response += "\n"

        avg_sharpe = np.mean([r['sharpe_ratio'] for r in with_sharpe])
        response += f"Average Sharpe ratio: {avg_sharpe:.3f}\n"

        response += "\n💡 Insight: "
        response += "Sharpe ratio measures return per unit of risk. "
        response += "Higher is better. Aim for >1.0 for good risk-adjusted performance."

        return response

    def _generate_summary(self, run: Dict, results: List[Dict]) -> str:
        """Generate comprehensive summary."""
        response = f"📊 Training Run Summary\n\n"
        response += f"Run ID: {run['run_id']}\n"
        response += f"Timestamp: {run['timestamp']}\n"
        response += f"Universe: {run['universe_name']}\n"
        response += f"Stocks trained: {run['num_stocks_trained']}/{run['num_stocks_trained'] + run['num_stocks_failed']}\n"
        response += f"Alpha Vantage: {'Enabled' if run['alpha_vantage_enabled'] else 'Disabled'}\n\n"

        if not results:
            return response + "No results available."

        # Calculate key metrics
        returns_60d = [r['return_60d'] for r in results if r['return_60d'] is not None]
        sharpes = [r['sharpe_ratio'] for r in results if r['sharpe_ratio'] is not None]
        regime_changes = len([r for r in results if r['regime_change']])

        response += f"Key Metrics:\n"
        if returns_60d:
            response += f"  Avg 60-day return: {np.mean(returns_60d)*100:+.2f}%\n"
            response += f"  Best performer: {max(returns_60d)*100:+.2f}%\n"
            response += f"  Worst performer: {min(returns_60d)*100:+.2f}%\n"
        if sharpes:
            response += f"  Avg Sharpe ratio: {np.mean(sharpes):.3f}\n"
        response += f"  Regime changes: {regime_changes} stocks\n"

        return response

    def _generate_recommendations(self, results: List[Dict]) -> str:
        """Generate trading recommendations based on data."""
        response = "🎯 AI Trading Recommendations:\n\n"

        # Find stocks with good Sharpe, positive returns, and no regime shift
        candidates = [
            r for r in results
            if r['sharpe_ratio'] is not None
            and r['sharpe_ratio'] > 0.5
            and r['return_60d'] is not None
            and r['return_60d'] > 0
            and not r['regime_change']
        ]

        if not candidates:
            response += "No clear buy candidates at the moment based on current criteria.\n"
            return response

        # Sort by combination of Sharpe and return
        candidates.sort(
            key=lambda x: x['sharpe_ratio'] * (1 + x['return_60d']),
            reverse=True
        )

        response += "🟢 LONG Candidates (Strong fundamentals, stable regime):\n\n"
        for i, r in enumerate(candidates[:3], 1):
            ret = r['return_60d'] * 100
            sharpe = r['sharpe_ratio']
            response += f"{i}. {r['ticker']}\n"
            response += f"   60-day return: {ret:+.2f}%\n"
            response += f"   Sharpe ratio: {sharpe:.3f}\n"
            response += f"   Regime: Stable\n"
            if r['av_technical_score']:
                response += f"   Technical score: {r['av_technical_score']:.3f}\n"
            response += "\n"

        # Find regime shift stocks for potential reversal plays
        regime_shifts = [
            r for r in results
            if r['regime_change']
            and r['return_60d'] is not None
        ]

        if regime_shifts:
            response += "⚠️  WATCH (Regime shifts - potential opportunities):\n\n"
            for r in regime_shifts[:2]:
                ret = r['return_60d'] * 100
                response += f"  {r['ticker']}: {r['regime_state']} "
                response += f"(return: {ret:+.2f}%)\n"

        response += "\n⚠️  Disclaimer: These are data-driven insights, not financial advice. "
        response += "Always do your own research and manage risk appropriately."

        return response

    def _show_menu(self):
        """Display the menu options."""
        print("\n" + "="*70)
        print("🤖 AI Stock Insight Agent - Menu")
        print("="*70)
        print("\n📊 QUICK QUERIES:")
        print("  [1] Top Performers (60-day returns)")
        print("  [2] Regime Changes Detected")
        print("  [3] Risk & Volatility Analysis")
        print("  [4] Sharpe Ratio Rankings")
        print("  [5] Trading Recommendations")
        print("  [6] Market Summary & Overview")
        print("  [7] Most Volatile Stocks")
        print("  [8] Best Risk-Adjusted Returns")
        print("  [9] Bottom Performers")
        print("  [10] Alpha Vantage Technical Analysis")
        print("\n💬 FREE-FORM:")
        print("  [70] Free Chat Mode (ask anything)")
        print("\n  [0] Exit")
        print("="*70)

    def menu(self):
        """Interactive menu-driven interface."""
        print("\n🚀 Welcome to AI Stock Insight Agent!")

        # Check if we have any training data
        latest_run = self.db.get_latest_run()
        if not latest_run:
            print("\n⚠️  No training data found!")
            print("Run training first: python stock_training_db.py --train\n")
            return

        while True:
            try:
                self._show_menu()

                choice = input("\nSelect option (0 to exit): ").strip()

                if choice == '0' or choice.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break

                if not choice:
                    continue

                # Handle the choice
                response = None

                if choice == '1':
                    response = self._analyze_top_performers(
                        [r for r in self.db.get_run_results(latest_run['run_id'])
                         if not r['error_message']]
                    )
                elif choice == '2':
                    response = self._analyze_regime_changes(
                        [r for r in self.db.get_run_results(latest_run['run_id'])
                         if not r['error_message']]
                    )
                elif choice == '3':
                    response = self._analyze_risk_metrics(
                        [r for r in self.db.get_run_results(latest_run['run_id'])
                         if not r['error_message']]
                    )
                elif choice == '4':
                    response = self._analyze_sharpe_ratios(
                        [r for r in self.db.get_run_results(latest_run['run_id'])
                         if not r['error_message']]
                    )
                elif choice == '5':
                    response = self._generate_recommendations(
                        [r for r in self.db.get_run_results(latest_run['run_id'])
                         if not r['error_message']]
                    )
                elif choice == '6':
                    response = self._generate_summary(
                        latest_run,
                        [r for r in self.db.get_run_results(latest_run['run_id'])
                         if not r['error_message']]
                    )
                elif choice == '7':
                    response = self.query("Show me the most volatile stocks")
                elif choice == '8':
                    response = self.query("Which stocks have the best risk-adjusted returns?")
                elif choice == '9':
                    results = [r for r in self.db.get_run_results(latest_run['run_id'])
                              if not r['error_message'] and r['return_60d'] is not None]
                    bottom_perf = sorted(results, key=lambda x: x['return_60d'])[:5]

                    response = "📉 Bottom 5 Performers (60-day return):\n\n"
                    for i, r in enumerate(bottom_perf, 1):
                        ret = r['return_60d'] * 100
                        sharpe = r['sharpe_ratio'] or 0
                        response += f"{i}. {r['ticker']:6s}: {ret:+7.2f}% (Sharpe: {sharpe:.3f})\n"

                    response += "\n💡 Insight: Consider these for potential reversal plays or avoid if downtrend continues."

                elif choice == '10':
                    results = [r for r in self.db.get_run_results(latest_run['run_id'])
                              if not r['error_message'] and r['av_technical_score'] is not None]

                    if results:
                        sorted_av = sorted(results, key=lambda x: x['av_technical_score'], reverse=True)[:8]
                        response = "📊 Alpha Vantage Technical Analysis (Top 8):\n\n"
                        for i, r in enumerate(sorted_av, 1):
                            response += f"{i}. {r['ticker']:6s}: Score {r['av_technical_score']:.3f}"
                            if r['av_rsi']:
                                response += f" | RSI: {r['av_rsi']:.1f}"
                            if r['av_pe_ratio']:
                                response += f" | P/E: {r['av_pe_ratio']:.1f}"
                            response += "\n"
                        response += "\n💡 Higher technical scores indicate better technical setup."
                    else:
                        response = "No Alpha Vantage data available. Make sure ALPHA_VANTAGE_KEY is set and run training with API enabled."

                elif choice == '70':
                    # Enter free chat mode
                    self._free_chat_mode()
                    continue

                else:
                    print("\n❌ Invalid option. Please select a number from the menu.")
                    continue

                # Display the response
                if response:
                    print("\n" + "="*70)
                    print(response)
                    print("="*70)

                # Ask if user wants to continue
                while True:
                    continue_choice = input("\n🔄 Run another query? (y/n): ").strip().lower()
                    if continue_choice in ['y', 'yes']:
                        break  # Go back to menu
                    elif continue_choice in ['n', 'no']:
                        print("\n👋 Goodbye!\n")
                        return
                    else:
                        print("Please enter 'y' or 'n'")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()

    def _free_chat_mode(self):
        """Free-form chat mode (option 70)."""
        print("\n" + "="*70)
        print("💬 Free Chat Mode - Ask me anything!")
        print("="*70)
        print("Type 'menu' to return to menu, 'quit' to exit completely.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!\n")
                    exit(0)

                if user_input.lower() in ['menu', 'back']:
                    print("\n↩️  Returning to menu...\n")
                    break

                if not user_input:
                    continue

                response = self.query(user_input)
                print(f"\n🤖 Agent:\n{response}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                exit(0)
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def chat(self):
        """Legacy chat interface - redirects to menu."""
        self.menu()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stock Training Database with AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training on tech stocks"
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start AI agent chat interface"
    )

    parser.add_argument(
        "--history",
        type=int,
        default=365,
        help="Historical data period in days (default: 365)"
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback period for regime detection (default: 60)"
    )

    parser.add_argument(
        "--db-path",
        default="./stock_training.db",
        help="Database file path (default: ./stock_training.db)"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Initialize database
    db = StockTrainingDatabase(args.db_path)

    # Initialize Alpha Vantage if available
    av_provider = None
    av_key = os.getenv("ALPHA_VANTAGE_KEY")
    if av_key:
        av_provider = AlphaVantageProvider(av_key, max_calls_per_minute=5)
        print("✓ Alpha Vantage integration enabled\n")

    # Run training if requested
    if args.train:
        trainer = StockTrainer(
            db=db,
            av_provider=av_provider,
            lookback_regime=args.lookback,
            history_days=args.history
        )

        trainer.train_all_stocks(
            tickers=TECH_STOCKS_20,
            universe_name="tech_20"
        )

    # Start chat if requested
    if args.chat:
        agent = AIInsightAgent(db=db)
        agent.chat()

    # If neither flag, show help
    if not args.train and not args.chat:
        parser.print_help()
        print("\nQuick start:")
        print("  python stock_training_db.py --train          # Run training")
        print("  python stock_training_db.py --chat           # Chat with AI agent")
        print("  python stock_training_db.py --train --chat   # Train then chat")

    db.close()


if __name__ == "__main__":
    main()
