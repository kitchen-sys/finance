#!/usr/bin/env python3
"""
Unified AI-Stock Market Terminal
Combines real-time stock data display with AI-powered analysis
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from stock_training_db import StockTrainingDatabase, AIInsightAgent

# Load environment variables
load_dotenv()


# Stock Universe Definitions
TECH_STOCKS_20 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AVGO",
    "ORCL", "ADBE", "CRM", "CSCO", "INTC", "AMD", "QCOM", "TXN",
    "INTU", "IBM", "NOW", "SHOP"
]

AI_STOCKS_20 = [
    "NVDA",   # NVIDIA - AI chips
    "MSFT",   # Microsoft - Azure AI, OpenAI partnership
    "GOOGL",  # Google - Gemini, DeepMind
    "META",   # Meta - LLaMA, AI research
    "AMZN",   # Amazon - AWS AI services
    "TSLA",   # Tesla - Autopilot, FSD
    "AMD",    # AMD - AI accelerators
    "ORCL",   # Oracle - Cloud AI
    "CRM",    # Salesforce - Einstein AI
    "ADBE",   # Adobe - Firefly AI
    "NOW",    # ServiceNow - AI automation
    "PLTR",   # Palantir - AI platforms
    "SNOW",   # Snowflake - Data AI
    "DDOG",   # Datadog - AI monitoring
    "PATH",   # UiPath - AI automation
    "AI",     # C3.ai - Enterprise AI
    "BBAI",   # BigBear.ai
    "SOUN",   # SoundHound AI
    "BZFD",   # BuzzFeed - AI content
    "RELY",   # Remitly - AI fintech
]

ALL_STOCKS_TOP_20 = [
    # Mega cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Finance
    "JPM", "BAC", "WFC",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Consumer
    "WMT", "PG", "KO",
    # Energy
    "XOM", "CVX",
    # Other
    "V", "MA"
]


class StockDataFetcher:
    """Fetches real-time stock data from various sources."""

    @staticmethod
    def get_stock_data(tickers: List[str], period: str = "5d") -> pd.DataFrame:
        """Fetch stock data for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            DataFrame with stock data
        """
        data = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)

                if hist.empty:
                    continue

                # Get current and previous prices
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[0]

                # Calculate returns
                return_pct = ((current_price - prev_price) / prev_price) * 100

                # Get additional info
                info = stock.info
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', None)
                volume = hist['Volume'].iloc[-1] if len(hist) > 0 else 0

                data.append({
                    'ticker': ticker,
                    'price': current_price,
                    'return_pct': return_pct,
                    'market_cap': market_cap,
                    'pe_ratio': pe_ratio,
                    'volume': volume,
                    'name': info.get('longName', ticker)
                })

            except Exception as e:
                print(f"⚠️  Error fetching {ticker}: {str(e)[:50]}")
                continue

        return pd.DataFrame(data)


class UnifiedTerminal:
    """Unified terminal combining stock data display and AI analysis."""

    def __init__(self):
        self.db = StockTrainingDatabase()
        self.agent = AIInsightAgent(db=self.db)
        self.fetcher = StockDataFetcher()
        self.last_displayed_data: Optional[pd.DataFrame] = None
        self.session_history: List[Dict[str, Any]] = []

    def _format_market_cap(self, market_cap: float) -> str:
        """Format market cap in human-readable format."""
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"${market_cap/1e6:.2f}M"
        else:
            return f"${market_cap:.0f}"

    def display_stock_table(self, df: pd.DataFrame, title: str):
        """Display stock data in formatted table."""
        if df.empty:
            print("\n📭 No data available\n")
            return

        # Sort by return percentage
        df = df.sort_values('return_pct', ascending=False)

        print("\n" + "="*100)
        print(f"📊 {title}")
        print("="*100)
        print(f"{'#':<4} {'Ticker':<8} {'Name':<30} {'Price':<12} {'Return %':<12} {'Market Cap':<15} {'P/E':<8}")
        print("-"*100)

        for idx, row in df.iterrows():
            rank = df.index.get_loc(idx) + 1
            ticker = row['ticker']
            name = row['name'][:27] + "..." if len(row['name']) > 30 else row['name']
            price = f"${row['price']:.2f}"
            return_pct = f"{row['return_pct']:+.2f}%"
            market_cap = self._format_market_cap(row['market_cap'])
            pe = f"{row['pe_ratio']:.2f}" if row['pe_ratio'] else "N/A"

            # Color coding for returns
            if row['return_pct'] > 0:
                return_symbol = "🟢"
            elif row['return_pct'] < 0:
                return_symbol = "🔴"
            else:
                return_symbol = "⚪"

            print(f"{rank:<4} {ticker:<8} {name:<30} {price:<12} {return_symbol} {return_pct:<10} {market_cap:<15} {pe:<8}")

        print("="*100)

        # Summary statistics
        avg_return = df['return_pct'].mean()
        positive_count = (df['return_pct'] > 0).sum()
        negative_count = (df['return_pct'] < 0).sum()

        print(f"\n📈 Summary:")
        print(f"   Average Return: {avg_return:+.2f}%")
        print(f"   Positive: {positive_count} | Negative: {negative_count}")
        print(f"   Top Performer: {df.iloc[0]['ticker']} ({df.iloc[0]['return_pct']:+.2f}%)")
        print(f"   Bottom Performer: {df.iloc[-1]['ticker']} ({df.iloc[-1]['return_pct']:+.2f}%)")
        print()

        # Store for AI analysis
        self.last_displayed_data = df
        self.session_history.append({
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'data': df.to_dict('records')
        })

    def show_tech_stocks(self):
        """Display top 20 tech stocks."""
        print("\n🔍 Fetching top 20 tech stocks...")
        df = self.fetcher.get_stock_data(TECH_STOCKS_20, period="5d")
        self.display_stock_table(df, "Top 20 Tech Stocks (5-Day Performance)")

    def show_ai_stocks(self):
        """Display top 20 AI stocks."""
        print("\n🔍 Fetching top 20 AI stocks...")
        df = self.fetcher.get_stock_data(AI_STOCKS_20, period="5d")
        self.display_stock_table(df, "Top 20 AI Stocks (5-Day Performance)")

    def show_all_stocks(self):
        """Display top 20 stocks across all categories."""
        print("\n🔍 Fetching top 20 stocks across all categories...")
        df = self.fetcher.get_stock_data(ALL_STOCKS_TOP_20, period="5d")
        self.display_stock_table(df, "Top 20 Stocks - All Categories (5-Day Performance)")

    def show_custom_stocks(self, tickers: List[str], period: str = "5d"):
        """Display custom list of stocks."""
        print(f"\n🔍 Fetching data for {len(tickers)} stocks...")
        df = self.fetcher.get_stock_data(tickers, period=period)
        self.display_stock_table(df, f"Custom Stock List ({period} Performance)")

    def analyze_with_ai(self, query: Optional[str] = None):
        """Analyze current data with AI agent."""
        if self.last_displayed_data is None:
            print("\n⚠️  No stock data displayed yet. Please display some stocks first.")
            print("   Try: show tech, show ai, or show all")
            return

        # Build context from last displayed data
        df = self.last_displayed_data

        if query is None:
            # Auto-generate analysis
            context = f"""Analyze the following stock performance data:

Top 5 Performers:
"""
            for idx, row in df.head(5).iterrows():
                context += f"- {row['ticker']}: {row['return_pct']:+.2f}% (Market Cap: {self._format_market_cap(row['market_cap'])})\n"

            context += f"\nBottom 5 Performers:\n"
            for idx, row in df.tail(5).iterrows():
                context += f"- {row['ticker']}: {row['return_pct']:+.2f}% (Market Cap: {self._format_market_cap(row['market_cap'])})\n"

            context += f"\nAverage Return: {df['return_pct'].mean():+.2f}%\n"
            context += f"\nProvide insights on this market performance, potential trends, and trading considerations."

            query = context

        print("\n🤖 AI Agent Analysis:")
        print("="*100)

        # Use the AI agent to analyze
        try:
            # For now, just provide structured analysis based on data
            # In full implementation, this would use the agent's query method
            self._provide_structured_analysis(df)
        except Exception as e:
            print(f"❌ Error during AI analysis: {e}")

    def _provide_structured_analysis(self, df: pd.DataFrame):
        """Provide structured analysis of stock data."""
        print("\n📊 Market Analysis:\n")

        # Trend analysis
        avg_return = df['return_pct'].mean()
        if avg_return > 2:
            trend = "🟢 BULLISH"
            sentiment = "Strong positive momentum across the board"
        elif avg_return > 0:
            trend = "🟢 SLIGHTLY BULLISH"
            sentiment = "Modest gains, cautiously optimistic"
        elif avg_return > -2:
            trend = "🔴 SLIGHTLY BEARISH"
            sentiment = "Minor losses, market uncertainty"
        else:
            trend = "🔴 BEARISH"
            sentiment = "Significant downward pressure"

        print(f"Market Trend: {trend}")
        print(f"Sentiment: {sentiment}")
        print(f"Average Return: {avg_return:+.2f}%\n")

        # Sector leaders and laggards
        print("🏆 Top 3 Performers:")
        for idx, row in df.head(3).iterrows():
            print(f"   {df.index.get_loc(idx) + 1}. {row['ticker']} ({row['name'][:40]})")
            print(f"      Return: {row['return_pct']:+.2f}% | Market Cap: {self._format_market_cap(row['market_cap'])}")

        print("\n⚠️  Bottom 3 Performers:")
        for idx, row in df.tail(3).iterrows():
            print(f"   {len(df) - df[::-1].index.get_loc(idx)}. {row['ticker']} ({row['name'][:40]})")
            print(f"      Return: {row['return_pct']:+.2f}% | Market Cap: {self._format_market_cap(row['market_cap'])}")

        # Risk analysis
        volatility = df['return_pct'].std()
        print(f"\n📊 Volatility: {volatility:.2f}%")

        if volatility > 5:
            print("   ⚠️  High volatility - increased risk")
        elif volatility > 3:
            print("   🟡 Moderate volatility - normal market conditions")
        else:
            print("   ✅ Low volatility - stable market")

        # Trading recommendations
        print("\n💡 Trading Insights:")

        # Identify strong performers
        strong_performers = df[df['return_pct'] > df['return_pct'].quantile(0.75)]
        if len(strong_performers) > 0:
            print(f"   • {len(strong_performers)} stocks showing strong momentum")
            print(f"     Consider: {', '.join(strong_performers['ticker'].head(3).tolist())}")

        # Identify potential value plays
        weak_performers = df[df['return_pct'] < df['return_pct'].quantile(0.25)]
        if len(weak_performers) > 0:
            print(f"   • {len(weak_performers)} stocks underperforming")
            print(f"     Watch for reversal: {', '.join(weak_performers['ticker'].head(3).tolist())}")

        # Market cap analysis
        large_cap = df[df['market_cap'] > 100e9]
        if len(large_cap) > 0:
            large_cap_return = large_cap['return_pct'].mean()
            print(f"   • Large-cap avg return: {large_cap_return:+.2f}%")

        print("\n" + "="*100 + "\n")

    def show_help(self):
        """Display help information."""
        print("\n" + "="*100)
        print("📚 Unified AI-Stock Market Terminal - Command Reference")
        print("="*100)
        print("\n📊 STOCK DISPLAY COMMANDS:")
        print("  show tech              - Display top 20 tech stocks")
        print("  show ai                - Display top 20 AI stocks")
        print("  show all               - Display top 20 stocks across all categories")
        print("  show custom <tickers>  - Display custom stock list (comma-separated)")
        print("                           Example: show custom AAPL,MSFT,GOOGL")
        print("\n🤖 AI ANALYSIS COMMANDS:")
        print("  analyze                - AI analysis of last displayed stocks")
        print("  ask <question>         - Ask AI agent a specific question")
        print("  insights               - Get AI insights on current market data")
        print("\n💾 DATA COMMANDS:")
        print("  history                - Show command history for this session")
        print("  export <filename>      - Export last displayed data to CSV")
        print("\n🔧 SYSTEM COMMANDS:")
        print("  menu                   - Access original stock training menu")
        print("  pdf                    - Access PDF ingestion system (option 71)")
        print("  clear                  - Clear screen")
        print("  help / ?               - Show this help message")
        print("  quit / exit            - Exit terminal")
        print("\n💡 TIPS:")
        print("  • Commands are case-insensitive")
        print("  • After displaying stocks, use 'analyze' for AI insights")
        print("  • Combine commands: display stocks → analyze → ask questions")
        print("="*100 + "\n")

    def show_history(self):
        """Show session command history."""
        if not self.session_history:
            print("\n📭 No commands executed in this session yet\n")
            return

        print("\n" + "="*100)
        print("📜 Session History")
        print("="*100)

        for i, entry in enumerate(self.session_history, 1):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
            print(f"\n{i}. [{timestamp}] {entry['title']}")
            print(f"   Stocks displayed: {len(entry['data'])}")

        print("\n" + "="*100 + "\n")

    def export_data(self, filename: str):
        """Export last displayed data to CSV."""
        if self.last_displayed_data is None:
            print("\n⚠️  No data to export. Display some stocks first.\n")
            return

        try:
            filepath = Path(filename)
            if not filepath.suffix:
                filepath = filepath.with_suffix('.csv')

            self.last_displayed_data.to_csv(filepath, index=False)
            print(f"\n✅ Data exported to: {filepath}\n")
        except Exception as e:
            print(f"\n❌ Export failed: {e}\n")

    def run(self):
        """Run the unified terminal."""
        print("\n" + "="*100)
        print("🚀 Unified AI-Stock Market Terminal")
        print("="*100)
        print("\nCombining real-time stock data with AI-powered analysis")
        print("Type 'help' for available commands\n")

        while True:
            try:
                user_input = input("Terminal> ").strip()

                if not user_input:
                    continue

                # Parse command
                parts = user_input.lower().split(maxsplit=1)
                command = parts[0]
                args = parts[1] if len(parts) > 1 else ""

                # Handle quit/exit
                if command in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break

                # Handle help
                elif command in ['help', '?', 'h']:
                    self.show_help()

                # Handle stock display commands
                elif command == 'show':
                    if not args:
                        print("⚠️  Usage: show <tech|ai|all|custom TICKERS>")
                        continue

                    subcommand = args.split()[0]

                    if subcommand == 'tech':
                        self.show_tech_stocks()
                    elif subcommand == 'ai':
                        self.show_ai_stocks()
                    elif subcommand == 'all':
                        self.show_all_stocks()
                    elif subcommand == 'custom':
                        # Parse custom tickers
                        ticker_str = ' '.join(args.split()[1:])
                        tickers = [t.strip().upper() for t in ticker_str.replace(',', ' ').split()]
                        if tickers:
                            self.show_custom_stocks(tickers)
                        else:
                            print("⚠️  Usage: show custom AAPL,MSFT,GOOGL")
                    else:
                        print(f"⚠️  Unknown show command: {subcommand}")
                        print("   Available: tech, ai, all, custom")

                # Handle AI analysis
                elif command in ['analyze', 'analysis']:
                    self.analyze_with_ai()

                elif command in ['ask', 'query']:
                    if args:
                        # Use actual AI agent if available
                        print("\n🤖 AI Agent Response:")
                        print("="*100)
                        try:
                            response = self.agent.query(args)
                            print(response)
                        except Exception as e:
                            print(f"❌ Error: {e}")
                            print("\nNote: Make sure training data exists (run with --train)")
                        print("="*100 + "\n")
                    else:
                        print("⚠️  Usage: ask <your question>")

                elif command in ['insights', 'insight']:
                    self.analyze_with_ai()

                # Handle data commands
                elif command == 'history':
                    self.show_history()

                elif command == 'export':
                    if args:
                        self.export_data(args)
                    else:
                        print("⚠️  Usage: export <filename>")

                # Handle system commands
                elif command == 'menu':
                    print("\n🔄 Switching to stock training menu...\n")
                    self.agent.menu()

                elif command == 'pdf':
                    print("\n🔄 Switching to PDF ingestion system...\n")
                    self.agent._pdf_ingest_mode()

                elif command == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')

                else:
                    print(f"❌ Unknown command: '{command}'")
                    print("   Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified AI-Stock Market Terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick demo of the terminal"
    )

    args = parser.parse_args()

    terminal = UnifiedTerminal()

    if args.demo:
        print("\n🎬 Running Demo Mode...\n")
        print("Displaying Tech Stocks:")
        terminal.show_tech_stocks()
        print("\nAnalyzing with AI:")
        terminal.analyze_with_ai()
        print("\n✅ Demo complete. Run without --demo for interactive mode.\n")
    else:
        terminal.run()


if __name__ == "__main__":
    main()
