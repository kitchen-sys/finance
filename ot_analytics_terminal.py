#!/usr/bin/env python3
"""
OT Analytics Terminal - Comprehensive Quantitative Finance System
Combines real-time data, AI analysis, news, and RAG memory with institutional-grade LLM
"""
from __future__ import annotations

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from stock_training_db import StockTrainingDatabase, AIInsightAgent, PDFVectorStore, PDF_SUPPORT_AVAILABLE
from unified_terminal import (
    TECH_STOCKS_20, AI_STOCKS_20, ALL_STOCKS_TOP_20,
    StockDataFetcher, UnifiedTerminal
)

# Load environment variables
load_dotenv()


class NewsProvider:
    """Fetches financial news from various sources."""

    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_KEY")
        self.base_url = "https://www.alphavantage.co/query"

    def get_news_sentiment(self, tickers: Optional[List[str]] = None,
                          topics: Optional[List[str]] = None,
                          limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch news and sentiment from Alpha Vantage.

        Args:
            tickers: List of ticker symbols
            topics: List of topics (blockchain, earnings, ipo, mergers, etc.)
            limit: Maximum number of news items

        Returns:
            List of news articles with sentiment scores
        """
        if not self.alpha_vantage_key:
            print("⚠️  ALPHA_VANTAGE_KEY not set. News features unavailable.")
            return []

        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.alpha_vantage_key,
                "limit": limit
            }

            if tickers:
                params["tickers"] = ",".join(tickers)

            if topics:
                params["topics"] = ",".join(topics)

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if "feed" in data:
                return data["feed"]
            else:
                print(f"⚠️  Unexpected API response: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
                return []

        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            return []

    def format_news_summary(self, news_items: List[Dict[str, Any]], max_items: int = 10) -> str:
        """Format news items into readable summary."""
        if not news_items:
            return "No news available"

        summary = []
        summary.append(f"📰 Latest News ({len(news_items[:max_items])} items)\n")
        summary.append("=" * 100)

        for i, item in enumerate(news_items[:max_items], 1):
            title = item.get("title", "N/A")
            source = item.get("source", "Unknown")
            time_published = item.get("time_published", "")

            # Format timestamp
            if time_published:
                try:
                    dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    time_str = time_published

            # Overall sentiment
            overall_sentiment = item.get("overall_sentiment_label", "N/A")
            sentiment_score = item.get("overall_sentiment_score", 0)

            # Sentiment emoji
            if overall_sentiment == "Bullish":
                sentiment_emoji = "🟢"
            elif overall_sentiment == "Bearish":
                sentiment_emoji = "🔴"
            elif overall_sentiment == "Neutral":
                sentiment_emoji = "⚪"
            else:
                sentiment_emoji = "❔"

            summary.append(f"\n{i}. [{time_str}] {sentiment_emoji} {overall_sentiment} (Score: {sentiment_score:.3f})")
            summary.append(f"   Source: {source}")
            summary.append(f"   Title: {title[:97]}..." if len(title) > 100 else f"   Title: {title}")

            # Ticker-specific sentiment
            if "ticker_sentiment" in item and item["ticker_sentiment"]:
                tickers_sentiment = []
                for ts in item["ticker_sentiment"][:3]:  # Show top 3
                    ticker = ts.get("ticker", "")
                    relevance = float(ts.get("relevance_score", 0))
                    sentiment_label = ts.get("ticker_sentiment_label", "")
                    tickers_sentiment.append(f"{ticker} ({sentiment_label}, {relevance:.2f})")
                if tickers_sentiment:
                    summary.append(f"   Tickers: {', '.join(tickers_sentiment)}")

        summary.append("\n" + "=" * 100)
        return "\n".join(summary)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, model_name: str = "ot-analytics-quant", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m.get("name", "").startswith(self.model_name) for m in models)
            return False
        except:
            return False

    def generate(self, prompt: str, context: Optional[str] = None, stream: bool = False) -> str:
        """Generate response from Ollama model.

        Args:
            prompt: User prompt
            context: Additional context (stock data, news, etc.)
            stream: Whether to stream the response

        Returns:
            Model response
        """
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\nUser Query: {prompt}"

        try:
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": stream
            }

            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()

            if stream:
                # Handle streaming response
                full_response = []
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            chunk = data["response"]
                            print(chunk, end="", flush=True)
                            full_response.append(chunk)
                print()  # New line after streaming
                return "".join(full_response)
            else:
                data = response.json()
                return data.get("response", "No response generated")

        except Exception as e:
            return f"❌ Error generating response: {e}"


class OTAnalyticsTerminal:
    """
    Comprehensive OT Analytics Terminal
    Combines stock data, news, AI analysis, RAG memory, and institutional-grade LLM
    """

    def __init__(self):
        # Core components
        self.db = StockTrainingDatabase()
        self.agent = AIInsightAgent(db=self.db)
        self.fetcher = StockDataFetcher()
        self.news_provider = NewsProvider()

        # Ollama integration
        self.ollama = OllamaClient()
        self.ollama_available = self.ollama.is_available()

        # RAG memory (PDF vector store)
        self._pdf_store: Optional[PDFVectorStore] = None

        # Session data
        self.last_displayed_data: Optional[pd.DataFrame] = None
        self.session_history: List[Dict[str, Any]] = []
        self.latest_news: List[Dict[str, Any]] = []
        self.auto_loaded_data: Dict[str, Any] = {}

    @property
    def pdf_store(self) -> Optional[PDFVectorStore]:
        """Lazy initialization of PDF vector store."""
        if not PDF_SUPPORT_AVAILABLE:
            return None
        if self._pdf_store is None:
            try:
                self._pdf_store = PDFVectorStore()
            except Exception as e:
                print(f"⚠️  PDF store initialization failed: {e}")
                return None
        return self._pdf_store

    def startup_sequence(self):
        """Auto-load data on startup."""
        print("\n" + "="*100)
        print("🚀 OT ANALYTICS TERMINAL - Institutional-Grade Quantitative Finance System")
        print("="*100)
        print("\n⚡ Initializing components...")

        # Check Ollama
        if self.ollama_available:
            print(f"✅ Ollama connected: {self.ollama.model_name}")
        else:
            print("⚠️  Ollama not available (install: ollama pull llama3:8b && ollama create ot-analytics-quant -f Modelfile)")

        # Check training database
        latest_run = self.db.get_latest_run()
        if latest_run:
            print(f"✅ Training database: {latest_run['num_stocks_trained']} stocks (Run ID: {latest_run['run_id']})")
            self.auto_loaded_data['training_run'] = latest_run
        else:
            print("⚠️  No training data (run: python stock_training_db.py --train)")

        # Check PDF store
        if self.pdf_store:
            stats = self.pdf_store.get_stats()
            if stats['num_pdfs'] > 0:
                print(f"✅ RAG Memory: {stats['num_pdfs']} PDFs, {stats['num_chunks']} chunks")
                self.auto_loaded_data['pdf_stats'] = stats
            else:
                print("💡 RAG Memory: Ready (no PDFs ingested yet)")
        else:
            print("⚠️  RAG Memory: Unavailable (install: pip install chromadb pypdf langchain-text-splitters)")

        # Auto-load latest news
        print("\n📰 Loading latest market news...")
        self.latest_news = self.news_provider.get_news_sentiment(
            topics=["financial_markets", "economy_macro", "technology"],
            limit=20
        )
        if self.latest_news:
            print(f"✅ Loaded {len(self.latest_news)} news articles")
            self.auto_loaded_data['news_count'] = len(self.latest_news)
        else:
            print("⚠️  News unavailable (set ALPHA_VANTAGE_KEY in .env)")

        # Auto-load market snapshot
        print("\n📊 Loading market snapshot...")
        try:
            # Quick snapshot of major indices/stocks
            snapshot_tickers = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"]
            snapshot_df = self.fetcher.get_stock_data(snapshot_tickers, period="1d")
            if not snapshot_df.empty:
                print(f"✅ Market snapshot: {len(snapshot_df)} securities")
                self.auto_loaded_data['market_snapshot'] = snapshot_df.to_dict('records')
            else:
                print("⚠️  Market snapshot unavailable")
        except Exception as e:
            print(f"⚠️  Market snapshot error: {e}")

        print("\n" + "="*100)
        print("✅ System ready. Type 'help' for commands or 'auto' for automatic briefing")
        print("="*100 + "\n")

    def generate_automatic_briefing(self):
        """Generate comprehensive market briefing using all available data."""
        print("\n" + "="*100)
        print("📋 AUTOMATIC MARKET BRIEFING")
        print("="*100 + "\n")

        briefing_parts = []

        # 1. Market snapshot
        if 'market_snapshot' in self.auto_loaded_data:
            briefing_parts.append("📊 MARKET SNAPSHOT:")
            snapshot = pd.DataFrame(self.auto_loaded_data['market_snapshot'])
            avg_return = snapshot['return_pct'].mean()
            briefing_parts.append(f"Average 1-day return: {avg_return:+.2f}%")
            for _, row in snapshot.iterrows():
                briefing_parts.append(f"  • {row['ticker']}: ${row['price']:.2f} ({row['return_pct']:+.2f}%)")
            briefing_parts.append("")

        # 2. Latest news summary
        if self.latest_news:
            briefing_parts.append("📰 NEWS HIGHLIGHTS (Top 5):")
            for i, news in enumerate(self.latest_news[:5], 1):
                sentiment = news.get("overall_sentiment_label", "N/A")
                title = news.get("title", "")[:80]
                briefing_parts.append(f"{i}. [{sentiment}] {title}...")
            briefing_parts.append("")

        # 3. Training database insights
        if 'training_run' in self.auto_loaded_data:
            latest_run = self.auto_loaded_data['training_run']
            results = self.db.get_run_results(latest_run['run_id'])
            successful = [r for r in results if not r['error_message']]

            if successful:
                briefing_parts.append("📈 TRAINING DATABASE INSIGHTS:")
                # Top performers
                top_perf = sorted(
                    [r for r in successful if r['return_60d'] is not None],
                    key=lambda x: x['return_60d'],
                    reverse=True
                )[:3]
                briefing_parts.append("Top 60-day performers:")
                for r in top_perf:
                    briefing_parts.append(f"  • {r['ticker']}: {r['return_60d']*100:+.2f}%")

                # Regime changes
                regime_changes = [r for r in successful if r['regime_change']]
                if regime_changes:
                    briefing_parts.append(f"\n⚠️  Regime changes detected: {len(regime_changes)} stocks")
                    for r in regime_changes[:3]:
                        briefing_parts.append(f"  • {r['ticker']}: {r['regime_state']}")
                briefing_parts.append("")

        # 4. RAG Memory status
        if 'pdf_stats' in self.auto_loaded_data:
            stats = self.auto_loaded_data['pdf_stats']
            briefing_parts.append(f"💾 RAG MEMORY: {stats['num_pdfs']} documents indexed")
            briefing_parts.append("")

        briefing_text = "\n".join(briefing_parts)
        print(briefing_text)

        # Generate AI insights if Ollama available
        if self.ollama_available:
            print("🤖 GENERATING AI ANALYSIS (OT-Analytics-Quant-Model)...")
            print("-" * 100 + "\n")

            context = f"""You are analyzing current market conditions. Here is the data:

{briefing_text}

Provide institutional-grade analysis including:
1. Market regime assessment
2. Key risks and opportunities
3. Specific trading recommendations with entry/exit levels
4. Risk management considerations
"""

            response = self.ollama.generate(
                "Analyze the current market conditions and provide trading insights.",
                context=context,
                stream=True
            )
        else:
            print("💡 Install Ollama for AI-powered analysis\n")

        print("\n" + "="*100 + "\n")

    def query_rag_memory(self, query: str, n_results: int = 3) -> str:
        """Query RAG memory (PDF vector store) for relevant context."""
        if not self.pdf_store:
            return ""

        try:
            results = self.pdf_store.search(query, n_results=n_results)
            if not results:
                return ""

            context_parts = [f"=== RELEVANT DOCUMENT EXCERPTS ({len(results)} found) ===\n"]
            for i, result in enumerate(results, 1):
                meta = result['metadata']
                text = result['text'][:500]  # Limit context size
                context_parts.append(f"[{i}] From {meta['filename']} (Page {meta['page_number']}):")
                context_parts.append(f"{text}\n")

            return "\n".join(context_parts)
        except Exception as e:
            print(f"⚠️  RAG query error: {e}")
            return ""

    def analyze_with_ollama(self, query: str, include_news: bool = True, include_rag: bool = True):
        """Analyze using Ollama with full context."""
        if not self.ollama_available:
            print("❌ Ollama not available. Install and create model first.")
            print("   1. Install Ollama: https://ollama.ai")
            print("   2. ollama pull llama3:8b")
            print("   3. ollama create ot-analytics-quant -f Modelfile")
            return

        print("\n🤖 OT-Analytics-Quant-Model Analysis")
        print("="*100 + "\n")

        # Build comprehensive context
        context_parts = []

        # Current stock data
        if self.last_displayed_data is not None:
            context_parts.append("=== CURRENT STOCK DATA ===")
            df = self.last_displayed_data
            context_parts.append(f"Showing {len(df)} stocks:")
            for _, row in df.head(10).iterrows():
                context_parts.append(
                    f"  {row['ticker']}: ${row['price']:.2f} ({row['return_pct']:+.2f}%), "
                    f"MCap: ${row['market_cap']/1e9:.1f}B, P/E: {row['pe_ratio']:.1f if row['pe_ratio'] else 'N/A'}"
                )
            context_parts.append("")

        # News context
        if include_news and self.latest_news:
            context_parts.append("=== RECENT NEWS ===")
            for news in self.latest_news[:5]:
                sentiment = news.get("overall_sentiment_label", "N/A")
                title = news.get("title", "")[:100]
                context_parts.append(f"[{sentiment}] {title}")
            context_parts.append("")

        # RAG memory context
        if include_rag:
            rag_context = self.query_rag_memory(query, n_results=3)
            if rag_context:
                context_parts.append(rag_context)

        # Training database context
        latest_run = self.db.get_latest_run()
        if latest_run:
            results = self.db.get_run_results(latest_run['run_id'])
            successful = [r for r in results if not r['error_message']]
            if successful:
                context_parts.append("=== TRAINING DATABASE STATS ===")
                context_parts.append(f"Total stocks analyzed: {len(successful)}")
                avg_return = sum(r['return_60d'] for r in successful if r['return_60d']) / len([r for r in successful if r['return_60d']])
                context_parts.append(f"Average 60-day return: {avg_return*100:.2f}%")
                context_parts.append("")

        full_context = "\n".join(context_parts)

        # Generate response
        print("Generating analysis...\n")
        response = self.ollama.generate(query, context=full_context, stream=True)
        print("\n" + "="*100 + "\n")

    def show_news(self, tickers: Optional[List[str]] = None, limit: int = 10):
        """Display latest news."""
        print(f"\n📰 Fetching latest news{f' for {len(tickers)} tickers' if tickers else ''}...")

        news = self.news_provider.get_news_sentiment(tickers=tickers, limit=limit)
        if news:
            self.latest_news = news
            print(self.news_provider.format_news_summary(news, max_items=limit))
        else:
            print("📭 No news available\n")

    def run(self):
        """Run the terminal."""
        # Run startup sequence
        self.startup_sequence()

        # Inherit from unified terminal for base commands
        base_terminal = UnifiedTerminal()
        base_terminal.db = self.db
        base_terminal.agent = self.agent
        base_terminal.fetcher = self.fetcher

        while True:
            try:
                user_input = input("OT-Analytics> ").strip()

                if not user_input:
                    continue

                # Parse command
                parts = user_input.lower().split(maxsplit=1)
                command = parts[0]
                args = parts[1] if len(parts) > 1 else ""

                # Handle quit/exit
                if command in ['quit', 'exit', 'q']:
                    print("\n👋 Session ended. Happy trading!\n")
                    break

                # OT Analytics specific commands
                elif command == 'auto':
                    self.generate_automatic_briefing()

                elif command == 'news':
                    if args:
                        # Parse tickers
                        ticker_str = args
                        tickers = [t.strip().upper() for t in ticker_str.replace(',', ' ').split()]
                        self.show_news(tickers=tickers)
                    else:
                        self.show_news()

                elif command in ['ai', 'ask', 'query']:
                    if args:
                        self.analyze_with_ollama(args)
                    else:
                        print("⚠️  Usage: ai <your question>")

                elif command == 'rag':
                    if args:
                        if self.pdf_store:
                            results = self.pdf_store.search(args, n_results=5)
                            if results:
                                print(f"\n🔍 RAG Memory Search: '{args}'")
                                print("="*100 + "\n")
                                for i, result in enumerate(results, 1):
                                    meta = result['metadata']
                                    text = result['text'][:300]
                                    print(f"{i}. {meta['filename']} (Page {meta['page_number']})")
                                    print(f"   {text}...\n")
                            else:
                                print("📭 No relevant documents found\n")
                        else:
                            print("⚠️  RAG memory not available\n")
                    else:
                        print("⚠️  Usage: rag <search query>")

                elif command == 'reload':
                    print("\n♻️  Reloading data...")
                    self.startup_sequence()

                # Delegate to base terminal commands
                elif command == 'show':
                    self.last_displayed_data = base_terminal.last_displayed_data
                    if not args:
                        print("⚠️  Usage: show <tech|ai|all|custom TICKERS>")
                        continue

                    subcommand = args.split()[0]

                    if subcommand == 'tech':
                        base_terminal.show_tech_stocks()
                        self.last_displayed_data = base_terminal.last_displayed_data
                    elif subcommand == 'ai':
                        base_terminal.show_ai_stocks()
                        self.last_displayed_data = base_terminal.last_displayed_data
                    elif subcommand == 'all':
                        base_terminal.show_all_stocks()
                        self.last_displayed_data = base_terminal.last_displayed_data
                    elif subcommand == 'custom':
                        ticker_str = ' '.join(args.split()[1:])
                        tickers = [t.strip().upper() for t in ticker_str.replace(',', ' ').split()]
                        if tickers:
                            base_terminal.show_custom_stocks(tickers)
                            self.last_displayed_data = base_terminal.last_displayed_data
                        else:
                            print("⚠️  Usage: show custom AAPL,MSFT,GOOGL")
                    else:
                        print(f"⚠️  Unknown show command: {subcommand}")

                elif command == 'analyze':
                    base_terminal.analyze_with_ai()

                elif command == 'history':
                    base_terminal.show_history()

                elif command == 'export':
                    if args:
                        base_terminal.export_data(args)
                    else:
                        print("⚠️  Usage: export <filename>")

                elif command == 'menu':
                    print("\n🔄 Switching to stock training menu...\n")
                    self.agent.menu()

                elif command == 'pdf':
                    print("\n🔄 Switching to PDF ingestion system...\n")
                    self.agent._pdf_ingest_mode()

                elif command == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')

                elif command in ['help', '?', 'h']:
                    self.show_help()

                else:
                    print(f"❌ Unknown command: '{command}'")
                    print("   Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n\n👋 Session ended. Happy trading!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()

    def show_help(self):
        """Display comprehensive help."""
        print("\n" + "="*100)
        print("📚 OT ANALYTICS TERMINAL - Command Reference")
        print("="*100)
        print("\n🚀 AUTO-LOADING COMMANDS:")
        print("  auto                   - Generate automatic market briefing with AI analysis")
        print("  reload                 - Reload all data (news, market snapshot, training data)")
        print("\n📊 STOCK DISPLAY:")
        print("  show tech              - Display top 20 tech stocks")
        print("  show ai                - Display top 20 AI stocks")
        print("  show all               - Display top 20 stocks across all categories")
        print("  show custom <tickers>  - Display custom stock list (e.g., show custom AAPL,MSFT)")
        print("\n🤖 AI ANALYSIS:")
        print("  ai <question>          - Query OT-Analytics-Quant-Model with full context")
        print("  ask <question>         - Alias for 'ai'")
        print("  analyze                - Analyze last displayed stocks (structured analysis)")
        print("\n📰 NEWS:")
        print("  news                   - Show latest market news")
        print("  news AAPL,MSFT        - Show news for specific tickers")
        print("\n💾 RAG MEMORY:")
        print("  rag <query>            - Search PDF vector database for relevant context")
        print("  pdf                    - Access PDF ingestion system")
        print("\n💿 DATA MANAGEMENT:")
        print("  history                - Show session command history")
        print("  export <filename>      - Export last displayed data to CSV")
        print("\n🔧 SYSTEM:")
        print("  menu                   - Access stock training database menu")
        print("  clear                  - Clear screen")
        print("  help / ?               - Show this help message")
        print("  quit / exit            - Exit terminal")
        print("\n💡 WORKFLOW EXAMPLE:")
        print("  1. auto                    # Get automatic briefing")
        print("  2. show tech               # Display tech stocks")
        print("  3. news NVDA,AMD           # Get news for specific stocks")
        print("  4. ai analyze nvidia       # Get AI analysis with full context")
        print("  5. export analysis.csv     # Save data")
        print("\n🔑 FEATURES:")
        print("  • Automatic data loading on startup")
        print("  • Real-time stock data + news integration")
        print("  • Institutional-grade AI model (OT-Analytics-Quant)")
        print("  • RAG memory with PDF vector database")
        print("  • Training database with historical analysis")
        print("  • Comprehensive context for AI queries")
        print("="*100 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="OT Analytics Terminal - Institutional-Grade Quantitative Finance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run automatic briefing demo"
    )

    args = parser.parse_args()

    terminal = OTAnalyticsTerminal()

    if args.demo:
        terminal.startup_sequence()
        terminal.generate_automatic_briefing()
    else:
        terminal.run()


if __name__ == "__main__":
    main()
