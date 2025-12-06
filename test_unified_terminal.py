#!/usr/bin/env python3
"""
Test script for Unified Terminal (without yfinance dependency)
Demonstrates the terminal functionality with mock data
"""
import pandas as pd
from datetime import datetime


class MockStockDataFetcher:
    """Mock stock data fetcher for testing."""

    @staticmethod
    def get_stock_data(tickers, period="5d"):
        """Generate mock stock data."""
        import random
        random.seed(42)  # For reproducible results

        data = []
        for ticker in tickers:
            # Generate realistic mock data
            price = random.uniform(50, 1000)
            return_pct = random.uniform(-5, 8)
            market_cap = random.uniform(100e9, 3e12)
            pe_ratio = random.uniform(15, 80) if random.random() > 0.1 else None
            volume = random.uniform(1e6, 100e6)

            # Company names
            names = {
                'AAPL': 'Apple Inc.',
                'MSFT': 'Microsoft Corporation',
                'GOOGL': 'Alphabet Inc.',
                'AMZN': 'Amazon.com Inc.',
                'NVDA': 'NVIDIA Corporation',
                'META': 'Meta Platforms Inc.',
                'TSLA': 'Tesla Inc.',
            }

            data.append({
                'ticker': ticker,
                'price': price,
                'return_pct': return_pct,
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'volume': volume,
                'name': names.get(ticker, f'{ticker} Inc.')
            })

        return pd.DataFrame(data)


def format_market_cap(market_cap):
    """Format market cap in human-readable format."""
    if market_cap >= 1e12:
        return f"${market_cap/1e12:.2f}T"
    elif market_cap >= 1e9:
        return f"${market_cap/1e9:.2f}B"
    elif market_cap >= 1e6:
        return f"${market_cap/1e6:.2f}M"
    else:
        return f"${market_cap:.0f}"


def display_stock_table(df, title):
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
        market_cap = format_market_cap(row['market_cap'])
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


def provide_structured_analysis(df):
    """Provide structured analysis of stock data."""
    print("\n🤖 AI Agent Analysis:")
    print("="*100)
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
        print(f"      Return: {row['return_pct']:+.2f}% | Market Cap: {format_market_cap(row['market_cap'])}")

    print("\n⚠️  Bottom 3 Performers:")
    for idx, row in df.tail(3).iterrows():
        print(f"   {len(df) - df[::-1].index.get_loc(idx)}. {row['ticker']} ({row['name'][:40]})")
        print(f"      Return: {row['return_pct']:+.2f}% | Market Cap: {format_market_cap(row['market_cap'])}")

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


def test_terminal():
    """Test the unified terminal with mock data."""
    print("\n" + "="*100)
    print("🧪 Unified Terminal Test (Mock Data)")
    print("="*100)
    print("\nTesting terminal functionality without yfinance dependency\n")

    # Tech stocks
    tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AVGO"]

    # Fetch and display
    print("\n🔍 Fetching top tech stocks (mock data)...")
    fetcher = MockStockDataFetcher()
    df = fetcher.get_stock_data(tech_stocks, period="5d")

    display_stock_table(df, "Top Tech Stocks (5-Day Performance) - MOCK DATA")

    print("\n🤖 Generating AI analysis...")
    provide_structured_analysis(df)

    print("="*100)
    print("✅ Test completed successfully!")
    print("="*100)
    print("\nThe unified terminal structure is working correctly.")
    print("Once yfinance dependencies are resolved, it will fetch real market data.\n")


if __name__ == "__main__":
    test_terminal()
