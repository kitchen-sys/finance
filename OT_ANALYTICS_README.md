# OT Analytics Terminal

**Institutional-Grade Quantitative Finance System with AI, RAG Memory, and Auto-Loading**

The complete, production-ready terminal combining real-time market data, news, AI analysis, and RAG memory with an institutional-grade LLM optimized for quantitative finance.

## 🌟 Overview

OT Analytics Terminal is the third and final wrapper that brings together:

1. **Real-Time Stock Data** - Live market data for Tech, AI, and All category stocks
2. **News Integration** - Alpha Vantage News API with sentiment analysis
3. **AI Analysis** - Ollama-powered institutional-grade quant model (OT-Analytics-Quant-Model)
4. **RAG Memory** - PDF vector database for document Q&A
5. **Training Database** - Historical stock analysis and regime detection
6. **Automatic Data Loading** - All data auto-loads on startup

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│           OT Analytics Terminal (Main Interface)          │
│  Auto-loading │ News │ AI │ RAG │ Stock Data │ Training   │
└────────────┬───────────────────────────────────────────────┘
             │
     ┌───────┼────────┬─────────────┬────────────┬──────────┐
     │       │        │             │            │          │
     ▼       ▼        ▼             ▼            ▼          ▼
┌─────────┐ ┌────┐ ┌──────┐  ┌──────────┐  ┌────────┐  ┌──────┐
│ Unified │ │News│ │Ollama│  │PDF Vector│  │ Stock  │  │Train │
│Terminal │ │API │ │Model │  │Database  │  │Fetcher │  │  DB  │
└─────────┘ └────┘ └──────┘  └──────────┘  └────────┘  └──────┘
     │        │       │           │             │          │
     ▼        ▼       ▼           ▼             ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│ Data Sources: yfinance, Alpha Vantage, ChromaDB, SQLite   │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

```bash
# Python 3.11+
# Ollama (for AI features)

# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull base model
ollama pull llama3:8b
```

### Dependencies

```bash
cd /home/user/finance
pip install -r requirements.txt
```

### Create Custom Model

```bash
# Create the OT-Analytics-Quant-Model
ollama create ot-analytics-quant -f Modelfile
```

### Environment Variables

Create `.env` file:

```bash
# Required for news
ALPHA_VANTAGE_KEY=your_alpha_vantage_api_key_here

# Optional for enhanced AI features
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## 🚀 Quick Start

### Interactive Mode

```bash
python ot_analytics_terminal.py
```

### Demo Mode

```bash
python ot_analytics_terminal.py --demo
```

## 📖 Command Reference

### 🚀 Auto-Loading Commands

| Command | Description |
|---------|-------------|
| `auto` | Generate automatic market briefing with AI analysis |
| `reload` | Reload all data (news, market snapshot, training data) |

### 📊 Stock Display

| Command | Description | Example |
|---------|-------------|---------|
| `show tech` | Display top 20 tech stocks | `show tech` |
| `show ai` | Display top 20 AI stocks | `show ai` |
| `show all` | Display top 20 stocks (all categories) | `show all` |
| `show custom <tickers>` | Display custom stock list | `show custom AAPL,MSFT,NVDA` |

### 🤖 AI Analysis (OT-Analytics-Quant-Model)

| Command | Description | Example |
|---------|-------------|---------|
| `ai <question>` | Query AI with full context | `ai analyze nvidia performance` |
| `ask <question>` | Alias for 'ai' | `ask what's driving tech stocks?` |
| `analyze` | Structured analysis of displayed stocks | `analyze` |

### 📰 News Integration

| Command | Description | Example |
|---------|-------------|---------|
| `news` | Show latest market news (all topics) | `news` |
| `news <tickers>` | Show news for specific tickers | `news NVDA,AMD,INTC` |

### 💾 RAG Memory (PDF Vector Database)

| Command | Description | Example |
|---------|-------------|---------|
| `rag <query>` | Search PDF vector database | `rag interest rate impact` |
| `pdf` | Access PDF ingestion system | `pdf` |

### 💿 Data Management

| Command | Description | Example |
|---------|-------------|---------|
| `history` | Show session command history | `history` |
| `export <filename>` | Export data to CSV | `export tech_analysis.csv` |

### 🔧 System Commands

| Command | Description |
|---------|-------------|
| `menu` | Access stock training database menu (options 1-71) |
| `clear` | Clear screen |
| `help` | Show help message |
| `quit` / `exit` | Exit terminal |

## 🎯 Typical Workflows

### Morning Market Briefing

```bash
OT-Analytics> auto
# System generates comprehensive briefing with:
# - Market snapshot
# - News highlights
# - Training database insights
# - AI-powered analysis
```

### Stock Research

```bash
OT-Analytics> show tech
OT-Analytics> news NVDA,AMD
OT-Analytics> ai analyze nvidia's competitive position in AI chips
OT-Analytics> export nvda_analysis.csv
```

### Document Q&A with RAG

```bash
OT-Analytics> pdf
# Ingest relevant PDFs (research reports, 10-Ks, etc.)

OT-Analytics> rag impact of rising rates on tech valuations
# Search vector database

OT-Analytics> ai how do rate changes affect high-growth tech stocks?
# AI uses RAG context + market data
```

### Deep Dive Analysis

```bash
OT-Analytics> show ai
OT-Analytics> news
OT-Analytics> ai perform detailed quant analysis on AI sector with recent news context
# Full analysis with stock data, news, RAG memory, and training DB
```

## 🔑 Key Features

### 1. Automatic Data Loading

On startup, the system automatically loads:

- ✅ Market snapshot (SPY, QQQ, major stocks)
- ✅ Latest news (20 articles with sentiment)
- ✅ Training database (historical analysis)
- ✅ RAG memory status (PDF documents)
- ✅ Ollama model availability

### 2. Institutional-Grade AI Model

**OT-Analytics-Quant-Model** features:

- **Temperature: 0.05** - Maximum precision
- **Expert in**: Technical analysis, fundamental analysis, options, risk management, algo trading
- **8-Phase Framework**: Systematic analysis from data collection to performance attribution
- **Structured Output**: Executive summary, quantitative analysis, risk factors, trading thesis

### 3. Comprehensive Context

AI queries automatically include:

- Current stock data (displayed in session)
- Recent news with sentiment scores
- RAG memory (relevant PDF excerpts)
- Training database statistics
- Market regime indicators

### 4. News Sentiment Analysis

Alpha Vantage News API provides:

- Overall sentiment (Bullish/Bearish/Neutral)
- Ticker-specific sentiment
- Relevance scores
- Real-time updates

### 5. RAG Memory Integration

PDF vector database enables:

- Semantic search across ingested documents
- Automatic context injection into AI queries
- Research report Q&A
- 10-K/10-Q analysis

## 📊 Example Output

### Automatic Briefing

```
📋 AUTOMATIC MARKET BRIEFING
====================================================================================================

📊 MARKET SNAPSHOT:
Average 1-day return: +0.75%
  • SPY: $450.25 (+0.52%)
  • QQQ: $375.80 (+1.23%)
  • NVDA: $875.50 (+2.10%)
  • AAPL: $185.30 (+0.45%)
  • MSFT: $425.60 (+0.80%)

📰 NEWS HIGHLIGHTS (Top 5):
1. [Bullish] Nvidia announces breakthrough in AI chip efficiency...
2. [Neutral] Fed maintains interest rates, signals data-dependent approach...
3. [Bullish] Tech sector earnings exceed expectations for Q4...
4. [Bearish] Geopolitical tensions weigh on global markets...
5. [Neutral] Energy sector rotation continues amid supply concerns...

📈 TRAINING DATABASE INSIGHTS:
Top 60-day performers:
  • NVDA: +45.23%
  • AMD: +32.15%
  • META: +28.90%

⚠️  Regime changes detected: 3 stocks
  • INTC: Mean-Reverting
  • CSCO: Trending
  • IBM: High-Volatility

💾 RAG MEMORY: 5 documents indexed

🤖 GENERATING AI ANALYSIS (OT-Analytics-Quant-Model)...
----------------------------------------------------------------------------------------------------

📊 EXECUTIVE SUMMARY
Current market regime exhibits bullish momentum with above-average volatility. Tech sector leadership
driven by AI narrative, supported by strong earnings. Recommend momentum strategies with tight risk
management given elevated valuations.

📈 MARKET CONTEXT
• SPY: $450.25 (+0.52% 1D)
• QQQ Outperformance: +0.71% vs SPY
• VIX: Elevated at 16.5 (30-day avg: 14.2)
• Sector Rotation: Technology +1.2%, Energy -0.8%

🔍 QUANTITATIVE ANALYSIS
[Technical Indicators]
• Trend: Bullish (QQQ > 20/50/200 SMA)
• Momentum: Overbought (RSI: 72)
• Volatility: Expanding (ATR increasing)
• Volume: Above average (+15%)

[Statistical Metrics]
• Correlation Regime: Risk-on (Tech/SPY correlation: 0.85)
• Volatility Forecast: Mean-reverting from highs
• Drawdown Risk: 5% from ATH

⚠️ RISK FACTORS
• Valuation: Tech P/E at 95th percentile vs 10Y history
• Sentiment: Extreme bullishness (contrarian indicator)
• Macro: Rate uncertainty remains elevated
• Technical: RSI overbought suggests near-term consolidation

💡 TRADING THESIS
• Direction: Selective Long with hedges
• Entry: Pullbacks to 20-day SMA
• Profit Target: +5-8% on mo momentum plays
• Stop Loss: 3% below entry (1.5-2.5 R:R ratio)
• Position Size: 2-3% per position (Kelly-optimal)
• Hedges: VIX calls, QQQ puts (10% of long exposure)

====================================================================================================
```

## 🔐 Security & Best Practices

### API Keys

- Store in `.env` file (never commit)
- Alpha Vantage: Free tier sufficient for testing
- Rate limits: Respect API quotas

### Data Privacy

- All data stored locally (SQLite, ChromaDB)
- No external transmission except API calls
- Ollama runs locally (no cloud dependency)

### Risk Management

- ⚠️ **PAST PERFORMANCE ≠ FUTURE RESULTS**
- All recommendations are educational only
- Never risk more than you can afford to lose
- Implement proper position sizing
- Use stop-losses

## 🛠️ Troubleshooting

### Ollama Not Available

```bash
# Check if Ollama is running
ollama list

# If not running
systemctl start ollama  # Linux
# or restart Ollama app (macOS/Windows)

# Recreate model if needed
ollama create ot-analytics-quant -f Modelfile
```

### News Not Loading

```bash
# Check API key
cat .env | grep ALPHA_VANTAGE_KEY

# Get free key
# https://www.alphavantage.co/support/#api-key

# Add to .env
echo "ALPHA_VANTAGE_KEY=YOUR_KEY" >> .env
```

### RAG Memory Issues

```bash
# Install dependencies
pip install chromadb pypdf langchain-text-splitters

# Check ChromaDB
python -c "import chromadb; print('ChromaDB OK')"
```

### Training Database Empty

```bash
# Run training
python stock_training_db.py --train

# This may take 5-10 minutes
# Creates historical analysis for all stocks
```

## 📈 Performance Metrics

### Startup Time

- Ollama check: <100ms
- Training DB load: <200ms
- News fetch: 1-2s (API dependent)
- Market snapshot: 2-3s (yfinance)
- **Total**: ~3-5 seconds

### AI Response Time

- Simple query: 2-5s
- Complex analysis: 10-30s
- Streaming mode: Real-time display

### Memory Usage

- Base: ~200MB
- With ChromaDB: ~500MB
- With Ollama: ~4-8GB (depends on model)

## 🔄 Comparison: Three Wrappers

### Wrapper 1: `stock_training_db.py`

- Training database with AI agent
- Menu options 1-70
- Historical analysis
- Regime detection

### Wrapper 2: `unified_terminal.py`

- Real-time stock data
- Multiple categories (Tech, AI, All)
- AI analysis
- Export capabilities
- Wrapper 1 features + live data

### Wrapper 3: `ot_analytics_terminal.py` ⭐ **THIS**

- **Everything from Wrappers 1 & 2**
- **+ Ollama integration (custom quant model)**
- **+ News with sentiment analysis**
- **+ RAG memory (PDF vector DB)**
- **+ Automatic data loading**
- **+ Comprehensive context for AI**
- **= Complete production system**

## 🎓 Learning Resources

### Quantitative Finance

- Model system prompt contains comprehensive quant finance curriculum
- Covers: Technical analysis, fundamental analysis, options, risk management
- Statistical methods: ARIMA, GARCH, Monte Carlo, Bayesian methods
- Trading strategies: Momentum, mean reversion, arbitrage

### Using the System

1. Start with `auto` to get bearings
2. Use `show` commands to display stocks
3. Read `news` for latest developments
4. Query `ai` with specific questions
5. Use `rag` to search documents
6. `export` for further analysis

## 🚀 Advanced Usage

### Custom Model Parameters

Edit `Modelfile` to adjust:

```modelfile
PARAMETER temperature 0.05    # Lower = more deterministic
PARAMETER top_p 0.85          # Nucleus sampling
PARAMETER repeat_penalty 1.15 # Reduce repetition
```

### Batch Analysis

```bash
# Create script
echo "show tech" > batch.txt
echo "export tech.csv" >> batch.txt
echo "show ai" >> batch.txt
echo "export ai.csv" >> batch.txt

# Run (requires modification for batch mode)
```

### Integration with Other Tools

```python
from ot_analytics_terminal import OTAnalyticsTerminal

terminal = OTAnalyticsTerminal()
terminal.startup_sequence()

# Use programmatically
terminal.show_news(tickers=["NVDA", "AMD"])
terminal.analyze_with_ollama("What's driving semiconductor stocks?")
```

## 📝 Contributing

Suggested improvements:

- [ ] WebSocket for real-time data streaming
- [ ] Portfolio tracking and P&L
- [ ] Backtesting integration
- [ ] Chart generation (ASCII/matplotlib)
- [ ] Alert system for price movements
- [ ] Multi-model support (Claude, GPT-4, etc.)
- [ ] Web UI (Flask/FastAPI)

## 📄 License

Same as parent project.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading and investing involve substantial risk of loss. The authors and contributors are not responsible for any trading losses incurred from using this software.

**USE AT YOUR OWN RISK.**

---

**Built with ❤️ for serious quantitative finance**

**OT Analytics Terminal - Where AI meets institutional-grade trading**

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║             OT ANALYTICS TERMINAL - QUICK REFERENCE            ║
╠════════════════════════════════════════════════════════════════╣
║ STARTUP                                                        ║
║  python ot_analytics_terminal.py          Start terminal       ║
║  auto                                      Auto briefing       ║
║                                                                ║
║ STOCK DATA                                                     ║
║  show tech | ai | all | custom TICKERS    Display stocks      ║
║  news [TICKERS]                            Get news            ║
║  analyze                                   Analyze data        ║
║                                                                ║
║ AI QUERIES                                                     ║
║  ai <question>                             Query with context  ║
║  rag <query>                               Search PDF DB       ║
║                                                                ║
║ SYSTEM                                                         ║
║  export <file>     Export data                                ║
║  history           Session history                            ║
║  menu              Training DB menu                           ║
║  pdf               PDF ingestion                              ║
║  help              Full help                                  ║
║  quit              Exit                                       ║
╚════════════════════════════════════════════════════════════════╝
```
