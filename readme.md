# OT Analytics Finance System

**Institutional-Grade Quantitative Finance Platform with AI, RAG Memory, and Real-Time Analysis**

This project provides a comprehensive quantitative finance platform combining optimal-transport inspired regime detection, real-time market data, AI-powered analysis, RAG memory, and automated trading insights. The system features three progressive interfaces from basic Discord scanning to a full-featured terminal with institutional-grade AI.

## 🌟 What's New

This repository now includes a complete evolution from Discord bot to production trading terminal:

1. **Discord Bot** - Original scanner with OT regime detection
2. **Stock Training Database** (Wrapper 1) - Historical analysis with 71 menu options
3. **Unified Terminal** (Wrapper 2) - Real-time data + AI analysis
4. **OT Analytics Terminal** (Wrapper 3) ⭐ **PRODUCTION READY** - Complete system with Ollama AI, news, and RAG memory

## 🚀 Quick Start

### New Users - Start Here

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Add your ALPHA_VANTAGE_KEY

# 3. Run the production terminal
python ot_analytics_terminal.py

# 4. (Optional) Install Ollama for AI
curl https://ollama.ai/install.sh | sh
ollama pull llama3:8b
ollama create ot-analytics-quant -f Modelfile
```

**📖 For detailed setup instructions, see [SETUP.md](SETUP.md)**

## 📦 System Components

### 1. OT Analytics Terminal (⭐ **RECOMMENDED**)

**File:** `ot_analytics_terminal.py`

The complete production system with automatic data loading, news integration, and institutional-grade AI.

```bash
python ot_analytics_terminal.py
```

**Features:**
- ✅ Automatic market briefing on startup
- ✅ Real-time stock data (Tech, AI, All categories)
- ✅ News sentiment analysis (Alpha Vantage)
- ✅ Ollama-powered AI (OT-Analytics-Quant-Model)
- ✅ RAG memory (PDF vector database)
- ✅ Training database integration
- ✅ Export to CSV
- ✅ Comprehensive context for AI queries

**Commands:**
- `auto` - Automatic market briefing with AI
- `show tech/ai/all` - Display stock categories
- `news [TICKERS]` - Latest news with sentiment
- `ai <question>` - Query AI with full context
- `rag <query>` - Search PDF documents

**Documentation:** [OT_ANALYTICS_README.md](OT_ANALYTICS_README.md)

### 2. Unified Terminal

**File:** `unified_terminal.py`

Real-time stock data with AI-powered analysis.

```bash
python unified_terminal.py
```

**Features:**
- Real-time data for 20 tech stocks, 20 AI stocks, top 20 all categories
- Custom stock lists
- AI analysis
- Export capabilities
- Session tracking

**Documentation:** [UNIFIED_TERMINAL_README.md](UNIFIED_TERMINAL_README.md)

### 3. Stock Training Database

**File:** `stock_training_db.py`

Historical stock analysis with regime detection and 71 menu options.

```bash
# Initial training (5-10 minutes)
python stock_training_db.py --train

# Interactive menu
python stock_training_db.py --chat
```

**Features:**
- Options 1-70: Quick queries and AI analysis
- Option 71: PDF ingestion with vector database
- Regime change detection
- Risk metrics (VaR, CVaR, Sharpe)
- ARIMA forecasting

**Documentation:** [STOCK_TRAINING_DB_README.md](STOCK_TRAINING_DB_README.md)

### 4. Discord Bot (Original)

**File:** `main.py`

Live market scanner posting to Discord channels.

```bash
python main.py --channels YOUR_CHANNEL_ID
```

**Features:**
- Automated scanning every 30 minutes
- Top 5 "buy" and "boom" candidates
- S&P 500, NASDAQ 100, Dow, Most Active
- Slash commands: `/scan`

**Setup:**
```bash
# Set environment variables
DISCORD_TOKEN=your_bot_token
DISCORD_CHANNELS=channel_id_1,channel_id_2
```

## 🎯 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 OT Analytics Terminal                       │
│  (Production system with auto-loading, news, AI, RAG)      │
└────────────┬────────────────────────────────────────────────┘
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
│   Data: yfinance, Alpha Vantage, ChromaDB, SQLite          │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Key Technologies

### Core Financial Models
- **Optimal Transport**: Wasserstein distance-based regime detection
- **GARCH/EGARCH**: Volatility forecasting
- **ARIMA**: Time series prediction
- **Hidden Markov Models**: Regime classification
- **Mean-Variance Optimization**: Portfolio construction
- **Risk Parity**: Equal risk contribution

### AI & Machine Learning
- **Ollama**: Local LLM inference (llama3:8b)
- **Custom Model**: OT-Analytics-Quant-Model (temp=0.05 for precision)
- **ChromaDB**: Vector database for RAG
- **Scikit-learn**: Cross-sectional analysis
- **Factor Models**: Fama-French, momentum, value

### Data Sources
- **yfinance**: Real-time stock data, historical prices
- **Alpha Vantage**: News sentiment, technical indicators, fundamentals
- **PDF Documents**: Research reports, 10-Ks (via RAG)

## 💾 Databases

The system maintains several databases:

| Database | Type | Purpose |
|----------|------|---------|
| `stock_training.db` | SQLite | Historical training results, regime changes |
| `pdf_metadata.db` | SQLite | PDF document metadata, chunks |
| `chroma_db/` | ChromaDB | Vector embeddings for semantic search |

## 🤖 OT-Analytics-Quant-Model

Custom Ollama model optimized for quantitative finance with institutional-grade precision.

### Parameters

```modelfile
temperature: 0.05          # Maximum precision
top_p: 0.85
repeat_penalty: 1.15
presence_penalty: 0.0
frequency_penalty: 0.0
num_ctx: 8192             # Extended context
num_predict: 4096         # Longer outputs
```

### Expertise Areas

1. **Quantitative Analysis & Statistical Methods**
   - Time Series Analysis: ARIMA, GARCH, VAR models, cointegration
   - Stochastic Calculus: Brownian motion, Ito's lemma
   - Statistical Arbitrage: Pairs trading, mean reversion
   - Bayesian Methods: Probabilistic forecasting
   - Monte Carlo Simulation: VaR estimation, option pricing

2. **Risk Management & Portfolio Theory**
   - Modern Portfolio Theory: Efficient frontier, CAPM
   - Risk Metrics: VaR, CVaR, Expected Shortfall
   - Risk-Adjusted Returns: Sharpe, Sortino, Calmar ratios
   - Kelly Criterion: Optimal position sizing
   - Scenario Analysis: Stress testing, extreme value theory

3. **Technical Analysis & Market Microstructure**
   - Trend Indicators: Moving averages, ADX, Parabolic SAR
   - Momentum Oscillators: RSI, Stochastic, Williams %R
   - Volatility Indicators: Bollinger Bands, ATR, Keltner
   - Volume Analysis: OBV, Accumulation/Distribution
   - Order Flow: Bid-ask spread, market depth, VWAP

4. **Fundamental Analysis & Valuation**
   - Valuation Models: DCF, DDM, comparable multiples
   - Profitability Metrics: ROE, ROA, profit margins
   - Quality Metrics: Piotroski F-Score, Altman Z-Score
   - Sector-Specific Analysis: REIT, bank, tech SaaS metrics

5. **Algorithmic Trading Strategies**
   - Momentum: Cross-sectional, time-series, trend following
   - Mean Reversion: Bollinger bounce, RSI extremes
   - Breakout Systems: Donchian channels, volatility expansion
   - Event-Driven: Earnings surprises, M&A arbitrage
   - Machine Learning: Random forests, neural networks, LSTM

6-9. **Plus**: Derivatives, Options, Market Regimes, Execution Analysis

### 8-Phase Analytical Framework

1. Data Collection & Preprocessing
2. Descriptive Statistics
3. Regime Identification
4. Signal Generation
5. Risk Assessment
6. Position Sizing & Portfolio Construction
7. Execution Planning
8. Performance Attribution

### Create the Model

```bash
ollama pull llama3:8b
ollama create ot-analytics-quant -f Modelfile
```

## 📊 Features Deep Dive

### Optimal Transport Regime Detection

The core innovation uses Wasserstein distance to detect market regime changes:

```python
from finance_bot.model import OTFinanceModel

model = OTFinanceModel()
regime_change, distance = model.detect_regime_change(returns)
```

**Key Metrics:**
- Regime states: Trending, Mean-Reverting, High-Volatility
- Distance threshold for change detection
- Rolling window analysis

### News Sentiment Analysis

Alpha Vantage integration provides real-time sentiment:

```python
from ot_analytics_terminal import NewsProvider

news_provider = NewsProvider()
news = news_provider.get_news_sentiment(tickers=["NVDA", "AMD"])
```

**Sentiment Scores:**
- Overall: Bullish/Bearish/Neutral
- Ticker-specific relevance
- Real-time updates

### RAG Memory System

PDF vector database for document Q&A:

```python
# Ingest research report
pdf_store.ingest_pdf("research_report.pdf")

# Search for relevant context
results = pdf_store.search("AI chip market analysis")
```

**Features:**
- Semantic search across documents
- Automatic chunking (1000 chars, 200 overlap)
- ChromaDB vector embeddings
- SQL metadata storage

### Risk Metrics

Comprehensive risk analysis:

- **VaR (Value at Risk)**: 95th percentile loss
- **CVaR (Conditional VaR)**: Expected shortfall
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Beta**: Market correlation

## 🔄 Typical Workflows

### Morning Market Briefing

```bash
python ot_analytics_terminal.py

OT-Analytics> auto
# System generates:
# - Market snapshot (SPY, QQQ, major stocks)
# - News highlights (top 5 with sentiment)
# - Training database insights
# - AI-powered comprehensive analysis
```

### Stock Research

```bash
OT-Analytics> show tech
# Display 20 tech stocks

OT-Analytics> news NVDA,AMD
# Get latest semiconductor news

OT-Analytics> ai analyze nvidia's position in AI chips with recent news context
# AI provides comprehensive analysis using:
# - Stock performance data
# - News sentiment
# - RAG memory (if docs ingested)
# - Training database history

OT-Analytics> export semiconductor_analysis.csv
```

### Document Analysis with RAG

```bash
OT-Analytics> pdf
# Ingest research reports

OT-Analytics> rag impact of interest rates on tech valuations
# Search vector database

OT-Analytics> ai how do rising rates affect high-growth tech stocks?
# AI uses RAG context + market data
```

## 📈 Stock Categories

### Tech Stocks (20)
AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, AVGO, ORCL, ADBE, CRM, CSCO, INTC, AMD, QCOM, TXN, INTU, IBM, NOW, SHOP

### AI Stocks (20)
NVDA, MSFT, GOOGL, META, AMZN, TSLA, AMD, ORCL, CRM, ADBE, NOW, PLTR, SNOW, DDOG, PATH, AI, BBAI, SOUN, BZFD, RELY

### All Categories (20)
Tech + Finance (JPM, BAC, WFC, V, MA) + Healthcare (JNJ, UNH, PFE) + Consumer (WMT, PG, KO) + Energy (XOM, CVX)

## 🛠️ Configuration

### Environment Variables

Create `.env` file:

```bash
# Required for news and enhanced data
ALPHA_VANTAGE_KEY=your_key_here

# Optional for AI agent features
ANTHROPIC_API_KEY=your_key_here

# Optional for Discord bot
DISCORD_TOKEN=your_token_here
DISCORD_CHANNELS=channel_id_1,channel_id_2
```

### Custom Stock Lists

Edit `unified_terminal.py` or `ot_analytics_terminal.py`:

```python
CUSTOM_STOCKS = [
    "YOUR", "CUSTOM", "TICKERS"
]
```

### Model Parameters

Edit `Modelfile` to adjust AI behavior:

```modelfile
PARAMETER temperature 0.05    # Lower = more deterministic
PARAMETER top_p 0.85          # Nucleus sampling threshold
```

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Complete installation guide
- **[OT_ANALYTICS_README.md](OT_ANALYTICS_README.md)** - OT Analytics Terminal
- **[UNIFIED_TERMINAL_README.md](UNIFIED_TERMINAL_README.md)** - Unified Terminal
- **[STOCK_TRAINING_DB_README.md](STOCK_TRAINING_DB_README.md)** - Training Database
- **[PDF_VECTOR_DB_SCHEMA.md](PDF_VECTOR_DB_SCHEMA.md)** - RAG system schema
- **[Modelfile](Modelfile)** - AI model configuration

## 🧪 Testing

```bash
# Test PDF system
python test_pdf_simple.py

# Test unified terminal (with mock data)
python test_unified_terminal.py

# Run full training
python stock_training_db.py --train

# Demo mode
python ot_analytics_terminal.py --demo
```

## 🔒 Security & Disclaimers

### Security
- Never commit `.env` file
- Rotate API keys regularly
- Use environment variables in production
- Backup databases regularly

### Trading Disclaimers

⚠️ **CRITICAL WARNINGS**

- **PAST PERFORMANCE ≠ FUTURE RESULTS**
- All trading involves substantial risk of loss
- This software is for educational purposes only
- Not financial advice
- Authors not responsible for trading losses
- **NEVER RISK MORE THAN YOU CAN AFFORD TO LOSE**

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Real-time streaming data
- [ ] Additional technical indicators
- [ ] Chart generation (ASCII/matplotlib)
- [ ] Portfolio tracking with P&L
- [ ] Backtesting framework
- [ ] Multi-model support (Claude, GPT-4)
- [ ] Web UI (Flask/FastAPI)
- [ ] Mobile app integration

## 📊 Performance

### Startup Time
- OT Analytics Terminal: 3-5 seconds
- Includes: Ollama check, DB load, news fetch, market snapshot

### AI Response Time
- Simple query: 2-5 seconds
- Complex analysis: 10-30 seconds
- Streaming mode available for better UX

### Memory Usage
- Base system: ~200MB
- With ChromaDB: ~500MB
- With Ollama: ~4-8GB (llama3:8b)

## 🗺️ Roadmap

**Q1 2025:**
- ✅ PDF vector database (RAG memory)
- ✅ Ollama integration
- ✅ News sentiment analysis
- ✅ Auto-loading data system

**Q2 2025:**
- [ ] Real-time WebSocket data streaming
- [ ] Advanced portfolio optimization
- [ ] Backtesting framework
- [ ] Multi-model AI support

**Q3 2025:**
- [ ] Web UI with Flask/FastAPI
- [ ] Mobile app (React Native)
- [ ] Cloud deployment options
- [ ] API for external integrations

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: See docs/ folder
- **Questions**: Check SETUP.md troubleshooting section

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with:
- **Ollama** - Local LLM inference
- **ChromaDB** - Vector database
- **Alpha Vantage** - Financial data API
- **yfinance** - Yahoo Finance data
- **Discord.py** - Discord integration

---

**OT Analytics Finance System - Where AI meets institutional-grade trading**

*For detailed setup instructions, see [SETUP.md](SETUP.md)*

*For the complete production system, use `ot_analytics_terminal.py`*
