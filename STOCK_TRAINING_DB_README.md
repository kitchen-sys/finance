# Stock Training Database with AI Agent

A comprehensive stock training system with persistent database storage and RAG-powered AI agent for intelligent insights on 20 top tech stocks using Alpha Vantage data.

## Features

### 🗄️ Database System
- **SQLite database** for persistent training run tracking
- Comprehensive schema tracking:
  - Training runs with metadata
  - Stock-level results with full metrics
  - Regime detection signals
  - Risk metrics (VaR, CVaR, Sharpe ratio)
  - Volatility forecasts
  - ARIMA price forecasts
  - Alpha Vantage technical and fundamental data

### 📊 Training System
- Train on 20 top tech stocks: AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, and more
- Integration with **Alpha Vantage API** for:
  - Technical indicators (RSI, MACD, ADX)
  - Fundamental data (P/E ratio, ROE, profit margins)
  - Technical scoring
- Optimal Transport (OT) based regime detection
- ARIMA price forecasting
- GARCH volatility forecasting
- Risk metrics calculation
- All results persisted to database with full history

### 🤖 AI Insight Agent
- **RAG-powered AI agent** for natural language queries
- Intelligent analysis of training data
- Query capabilities:
  - Top performers analysis
  - Regime change detection
  - Risk and volatility metrics
  - Sharpe ratio rankings
  - Trading recommendations
- Interactive chat interface
- Context-aware responses based on latest training data

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your ALPHA_VANTAGE_KEY
```

## Usage

### Run Training

Train the model on all 20 tech stocks and save to database:

```bash
python stock_training_db.py --train
```

Options:
- `--history DAYS`: Historical data period (default: 365 days)
- `--lookback DAYS`: Regime detection lookback period (default: 60 days)
- `--db-path PATH`: Database file path (default: ./stock_training.db)

Example:
```bash
python stock_training_db.py --train --history 730 --lookback 90
```

### Chat with AI Agent

Launch the interactive AI agent to analyze training results:

```bash
python stock_training_db.py --chat
```

Example queries:
- "What are the top performers?"
- "Which stocks show regime changes?"
- "Analyze risk metrics"
- "Give me trading recommendations"
- "What's the general market summary?"

### Combined Workflow

Train and then immediately chat with the agent:

```bash
python stock_training_db.py --train --chat
```

## Database Schema

### training_runs
- `run_id`: Unique identifier for each training session
- `timestamp`: When the training run started
- `universe_name`: Name of stock universe (e.g., "tech_20")
- `lookback_regime`: Regime detection lookback period
- `history_days`: Historical data period used
- `status`: Run status (pending, running, completed, failed)
- `num_stocks_trained`: Number of successfully trained stocks
- `num_stocks_failed`: Number of failed stocks
- `alpha_vantage_enabled`: Whether Alpha Vantage was used

### stock_training_results
Complete results for each stock in a training run:

**Identification:**
- `result_id`: Unique identifier
- `run_id`: Parent training run
- `ticker`: Stock symbol
- `training_timestamp`: When this stock was trained
- `data_points`: Number of historical data points used
- `date_start`, `date_end`: Date range of training data

**Regime Metrics:**
- `regime_change`: Boolean - regime shift detected
- `regime_distance`: Wasserstein distance metric
- `regime_state`: Current regime state
- `regime_threshold`: Detection threshold

**Risk Metrics:**
- `var_95`: Value at Risk (95% confidence)
- `cvar_95`: Conditional VaR
- `sharpe_ratio`: Risk-adjusted return metric

**Volatility Metrics:**
- `current_vol`: Current volatility forecast
- `mean_vol`: Average volatility
- `max_vol`: Maximum volatility

**Forecast Metrics:**
- `forecast_5d`: 5-day price forecast
- `forecast_stderr`: Standard error
- `forecast_conf_lower`, `forecast_conf_upper`: Confidence interval

**Current Metrics:**
- `price`: Latest price
- `return_1d`, `return_5d`, `return_20d`, `return_60d`: Returns over various periods

**Alpha Vantage Metrics:**
- `av_rsi`: Relative Strength Index
- `av_technical_score`: Aggregate technical score
- `av_pe_ratio`: Price-to-Earnings ratio
- `av_roe`: Return on Equity
- `av_profit_margin`: Profit margin

**Errors:**
- `error_message`: Error details if training failed

## Architecture

### StockTrainingDatabase
Core database class handling all persistence:
- Table creation and schema management
- CRUD operations for training runs
- Query methods for analysis
- Indexing for performance

### StockTrainer
Enhanced trainer with database integration:
- Downloads stock data via yfinance
- Trains OTFinanceModel on each stock
- Fetches Alpha Vantage data (if API key available)
- Persists all results to database
- Prints training summaries

### AIInsightAgent
RAG-powered AI agent for insights:
- Builds context from database
- Answers natural language queries
- Provides actionable insights
- Interactive chat interface
- Rule-based analysis (extensible to Anthropic Claude API)

## Example Workflow

```python
from stock_training_db import (
    StockTrainingDatabase,
    StockTrainer,
    AIInsightAgent,
    TECH_STOCKS_20
)
from finance_bot.alpha_vantage_provider import AlphaVantageProvider

# Initialize database
db = StockTrainingDatabase("./my_training.db")

# Initialize Alpha Vantage (optional)
av_provider = AlphaVantageProvider(api_key="YOUR_KEY")

# Create trainer
trainer = StockTrainer(
    db=db,
    av_provider=av_provider,
    lookback_regime=60,
    history_days=365
)

# Run training
run_id = trainer.train_all_stocks(
    tickers=TECH_STOCKS_20,
    universe_name="tech_20"
)

# Query results
latest_run = db.get_latest_run()
top_performers = db.get_top_performers(metric="sharpe_ratio", limit=10)
regime_changes = db.get_regime_changes()

# Use AI agent
agent = AIInsightAgent(db=db)
response = agent.query("What are the best trading opportunities?")
print(response)

# Interactive chat
agent.chat()

db.close()
```

## AI Agent Capabilities

The AI agent provides intelligent analysis across multiple dimensions:

### Performance Analysis
- Top/bottom performers by return
- Risk-adjusted performance (Sharpe ratios)
- Momentum analysis across timeframes

### Regime Detection
- Stocks experiencing regime shifts
- Strength of regime changes
- Implications for trading

### Risk Analysis
- Volatility rankings
- VaR and CVaR metrics
- Risk-adjusted opportunity identification

### Trading Recommendations
- Long candidates (stable, good fundamentals)
- Watch list (regime shifts, potential reversals)
- Risk management suggestions

### Market Summary
- Overall market conditions
- Key statistics and trends
- Comparative analysis

## Technical Details

### Optimal Transport Regime Detection
Uses Wasserstein distance to detect distribution shifts in returns:
- Compares recent vs historical return distributions
- Detects regime changes before traditional indicators
- Provides distance metrics for shift strength

### Risk Metrics
- **VaR/CVaR**: Tail risk quantification
- **Sharpe Ratio**: Risk-adjusted returns
- **GARCH Volatility**: Dynamic volatility forecasting

### Forecasting
- **ARIMA**: Price forecasting with confidence intervals
- **Auto-ARIMA**: Automatic order selection
- **Multi-horizon**: 1-day, 5-day, 20-day, 60-day returns

### Alpha Vantage Integration
- Technical indicators: RSI, MACD, ADX, Bollinger Bands
- Fundamental data: Financial ratios, margins
- Aggregate scoring for signal generation

## Database Queries

Common SQL queries you can run:

```python
# Get all regime changes from latest run
db.execute_query("""
    SELECT ticker, regime_state, regime_distance, return_60d
    FROM stock_training_results
    WHERE regime_change = 1
    ORDER BY regime_distance DESC
""")

# Compare performance across runs
db.execute_query("""
    SELECT r.ticker, r.return_60d, r.sharpe_ratio, t.timestamp
    FROM stock_training_results r
    JOIN training_runs t ON r.run_id = t.run_id
    WHERE r.ticker = 'AAPL'
    ORDER BY t.timestamp DESC
""")

# Top stocks by multiple criteria
db.execute_query("""
    SELECT ticker, return_60d, sharpe_ratio, current_vol
    FROM stock_training_results
    WHERE run_id = (SELECT MAX(run_id) FROM training_runs)
    AND sharpe_ratio > 0.5
    AND return_60d > 0
    ORDER BY sharpe_ratio DESC
    LIMIT 10
""")
```

## Extending the AI Agent

The current AI agent uses rule-based responses. To integrate with Anthropic Claude API:

1. Set `ANTHROPIC_API_KEY` in your environment
2. Install: `pip install anthropic`
3. Modify `AIInsightAgent._generate_response()` to call the API:

```python
def _generate_response(self, context: str, query: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=self.api_key)

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system="You are an expert financial analyst...",
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return message.content[0].text
```

## Future Enhancements

- [ ] Real-time data streaming
- [ ] Portfolio optimization using training results
- [ ] Backtesting framework
- [ ] Advanced RAG with vector embeddings
- [ ] Multi-model ensemble predictions
- [ ] Alerts and notifications
- [ ] Web dashboard
- [ ] Integration with broker APIs

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

For questions or issues, please open a GitHub issue.
