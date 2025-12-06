# Unified AI-Stock Market Terminal

A powerful command-line interface that combines real-time stock market data with AI-powered analysis. Display stock performance, get instant insights, and interact with an intelligent agent—all in one terminal.

## Features

### 📊 Real-Time Stock Data Display
- **Top 20 Tech Stocks**: AAPL, MSFT, GOOGL, AMZN, NVDA, and more
- **Top 20 AI Stocks**: NVDA, MSFT, PLTR, AI, and other AI-focused companies
- **Top 20 All Categories**: Diversified selection across tech, finance, healthcare, energy
- **Custom Stock Lists**: Display any stocks you want to track

### 🤖 AI-Powered Analysis
- Automatic market trend analysis
- Performance insights on displayed stocks
- Risk and volatility assessment
- Trading recommendations
- Natural language queries to the AI agent

### 💡 Intelligent Features
- Session history tracking
- Data export to CSV
- Integrated with stock training database
- Access to PDF ingestion system
- Color-coded performance indicators

## Installation

```bash
# Dependencies are already in requirements.txt
pip install -r requirements.txt
```

## Quick Start

### Interactive Mode

```bash
python unified_terminal.py
```

### Demo Mode

```bash
python unified_terminal.py --demo
```

## Command Reference

### Stock Display Commands

| Command | Description | Example |
|---------|-------------|---------|
| `show tech` | Display top 20 tech stocks | `show tech` |
| `show ai` | Display top 20 AI stocks | `show ai` |
| `show all` | Display top 20 stocks across all categories | `show all` |
| `show custom <tickers>` | Display custom stock list | `show custom AAPL,MSFT,GOOGL` |

### AI Analysis Commands

| Command | Description | Example |
|---------|-------------|---------|
| `analyze` | AI analysis of last displayed stocks | `analyze` |
| `ask <question>` | Ask AI agent a specific question | `ask what are the best tech stocks?` |
| `insights` | Get AI insights on current market data | `insights` |

### Data Commands

| Command | Description | Example |
|---------|-------------|---------|
| `history` | Show command history for this session | `history` |
| `export <filename>` | Export last displayed data to CSV | `export tech_stocks.csv` |

### System Commands

| Command | Description |
|---------|-------------|
| `menu` | Access original stock training menu |
| `pdf` | Access PDF ingestion system (option 71) |
| `clear` | Clear screen |
| `help` / `?` | Show help message |
| `quit` / `exit` | Exit terminal |

## Stock Categories

### Tech Stocks (20)
- **Mega Cap**: AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA
- **Semiconductors**: AVGO, INTC, AMD, QCOM, TXN
- **Software**: ORCL, ADBE, CRM, INTU, NOW
- **Other**: CSCO, IBM, SHOP

### AI Stocks (20)
- **AI Infrastructure**: NVDA, AMD (AI chips)
- **Cloud AI**: MSFT, GOOGL, AMZN, ORCL
- **AI Platforms**: PLTR, AI, SNOW, DDOG
- **AI Applications**: META, ADBE, CRM, NOW
- **Autonomous**: TSLA (Autopilot/FSD)
- **Emerging AI**: BBAI, SOUN, PATH, UiPath
- **AI Content**: BZFD
- **AI Fintech**: RELY

### All Categories (20)
- **Tech**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Finance**: JPM, BAC, WFC, V, MA
- **Healthcare**: JNJ, UNH, PFE
- **Consumer**: WMT, PG, KO
- **Energy**: XOM, CVX

## Usage Examples

### Example 1: View Tech Stocks and Get Analysis

```
Terminal> show tech
[Displays top 20 tech stocks with 5-day performance]

Terminal> analyze
[AI provides market trend analysis, identifies leaders/laggards, risk assessment]
```

### Example 2: Compare AI Stocks

```
Terminal> show ai
[Displays top 20 AI stocks]

Terminal> ask which AI stocks have the best momentum?
[AI agent analyzes and responds based on training data]
```

### Example 3: Custom Stock Tracking

```
Terminal> show custom AAPL,MSFT,GOOGL,AMZN,META
[Displays custom list of 5 mega-cap tech stocks]

Terminal> export faang_stocks.csv
[Exports data to CSV file]
```

### Example 4: Session Workflow

```
Terminal> show tech
Terminal> analyze
Terminal> show ai
Terminal> analyze
Terminal> history
[Shows all commands executed in session]
```

## Display Format

Stock data is displayed in a formatted table:

```
====================================================================================================
📊 Top 20 Tech Stocks (5-Day Performance)
====================================================================================================
#    Ticker   Name                           Price        Return %     Market Cap      P/E
----------------------------------------------------------------------------------------------------
1    NVDA     NVIDIA Corporation             $875.50      🟢 +8.23%    $2.16T         65.32
2    AMD      Advanced Micro Devices, Inc.   $142.30      🟢 +5.67%    $230.12B       48.91
3    MSFT     Microsoft Corporation          $425.80      🟢 +3.45%    $3.17T         36.25
...
====================================================================================================

📈 Summary:
   Average Return: +2.85%
   Positive: 15 | Negative: 5
   Top Performer: NVDA (+8.23%)
   Bottom Performer: INTC (-2.45%)
```

## AI Analysis Output

After displaying stocks, use `analyze` to get insights:

```
🤖 AI Agent Analysis:
====================================================================================================

📊 Market Analysis:

Market Trend: 🟢 BULLISH
Sentiment: Strong positive momentum across the board
Average Return: +2.85%

🏆 Top 3 Performers:
   1. NVDA (NVIDIA Corporation)
      Return: +8.23% | Market Cap: $2.16T
   2. AMD (Advanced Micro Devices, Inc.)
      Return: +5.67% | Market Cap: $230.12B
   3. MSFT (Microsoft Corporation)
      Return: +3.45% | Market Cap: $3.17T

⚠️  Bottom 3 Performers:
   20. INTC (Intel Corporation)
      Return: -2.45% | Market Cap: $180.45B
   ...

📊 Volatility: 3.42%
   🟡 Moderate volatility - normal market conditions

💡 Trading Insights:
   • 15 stocks showing strong momentum
     Consider: NVDA, AMD, MSFT
   • 5 stocks underperforming
     Watch for reversal: INTC, CSCO, IBM
   • Large-cap avg return: +3.12%

====================================================================================================
```

## Integration with Other Systems

### Stock Training Database
Access the original menu system for detailed stock training analysis:
```
Terminal> menu
```

### PDF Ingestion System
Access the PDF vector database for document analysis:
```
Terminal> pdf
```

## Data Sources

- **Stock Data**: Yahoo Finance (via yfinance)
- **Real-time Prices**: 15-minute delayed quotes
- **Historical Data**: Configurable periods (1d, 5d, 1mo, 3mo, 6mo, 1y)
- **Market Info**: Market cap, P/E ratios, volume, company names

## Performance

- **Data Fetching**: ~1-2 seconds per stock (parallel fetching recommended)
- **Display**: Instant rendering of tables
- **AI Analysis**: ~2-3 seconds for structured analysis
- **Export**: Instant CSV generation

## Error Handling

The terminal gracefully handles:
- Missing or invalid ticker symbols
- Network errors during data fetching
- Empty datasets
- Invalid commands
- Interrupted connections

## Tips for Best Results

1. **Start with a category**: Use `show tech`, `show ai`, or `show all`
2. **Analyze immediately**: Run `analyze` after displaying stocks
3. **Ask specific questions**: Use `ask` for targeted queries
4. **Track your session**: Use `history` to review commands
5. **Export important data**: Use `export` to save datasets
6. **Combine with training data**: Use `menu` for historical analysis

## Workflow Examples

### Daily Market Check
```bash
# Morning routine
Terminal> show tech
Terminal> analyze
Terminal> show ai
Terminal> analyze
Terminal> export morning_analysis.csv
```

### Research Workflow
```bash
# Deep dive on specific stocks
Terminal> show custom NVDA,AMD,INTC
Terminal> analyze
Terminal> ask what's driving nvidia's performance?
Terminal> menu  # Access historical training data
```

### Portfolio Monitoring
```bash
# Track your holdings
Terminal> show custom <your tickers>
Terminal> analyze
Terminal> export my_portfolio.csv
Terminal> history
```

## Troubleshooting

### No data displayed
- Check internet connection
- Verify ticker symbols are correct
- Some stocks may have limited data availability

### AI agent errors
- Ensure stock training database exists
- Run training: `python stock_training_db.py --train`
- Check ANTHROPIC_API_KEY in .env file

### Slow performance
- Reduce number of stocks in custom lists
- Check network connection
- Consider using cached data

## Architecture

```
┌─────────────────────────────────────────┐
│     Unified Terminal Interface          │
│  (Interactive CLI with command parser)  │
└───────────┬─────────────────────────────┘
            │
            ├──────────────┬──────────────────────┐
            │              │                      │
            ▼              ▼                      ▼
   ┌────────────────┐  ┌──────────────┐  ┌──────────────┐
   │ StockDataFetcher│ │ AIInsightAgent│ │StockTraining │
   │  (yfinance)    │  │ (AI Analysis)│  │  Database    │
   └────────────────┘  └──────────────┘  └──────────────┘
            │              │                      │
            ▼              ▼                      ▼
   ┌────────────────┐  ┌──────────────┐  ┌──────────────┐
   │ Yahoo Finance  │  │ Anthropic API│  │   SQLite     │
   │   Real-time    │  │   (Claude)   │  │Training Data │
   └────────────────┘  └──────────────┘  └──────────────┘
```

## Future Enhancements

- [ ] Real-time streaming quotes
- [ ] Technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Chart generation (ASCII or matplotlib)
- [ ] Alert system for price movements
- [ ] Portfolio tracking with P&L
- [ ] Historical comparison mode
- [ ] Sector rotation analysis
- [ ] News integration
- [ ] Multi-timeframe analysis
- [ ] Backtesting capabilities

## Contributing

To add new stock categories:
1. Define ticker list in `unified_terminal.py`
2. Add display method
3. Add command to `run()` method
4. Update help text

## License

Same as parent project.

## Support

For issues or questions:
1. Check this README
2. Review command help: `help`
3. Check stock_training_db documentation
4. Review source code comments

---

**Happy Trading! 📈🚀**
