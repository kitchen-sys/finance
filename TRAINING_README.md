# Stock Trainer CLI

Terminal wrapper for training the OT Finance Model on top tech and AI stocks.

## Features

- **Top 20 Tech Stocks**: Train on major technology companies (AAPL, MSFT, GOOGL, NVDA, etc.)
- **Top 20 AI Stocks**: Train on AI-focused companies and heavy AI investors (NVDA, MSFT, PLTR, AI, etc.)
- **Comprehensive Analysis**:
  - Regime detection using optimal transport
  - ARIMA forecasting (5-day ahead)
  - Volatility forecasting (GARCH models)
  - Risk metrics (VaR, CVaR, Sharpe Ratio)
  - Stationarity testing
  - Momentum and performance metrics
- **Results Export**: Save detailed JSON results for further analysis

## Quick Start

### Basic Usage

Train on both tech and AI stocks:
```bash
python train.py
```

Train on tech stocks only:
```bash
python train.py --universe tech
```

Train on AI stocks only:
```bash
python train.py --universe ai
```

### Advanced Options

Use 2 years of historical data:
```bash
python train.py --history 730
```

Customize regime detection lookback:
```bash
python train.py --lookback-regime 90
```

Specify custom output directory:
```bash
python train.py --output-dir ./my_results
```

Use Alpha Vantage for enhanced analysis:
```bash
python train.py --av-key YOUR_ALPHA_VANTAGE_KEY
```

## Stock Lists

### Top 20 Tech Stocks
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Alphabet)
- AMZN (Amazon)
- META (Meta)
- TSLA (Tesla)
- NVDA (NVIDIA)
- AVGO (Broadcom)
- ORCL (Oracle)
- ADBE (Adobe)
- CRM (Salesforce)
- CSCO (Cisco)
- INTC (Intel)
- AMD (AMD)
- QCOM (Qualcomm)
- TXN (Texas Instruments)
- INTU (Intuit)
- IBM (IBM)
- NOW (ServiceNow)
- SHOP (Shopify)

### Top 20 AI Stocks
- NVDA (NVIDIA - AI chips)
- MSFT (Microsoft - OpenAI, Azure AI)
- GOOGL (Alphabet - DeepMind, Bard)
- META (Meta - LLaMA, AI Research)
- AMZN (Amazon - AWS AI, Alexa)
- TSLA (Tesla - Autopilot, Dojo)
- AMD (AMD - AI chips)
- ORCL (Oracle - AI infrastructure)
- AVGO (Broadcom - AI networking)
- QCOM (Qualcomm - AI mobile chips)
- ADBE (Adobe - Firefly AI)
- CRM (Salesforce - Einstein AI)
- PLTR (Palantir - AI platforms)
- SNOW (Snowflake - AI data)
- NOW (ServiceNow - AI workflows)
- PATH (UiPath - AI automation)
- AI (C3.ai - Enterprise AI)
- BBAI (BigBear.ai - AI solutions)
- SOUN (SoundHound - Voice AI)
- AMBA (Ambarella - AI vision chips)

## Output

The trainer generates:

1. **Console Summary**:
   - Download progress
   - Training progress
   - Top performers (60-day returns)
   - Bottom performers
   - Highest Sharpe ratios
   - Most volatile stocks
   - Regime change alerts

2. **JSON Results**:
   Detailed JSON files saved to `./training_results/` (or custom output directory) containing:
   - Regime detection results
   - ARIMA forecasts
   - Volatility metrics
   - Risk metrics (VaR, CVaR, Sharpe)
   - Stationarity tests
   - Current price and momentum metrics

## Example Output

```
============================================================
🚀 Stock Trainer - OT Finance Model
============================================================
History period: 365 days
Regime lookback: 60 days
Output directory: ./training_results

============================================================
Training on tech Universe
============================================================

📊 Downloading data for 20 stocks...
  [1/20] Fetching AAPL... ✓
  [2/20] Fetching MSFT... ✓
  ...

✓ Successfully downloaded 20/20 stocks

🧠 Training model on each stock...
  [1/20] Training AAPL... ✓
  [2/20] Training MSFT... ✓
  ...

💾 Results saved to: ./training_results/tech_20251206_010000.json

============================================================
tech Training Summary
============================================================

✓ Successfully trained: 20/20 stocks

⚠️  Regime changes detected: 3 stocks
    • NVDA: shift
    • TSLA: shift
    • AMD: shift

📈 Top 5 performers (60-day return):
    • NVDA: +45.23%
    • META: +32.10%
    • AVGO: +28.50%
    • AMD: +25.80%
    • MSFT: +18.90%

📉 Bottom 5 performers (60-day return):
    • INTC: -12.30%
    • CSCO: -5.20%
    • IBM: -3.10%
    • ORCL: +2.40%
    • TXN: +5.60%

⭐ Top 5 by Sharpe Ratio:
    • MSFT: 1.850
    • AAPL: 1.720
    • GOOGL: 1.650
    • META: 1.580
    • CRM: 1.450

💥 Most volatile (current):
    • TSLA: 3.45%
    • NVDA: 2.89%
    • AMD: 2.67%
    • META: 2.34%
    • SHOP: 2.12%
```

## Command Line Options

```
usage: train.py [-h] [--universe {tech,ai,both}] [--history HISTORY]
                [--lookback-regime LOOKBACK_REGIME] [--output-dir OUTPUT_DIR]
                [--av-key AV_KEY]

options:
  --universe {tech,ai,both}
                        Stock universe to train on (default: both)
  --history HISTORY
                        Historical data period in days (default: 365)
  --lookback-regime LOOKBACK_REGIME
                        Lookback period for regime detection (default: 60)
  --output-dir OUTPUT_DIR
                        Output directory for results (default: ./training_results)
  --av-key AV_KEY
                        Alpha Vantage API key (overrides ALPHA_VANTAGE_KEY env)
```

## Environment Variables

You can also set the Alpha Vantage API key via environment variable:

```bash
export ALPHA_VANTAGE_KEY=your_api_key_here
python train.py
```

Or use a `.env` file:
```
ALPHA_VANTAGE_KEY=your_api_key_here
```

## Requirements

All required packages are listed in `requirements.txt`. The trainer uses:
- yfinance for stock data
- numpy, pandas for data manipulation
- statsmodels for ARIMA modeling
- arch for GARCH/volatility modeling
- scipy for statistical tests and optimal transport
- cvxpy for portfolio optimization

## Tips

- **First run**: The first time you run the trainer, yfinance will download historical data. This may take a few minutes.
- **Data quality**: Some stocks may have insufficient data or download failures. These will be skipped automatically.
- **Regime changes**: Stocks showing regime changes may indicate significant market shifts and warrant closer attention.
- **Alpha Vantage**: For enhanced analysis with additional technical indicators, provide an Alpha Vantage API key (free at https://www.alphavantage.co/).
