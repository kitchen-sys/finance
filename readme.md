# OT-Powered Finance Scanner & Discord Bot

This project provides a lightweight implementation of an optimal-transport inspired regime detection engine, portfolio utilities, and a live Discord bot that scans equities in near real time using **yfinance** and **Alpha Vantage**. The scanner ranks the top 50 tickers per universe and surfaces the top 5 "buy" and "boom" candidates per run.

## Features
- **Optimal Transport edge**: Wasserstein-based regime detection, OT distance matrices, distributionally robust portfolio weights, and OT-influenced rebalancing utilities.
- **Time series toolkit**: Auto-ARIMA fitting and forecasting, Kalman filtering for latent regimes, and GARCH/EGARCH/GJR-GARCH/FIGARCH volatility forecasting to support VaR/CVaR-style risk metrics.
- **Portfolio construction**: Mean-variance, risk-parity, Wasserstein-robust, and regime-adaptive allocation helpers plus a simple event-driven backtester.
- **Discord live scanner**: Pulls the latest data for S&P 500, NASDAQ 100, Dow, and most-active lists; outputs the top 5 buy and boom setups for each category.
- **Alpha / signal layer**: Cross-sectional models (Fama–French style factors, factor regressions, idiosyncratic return isolation), ML forecasters (tree-based, regularized linear, sequence models), and causal/event estimators (causal forests, doubly robust learners, OT-based treated vs control comparison) to drive "own this, short that, right now" decisions.
- **Alpha Vantage integration**: Enhanced technical indicators (RSI, MACD, Bollinger Bands, ADX, Stochastic), fundamental data (ROE, ROA, profit margins, P/E ratios), and macro economic regime detection for superior signal quality.

## Quickstart
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set environment variables** (or pass CLI args):
   - `DISCORD_TOKEN`: your bot token
   - `DISCORD_CHANNELS`: comma-separated channel IDs for periodic scans
   - `ALPHA_VANTAGE_KEY`: your Alpha Vantage API key (get free key at https://www.alphavantage.co/support/#api-key)
   - `SCAN_CATEGORIES`: optional list of universes (sp500,nasdaq100,dow,most_active)

   Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   # Edit .env with your actual keys
   ```

3. **Run the bot**
   ```bash
   python main.py --channels 123456789012345678 234567890123456789
   ```
   Use `--history`, `--lookback-regime`, `--lookback-buy`, and `--lookback-boom` to tune the model.

   Optional Alpha Vantage flags:
   - `--av-key YOUR_KEY`: Pass Alpha Vantage key via CLI
   - `--disable-av-technicals`: Disable technical indicator integration
   - `--disable-av-fundamentals`: Disable fundamental data integration

## Discord commands
- Slash command `/scan` or legacy command `!scan` triggers an on-demand scan.
- The bot also posts scans to configured channels every 30 minutes.

## OT model overview
- **Regime detection**: Wasserstein distance between rolling windows of log returns.
- **Volatility forecasting**: GARCH/EGARCH/GJR-GARCH powered forecasts.
- **Portfolio construction**: Mean-variance, Wasserstein-stability tilt, and DRO-style penalty adjustments.
- **Risk metrics**: Rolling Sharpe, CVaR estimates, and event-driven backtesting with basic transaction costs.

## Alpha / signal layer
The signal stack complements the OT/optimization pipeline with cross-sectional predictors so the system can select what to own or short right now.

- **Cross-sectional factor backbone**
  - Fama–French style factors: size, value, quality, momentum, low volatility, and more.
  - Per-date cross-sectional regressions of returns on factors to estimate loadings and isolate residual (idiosyncratic) returns to trade while managing factor risk.
- **Machine learning forecasters**
  - Tree models (XGBoost / LightGBM) for next-period return, direction, or rank.
  - Linear / regularized models (Lasso, Elastic Net) for high-dimensional feature sets.
  - Sequence models (basic RNN, LSTM, temporal CNN, transformer) for multi-horizon forecasts.
- **Causal & event models**
  - Treatment-effect estimators for events such as earnings, Fed meetings, and macro releases.
  - Causal forests and doubly robust learners.
  - Optimal transport to compare treated vs control distributions around events.

### Cross-sectional usage example
```python
import pandas as pd
from finance_bot import CrossSectionalSignalLayer, CrossSectionalMLForecaster

# prices and volumes are DataFrames indexed by date with ticker columns
signal_layer = CrossSectionalSignalLayer()
exposures = signal_layer.compute_factor_exposures(prices, volumes)
returns = prices.pct_change().dropna()

# Cross-sectional regressions of returns on factors
regression = signal_layer.regress_factors(returns, exposures)
idiosyncratic = regression.residuals
zscore_alpha = signal_layer.idiosyncratic_zscore(idiosyncratic)

# Cross-sectional ML forecast using MultiIndex features (date, ticker)
features = exposures.stack().swaplevel().sort_index()
target = returns.stack().sort_index()
forecaster = CrossSectionalMLForecaster()
forecaster.fit(features, target)
latest_signals = forecaster.predict_latest(features)
```

## Alpha Vantage Integration

The bot now integrates with Alpha Vantage to provide enhanced signals beyond yfinance data.

### Technical Indicators
- **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Momentum and trend direction
- **ADX (Average Directional Index)**: Trend strength measurement
- **Bollinger Bands**: Volatility-based support/resistance
- **Stochastic Oscillator**: Momentum indicator

These indicators are integrated into the scanner's buy/boom scoring:
- **Buy score**: 70% traditional metrics + 30% Alpha Vantage technical composite
- **Boom score**: Base breakout score amplified by ADX trend strength

### Fundamental Data
The system can fetch and integrate fundamental metrics:
- **Quality metrics**: ROE, ROA, profit margin, operating margin
- **Valuation ratios**: P/E, PEG, Price-to-Book
- **Financial health**: Debt-to-Equity, Current Ratio, Quick Ratio

These enhance the cross-sectional quality factor in the alpha layer.

### Macro Economic Indicators
The `MacroIndicatorLayer` provides:
- **Growth regime detection**: Expansion/contraction/neutral based on market trends
- **Volatility regime**: High/low/normal volatility classification
- **Risk regime**: Risk-on/risk-off/neutral sentiment
- **Market breadth**: Advance/decline ratios, new highs/lows
- **Sector rotation signals**: Relative strength across sectors

### Rate Limits
Alpha Vantage free tier allows:
- 5 API calls per minute
- 500 API calls per day

The provider includes automatic rate limiting. For production use with many tickers, consider upgrading to premium tier or implementing caching.

### Usage Example
```python
from finance_bot.alpha_vantage_provider import AlphaVantageProvider
from finance_bot.alpha import CrossSectionalSignalLayer

# Initialize provider
av_provider = AlphaVantageProvider(api_key="YOUR_KEY")

# Get technical indicators
indicators = av_provider.get_technical_indicators("AAPL", indicators=['RSI', 'MACD', 'ADX'])
score, individual_scores = av_provider.compute_technical_score(indicators)

# Get fundamental data
fundamentals = av_provider.get_fundamentals("AAPL")
quality_metrics = av_provider.extract_quality_metrics(fundamentals)

# Use with cross-sectional layer
signal_layer = CrossSectionalSignalLayer(
    av_provider=av_provider,
    use_av_technicals=True
)
exposures = signal_layer.compute_factor_exposures(prices, volumes)
enhanced_quality = signal_layer.compute_enhanced_quality_factor(prices, tickers)
```

## Notes
- The scanner uses yfinance universes and limits each category to the top 50 tickers for responsiveness.
- Boom scoring favors recent breakouts with volume expansion; buy scoring rewards positive momentum vs. volatility and distance above the moving average.
- Alpha Vantage integration is optional - the bot works with yfinance only if no API key is provided.
- Ensure your Discord bot has the `MESSAGE CONTENT INTENT` enabled for legacy commands.
