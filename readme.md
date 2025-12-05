# OT-Powered Finance Scanner & Discord Bot

This project provides a lightweight implementation of an optimal-transport inspired regime detection engine, portfolio utilities, and a live Discord bot that scans equities in near real time using **yfinance**. The scanner ranks the top 50 tickers per universe and surfaces the top 5 "buy" and "boom" candidates per run.

## Features
- **Optimal Transport edge**: Wasserstein-based regime detection, OT distance matrices, distributionally robust portfolio weights, and OT-influenced rebalancing utilities.
- **Time series toolkit**: Auto-ARIMA fitting and forecasting, Kalman filtering for latent regimes, and GARCH/EGARCH/GJR-GARCH/FIGARCH volatility forecasting to support VaR/CVaR-style risk metrics.
- **Portfolio construction**: Mean-variance, risk-parity, Wasserstein-robust, and regime-adaptive allocation helpers plus a simple event-driven backtester.
- **Discord live scanner**: Pulls the latest data for S&P 500, NASDAQ 100, Dow, and most-active lists; outputs the top 5 buy and boom setups for each category.
- **Alpha / signal layer**: Cross-sectional models (Fama–French style factors, factor regressions, idiosyncratic return isolation), ML forecasters (tree-based, regularized linear, sequence models), and causal/event estimators (causal forests, doubly robust learners, OT-based treated vs control comparison) to drive “own this, short that, right now” decisions.
- **Factor risk model**: Systematic vs idiosyncratic risk decomposition, factor covariance vs idio variances, marginal/component risk contributions per factor and per asset, and liquidity-aware haircuts.
- **Dependence & tail risk**: Dynamic-correlation (DCC-style) matrices, copula-derived tail dependence, OT-based joint dependence gaps, extreme-value POT fits, and liquidity/funding-aware constraints.

## Quickstart
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set environment variables** (or pass CLI args):
   - `DISCORD_TOKEN`: your bot token
   - `DISCORD_CHANNELS`: comma-separated channel IDs for periodic scans
   - `SCAN_CATEGORIES`: optional list of universes (sp500,nasdaq100,dow,most_active)
3. **Run the bot**
   ```bash
   python main.py --channels 123456789012345678 234567890123456789
   ```
   Use `--history`, `--lookback-regime`, `--lookback-buy`, and `--lookback-boom` to tune the model.

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

### Factor risk & dependence example
```python
import pandas as pd
from finance_bot import FactorRiskModel

# factor_returns: DataFrame of factor return series
# exposures: MultiIndex columns (factor, asset)
# residuals: idiosyncratic residuals from factor regression
# weights: portfolio weights as a Series indexed by asset
risk_model = FactorRiskModel()
decomp = risk_model.decompose(weights, exposures, factor_returns, residuals)

# Dynamic correlations, copula tail dependence, and OT-based joint gaps
dcc_corr = risk_model.dcc_dynamic_correlation(returns)
tail_dep = risk_model.copula_tail_dependence(returns)
ot_joint = risk_model.ot_dependence_matrix(returns)

# Extreme value tail fit and liquidity-aware haircuts
shape, threshold, scale = risk_model.pot_extreme_value(returns["SPY"])
days_to_liq = risk_model.liquidity_score(weights, prices, volumes)
constrained_weights = risk_model.apply_haircuts(weights, haircuts, leverage_limit=1.5)
```

## Notes
- The scanner uses yfinance universes and limits each category to the top 50 tickers for responsiveness.
- Boom scoring favors recent breakouts with volume expansion; buy scoring rewards positive momentum vs. volatility and distance above the moving average.
- Ensure your Discord bot has the `MESSAGE CONTENT INTENT` enabled for legacy commands.
