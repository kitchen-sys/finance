# OT-Powered Finance Scanner & Discord Bot

This project provides a lightweight implementation of an optimal-transport inspired regime detection engine, portfolio utilities, and a live Discord bot that scans equities in near real time using **yfinance**. The scanner ranks the top 50 tickers per universe and surfaces the top 5 "buy" and "boom" candidates per run.

## Features
- **Optimal Transport edge**: Wasserstein-based regime detection, OT distance matrices, distributionally robust portfolio weights, and OT-influenced rebalancing utilities.
- **Time series toolkit**: Auto-ARIMA fitting and forecasting, Kalman filtering for latent regimes, and GARCH/EGARCH/GJR-GARCH/FIGARCH volatility forecasting to support VaR/CVaR-style risk metrics.
- **Portfolio construction**: Mean-variance, risk-parity, Wasserstein-robust, and regime-adaptive allocation helpers plus a simple event-driven backtester.
- **Discord live scanner**: Pulls the latest data for S&P 500, NASDAQ 100, Dow, and most-active lists; outputs the top 5 buy and boom setups for each category.

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

## Notes
- The scanner uses yfinance universes and limits each category to the top 50 tickers for responsiveness.
- Boom scoring favors recent breakouts with volume expansion; buy scoring rewards positive momentum vs. volatility and distance above the moving average.
- Ensure your Discord bot has the `MESSAGE CONTENT INTENT` enabled for legacy commands.
