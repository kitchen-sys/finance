from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import cvxpy as cp


@dataclass
class RegimeDetectionResult:
    regime_change: bool
    distance: float
    current_state: str
    threshold: float


@dataclass
class PortfolioWeights:
    weights: pd.Series
    risk_contribution: pd.Series


@dataclass
class VaRResult:
    var: float
    cvar: float
    alpha: float


@dataclass
class ForecastResult:
    forecast: float
    stderr: float
    conf_int: Tuple[float, float]


class OTFinanceModel:
    """End-to-end model combining OT-driven regime detection and allocation logic."""

    def __init__(self, lookback_regime: int = 60, risk_free_rate: float = 0.02) -> None:
        self.lookback_regime = lookback_regime
        self.risk_free_rate = risk_free_rate

    def _rolling_wasserstein(
        self, returns: pd.Series, window: int
    ) -> List[Tuple[pd.Timestamp, float]]:
        scores: List[Tuple[pd.Timestamp, float]] = []
        if len(returns) < window * 2:
            return scores
        for end in range(window * 2, len(returns) + 1):
            ref = returns.iloc[end - window * 2 : end - window]
            cur = returns.iloc[end - window : end]
            distance = wasserstein_distance(ref, cur)
            scores.append((returns.index[end - 1], distance))
        return scores

    def detect_regime(self, prices: pd.Series) -> RegimeDetectionResult:
        log_returns = np.log(prices).diff().dropna()
        scores = self._rolling_wasserstein(log_returns, self.lookback_regime)
        if not scores:
            return RegimeDetectionResult(False, 0.0, "stable", 0.0)
        dates, distances = zip(*scores)
        distance_series = pd.Series(distances, index=pd.DatetimeIndex(dates))
        threshold = distance_series.rolling(20).mean().iloc[-1] + distance_series.rolling(20).std().iloc[-1]
        regime_change = distance_series.iloc[-1] > threshold
        current_state = "shift" if regime_change else "stable"
        return RegimeDetectionResult(regime_change, distance_series.iloc[-1], current_state, threshold)

    def fit_arima(self, series: pd.Series, order: Tuple[int, int, int]) -> ARIMA:
        model = ARIMA(series, order=order)
        return model.fit()

    def auto_arima_order(
        self, series: pd.Series, p_range: range = range(0, 3), d_range: range = range(0, 2), q_range: range = range(0, 3)
    ) -> Tuple[int, int, int]:
        best_aic = np.inf
        best_order = (1, 0, 0)
        series = series.dropna()
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    if p == d == q == 0:
                        continue
                    try:
                        res = ARIMA(series, order=(p, d, q)).fit()
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue
        return best_order

    def forecast_arima(self, series: pd.Series, steps: int = 1) -> ForecastResult:
        order = self.auto_arima_order(series)
        fitted = self.fit_arima(series, order)
        forecast = fitted.get_forecast(steps=steps)
        mean_forecast = float(forecast.predicted_mean.iloc[-1])
        stderr = float(np.sqrt(forecast.var_pred_mean.iloc[-1]))
        lower, upper = forecast.conf_int().iloc[-1]
        return ForecastResult(mean_forecast, stderr, (float(lower), float(upper)))

    def fit_garch(
        self,
        returns: pd.Series,
        model: str = "GARCH",
        p: int = 1,
        o: int = 0,
        q: int = 1,
        power: float = 2.0,
    ):
        model = model.upper()
        garch = arch_model(returns * 100, vol=model, p=p, o=o, q=q, power=power)
        return garch.fit(disp="off")

    def forecast_volatility(self, returns: pd.Series, garch_type: str = "GARCH") -> pd.Series:
        garch_types = {
            "GARCH": {"p": 1, "o": 0, "q": 1, "power": 2.0},
            "EGARCH": {"p": 1, "o": 1, "q": 1, "power": 2.0},
            "GJR-GARCH": {"p": 1, "o": 1, "q": 1, "power": 2.0},
            "FIGARCH": {"p": 1, "o": 0, "q": 1, "power": 2.0},
        }
        params = garch_types.get(garch_type.upper(), garch_types["GARCH"])
        res = self.fit_garch(returns, model=garch_type, **params)
        return res.conditional_volatility / 100

    def kalman_state_space(self, observations: pd.Series) -> KalmanFilter:
        kf = KalmanFilter(k_endog=1, k_states=1)
        kf.bind(observations.values[:, None])
        return kf

    def kalman_regime_filter(
        self, observations: pd.Series, transition_std: float = 0.05
    ) -> pd.Series:
        """Simple Kalman filter smoothing to infer latent regime levels."""

        kf = KalmanFilter(k_endog=1, k_states=1)
        kf.initialize_known(np.array([observations.iloc[0]]), np.eye(1))
        kf['transition'] = np.eye(1)
        kf['selection'] = np.eye(1)
        kf['state_cov'] = np.eye(1) * transition_std
        kf.bind(observations.values[:, None])
        res = kf.filter()
        smoothed = res.filtered_state[0]
        return pd.Series(smoothed, index=observations.index)

    def cvar(self, returns: pd.Series, alpha: float = 0.95) -> float:
        sorted_returns = returns.sort_values()
        cutoff = int((1 - alpha) * len(sorted_returns))
        tail = sorted_returns.iloc[:cutoff]
        return tail.mean()

    def value_at_risk(self, returns: pd.Series, alpha: float = 0.95) -> VaRResult:
        var = float(returns.quantile(1 - alpha))
        cvar = float(self.cvar(returns, alpha))
        return VaRResult(var=var, cvar=cvar, alpha=alpha)

    def sharpe_ratio(self, returns: pd.Series) -> float:
        excess = returns - self.risk_free_rate / 252
        return np.sqrt(252) * excess.mean() / excess.std(ddof=1)

    def mean_variance_portfolio(self, returns: pd.DataFrame) -> PortfolioWeights:
        covariance = returns.cov()
        means = returns.mean()
        inv_cov = np.linalg.pinv(covariance)
        raw_weights = inv_cov.dot(means)
        weights = raw_weights / raw_weights.sum()
        risk_contribution = weights * covariance.dot(weights)
        return PortfolioWeights(pd.Series(weights, index=returns.columns), pd.Series(risk_contribution, index=returns.columns))

    def risk_parity_weights(self, returns: pd.DataFrame) -> PortfolioWeights:
        covariance = returns.cov().values
        n_assets = covariance.shape[0]
        w = cp.Variable(n_assets)
        port_var = cp.quad_form(w, covariance)
        rc = cp.multiply(w, covariance @ w)
        objective = cp.Minimize(cp.sum_squares(rc - port_var / n_assets))
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(warm_start=True)
        weights = np.array(w.value).flatten()
        weights = weights / weights.sum()
        risk_contribution = weights * covariance.dot(weights)
        return PortfolioWeights(pd.Series(weights, index=returns.columns), pd.Series(risk_contribution, index=returns.columns))

    def wasserstein_rebalance(self, returns: pd.DataFrame) -> PortfolioWeights:
        pairwise = returns.apply(lambda col: self._rolling_wasserstein(col, self.lookback_regime), axis=0)
        avg_distance = pd.Series({
            col: np.mean([d for _, d in distances]) if distances else 0.0
            for col, distances in pairwise.items()
        })
        stability = 1 / (1 + avg_distance)
        weights = stability / stability.sum()
        covariance = returns.cov()
        risk_contribution = weights * covariance.dot(weights)
        return PortfolioWeights(weights, pd.Series(risk_contribution, index=returns.columns))

    def regime_adaptive_weights(self, prices: pd.DataFrame) -> PortfolioWeights:
        returns = prices.pct_change().dropna()
        detection = self.detect_regime(prices.mean(axis=1))
        if detection.regime_change:
            return self.wasserstein_rebalance(returns.tail(self.lookback_regime))
        return self.mean_variance_portfolio(returns.tail(self.lookback_regime))

    def is_stationary(self, series: pd.Series, alpha: float = 0.05) -> bool:
        statistic, p_value, *_ = adfuller(series.dropna())
        return p_value < alpha

    def dro_wasserstein_radius(self, returns: pd.Series, confidence: float = 0.9) -> float:
        return np.sqrt(2 * np.log(1 / (1 - confidence)) / len(returns))

    def distributionally_robust_weights(
        self, returns: pd.DataFrame, confidence: float = 0.9
    ) -> PortfolioWeights:
        mu = returns.mean().values
        sigma = returns.cov().values
        n_assets = len(mu)
        w = cp.Variable(n_assets)
        radius = self.dro_wasserstein_radius(returns.stack(), confidence)
        # Penalize variance and introduce Wasserstein robustness through L2 ball
        objective = cp.Maximize(mu @ w - radius * cp.norm(w, 2) - 0.5 * cp.quad_form(w, sigma))
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(warm_start=True)
        weights = np.array(w.value).flatten()
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        covariance = returns.cov()
        risk_contribution = weights * covariance.dot(weights)
        return PortfolioWeights(pd.Series(weights, index=returns.columns), pd.Series(risk_contribution, index=returns.columns))

    def backtest_event_driven(
        self,
        prices: pd.DataFrame,
        strategy_weights: PortfolioWeights,
        rebalance_cost: float = 0.0005,
    ) -> pd.Series:
        returns = prices.pct_change().dropna()
        portfolio_returns = returns.dot(strategy_weights.weights)
        turnover = strategy_weights.weights.diff().abs().sum()
        costs = turnover * rebalance_cost
        after_cost = portfolio_returns - costs
        return (1 + after_cost).cumprod()

    def ot_distance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        distances = pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)
        for i in returns.columns:
            for j in returns.columns:
                distances.loc[i, j] = wasserstein_distance(returns[i].dropna(), returns[j].dropna())
        return distances

    def regime_path(self, prices: pd.Series) -> pd.Series:
        log_returns = np.log(prices).diff().dropna()
        distances = self._rolling_wasserstein(log_returns, self.lookback_regime)
        if not distances:
            return pd.Series(dtype=float)
        dates, vals = zip(*distances)
        series = pd.Series(vals, index=pd.DatetimeIndex(dates))
        threshold = series.rolling(20).mean() + series.rolling(20).std()
        states = np.where(series > threshold, 1, 0)
        return pd.Series(states, index=series.index)
