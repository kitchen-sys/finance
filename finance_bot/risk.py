from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import genpareto, rankdata, wasserstein_distance


@dataclass
class FactorRiskBreakdown:
    """Outputs from a factor risk decomposition."""

    factor_covariance: pd.DataFrame
    idiosyncratic_variance: pd.Series
    systematic_variance: pd.Series
    total_variance: pd.Series
    factor_marginal_contrib: pd.Series
    factor_component_contrib: pd.Series
    asset_marginal_contrib: pd.Series
    asset_component_contrib: pd.Series


class FactorRiskModel:
    """Systematic vs idiosyncratic risk modeling with dependence overlays."""

    def __init__(self, dcc_a: float = 0.01, dcc_b: float = 0.98, tail_alpha: float = 0.95) -> None:
        if dcc_a + dcc_b >= 1:
            raise ValueError("dcc_a + dcc_b must be < 1 for stationarity")
        self.dcc_a = dcc_a
        self.dcc_b = dcc_b
        self.tail_alpha = tail_alpha

    def factor_covariance(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Compute the covariance of factor returns."""

        return factor_returns.dropna().cov()

    def decompose(
        self,
        weights: pd.Series,
        exposures: pd.DataFrame,
        factor_returns: pd.DataFrame,
        residuals: pd.DataFrame,
    ) -> FactorRiskBreakdown:
        """Systematic vs idiosyncratic risk decomposition with contributions."""

        latest_date = exposures.index.get_level_values(0).max()
        exposures_today = exposures.loc[latest_date]
        loading_matrix = exposures_today.unstack(level=0).fillna(0)

        assets = loading_matrix.index
        weights_vec = weights.reindex(assets).fillna(0).values
        weights_vec = weights_vec / np.sum(np.abs(weights_vec)) if np.sum(np.abs(weights_vec)) != 0 else weights_vec

        factor_cov = self.factor_covariance(factor_returns)
        systematic_cov = loading_matrix.values @ factor_cov.values @ loading_matrix.values.T

        idio_var = residuals.var().reindex(assets).fillna(0)
        idio_cov = np.diag(idio_var.values)
        total_cov = systematic_cov + idio_cov

        systematic_var = pd.Series(np.diag(systematic_cov), index=assets)
        total_var = pd.Series(np.diag(total_cov), index=assets)

        asset_marginal = pd.Series(total_cov @ weights_vec, index=assets)
        asset_component = pd.Series(weights_vec * asset_marginal.values, index=assets)

        factor_exposure = pd.Series(loading_matrix.T.values @ weights_vec, index=loading_matrix.columns)
        factor_marginal = pd.Series(factor_cov.values @ factor_exposure.values, index=loading_matrix.columns)
        factor_component = factor_exposure * factor_marginal

        return FactorRiskBreakdown(
            factor_covariance=factor_cov,
            idiosyncratic_variance=idio_var,
            systematic_variance=systematic_var,
            total_variance=total_var,
            factor_marginal_contrib=factor_marginal,
            factor_component_contrib=factor_component,
            asset_marginal_contrib=asset_marginal,
            asset_component_contrib=asset_component,
        )

    def dcc_dynamic_correlation(self, returns: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Approximate DCC-style dynamic correlation matrices."""

        standardized = returns.sub(returns.mean()).div(returns.std(ddof=1)).dropna()
        unconditional = standardized.corr().values
        q_prev = unconditional.copy()
        results: Dict[pd.Timestamp, pd.DataFrame] = {}

        for date, row in standardized.iterrows():
            eps = row.values[:, None]
            q_t = (1 - self.dcc_a - self.dcc_b) * unconditional + self.dcc_a * (eps @ eps.T) + self.dcc_b * q_prev
            d_inv = np.diag(1 / np.sqrt(np.diag(q_t) + 1e-8))
            r_t = d_inv @ q_t @ d_inv
            results[date] = pd.DataFrame(r_t, index=standardized.columns, columns=standardized.columns)
            q_prev = q_t

        return results

    def copula_tail_dependence(self, returns: pd.DataFrame, tail: float = 0.05) -> pd.DataFrame:
        """Estimate lower- and upper-tail dependence coefficients via empirical copula ranks."""

        uniforms = returns.apply(lambda x: rankdata(x) / (len(x) + 1))
        lower = pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)
        upper = pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)

        for i in returns.columns:
            for j in returns.columns:
                u_i, u_j = uniforms[i], uniforms[j]
                lower.loc[i, j] = np.mean((u_i <= tail) & (u_j <= tail)) / tail
                upper.loc[i, j] = np.mean((u_i >= 1 - tail) & (u_j >= 1 - tail)) / tail

        lower_upper = (lower + upper) / 2
        lower_upper.index.name = "asset"
        lower_upper.columns.name = "asset"
        return lower_upper

    def ot_dependence_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """OT-based dependence using Wasserstein gaps between joint vs shuffled (independent) samples."""

        distances = pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)
        for i in returns.columns:
            for j in returns.columns:
                pair = returns[[i, j]].dropna()
                if pair.empty:
                    distances.loc[i, j] = np.nan
                    continue
                joint = np.linalg.norm(pair.values, axis=1)
                shuffled = np.linalg.norm(np.column_stack([pair[i].sample(frac=1).values, pair[j].values]), axis=1)
                distances.loc[i, j] = wasserstein_distance(joint, shuffled)
        return distances

    def pot_extreme_value(self, returns: pd.Series, threshold_quantile: float = 0.9) -> Tuple[float, float, float]:
        """Peaks-over-threshold fit for downside tail using a Generalized Pareto."""

        threshold = returns.quantile(threshold_quantile)
        tail_losses = -(returns[returns < threshold] - threshold)
        if tail_losses.empty:
            return 0.0, float(threshold), 0.0
        shape, loc, scale = genpareto.fit(tail_losses, floc=0)
        return float(shape), float(threshold), float(scale)

    def liquidity_score(
        self,
        positions: pd.Series,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        adv_window: int = 20,
        horizon_days: int = 5,
    ) -> pd.Series:
        """Estimate days-to-liquidate style liquidity scores per asset."""

        latest_date = prices.index.max()
        dollar_vol = prices * volumes
        adv = dollar_vol.rolling(adv_window).mean().loc[latest_date]
        position_value = positions * prices.loc[latest_date]
        days_to_liquidate = position_value.abs() / (adv * horizon_days + 1e-8)
        return days_to_liquidate.fillna(np.inf)

    def apply_haircuts(self, weights: pd.Series, haircuts: pd.Series, leverage_limit: float = 1.0) -> pd.Series:
        """Apply margin haircuts and rescale to meet a leverage limit."""

        adjusted = weights * (1 - haircuts.reindex(weights.index).fillna(0))
        gross = adjusted.abs().sum()
        if gross > leverage_limit:
            adjusted = adjusted * (leverage_limit / gross)
        return adjusted / adjusted.sum()
