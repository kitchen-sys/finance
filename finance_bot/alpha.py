from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet


@dataclass
class CrossSectionalRegressionResult:
    """Outputs from cross-sectional factor regression."""

    factor_returns: pd.DataFrame
    residuals: pd.DataFrame


class CrossSectionalSignalLayer:
    """Compute cross-sectional factor exposures and idiosyncratic signals."""

    def __init__(
        self,
        momentum_window: int = 21,
        value_window: int = 63,
        volatility_window: int = 21,
        quality_window: int = 21,
        size_window: int = 63,
    ) -> None:
        self.momentum_window = momentum_window
        self.value_window = value_window
        self.volatility_window = volatility_window
        self.quality_window = quality_window
        self.size_window = size_window

    def compute_factor_exposures(
        self, prices: pd.DataFrame, volumes: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Derive factor exposures per asset and date.

        Returns a DataFrame with a MultiIndex on columns of the shape
        ``(factor, asset)``. Supported factors include momentum, value,
        quality, low volatility, and (optionally) size when volumes are
        provided.
        """

        returns = prices.pct_change()
        momentum = prices.pct_change(self.momentum_window)
        value = prices.rolling(self.value_window).mean() / prices
        volatility = returns.rolling(self.volatility_window).std()
        quality = returns.rolling(self.quality_window).mean() / (volatility + 1e-8)

        factors: Dict[str, pd.DataFrame] = {
            "momentum": momentum,
            "value": value,
            "quality": quality,
            "low_vol": -volatility,
        }

        if volumes is not None:
            dollar_vol = prices * volumes
            size = np.log(dollar_vol.rolling(self.size_window).mean())
            factors["size"] = size

        exposures = pd.concat(factors, axis=1)
        return exposures

    def regress_factors(
        self, returns: pd.DataFrame, exposures: pd.DataFrame
    ) -> CrossSectionalRegressionResult:
        """Run per-date cross-sectional regressions of returns on factors."""

        factor_returns: list[pd.Series] = []
        residuals: Dict[pd.Timestamp, pd.Series] = {}

        for date in returns.index:
            y = returns.loc[date].dropna()
            if y.empty:
                continue

            exposures_for_date = exposures.loc[date]
            exposures_matrix = exposures_for_date.unstack(level=0)
            exposures_matrix = exposures_matrix.loc[y.index].dropna()

            if exposures_matrix.shape[0] <= exposures_matrix.shape[1]:
                continue

            y_aligned = y.loc[exposures_matrix.index]
            X = exposures_matrix.values
            X_with_const = np.column_stack([np.ones(len(X)), X])
            beta, *_ = np.linalg.lstsq(X_with_const, y_aligned.values, rcond=None)

            factor_ret = pd.Series(beta[1:], index=exposures_matrix.columns, name=date)
            factor_returns.append(factor_ret)

            fitted = X_with_const @ beta
            residuals[date] = pd.Series(y_aligned.values - fitted, index=y_aligned.index)

        factor_returns_df = pd.DataFrame(factor_returns)
        residuals_df = pd.DataFrame(residuals).T
        return CrossSectionalRegressionResult(factor_returns=factor_returns_df, residuals=residuals_df)

    def idiosyncratic_zscore(self, residuals: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Compute rolling z-scores of idiosyncratic returns for selection."""

        mean = residuals.rolling(window).mean()
        std = residuals.rolling(window).std(ddof=1)
        return (residuals - mean) / (std + 1e-8)


class CrossSectionalMLForecaster:
    """Simple tree + elastic-net ensemble for cross-sectional forecasting."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self.tree_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        self.elastic_net = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Fit models on a MultiIndex feature set with ``(date, asset)`` index."""

        aligned = features.join(target.rename("target"), how="inner")
        aligned = aligned.dropna()
        if aligned.empty:
            raise ValueError("No overlapping features and targets to train on.")

        X = aligned[features.columns].values
        y = aligned["target"].values
        self.tree_model.fit(X, y)
        self.elastic_net.fit(X, y)

    def predict_latest(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-sectional forecasts for the latest date in features."""

        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("Features must use a MultiIndex of (date, asset).")

        latest_date = features.index.get_level_values(0).max()
        latest = features.xs(latest_date, level=0).dropna()
        if latest.empty:
            raise ValueError("No features available for the latest date.")

        tree_pred = self.tree_model.predict(latest)
        elastic_pred = self.elastic_net.predict(latest)
        blended = (tree_pred + elastic_pred) / 2
        return pd.DataFrame(
            {
                "tree": tree_pred,
                "elastic_net": elastic_pred,
                "blended": blended,
            },
            index=latest.index,
        )
