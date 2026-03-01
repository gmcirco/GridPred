from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Union, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skexplain

class GridPredPredictor(ABC):
    def __init__(self, ale_params: Optional[dict] = None, **model_params):
        self.model = None
        self.model_params = model_params
        
        # default ale params
        self.ale_params = {
            'n_bootstrap': 5, 
            'subsample': 500, 
            'n_bins': 10, 
            'n_jobs': 1
        }

        # or override user-provided dict if it exists
        if ale_params:
            self.ale_params.update(ale_params)

        self._X_fit = None
        self._y_fit = None
        self._explainer = None
        self._ale_ds_1d = None
        self._ale_ds_2d = {}

    @abstractmethod
    def build_model(self) -> Any:
        """Return an ML model instance. Must be implemented in subclass."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model and cache data for ALE calculations."""
        if self.model is None:
            self.model = self.build_model()
        
        self.model.fit(X, y)
        
        # Cache data for post-hoc explanations
        self._X_fit = X.copy()
        self._y_fit = y.copy()

        # Reset explainer if re-fitting
        self._explainer = None 
        self._ale_ds_1d = None
        self._ale_ds_2d = {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions."""
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.model.predict(X)

    def get_feature_importances(self) -> pd.Series:
        """Provides feature importances if supported."""
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")

        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_, index=self._X_fit.columns)
        else:
            raise AttributeError(f"{type(self.model).__name__} does not provide feature importances.")

    # --- ALE Computation Logic ---

    def _init_explainer(self):
        """Lazy-load the skexplain toolkit."""
        if self._explainer is None:
            if self.model is None:
                raise RuntimeError("Model must be fit before explaining.")
            # skexplain requires a name-model tuple
            self._explainer = skexplain.ExplainToolkit(
                estimators=(type(self.model).__name__, self.model),
                X=self._X_fit,
                y=self._y_fit,
            )
        return self._explainer

    def run_ale(self, features: Union[str, List] = 'all', **kwargs):
        """
        Computes ALE using merged parameters.
        **kwargs here take the highest priority (e.g. model.run_ale(n_bins=50)).
        """
        explainer = self._init_explainer()
        
        # 3. Merge: Global Defaults < Class Init Params < Method Call Params
        run_params = {**self.ale_params, **kwargs}

        if isinstance(features, list) and len(features) > 0 and isinstance(features[0], tuple):
            ds = explainer.ale(features=features, **run_params)
            for pair in features:
                self._ale_ds_2d[tuple(sorted(pair))] = ds
        else:
            self._ale_ds_1d = explainer.ale(features=features, **run_params)
        
        return self 

    # --- ALE Plotting Logic ---

    def plot_ale_1d(self, feature: str, xlim=None, ylim=None):
        """Plots the 1D ALE for a specific feature."""
        if self._ale_ds_1d is None:
            self.run_ale(features='all')

        ds = self._ale_ds_1d
        model_name = type(self.model).__name__
        
        v_bin = ds[f"{feature}__bin_values"].values
        v_ale = ds[f"{feature}__{model_name}__ale"].values

        # Aggregate across bootstrap iterations
        val_min, val_avg, val_max = v_ale.min(axis=0), v_ale.mean(axis=0), v_ale.max(axis=0)

        plt.figure(figsize=(5, 4))
        plt.fill_between(v_bin, val_min, val_max, alpha=0.2, color='#0077BB')
        sns.lineplot(x=v_bin, y=val_avg, color='#0077BB', label='Mean ALE')
        plt.axhline(y=0, color='#CC3311', linestyle='--', linewidth=1)
        
        plt.title(f"1D ALE: {feature}")
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        sns.despine()
        plt.show()

    def plot_ale_2d(self, feat_x: str, feat_y: str, clim=None):
        """Plots the 2D interaction ALE for two features."""
        pair = tuple(sorted((feat_x, feat_y)))
        if pair not in self._ale_ds_2d:
            self.run_ale(features=[pair])

        ds = self._ale_ds_2d[pair]
        model_name = type(self.model).__name__

        # 1. Identify which feature is indexed as 'dim_0' vs 'dim_1' in the ALE array
        # skexplain usually lists them in the order they were passed to run_ale
        key = f"{feat_x}__{feat_y}__{model_name}__ale"
        if key in ds:
            ale_vals = ds[key].values.mean(axis=0)
        else:
            # If the user swapped the order, we look for the alternative key
            # and transpose the ALE values to match (feat_x, feat_y)
            key = f"{feat_y}__{feat_x}__{model_name}__ale"
            ale_vals = ds[key].values.mean(axis=0).T # Transpose to align

        x_bin = ds[f"{feat_x}__bin_values"].values
        y_bin = ds[f"{feat_y}__bin_values"].values

        # 2. Use indexing="ij" for the meshgrid (Matrix style)
        X, Y = np.meshgrid(x_bin, y_bin, indexing="ij")

        # 3. Handle Scaling
        vmax = np.abs(ale_vals).max() if clim is None else clim[1]
        vmin = -vmax if clim is None else clim[0]

        # 4. Plot
        plt.figure(figsize=(7, 6))
        # Note: We pass ale_vals directly now because it is aligned with X and Y
        contour = plt.contourf(X, Y, ale_vals, levels=20, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        
        plt.colorbar(contour, label="ALE Effect")
        plt.title(f"2D ALE Interaction: {feat_x} vs {feat_y}")
        plt.xlabel(feat_x)
        plt.ylabel(feat_y)
        plt.show()