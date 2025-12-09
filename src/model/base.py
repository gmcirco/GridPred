from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import numpy as np


class GridPredPredictor(ABC):
    """
    Base class defining the interface for ML models used with GridPred.
    """

    def __init__(self, **model_params):
        self.model = None
        self.model_params = model_params

    @abstractmethod
    def build_model(self) -> Any:
        """Return an ML model instance. Must be implemented in subclass."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model using prepared GridPred data.
        """
        if self.model is None:
            self.model = self.build_model()
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return predictions.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.model.predict(X)

    def get_feature_importances(self) -> pd.Series:
        """
        Provides feature importances if supported, else raises helpful error.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")

        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        else:
            raise AttributeError(f"{type(self.model).__name__} does not provide feature importances.")
