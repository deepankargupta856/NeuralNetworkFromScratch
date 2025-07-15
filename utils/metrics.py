import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Abstract base class for all metrics.
    Ensures consistency across metric interfaces.
    """
    def __init__(self):
        self.result = None

    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the metric value and store in self.result.
        Should return a float value.
        """
        pass

    def reset(self):
        """Reset metric state if accumulated across batches."""
        self.result = None

class CategoricalAccuracy(Metric):
    """
    Computes classification accuracy for one-hot or sparse targets.
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_true == y_pred)
        self.result = accuracy
        return accuracy

class MeanSquaredError(Metric):
    """
    Computes Mean Squared Error (MSE).
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        self.result = mse
        return mse

class MeanAbsoluteError(Metric):
    """
    Computes Mean Absolute Error (MAE).
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mae = np.mean(np.abs(y_true - y_pred))
        self.result = mae
        return mae
