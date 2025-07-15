import numpy as np
from typing import List, Optional, Callable, Dict, Any

class Sequential:
    """
    A minimal deep learning sequential model for training feedforward networks.
    Allows stacking layers, compiling with loss and optimizer, and fitting on data.
    """

    def __init__(self) -> None:
        self.layers: List[Any] = []
        self.loss: Optional[Any] = None
        self.optimizer: Optional[Any] = None
        self.metrics: List[Callable[[np.ndarray, np.ndarray], float]] = []

    def add(self, layer: Any) -> None:
        """Append a layer to the model's computation graph."""
        self.layers.append(layer)

    def compile(
        self,
        *,
        loss: Any,
        optimizer: Any,
        metrics: Optional[List[Callable[[np.ndarray, np.ndarray], float]]] = None
    ) -> None:
        """
        Set the loss, optimizer, and metrics for training.

        Args:
            loss: A loss class instance (with forward and backward methods)
            optimizer: Optimizer instance with update_params method
            metrics: List of metric functions
        """
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics or []

    def forward(self, X: np.ndarray,Y = None) -> np.ndarray:
        """
        Pass data through each layer sequentially.

        Args:
            X: Input features

        Returns:
            Output of the final layer
        """

        if Y != None :
            layer.forward(X,Y)
        for layer in self.layers:
            layer.forward(output)
            output = layer.outputs
        return output

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Perform backpropagation from output to input layer.

        Args:
            dvalues: Gradient of the loss with respect to the output
        """
        grad = dvalues
        for layer in reversed(self.layers):
            layer.backward(grad)
            grad = layer.dinputs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 1,
        batch_size: int = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model using forward and backward passes.

        Args:
            X: Input data
            y: Target labels
            epochs: Number of passes over the data
            batch_size: Size of training batches (None = full batch)
            verbose: Print progress if True

        Returns:
            Dictionary containing loss and metrics per epoch
        """
        n_samples = X.shape[0]
        batch_size = batch_size or n_samples
        history: Dict[str, List[float]] = {'loss': [], **{m.__name__: [] for m in self.metrics}}

        for epoch in range(1, epochs + 1):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y[indices]

            epoch_losses = []
            epoch_metrics = {m.__name__: [] for m in self.metrics}

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                try:
                    predictions = self.forward(X_batch)
                except TypeError:
                    predictions = self.forward(X_batch, y_batch)

                batch_loss = np.mean(self.loss.forward(predictions, y_batch))

                for m in self.metrics:
                    epoch_metrics[m.__name__].append(m(y_batch, predictions))

                dvalues = self.loss.backward(predictions, y_batch)
                self.backward(dvalues)

                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        self.optimizer.update_params(layer)

                epoch_losses.append(batch_loss)

            avg_loss = float(np.mean(epoch_losses))
            history['loss'].append(avg_loss)

            if verbose:
                msg = f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}"
                for name, vals in epoch_metrics.items():
                    avg_val = float(np.mean(vals))
                    history[name].append(avg_val)
                    msg += f" - {name}: {avg_val:.4f}"
                print(msg)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions on new data.

        Args:
            X: Input data

        Returns:
            Model output (e.g., logits or probabilities)
        """
        return self.forward(X)
