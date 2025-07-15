import numpy as np
from losses.Loss import Loss

class LossCategoricalCrossentropy(Loss):
    """
    Categorical Cross-Entropy loss function.

    Computes the loss and gradient for both sparse (integer) labels
    and one-hot encoded labels.

    Attributes:
        dinputs (np.ndarray): Gradient of the loss with respect to predictions.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculate the negative log likelihood loss for each sample.

        Args:
            y_pred (np.ndarray): Predicted probabilities, shape (batch_size, n_classes).
            y_true (np.ndarray): True labels, either sparse (shape (batch_size,))
                                 or one-hot encoded (shape (batch_size, n_classes)).

        Returns:
            np.ndarray: Array of per-sample losses, shape (batch_size,).
        """
        # Number of samples in batch
        samples = y_pred.shape[0]

        # Clip predictions to avoid log(0) and 1 - clipped to avoid infinite gradient
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Select the predicted probability for the correct class
        if y_true.ndim == 1:  # sparse labels
            correct_confidences = y_pred_clipped[np.arange(samples), y_true]
        else:  # one-hot encoded labels
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Compute negative log likelihoods
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Calculate the gradient of the loss with respect to the predictions.

        Args:
            dvalues (np.ndarray): Predicted probabilities (same as forward input),
                                  shape (batch_size, n_classes).
            y_true (np.ndarray): True labels, either sparse or one-hot encoded.
        """
        # Number of samples and number of classes
        samples = dvalues.shape[0]
        n_classes = dvalues.shape[1]

        # Convert sparse labels to one-hot encoding
        if y_true.ndim == 1:
            y_true = np.eye(n_classes)[y_true]

        # Gradient: -y_true / y_pred
        self.dinputs = -y_true / dvalues
        # Average gradients across the batch
        self.dinputs = self.dinputs / samples
