import numpy as np
from layers.activations.softmax import ActivationSoftmax
from losses.CategoricalCrossEntropy import LossCategoricalCrossentropy

class ActivationSoftmaxLossCategoricalCrossentropy:
    """
    Combines Softmax activation with Categorical Cross-Entropy loss in one module
    for improved numerical stability and a streamlined backward computation.

    Attributes:
        activation (ActivationSoftmax): Softmax activation instance.
        loss (LossCategoricalCrossentropy): Categorical cross-entropy loss instance.
        output (np.ndarray): Softmax probabilities from the forward pass.
        dinputs (np.ndarray): Gradients of the loss with respect to inputs.
    """

    def __init__(self) -> None:
        # Initialize constituent activation and loss objects
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()
        # Placeholders for outputs and gradients
        self.output: np.ndarray
        self.dinputs: np.ndarray

    def forward(self, inputs: np.ndarray, y_true: np.ndarray) -> float:
        """
        Forward pass: applies Softmax then computes loss.

        Args:
            inputs (np.ndarray): Logits array of shape (batch_size, n_classes).
            y_true (np.ndarray): True labels, either integer array
                                 of shape (batch_size,) or one-hot
                                 encoded of shape (batch_size, n_classes).

        Returns:
            float: Mean categorical cross-entropy loss over the batch.
        """
        # Ensure correct dimensions
        assert inputs.ndim == 2, "inputs must be 2D"
        assert y_true.shape[0] == inputs.shape[0], "batch size mismatch"

        # Compute softmax probabilities
        self.activation.forward(inputs)
        self.output = self.activation.output

        # Compute and return the loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Backward pass: computes gradient of combined Softmax + loss.

        Uses the identity: dL/dz = (softmax_output - true_labels) / batch_size

        Args:
            dvalues (np.ndarray): Softmax output probabilities, shape (batch_size, n_classes).
            y_true (np.ndarray): True labels in same format as forward.
        """
        # Number of samples in batch
        samples = dvalues.shape[0]

        # Convert one-hot labels to class indices if necessary
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy probabilities to modify in-place
        self.dinputs = dvalues.copy()

        # Subtract 1.0 at the positions of the correct classes
        self.dinputs[np.arange(samples), y_true] -= 1

        # Normalize gradient to average over batch
        self.dinputs /= samples
