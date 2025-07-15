import numpy as np

class ActivationReLU:
    """
    Rectified Linear Unit (ReLU) activation layer.

    Applies the element-wise function:
        output = max(0, input)

    Attributes:
        inputs (np.ndarray): Cached input values from the forward pass.
        output (np.ndarray): Output values after applying ReLU.
        dinputs (np.ndarray): Gradients of the loss with respect to inputs.
    """

    def forward(self, inputs: np.ndarray) -> None:
        """
        Compute the forward pass of the ReLU activation.

        Args:
            inputs (np.ndarray): Input data of any shape.
        """
        # Cache the input values for use during backpropagation
        self.inputs = inputs
        # Apply ReLU: zero out negative values
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Compute the backward pass of the ReLU activation.

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the layer's outputs,
                                  same shape as `inputs`.
        """
        # Copy upstream gradients to avoid modifying the original array
        self.dinputs = dvalues.copy()
        # Zero out gradients where the original inputs were negative or zero
        self.dinputs[self.inputs <= 0] = 0
