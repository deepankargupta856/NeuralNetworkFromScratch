import numpy as np

class ActivationSoftmax:
    """
    Softmax activation layer.

    Applies the softmax function to each input row:
        output_i = exp(input_i) / sum_j exp(input_j)

    Attributes:
        inputs (np.ndarray): Cached inputs from forward pass.
        output (np.ndarray): Softmax probabilities.
        dinputs (np.ndarray): Gradients of the loss w.r.t. inputs.
    """

    def forward(self, inputs: np.ndarray) -> None:
        """
        Compute the forward pass of softmax activation.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, n_features).
        """
        # Cache inputs for use in backward pass
        self.inputs = inputs
        # Numeric stability: subtract max per-row
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize to get probabilities
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Compute the backward pass of softmax activation.

        Args:
            dvalues (np.ndarray): Gradient of the loss w.r.t. 
                                  softmax outputs, shape (batch_size, n_features).
        """
        # Allocate space for the gradient w.r.t. inputs
        self.dinputs = np.zeros_like(dvalues)

        # Iterate over each sample to apply the Jacobian matrix
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array to column vector
            y = single_output.reshape(-1, 1)
            # Compute Jacobian: diag(y) - y @ y.T
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            # Gradient from this sample: J · dL/dy
            self.dinputs[index] = np.dot(jacobian, single_dvalues)
