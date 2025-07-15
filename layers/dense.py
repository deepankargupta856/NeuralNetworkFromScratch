import numpy as np

class LayerDense:
    """
    A fully connected (dense) neural network layer.

    Attributes:
        weights (np.ndarray): Weight matrix of shape (n_inputs, n_neurons).
        biases (np.ndarray): Bias vector of shape (1, n_neurons).
        weight_regularizer_l1 (float): L1 regularization strength for weights.
        weight_regularizer_l2 (float): L2 regularization strength for weights.
        bias_regularizer_l1 (float): L1 regularization strength for biases.
        bias_regularizer_l2 (float): L2 regularization strength for biases.
        inputs (np.ndarray): Cached inputs from the forward pass.
        outputs (np.ndarray): Output values from the forward pass.
        dweights (np.ndarray): Gradient of the loss w.r.t. weights.
        dbiases (np.ndarray): Gradient of the loss w.r.t. biases.
        dinputs (np.ndarray): Gradient of the loss w.r.t. inputs.
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        weight_regularizer_l1: float = 0.0,
        weight_regularizer_l2: float = 0.0,
        bias_regularizer_l1: float = 0.0,
        bias_regularizer_l2: float = 0.0,
    ) -> None:
        # Initialize learnable parameters
        self.weights = 0.01 * np.random.random((n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))  # row vector for broadcasting

        # Regularization coefficients
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs: np.ndarray) -> None:
        """
        Perform the forward pass.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, n_inputs).
        """
        self.inputs = inputs
        # Weighted sum plus bias
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Perform the backward pass (compute gradients).

        Args:
            dvalues (np.ndarray): Gradient of the loss w.r.t. this layer's outputs,
                                  shape (batch_size, n_neurons).
        """
        # Gradients on weights and biases
        self.dweights = np.dot(self.inputs.T, dvalues)  # shape: (n_inputs, n_neurons)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)  # shape: (1, n_neurons)

        # ----- Weight regularization -----
        # L1 regularization (adds |W| penalty)
        if self.weight_regularizer_l1 > 0:
            sign_weights = np.where(self.weights >= 0, 1, -1)
            self.dweights += self.weight_regularizer_l1 * sign_weights

        # L2 regularization (adds W^2 penalty)
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # ----- Bias regularization -----
        # L1 regularization on biases
        if self.bias_regularizer_l1 > 0:
            sign_biases = np.where(self.biases >= 0, 1, -1)
            self.dbiases += self.bias_regularizer_l1 * sign_biases

        # L2 regularization on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on inputs to pass to previous layer
        self.dinputs = np.dot(dvalues, self.weights.T)  # shape: (batch_size, n_inputs)
