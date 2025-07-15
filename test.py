import numpy as np


# import matplotlib.pyplot as plt

def generate_spiral_data(samples: int, classes: int):
    """
    Generate a 2D spiral dataset.

    Args:
        samples (int): Number of points per class.
        classes (int): Number of spiral arms (classes).

    Returns:
        X (np.ndarray): Features of shape (samples*classes, 2)
        y (np.ndarray): Labels of shape (samples*classes,)
    """
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype='uint8')

    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number

    return X, y


# Generate dataset
X, y = generate_spiral_data(samples=1000, classes=3)

# Imports from custom modules
from layers.dense import LayerDense
from layers.dropout import Layer_Dropout
from layers.activations.ReLU import ActivationReLU
from layers.activations.softmax_crossentropy import ActivationSoftmaxLossCategoricalCrossentropy
from losses.CategoricalCrossEntropy import LossCategoricalCrossentropy
from optimizers.Adam import Optimizer_Adam

# Define model layers
dense1 = LayerDense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = ActivationReLU()
dropout1 = Layer_Dropout(0.1)
dense2 = LayerDense(64, 3)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.07, decay=5e-5)

# Training loop
for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dropout1.forward(activation1.outputs)
    dense2.forward(dropout1.outputs)
    data_loss = loss_activation.forward(dense2.outputs, y)

    # Regularization
    reg_loss = (
            loss_activation.loss.regularization_loss(dense1) +
            loss_activation.loss.regularization_loss(dense2)
    )
    loss = data_loss + reg_loss

    # Accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Logging
    if not epoch % 100:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f} "
              f"(data_loss: {data_loss:.3f}, reg_loss: {reg_loss:.3f}), "
              f"lr: {optimizer.current_learning_rate}")

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validation
X_test, y_test = generate_spiral_data(samples=1000, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.outputs)
dense2.forward(activation1.outputs)
val_loss = loss_activation.forward(dense2.outputs, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
val_accuracy = np.mean(predictions == y_test)

print(f"validation, acc: {val_accuracy:.3f}, loss: {val_loss:.3f}")
