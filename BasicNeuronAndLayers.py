import numpy as np

#  Single Neuron
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

# Manual implementation
neuron_output = 0
for i, w in zip(inputs, weights):
    neuron_output += i * w
neuron_output += bias
print("Single Neuron Output (Manual):", neuron_output)

# NumPy implementation
neuron_output = np.dot(np.array(inputs), np.array(weights)) + bias
print("Single Neuron Output (NumPy):", neuron_output)


#  One Layer of Neurons
inputs = np.random.random(4)
weights = np.random.random((4, 3))  # 4 inputs -> 3 neurons
bias = np.random.random()
neuron_output = np.dot(inputs, weights) + bias

print("\n--- Single Layer ---")
print("Inputs:\n", inputs)
print("Weights:\n", weights)
print("Bias:\n", bias)
print("Output:\n", neuron_output)


#  Stacked Layers
inputs = np.random.random(4)  # 4 input features
weights_1 = np.random.random((4, 3))  # Layer 1: 4 inputs -> 3 outputs
biases_1 = np.random.random(3)       # One bias per neuron in layer 1

weights_2 = np.random.random((3, 3))  # Layer 2: 3 inputs -> 3 outputs
biases_2 = np.random.random(3)        # One bias per neuron in layer 2

layer_1_output = np.dot(inputs, weights_1) + biases_1
layer_2_output = np.dot(layer_1_output, weights_2) + biases_2

print("\n--- Two-Layer Network ---")
print("Inputs:\n", inputs)
print("Layer 1 Weights:\n", weights_1)
print("Layer 1 Biases:\n", biases_1)
print("Output of Layer 1:\n", layer_1_output)

print("\nLayer 2 Weights:\n", weights_2)
print("Layer 2 Biases:\n", biases_2)
print("Output of Layer 2:\n", layer_2_output)

