# Implementing Neural Networks from Scratch and using the same for digit recognition

## Overview

This repository contains a custom implementation of a neural network framework built from scratch in Python. The framework includes essential components like linear layers, activation functions, loss functions, and an optimizer. The framework is demonstrated using the MNIST dataset, a standard dataset for handwritten digit recognition.

## Components

### 1. Linear Layer
- **Purpose:** Implements a fully connected layer, which performs a linear transformation of the input.
- **Methods:** 
  - `forward(x)`: Computes the output of the layer.
  - `backward(grad_output)`: Computes gradients with respect to inputs, weights, and biases.

### 2. Activation Functions
- **ReLU (Rectified Linear Unit):** Applies the ReLU function, which introduces non-linearity by outputting the input directly if positive, otherwise zero.
- **Sigmoid:** Applies the sigmoid function, which squashes values to the range [0, 1].
- **Tanh (Hyperbolic Tangent):** Applies the tanh function, which outputs values in the range [-1, 1].
- **Softmax:** Converts logits (raw prediction scores) into probabilities that sum to one.

### 3. Loss Functions
- **CrossEntropyLoss:** Computes the cross-entropy loss for classification tasks. It measures the difference between the predicted probabilities and the true labels.
- **MSELoss (Mean Squared Error):** Computes the mean squared error loss for regression tasks.

### 4. Optimizer
- **SGD (Stochastic Gradient Descent):** Optimizer that updates weights and biases based on gradients computed during backpropagation.
