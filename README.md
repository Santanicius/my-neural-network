# Multilayer Perceptron Implementation in Python from Scratch

![MLP](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/500px-Artificial_neural_network.svg.png)

This repository contains a Python implementation of a Multilayer Perceptron (MLP) from scratch. The Multilayer Perceptron is a type of artificial neural network with multiple layers of interconnected neurons, commonly used for various machine learning tasks like classification and regression.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

## Overview

A Multilayer Perceptron consists of an input layer, one or more hidden layers, and an output layer. Each layer contains multiple neurons (also known as nodes or units), and each neuron is connected to every neuron in the previous and subsequent layers. The network uses a combination of weighted inputs, activation functions, and backpropagation to learn and make predictions.

This implementation provides a basic framework for creating and training a Multilayer Perceptron for various tasks. It includes support for customizable network architecture, activation functions (e.g., sigmoid, ReLU), and training using gradient descent.

## Features

- Implementation of a Multilayer Perceptron from scratch in Python.
- Customizable network architecture (number of layers, number of neurons per layer).
- Support for different activation functions (sigmoid, ReLU, etc.).
- Training using gradient descent with backpropagation.
- Easy-to-use interface for training and making predictions.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed.
- NumPy library installed (`pip install numpy`).
- Streamlit library installed (`pip install streamlit`).
- Pandas library installed (`pip install pandas`).

## Getting Started

To get started, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/MLP-from-scratch.git
   ```

2. Change to the project directory:

   ```bash
   cd MLP-from-scratch
   ```

3. Start streamlit server and use:
   
   ```bash
   streamlit run main.py
   ```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve this implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
