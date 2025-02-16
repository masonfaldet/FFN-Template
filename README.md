# FFN Template with Regularization Options

This repository contains an annotated template for a feed-forward neural network (FFN) implemented in PyTorch. The goal of this project is to introduce fundamental PyTorch syntax and functionality to beginners. The template allows users to easily toggle various regularization techniques for comparison, including dropout, layer normalization, L1/L2 weight decay, and early stopping.

In its current configuration, the model is set up for regression tasks (e.g., learning the sine function). However, you can readily modify the output activation (for example, by adding a softmax unit) or simply use `torch.nn.CrossEntropyLoss()` to tackle classification tasks. Additionally, you can extend the architecture by adding convolution or residual layers if desired.

## Features

- **Variable Input/Output Dimensions:**  
  Easily adjust the number of input and output features.
- **Customizable Architecture:**  
  Change the depth (number of hidden layers) and width (number of neurons per layer) to suit your needs.
- **Flexible Training:**  
  Configure training duration and choose among different loss functions and optimization algorithms.
- **Regularization Techniques:**  
  Toggle the following options on/off to compare their effects:
  - Dropout
  - Layer Normalization
  - L1 and/or L2 Weight Decay
  - Early Stopping

## Examples

The following examples demonstrate how to run experiments quickly (in about a minute or less). Consequently, the posted results are not optimized.

- **One-Dimensional Regression Task:**  
  The `Example_Regression_Task.py` uses the template in  `Flexible_FFN_Template.py` to solve a simple regression task with detailed comments that explain all available options. You can find the results in the `Regression_Results` folder.

- **MNIST Classification Problem:**  
  The `Example_MNIST_Classification.py` script uses the template in  `Flexible_FFN_Template.py` to classify the MNIST dataset. Because the comments are less detailed in this example, it is recommended to review the regression task example first. Results from this experiment are stored in the `MNIST_Results` folder.

  

## Installation

This project requires Python 3.7+ and PyTorch (tested with PyTorch 1.8+). You can install the necessary dependencies using pip:

```bash
pip install torch numpy matplotlib
