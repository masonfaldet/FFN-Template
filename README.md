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
The examples below will train in about a minute or less, thus the corresponding results posted are by no means optimized. 
- **One-Dimensional Regression Task:**
  A ready to run simple regression task is implemented in Example_Regression_Task.py. This example has detailed comments outlining the options avalible to the user. Results from this simple experiment can be found in Example_Regression_Results folder.
- **MNIST Classification Probem:**
  A ready to run instance of Flexible_FFN_Template to classify the MNIST dataset can be found in Example_MNIST_Classification.py. This implementation has less detailed comments, so I recommend looking at the one-dimensional regression example first. Results from this MNIST run can be found in the MNIST_Result folder.
  

## Installation

This project requires Python 3.7+ and PyTorch (tested with PyTorch 1.8+). You can install the necessary dependencies using pip:

```bash
pip install torch numpy matplotlib
