import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from Flexible_FFN_Template import FlexibleMLP, train
from torch.utils.data import TensorDataset, DataLoader

# --------------------------------
# Example Usage
# --------------------------------
# Task: Learn f(x) = sin(2 pi x) + sin(7 pi x) - cos(5 pi x) + noise.
# Details: No regularization techniques used. 300 training examples, 200 validation example, 1000 epochs
# Results: Performance statistics and plots are saved to Regression_Results folder in repo.

if __name__ == '__main__':
    # ------------------------------
    # Data Generation and Preparation
    # ------------------------------
    np.random.seed(42)
    x = np.linspace(-2, 2, 500).reshape(-1, 1)
    y = np.sin(2 * np.pi * x) + np.sin(7*np.pi * x)- np.cos(5 * np.pi * x)+ 0.1 * np.random.randn(500, 1)

    x_all = torch.tensor(x, dtype=torch.float32)
    y_all = torch.tensor(y, dtype=torch.float32)

    # Shuffle the data before splitting. This ensures validation is randomly distributed across training data
    torch.manual_seed(42)
    indices = torch.randperm(x_all.size(0))
    x_all = x_all[indices]
    y_all = y_all[indices]

    # Split into training and validation sets (60% training, 40% validation)
    val_split = int(0.6 * len(x_all))
    x_train = x_all[:val_split]
    y_train = y_all[:val_split]
    x_val = x_all[val_split:]
    y_val = y_all[val_split:]

    # Use DataLoader to enable mini-batching (set batch_size = len(x_train) for full batch training)
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_dataset = TensorDataset(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)


    # ------------------------------
    # Model, Loss Function, and Optimizer Setup
    # ------------------------------

    # Create FlexibleMLP instance.
    # Provide number of input and output features in addition to depth and width of network
    # Optional: dropout regularization (dropout = 0 for no dropout)
    # Optional: Layer normalization regularization with learned gain and bias
    model = FlexibleMLP(depth=2, widths=[32, 32], input_size=1, output_size=1,
                        dropout=0, layer_norm=False)

    # Main loss function:
    # For regression:
    #   nn.MSELoss() <- Mean squared error.
    #   nn.L1Loss()  <- Mean absolute error, sensitive to outliers
    #   nn.SmoothL1Loss() <- Huber Loss, sensitive to outliers
    # For Classification:
    #   nn.CrossEntropyLoss() <- Combines nn.LogSoftmax() and nn.NLLLoss()
    #   nn.NLLLoss() <- Negative log likely hood
    #   nn.BCEWithLogitsLoss() <- Combines a sigmoid layer with binary
    #                             cross-entropy loss, for binary classification
    # Other:
    #   nn.KLDivLoss() <- Kullback-Leibler divergence loss, used for comparing two
    #                     probability distributions
    #   Implement your own by subclassing nn.Module
    criterion = nn.MSELoss()

    # Set optimization algorithm:
    #   optim.Adam <- Adaptive Learning rate for each parameter. Estimates first and second order
    #                moment with a weighted average and incorporates a correction for first order
    #                moment bias
    #   optim.SGD <- Supports momentum and Nesterov momentum (the latter computes momentum
    #                after taking a step).
    #   optim.RMSprop <- Adaptive learning rate for each parameter. Learning rate for a parameter
    #                    is inversely proportional to an exponentially weighted moving average of
    #                    historical partial derivatives w.r.t. the parameter.
    #   optim.Adagrad <- Adaptive learning rate for each parameter. Learning rate for parameter
    #                    is inversely proportional to the sqrt of the sum of historical squared
    #                    partial derivatives w.r.t the parameter.
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # ------------------------------
    # Train the Model and Collect Loss Vectors
    # ------------------------------

    # Train instance of FlexibleMLP
    # Provide model, main loss function, optimization algorithm, and training data
    # (Optional) Validation set (required for early stopping regularization)
    # (Optional) L1 and/or L2 weight decay regularization.
    # (Optional) Early stopping regularization (set early_stopping_patience = 0 to disable)
    train_losses, val_losses = train(model, criterion, optimizer, train_dataloader,
                                     val_dataloader, epochs=10000,
                                     l1_reg=0, l2_reg=0, early_stopping_patience=0)

    # ------------------------------
    # Evaluation and Visualization (Only make this plot if regression task learns a function from R -> R)
    # ------------------------------

    # Define the output folder path (change as needed)
    output_folder = 'FF_1DRegression_No_Reg'
    os.makedirs(output_folder, exist_ok=True)

    # Resort data to make smooth plots.
    x_train_sorted, train_sort_indices = torch.sort(x_train, dim=0)
    y_train_sorted = y_train[train_sort_indices.squeeze()]

    x_val_sorted, val_sort_indices = torch.sort(x_val, dim=0)
    y_val_sorted = y_val[val_sort_indices.squeeze()]

    # Pass data through trained network. no_grad() prevents PyTorch from tracking gradients since
    # this is unnecessary for model evaluation/inference
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode (disables dropout)
        y_pred_val = model(x_val_sorted).numpy()
        y_pred_train = model(x_train_sorted).numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(x_train_sorted.numpy(), y_train_sorted.numpy(), label='Ground Truth', color='blue')
    plt.plot(x_train_sorted.numpy(), y_pred_train, label='Model Prediction', color='red', linewidth=2)
    plt.title('Prediction vs. Ground Truth on Training Set')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'train_plot.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(x_val_sorted.numpy(), y_val_sorted.numpy(), label='Ground Truth', color='blue')
    plt.plot(x_val_sorted.numpy(), y_pred_val, label='Model Prediction', color='red', linewidth=2)
    plt.title('Prediction vs. Ground Truth on Validation Set')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'val_plot.png'))
    plt.close()

    # ------------------------------
    # Plot the Training and Validation Losses
    # ------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if any(val is not None for val in val_losses):
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss' )
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'loss_plot.png'))
    plt.close()

    # ------------------------------
    # Compute Summary Statistics
    # ------------------------------
    total_epochs = len(train_losses)
    final_train_loss = train_losses[-1]

    if any(val is not None for val in val_losses):
        # Filter out None values (if any)
        valid_val_losses = [val for val in val_losses if val is not None]
        final_val_loss = valid_val_losses[-1]
        best_val_loss = min(valid_val_losses)

        # Calculate errors on the validation set
        # Assuming y_val_sorted and y_pred_val are numpy arrays
        errors = y_val_sorted.numpy() - y_pred_val
        # Expected error: mean absolute error
        expected_error = np.mean(np.abs(errors))
        # Error variance: variance of the differences
        error_variance = np.var(errors)
    else:
        final_val_loss = None
        best_val_loss = None
        expected_error = None
        error_variance = None

    # ------------------------------
    # Write Performance Statistics to a Text File
    # ------------------------------
    stats_file_path = os.path.join(output_folder, 'performance_stats.txt')
    with open(stats_file_path, 'w') as f:
        f.write("Performance Statistics\n")
        f.write("=======================\n")
        f.write(f"Total epochs: {total_epochs}\n")
        f.write(f"Final Training Loss: {final_train_loss:.6f}\n")
        if final_val_loss is not None:
            f.write(f"Final Validation Loss: {final_val_loss:.6f}\n")
            f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
            f.write(f"Expected Error on Validation Set (Mean Absolute Error): {expected_error:.6f}\n")
            f.write(f"Error Variance on Validation Set: {error_variance:.6f}\n")
        else:
            f.write("No validation loss available.\n")
