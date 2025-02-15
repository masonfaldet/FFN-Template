import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------
# Define a Flexible Feedforward Neural Network with Optional Layer Normalization
# ------------------------------
class FlexibleMLP(nn.Module):
    def __init__(self, depth, widths, input_size=1, output_size=1, dropout=0.0, layer_norm=False):
        """
        Parameters:
            depth (int): Number of hidden layers.
            widths (list): List specifying the number of neurons in each hidden layer.
                           Its length must equal 'depth'.
            input_size (int): Dimensionality of input features.
            output_size (int): Dimensionality of the output.
            dropout (float): Dropout probability (if > 0, dropout layers are added after activations).
            layer_norm (bool): If True, adds Layer Normalization after each linear layer (before activation).
        """

        # Use super to inherit methods from nn.Module
        super(FlexibleMLP, self).__init__()

        if len(widths) != depth:
            raise ValueError("Length of 'widths' must equal the 'depth' parameter.")


        layers = []
        # --- First Hidden Layer ---
        layers.append(nn.Linear(input_size, widths[0]))
        if layer_norm:
            layers.append(nn.LayerNorm(widths[0]))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # --- Additional Hidden Layers ---
        for i in range(1, depth):
            layers.append(nn.Linear(widths[i - 1], widths[i]))
            if layer_norm:
                layers.append(nn.LayerNorm(widths[i]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # --- Output Layer ---
        layers.append(nn.Linear(widths[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ------------------------------
# Define a Training Function with Early Stopping and Loss Tracking
# ------------------------------
def train(model, criterion, optimizer, dataloader,
          x_val=None, y_val=None, epochs=500, l1_reg=0.0, l2_reg=0.0,
          early_stopping_patience=0):
    """
    Trains the model with options for L1/L2 regularization and early stopping.
    Returns vectors containing training losses and validation losses (errors) for each epoch.

    Parameters:
        model (nn.Module): The neural network model.
        criterion (loss function): e.g., nn.MSELoss()
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        dataloader (DataLoader): Loader for mini-batches.
        x_val (Tensor): Validation inputs (optional for early stopping).
        y_val (Tensor): Validation targets.
        epochs (int): Maximum number of epochs.
        l1_reg (float): L1 regularization strength.
        l2_reg (float): L2 regularization strength.
        early_stopping_patience (int): Number of epochs to wait for improvement in validation loss before stopping.
                                       Set to 0 to disable early stopping.

    Returns:
        train_losses (list): Average training loss per epoch.
        val_losses (list): Validation loss per epoch (if validation data is provided, else an empty list).
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()  # Ensure model is in training mode (enable dropout)
        epoch_loss = 0.0

        # Process the (full) batch
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()  # Reset gradients

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # Apply L1 regularization if specified
            if l1_reg > 0:
                l1_loss = 0.0
                for param in model.parameters():
                    l1_loss += torch.sum(torch.abs(param))
                loss += l1_reg * l1_loss

            # Apply L2 regularization if specified
            if l2_reg > 0:
                l2_loss = 0.0
                for param in model.parameters():
                    l2_loss += torch.sum(param ** 2)
                loss += l2_reg * l2_loss

            # loss.backward() uses PyTorch's automatic differentiation system autograd to compute
            # the gradient of the entire loss. Autograd automatically includes additional
            # regularization terms like L1/L2 loss in addition to the primary loss provided by criterion
            loss.backward()
            optimizer.step()  # Update parameters
            epoch_loss += loss.item()

        # Average loss for this epoch
        epoch_loss /= len(dataloader)
        if epoch % 50 == 0:
            train_losses.append(epoch_loss)

        # Check validation loss if validation data is provided
        if x_val is not None and y_val is not None:
            model.eval()  # Switch to evaluation mode (disables dropout)
            with torch.no_grad():
                val_predictions = model(x_val)
                val_loss = criterion(val_predictions, y_val).item()
            if epoch % 50 == 0:
                val_losses.append(val_loss)

            # Early stopping logic based on validation loss
            if early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    break
        else:
            # If no validation set is provided, just log training loss
            val_losses.append(None)

    return train_losses, val_losses

if __name__ == '__main__':

    #       Example Usage
    #-------------------------------
    # ------------------------------
    # Data Generation and Preparation
    # ------------------------------
    np.random.seed(42)
    x = np.linspace(-2, 2, 500).reshape(-1, 1)
    y = np.sin(np.pi * x) + 0.1 * np.random.randn(500, 1)

    x_all = torch.tensor(x, dtype=torch.float32)
    y_all = torch.tensor(y, dtype=torch.float32)

    # Shuffle the data before splitting. This ensures validation is randomly distributed across training data
    torch.manual_seed(42)
    indices = torch.randperm(x_all.size(0))
    x_all = x_all[indices]
    y_all = y_all[indices]

    # Split into training and validation sets (80% training, 20% validation)
    val_split = int(0.8 * len(x_all))
    x_train = x_all[:val_split]
    y_train = y_all[:val_split]
    x_val = x_all[val_split:]
    y_val = y_all[val_split:]

    # Use DataLoader to enable mini-batching (set batch_size = len(x_train) for full batch training)
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)


    # ------------------------------
    # Model, Loss Function, and Optimizer Setup
    # ------------------------------

    # Create FlexibleMLP instance.
    # Provide number of input and output features in addition to depth and width of network
    # Optional: dropout regularization (dropout = 0 for no dropout)
    # Optional: Layer normalization regularization with learned gain and bias
    model = FlexibleMLP(depth=3, widths=[50,50,20], input_size=1, output_size=1,
                        dropout=0.2, layer_norm=False)

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
    train_losses, val_losses = train(model, criterion, optimizer, dataloader,
                                     x_val=x_val, y_val=y_val, epochs=10000,
                                     l1_reg=0, l2_reg=1e-3, early_stopping_patience=0)

    # ------------------------------
    # Evaluation and Visualization (Only make this plot if regression task learns a function from R -> R)
    # ------------------------------

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
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(x_val_sorted.numpy(), y_val_sorted.numpy(), label='Ground Truth', color='blue')
    plt.plot(x_val_sorted.numpy(), y_pred_val, label='Model Prediction', color='red', linewidth=2)
    plt.title('Prediction vs. Ground Truth on Validation Set')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # ------------------------------
    # Plot the Training and Validation Losses
    # ------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if any(val is not None for val in val_losses):
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.show()

