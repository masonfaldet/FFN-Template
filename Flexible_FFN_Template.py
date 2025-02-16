import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
        train_losses.append(epoch_loss)

        # Check validation loss if validation data is provided
        if x_val is not None and y_val is not None:
            model.eval()  # Switch to evaluation mode (disables dropout)
            with torch.no_grad():
                val_predictions = model(x_val)
                val_loss = criterion(val_predictions, y_val).item()
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



