import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import Flexible_FFN_Template

# ----------------------------------------------
# Example Usage: CIFAR 10 classification
# ----------------------------------------------
# Details: We use the Flexible FFN Template to classify the CIFAR dataset. See the Regression
# for a more thorough explanation of how to use the Flexible_FFN_Template. Results after training
# for 50 epochs with various regularization techniques can be found in MNIST_Results folder on repo


def evaluate_and_record_stats(model, test_dataloader, train_losses, val_losses, output_dir, regularization):

    """
    Runs inference on the test set, computes performance metrics, and writes them to a text file.

    Parameters:
        model (nn.Module): The trained model.
        test_dataloader (DataLoader): DataLoader for the test set (about 2000 MNIST examples).
        train_losses (list): List of training losses from your training routine.
        val_losses (list): List of validation losses from your training routine.
        output_dir (str): Path to the folder where the text file will be saved.
        regularization (str): Regularization techniques used while training the model.
    """

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
    plt.savefig(os.path.join(output_dir, regularization + '_loss_plot.png'))
    plt.close()

    # ---------------------------------
    # Compute and log statistics
    # ---------------------------------
    model.eval()  # Switch model to evaluation mode

    all_preds = []
    all_labels = []
    all_confidences = []

    # Inference loop: accumulate predictions, labels, and confidence values.
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            logits = model(batch_x)
            # Compute softmax probabilities (assuming model outputs logits)
            probs = F.softmax(logits, dim=1)
            # For each sample, get the predicted class and its associated confidence (max probability)
            confidences, preds = torch.max(probs, dim=1)
            all_preds.append(preds)
            all_labels.append(batch_y)
            all_confidences.append(confidences)

    # Concatenate batch outputs
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_confidences = torch.cat(all_confidences)

    # Compute performance metrics
    total_samples = all_labels.size(0)
    correct = (all_preds == all_labels).sum().item()
    accuracy = correct / total_samples
    error_rate = 1 - accuracy
    avg_confidence = all_confidences.mean().item()

    # Retrieve final training and validation losses (if available)
    final_train_loss = train_losses[-1] if train_losses else None
    final_val_loss = val_losses[-1] if val_losses and val_losses[-1] is not None else None

    # Create a string with all performance statistics
    stats_text = (
        "Performance Statistics:\n"
        "-----------------------\n"
        f"Total Test Samples: {total_samples}\n"
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Error Rate: {error_rate * 100:.2f}%\n"
        f"Average Confidence: {avg_confidence * 100:.2f}%\n\n"
        f"Final Training Loss: {final_train_loss}\n"
        f"Final Validation Loss: {final_val_loss}\n"
    )

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the performance stats to the text file
    file_path = os.path.join(output_dir, regularization + '_performance_stats.txt')
    with open(file_path, "w") as f:
        f.write(stats_text)

    print(f"Performance stats saved to: {file_path}")


if __name__ == '__main__':

    # Define the transformation for the images:
    # 1. Convert the image to a tensor.
    # 2. Normalize using CIFAR-10's mean and std.
    # 3. Flatten the 32x32x3 image into a 3072-dimensional vector.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Download and load the full CIFAR-10 training dataset.
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Split the full training dataset into training (80%) and validation (20%) sets.
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create DataLoaders for the training and validation sets.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # (Optional) Load the CIFAR-10 test dataset similarly.
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Instantiate the model.
    # For CIFAR-10, the input vector size is 3*32*32 = 3072, and there are 10 classes.
    model = Flexible_FFN_Template.FlexibleMLP(depth=2, widths=[1024, 512], input_size= 3072,
                                              output_size= 10, dropout=0.1, layer_norm=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.01)

    train_losses, val_losses = Flexible_FFN_Template.train(
        model, criterion, optimizer, train_loader, val_loader,
        epochs=50, l1_reg=0, l2_reg=1e-3, early_stopping_patience=0
    )

    output_folder = 'CIFAR10_Results'
    regularization = 'Dropout_Norm_L2_Decay'
    os.makedirs(output_folder, exist_ok=True)
    evaluate_and_record_stats(model, test_loader, train_losses, val_losses, output_folder, regularization)
