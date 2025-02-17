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
# Example Usage: MNIST classification
# ----------------------------------------------
# Details: We use the Flexible FFN Template to classify the MNIST dataset. See the Regression
# for a more thorough explanation of how to use the Flexible_FFN_Template. Results after training
# for 20 epochs with various regularization techniques can be found in MNIST_Results folder on repo


def evaluate_and_record_stats(model, test_dataloader, train_losses, val_losses, output_dir,
                              filename="performance_stats.txt"):
    """
    Runs inference on the test set, computes performance metrics, and writes them to a text file.

    Parameters:
        model (nn.Module): The trained model.
        test_dataloader (DataLoader): DataLoader for the test set (about 2000 MNIST examples).
        train_losses (list): List of training losses from your training routine.
        val_losses (list): List of validation losses from your training routine.
        output_dir (str): Path to the folder where the text file will be saved.
        filename (str): Name of the text file (default is "performance_stats.txt").
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
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
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
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w") as f:
        f.write(stats_text)

    print(f"Performance stats saved to: {file_path}")


if __name__ == '__main__':

    # Define the transformation for the images:
    # 1. Convert the image to a tensor.
    # 2. Normalize using MNIST's mean and std.
    # 3. Flatten the 28x28 image into a 784-dimensional vector.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Download and load the full MNIST training dataset.
    # No target_transform is provided, so labels remain as integers.
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Split the full training dataset into training (80%) and validation (20%) sets.
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create DataLoaders for the training and validation sets.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # (Optional) Load the test dataset similarly.
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = Flexible_FFN_Template.FlexibleMLP(2, [256, 128], 784, 10,0.2,True)

    model.make_autograd_diagram('example_MNIST_dropout_norm')

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), 0.01)

    train_losses, val_losses = Flexible_FFN_Template.train(model, criterion, optimizer, train_loader,
                                     val_loader, epochs=20,
                                     l1_reg=0, l2_reg=0, early_stopping_patience=0)

    output_folder = 'MNIST_Results'
    os.makedirs(output_folder, exist_ok=True)
    evaluate_and_record_stats(model, test_loader, train_losses, val_losses, output_folder)


