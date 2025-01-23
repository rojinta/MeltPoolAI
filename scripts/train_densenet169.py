import os
import torch
from torchvision import models
import torch.nn as nn
import itertools
from src.data_loader import get_dataloaders
from src.training import train_and_validate
from src.utils import plot_metrics
from src.systematic_search import systematic_search

DATA_DIR = "../data"
RESULTS_DIR = "../results"
BATCH_SIZES = [16, 64]
LEARNING_RATES = [1e-3, 1e-5]
OPTIMIZERS = ['Adam', 'SGD']
EPOCHS = 10

# Ensure results subfolders exist
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)

# Define the DenseNet169 model setup function
def get_densenet169():
    model = models.densenet169(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Log file setup
    log_file_path = os.path.join(RESULTS_DIR, "logs", "densenet169_training.log")

    # Generate hyperparameter combinations
    combinations = list(itertools.product(LEARNING_RATES, BATCH_SIZES, OPTIMIZERS))

    # Perform hyperparameter search
    best_params, best_metrics = systematic_search(
        combinations=combinations,
        data_dir=DATA_DIR,
        device=device,
        model_func=get_densenet169,
        criterion=torch.nn.BCELoss(),
        num_epochs=EPOCHS,
        log_file_path=log_file_path
    )

    # Save and log final results
    with open(log_file_path, "a") as log_file:
        log_file.write("\nBest Hyperparameters:\n")
        log_file.write(f"Learning Rate: {best_params['lr']}, Batch Size: {best_params['batch_size']}, Optimizer: {best_params['optimizer']}\n")
        log_file.write("\nBest Metrics:\n")
        log_file.write(f"Training Accuracy: {best_metrics['train_accuracy']:.2f}%\n")
        log_file.write(f"Validation Accuracy: {best_metrics['val_accuracy']:.2f}%\n")

    # Save plot of metrics
    plot_path = os.path.join(RESULTS_DIR, "plots", "densenet169_best_metrics.png")
    plot_metrics(best_metrics['train_losses'], best_metrics['train_accuracies'], best_metrics['val_accuracies'], {
        "lr": best_params['lr'],
        "batch_size": best_params['batch_size'],
        "model": "densenet169",
    }, save_path=plot_path)
    print(f"Metrics plot saved to {plot_path}")

if __name__ == "__main__":
    main()
