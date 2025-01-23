import torch
import torch.optim as optim
from src.training import train_and_validate
from src.data_loader import get_dataloaders

def systematic_search(combinations, data_dir, device, model_func, criterion, num_epochs, log_file_path):
    """ 
    Performs systematic search over hyperparameter combinations to find the best configuration.

    Args:
        combinations (list): List of tuples containing hyperparameter combinations (lr, batch_size, optimizer_name)
        data_dir (str): Path to the dataset directory
        device (torch.device): Device to use for training ('cuda' or 'cpu')
        model_func (callable): Function that returns a fresh instance of the model
        criterion: Loss function for training
        num_epochs (int): Number of epochs to train for each combination
        log_file_path (str): Path to the log file

    Returns:
        best_params (dict): Dictionary containing the best hyperparameters
        best_metrics (dict): Dictionary containing the training and validation metrics for the best configuration
    """
    best_accuracy = 0
    best_params = {}
    best_metrics = {}

    for idx, (lr, batch_size, optimizer_name) in enumerate(combinations):
        model = model_func().to(device)
        train_loader, val_loader = get_dataloaders(data_dir, batch_size)

        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        train_losses, train_accuracies, val_accuracies = train_and_validate(
            model, train_loader, val_loader, criterion, optimizer, device, num_epochs
        )

        val_accuracy = val_accuracies[-1]

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = {'lr': lr, 'batch_size': batch_size, 'optimizer': optimizer_name}
            best_metrics = {
                'train_accuracy': train_accuracies[-1],
                'val_accuracy': val_accuracy
            }

    # Log only the best parameters and metrics (excluding training loss)
    with open(log_file_path, "w") as log_file:
        log_file.write("Best Hyperparameters:\n")
        log_file.write(f"Learning Rate: {best_params['lr']}\n")
        log_file.write(f"Batch Size: {best_params['batch_size']}\n")
        log_file.write(f"Optimizer: {best_params['optimizer']}\n")
        log_file.write("\nBest Metrics:\n")
        log_file.write(f"Training Accuracy: {best_metrics['train_accuracy']:.2f}%\n")
        log_file.write(f"Validation Accuracy: {best_metrics['val_accuracy']:.2f}%\n")

    return best_params, best_metrics
