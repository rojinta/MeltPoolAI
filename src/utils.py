import matplotlib.pyplot as plt

def plot_metrics(train_losses, train_accuracies, val_accuracies, hyperparameters=None, save_path=None):
    """ 
    Plots training loss, training accuracy, and validation accuracy. Saves the plot if a path is provided.

    Args:
        train_losses (list): Training losses per epoch
        train_accuracies (list): Training accuracies per epoch
        val_accuracies (list): Validation accuracies per epoch
        hyperparameters (dict, optional): Dictionary of hyperparameters to display in the title
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    title = "Training and Validation Metrics"
    if hyperparameters:
        title += f" (LR={hyperparameters['lr']}, Batch={hyperparameters['batch_size']}, Model={hyperparameters['model']})"
    plt.suptitle(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
