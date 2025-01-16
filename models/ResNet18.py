import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools

# Define transformations for training and validation data
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder('C:/Users/rojin/Desktop/Rojin/UPitt/Deep Learning/Project/Data/mp_dataset/train', transform=train_transform)
val_dataset = ImageFolder('C:/Users/rojin/Desktop/Rojin/UPitt/Deep Learning/Project/Data/mp_dataset/val', transform=val_transform)

def get_model():
    model = models.resnet18(pretrained=True)
    # Freeze all the layers before the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()  # Ensure the output is between 0 and 1
    )
    return model

# Define hyperparameters
learning_rates = [1e-3, 1e-5]
batch_sizes = [16, 64]
optimizers = ['Adam', 'SGD']

# Generate combinations of hyperparameters
combinations = list(itertools.product(learning_rates, batch_sizes, optimizers))

# Define training and validation function
def train_and_validate(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10):
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            predicted = outputs > 0.5
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                outputs = model(inputs)
                predicted = outputs > 0.5
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_accuracy:.2f}%, Val Acc {val_accuracy:.2f}%')

    return train_losses, train_accuracies, val_accuracies

# Define the systematic search function
def systematic_search(combinations, device):
    best_accuracy = 0
    best_params = {}
    best_metrics = {}

    for idx, (lr, batch_size, optimizer_name) in enumerate(combinations):
        model = get_model().to(device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        criterion = nn.BCELoss()

        print(f"Testing combination {idx+1}/{len(combinations)}: Optimizer={optimizer_name}, LR={lr}, Batch size={batch_size}")
        train_losses, train_accuracies, val_accuracies = train_and_validate(model, criterion, optimizer, train_loader, val_loader, device)
        
        val_accuracy = val_accuracies[-1]
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = {'lr': lr, 'batch_size': batch_size, 'optimizer': optimizer_name}
            best_metrics = {'train_losses': train_losses, 'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}
            print(f"New best parameters found: {best_params}, Accuracy: {best_accuracy}%")

    plot_metrics(best_metrics['train_losses'], best_metrics['train_accuracies'], best_metrics['val_accuracies'], best_params)
    return best_params

# Plot metrics
def plot_metrics(train_losses, train_accuracies, val_accuracies, hyperparameters):
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

    plt.suptitle(f"Best Hyperparameters: LR={hyperparameters['lr']}, Batch Size={hyperparameters['batch_size']}, Optimizer={hyperparameters['optimizer']}")
    plt.show()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Execute the search
best_hyperparameters = systematic_search(combinations, device)
print("Best Hyperparameters:", best_hyperparameters)

# Epoch data
epochs = list(range(1, 11))

# Training loss per epoch
train_loss = [0.4788, 0.4150, 0.3962, 0.4005, 0.3850, 0.4157, 0.3580, 0.3777, 0.3380, 0.3555]

# Training accuracy per epoch
train_acc = [75.81, 81.01, 83.28, 81.90, 83.20, 80.76, 83.36, 82.47, 85.80, 85.06]

# Validation accuracy per epoch
val_acc = [82.14, 68.18, 75.32, 67.21, 68.51, 74.35, 63.31, 72.73, 70.13, 72.40]

# Create a figure and a set of subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot training loss
axs[0].plot(epochs, train_loss, 'o-', color="r", label="Train Loss")
axs[0].set_title("Training Loss per Epoch")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].grid(True)

# Plot training accuracy
axs[1].plot(epochs, train_acc, 'o-', color="b", label="Train Accuracy")
axs[1].set_title("Training Accuracy per Epoch")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy (%)")
axs[1].legend()
axs[1].grid(True)

# Plot validation accuracy
axs[2].plot(epochs, val_acc, 'o-', color="g", label="Validation Accuracy")
axs[2].set_title("Validation Accuracy per Epoch")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Accuracy (%)")
axs[2].legend()
axs[2].grid(True)

# Adjust layout to prevent overlap
fig.tight_layout()

# Save the entire figure as a PDF in your current working directory
plt.savefig('resnet18_training_results.pdf') 

# Save the entire figure as a PNG file with higher resolution
plt.savefig('resnet18_training_results.png', dpi=300)

# Show plot
plt.show()