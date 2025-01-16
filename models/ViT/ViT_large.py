import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
import timm
from google.colab import drive

drive.mount('/content/drive')
!unzip "/content/drive/My Drive/mp_dataset.zip" -d "/content/data"

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
train_dataset = datasets.ImageFolder('/content/data/mp_dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder('/content/data/mp_dataset/val', transform=val_transform)

# Define model setup function
def get_model():
    model = timm.create_model('vit_large_patch16_224', pretrained=True)  # Load a pre-trained ViT
    # Adapt the classifier to binary classification
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 1)  # Replace the pre-trained head with a new one for binary classification
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

        # Define criterion for binary classification
        criterion = nn.BCEWithLogitsLoss()

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

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Execute the search
best_hyperparameters, best_metrics = systematic_search(combinations, device)
print("Best Hyperparameters:", best_hyperparameters)