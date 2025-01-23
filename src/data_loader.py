import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size, augment=True):
    """ 
    Prepares training and validation DataLoaders with optional data augmentation.
    
    Args:
        data_dir (str): Path to the dataset directory (expects 'train' and 'val' subfolders)
        batch_size (int): Batch size for data loader
        augment (bool): Whether to apply data augmentation to training data

    Returns:
        train_loader (DataLoader): DataLoader for training set
        val_loader (DataLoader): DataLoader for validation set
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if augment else transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
