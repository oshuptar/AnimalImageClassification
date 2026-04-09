import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, Subset, DataLoader
from torchvision.datasets import ImageFolder

def transform_images(dir_name = "data/train"):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5])
    ])
    return load_images(dir_name=dir_name, transform=transform)

def load_images(dir_name, transform = None):
    return ImageFolder(dir_name, transform = transform)

def get_datasets() -> tuple[Dataset, Dataset, Dataset, dict[str, int]]:
    full_train_dataset = transform_images()
    test_dataset = load_images(dir_name="data/test")

    class_to_idx = full_train_dataset.class_to_idx
    indeces = list(range(len(full_train_dataset)))
    targets = full_train_dataset.targets
    val_ratio = 0.2
    train_indices, val_indices = train_test_split(indeces,
                    stratify=targets,
                    test_size=val_ratio,
                    random_state=67)
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    return train_dataset, val_dataset, test_dataset, class_to_idx

def get_data_loaders(batch_size = 64) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    train_dataset, val_dataset, test_dataset, class_to_idx = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, class_to_idx
    
    
