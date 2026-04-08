from src.model import ConvolutionModel
from src.utils import (get_device)
from src.dataset import (get_data_loaders, get_datasets)
from src.train import (train_model, evaluate_model)
import torch.optim as optim
import torch.nn as nn


def main():
    device = get_device()
    train_loader, val_loader, test_loader, num_classes = get_data_loaders()
    criterion = nn.CrossEntropyLoss()
    model = ConvolutionModel(in_channels=3, out_channels=3, num_classes=num_classes)
    optimizer = optim.SGD(model.parameters(), lr = 0.001)
    print("\nQ1:")
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device = device)


if __name__ == "__main__":
    main()