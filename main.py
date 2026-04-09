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
    model = ConvolutionModel(in_channels=3, out_channels=32, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr = 0.003)
    print("\nQ5:")
    model, best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, device = device)
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(name, param.grad.norm().item())
    print(f"Best accuracy over epochs: {best_acc}")


if __name__ == "__main__":
    main()