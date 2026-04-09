from src.model import ConvolutionModel
from src.utils import (get_device, compute_macros_for_class, f1_score_per_class)
from src.dataset import (get_data_loaders, get_datasets)
from src.train import (train_model, evaluate_model)
from src.plotting import (plot_loss_and_accuracy_over_epochs)
import torch.optim as optim
import torch.nn as nn


def main():
    device = get_device()
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders()
    num_classes = len(class_to_idx)
    criterion = nn.CrossEntropyLoss()
    model = ConvolutionModel(in_channels=3, out_channels=32, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    print("\nQ5:")
    model, history, confusion_matrix = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, device = device, num_classes=num_classes)
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(name, param.grad.norm().item())
    print(f"Best accuracy over epochs: {max(history['acc'])}")
    plot_loss_and_accuracy_over_epochs(history)
    print ("\nQ6")

    f1_scores = f1_score_per_class(confusion_matrix, class_to_idx)
    for class_name, f1_score in f1_scores:
        print(f"F1-score for class : {class_name} - {f1_score}")

    sorted_scores = sorted(f1_scores, key = lambda x: x[1])
    weakest_three = sorted_scores[:3]
    print(f"The weakest three classes:")
    i = 1;
    for class_name, f1_score in weakest_three:
        print(f"{i}. {class_name} - {f1_score}")
        i += 1


if __name__ == "__main__":
    main()