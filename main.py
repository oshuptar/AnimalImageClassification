from src.model import ConvolutionModel
from src.utils import (get_device, compute_f1_score_per_class,
                       get_batch_size, get_filter_size, get_learning_rate, get_weight_decay,
                       compute_f1_macro, save_test_results)
from src.dataset import (get_data_loaders, get_datasets)
from src.train import (train_model, evaluate_model, test_model)
from src.plotting import (plot_loss_and_accuracy_over_epochs, plot_confusion_matrix)
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
    model, history, _ = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, class_to_idx=class_to_idx, device = device)
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(name, param.grad.norm().item())
    print(f"Best accuracy over epochs: {max(history['acc'])}")
    plot_loss_and_accuracy_over_epochs(history)

    print("\nQ7- Hyperparameter tuning")
    best_configuration = grid_experiment(criterion,device)
    best_model = best_configuration["model"]

    print("\nQ8")
    _, _, _, confusion_matrix = evaluate_model(best_model, val_loader, criterion, class_to_idx, device)
    plot_confusion_matrix(confusion_matrix, class_to_idx)

    print("\nQ9")
    test_results = test_model(best_model, test_loader, device)
    save_test_results(folder_name = "testing",filename = "results", history=test_results, class_to_idx=class_to_idx)

def grid_experiment(criterion, device):
    experiments = []
    for filter_size in get_filter_size():
        for lr in get_learning_rate():
            for batch_size in get_batch_size():
                for weight_decay in get_weight_decay():
                    print("\n" + ("-" * 50))
                    print(f"\nfilter_size={filter_size}, lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}")
                    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(batch_size=batch_size)
                    model = ConvolutionModel(in_channels=3, out_channels=filter_size, num_classes=len(class_to_idx))
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    trained_model, _, confusion_matrix = train_model(model,
                                                                        train_loader,
                                                                        val_loader,
                                                                        criterion,
                                                                        optimizer,
                                                                        epochs = 3,
                                                                        class_to_idx = class_to_idx,
                                                                        device = device)
                    f1_scores = compute_f1_score_per_class(confusion_matrix, class_to_idx)
                    f1_macro = compute_f1_macro(f1_scores)
                    experiments.append({
                        "filter_size": filter_size,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                        "f1_macro": f1_macro,
                        "model": trained_model
                    })
    experiments.sort(key=lambda x: x["f1_macro"])
    print("\nBest configuration:")
    print(experiments[-1])

    return experiments[-1]

if __name__ == "__main__":
    main()