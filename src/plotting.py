import matplotlib.pyplot as plt
import os

from src.utils import row_normalise_confusion_matrix 

def plot_loss_and_accuracy_over_epochs(history):
    if len(history["loss"]) != len(history["acc"]):
        raise ValueError("The dimensions of loss and accuracy do not match")

    epochs = list(range(1, len(history["loss"]) + 1))
    loss = history["loss"]
    accuracy = history["acc"]

    plt.figure(figsize=(14,5))
    plt.plot(epochs, loss, label = "Validation loss")
    plt.plot(epochs, accuracy, label = "Validation accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.tight_layout()

    output_path = get_plot_path("val_acc_per_epochs.png")
    plt.savefig(output_path)
    print(f"The plot was saved to {output_path}")

def plot_confusion_matrix(confusion_matrix, class_to_idx):
    class_names = [class_name for class_name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    row_normalised_cm = row_normalise_confusion_matrix(confusion_matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(row_normalised_cm)
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    output_path = get_plot_path("confusion_matrix_heatmap.png")
    plt.savefig(output_path)
    print(f"The plot was saved to {output_path}")

def get_plot_path(filename):
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, filename)
    return output_path

