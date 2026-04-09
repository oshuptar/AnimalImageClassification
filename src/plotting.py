import matplotlib.pyplot as plt
import os 

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

    root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    plot_dir = os.path.join(root_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, "val_acc_per_epochs.png")
    plt.savefig(output_path)
    print(f"The plot was saved to {output_path}")

