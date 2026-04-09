import torch.nn as nn
import numpy as np
import torch

from src.utils import compute_f1_score_per_class

def train_model(model: nn.Module, train_loader, val_loader, criterion, optimizer, epochs, class_to_idx, device, verbose = True):
    model.to(device)

    history = {
        "loss": [],
        "acc": []
    }
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0; total = 0;
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            epoch_loss += loss.item()
            correct += (pred.argmax(dim = 1) == y).sum().item()
            total += y.size(dim = 0)
            optimizer.step()

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        total_val_loss, val_correct, val_total, confusion_matrix = evaluate_model(model, val_loader, criterion, class_to_idx, device)
        val_loss = total_val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history["loss"].append(val_loss)
        history["acc"].append(val_acc)
        if verbose:
            report_epoch_summary(epoch, epochs, train_loss, train_acc, val_loss, val_acc)
            report_metrics(confusion_matrix, class_to_idx)

    return model, history, confusion_matrix

def evaluate_model(model: nn.Module, val_loader, criterion, class_to_idx, device):
    num_classes = len(class_to_idx)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int) # rows correspond to the true value, columns correspond to the predicted value
    model.to(device)
    model.eval()
    total_loss = 0.0; correct = 0; total = 0;
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y).item()
            total_loss += loss
            pred_labels = pred.argmax(dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(dim = 0)
            for true_label, pred_label in zip(y, pred_labels):
                confusion_matrix[true_label.item(), pred_label.item()] += 1

    return total_loss, correct, total, confusion_matrix

def report_epoch_summary(epoch: int, epochs: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float):
    print(f"\nEpoch {epoch + 1}/{epochs}:")
    print(f"Train Acc: {train_acc:.4f}. Train Loss: {train_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}. Val Loss: {val_loss:.4f}")

def report_metrics(confusion_matrix, class_to_idx):
    f1_scores = compute_f1_score_per_class(confusion_matrix, class_to_idx)
    for class_name, f1_score in f1_scores:
        print(f"F1-score for class : {class_name} - {f1_score}")

    sorted_scores = sorted(f1_scores, key = lambda x: x[1])
    weakest_three = sorted_scores[:3]
    print(f"The weakest three classes:")
    i = 1;
    for class_name, f1_score in weakest_three:
        print(f"{i}. {class_name} - {f1_score}")
        i += 1