import torch.nn as nn
import torch

def train_model(model: nn.Module, train_loader, val_loader, criterion, optimizer, epochs, device):
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

        total_val_loss, val_correct, val_total = evaluate_model(model, val_loader, criterion, device)
        val_acc = val_correct/val_total
        val_loss = total_val_loss / len(val_loader)
        history["loss"].append(val_loss)
        history["acc"].append(val_acc)
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Train Acc: {correct/total}. Train Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"Val Acc: {val_acc}. Val Loss: {val_loss}")

    return model, history

def evaluate_model(model: nn.Module, val_loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0.0; correct = 0; total = 0;
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y).item()
            total_loss += loss
            correct += (pred.argmax(dim = 1) == y).sum().item()
            total += y.size(dim = 0)

    return total_loss, correct, total