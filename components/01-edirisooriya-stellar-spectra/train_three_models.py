import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

from model_architectures import ResNet1D, StandardCNN1D, CNNBiLSTM1D

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_split(name):
    path = os.path.join(PROCESSED_DIR, f"{name}.npz")
    data = np.load(path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.long)
    return TensorDataset(X, y)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * X.size(0)
    return loss_sum / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            loss_sum += loss.item() * X.size(0)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    acc = accuracy_score(all_targets, all_preds)
    return loss_sum / len(loader.dataset), acc, all_targets, all_preds

def train_and_evaluate(model, train_loader, val_loader, test_loader, device, epochs=30, lr=1e-3):
    print(f"\n=======================================================")
    print(f"Training Model: {model.name}")
    print(f"=======================================================")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} - TrainLoss: {tr_loss:.4f} - ValLoss: {val_loss:.4f} - ValAcc: {val_acc:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save model weights
    save_path = os.path.join(MODELS_DIR, f"{model.name.lower().replace('-', '_')}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model weights to: {save_path}")

    # Final evaluation on untouched test set
    test_loss, test_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"--- {model.name} Test Results ---")
    print(f"Test Accuracy:          {test_acc:.4f}")
    print(f"Macro F1-Score:         {macro_f1:.4f}")
    print(f"Balanced Accuracy:      {bal_acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "model_name": model.name,
        "test_accuracy": float(test_acc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist()
    }

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    train_set = load_split('train')
    val_set = load_split('val')
    test_set = load_split('test')

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    num_classes = int(max(train_set.tensors[1].max(), val_set.tensors[1].max(), test_set.tensors[1].max()).item()) + 1
    print(f"Number of classes: {num_classes}")

    # Instantiate the 3 models
    models = [
        ResNet1D(num_classes=num_classes),
        StandardCNN1D(num_classes=num_classes),
        CNNBiLSTM1D(num_classes=num_classes)
    ]

    all_results = {}
    for m in models:
        res = train_and_evaluate(m, train_loader, val_loader, test_loader, device, epochs=30)
        all_results[m.name] = res

    # Save comparative metrics summary
    out_json = os.path.join(BASE_DIR, "three_models_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved 3-model benchmark report to: {out_json}")

if __name__ == '__main__':
    main()
