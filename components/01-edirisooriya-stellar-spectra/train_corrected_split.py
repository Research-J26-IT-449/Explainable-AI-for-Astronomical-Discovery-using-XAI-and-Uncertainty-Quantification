import os, sys, json, copy
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from model_architectures import ResNet1D, StandardCNN1D, CNNBiLSTM1D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on device:', device)

def load_data(split):
    d = np.load(ROOT / 'processed' / 'corrected_clean' / f'{split}.npz')
    return TensorDataset(torch.tensor(d['X'], dtype=torch.float32), torch.tensor(d['y'], dtype=torch.long))

train_loader = DataLoader(load_data('train'), batch_size=16, shuffle=True)
val_loader = DataLoader(load_data('val'), batch_size=16, shuffle=False)
test_loader = DataLoader(load_data('test'), batch_size=16, shuffle=False)

def train_eval(model, epochs=25, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float('inf')
    best_w = None
    for ep in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                v_loss += criterion(model(X), y).item() * X.size(0)
        v_loss /= len(val_loader.dataset)
        if v_loss < best_loss:
            best_loss = v_loss
            best_w = copy.deepcopy(model.state_dict())  # deep-copy ensures snapshot is independent of future parameter updates
        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:02d}/{epochs:02d} - Val Loss: {v_loss:.4f}")

    model.load_state_dict(best_w)
    
    # Save weights
    out_dir = ROOT / 'models' / 'corrected_split'
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = model.name.lower().replace('-', '_') + '.pth'
    torch.save(model.state_dict(), out_dir / fname)
    
    # Evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = torch.argmax(model(X), dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f'{model.name} -> Acc: {acc:.4f}, Macro-F1: {f1:.4f}, Bal-Acc: {bal:.4f}')
    print(f'CM:\n{cm}')
    return {'accuracy': acc, 'macro_f1': f1, 'balanced_acc': bal, 'cm': cm.tolist()}

results = {}
for m in [CNNBiLSTM1D(num_classes=5), ResNet1D(num_classes=5), StandardCNN1D(num_classes=5)]:
    print(f'\nTraining {m.name}...')
    results[m.name] = train_eval(m, epochs=25)

with open(ROOT / 'results' / 'corrected_split_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nCorrected split benchmark saved to results/corrected_split_benchmark.json')
