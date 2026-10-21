import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Paths (relative to this script)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, 'dataset_manifest_labeled.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pth')

# Load manifest to map labels
manifest = pd.read_csv(MANIFEST_PATH)
label_to_id = {lbl: idx for idx, lbl in enumerate(sorted(manifest['label'].unique()))}
num_classes = len(label_to_id)

# Define model architecture (same as training script)
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, num_classes=5, channels=[64, 128, 256]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        layers = []
        in_c = channels[0]
        for out_c in channels:
            layers.append(BasicBlock(in_c, out_c))
            in_c = out_c
        self.layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_c, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, length)
        x = self.stem(x)
        x = self.layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load model
model = ResNet1D(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# Load test set
test_npz = np.load(os.path.join(PROCESSED_DIR, 'test.npz'))
X_test = torch.tensor(test_npz['X'], dtype=torch.float32)
y_test = torch.tensor(test_npz['y'], dtype=torch.long)

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        _, pred = torch.max(logits, dim=1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()

accuracy = correct / total if total > 0 else 0.0
print(f"Test set size: {total}")
print(f"Number of classes: {num_classes}")
print(f"Class distribution (id -> label): {label_to_id}")
print(f"Test accuracy: {accuracy:.4f}")
