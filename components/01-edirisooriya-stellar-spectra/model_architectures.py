import torch
import torch.nn as nn

# -------------------------------------------------------------
# 1. Model 1: 1D-ResNet (Deep Residual Network)
# -------------------------------------------------------------
class BasicBlock1D(nn.Module):
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
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, num_classes=5, channels=[64, 128, 256]):
        super().__init__()
        self.name = "1D-ResNet"
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        layers = []
        in_c = channels[0]
        for out_c in channels:
            layers.append(BasicBlock1D(in_c, out_c))
            in_c = out_c
        self.layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_c, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# -------------------------------------------------------------
# 2. Model 2: Standard 1D-CNN (Multi-layer ConvNet Baseline)
# -------------------------------------------------------------
class StandardCNN1D(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.name = "Standard-1D-CNN"
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)

# -------------------------------------------------------------
# 3. Model 3: Hybrid 1D-CNN + BiLSTM (Sequential Deep Network)
# -------------------------------------------------------------
class CNNBiLSTM1D(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=64):
        super().__init__()
        self.name = "1D-CNN-BiLSTM"
        self.conv_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Input to LSTM will have 64 feature channels
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Conv feature extraction: (B, 1, 4000) -> (B, 64, L_reduced)
        feat = self.conv_stem(x)
        # Permute for LSTM: (B, seq_len, features)
        feat = feat.permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        # Global max pooling across time/sequence dimension
        pooled, _ = torch.max(lstm_out, dim=1)
        return self.classifier(pooled)
