import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import shap
from model_architectures import CNNBiLSTM1D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running SHAP attribution on verified clean split using device:', device)

model = CNNBiLSTM1D(num_classes=5)
model.load_state_dict(torch.load(ROOT / 'models' / 'corrected_split' / '1d_cnn_bilstm.pth', map_location=device))
model.to(device)
model.eval()

test_data = np.load(ROOT / 'processed' / 'corrected_clean' / 'test.npz')
X_test = torch.tensor(test_data['X'], dtype=torch.float32).to(device)
y_test = test_data['y']

# Use representative 30 background samples from clean training set
train_data = np.load(ROOT / 'processed' / 'corrected_clean' / 'train.npz')
background = torch.tensor(train_data['X'][:30], dtype=torch.float32).to(device)

print(f'Computing GradientExplainer over {len(X_test)} test spectra...')
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_test)  # (150, 4000, 5)

print('Computed SHAP tensor shape:', shap_values.shape)
assert not np.isnan(shap_values).any(), 'NaN values detected in SHAP attributions!'
assert not np.isinf(shap_values).any(), 'Inf values detected in SHAP attributions!'

# Save full 3D attribution tensor and summary 2D
out_dir = ROOT / 'results'
out_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_dir / 'shap_attributions_clean.npz',
    shap_tensor=shap_values.astype(np.float32),
    mean_abs_per_class=np.mean(np.abs(shap_values), axis=0).astype(np.float32),  # (4000, 5)
    global_mean_abs=np.mean(np.abs(shap_values), axis=(0, 2)).astype(np.float32)  # (4000,)
)
print('Saved verified SHAP attributions to results/shap_attributions_clean.npz')
