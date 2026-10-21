# 03_shap_attributions.py
"""
SHAP Attribution Generation for Trained Stellar Spectra Model
================================================================
Loads the trained best model and computes SHAP values for the test set
using shap.GradientExplainer (DeepSHAP). Saves per-wavelength attribution
arrays as a NumPy .npz file.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import shap
from torch.utils.data import TensorDataset, DataLoader
from model_architectures import ResNet1D, StandardCNN1D, CNNBiLSTM1D

def load_test_data(npz_path):
    data = np.load(npz_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.long)
    return TensorDataset(X, y)

def load_best_model(model_path, device):
    state = torch.load(model_path, map_location=device)
    if 'lstm.weight_ih_l0' in state:
        num_classes = state['classifier.3.weight'].shape[0]
        model = CNNBiLSTM1D(num_classes=num_classes)
    elif 'features.0.weight' in state:
        num_classes = state['classifier.3.weight'].shape[0]
        model = StandardCNN1D(num_classes=num_classes)
    else:
        num_classes = state['fc.weight'].shape[0]
        model = ResNet1D(num_classes=num_classes)
    
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def main(model_path, test_dir, out_path, max_samples=75):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_best_model(model_path, device)

    # Load test set
    test_set = load_test_data(os.path.join(test_dir, 'test.npz'))
    loader = DataLoader(test_set, batch_size=64, shuffle=False)
    X_all = []
    for X_batch, _ in loader:
        X_all.append(X_batch)
    X_tensor = torch.cat(X_all, dim=0)[:max_samples].to(device)

    # Background = first min(30, N) samples
    n_bg = min(30, X_tensor.shape[0])
    background = X_tensor[:n_bg]

    print(f"Computing SHAP on {X_tensor.shape[0]} test spectra (background={n_bg})...")
    explainer  = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_tensor)

    # shap_values: list of arrays, one per class, each (N, length)
    # Mean absolute SHAP across classes -> (N, length)
    shap_array = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)

    np.savez(out_path, shap=shap_array)
    print(f"Saved SHAP attributions to {out_path}, shape={shap_array.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SHAP attributions for stellar spectra model')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--test_dir',   required=True)
    parser.add_argument('--out_path',   required=True)
    args = parser.parse_args()
    main(args.model_path, args.test_dir, args.out_path)
