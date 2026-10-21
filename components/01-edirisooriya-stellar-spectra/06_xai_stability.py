"""
06_xai_stability.py
===================
Evaluates the objective Stability of SHAP explanations under controlled
observational noise using Jaccard region similarity.
"""

import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from model_architectures import CNNBiLSTM1D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_top_regions(attr, top_k=200, window=10):
    """Identifies top-k bins and expands them into continuous regions to avoid single-bin boundary noise."""
    ranks = np.argsort(attr)[::-1][:top_k]
    region_mask = np.zeros_like(attr, dtype=bool)
    for r in ranks:
        start = max(0, r - window)
        end = min(len(attr), r + window + 1)
        region_mask[start:end] = True
    return set(np.where(region_mask)[0])

def jaccard_similarity(set_a, set_b):
    if len(set_a | set_b) == 0:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

def main():
    print("Executing XAI Stability Evaluation under SNR Perturbations...")
    # Load model
    model = CNNBiLSTM1D(num_classes=5)
    model.load_state_dict(torch.load(str(ROOT / 'models' / 'corrected_split' / '1d_cnn_bilstm.pth'), map_location=device))
    model.to(device)
    model.eval()

    # Load test data and background
    test_data = np.load(ROOT / 'processed' / 'corrected_clean' / 'test.npz')
    train_data = np.load(ROOT / 'processed' / 'corrected_clean' / 'train.npz')
    X_test = test_data['X']
    y_test = test_data['y']
    background = torch.tensor(train_data['X'][:20], dtype=torch.float32).to(device)

    # Test cohort of 25 spectra
    N_EVAL = 25
    eval_indices = np.arange(N_EVAL)

    explainer = shap.GradientExplainer(model, background)

    # Pre-compute original SHAP attributions for test cohort
    t_orig = torch.tensor(X_test[eval_indices], dtype=torch.float32).to(device)
    orig_shap = explainer.shap_values(t_orig)  # (N_EVAL, 4000, 5)

    # Perturbation noise levels (simulating observational SNR = 100, 50, 25)
    # APOGEE SNR typically 50 - 100
    noise_stds = [0.01, 0.02, 0.04]  # 1%, 2%, 4% flux noise
    stability_by_noise = {}

    rng = np.random.default_rng(42)

    for sigma in noise_stds:
        print(f"Testing noise level sigma={sigma} (equivalent to SNR ~ {int(1/sigma)})...")
        jaccard_scores = []
        # Create perturbed spectra
        noisy_X = X_test[eval_indices] + rng.normal(0, sigma, size=X_test[eval_indices].shape).astype(np.float32)
        t_noisy = torch.tensor(noisy_X, dtype=torch.float32).to(device)
        noisy_shap = explainer.shap_values(t_noisy)

        for i, idx in enumerate(eval_indices):
            target_class = y_test[idx]
            attr_orig = np.abs(orig_shap[i, :, target_class])
            attr_noisy = np.abs(noisy_shap[i, :, target_class])

            regions_orig = compute_top_regions(attr_orig, top_k=150, window=8)
            regions_noisy = compute_top_regions(attr_noisy, top_k=150, window=8)

            j_score = jaccard_similarity(regions_orig, regions_noisy)
            jaccard_scores.append(j_score)

        mean_j = float(np.mean(jaccard_scores))
        std_j = float(np.std(jaccard_scores))
        print(f"  Sigma {sigma} -> Mean Jaccard Similarity: {mean_j:.4f} +/- {std_j:.4f}")
        stability_by_noise[f"SNR_{int(1/sigma)} (sigma={sigma})"] = {
            "mean_jaccard": mean_j,
            "std_jaccard": std_j,
            "all_scores": [float(s) for s in jaccard_scores]
        }

    # Summary
    results = {
        "evaluation_cohort_size": N_EVAL,
        "region_window_bins": 8,
        "top_k_bins": 150,
        "stability_by_snr": stability_by_noise
    }

    with open(str(ROOT / 'results' / 'xai_stability_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot Stability vs Noise
    snrs = [100, 50, 25]
    means = [stability_by_noise[f"SNR_{snr} (sigma={1/snr:.2f})"]["mean_jaccard"] for snr in snrs]
    stds = [stability_by_noise[f"SNR_{snr} (sigma={1/snr:.2f})"]["std_jaccard"] for snr in snrs]

    plt.figure(figsize=(6, 4))
    plt.errorbar(snrs, means, yerr=stds, fmt='o-', color='#1b9e77', ecolor='darkgreen', elinewidth=1.5, capsize=4, lw=2)
    plt.title('XAI Stability: Jaccard Region Overlap vs Signal-to-Noise Ratio', fontsize=11, fontweight='bold')
    plt.xlabel('Simulated Observational SNR', fontsize=10)
    plt.ylabel('Mean Jaccard Region Overlap', fontsize=10)
    plt.ylim(0.4, 1.0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plot_dir = ROOT / 'plots' / 'xai_validation'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_dir / 'xai_stability_jaccard.png'), dpi=200)
    plt.close()
    print("Stability evaluation artifacts written to results/ and plots/xai_validation/")

if __name__ == '__main__':
    main()
