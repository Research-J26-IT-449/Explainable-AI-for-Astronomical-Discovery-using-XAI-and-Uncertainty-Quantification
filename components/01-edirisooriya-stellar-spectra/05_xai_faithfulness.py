"""
05_xai_faithfulness.py
======================
Evaluates the objective Faithfulness of SHAP attributions using Area Over
Perturbation Curve (AOPC). Sequentially masks top-k important wavelength regions
vs random regions and measures model confidence degradation.
"""

import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model_architectures import CNNBiLSTM1D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("Executing XAI Faithfulness Evaluation (AOPC)...")
    # Load model
    model = CNNBiLSTM1D(num_classes=5)
    model.load_state_dict(torch.load(str(ROOT / 'models' / 'corrected_split' / '1d_cnn_bilstm.pth'), map_location=device))
    model.to(device)
    model.eval()

    # Load test data and clean SHAP attributions
    test_data = np.load(ROOT / 'processed' / 'corrected_clean' / 'test.npz')
    X_test = test_data['X']  # (150, 4000)
    y_test = test_data['y']

    shap_data = np.load(ROOT / 'results' / 'shap_attributions_clean.npz')
    shap_tensor = shap_data['shap_tensor']  # (150, 4000, 5)

    # Evaluate on representative subset of 50 correctly predicted test spectra
    correct_indices = []
    with torch.no_grad():
        preds = torch.argmax(model(torch.tensor(X_test, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
        for i in range(len(y_test)):
            if preds[i] == y_test[i]:
                correct_indices.append(i)

    eval_indices = correct_indices[:50]
    print(f"Evaluating faithfulness across {len(eval_indices)} correctly classified test stars...")

    steps = [0, 50, 100, 200, 300, 500, 800, 1200]  # Number of perturbed bins
    aopc_top_drops = []
    aopc_rand_drops = []

    rng = np.random.default_rng(42)

    for idx in eval_indices:
        x_orig = X_test[idx].copy()
        y_target = y_test[idx]
        
        # Attribution for the predicted target class
        attr = np.abs(shap_tensor[idx, :, y_target])
        sorted_ranks = np.argsort(attr)[::-1]  # Highest importance first

        # Baseline original probability
        with torch.no_grad():
            t_orig = torch.tensor(x_orig[np.newaxis, :], dtype=torch.float32).to(device)
            p_orig = F.softmax(model(t_orig), dim=1)[0, y_target].item()

        top_drop_curve = []
        rand_drop_curve = []

        for k in steps:
            if k == 0:
                top_drop_curve.append(0.0)
                rand_drop_curve.append(0.0)
                continue

            # Perturb top-k: replace with local continuum mean (1.0)
            x_top_perturbed = x_orig.copy()
            x_top_perturbed[sorted_ranks[:k]] = 1.0

            # Perturb random-k: replace with 1.0
            x_rand_perturbed = x_orig.copy()
            rand_bins = rng.choice(4000, size=k, replace=False)
            x_rand_perturbed[rand_bins] = 1.0

            with torch.no_grad():
                p_top = F.softmax(model(torch.tensor(x_top_perturbed[np.newaxis, :], dtype=torch.float32).to(device)), dim=1)[0, y_target].item()
                p_rand = F.softmax(model(torch.tensor(x_rand_perturbed[np.newaxis, :], dtype=torch.float32).to(device)), dim=1)[0, y_target].item()

            top_drop_curve.append(p_orig - p_top)
            rand_drop_curve.append(p_orig - p_rand)

        aopc_top_drops.append(top_drop_curve)
        aopc_rand_drops.append(rand_drop_curve)

    mean_top_drops = np.mean(aopc_top_drops, axis=0)
    mean_rand_drops = np.mean(aopc_rand_drops, axis=0)

    # Compute AOPC (Area Over Perturbation Curve) via trapezoidal integration
    aopc_score_top = float(np.trapezoid(mean_top_drops, steps) / max(steps))
    aopc_score_rand = float(np.trapezoid(mean_rand_drops, steps) / max(steps))

    print(f"Mean AOPC Score (Top-k Attributions): {aopc_score_top:.4f}")
    print(f"Mean AOPC Score (Random Baselines):  {aopc_score_rand:.4f}")
    print(f"Faithfulness Ratio (Top-k / Random): {aopc_score_top / aopc_score_rand:.2f}x")

    faithfulness_results = {
        "evaluation_samples": len(eval_indices),
        "steps_k_bins": steps,
        "mean_confidence_drop_top_k": [float(v) for v in mean_top_drops],
        "mean_confidence_drop_random": [float(v) for v in mean_rand_drops],
        "aopc_top_k": aopc_score_top,
        "aopc_random": aopc_score_rand,
        "faithfulness_ratio": float(aopc_score_top / aopc_score_rand)
    }

    with open(str(ROOT / 'results' / 'xai_faithfulness_metrics.json'), 'w') as f:
        json.dump(faithfulness_results, f, indent=2)

    # Plot AOPC curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, mean_top_drops, 'o-', color='#d95f02', lw=2, label=f'Top-k SHAP (AOPC={aopc_score_top:.3f})')
    plt.plot(steps, mean_rand_drops, 's--', color='#7570b3', lw=1.5, label=f'Random Perturbations (AOPC={aopc_score_rand:.3f})')
    plt.fill_between(steps, mean_top_drops, mean_rand_drops, color='#d95f02', alpha=0.15, label='Faithfulness Advantage')
    plt.title('XAI Faithfulness: Area Over Perturbation Curve (AOPC)', fontsize=11, fontweight='bold')
    plt.xlabel('Number of Masked Wavelength Bins (k)', fontsize=10)
    plt.ylabel('Mean Target Confidence Drop', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plot_dir = ROOT / 'plots' / 'xai_validation'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_dir / 'xai_faithfulness_aopc.png'), dpi=200)
    plt.close()
    print("Faithfulness evaluation artifacts saved to results/ and plots/xai_validation/")

if __name__ == '__main__':
    main()
