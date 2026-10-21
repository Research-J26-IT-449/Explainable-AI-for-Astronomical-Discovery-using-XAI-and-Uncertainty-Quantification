"""
04_scientific_validation.py
===========================
Scientific Atomic-Line Matching, Alignment Score, and Monte Carlo Random Baseline.
Evaluates whether SHAP attribution peaks align with known stellar transitions
significantly better than chance.
"""

import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Literature NIR atomic transitions in APOGEE H-band (15,100 - 17,000 A)
# References: NIST Atomic Spectra Database, Shetrone et al. (2015), Smith et al. (2013)
ASTRO_LINES = {
    "H_Br14":    15264.7,
    "H_Br13":    15560.7,
    "H_Br12":    16113.7,
    "H_Br11":    16811.1,
    "Fe_I_15194": 15194.5,
    "Fe_I_15207": 15207.5,
    "Fe_I_15395": 15395.7,
    "Fe_I_15490": 15490.3,
    "Fe_I_15648": 15648.5,
    "Fe_I_15964": 15964.4,
    "Fe_I_16040": 16040.7,
    "Fe_I_16153": 16153.2,
    "Fe_I_16165": 16165.0,
    "Ti_I_15334": 15334.8,
    "Ti_I_15543": 15543.8,
    "Mg_I_15745": 15745.0,
    "Mg_I_15770": 15770.1,
    "Mg_I_16397": 16397.9,
    "Si_I_15960": 15960.1,
    "Si_I_16061": 16060.0,
    "Si_I_16094": 16094.8,
    "Al_I_16718": 16718.9,
    "Al_I_16763": 16763.4,
    "N_I_15961":  15961.3,
    "N_I_16764":  16764.3,
    "O_I_15569":  15569.0,
    "O_I_15871":  15871.0,
    "C_I_16005":  16005.0,
    "C_I_16895":  16895.0,
    "Ca_I_16150": 16150.8,
    "Ca_I_16157": 16157.4
}

def main():
    # 1. Load clean SHAP attributions
    shap_data = np.load('results/shap_attributions_clean.npz')
    global_shap = shap_data['global_mean_abs']  # (4000,)
    mean_abs_per_class = shap_data['mean_abs_per_class']  # (4000, 5)

    # 2. Build exact model wavelength grid
    log_min, log_max = 4.179, 4.179 + 6e-6 * 8574
    wave_grid = 10 ** np.linspace(log_min, log_max, 4000)

    # 3. Peak detection on global attribution curve
    # Prominence threshold at 85th percentile of attributions
    prominence_thresh = np.percentile(global_shap, 85)
    peak_indices, properties = find_peaks(global_shap, prominence=prominence_thresh, distance=15)
    peak_waves = wave_grid[peak_indices]
    peak_strengths = global_shap[peak_indices]

    print(f"Detected {len(peak_waves)} prominent global SHAP attribution peaks.")

    # 4. Match against NIST/APOGEE reference lines (tolerance = 3.5 A)
    TOLERANCE = 3.5  # Angstrom
    matched_peaks = []
    unmatched_peaks = []
    line_match_records = []

    for pw, ps in zip(peak_waves, peak_strengths):
        # Find closest reference line
        closest_line = None
        min_dist = float('inf')
        for lname, lwave in ASTRO_LINES.items():
            dist = abs(pw - lwave)
            if dist < min_dist:
                min_dist = dist
                closest_line = (lname, lwave)

        if min_dist <= TOLERANCE:
            matched_peaks.append({
                'peak_wavelength': float(pw),
                'attribution_strength': float(ps),
                'matched_line': closest_line[0],
                'reference_wavelength': float(closest_line[1]),
                'wavelength_diff': float(min_dist)
            })
            line_match_records.append(matched_peaks[-1])
        else:
            unmatched_peaks.append({
                'peak_wavelength': float(pw),
                'attribution_strength': float(ps),
                'nearest_line': closest_line[0],
                'distance': float(min_dist)
            })

    total_peaks = len(peak_waves)
    n_matched = len(matched_peaks)
    alignment_score = (n_matched / total_peaks) * 100.0 if total_peaks > 0 else 0.0

    print(f"Alignment Score: {alignment_score:.2f}% ({n_matched} / {total_peaks} peaks matched within {TOLERANCE} A)")

    # 5. Monte Carlo Random Baseline Comparison
    N_TRIALS = 1000
    random_scores = []
    rng = np.random.default_rng(42)

    for _ in range(N_TRIALS):
        # Draw same number of random peaks across the passband
        rand_waves = rng.uniform(wave_grid.min(), wave_grid.max(), size=total_peaks)
        r_matched = 0
        for rw in rand_waves:
            if any(abs(rw - lw) <= TOLERANCE for lw in ASTRO_LINES.values()):
                r_matched += 1
        random_scores.append((r_matched / total_peaks) * 100.0)

    mean_random_score = float(np.mean(random_scores))
    std_random_score = float(np.std(random_scores))
    p_value = float(np.sum(np.array(random_scores) >= alignment_score) / N_TRIALS)

    print(f"Random Baseline Mean Alignment: {mean_random_score:.2f}% +/- {std_random_score:.2f}%")
    print(f"Statistical Significance: Empirical p-value = {p_value:.4f}")

    # 6. Save line matches and validation summary
    df_matches = pd.DataFrame(line_match_records).sort_values('attribution_strength', ascending=False)
    df_matches.to_csv(str(ROOT / 'results' / 'scientific_line_matches_verified.csv'), index=False)

    summary = {
        'total_detected_peaks': total_peaks,
        'matched_peaks': n_matched,
        'unmatched_peaks': len(unmatched_peaks),
        'tolerance_angstrom': TOLERANCE,
        'alignment_score_pct': alignment_score,
        'random_baseline_mean_pct': mean_random_score,
        'random_baseline_std_pct': std_random_score,
        'empirical_p_value': p_value,
        'statistically_significant': bool(p_value < 0.05),
        'top_matched_lines': df_matches[['matched_line', 'peak_wavelength', 'attribution_strength']].to_dict(orient='records')
    }

    with open(str(ROOT / 'results' / 'scientific_validation_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # 7. Diagnostic Plot: Overlay with Line Indicators & Random Distribution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [2, 1.2]})

    # Top: Global SHAP curve with detected peaks and matched lines
    ax1.plot(wave_grid, global_shap, color='#1f77b4', lw=1.2, label='Global SHAP Importance')
    for m in matched_peaks:
        ax1.axvline(m['reference_wavelength'], color='crimson', ls='--', alpha=0.5, lw=1.0)
        ax1.scatter([m['peak_wavelength']], [m['attribution_strength']], color='red', s=35, zorder=5)
        ax1.text(m['peak_wavelength'], m['attribution_strength'] * 1.05, m['matched_line'],
                 fontsize=7, rotation=45, ha='left', va='bottom', color='darkred', fontweight='bold')

    ax1.set_title(f"SHAP Feature Alignment with NIST Stellar Lines (Alignment Score: {alignment_score:.1f}%)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Wavelength (Angstrom)", fontsize=10)
    ax1.set_ylabel("Mean |SHAP Attribution|", fontsize=10)
    ax1.set_xlim(15100, 17000)
    ax1.legend(loc='upper right', fontsize=9)

    # Bottom: Monte Carlo Random Baseline Histogram
    ax2.hist(random_scores, bins=25, color='#7f7f7f', alpha=0.7, edgecolor='black', label=f'Monte Carlo Trials (N={N_TRIALS})')
    ax2.axvline(alignment_score, color='red', lw=2.5, ls='-', label=f'Observed Model Alignment ({alignment_score:.1f}%, p={p_value:.4f})')
    ax2.axvline(mean_random_score, color='blue', lw=2.0, ls=':', label=f'Random Expectation ({mean_random_score:.1f}%)')
    ax2.set_title("Statistical Significance vs. Random Attribution Baseline", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Line Alignment Score (%)", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plot_dir = ROOT / 'plots' / 'xai_validation'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_dir / 'scientific_line_alignment.png'), dpi=200)
    plt.close()
    print("Scientific validation complete. Artifacts written to results/ and plots/xai_validation/")

if __name__ == '__main__':
    main()
