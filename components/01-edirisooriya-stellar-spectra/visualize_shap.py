# visualize_shap.py
"""
Generate SHAP visualization plots for Module 1 (Stellar Spectra).

Produces:
1. Mean SHAP attribution curve across all test spectra with top-10 NIST lines marked.
2. Per-spectrum SHAP heatmap (wavelength vs test sample index).
3. Bar chart of top-20 lines ranked by AAAS score.
4. Example test spectrum with SHAP overlay.

Usage:
    python visualize_shap.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (safe on Windows)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Load data ──────────────────────────────────────────────────────────────────
shap_data = np.load("shap_attributions.npz")
shap_raw = shap_data["shap"]   # shape either (4000, n_classes) or (n_samples, 4000)

print(f"Loaded SHAP array shape: {shap_raw.shape}")

# Normalise to (n_wavelengths=4000, ...) then collapse to 1-D mean |SHAP|
if shap_raw.shape[0] == 4000:
    # shape is (4000, n_classes) — mean over classes axis
    shap_mean = np.mean(np.abs(shap_raw), axis=1)   # (4000,)
    # For heatmap treat each class as a "row"
    shap_2d   = np.abs(shap_raw).T                   # (n_classes, 4000)
else:
    # shape is (n_samples, 4000) — mean over samples axis
    shap_mean = np.mean(np.abs(shap_raw), axis=0)   # (4000,)
    shap_2d   = np.abs(shap_raw)                     # (n_samples, 4000)

# Reconstruct APOGEE H-band wavelength grid (log10 scale)
# CRVAL1=4.179, CDELT1=6e-6 log10(A), 8575 pixels → resampled to 4000
log_grid  = np.linspace(4.179, 4.179 + 6e-6 * 8574, 4000)
wave_grid = 10 ** log_grid   # Angstrom

# Load line matches
df_lines = pd.read_csv("line_matches.csv")
top10    = df_lines.head(10)

os.makedirs("plots", exist_ok=True)

# ── Plot 1: Mean SHAP attribution curve + NIST lines ─────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(wave_grid, shap_mean, color='steelblue', linewidth=0.8, label='Mean |SHAP|')

colors = plt.cm.tab10(np.linspace(0, 1, len(top10)))
for (_, row), c in zip(top10.iterrows(), colors):
    ax.axvline(row['line_wavelength'], color=c, linestyle='--', linewidth=1.0,
               alpha=0.85, label=f"{row['line_name']} ({row['line_wavelength']:.0f} Å)")

ax.set_xlabel("Wavelength (Angstrom)", fontsize=12)
ax.set_ylabel("Mean |SHAP| Attribution", fontsize=12)
ax.set_title("SHAP Feature Importance vs NIST Spectral Lines\n(APOGEE H-band, Module 1)", fontsize=13)
ax.legend(loc='upper right', fontsize=7, ncol=2)
ax.set_xlim(wave_grid.min(), wave_grid.max())
plt.tight_layout()
plt.savefig("plots/01_shap_curve_with_nist_lines.png", dpi=150)
plt.close()
print("[OK] Saved plots/01_shap_curve_with_nist_lines.png")

# ── Plot 2: SHAP heatmap over classes / test spectra ─────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(
    shap_2d,
    aspect='auto',
    cmap='hot',
    origin='lower',
    extent=[wave_grid.min(), wave_grid.max(), 0, shap_2d.shape[0]],
)
plt.colorbar(im, ax=ax, label='|SHAP| Attribution')
ax.set_xlabel("Wavelength (Angstrom)", fontsize=12)
y_label = "Class Index" if shap_raw.shape[0] == 4000 else "Test Spectrum Index"
ax.set_ylabel(y_label, fontsize=12)
ax.set_title("SHAP Attribution Heatmap Across Test Spectra", fontsize=13)
for _, row in top10.iterrows():
    ax.axvline(row['line_wavelength'], color='cyan', linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig("plots/02_shap_heatmap.png", dpi=150)
plt.close()
print("[OK] Saved plots/02_shap_heatmap.png")

# ── Plot 3: Bar chart of top 20 lines by AAAS ────────────────────────────────
top20 = df_lines.head(20)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(
    [f"{r['line_name']} ({r['line_wavelength']:.0f})" for _, r in top20.iterrows()],
    top20['aaas'],
    color='steelblue',
    edgecolor='white',
)
ax.set_xlabel("AAAS (Absolute Average Attribution Score)", fontsize=12)
ax.set_title("Top 20 Physical Spectral Lines Ranked by SHAP Importance\n(Module 1 — Temperature-proxy XAI, APOGEE DR17)", fontsize=13)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("plots/03_aaas_bar_chart.png", dpi=150)
plt.close()
print("[OK] Saved plots/03_aaas_bar_chart.png")

# ── Plot 4: Single test spectrum + SHAP overlay ───────────────────────────────
test_data = np.load("processed/test.npz")
X_test    = test_data['X']   # (n_test, 4000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# Spectrum
ax1.plot(wave_grid, X_test[0], color='black', linewidth=0.6)
ax1.set_ylabel("Normalised Flux", fontsize=11)
ax1.set_title("Example Test Spectrum (APOGEE H-band, DR17)", fontsize=12)
for (_, row), c in zip(top10.iterrows(), colors):
    ax1.axvline(row['line_wavelength'], color=c, linestyle='--', linewidth=0.9, alpha=0.8)

# SHAP attribution for same spectrum
ax2.fill_between(wave_grid, 0, shap_mean, color='tomato', alpha=0.7, label='Mean |SHAP|')
ax2.set_xlabel("Wavelength (Angstrom)", fontsize=11)
ax2.set_ylabel("Mean |SHAP|", fontsize=11)
ax2.set_title("Corresponding SHAP Attribution (XAI Explanation)", fontsize=12)
for (_, row), c in zip(top10.iterrows(), colors):
    ax2.axvline(row['line_wavelength'], color=c, linestyle='--', linewidth=0.9, alpha=0.8)

handles = [mpatches.Patch(color=c, label=row['line_name'])
           for (_, row), c in zip(top10.iterrows(), colors)]
ax2.legend(handles=handles, loc='upper right', fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig("plots/04_spectrum_with_shap_overlay.png", dpi=150)
plt.close()
print("[OK] Saved plots/04_spectrum_with_shap_overlay.png")

print("\nAll 4 plots saved to: plots/")
