# 04_nist_line_matching.py
"""
Physical Line Matching for SHAP Peaks in Stellar Spectra
=========================================================
Compares SHAP attribution peaks with known atomic/molecular
transition wavelengths from the NIST Atomic Spectra Database.
Computes the Absolute Average Attribution Score (AAAS) per line.

Usage:
    python 04_nist_line_matching.py \
        --shap_path shap_attributions.npz \
        --out_path line_matches.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

# Known atomic/molecular spectral lines (wavelength in Angstrom).
# APOGEE covers the near-infrared: ~15140-17000 A (H-band).
ASTRO_LINES = {
    # Hydrogen Brackett series (NIR)
    "H_Br10":    17366.0,
    "H_Br11":    16811.0,
    "H_Br12":    16114.0,
    "H_Br13":    15560.0,
    "H_Br14":    15264.0,
    # Carbon lines
    "C_I_16895": 16895.0,
    "C_I_16005": 16005.0,
    # Nitrogen lines
    "N_I_15961": 15961.0,
    "N_I_16764": 16764.0,
    # Oxygen lines
    "O_I_15569": 15569.0,
    "O_I_15871": 15871.0,
    # Iron lines (Fe I)
    "Fe_I_15194": 15194.0,
    "Fe_I_15207": 15207.0,
    "Fe_I_15395": 15395.0,
    "Fe_I_15490": 15490.0,
    "Fe_I_15648": 15648.0,
    "Fe_I_15964": 15964.0,
    "Fe_I_16040": 16040.0,
    "Fe_I_16153": 16153.0,
    "Fe_I_16165": 16165.0,
    # Magnesium lines
    "Mg_I_15745": 15745.0,
    "Mg_I_15770": 15770.0,
    "Mg_I_16397": 16397.0,
    # Silicon lines
    "Si_I_15960": 15960.0,
    "Si_I_16061": 16061.0,
    "Si_I_16094": 16094.0,
    # Calcium lines
    "Ca_I_16150": 16150.0,
    "Ca_I_16157": 16157.0,
    # Titanium lines
    "Ti_I_15334": 15334.0,
    "Ti_I_15543": 15543.0,
    # Aluminum lines
    "Al_I_16718": 16718.0,
    "Al_I_16763": 16763.0,
}


def match_shap_peaks_to_physics(shap_1d, wavelengths, tolerance=2.0):
    """
    Match SHAP attribution peaks to known spectral lines.

    Parameters
    ----------
    shap_1d      : np.ndarray, shape (n_wavelengths,)  Mean absolute SHAP per wavelength.
    wavelengths  : np.ndarray, shape (n_wavelengths,)  Wavelength grid in Angstrom.
    tolerance    : float  Max wavelength difference (A) to count as a match.

    Returns
    -------
    pd.DataFrame  Ranked by AAAS (Absolute Average Attribution Score).
    """
    results = []
    for name, line_wave in ASTRO_LINES.items():
        idx = np.where(np.abs(wavelengths - line_wave) <= tolerance)[0]
        if len(idx) == 0:
            results.append(dict(line_name=name, line_wavelength=line_wave,
                                mean_attribution=0.0, std_attribution=0.0, aaas=0.0))
            continue
        vals = shap_1d[idx]
        results.append(dict(
            line_name=name,
            line_wavelength=line_wave,
            mean_attribution=float(np.mean(vals)),
            std_attribution=float(np.std(vals)),
            aaas=float(np.mean(np.abs(vals))),
        ))
    df = pd.DataFrame(results).sort_values("aaas", ascending=False)
    return df


def main(shap_path, out_path, tolerance=2.0):
    data = np.load(shap_path)
    shap = data["shap"]           # could be (n_samples, n_wave) or (n_wave, n_classes)
    print(f"Loaded SHAP array shape: {shap.shape}")

    # Collapse to 1-D per-wavelength importance
    if shap.ndim == 1:
        shap_1d = np.abs(shap)
    elif shap.ndim == 2:
        # If (n_wave, n_classes) take mean over classes
        # If (n_samples, n_wave) take mean over samples
        # The wavelength axis is always 4000 (the longer dimension)
        if shap.shape[0] == 4000:
            shap_1d = np.mean(np.abs(shap), axis=1)  # (4000,)
        else:
            shap_1d = np.mean(np.abs(shap), axis=0)  # (4000,)
    else:
        shap_1d = np.mean(np.abs(shap), axis=tuple(range(shap.ndim - 1)))

    n_points = len(shap_1d)
    # Reconstruct APOGEE H-band log-wavelength grid
    # APOGEE CRVAL1=4.179, CDELT1=6e-6, 8575 pixels — we resampled to 4000
    log_min = 4.179
    log_max = 4.179 + 6e-6 * 8574
    log_grid = np.linspace(log_min, log_max, n_points)
    wave_grid = 10 ** log_grid

    print(f"Wavelength range: {wave_grid.min():.1f} - {wave_grid.max():.1f} A")
    df = match_shap_peaks_to_physics(shap_1d, wave_grid, tolerance)
    df.to_csv(out_path, index=False)
    print(f"Line-matching results saved to: {out_path}")
    print("\nTop 10 most important lines (by AAAS):")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match SHAP peaks to NIST lines")
    parser.add_argument("--shap_path",  required=True)
    parser.add_argument("--out_path",   required=True)
    parser.add_argument("--tolerance",  type=float, default=2.0,
                        help="Wavelength tolerance in Angstrom (default 2.0)")
    args = parser.parse_args()
    main(args.shap_path, args.out_path, args.tolerance)
