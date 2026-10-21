"""
visualize_dataset_types.py
==========================
Shows physically what the APOGEE FITS dataset looks like:
- Raw FITS file contents
- Spectrum plot (flux vs wavelength)
- Multiple spectra comparison across classes
- Preprocessed array vs raw
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import pandas as pd

RAW_DIR = r"C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\module1_stellar_spectra\raw_spectra"
MANIFEST = r"C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\module1_stellar_spectra\dataset_manifest_labeled.csv"
OUT_DIR = r"C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\module1_stellar_spectra\plots"
PROCESSED_DIR = r"C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\module1_stellar_spectra\processed"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(MANIFEST)

# ─────────────────────────────────────────────────────────────────
# 1. Show what a single FITS file looks like inside
# ─────────────────────────────────────────────────────────────────
sample_file = df.iloc[0]['file_name']
fits_path = os.path.join(RAW_DIR, sample_file)

with fits.open(fits_path) as hdul:
    hdul.info()
    hdr = hdul[0].header
    flux = hdul[1].data
    if flux.ndim > 1:
        flux = flux[0]   # combined spectrum
    naxis1 = hdr.get('NAXIS1', len(flux))
    crval1 = hdr.get('CRVAL1', 4.179)
    cdelt1 = hdr.get('CDELT1', 6e-6)
    objid  = hdr.get('APSTAR_ID', hdr.get('OBJID', 'Unknown'))
    teff   = hdr.get('RV_TEFF', '?')
    ra     = hdr.get('RA', '?')
    dec    = hdr.get('DEC', '?')

# Build wavelength grid from FITS WCS header
log_wave = np.array([crval1 + cdelt1 * i for i in range(len(flux))])
wave_ang = 10 ** log_wave   # Angstrom

print(f"FITS file: {sample_file}")
print(f"Object ID: {objid}")
print(f"TEFF: {teff} K | RA: {ra} | Dec: {dec}")
print(f"Flux array shape: {flux.shape} pixels")
print(f"Wavelength range: {wave_ang.min():.1f} - {wave_ang.max():.1f} Angstrom")

# ─────────────────────────────────────────────────────────────────
# 2. Plot: Single raw spectrum (what 1 star looks like)
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(wave_ang, flux, color='black', linewidth=0.5, alpha=0.9)
ax.set_xlabel("Wavelength (Angstrom)", fontsize=12)
ax.set_ylabel("Flux (counts)", fontsize=12)
ax.set_title(f"Raw APOGEE H-band Spectrum\nObject: {objid} | TEFF={teff}K | RA={ra:.3f} Dec={dec:.3f}", fontsize=11)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_single_raw_spectrum.png"), dpi=150)
plt.close()
print("[OK] 05_single_raw_spectrum.png")

# ─────────────────────────────────────────────────────────────────
# 3. Plot: One spectrum per class (K, G, F, A, B side by side)
# ─────────────────────────────────────────────────────────────────
class_colors = {'K': 'crimson', 'G': 'darkorange', 'F': 'gold',
                'A': 'steelblue', 'B': 'mediumpurple'}
class_labels_present = sorted(df['label'].unique())

fig, axes = plt.subplots(len(class_labels_present), 1, figsize=(14, 3*len(class_labels_present)), sharex=True)

for i, cls in enumerate(class_labels_present):
    subset = df[df['label'] == cls]
    row = subset.iloc[0]
    fpath = os.path.join(RAW_DIR, row['file_name'])
    try:
        with fits.open(fpath) as hdul:
            f = hdul[1].data
            if f.ndim > 1:
                f = f[0]
            hdr = hdul[0].header
            crv = hdr.get('CRVAL1', 4.179)
            cdt = hdr.get('CDELT1', 6e-6)
        w = 10 ** np.array([crv + cdt * j for j in range(len(f))])
        # Normalise for comparison
        med = np.nanmedian(f)
        scale = np.nanpercentile(np.abs(f - med), 99)
        if scale > 0:
            f_norm = (f - med) / scale
        else:
            f_norm = f - med

        ax = axes[i] if len(class_labels_present) > 1 else axes
        ax.plot(w, f_norm, color=class_colors.get(cls, 'black'), linewidth=0.6)
        ax.set_ylabel(f"Class {cls}\nTEFF={row['teff']:.0f}K", fontsize=10, color=class_colors.get(cls, 'black'))
        ax.set_ylim(-3, 3)
        ax.axhline(0, color='gray', linewidth=0.4, linestyle='--')
        ax.set_xlim(w.min(), w.max())
    except Exception as e:
        print(f"Error loading {cls}: {e}")

axes[-1].set_xlabel("Wavelength (Angstrom)", fontsize=12)
fig.suptitle("One APOGEE H-band Spectrum per Stellar Class\n(TEFF-based proxy labels, SDSS DR17)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_one_spectrum_per_class.png"), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 06_one_spectrum_per_class.png")

# ─────────────────────────────────────────────────────────────────
# 4. Plot: Preprocessed array (what the model actually sees)
# ─────────────────────────────────────────────────────────────────
test_data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
X_test = test_data['X']
y_test = test_data['y']

wave_resampled = 10 ** np.linspace(4.179, 4.179 + 6e-6 * 8574, 4000)
label_map = {0:'A', 1:'B', 2:'F', 3:'G', 4:'K'}

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

# Raw preprocessed
axes[0].plot(wave_resampled, X_test[0], color='steelblue', linewidth=0.7)
axes[0].set_ylabel("Normalised Flux", fontsize=11)
axes[0].set_title(f"Preprocessed Spectrum (what model sees) — Class: {label_map[y_test[0]]}", fontsize=11)

# 5 random spectra overlaid
for idx in range(min(5, len(X_test))):
    cls = label_map[y_test[idx]]
    axes[1].plot(wave_resampled, X_test[idx], linewidth=0.5, alpha=0.7,
                 color=class_colors.get(cls, 'gray'), label=f"Class {cls}")
axes[1].set_xlabel("Wavelength (Angstrom)", fontsize=11)
axes[1].set_ylabel("Normalised Flux", fontsize=11)
axes[1].set_title("5 Preprocessed Spectra Overlaid (4000 channels, normalised)", fontsize=11)

# Deduplicate legend
handles, labels_leg = axes[1].get_legend_handles_labels()
by_label = dict(zip(labels_leg, handles))
axes[1].legend(by_label.values(), by_label.keys(), fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_preprocessed_model_input.png"), dpi=150)
plt.close()
print("[OK] 07_preprocessed_model_input.png")

# ─────────────────────────────────────────────────────────────────
# 5. Plot: Full dataset as heatmap (all 1000 spectra as image)
# ─────────────────────────────────────────────────────────────────
X_train = np.load(os.path.join(PROCESSED_DIR, "train.npz"))['X']
X_val   = np.load(os.path.join(PROCESSED_DIR, "val.npz"))['X']
X_all   = np.concatenate([X_train, X_test, X_val], axis=0)

# Clamp for display
X_display = np.clip(X_all, -2.5, 2.5)

fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(X_display, aspect='auto', cmap='inferno',
               extent=[wave_resampled.min(), wave_resampled.max(), len(X_display), 0],
               vmin=-2.5, vmax=2.5)
plt.colorbar(im, ax=ax, label='Normalised Flux')
ax.set_xlabel("Wavelength (Angstrom)", fontsize=12)
ax.set_ylabel("Spectrum Index (each row = 1 star)", fontsize=12)
ax.set_title("All 1,000 APOGEE Spectra as an Image\n(Each row = 1 star, each column = 1 wavelength channel)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_full_dataset_as_image.png"), dpi=120)
plt.close()
print("[OK] 08_full_dataset_as_image.png")

print("\nAll visualizations saved to plots/")
print(f"\nSummary:")
print(f"  - Each FITS file = 1 star's spectrum = {len(flux)} wavelength pixels")
print(f"  - After resampling: 4000 channels per spectrum")
print(f"  - 1000 spectra total = 1000 rows in the dataset image")
print(f"  - Data TYPE: 1D numerical arrays (NOT images)")
print(f"  - Wavelength range: ~15,100 - 17,000 Angstrom (Near-Infrared H-band)")
