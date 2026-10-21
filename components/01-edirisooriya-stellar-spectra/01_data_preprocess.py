import os
import argparse
import numpy as np
import pandas as pd
import astropy.io.fits as fits
from scipy.ndimage import median_filter

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMP_BINS = [0, 3500, 5000, 6000, 7500, 10000, 30000, 50000]
CLASS_NAMES = ['M', 'K', 'G', 'F', 'A', 'B', 'O']

def teff_to_class(teff):
    for i in range(len(TEMP_BINS) - 1):
        if TEMP_BINS[i] <= teff < TEMP_BINS[i + 1]:
            return CLASS_NAMES[i]
    return 'UNKNOWN'

def continuum_normalize(flux, kernel_size=101):
    flux32 = flux.astype(np.float32)
    cont = median_filter(flux32, size=kernel_size, mode='reflect')
    cont[cont == 0] = 1e-6
    return (flux32 / cont).astype(np.float32)

def resample_log_wave(wave, flux, n_points=4000):
    valid = np.isfinite(flux) & np.isfinite(wave) & (wave > 0)
    if valid.sum() < 50:
        raise ValueError("Too few valid pixels after masking.")
    wave_v = wave[valid]
    flux_v = flux[valid]
    log_min = np.log10(wave_v.min())
    log_max = np.log10(wave_v.max())
    log_grid = np.linspace(log_min, log_max, n_points)
    wave_grid = 10 ** log_grid
    flux_interp = np.interp(wave_grid, wave_v, flux_v)
    return wave_grid, flux_interp.astype(np.float32)

def process_one_spectrum(fits_path):
    with fits.open(fits_path) as hdul:
        hdr = hdul[1].header
        flux_raw = hdul[1].data
    if flux_raw.ndim == 2:
        flux = flux_raw[0].astype(np.float64)
    else:
        flux = flux_raw.astype(np.float64)
    naxis1 = hdr.get('NAXIS1', len(flux))
    crval1 = hdr.get('CRVAL1', 4.179)
    cdelt1 = hdr.get('CDELT1', 6e-6)
    pixels = np.arange(naxis1)
    log_wave = crval1 + cdelt1 * pixels
    wave = 10 ** log_wave
    flux[flux <= 0] = np.nan
    wave_grid, flux_resampled = resample_log_wave(wave, flux, n_points=4000)
    norm_flux = continuum_normalize(flux_resampled, kernel_size=101)
    return norm_flux

def main(raw_dir, out_dir, manifest_path, test_split=0.15, val_split=0.15):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} entries.")
    
    label_counts = df['label'].value_counts()
    print("Label distribution in manifest:")
    print(label_counts)

    # Encode labels
    classes = sorted([c for c in df['label'].unique() if c != 'UNKNOWN'])
    label_to_id = {lbl: i for i, lbl in enumerate(classes)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}
    print(f"Target classes ({len(classes)}): {label_to_id}")

    X, y, ids = [], [], []
    for _, row in df.iterrows():
        fname = row['file_name']
        lbl = row['label']
        if lbl not in label_to_id:
            continue
        fits_p = os.path.join(raw_dir, fname)
        if not os.path.exists(fits_p):
            continue
        try:
            flux = process_one_spectrum(fits_p)
            X.append(flux)
            y.append(label_to_id[lbl])
            ids.append(str(row['objid']))
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    ids = np.array(ids)
    print(f"\nSuccessfully processed spectra: {len(X)}")

    # Stratified split attempt with fallback
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
        train_idx, test_idx = next(strat_split.split(X, y))
        val_split_adj = val_split / (1.0 - test_split)
        strat_split_val = StratifiedShuffleSplit(n_splits=1, test_size=val_split_adj, random_state=42)
        train_idx, val_idx = next(strat_split_val.split(X[train_idx], y[train_idx]))
        
        train_mask = np.zeros(len(y), dtype=bool)
        val_mask = np.zeros(len(y), dtype=bool)
        test_mask = np.zeros(len(y), dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        print("Successfully applied StratifiedShuffleSplit!")
    except Exception as e:
        print(f"Stratified split failed ({e}), falling back to unique OBJID split...")
        unique_ids = np.unique(ids)
        rng = np.random.default_rng(42)
        rng.shuffle(unique_ids)
        n_test = max(1, int(test_split * len(unique_ids)))
        n_val = max(1, int(val_split * len(unique_ids)))
        test_ids = set(unique_ids[:n_test])
        val_ids = set(unique_ids[n_test:n_test + n_val])
        train_ids = set(unique_ids[n_test + n_val:])
        train_mask = np.isin(ids, list(train_ids))
        val_mask = np.isin(ids, list(val_ids))
        test_mask = np.isin(ids, list(test_ids))

    np.savez(os.path.join(out_dir, 'train.npz'), X=X[train_mask], y=y[train_mask])
    np.savez(os.path.join(out_dir, 'val.npz'), X=X[val_mask], y=y[val_mask])
    np.savez(os.path.join(out_dir, 'test.npz'), X=X[test_mask], y=y[test_mask])

    print("\nSplits saved successfully:")
    print(f"  Train: {train_mask.sum()} samples")
    print(f"  Val:   {val_mask.sum()} samples")
    print(f"  Test:  {test_mask.sum()} samples")
    print("Class mapping:", label_to_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default=os.path.join(BASE_DIR, 'raw_spectra'))
    parser.add_argument('--out_dir', default=os.path.join(BASE_DIR, 'processed'))
    parser.add_argument('--manifest', default=os.path.join(BASE_DIR, 'dataset_manifest_labeled.csv'))
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.manifest)
