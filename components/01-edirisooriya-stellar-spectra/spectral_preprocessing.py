"""
spectral_preprocessing.py
=========================
Scientific preprocessing and input validation engine for stellar spectra (APOGEE DR17).
Provides uniform, contract-validated pipelines for FITS, CSV, NPY, and NPZ inputs.
"""

from pathlib import Path
import numpy as np
import io
import pandas as pd
from scipy.ndimage import median_filter

APOGEE_MIN_WAVE = 15100.8
APOGEE_MAX_WAVE = 16999.8
N_WAVE_DEFAULT = 4000


def build_wave_grid(n_points: int = N_WAVE_DEFAULT) -> np.ndarray:
    """Build the canonical 4,000-channel logarithmic wavelength grid across APOGEE H-band."""
    log_grid = np.linspace(4.179, 4.179 + 6e-6 * 8574, n_points)
    return 10 ** log_grid


def continuum_normalize(flux: np.ndarray, kernel_size: int = 101) -> np.ndarray:
    """
    Continuum normalization using running median filter, matching APOGEE training pipeline.
    Rejects non-finite values and preserves emission/absorption profiles.
    """
    if not np.isfinite(flux).all():
        raise ValueError("Input flux contains non-finite values (NaN or Inf).")
    flux32 = flux.astype(np.float32)
    cont = median_filter(flux32, size=kernel_size, mode='reflect')
    cont[cont == 0] = 1e-6
    return (flux32 / cont).astype(np.float32)


def resample_log_wave(wave: np.ndarray, flux: np.ndarray, n_points: int = N_WAVE_DEFAULT) -> tuple:
    """
    Resample arbitrary wavelength/flux pairs to the canonical 4000-bin logarithmic grid.
    Validates wavelength coverage and rejects non-overlapping spectra.
    """
    if wave is None or flux is None or len(wave) == 0 or len(flux) == 0:
        raise ValueError("Wavelength and flux arrays must be non-empty.")
    if len(wave) != len(flux):
        raise ValueError(f"Length mismatch: wave has {len(wave)} elements, flux has {len(flux)} elements.")

    valid = np.isfinite(flux) & np.isfinite(wave) & (wave > 0)
    if valid.sum() < 50:
        raise ValueError("Too few valid pixels after masking invalid/negative flux.")

    wave_v = wave[valid]
    flux_v = flux[valid]

    # Check overlap with APOGEE passband
    if wave_v.max() < APOGEE_MIN_WAVE or wave_v.min() > APOGEE_MAX_WAVE:
        raise ValueError(
            f"Wavelength range [{wave_v.min():.1f}, {wave_v.max():.1f} Å] does not overlap "
            f"with APOGEE H-band [{APOGEE_MIN_WAVE}, {APOGEE_MAX_WAVE} Å]."
        )

    sort_idx = np.argsort(wave_v)
    wave_v = wave_v[sort_idx]
    flux_v = flux_v[sort_idx]

    log_min = np.log10(max(wave_v.min(), APOGEE_MIN_WAVE))
    log_max = np.log10(min(wave_v.max(), APOGEE_MAX_WAVE))
    log_grid = np.linspace(log_min, log_max, n_points)
    wave_grid = 10 ** log_grid
    flux_interp = np.interp(wave_grid, wave_v, flux_v)

    if not np.isfinite(flux_interp).all():
        raise ValueError("Interpolated flux contains non-finite values (NaN or Inf).")

    return wave_grid, flux_interp.astype(np.float32)


def validate_and_preprocess_spectrum(
    raw_bytes: bytes,
    file_ext: str,
    wavelength_col: str = None,
    flux_col: str = None,
    sample_idx: int = 0
) -> tuple:
    """
    Unified, contract-validated ingestion for FITS, CSV, NPY, and NPZ spectra.

    Returns:
        spectrum_4000: np.ndarray of shape (4000,) float32
        wave_grid: np.ndarray of shape (4000,) float64
        status: descriptive string of processing status
    """
    ext = file_ext.lower().strip().replace(".", "")
    wave_grid = build_wave_grid(N_WAVE_DEFAULT)

    if ext in ("fits", "fit"):
        from astropy.io import fits
        with fits.open(io.BytesIO(raw_bytes)) as hdul:
            hdu_idx = 1 if len(hdul) > 1 else 0
            flux_raw = hdul[hdu_idx].data
            if flux_raw is None:
                raise ValueError("Selected FITS HDU contains no data array.")
            if flux_raw.ndim > 1:
                flux = flux_raw[0].astype(np.float64)
            else:
                flux = flux_raw.astype(np.float64)

            hdr = hdul[hdu_idx].header
            crval1 = hdr.get("CRVAL1", 4.179)
            cdelt1 = hdr.get("CDELT1", 6e-6)
            crpix1 = hdr.get("CRPIX1", 1.0)
            naxis1 = hdr.get("NAXIS1", len(flux))
            pixels = np.arange(naxis1)
            log_wave = crval1 + cdelt1 * (pixels - (crpix1 - 1.0))
            wave_raw = 10 ** log_wave

            flux[flux <= 0] = np.nan
            _, flux_interp = resample_log_wave(wave_raw, flux, n_points=N_WAVE_DEFAULT)
            spectrum_4000 = continuum_normalize(flux_interp, kernel_size=101)
            status = "Processed from raw FITS (WCS wavelength extraction -> 4,000 log-grid resample -> 101-pixel median continuum normalization)"

    elif ext == "csv":
        df = pd.read_csv(io.StringIO(raw_bytes.decode("utf-8", errors="replace")))
        has_wave = wavelength_col and wavelength_col in df.columns
        has_flux = flux_col and flux_col in df.columns

        if has_wave and has_flux:
            wave_raw = df[wavelength_col].values.astype(float)
            flux_raw = df[flux_col].values.astype(float)
            _, flux_interp = resample_log_wave(wave_raw, flux_raw, n_points=N_WAVE_DEFAULT)
            spectrum_4000 = continuum_normalize(flux_interp, kernel_size=101)
            status = "Processed from raw CSV (resampled to 4,000 log bins -> 101-pixel median continuum normalization)"
        else:
            # 1D flux array directly
            flux_raw = df[flux_col].values.astype(float) if has_flux else df.iloc[:, 0].values.astype(float)
            if len(flux_raw) != N_WAVE_DEFAULT:
                raise ValueError(
                    f"1D flux array length ({len(flux_raw)}) != {N_WAVE_DEFAULT}. "
                    f"Expected exactly {N_WAVE_DEFAULT} channels matching APOGEE H-band grid (15,100–17,000 Å). "
                    "For arbitrary wavelength channels, please provide a wavelength column."
                )
            if not np.isfinite(flux_raw).all():
                raise ValueError("Spectral array contains non-finite values (NaN or Inf).")

            med = float(np.median(flux_raw))
            if med < 0.2 or med > 5.0:
                spectrum_4000 = continuum_normalize(flux_raw, kernel_size=101)
                status = "Normalized 4,000-channel array (raw flux detected -> applied 101-pixel median continuum normalization)"
            else:
                spectrum_4000 = flux_raw.astype(np.float32)
                status = "Validated pre-normalized 4,000-channel spectrum (canonical log-grid 15,100–17,000 Å; continuum ~ 1.0)"

    elif ext == "npy":
        arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
        if not np.isfinite(arr).all():
            raise ValueError("NPY array contains non-finite values (NaN or Inf).")

        if arr.ndim == 1:
            if len(arr) != N_WAVE_DEFAULT:
                raise ValueError(f"Invalid 1D NPY length: {len(arr)}. Expected {N_WAVE_DEFAULT} channels.")
            flux_arr = arr
        elif arr.ndim == 2:
            if arr.shape[1] != N_WAVE_DEFAULT:
                raise ValueError(f"Invalid 2D NPY shape: {arr.shape}. Expected (N, {N_WAVE_DEFAULT}).")
            if sample_idx < 0 or sample_idx >= len(arr):
                raise ValueError(f"sample_idx {sample_idx} out of range for NPY of length {len(arr)}.")
            flux_arr = arr[sample_idx]
        else:
            raise ValueError(f"Unsupported NPY dimension: {arr.ndim}. Expected 1D or 2D array.")

        med = float(np.median(flux_arr))
        if med < 0.2 or med > 5.0:
            spectrum_4000 = continuum_normalize(flux_arr, kernel_size=101)
            status = "Normalized 4,000-channel array (raw flux detected -> applied 101-pixel median continuum normalization)"
        else:
            spectrum_4000 = flux_arr.astype(np.float32)
            status = "Validated pre-normalized 4,000-channel spectrum (canonical log-grid 15,100–17,000 Å; continuum ~ 1.0)"

    elif ext == "npz":
        d = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
        key = "X" if "X" in d else list(d.keys())[0]
        arr = d[key]
        if not np.isfinite(arr).all():
            raise ValueError("NPZ array contains non-finite values (NaN or Inf).")

        if arr.ndim == 1:
            if len(arr) != N_WAVE_DEFAULT:
                raise ValueError(f"Invalid 1D NPZ length: {len(arr)}. Expected {N_WAVE_DEFAULT} channels.")
            flux_arr = arr
        elif arr.ndim == 2:
            if arr.shape[1] != N_WAVE_DEFAULT:
                raise ValueError(f"Invalid 2D NPZ shape: {arr.shape}. Expected (N, {N_WAVE_DEFAULT}).")
            if sample_idx < 0 or sample_idx >= len(arr):
                raise ValueError(f"sample_idx {sample_idx} out of range for NPZ of length {len(arr)}.")
            flux_arr = arr[sample_idx]
        else:
            raise ValueError(f"Unsupported NPZ dimension: {arr.ndim}. Expected 1D or 2D array.")

        med = float(np.median(flux_arr))
        if med < 0.2 or med > 5.0:
            spectrum_4000 = continuum_normalize(flux_arr, kernel_size=101)
            status = "Normalized 4,000-channel array (raw flux detected -> applied 101-pixel median continuum normalization)"
        else:
            spectrum_4000 = flux_arr.astype(np.float32)
            status = "Validated pre-normalized 4,000-channel spectrum (canonical log-grid 15,100–17,000 Å; continuum ~ 1.0)"

    else:
        raise ValueError(f"Unsupported file format '.{ext}'. Supported formats: .fits, .csv, .npy, .npz")

    if not np.isfinite(spectrum_4000).all():
        raise ValueError("Processed spectrum contains non-finite values (NaN or Inf).")

    return spectrum_4000, wave_grid, status
