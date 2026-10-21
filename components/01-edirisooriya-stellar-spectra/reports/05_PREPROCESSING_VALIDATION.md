# Report 05: Preprocessing Validation and Spectral Resolution

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Preprocessing Pipeline Architecture

The preprocessing pipeline transforms raw APOGEE FITS spectra into standardized numerical tensors suitable for 1D deep convolutional and recurrent architectures:

```
Raw FITS (HDU 1: 8575 pixels)
  │
  ├── 1. Wavelength Reconstruction: log10(lambda) = CRVAL1 + CDELT1 * pixel
  │      lambda = 10^(log10(lambda))  [Range: 15,100.8 A - 16,999.8 A]
  │
  ├── 2. Quality Masking: Mask non-positive flux and sky emission spikes
  │
  ├── 3. Log-Linear Resampling: Resample 8575 raw channels to 4,000 uniform points
  │      dispersion: ~0.475 A/pixel
  │
  ├── 4. Continuum Normalization: Local median filter (kernel = 101 pixels)
  │      norm_flux = flux / cont_flux
  │
  └── 5. Tensor Output: (N, 4000) float32 array
```

---

## 2. Spectral Feature Preservation

A key concern in astrophysical machine learning is whether downsampling from 8,575 channels to 4,000 channels destroys narrow atomic absorption lines.

### Resolution Analysis:
- Total wavelength coverage: $\Delta\lambda = 16,999.8 - 15,100.8 = 1,899.0$ Å.
- Raw APOGEE channel spacing: $\delta\lambda_{\text{raw}} \approx 0.221$ Å/pixel ($R \approx 22,500$, FWHM $\approx 0.7$ Å).
- Resampled channel spacing: $\delta\lambda_{\text{resampled}} = 1,899.0 / 4,000 \approx 0.475$ Å/pixel.

### Finding:
At 0.475 Å/pixel, the Nyquist sampling criterion ($\ge 2$ pixels across line FWHM) is satisfied for typical stellar line profiles in giant stars (which undergo thermal, rotational, and instrumental broadening to FWHM $\ge 1.0$ Å). Key diagnostic lines (e.g., Al I at 16,718 Å, Si I at 15,960 Å, H Brackett-14 at 15,264 Å) are preserved as distinct, multi-pixel absorption troughs.
