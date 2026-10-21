# Report 02: Dataset Scientific Integrity Audit Report

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Provenance of Astronomical Data

The primary dataset consists of genuine astronomical spectra from the **Sloan Digital Sky Survey IV (SDSS-IV) APOGEE-2 (Apache Point Observatory Galactic Evolution Experiment 2) Data Release 17 (DR17)**.
* **Telescope & Instruments:** APO 2.5m telescope (Sunspot, NM) and LCO 2.5m Irénée du Pont telescope (Las Campanas, Chile).
* **Spectrograph:** High-resolution cryogenic near-infrared (NIR) multi-fiber spectrograph ($R \equiv \lambda / \Delta\lambda \approx 22,500$).
* **Spectral Coverage:** Astronomical near-infrared $H$-band ($1.51\,\mu\text{m} - 1.70\,\mu\text{m}$ or $15,100 - 17,000$ Å).
* **Data Origin:** Directly downloaded via HTTP Science Archive Server (`data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/`).
* **Software Reduction Pipeline:** `APOGEE Reduction Pipeline version 0.17.22`, ASPCAP (APOGEE Stellar Parameters and Chemical Abundances Pipeline) `synspec_rev1`.

### FITS HDU Structure Verification
Inspection of random files in `raw_spectra/*.fits` demonstrates standard APOGEE `apStar` multi-extension FITS format:
- **HDU 0 (PrimaryHDU):** Summary header containing astronomical coordinates (`RA`, `DEC`, `GLON`, `GLAT`), targeting flags (`APOGEE_TARGET1`, `APOGEE2_TARGET1`), 2MASS photometry (`J`, `H`, `K`), and survey identification (`FIELD`, `OBJID`).
- **HDU 1 (ImageHDU):** Spectrum array `(N_visits + 1, 8575)` pixels containing combined rest-frame relative flux as row 0 and individual visit exposures as subsequent rows. Wavelength dispersion is logarithmically sampled (`CRVAL1 = 4.179`, `CDELT1 = 6e-6`, `CTYPE1 = LOG-LINEAR`).
- **HDU 2 (ImageHDU):** Flux uncertainty / error array $\sigma_\lambda$.
- **HDU 3 (ImageHDU):** Bitmask pixel flags (`PIXMASK`).
- **HDU 9 & 10 (BinTableHDU):** ASPCAP parameter tables containing stellar parameters (`teff`, `logg`, `feh`, `alpha_m`) for combined and visit spectra.

---

## 2. Dataset Counts & Exclusion Analysis

| Category | Count | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Total FITS files on disk (`raw_spectra/`)** | 1,033 | Verified | Total files downloaded across multiple runs |
| **Cataloged in Manifest (`dataset_manifest_labeled.csv`)** | 1,004 | Verified | Complete valid FITS with ASPCAP parameters |
| **Active Machine Learning Cohort** | 1,000 | Verified | First 1,000 complete stars used in model experiments |
| **Files excluded from active ML cohort** | 33 | Verified | 29 incomplete/truncated downloads + 4 newest additions |

### Detailed Exclusion Reasons
1. **Truncated/Incomplete Downloads (28 files):** Interrupted downloads where files contained fewer than 10 HDUs (e.g., failed network socket termination during overnight script). These files triggered Astropy truncation warnings and were excluded to prevent corrupt tensor generation.
2. **Post-Batch Valid Files (5 files):** 1 file (`apStar-dr17-2M21241863%2B1240122.fits`) and 4 newest downloads (`apStar-dr17-2M21235315...`) are valid but downloaded after the 1,000-sample training pipeline was frozen.

---

## 3. Data Integrity & Missing Value Audit

- **Flux Values:** All 1,000 spectra in the active cohort were preprocessed without runtime exceptions. Non-positive flux values ($\le 0$) and cosmic ray artifacts were replaced with local median continuum values during log-linear resampling.
- **ASPCAP Physical Parameters:**
  - $\text{T}_{\text{eff}}$: 1,000 valid values (Range: $3,500.0\,\text{K}$ to $27,712.4\,\text{K}$, Mean: $4,917.4\,\text{K}$). Missing value count = 0.
  - $\log g$: 1,000 valid values (Range: $0.00$ to $5.00\,\text{dex}$, Mean: $2.49\,\text{dex}$). Missing value count = 0.
  - $[\text{Fe}/\text{H}]$: 1,000 valid values (Range: $-2.50$ to $+0.30\,\text{dex}$, Mean: $-0.62\,\text{dex}$). Missing value count = 0.
- **Physical Plausibility:** The Kiel diagram ($\log g$ vs $\text{T}_{\text{eff}}$) demonstrates typical red giant branch (RGB) and red clump (RC) concentrations consistent with APOGEE's infrared target selection function.
