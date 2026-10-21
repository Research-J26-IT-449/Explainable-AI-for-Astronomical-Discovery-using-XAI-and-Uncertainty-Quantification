# Dataset Provenance: SDSS-IV APOGEE DR17

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  

---

## 1. Observational Survey Background
The spectra used in this module are drawn from the **Apache Point Observatory Galactic Evolution Experiment (APOGEE-2)**, a core component of the Sloan Digital Sky Survey IV (SDSS-IV), published as part of **Data Release 17 (DR17)** in December 2021.

- **Primary URL:** `https://data.sdss.org/sas/dr17/apogee/`
- **Spectrograph:** Cryogenic near-infrared, 300-fiber, high-resolution spectrograph ($R \approx 22,500$).
- **Wavelength Range:** Vacuum near-infrared $H$-band: $1.51 - 1.70\,\mu\text{m}$ ($15,100 - 17,000$ Å).
- **Target Selection:** Red giant branch (RGB) and red clump (RC) stars selected to map Galactic chemical evolution across heavy interstellar dust extinction.

---

## 2. Astronomical File Structure
Each downloaded spectrum corresponds to an individual APOGEE astronomical target identified by its 2MASS coordinate identifier (e.g., `2M00013313+7537342`):
- Format: Multi-Extension FITS (Flexible Image Transport System).
- HDU 1 contains combined rest-frame relative flux arrays.
- HDU 9 & 10 contain ASPCAP stellar parameters ($\text{T}_{\text{eff}}$, $\log g$, $[\text{Fe}/\text{H}]$).
- Reduction Pipeline: Version 0.17.22.
