# Label Source Documentation

This file records how stellar labels were obtained for the 62 APOGEE DR17 spectra used in Module 1.

## Primary source attempted
- SDSS‑IV APOGEE DR17 ASPCAP catalogue (table **apogeeStar**) – queried for a spectral type column.
- The query returned no usable spectral types for the objects.

## Outcome
All objects fell back to a temperature‑based mapping (see below).

## Temperature‑bin fallback mapping
Effective temperature (TEFF) from the FITS header is mapped to a Harvard class via the following bins (K):

| Lower (K) | Upper (K) | Class |
|----------|----------|-------|
| 0 | 3500 | M |
| 3500 | 5000 | K |
| 5000 | 6000 | G |
| 6000 | 7500 | F |
| 7500 | 10000 | A |
| 10000 | 30000 | B |
| 30000 | 50000 | O |

Objects with TEFF outside these ranges are labeled **UNKNOWN**.

## Manifest columns
- `file_name`: name of the FITS file (relative to `raw_spectra`).
- `objid`: APOGEE OBJID from the header (may be None).
- `teff`, `logg`, `feh`: stellar parameters from the header.
- `label`: spectral type string (e.g., `G`) derived from the temperature fallback.

All steps were performed programmatically; no manual edits were made to the labels.
