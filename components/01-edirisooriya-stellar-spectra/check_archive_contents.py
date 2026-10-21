import zipfile

zip_path = r"C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\MODULE1_STELLAR_SPECTRA_FINAL_AUDIT.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    fits = [name for name in z.namelist() if name.lower().endswith('.fits')]
    npz = [name for name in z.namelist() if name.lower().endswith('.npz')]
    csv = [name for name in z.namelist() if name.lower().endswith('.csv')]
    print("FITS list (first 10):", fits[:10])
    print("NPZ list:", npz)
    print("CSV list (first 10):", csv[:10])
