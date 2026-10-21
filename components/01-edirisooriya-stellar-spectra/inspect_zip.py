import zipfile

zip_path = r"C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\MODULE1_STELLAR_SPECTRA_FINAL_AUDIT.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    fits = [name for name in z.namelist() if name.lower().endswith('.fits')]
    npz = [name for name in z.namelist() if name.lower().endswith('.npz')]
    csv = [name for name in z.namelist() if name.lower().endswith('.csv')]
    top_dirs = set(name.split('/')[0] for name in z.namelist())
    print(f"Total entries: {len(z.namelist())}")
    print(f"Top-level entries: {top_dirs}")
    print(f"FITS files: {len(fits)}")
    print(f"NPZ files: {len(npz)}")
    print(f"CSV files: {len(csv)}")
