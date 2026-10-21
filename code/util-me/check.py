import os
import glob
import nibabel as nib
import gzip
from pathlib import Path


def check_nii_gz(file_path):
    try:
        # Step 1: Check if file is a valid gzip
        with open(file_path, 'rb') as f:
            gzip.GzipFile(fileobj=f, mode='rb').read(1)  # Try reading a small portion
        # Step 2: Try loading as NIfTI
        img = nib.load(file_path)
        data = img.get_fdata()  # Attempt to access data
        if data.size == 0:
            return False, "Empty data"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def check_directory(dir_path):
    nii_files = glob.glob(os.path.join(dir_path, "*.nii.gz"))
    corrupted_files = []

    for file in nii_files:
        print(f"Checking {file}...")
        is_valid, message = check_nii_gz(file)
        if not is_valid:
            print(f"{file}: CORRUPTED ({message})")
            corrupted_files.append((file, message))
        else:
            print(f"{file}: OK")

    # Save corrupted files to a report
    with open("corrupted_nii_files.txt", "w") as f:
        for file, reason in corrupted_files:
            f.write(f"{file}: {reason}\n")

    return corrupted_files


if __name__ == "__main__":
    dir_path = "/workspace/3122006227/nnUNet_raw/Dataset105_full_image/imagesTr"
    corrupted_files = check_directory(dir_path)
    print(f"\nFound {len(corrupted_files)} corrupted files.")