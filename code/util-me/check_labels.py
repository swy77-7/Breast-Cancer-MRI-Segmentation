
import os
import glob
import SimpleITK as sitk
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
import psutil
import gc
from multiprocessing import Pool, Lock, Manager
import signal
import sys

# Global lock and results for multiprocessing
global_lock = None
global_results = None


def init_pool(lock, results):
    """Initializer for multiprocessing pool."""
    global global_lock, global_results
    global_lock = lock
    global_results = results


class LabelChecker:
    """A class to check original and resampled segmentation labels for issues."""

    def __init__(self, images_folder, output_folder, segmentations_folder, log_file_path, num_processes=8):
        self.images_folder = images_folder
        self.output_folder = output_folder
        self.segmentations_folder = segmentations_folder
        self.log_file_path = log_file_path
        self.output_labels_folder = os.path.join(os.path.dirname(output_folder.rstrip('/\\')), "labels")
        self.error_log_path = os.path.join(os.path.dirname(log_file_path), "check_labels_error_log.txt")
        self.num_processes = min(num_processes, os.cpu_count(),
                                 int(psutil.virtual_memory().available / (0.6 * 1024 ** 3)))  # ~0.6GB per process
        os.makedirs(os.path.dirname(self.error_log_path), exist_ok=True)
        print(f"Using {self.num_processes} processes for parallel checking.")

    def _load_processed_patients(self):
        if not os.path.exists(self.log_file_path):
            print(f"Error: Log file {self.log_file_path} not found.")
            return []
        try:
            with open(self.log_file_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading log file {self.log_file_path}: {e}")
            return []

    def _read_image(self, patient_id):
        image_path = os.path.join(self.output_folder, f"{patient_id}_0000.nii.gz")
        if not os.path.exists(image_path):
            return None
        try:
            return sitk.ReadImage(image_path, sitk.sitkFloat32)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None

    def _read_original_mask(self, patient_id):
        mask_path = os.path.join(self.segmentations_folder, f"{patient_id}.nii.gz")
        if not os.path.exists(mask_path):
            return None
        try:
            return sitk.ReadImage(mask_path, sitk.sitkUInt8)
        except Exception as e:
            print(f"Error reading original mask {mask_path}: {e}")
            return None

    def _read_resampled_mask(self, patient_id):
        mask_path = os.path.join(self.output_labels_folder, f"{patient_id}.nii.gz")
        if not os.path.exists(mask_path):
            return None
        try:
            return sitk.ReadImage(mask_path, sitk.sitkUInt8)
        except Exception as e:
            print(f"Error reading resampled mask {mask_path}: {e}")
            return None

    def _get_image_properties(self, image_sitk):
        if image_sitk is None:
            return None
        try:
            array = sitk.GetArrayFromImage(image_sitk)
            props = {
                'size': image_sitk.GetSize(),
                'spacing': image_sitk.GetSpacing(),
                'origin': image_sitk.GetOrigin(),
                'direction': image_sitk.GetDirection(),
                'unique_values': np.unique(array).tolist(),
                'pixel_type': image_sitk.GetPixelIDTypeAsString()
            }
            del array
            return props
        except Exception as e:
            print(f"Error extracting properties: {e}")
            return None

    def check_patient(self, patient_id):
        result = {'patient_id': patient_id, 'status': 'OK', 'issues': []}
        mem_before = psutil.virtual_memory().used / (1024 ** 3)

        try:
            image_sitk = self._read_image(patient_id)
            original_mask_sitk = self._read_original_mask(patient_id)
            resampled_mask_sitk = self._read_resampled_mask(patient_id)

            if image_sitk is None:
                result['status'] = 'Error'
                result['issues'].append(f"Preprocessed image missing: {patient_id}_0000.nii.gz")
            if original_mask_sitk is None:
                result['status'] = 'Warning'
                result['issues'].append(f"Original mask missing: {patient_id}.nii.gz")
            if resampled_mask_sitk is None:
                result['status'] = 'Error'
                result['issues'].append(f"Resampled mask missing: {patient_id}.nii.gz")

            image_props = self._get_image_properties(image_sitk)
            original_mask_props = self._get_image_properties(original_mask_sitk)
            resampled_mask_props = self._get_image_properties(resampled_mask_sitk)

            if image_props and resampled_mask_props:
                if image_props['size'] != resampled_mask_props['size']:
                    result['status'] = 'Error'
                    result['issues'].append(
                        f"Size mismatch: Image {image_props['size']} vs Resampled Mask {resampled_mask_props['size']}")
                if image_props['spacing'] != resampled_mask_props['spacing']:
                    result['status'] = 'Error'
                    result['issues'].append(
                        f"Spacing mismatch: Image {image_props['spacing']} vs Resampled Mask {resampled_mask_props['spacing']}")
                if image_props['origin'] != resampled_mask_props['origin']:
                    result['status'] = 'Warning'
                    result['issues'].append(
                        f"Origin mismatch: Image {image_props['origin']} vs Resampled Mask {resampled_mask_props['origin']}")
                if image_props['direction'] != resampled_mask_props['direction']:
                    result['status'] = 'Warning'
                    result['issues'].append(
                        f"Direction mismatch: Image {image_props['direction']} vs Resampled Mask {resampled_mask_props['direction']}")

                if resampled_mask_props['unique_values'] == [0]:
                    result['status'] = 'Error'
                    result['issues'].append(f"Resampled mask is all zeros: {resampled_mask_props['unique_values']}")
                elif original_mask_props and resampled_mask_props['unique_values'] != original_mask_props[
                    'unique_values']:
                    result['status'] = 'Warning'
                    result['issues'].append(
                        f"Label values changed: Original {original_mask_props['unique_values']} vs Resampled {resampled_mask_props['unique_values']}")

                if resampled_mask_props['pixel_type'] != '8-bit unsigned integer':
                    result['status'] = 'Warning'
                    result['issues'].append(
                        f"Resampled mask pixel type incorrect: {resampled_mask_props['pixel_type']}")

            result['image_props'] = image_props
            result['original_mask_props'] = original_mask_props
            result['resampled_mask_props'] = resampled_mask_props

        except Exception as e:
            result['status'] = 'Error'
            result['issues'].append(f"Exception processing patient {patient_id}: {str(e)}")
            with global_lock:
                with open(self.error_log_path, 'a') as f:
                    f.write(f"Error processing {patient_id}: {str(e)}\n")
                    f.write(traceback.format_exc() + '\n')

        mem_after = psutil.virtual_memory().used / (1024 ** 3)
        print(
            f"Checked patient {patient_id}: {result['status']} - Issues: {', '.join(result['issues']) or 'None'} - Memory: {mem_before:.2f}GB -> {mem_after:.2f}GB")

        del image_sitk, original_mask_sitk, resampled_mask_sitk, image_props, original_mask_props, resampled_mask_props
        gc.collect()

        with global_lock:
            global_results.append(result)

        return patient_id, result['status']

    def check_all(self, batch_size=50):
        patient_ids = self._load_processed_patients()
        print(f"Found {len(patient_ids)} patients to check.")

        def signal_handler(sig, frame):
            print("\nReceived interrupt signal, shutting down pool...")
            pool.terminate()
            pool.join()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        manager = Manager()
        results = manager.list()
        lock = Lock()

        for i in range(0, len(patient_ids), batch_size):
            batch_ids = sorted(patient_ids[i:i + batch_size])
            print(f"Processing batch {i // batch_size + 1}: Patients {batch_ids[0]} to {batch_ids[-1]}")

            try:
                with Pool(processes=self.num_processes, initializer=init_pool, initargs=(lock, results)) as pool:
                    for patient_id, status in pool.imap_unordered(self.check_patient, batch_ids):
                        print(f"Completed {patient_id}: {status}")
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt detected, closing pool...")
                pool.terminate()
                pool.join()
                break
            except Exception as e:
                print(f"Unexpected error in pool: {e}")
                pool.terminate()
                pool.join()
                break

            self.results = list(results)  # Copy results
            self._save_results()
            results[:] = []  # Clear shared results
            gc.collect()

    def _save_results(self):
        results_df = pd.DataFrame([
            {
                'Patient_ID': r['patient_id'],
                'Status': r['status'],
                'Issues': '; '.join(r['issues']),
                'Image_Size': r['image_props']['size'] if r.get('image_props') else None,
                'Image_Spacing': r['image_props']['spacing'] if r.get('image_props') else None,
                'Image_Origin': r['image_props']['origin'] if r.get('image_props') else None,
                'Image_Direction': r['image_props']['direction'] if r.get('image_props') else None,
                'Image_Unique_Values': r['image_props']['unique_values'] if r.get('image_props') else None,
                'Original_Mask_Size': r['original_mask_props']['size'] if r.get('original_mask_props') else None,
                'Original_Mask_Spacing': r['original_mask_props']['spacing'] if r.get('original_mask_props') else None,
                'Original_Mask_Origin': r['original_mask_props']['origin'] if r.get('original_mask_props') else None,
                'Original_Mask_Direction': r['original_mask_props']['direction'] if r.get(
                    'original_mask_props') else None,
                'Original_Mask_Unique_Values': r['original_mask_props']['unique_values'] if r.get(
                    'original_mask_props') else None,
                'Resampled_Mask_Size': r['resampled_mask_props']['size'] if r.get('resampled_mask_props') else None,
                'Resampled_Mask_Spacing': r['resampled_mask_props']['spacing'] if r.get(
                    'resampled_mask_props') else None,
                'Resampled_Mask_Origin': r['resampled_mask_props']['origin'] if r.get('resampled_mask_props') else None,
                'Resampled_Mask_Direction': r['resampled_mask_props']['direction'] if r.get(
                    'resampled_mask_props') else None,
                'Resampled_Mask_Unique_Values': r['resampled_mask_props']['unique_values'] if r.get(
                    'resampled_mask_props') else None,
                'Resampled_Mask_Pixel_Type': r['resampled_mask_props']['pixel_type'] if r.get(
                    'resampled_mask_props') else None
            }
            for r in self.results
        ])
        output_csv = os.path.join(os.path.dirname(self.log_file_path),
                                  f"check_labels_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if os.path.exists(output_csv):
            results_df.to_csv(output_csv, mode='a', header=False, index=False)
        else:
            results_df.to_csv(output_csv, index=False)
        print(f"Saved partial results to {output_csv}")


if __name__ == "__main__":
    IMAGES_FOLDER = '/workspace/3122006213/nnUNet_raw/data/images'
    OUTPUT_FOLDER = '/workspace/3122006213/nnUNet_raw/data_pre/images'
    SEGMENTATIONS_FOLDER = '/workspace/3122006213/nnUNet_raw/data/labels'
    LOG_FILE_PATH = '/workspace/3122006213/nnUNet_raw/data_pre/processed_log.txt'
    NUM_PROCESSES = 16  # Adjust based on system

    checker = LabelChecker(
        images_folder=IMAGES_FOLDER,
        output_folder=OUTPUT_FOLDER,
        segmentations_folder=SEGMENTATIONS_FOLDER,
        log_file_path=LOG_FILE_PATH,
        num_processes=NUM_PROCESSES
    )
    checker.check_all(batch_size=50)

'''
### 3. Key Changes in the Updated Script

1. ** Parallel
Processing **:
- Uses
`multiprocessing.Pool`
with `num_processes=8` (configurable in `__main__`).
- Processes
patients in parallel
via
`pool.imap_unordered`
for efficient load balancing.
- Initializes
a
global
`Lock` and shared
`Manager.list`
for thread - safe result collection.

2. ** Batch
Processing **:
- Retains
batch_size = 50
to
save
partial
CSV
results and clear
memory.
- Each
batch is processed in parallel, reducing
total
runtime.

3. ** Memory
Management **:
- Deletes
SimpleITK
images and NumPy
arrays
after
use.
- Calls
`gc.collect()`
per
patient and batch.
- Limits
`num_processes`
based
on
available
RAM(~0.6
GB
per
process).
- Clears
shared
results
after
each
batch.

4. ** Error
Handling **:
- Uses
`Lock`
for thread - safe error logging and result appending.
- Handles
`KeyboardInterrupt` and pool
errors
gracefully, saving
partial
results.
- Logs
memory
usage
per
patient
to
detect
spikes.

5. ** Performance
Estimate **:
- ** Runtime **: ~171
patients / 8
processes * 1.5
s ≈ 32
s + I / O(~40 - 60
s
total).
- ** Memory **: 8 * 0.6
GB ≈ 4.8
GB, leaving
~11.2
GB
for OS and I / O, safer than sequential accumulation.

---

### 4. How to Use the Parallel Script

1. ** Save
the
Script **:
- Save as ` / workspace / 3122006213 / check_labels.py
`.
- Verify
paths in `__main__`:
```python
IMAGES_FOLDER = '/workspace/3122006213/nnUNet_raw/data/images'
OUTPUT_FOLDER = '/workspace/3122006213/nnUNet_raw/data_pre/images'
SEGMENTATIONS_FOLDER = '/workspace/3122006213/nnUNet_raw/data/labels'
LOG_FILE_PATH = '/workspace/3122006213/nnUNet_raw/data_pre/processed_log.txt'
NUM_PROCESSES = 8
```

2. ** Install
Dependencies **:
```bash
pip
install
SimpleITK
numpy
pandas
psutil
```

3. ** Run
the
Script **:
```bash
python / workspace / 3122006213 / check_labels.py
```
- Monitor
memory:
```bash
watch - n
2
free - h
```
- If
killed, reduce
`NUM_PROCESSES`
to
4:
```python
NUM_PROCESSES = 4
```

4. ** Interpret
Results **:
- ** Console
Output **: Shows
per - patient
status, e.g.:
```
Checked
patient
PATIENT_0171: OK - Issues: None - Memory: 8.2
GB -> 8.3
GB
```
- ** CSV **: ` / workspace / 3122006213 / nnUNet_raw / data_pre / check_labels_results_ *.csv
`
- Filter
for `Status == Error` or `Resampled_Mask_Unique_Values ==[0]` to find blank masks.
- Check
`Resampled_Mask_Pixel_Type`, `Resampled_Mask_Size`, and `Resampled_Mask_Spacing`
for issues.
    - ** Error
Log **: ` / workspace / 3122006213 / nnUNet_raw / data_pre / check_labels_error_log.txt
`
for file read errors.

5. ** Diagnose
Blank
Masks **:
- ** All - Zero
Masks **: Reprocess
with updated `mri_preprocessor_par.py` (previous response):
    ```python
    patient_ids = ['PATIENT_XXXX']  # From CSV
    ```
- ** Spatial
Mismatches **: Compare
`Image_ * ` vs. `
Resampled_Mask_ * ` in CSV.
- ** Pixel
Type
Issues **: Convert
masks
to
`sitkUInt8`:
```python
mask = sitk.ReadImage('path/to/mask.nii.gz')
mask = sitk.Cast(mask, sitk.sitkUInt8)
sitk.WriteImage(mask, 'path/to/fixed_mask.nii.gz')
```

6. ** Verify in Slicer **:
- Load
a
problematic
mask and image:
```
/ workspace / 3122006213 / nnUNet_raw / data_pre / labels / PATIENT_XXXX.nii.gz
/ workspace / 3122006213 / nnUNet_raw / data_pre / images / PATIENT_XXXX_0000.nii.gz
```
- Use
Label
Map, apply ** GenericAnatomyColors **, and check
overlay(per
previous
Slicer
guide).

---

### 5. Recommendations

1. ** Backup
Files **:
```bash
cp - r / workspace / 3122006213 / nnUNet_raw / data_pre / workspace / 3122006213 / nnUNet_raw / data_pre_backup
cp - r / workspace / 3122006213 / nnUNet_raw / data / labels / workspace / 3122006213 / nnUNet_raw / data / labels_backup
```

2. ** Close
Other
Processes **:
```bash
pkill - u $USER
python
pkill - u $USER
slicer
```

3. ** Test
Small
Batch **:
- Modify
`check_all`
to
test
10
patients:
```python
patient_ids = self._load_processed_patients()[174:184]  # Start from PATIENT_0175
```
- Ensure
stability
before
full
run.

4. ** Check
Disk
I / O **:
- If
slow, verify
disk
performance:
```bash
iostat - x
2
```

5. ** Add
Swap(Optional) **:
```bash
sudo
fallocate - l
8
G / swapfile
sudo
chmod
600 / swapfile
sudo
mkswap / swapfile
sudo
swapon / swapfile
free - h
```

6. ** Share
Results **:
- If
issues
persist, share:
- CSV
rows
for problematic patients.
    - Memory
    logs or error
    log
    contents.
- Slicer
observations.

---

### Conclusion
The
updated
`check_labels.py`
uses
parallel
processing
with 8 processes to reduce runtime from ~4.3 minutes to ~40-60 seconds for 171 patients.It retains all checks (file existence, spatial alignment, label values, pixel type) while addressing memory issues through batch processing and cleanup.Run the script, review the CSV for blank mask causes (e.g., all-zero masks), and reprocess affected patients with the updated `mri_preprocessor_par.py`.The parallel approach leverages your 96 CPUs efficiently within 16GB RAM constraints.Let me know the CSV results or any errors for further assistance!
'''