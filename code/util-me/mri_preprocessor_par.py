
import os
import glob
import SimpleITK as sitk
import numpy as np
from skimage import restoration
from multiprocessing import Pool, Lock
import multiprocessing as mp
import psutil
import traceback
import signal
import sys

# Global lock and processed files list for multiprocessing
global_lock = None
global_processed_files = None


def init_pool(lock, processed_files):
    """Initializer for multiprocessing pool to set global lock and processed files."""
    global global_lock, global_processed_files
    global_lock = lock
    global_processed_files = processed_files


class MRIPreprocessor:
    """A class to preprocess 3D MRI images (single phase) using bias correction and denoising."""

    def __init__(self, images_folder, output_folder, segmentations_folder=None,
                 use_otsu_threshold=False, shrink_factor=0, patch_size=5,
                 patch_distance=6, h=0.8, num_processes=None):
        """
        Initialize the MRI preprocessor with explicit parameters.

        Parameters:
        -----------
        images_folder : str
            Path to the folder containing MRI images
        output_folder : str
            Path to save preprocessed images
        segmentations_folder : str, optional
            Path to segmentation masks
        use_otsu_threshold : bool
            Whether to use Otsu thresholding for bias correction
        shrink_factor : int
            Shrink factor for bias correction (0 for no shrinking)
        patch_size : int
            Size of patches for NLMeans denoising
        patch_distance : int
            Maximal distance for NLMeans patch search
        h : float
            Cut-off distance for NLMeans denoising
        num_processes : int, optional
            Number of parallel processes (auto-determined if None)
        """
        self.images_folder = images_folder
        self.output_folder = output_folder
        self.segmentations_folder = segmentations_folder
        self.use_otsu_threshold = use_otsu_threshold
        self.shrink_factor = shrink_factor
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.h = h

        # Calculate available memory
        mem_available = psutil.virtual_memory().available / (1024 ** 3)  # GB

        # Determine number of processes
        if num_processes is None:
            cpu_count = mp.cpu_count()  # 96 CPUs
            max_processes = min(cpu_count, int(mem_available / 0.6))  # ~83 with 50GB
            self.num_processes = max(1, min(12, int(max_processes * 0.75)))  # Cap at 12
            process_source = "auto-determined"
        else:
            self.num_processes = num_processes
            process_source = "user-specified"

        print(f"Using {self.num_processes} processes for parallel processing "
              f"({process_source}, based on {mem_available:.1f}GB available RAM).")

        # Ensure output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Setup for segmentation output and logging
        parent_output_dir = os.path.dirname(self.output_folder.rstrip('/\\'))
        self.segmentation_output_folder = None
        if self.segmentations_folder:
            self.segmentation_output_folder = os.path.join(parent_output_dir, "labels")
            if not os.path.exists(self.segmentation_output_folder):
                os.makedirs(self.segmentation_output_folder)

        self.log_file_path = os.path.join(parent_output_dir, "processed_log.txt")
        self.error_log_path = os.path.join(parent_output_dir, "error_log.txt")
        self.processed_files = self._load_processed_files()

    def _load_processed_files(self):
        """Loads the list of already processed patient IDs from the log file."""
        processed = []
        if os.path.exists(self.log_file_path):
            try:
                with open(self.log_file_path, 'r') as f:
                    processed = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Warning: Could not read log file {self.log_file_path}. Starting fresh. Error: {e}")
        return processed

    def read_mri(self, patient_id):
        """Read 3D MRI image for a given patient."""
        image_path = os.path.join(self.images_folder, f"{patient_id}_0000.nii.gz")
        phase_sitk = sitk.ReadImage(image_path, sitk.sitkFloat32)
        if len(phase_sitk.GetSize()) != 3:
            raise ValueError(f"Image {image_path} is not 3D")
        return phase_sitk

    def read_segmentation(self, patient_id):
        """Read 3D segmentation mask for a given patient."""
        if self.segmentations_folder is None:
            return None
        mask_path = os.path.join(self.segmentations_folder, f"{patient_id}.nii.gz")
        if not os.path.exists(mask_path):
            print(f"Warning: Segmentation mask not found for {patient_id} at {mask_path}")
            return None
        mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        if len(mask_sitk.GetSize()) != 3:
            raise ValueError(f"Segmentation {mask_path} is not 3D")
        return mask_sitk

    def bias_correction(self, image_sitk):
        """Apply N4 Bias Field Correction to the input 3D image."""
        if len(image_sitk.GetSize()) != 3:
            raise ValueError("Input image must be 3D for bias correction")

        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        if self.shrink_factor:
            mask_breast = sitk.OtsuThreshold(image_sitk, 0, 1)
            shrinked_image_sitk = sitk.Shrink(
                image_sitk, [self.shrink_factor] * image_sitk.GetDimension()
            )
            shrinked_mask_breast = sitk.Shrink(
                mask_breast, [self.shrink_factor] * mask_breast.GetDimension()
            )
            tmp_image = corrector.Execute(shrinked_image_sitk, shrinked_mask_breast)
            log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)
            corrected_image_sitk = image_sitk / sitk.Exp(log_bias_field)
        else:
            initial_img = image_sitk
            tmp_image = sitk.Cast(image_sitk, sitk.sitkFloat64)
            tmp_image_array = sitk.GetArrayFromImage(tmp_image)
            tmp_image_array[tmp_image_array == 0] = np.finfo(float).eps
            tmp_image = sitk.GetImageFromArray(tmp_image_array)
            tmp_image.CopyInformation(initial_img)
            if self.use_otsu_threshold:
                mask_image = sitk.OtsuThreshold(tmp_image, 0, 1)
                corrected_image_sitk = corrector.Execute(tmp_image, mask_image)
            else:
                corrected_image_sitk = corrector.Execute(tmp_image)
        return corrected_image_sitk

    def nlmeans_denoise(self, image_sitk):
        """Apply Non-Local Means denoising to the input 3D image."""
        if len(image_sitk.GetSize()) != 3:
            raise ValueError("Input image must be 3D for denoising")
        image_np = sitk.GetArrayFromImage(image_sitk)
        denoised_np = restoration.denoise_nl_means(
            image_np,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance,
            h=self.h,
            fast_mode=True,
            channel_axis=None
        )
        denoised_image_sitk = sitk.GetImageFromArray(denoised_np.astype(np.float32))
        denoised_image_sitk.CopyInformation(image_sitk)
        return denoised_image_sitk

    def save_image(self, image_sitk, patient_id):
        """Save the preprocessed 3D image to the output folder."""
        output_path = os.path.join(self.output_folder, f"{patient_id}_0000.nii.gz")
        sitk.WriteImage(image_sitk, output_path)

    def check_spatial_consistency(self, image_sitk, mask_sitk, patient_id):
        """Check if the image and segmentation mask are spatially consistent."""
        if mask_sitk is None:
            print(f"Warning: No segmentation mask for {patient_id}, skipping spatial consistency check.")
            return True
        if image_sitk.GetSize() != mask_sitk.GetSize():
            print(f"Error: Spatial inconsistency for {patient_id} - Size mismatch: "
                  f"Image size {image_sitk.GetSize()}, Mask size {mask_sitk.GetSize()}")
            return False
        if image_sitk.GetSpacing() != mask_sitk.GetSpacing():
            print(f"Error: Spatial inconsistency for {patient_id} - Spacing mismatch")
            return False
        if image_sitk.GetOrigin() != mask_sitk.GetOrigin():
            print(f"Warning: Spatial inconsistency for {patient_id} - Origin mismatch: "
                  f"Image origin {image_sitk.GetOrigin()}, Mask origin {mask_sitk.GetOrigin()}")
        if image_sitk.GetDirection() != mask_sitk.GetDirection():
            print(f"Warning: Spatial inconsistency for {patient_id} - Direction mismatch")
        return True

    def preprocess_patient(self, patient_id):
        """Preprocess a single 3D MRI image with preprocessing."""
        try:
            print(f"Processing patient {patient_id}...")
            image_sitk = self.read_mri(patient_id)
            corrected_image = self.bias_correction(image_sitk)
            denoised_image = self.nlmeans_denoise(corrected_image)
            return patient_id, denoised_image
        except Exception as e:
            with global_lock:
                with open(self.error_log_path, 'a') as f:
                    f.write(f"Error processing {patient_id}: {str(e)}\n")
                    f.write(traceback.format_exc() + '\n')
            return patient_id, None

    def process_patient_wrapper(self, patient_id):
        """Wrapper function to process a single patient and handle saving and logging."""
        output_image_path = os.path.join(self.output_folder, f"{patient_id}_0000.nii.gz")
        output_segmentation_path = None
        if self.segmentations_folder and self.segmentation_output_folder:
            output_segmentation_path = os.path.join(self.segmentation_output_folder, f"{patient_id}.nii.gz")

        if patient_id in global_processed_files:
            image_exists = os.path.exists(output_image_path)
            segmentation_file_check_passed = True
            if output_segmentation_path:
                segmentation_file_check_passed = os.path.exists(output_segmentation_path)
            if image_exists and segmentation_file_check_passed:
                print(f"Skipping {patient_id}: Already processed and output files exist.")
                return patient_id, True
            else:
                print(f"Log for {patient_id} exists, but output file(s) missing. Reprocessing.")

        patient_id, preprocessed_image_sitk = self.preprocess_patient(patient_id)

        if preprocessed_image_sitk is None:
            print(f"Skipping {patient_id} due to error in preprocess_patient.")
            return patient_id, False

        mask_sitk = self.read_segmentation(patient_id)
        if not self.check_spatial_consistency(preprocessed_image_sitk, mask_sitk, patient_id):
            print(f"Skipping {patient_id} due to spatial inconsistency.")
            with global_lock:
                with open(self.error_log_path, 'a') as f:
                    f.write(f"Error processing {patient_id}: Spatial inconsistency\n")
            return patient_id, False

        # Save image and log immediately
        self.save_image(preprocessed_image_sitk, patient_id)
        print(f"Saved preprocessed image for {patient_id} to {output_image_path}")

        if self.segmentations_folder and self.segmentation_output_folder and mask_sitk:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(preprocessed_image_sitk)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            resampled_mask_sitk = resampler.Execute(mask_sitk)
            sitk.WriteImage(resampled_mask_sitk, output_segmentation_path)
            print(f"Saved resampled segmentation for {patient_id} to {output_segmentation_path}")

        # Log patient ID immediately after saving all files
        with global_lock:
            with open(self.log_file_path, 'a') as f:
                f.write(patient_id + '\n')
            if patient_id not in global_processed_files:
                global_processed_files.append(patient_id)

        return patient_id, True

    def preprocess_folder(self):
        """Preprocess all 3D MRI images in the images_folder using parallel processing."""

        def signal_handler(sig, frame):
            print("\nReceived interrupt signal, shutting down pool gracefully...")
            pool.terminate()
            pool.join()
            sys.exit(0)

        image_files = sorted(glob.glob(os.path.join(self.images_folder, "*_0000.nii.gz")))
        total_files = len(image_files)
        print(f"Found {total_files} image files to potentially process.")

        patient_ids = [os.path.basename(f).replace("_0000.nii.gz", "") for f in image_files]

        # Register signal handler for KeyboardInterrupt
        signal.signal(signal.SIGINT, signal_handler)

        # Use imap_unordered for incremental result processing
        successful_results = []
        lock = Lock()
        try:
            with Pool(processes=self.num_processes, initializer=init_pool,
                      initargs=(lock, self.processed_files)) as pool:
                for patient_id, success in pool.imap_unordered(self.process_patient_wrapper, patient_ids):
                    if success:
                        successful_results.append((patient_id, True))
                    print(f"Completed processing or skipping {patient_id} (success: {success})")
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected, closing pool...")
            pool.terminate()
            pool.join()
            print("Pool closed, preserving logged patients.")
        except Exception as e:
            print(f"Unexpected error in pool: {e}")
            pool.terminate()
            pool.join()
            print("Pool closed, preserving logged patients.")

        print(f"\nFinished processing all files. {len(successful_results)} processed or skipped in this run.")
        return successful_results


if __name__ == "__main__":
    IMAGES_FOLDER = '/workspace/3122006213/nnUNet_raw/data/images'
    OUTPUT_FOLDER = '/workspace/3122006213/nnUNet_raw/data_pre/images'
    SEGMENTATIONS_FOLDER = '/workspace/3122006213/nnUNet_raw/data/labels'

    USE_OTSU_THRESHOLD = False
    SHRINK_FACTOR = 2
    PATCH_SIZE = 5
    PATCH_DISTANCE = 6
    H = 0.8
    NUM_PROCESSES = 4  # Specify 8 processes for 50GB RAM

    preprocessor = MRIPreprocessor(
        images_folder=IMAGES_FOLDER,
        output_folder=OUTPUT_FOLDER,
        segmentations_folder=SEGMENTATIONS_FOLDER,
        use_otsu_threshold=USE_OTSU_THRESHOLD,
        shrink_factor=SHRINK_FACTOR,
        patch_size=PATCH_SIZE,
        patch_distance=PATCH_DISTANCE,
        h=H,
        num_processes=NUM_PROCESSES
    )

    processed_results = preprocessor.preprocess_folder()

    print("\nSummary of images processed or skipped in this run:")
    if processed_results:
        for patient_id, _ in processed_results:
            print(f"Patient: {patient_id}")
    else:
        print("No images were processed or skipped in this run.")


