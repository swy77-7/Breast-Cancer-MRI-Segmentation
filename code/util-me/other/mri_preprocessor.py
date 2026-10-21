import os
import glob
import SimpleITK as sitk
import numpy as np
from skimage import restoration

class MRIPreprocessor:
    """A class to preprocess 3D MRI images (single phase) using bias correction and denoising."""
    
    def __init__(self, images_folder, output_folder, segmentations_folder=None, 
                 use_otsu_threshold=False, shrink_factor=0, patch_size=5, 
                 patch_distance=6, h=0.8):
        """
        Initialize the MRI preprocessor with explicit parameters.
        
        Parameters:
        -----------
        images_folder : str
            Path to the folder containing MRI images (e.g., /workspace/3122006213/nnUNet_raw/data/images)
        output_folder : str
            Path to save preprocessed images (e.g., /workspace/3122006213/nnUNet_raw/data_pre/images)
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
        """
        self.images_folder = images_folder
        self.output_folder = output_folder
        self.segmentations_folder = segmentations_folder
        self.use_otsu_threshold = use_otsu_threshold
        self.shrink_factor = shrink_factor
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.h = h
        
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
        self.processed_files = self._load_processed_files()

    def _load_processed_files(self):
        """Loads the set of already processed patient IDs from the log file."""
        processed = set()
        if os.path.exists(self.log_file_path):
            try:
                with open(self.log_file_path, 'r') as f:
                    processed = set(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"Warning: Could not read log file {self.log_file_path}. Starting fresh. Error: {e}")
        return processed

    def read_mri(self, patient_id):
        """
        Read 3D MRI image (phase 0) for a given patient.
        
        Parameters:
        -----------
        patient_id : str
            The patient ID (e.g., PATIENT_0001)
            
        Returns:
        --------
        SimpleITK.Image
            The loaded 3D MRI image
        """
        image_path = os.path.join(self.images_folder, f"{patient_id}_0000.nii.gz")
        phase_sitk = sitk.ReadImage(image_path, sitk.sitkFloat32)
        
        # Verify 3D image
        if len(phase_sitk.GetSize()) != 3:
            raise ValueError(f"Image {image_path} is not 3D")
            
        return phase_sitk

    def read_segmentation(self, patient_id):
        """
        Read 3D segmentation mask for a given patient.
        
        Parameters:
        -----------
        patient_id : str
            The patient ID (e.g., PATIENT_0001)
            
        Returns:
        --------
        SimpleITK.Image or None
            The loaded 3D segmentation mask or None if segmentations_folder is not set
        """
        if self.segmentations_folder is None:
            return None
        mask_path = os.path.join(self.segmentations_folder, f"{patient_id}.nii.gz")
        if not os.path.exists(mask_path):
            print(f"Warning: Segmentation mask not found for {patient_id} at {mask_path}")
            return None
        mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        
        # Verify 3D mask
        if len(mask_sitk.GetSize()) != 3:
            raise ValueError(f"Segmentation {mask_path} is not 3D")
            
        return mask_sitk

    def bias_correction(self, image_sitk):
        """
        Apply N4 Bias Field Correction to the input 3D image.
        
        Parameters:
        -----------
        image_sitk : SimpleITK.Image
            Input 3D MRI image
            
        Returns:
        --------
        SimpleITK.Image
            Bias-corrected 3D image
        """
        if len(image_sitk.GetSize()) != 3:
            raise ValueError("Input image must be 3D for bias correction")
            
        if self.shrink_factor:
            mask_breast = sitk.OtsuThreshold(image_sitk, 0, 1)
            shrinked_image_sitk = sitk.Shrink(
                image_sitk, [self.shrink_factor] * image_sitk.GetDimension()
            )
            shrinked_mask_breast = sitk.Shrink(
                mask_breast, [self.shrink_factor] * mask_breast.GetDimension()
            )
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
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
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            if self.use_otsu_threshold:
                mask_image = sitk.OtsuThreshold(tmp_image, 0, 1)
                corrected_image_sitk = corrector.Execute(tmp_image, mask_image)
            else:
                corrected_image_sitk = corrector.Execute(tmp_image)
                
        return corrected_image_sitk

    def nlmeans_denoise(self, image_sitk):
        """
        Apply Non-Local Means denoising to the input 3D image.
        
        Parameters:
        -----------
        image_sitk : SimpleITK.Image
            Input 3D MRI image
            
        Returns:
        --------
        SimpleITK.Image
            Denoised 3D image
        """
        if len(image_sitk.GetSize()) != 3:
            raise ValueError("Input image must be 3D for denoising")
            
        image_np = sitk.GetArrayFromImage(image_sitk)
        denoised_np = restoration.denoise_nl_means(
            image_np,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance,
            h=self.h,
            fast_mode=True,
            channel_axis=None # For 3D single channel
        )
        denoised_image_sitk = sitk.GetImageFromArray(denoised_np.astype(np.float32))
        denoised_image_sitk.CopyInformation(image_sitk)
        return denoised_image_sitk

    def save_image(self, image_sitk, patient_id):
        """
        Save the preprocessed 3D image to the output folder.
        
        Parameters:
        -----------
        image_sitk : SimpleITK.Image
            The preprocessed 3D image
        patient_id : str
            The patient ID (e.g., PATIENT_0001)
        """
        output_path = os.path.join(self.output_folder, f"{patient_id}_0000.nii.gz")
        sitk.WriteImage(image_sitk, output_path)

    def check_spatial_consistency(self, image_sitk, mask_sitk, patient_id):
        """
        Check if the image and segmentation mask are spatially consistent.
        
        Parameters:
        -----------
        image_sitk : SimpleITK.Image
            The preprocessed 3D MRI image
        mask_sitk : SimpleITK.Image
            The 3D segmentation mask
        patient_id : str
            The patient ID (e.g., PATIENT_0001)
            
        Returns:
        --------
        bool
            True if spatially consistent, False otherwise
        """
        if mask_sitk is None:
            print(f"Warning: No segmentation mask for {patient_id}, skipping spatial consistency check.")
            return True  # No mask to check, proceed without error

        # Check size, spacingಮ
        if image_sitk.GetSize() != mask_sitk.GetSize():
            print(f"Error: Spatial inconsistency for {patient_id} - Size mismatch: "
                  f"Image size {image_sitk.GetSize()}, Mask size {mask_sitk.GetSize()}")
            return False

        # Check spacing
        if image_sitk.GetSpacing() != mask_sitk.GetSpacing():
            print(f"Error: Spatial inconsistency for {patient_id} - Spacing mismatch: "
                  f"Image spacing {image_sitk.GetSpacing()}, Mask spacing {mask_sitk.GetSpacing()}")
            return False

        # Check origin
        if image_sitk.GetOrigin() != mask_sitk.GetOrigin():
            print(f"Warning: Spatial inconsistency for {patient_id} - Origin mismatch: "
                  f"Image origin {image_sitk.GetOrigin()}, Mask origin {mask_sitk.GetOrigin()}")

        # Check direction
        if image_sitk.GetDirection() != mask_sitk.GetDirection():
            print(f"Warning: Spatial inconsistency for {patient_id} - Direction mismatch: "
                  f"Image direction {image_sitk.GetDirection()}, Mask direction {mask_sitk.GetDirection()}")

        return True

    def preprocess_patient(self, patient_id):
        """
        Preprocess a single 3D MRI image (phase 0) with bias correction and denoising.
        This method performs the core processing steps for one patient.
        
        Parameters:
        -----------
        patient_id : str
            The patient ID (e.g., PATIENT_0001)
            
        Returns:
        --------
        SimpleITK.Image
            Preprocessed 3D MRI image, or None if reading fails.
        """
        try:
            image_sitk = self.read_mri(patient_id)
        except Exception as e:
            print(f"Error reading MRI for {patient_id}: {str(e)}")
            return None
        
        corrected_image = self.bias_correction(image_sitk)
        denoised_image = self.nlmeans_denoise(corrected_image)
        return denoised_image

    def preprocess_folder(self):
        """
        Preprocess all 3D MRI images (phase 0) in the images_folder.
        Skips already processed images based on a log file.
        Handles associated segmentations if provided.
        
        Returns:
        --------
        list of tuple
            List of (patient_id, SimpleITK.Image) for each successfully processed image in this run.
        """
        results = []
        image_files = sorted(glob.glob(os.path.join(self.images_folder, "*_0000.nii.gz"))) # Sort for consistent order
        
        total_files = len(image_files)
        print(f"Found {total_files} image files to potentially process.")

        for i, image_file_path in enumerate(image_files):
            filename = os.path.basename(image_file_path)
            patient_id = filename.replace("_0000.nii.gz", "")
            
            print(f"\nChecking {patient_id} ({i+1}/{total_files})...")

            output_image_path = os.path.join(self.output_folder, f"{patient_id}_0000.nii.gz")
            output_segmentation_path = None
            if self.segmentations_folder and self.segmentation_output_folder:
                output_segmentation_path = os.path.join(self.segmentation_output_folder, f"{patient_id}.nii.gz")

            if patient_id in self.processed_files:
                image_exists = os.path.exists(output_image_path)
                segmentation_file_check_passed = True
                if output_segmentation_path: # If we expect a segmentation output
                    segmentation_file_check_passed = os.path.exists(output_segmentation_path)
                
                if image_exists and segmentation_file_check_passed:
                    print(f"Skipping {patient_id}: Already processed and output files exist.")
                    continue
                else:
                    print(f"Log for {patient_id} exists, but output file(s) missing. Reprocessing.")
            
            try:
                print(f"Processing {patient_id}...")
                preprocessed_image_sitk = self.preprocess_patient(patient_id)
                
                if preprocessed_image_sitk is None: # Error during preprocess_patient (e.g. read error)
                    print(f"Skipping {patient_id} due to error in preprocess_patient.")
                    continue

                # Read segmentation mask if available
                mask_sitk = self.read_segmentation(patient_id)

                # Check spatial consistency between image and mask
                if not self.check_spatial_consistency(preprocessed_image_sitk, mask_sitk, patient_id):
                    print(f"Skipping {patient_id} due to spatial inconsistency.")
                    continue

                # Save preprocessed image
                self.save_image(preprocessed_image_sitk, patient_id)
                print(f"Saved preprocessed image for {patient_id} to {output_image_path}")

                if self.segmentations_folder and self.segmentation_output_folder and mask_sitk:
                    # Resample mask to match preprocessed image space
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(preprocessed_image_sitk)
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Crucial for masks
                    resampler.SetDefaultPixelValue(0)  # Background value for mask
                    resampled_mask_sitk = resampler.Execute(mask_sitk)
                    sitk.WriteImage(resampled_mask_sitk, output_segmentation_path)
                    print(f"Saved resampled segmentation for {patient_id} to {output_segmentation_path}")

                with open(self.log_file_path, 'a') as f:
                    f.write(patient_id + '\n')
                self.processed_files.add(patient_id) # Update in-memory set

                results.append((patient_id, preprocessed_image_sitk))
                print(f"Successfully processed and logged {patient_id}")

            except Exception as e:
                print(f"Error processing {patient_id}: {str(e)}. This patient will not be logged as processed.")
        
        print(f"\nFinished processing all files. {len(results)} processed in this run.")
        return results

if __name__ == "__main__":
    # Define paths and parameters
    IMAGES_FOLDER = '/root/autodl-tmp/workspace/3122006213/nnUNet_raw/data/images'  # Example path
    OUTPUT_FOLDER = '/root/autodl-tmp/workspace/3122006213/nnUNet_raw/data_pre/images' # Example path
    SEGMENTATIONS_FOLDER = '/root/autodl-tmp/workspace/3122006213/nnUNet_raw/data/labels' # Example: if you have labels
    # SEGMENTATIONS_FOLDER = None  # Replace with segmentation folder or None
    
    USE_OTSU_THRESHOLD = False # Usually False for primary images if values are important, True if for mask generation
    SHRINK_FACTOR = 2         # Example value, 0 means no shrinking for N4
    PATCH_SIZE = 5            # NLMeans parameter
    PATCH_DISTANCE = 6        # NLMeans parameter
    H = 0.8                   # NLMeans parameter, higher means more smoothing

    # Initialize the preprocessor
    preprocessor = MRIPreprocessor(
        images_folder=IMAGES_FOLDER,
        output_folder=OUTPUT_FOLDER,
        segmentations_folder=SEGMENTATIONS_FOLDER,
        use_otsu_threshold=USE_OTSU_THRESHOLD,
        shrink_factor=SHRINK_FACTOR,
        patch_size=PATCH_SIZE,
        patch_distance=PATCH_DISTANCE,
        h=H
    )

    # Process all images in the folder
    processed_results = preprocessor.preprocess_folder()

    # Print summary of images processed in this run
    print("\nSummary of images processed in this run:")
    if processed_results:
        for patient_id, image_obj in processed_results:
            print(f"Patient: {patient_id}, Output Shape: {sitk.GetArrayFromImage(image_obj).shape}")
    else:
        print("No new images were processed in this run.")