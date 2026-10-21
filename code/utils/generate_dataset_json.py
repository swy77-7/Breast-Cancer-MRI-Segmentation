import json
import os
from pathlib import Path
from collections import OrderedDict

def generate_dataset_json(task_id_str, base_dir, dataset_name, modalities, labels, description="", reference="N/A", licence="N/A", release_version="1.0"):
    """
    Generates a dataset.json file for nnUNet.

    Args:
        task_id_str (str): Task ID string, e.g., "Task101_BreastMRI".
        base_dir (Path): The base directory of nnUNet_raw (e.g., Path("D:/Project/MAMA_MIA/code/nnUNet_raw")).
        dataset_name (str): Name of the dataset (e.g., "BreastMRI").
        modalities (dict): Dictionary mapping modality keys (e.g., "0") to modality names (e.g., "DCE_Phase0").
        labels (dict): Dictionary mapping label keys (e.g., "0") to label names (e.g., "background").
        description (str, optional): Description of the dataset. Defaults to "".
        reference (str, optional): Reference for the dataset. Defaults to "N/A".
        licence (str, optional): Licence of the dataset. Defaults to "N/A".
        release_version (str, optional): Release version. Defaults to "1.0".
    """
    task_folder = base_dir / task_id_str
    images_tr_folder = task_folder / "imagesTr"
    #labels_tr_folder = task_folder / "labelsTr" # Assuming labelsTr exists or will be created

    # --- Sanity checks ---
    if not images_tr_folder.exists():
        print(f"Error: imagesTr folder not found at {images_tr_folder}")
        return
    # It's okay if labelsTr doesn't exist yet, but we'll need it for actual training.
    # if not labels_tr_folder.exists():
    #     print(f"Warning: labelsTr folder not found at {labels_tr_folder}. You will need this for training.")

    # --- Collect training cases ---
    training_cases = []
    # Expect filenames like PATIENTID_MODALITYKEY.nii.gz or PATIENTID_SUFFIX_MODALITYKEY.nii.gz
    # We need to identify unique patient IDs first.
    # The modality key is the part that matches the keys in the `modalities` dict.
    # For this script, we assume the first file found for a patient (e.g., _0000.nii.gz)
    # can represent the case in the dataset.json, and nnUNet will find other modalities.

    patient_ids = set()
    for f_name in os.listdir(images_tr_folder):
        if f_name.endswith(".nii.gz"):
            # Extract patient ID: everything before the last underscore if it's followed by a modality key
            parts = f_name.split('_')
            if len(parts) > 1:
                # Check if the part before .nii.gz and after the last '_' is a modality key
                potential_modality_key_with_ext = parts[-1]
                potential_modality_key = potential_modality_key_with_ext.split('.')[0] # remove .nii.gz
                if potential_modality_key in modalities:
                    patient_id = "_".join(parts[:-1])
                    patient_ids.add(patient_id)

    for patient_id in sorted(list(patient_ids)):
        # Create training cases for all modalities
        for modality_key in sorted(modalities.keys()):
            image_file_name = f"{patient_id}_{modality_key}.nii.gz"
            image_path_relative = f"./imagesTr/{image_file_name}"

            # Assume label file has the same patient_id
            label_file_name = f"{patient_id}.nii.gz"
            label_path_relative = f"./labelsTr/{label_file_name}"

            # Only add if the image file exists
            if (images_tr_folder / image_file_name).exists():
                training_cases.append(OrderedDict([("image", image_path_relative), ("label", label_path_relative)]))

    # num_training is now the total number of image-label pairs (patients × modalities)
    num_training = len(training_cases)

    # --- Create dataset_json content (OrderedDict to maintain order) ---
    dataset_info = OrderedDict()
    dataset_info["name"] = dataset_name
    dataset_info["description"] = description
    dataset_info["reference"] = reference
    dataset_info["licence"] = licence
    dataset_info["release"] = f"{release_version} {time.strftime('%Y/%m/%d')}"
    dataset_info["tensorImageSize"] = "3D" #  3D for DCE-MRI phases
    dataset_info["modality"] = OrderedDict(sorted(modalities.items())) # Sort for consistency
    dataset_info["labels"] = OrderedDict(sorted(labels.items())) # Sort for consistency
    dataset_info["numTraining"] = num_training
    dataset_info["numTest"] = 0 # Assuming no test cases are automatically found here
    dataset_info["training"] = training_cases
    dataset_info["test"] = [] # Placeholder for test cases
    dataset_info["file_ending"] = ".nii.gz"


    # --- Write to file ---
    output_file = task_folder / "dataset.json"
    with open(output_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Successfully generated dataset.json at {output_file}")
    print(f"Found {num_training} training cases.")

if __name__ == "__main__":
    import time # For release date

    # --- Configuration (Modify these values as needed) ---
    NNUNET_RAW_BASE_DIR = Path(r"D:\Project\MAMA_MIA\code\nnUNet\nnUNet_raw") # Path to your nnUNet_raw directory
    Dataset_ID_STRING = "Dataset101_BreastMRI" # e.g., "Dataset001_BrainTumour"
    DATASET_NAME = "BreastMRI"
    DATASET_DESCRIPTION = "DCE-MRI for breast tumor segmentation from MAMA-MIA dataset"

    # Modalities: key is what's used in filename (e.g., _0000.nii.gz -> key "0000"), value is descriptive name
    # For this script, we assume the keys in the modality dict are the suffixes in the filenames
    # e.g., if files are patient_0.nii.gz, patient_1.nii.gz, then keys should be "0", "1"
    MODALITIES = OrderedDict([
        ("0", "DCE_Phase0"), # Corresponds to _0.nii.gz or _0000.nii.gz if that's your file naming
        ("1", "DCE_Phase1"), # Corresponds to _1.nii.gz or _0001.nii.gz
        ("2", "DCE_Phase2"),
        ("3", "DCE_Phase3"),
        ("4", "DCE_Phase4")
    ])
    # Ensure your image files in imagesTr are named like: PATIENTID_MODALITYKEY.nii.gz
    # For example: duke_001_0.nii.gz, duke_001_1.nii.gz etc.
    # The script was previously trying to match PATIENTID_0000.nii.gz, PATIENTID_0001.nii.gz
    # The keys in MODALITIES dict ("0", "1", "2", "3", "4") must match the suffix in your filenames
    # (e.g., if your files are duke_001_0000.nii.gz, duke_001_0001.nii.gz, then keys should be "0000", "0001" etc.)
    # Let's adjust to match the `_000X` pattern from pre.py
    # The pre.py script saves files as PATIENTID_000PHASE.nii.gz
    # So, the modality keys here should reflect that.
    # However, the `pre.py` script uses `_000{phase}`.
    # The `dataset.json` `modality` keys are typically "0", "1", "2", etc.
    # And nnUNet expects filenames like `CASE_IDENTIFIER_XXXX.nii.gz` where XXXX is the modality index.
    # Let's assume the files in imagesTr will be named `duke_001_0000.nii.gz`, `duke_001_0001.nii.gz` etc.
    # And the modality keys in dataset.json should be "0000", "0001" etc. to match these.

    # Corrected MODALITIES based on typical nnUNet filename convention (CASE_IDENTIFIER_XXXX.nii.gz)
    # where XXXX is the modality index.
    # If your pre.py saves as duke_001_0000.nii.gz, duke_001_0001.nii.gz, etc.
    # then the keys here should be "0000", "0001", etc.
    MODALITIES_CORRECTED = OrderedDict([
        ("0000", "DCE_Phase0"),
        ("0001", "DCE_Phase1"),
        ("0002", "DCE_Phase2"),
        ("0003", "DCE_Phase3"),
        ("0004", "DCE_Phase4")
    ])


    LABELS = OrderedDict([
        ("0", "background"),
        ("1", "lesion") # Assuming "lesion" is label 1
    ])
    # --- End Configuration ---

    print("Generating dataset.json...")
    print(f"IMPORTANT: This script assumes your image files in '{NNUNET_RAW_BASE_DIR / Dataset_ID_STRING / 'imagesTr'}' "
          f"are named like 'PATIENTID_MODALITYKEY.nii.gz' (e.g., 'duke_001_0000.nii.gz', 'duke_001_0001.nii.gz').")
    print(f"The MODALITYKEY must match the keys in the MODALITIES_CORRECTED dictionary: {list(MODALITIES_CORRECTED.keys())}")
    print(f"Label files in 'labelsTr' should be named 'PATIENTID.nii.gz' (e.g., 'duke_001.nii.gz').")

    generate_dataset_json(
        task_id_str=Dataset_ID_STRING,
        base_dir=NNUNET_RAW_BASE_DIR,
        dataset_name=DATASET_NAME,
        modalities=MODALITIES_CORRECTED, # Use the corrected one
        labels=LABELS,
        description=DATASET_DESCRIPTION
    )