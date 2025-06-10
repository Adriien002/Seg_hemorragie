import os
import subprocess
import shutil
import glob
import numpy as np
import nibabel as nib
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# === CONFIGURATION ===

# DICOM input root
dicom_root = "/home/tibia/Documents/nnUNet/dicom_tests_rsna_ich"

# Intermediate folder for DICOM → NIfTI conversion
tmp_nifti_dir = "/home/tibia/Documents/nnUNet/tmp_niftis2"

# Final nnU-Net input (with _0000)
nnunet_input_dir = "/home/tibia/Documents/nnUNet/nnUNet_raw/Dataset005_WinMultiICHv5/imagesTs"

# Output normalized images for prediction
normalized_input_dir = "/home/tibia/Documents/nnUNet/nnUNet_raw/Dataset005_WinMultiICHv5/imagesTsgood"

# Output prediction folder
prediction_output_dir = "/home/tibia/Documents/nnUNet/prédictions2"

# Path to trained nnU-Net model
model_dir = "/home/tibia/Documents/nnUNet/docker_nnUNet/models/Dataset005_WinMultiICHv5/nnUNetTrainer__nnUNetPlans__3d_fullres"


# === STEP 1: Convert DICOM to NIfTI using dcm2niix ===

def convert_dicom_to_nifti(dicom_root, tmp_nifti_dir):
    os.makedirs(tmp_nifti_dir, exist_ok=True)
    patient_dirs = [os.path.join(dicom_root, d) for d in os.listdir(dicom_root) if os.path.isdir(os.path.join(dicom_root, d))]

    for patient_path in patient_dirs:
        patient_id = os.path.basename(patient_path)
        output_filename = f"patient_{patient_id}.nii.gz"
        output_fullpath = os.path.join(tmp_nifti_dir, output_filename)

        tmp_dir = os.path.join(tmp_nifti_dir, f"tmp_{patient_id}")
        os.makedirs(tmp_dir, exist_ok=True)

        cmd = [
            "dcm2niix",
            "-z", "y",
            "-f", "image",
            "-o", tmp_dir,
            patient_path
        ]

        print(f"[DICOM] Converting: {patient_id}")
        subprocess.run(cmd, check=True)

        converted_file = os.path.join(tmp_dir, "image.nii.gz")
        if os.path.exists(converted_file):
            os.rename(converted_file, output_fullpath)
            print(f"  -> Saved: {output_filename}")
        else:
            print(f"  !! Problem converting: {patient_id}")

        shutil.rmtree(tmp_dir)

    print("[DICOM] Conversion completed.")

# === STEP 2: Rename to nnU-Net compliant (_0000) and move ===

def prepare_nnunet_inputs(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    nifti_files = glob.glob(os.path.join(src_dir, "*.nii.gz"))

    for file_path in nifti_files:
        filename = os.path.basename(file_path)
        patient_id = filename.split("_")[-1].replace(".nii.gz", "")
        new_filename = f"patient_{patient_id}_0000.nii.gz"
        shutil.copy(file_path, os.path.join(dst_dir, new_filename))
        print(f"[NNUNet] Renamed: {filename} → {new_filename}")

# === STEP 3: Clip intensity values to a given HU range ===

def normalize_nii_images(input_folder, output_folder, min_val=-10, max_val=140):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            nii_image = nib.load(input_path)
            img_data = nii_image.get_fdata()
            img_data = np.clip(img_data, min_val, max_val)

            new_nii = nib.Nifti1Image(img_data, affine=nii_image.affine, header=nii_image.header)
            nib.save(new_nii, output_path)

            print(f"[Normalize] Processed: {filename}")

# === STEP 4: Initialize nnU-Net model and run prediction ===

def run_nnunet_prediction(input_dir, output_dir, model_dir):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name="checkpoint_final.pth"
    )

    os.makedirs(output_dir, exist_ok=True)

    print("[NNUNet] Running prediction...")
    predictor.predict_from_files(
        input_dir,
        output_dir,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    print("[NNUNet] Prediction completed.")

# === MAIN EXECUTION ===

if __name__ == "__main__":
    print("=== NNUNet Full Pipeline Started ===")
    
    convert_dicom_to_nifti(dicom_root, tmp_nifti_dir)
    prepare_nnunet_inputs(tmp_nifti_dir, nnunet_input_dir)
    normalize_nii_images(nnunet_input_dir, normalized_input_dir)
    run_nnunet_prediction(normalized_input_dir, prediction_output_dir, model_dir)

    print(f"=== See prediction at {prediction_output_dir} ===")
