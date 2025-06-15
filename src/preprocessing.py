import os
import numpy as np
import nibabel as nib  # for processing .nii and .nii.gz files (neuroimaging file formats)
import matplotlib.pyplot as plt  # To save slices as PNG images
from pathlib import Path  #  For safer path handling
from scipy.ndimage import zoom # To resize 2D slices
from tqdm import tqdm  # For displaying progress bars when iterating over files

input_dir="C:/Users/joshi/OneDrive/Desktop/Medical/MLHealthcare/data/raw"
output_dir="C:/Users/joshi/OneDrive/Desktop/Medical/MLHealthcare/data/processed"

# Min-Max Normalization Function(Good practice in neuroimaging)
def min_max_normalize(volume):
    # Computes the minimum voxel intensity value in the entire 3D volume.
    min_val=np.min(volume)
    max_val=np.max(volume)
    if max_val-min_val ==0:
        # returns an array of all zeros with the same shape and type as the input volume
        return np.zeros_like(volume)
    # This scales the values to lie between 0 and 1
    return (volume-min_val)/(max_val-min_val)

# Resize Slice Function
def resize_slice(slice_2d,target_size=(256,256)): # Default target size is (256,256)
    zoom_factors=(target_size[0]/slice_2d.shape[0],target_size[1]/slice_2d.shape[1])
    return zoom(slice_2d,zoom_factors)

# Save slices as png
def save_slices_as_png(volume,patient_id,scan_type,output_base):
    volume=min_max_normalize(volume)

    x=volume.shape[0]//2
    y=volume.shape[1]//2
    z=volume.shape[2]//2

    # Data is according to NIfTI RAS convention
    # Rotating by 90 anticlockwise gives good image views 
    # NIfTI data may be stored in a way that requires rotation for proper radiological view (left-right orientation)
    # For example, a sagittal slice might appear "lying down" without rotation.

    axial=resize_slice(np.rot90(volume[:,:,z]))
    coronal=resize_slice(np.rot90(volume[:,y,:]))
    sagittal=resize_slice(np.rot90(volume[x,:,:]))

    os.makedirs(output_base,exist_ok=True)
    # plt.imsave(path,image(numpy2d),cmap="gray")
    plt.imsave(f"{output_base}/{scan_type}_axial.png", axial, cmap='gray')
    plt.imsave(f"{output_base}/{scan_type}_coronal.png", coronal, cmap='gray')
    plt.imsave(f"{output_base}/{scan_type}_sagittal.png", sagittal, cmap='gray')

# Process All Patient Folders
def process_all_patients(input_dir,output_dir):
    for patient_folder in tqdm(os.listdir(input_dir)):
        patient_path = os.path.join(input_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for file in os.listdir(patient_path):
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(patient_path, file)
                scan_type = Path(file).stem

                try:
                    img = nib.load(file_path)
                    data = img.get_fdata()
                    save_slices_as_png(data, patient_id=patient_folder, scan_type=scan_type,
                                       output_base=os.path.join(output_dir, patient_folder))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

## To start preprocessing:
## This is top level code and can run when the script is imported
## use this thing: You use this block to protect executable code (like running preprocessing or training models) so it doesn't run automatically when you import the file somewhere else.

# Every Python file has a built-in variable called __name__

# If the file is run directly, Python sets __name__ = "__main__"

# If the file is imported into another file, then __name__ = "filename"

# Top-level code in Python is any code that is not:

# inside a function (def)

# inside a class (class)

# inside an if __name__ == "__main__": block


# It runs immediately when the Python file is:

# executed as a script (python myfile.py), or

# imported from another script or notebook.

if __name__=="__main__":
    process_all_patients(input_dir,output_dir)
