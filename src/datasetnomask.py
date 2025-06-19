import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# how to load a single sample (i.e. one patient’s MRI-CT pair across views). But deep learning models train in batches.
class PairedMultiViewDataset(Dataset):
    def __init__(self, root_dir: str, split_file: str, transform: Optional[transforms.Compose] = None):
        """
        Dataset for loading paired multi-view grayscale MRI and CT images.

        Args:
            root_dir (str): Directory containing patient folders with MRI and CT images.
            split_file (str): File containing list of patient IDs.
            transform (optional): Transformations to apply to each image.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.views = ['axial', 'coronal', 'sagittal']

        with open(split_file, 'r') as f:
            self.patient_ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_id = self.patient_ids[idx]
        patient_path = self.root_dir / patient_id

        mri_views = []
        ct_views = []

        for view in self.views:
            mri_path = patient_path / f"mr.nii_{view}.png"
            ct_path = patient_path / f"ct.nii_{view}.png"

            # Load as grayscale
            mri_img = Image.open(mri_path).convert("L")
            ct_img = Image.open(ct_path).convert("L")

            # Apply transforms
            mri_tensor = self.transform(mri_img)
            ct_tensor = self.transform(ct_img)

            mri_views.append(mri_tensor)
            ct_views.append(ct_tensor)

        # Stack to shape: [3, H, W] → each channel is a different view
        mri_stack = torch.stack(mri_views)
        ct_stack = torch.stack(ct_views)

        return mri_stack, ct_stack
