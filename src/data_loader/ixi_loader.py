import os
import nibabel as nib
import numpy as np
from glob import glob
from src.utils.logger import get_logger

logger = get_logger("IXI Loader", "logs/ixi_loader.log")

def load_nifti_file(file_path: str) -> np.ndarray:
    """Load a NIfTI MRI file and return a NumPy volume."""
    try:
        img = nib.load(file_path)
        volume = img.get_fdata()
        logger.info(f"Loaded: {file_path} | Shape: {volume.shape}")
        return volume
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def get_all_mri_paths(data_dir: str, extension="*.nii.gz") -> list:
    """Get list of all NIfTI files in the IXI data folder."""
    paths = glob(os.path.join(data_dir, extension))
    logger.info(f"Found {len(paths)} NIfTI files in {data_dir}")
    return paths

def load_all_volumes(data_dir: str) -> list:
    """Load all MRI volumes as NumPy arrays from directory."""
    paths = get_all_mri_paths(data_dir)
    volumes = []
    for path in paths:
        vol = load_nifti_file(path)
        if vol is not None:
            volumes.append(vol)
    logger.info(f"Loaded {len(volumes)} valid volumes")
    return volumes
