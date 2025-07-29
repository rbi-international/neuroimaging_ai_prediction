import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger("Preprocessing", "logs/preprocessing.log")

def normalize_volume(volume: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize MRI volume using z-score or min-max scaling."""
    if method == "zscore":
        mean, std = np.mean(volume), np.std(volume)
        volume = (volume - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val, max_val = np.min(volume), np.max(volume)
        volume = (volume - min_val) / (max_val - min_val + 1e-8)
    else:
        logger.warning(f"Unknown normalization method: {method}")
    return volume

def resize_volume(volume: np.ndarray, target_shape=(128, 128, 128)) -> np.ndarray:
    """Resize volume to target shape using interpolation."""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized = zoom(volume, factors, order=3)  # spline interpolation
    return resized

def preprocess_and_save(volume: np.ndarray, save_path: str, norm_method="zscore"):
    """Preprocess (normalize + resize) and save to NIfTI format."""
    try:
        volume = normalize_volume(volume, norm_method)
        volume = resize_volume(volume)
        ensure_dir(os.path.dirname(save_path))

        nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), save_path)
        logger.info(f"Saved preprocessed volume → {save_path}")
    except Exception as e:
        logger.error(f"Failed to preprocess & save volume: {e}")
