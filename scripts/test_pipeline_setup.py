import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))


# Import from our modular pipeline
from src.utils.helpers import load_config
from src.utils.logger import get_logger
from src.data_loader.slice_dataset import SliceDataset
from src.models.cnn_2d import SimpleCNN2D
from src.training import trainer
from src.evaluation import evaluator

logger = get_logger("TestScript", "logs/test_pipeline_setup.log")

def run_tests():
    logger.info("Starting test: Config + Dataset + Model")

    # Load config
    cfg = load_config(config_path)
    assert "data" in cfg and "model" in cfg and "train" in cfg

    # Load dataset
    dataset = SliceDataset(
        image_dir=cfg['data']['slices_dir'],
        metadata_csv=cfg['data']['metadata_csv']
    )

    assert len(dataset) > 0, "Dataset is empty!"
    logger.info(f"Loaded {len(dataset)} samples from {cfg['data']['dataset_name']}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    sample_batch = next(iter(dataloader))
    images, labels = sample_batch
    logger.info(f"Batch shape: {images.shape}, Labels: {labels.tolist()}")

    # Test model
    model = SimpleCNN2D(input_channels=1, num_classes=cfg['model']['num_classes'])
    output = model(images)
    assert output.shape[0] == images.shape[0], "Model output shape mismatch"

    logger.info("Forward pass successful")

    # Import train/eval functions
    assert hasattr(trainer, "train_model"), "train_model not found"
    assert hasattr(evaluator, "evaluate_model"), "evaluate_model not found"

    logger.info("Trainer and Evaluator modules import successfully")

    logger.info("All components are working! You're ready to run the full pipeline.")

if __name__ == "__main__":
    run_tests()
