import os
import sys
import torch
from torch.utils.data import DataLoader

# Ensure root path for src import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader.slice_dataset import SliceDataset
from src.models.cnn_2d import SimpleCNN2D
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model
from src.utils.helpers import load_config
from src.utils.seed_everything import seed_everything
from src.utils.logger import get_logger

logger = get_logger("Pipeline", "logs/run_pipeline.log")

def run_pipeline():
    # Load config and seed
    cfg = load_config("config/config.yaml")
    seed_everything(cfg['train']['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Using device: {device}")

    # Load dataset
    dataset_name = cfg['data']['dataset_name']
    logger.info(f"📦 Dataset: {dataset_name}")

    dataset = SliceDataset(
        image_dir=cfg['data']['slices_dir'],
        metadata_csv=cfg['data']['metadata_csv']
    )

    train_loader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=True)

    # Initialize model
    model = SimpleCNN2D(input_channels=1, num_classes=cfg['model']['num_classes'])

    # Train model
    train_model(model, train_loader, cfg, device)

    # Evaluate model
    acc, f1, report = evaluate_model(
        model=model,
        dataloader=train_loader,
        class_names=["Healthy", "Disease"],
        device=device,
        config=cfg,
        output_dir=cfg['output']['metrics_dir']
    )

    logger.info(f"✅ Final Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")

if __name__ == "__main__":
    run_pipeline()
