import os
import torch
from torch.utils.data import DataLoader

from src.utils.helpers import load_config
from src.utils.logger import get_logger
from src.data_loader.slice_dataset import SliceDataset
from src.models.cnn_2d import SimpleCNN2D
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model
from src.visualization.metrics_visualizer import plot_metrics
from src.utils.comparison_table_generator import generate_comparison_table

logger = get_logger("Pipeline", "logs/run_pipeline.log")

def main():
    cfg = load_config("config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Using device: {device}")

    # Load dataset
    dataset = SliceDataset(
        image_dir=cfg['data']['slices_dir'],
        metadata_csv=cfg['data']['metadata_csv']
    )
    dataloader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=True)

    # Initialize model
    model = SimpleCNN2D(input_channels=1, num_classes=cfg['model']['num_classes'])
    model.to(device)

    # Train
    model, train_loss_log = train_model(model, dataloader, cfg, device)

    # Evaluate
    class_names = sorted(dataset.df['label'].unique().tolist())
    acc, f1, report = evaluate_model(model, dataloader, class_names, device, cfg)

    # Visualize
    plot_metrics(train_loss_log, cfg['data']['dataset_name'])

    # Generate comparison
    generate_comparison_table()

    logger.info("🏁 Full pipeline execution completed!")

if __name__ == "__main__":
    main()
