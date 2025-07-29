import os
import matplotlib.pyplot as plt
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("MetricsVisualizer", "logs/metrics_visualizer.log")

def plot_metrics(train_loss_log, dataset_name, output_dir="outputs/visualizations/"):
    """
    Plots training loss curve and saves it.
    """
    ensure_dir(output_dir)

    if not train_loss_log:
        logger.warning("⚠️ No training loss data provided to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_log, marker='o')
    plt.title(f"Training Loss Curve - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"loss_curve_{dataset_name}.png")
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"📉 Saved loss curve plot at: {plot_path}")
