import pandas as pd
from datetime import datetime
import os
from src.utils.logger import get_logger

logger = get_logger("ReportGenerator", "logs/report_generator.log")

def generate_summary_report(results_csv: str, output_dir: str = "outputs/reports", top_k: int = 3):
    if not os.path.exists(results_csv):
        logger.error(f"❌ Results file not found: {results_csv}")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)

    if df.empty:
        logger.warning("⚠️ Results file is empty. No report generated.")
        return

    # Sort by accuracy descending
    df_sorted = df.sort_values(by="accuracy", ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"summary_report_{timestamp}.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("📊 Summary Report: Neuroimaging Disease Prediction\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Runs: {len(df)}\n\n")

        for i, row in df_sorted.head(top_k).iterrows():
            f.write(f"Rank #{i+1}\n")
            f.write(f"Dataset      : {row['dataset']}\n")
            f.write(f"Accuracy     : {row['accuracy']:.4f}\n")
            f.write(f"Precision    : {row['precision']:.4f}\n")
            f.write(f"Recall       : {row['recall']:.4f}\n")
            f.write(f"F1 Score     : {row['f1']:.4f}\n")
            f.write(f"Model Path   : {row['model_path']}\n")
            f.write("-" * 40 + "\n")

    logger.info(f"✅ Report saved to {summary_path}")
