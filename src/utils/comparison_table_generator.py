import os
import json
import pandas as pd
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("ComparisonTable", "logs/comparison_table.log")

def generate_comparison_table(metrics_dir="outputs/metrics/", output_path="outputs/metrics/comparison_table.csv"):
    """
    Aggregates classification reports for different datasets and generates a comparison table.
    """
    ensure_dir(metrics_dir)
    rows = []

    for file in os.listdir(metrics_dir):
        if file.startswith("classification_report_") and file.endswith(".json"):
            dataset_name = file.split("_")[-1].replace(".json", "")
            with open(os.path.join(metrics_dir, file), "r") as f:
                report = json.load(f)

            accuracy = report.get("accuracy", None)
            weighted_f1 = report.get("weighted avg", {}).get("f1-score", None)
            macro_f1 = report.get("macro avg", {}).get("f1-score", None)

            rows.append({
                "Dataset": dataset_name,
                "Accuracy": round(accuracy, 4) if accuracy else "N/A",
                "Macro F1": round(macro_f1, 4) if macro_f1 else "N/A",
                "Weighted F1": round(weighted_f1, 4) if weighted_f1 else "N/A"
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Dataset")

    # Save as CSV
    df.to_csv(output_path, index=False)
    logger.info(f"📋 Saved comparison table to {output_path}")

    # Save as LaTeX
    df.to_latex(output_path.replace(".csv", ".tex"), index=False, float_format="%.4f")
    logger.info(f"📄 Saved LaTeX version to {output_path.replace('.csv', '.tex')}")

    # Save as Markdown
    with open(output_path.replace(".csv", ".md"), "w") as f:
        f.write(df.to_markdown(index=False))
    logger.info(f"📝 Saved Markdown table to {output_path.replace('.csv', '.md')}")
