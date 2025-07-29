import torch
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir, save_json

logger = get_logger("Evaluator", "logs/evaluation.log")

def evaluate_model(model, dataloader, class_names, device, config):
    """
    Evaluates the model on a given dataloader and saves:
    - classification_report as .json, .csv, .tex
    - confusion_matrix as PNG
    Returns a dictionary with metrics and paths.
    """

    model.eval()
    y_true, y_pred = [], []
    dataset_name = config['data']['dataset_name']
    output_dir = os.path.join(config['output']['metrics_dir'], dataset_name)
    ensure_dir(output_dir)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 📊 Basic Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # 🧾 Log results
    logger.info(f"📊 Dataset: {dataset_name} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

    # 🧾 Save reports
    report_json_path = os.path.join(output_dir, f"classification_report_{dataset_name}.json")
    report_csv_path = os.path.join(output_dir, f"classification_report_{dataset_name}.csv")
    report_tex_path = os.path.join(output_dir, f"classification_report_{dataset_name}.tex")
    cm_png_path = os.path.join(output_dir, f"confusion_matrix_{dataset_name}.png")

    save_json(report_dict, report_json_path)

    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(report_csv_path)
    with open(report_tex_path, "w") as f:
        f.write(df.to_latex(float_format="%.2f"))

    # 🔵 Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_png_path)
    plt.close()

    return {
        "accuracy": acc,
        "f1_score": f1,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "paths": {
            "report_json": report_json_path,
            "report_csv": report_csv_path,
            "report_tex": report_tex_path,
            "confusion_matrix": cm_png_path,
        }
    }
