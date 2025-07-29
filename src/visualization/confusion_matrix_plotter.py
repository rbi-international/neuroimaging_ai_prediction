import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    return save_path


def print_classification_report(y_true, y_pred, class_names):
    print("📊 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    return report
