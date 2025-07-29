import os

FOLDERS = [
    "config",
    "data/raw/ixi",
    "data/raw/hcp",
    "data/raw/oasis",
    "data/raw/adni",
    "data/processed",
    "data/slices",
    "data/metadata",
    "models",
    "logs",
    "outputs/metrics",
    "outputs/visualizations",
    "outputs/predictions",
    "notebooks",
    "scripts",
    "src/data_loader",
    "src/preprocessing",
    "src/feature_engineering",
    "src/models",
    "src/training",
    "src/evaluation",
    "src/inference",
    "src/utils",
    "tests"
]

FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "config/config.yaml",
    "scripts/run_pipeline.py",
    "scripts/train_model.py",
    "scripts/evaluate_model.py",
    "scripts/predict_image.py",
    "src/__init__.py",
    "src/utils/logger.py",
    "src/utils/seed_everything.py",
    "src/utils/helpers.py",
    "tests/test_data_loader.py",
    "tests/test_model.py",
    "tests/test_preprocessing.py"
]

def create_structure():
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    for file_path in FILES:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("# Placeholder\n")
            print(f"Created file: {file_path}")
        else:
            print(f"File already exists: {file_path}")

if __name__ == "__main__":
    create_structure()
