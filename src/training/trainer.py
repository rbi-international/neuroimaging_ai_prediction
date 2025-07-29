import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.visualization.metrics_visualizer import plot_and_save_metrics
from src.utils.generate_report import generate_report
from src.utils.logger import get_logger

logger = get_logger("Trainer")

def train_model(model, train_dataset, val_dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Using device: {device}")

    # Load hyperparameters
    lr = config['train']['lr']
    epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    save_dir = config['train']['save_dir']
    dataset_name = config['data']['dataset_name']

    os.makedirs(save_dir, exist_ok=True)

    # Prepare loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        correct_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / len(train_dataset)

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / len(val_dataset)

        logger.info(f"📦 Epoch {epoch}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, f"{dataset_name}_model.pth"))
    logger.info(f"✅ Model saved to {os.path.join(save_dir, f'{dataset_name}_model.pth')}")

    # Save metrics plots
    plot_and_save_metrics(
        train_losses, val_losses,
        train_accs, val_accs,
        save_dir=os.path.join("outputs", "metrics"),
        dataset_name=dataset_name
    )
    
    # After training
    generate_report(
    train_losses=train_losses,
    val_losses=val_losses,
    train_accs=train_accs,
    val_accs=val_accs,
    save_dir=os.path.join("outputs", "reports"),
    dataset_name=dataset_name
)       
