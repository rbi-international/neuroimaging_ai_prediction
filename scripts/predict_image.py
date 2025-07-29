import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

from src.models.cnn_2d import SimpleCNN2D
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger("Predictor", "logs/predict.log")

def load_model(checkpoint_path, num_classes, device):
    model = SimpleCNN2D(input_channels=1, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path}")
    return model

def predict_image(model, image_path, class_names, device):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_class.item()]
    logger.info(f"Prediction: {predicted_label} | Confidence: {confidence.item():.2f}")
    return predicted_label, confidence.item()

