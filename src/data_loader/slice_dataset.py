import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class SliceDataset(Dataset):
    def __init__(self, image_dir: str, metadata_csv: str):
        self.df = pd.read_csv(metadata_csv)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("L")
        label = int(row['label'])
        return self.transform(image), label
