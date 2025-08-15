import bittensor as bt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
from typing import List, Tuple, Dict
import random
import numpy as np
import io # Added for FeedbackDataset

# Import label mappings from model.py to ensure consistency
from FruitsMaturityNet.model import FRUIT_TYPE_REV_MAP, RIPENESS_REV_MAP, FRUIT_TYPE_MAP, RIPENESS_MAP

class FruitDataset(Dataset):
    """
    Custom Dataset for the Kaggle Fruits dataset.
    Assumes the dataset is unzipped into 'dataset/training' and 'dataset/test'.
    Folder names like 'freshapples', 'rottenapples' contain both fruit type and ripeness info.
    """
    def __init__(self, root_dir: str, train: bool = True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths: List[str] = []
        self.labels: List[Tuple[int, int]] = [] # (fruit_type_id, ripeness_id)

        # Determine the dataset split folder
        data_folder = os.path.join(root_dir, 'training' if train else 'test')
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Dataset folder not found: {data_folder}. Please run data_prep.sh first.")

        # Collect image paths and assign labels
        for folder_name in os.listdir(data_folder):
            folder_path = os.path.join(data_folder, folder_name)
            if os.path.isdir(folder_path):
                # Parse folder name to extract fruit type and ripeness
                fruit_type, ripeness = self._parse_folder_name(folder_name)

                print(f"fruit type: {fruit_type}, ripeness: {ripeness}, folder_name: {folder_name}")
                
                fruit_type_id = FRUIT_TYPE_REV_MAP.get(fruit_type, -1)
                ripeness_id = RIPENESS_REV_MAP.get(ripeness, -1)
                
                if fruit_type_id == -1 or ripeness_id == -1:
                    print(f"Warning: Skipping unknown fruit/ripeness: {folder_name} -> {fruit_type}/{ripeness}")
                    continue

                for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                    for img_name in glob.glob(os.path.join(folder_path, ext)):
                        self.image_paths.append(img_name)
                        self.labels.append((fruit_type_id, ripeness_id))

        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # EfficientNet-B0 expects 224x224 input
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _parse_folder_name(self, folder_name: str) -> Tuple[str, str]:
        """
        Parse folder names like 'freshapples', 'rottenapples' into fruit type and ripeness.
        """
        folder_name = folder_name.lower()
        
        # Map folder prefixes to ripeness
        if folder_name.startswith('fresh'):
            ripeness = "Fresh"
            fruit_part = folder_name[5:]  # Remove 'fresh'
        elif folder_name.startswith('rotten'):
            ripeness = "Rotten"
            fruit_part = folder_name[6:]  # Remove 'rotten'
        else:
            # Default case
            ripeness = "Normal"
            fruit_part = folder_name
        
        # Map fruit part to fruit type
        fruit_type_mapping = {
            'apples': 'Apple',
            'banana': 'Banana', 
            'oranges': 'Orange'
        }
        
        fruit_type = fruit_type_mapping.get(fruit_part, fruit_part.capitalize())
        
        return fruit_type, ripeness

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        fruit_type_id, ripeness_id = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "fruit_type_label": torch.tensor(fruit_type_id, dtype=torch.long),
            "ripeness_label": torch.tensor(ripeness_id, dtype=torch.long),
            "image_path": img_path # Useful for debugging/feedback
        }

class FeedbackDataset(Dataset):
    """
    Dataset for fine-tuning based on accumulated feedback.
    """
    def __init__(self, feedback_data: List[Dict], transform=None):
        self.feedback_data = feedback_data
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.feedback_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.feedback_data[idx]
        image_bytes = item["image_bytes"]
        fruit_type_label_str = item["true_fruit_type"]
        ripeness_label_str = item["true_ripeness"]

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = self.transform(image)

        fruit_type_id = FRUIT_TYPE_REV_MAP.get(fruit_type_label_str, -1)
        ripeness_id = RIPENESS_REV_MAP.get(ripeness_label_str, -1)

        if fruit_type_id == -1 or ripeness_id == -1:
            raise ValueError(f"Invalid label in feedback: {fruit_type_label_str}, {ripeness_label_str}")

        return {
            "image": image,
            "fruit_type_label": torch.tensor(fruit_type_id, dtype=torch.long),
            "ripeness_label": torch.tensor(ripeness_id, dtype=torch.long),
        }
