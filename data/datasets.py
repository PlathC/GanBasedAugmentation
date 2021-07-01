import os
from PIL import Image
import random
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple


class CustomDataset(Dataset):
    def __init__(self,
                 class_folders: Dict[str, List[str]],
                 transform=None):
        super().__init__()
        self.class_folders = class_folders
        self.transform = transform

        self._x = []
        self._y = []
        for class_name in self.class_folders.keys():
            for folder in self.class_folders[class_name]:
                for path, dirs, files in os.walk(folder):
                    self._x += [os.path.join(path, file) for file in files]
                self._y += [class_name] * (len(self._x) - len(self._y))

        self.encoder = LabelEncoder()
        self._y = self.encoder.fit_transform(self._y)

    @property
    def classes(self) -> List[str]:
        return self.encoder.classes_

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self._x[idx]
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self._y[idx])
