"""
HAM10000 Dataset Classes - Albumentations ile
- Medikal-optimize augmentation pipeline
- Renk bilgisini KORUYAN işlemler
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd


# Sınıf etiketleri
MULTICLASS_LABELS = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
    'mel': 4, 'nv': 5, 'vasc': 6
}

BINARY_LABELS = {'benign': 0, 'malignant': 1}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size=224, mode='train', use_augmentation=True):
    """
    Albumentations transforms - MEDİKAL OPTİMİZE
    
    mode: 'train' | 'val' | 'test'
    use_augmentation: True ise augmentation uygulanır (sadece train'de)
    
    NOT: Renk bozucu işlemler (Solarize, Posterize, Equalize, Invert) 
         medikal görüntüler için KULLANILMAZ.
    """
    
    if mode == 'train' and use_augmentation:
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # GEOMETRİK AUGMENTATION (renk bozmaz)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
    		translate_percent=(-0.05, 0.05),
    		scale=(0.9, 1.1),
    		rotate=(-15, 15),
   		border_mode=cv2.BORDER_REFLECT,
    		p=0.5
		),
            
            # MEDİKAL ÖZEL (cilt dokusu için)
            A.OneOf([
                A.ElasticTransform(alpha=80, sigma=5, p=0.5),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=0.05, p=1.0),
            ], p=0.3),
            
            # KONTRAST (medikal standart)
            A.OneOf([
                A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.5
                ),
            ], p=0.5),
            
            # HAFİF RENK VARYASYONU (cilt tonu için, çok hafif)
            A.HueSaturationValue(
                hue_shift_limit=8,      # Çok hafif (default 20)
                sat_shift_limit=15,     # Çok hafif (default 30)
                val_shift_limit=10,     # Çok hafif (default 20)
                p=0.4
            ),
            
            # GAUSSIAN NOISE (gerçekçi)
            A.GaussNoise(std_range=(0.04, 0.10), p=0.2),
            
            # CUTOUT (modern teknik)
            A.CoarseDropout(
    		num_holes_range=(1, 3),
   		hole_height_range=(16, 28),
    		hole_width_range=(16, 28),
    		fill=0,
    		p=0.3
		),
            # Normalize ve tensor'a çevir
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        # Val/Test: SADECE resize + normalize
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])


class HAMDataset(Dataset):
    """
    HAM10000 PyTorch Dataset (Albumentations ile)
    
    Args:
        csv_path: CSV dosyası yolu
        task: 'multiclass' veya 'binary'
        transform: Albumentations transform
    """
    def __init__(self, csv_path, task='multiclass', transform=None):
        self.df = pd.read_csv(csv_path)
        self.task = task
        self.transform = transform
        
        if task == 'multiclass':
            self.label_col = 'dx'
            self.label_map = MULTICLASS_LABELS
        elif task == 'binary':
            self.label_col = 'binary'
            self.label_map = BINARY_LABELS
        else:
            raise ValueError(f"task '{task}' geçersiz!")
        
        # NaN path varsa filtrele
        self.df = self.df.dropna(subset=['image_path']).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Resmi OpenCV ile oku (Albumentations uyumlu)
        img = cv2.imread(row['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Etiket
        label = self.label_map[row[self.label_col]]
        
        # Transform uygula
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        return img, label


def get_weighted_sampler(dataset, target_per_class=500):
    """WeightedRandomSampler: azınlık sınıfları daha sık örnekler."""
    labels = []
    for i in range(len(dataset)):
        row = dataset.df.iloc[i]
        label = dataset.label_map[row[dataset.label_col]]
        labels.append(label)
    
    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels)
    weights = 1.0 / class_counts[labels].float()
    num_samples = target_per_class * len(class_counts)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True
    )
    
    print(f"WeightedRandomSampler hazır:")
    print(f"  Sınıf sayıları: {class_counts.tolist()}")
    print(f"  Hedef per class: {target_per_class}")
    print(f"  Toplam sample/epoch: {num_samples}")
    
    return sampler


def get_class_names(task='multiclass'):
    if task == 'multiclass':
        return ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    else:
        return ['benign', 'malignant']


def get_num_classes(task='multiclass'):
    return 7 if task == 'multiclass' else 2