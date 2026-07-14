"""
Pretrained Models - Transfer Learning
- ResNet50
- EfficientNet-B0
- DenseNet121

ImageNet'te pretrained, son katman değiştirilir (num_classes'a göre).
Tüm ağırlıklar fine-tune edilir.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_resnet50(num_classes=7, pretrained=True):
    """ResNet50 - klasik, güvenilir baseline"""
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50(weights=None)
    
    # Son fully connected layer'ı değiştir
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def get_efficientnet_b0(num_classes=7, pretrained=True):
    """EfficientNet-B0 - modern, hafif, güçlü"""
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Classifier son katmanı değiştir
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


def get_densenet121(num_classes=7, pretrained=True):
    """DenseNet121 - medikal görüntülemede popüler"""
    if pretrained:
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        model = models.densenet121(weights=None)
    
    # Classifier'ı değiştir
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model

def get_efficientnet_b3(num_classes=7, pretrained=True):
    """EfficientNet-B3 - daha büyük, daha güçlü (12M param)"""
    if pretrained:
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b3(weights=None)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

def get_efficientnet_b4(num_classes=7, pretrained=True):
    """EfficientNet-B4 - daha büyük, 19M param"""
    if pretrained:
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b4(weights=None)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


def get_convnext_tiny(num_classes=7, pretrained=True):
    """ConvNeXt-Tiny - modern CNN (2022), 28M param"""
    if pretrained:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    else:
        model = models.convnext_tiny(weights=None)
    
    # ConvNeXt classifier yapısı: [LayerNorm2d, Flatten, Linear]
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    return model


def count_parameters(model):
    """Eğitilebilir parametre sayısı"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Model factory - kolay erişim için
MODEL_FACTORY = {
    'resnet50': get_resnet50,
    'efficientnet_b0': get_efficientnet_b0,
    'efficientnet_b3': get_efficientnet_b3,
    'efficientnet_b4': get_efficientnet_b4,    # ← YENİ
    'convnext_tiny': get_convnext_tiny,        # ← YENİ
    'densenet121': get_densenet121,
}
def get_model(name, num_classes=7, pretrained=True):
    """
    Model adına göre döndürür.
    
    Args:
        name: 'resnet50', 'efficientnet_b0', 'densenet121'
        num_classes: Sınıf sayısı
        pretrained: ImageNet pretrained ağırlıkları
    """
    if name not in MODEL_FACTORY:
        raise ValueError(f"Bilinmeyen model: {name}. Mevcut: {list(MODEL_FACTORY.keys())}")
    return MODEL_FACTORY[name](num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    # Test
    for name in MODEL_FACTORY:
        model = get_model(name, num_classes=7)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"{name:20s} | Params: {count_parameters(model):>12,} | Out: {out.shape}")