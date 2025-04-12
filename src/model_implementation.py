import torch
import torchvision
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# Importy klas i funkcji z poprzedniego skryptu
from src.dataset_preparation import CLASSES

def get_faster_rcnn_model(num_classes=len(CLASSES), pretrained_backbone=True):
    """
    Tworzy model Faster R-CNN z predefiniowanym backbonem.
    
    Args:
        num_classes: Liczba klas (włączając tło)
        pretrained_backbone: Czy używać pretrenowanego backbonu
    
    Returns:
        Model Faster R-CNN
    """
    # Tworzymy backbone oparty na ResNet-50-FPN
    backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone)
    
    # Tworzymy model Faster R-CNN
    model = FasterRCNN(backbone, num_classes=num_classes)
    
    # Możemy dostroić parametry modelu
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features,
        num_classes
    )
    
    return model

def get_mask_rcnn_model(num_classes=len(CLASSES), pretrained_backbone=True):
    """
    Tworzy model Mask R-CNN z predefiniowanym backbonem.
    
    Args:
        num_classes: Liczba klas (włączając tło)
        pretrained_backbone: Czy używać pretrenowanego backbonu
    
    Returns:
        Model Mask R-CNN
    """
    # Tworzymy backbone oparty na ResNet-50-FPN
    backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone)
    
    # Tworzymy model Mask R-CNN
    model = MaskRCNN(backbone, num_classes=num_classes)
    
    # Dostrajamy predyktor pudełek (box predictor)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    
    # Dostrajamy predyktor masek (mask predictor)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model

def initialize_model(model_type="mask_rcnn", num_classes=len(CLASSES), pretrained_backbone=True):
    """
    Inicjalizuje model wybranego typu.
    
    Args:
        model_type: Typ modelu ('faster_rcnn' lub 'mask_rcnn')
        num_classes: Liczba klas (włączając tło)
        pretrained_backbone: Czy używać pretrenowanego backbonu
    
    Returns:
        Zainicjalizowany model
    """
    if model_type == "faster_rcnn":
        model = get_faster_rcnn_model(num_classes, pretrained_backbone)
    elif model_type == "mask_rcnn":
        model = get_mask_rcnn_model(num_classes, pretrained_backbone)
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")
    
    return model

# Dodatkowe funkcje pomocnicze do manipulacji modelami

def save_model(model, filename):
    """
    Zapisuje model do pliku.
    
    Args:
        model: Model do zapisania
        filename: Nazwa pliku
    """
    torch.save(model.state_dict(), filename)
    print(f"Model zapisany do {filename}")

def load_model(model_type, filename, num_classes=len(CLASSES)):
    """
    Wczytuje model z pliku.
    
    Args:
        model_type: Typ modelu ('faster_rcnn' lub 'mask_rcnn')
        filename: Nazwa pliku
        num_classes: Liczba klas (włączając tło)
    
    Returns:
        Wczytany model
    """
    model = initialize_model(model_type, num_classes, pretrained_backbone=False)
    model.load_state_dict(torch.load(filename))
    return model

if __name__ == "__main__":
    # Testujemy inicjalizację modeli
    faster_rcnn = initialize_model("faster_rcnn")
    mask_rcnn = initialize_model("mask_rcnn")
    
    print("Model Faster R-CNN:")
    print(faster_rcnn)
    
    print("\nModel Mask R-CNN:")
    print(mask_rcnn)
    
    # Sprawdzamy liczbę parametrów w modelach
    faster_rcnn_params = sum(p.numel() for p in faster_rcnn.parameters() if p.requires_grad)
    mask_rcnn_params = sum(p.numel() for p in mask_rcnn.parameters() if p.requires_grad)
    
    print(f"\nLiczba parametrów w Faster R-CNN: {faster_rcnn_params:,}")
    print(f"Liczba parametrów w Mask R-CNN: {mask_rcnn_params:,}")
    
    # Przygotowujemy przykładowe dane do testowania modeli
    dummy_image = torch.rand(3, 640, 480)
    dummy_target = {
        'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'masks': torch.zeros((1, 640, 480), dtype=torch.uint8)
    }
    
    # Testujemy inferencję modelu w trybie ewaluacji
    faster_rcnn.eval()
    with torch.no_grad():
        predictions = faster_rcnn([dummy_image])
    
    print("\nPrzykładowe predykcje z Faster R-CNN:")
    print(f"Liczba wykrytych obiektów: {len(predictions[0]['boxes'])}")