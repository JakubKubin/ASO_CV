# /src/model_implementation.py
import torch, torchvision
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights

from dataset_pipeline import CLASSES

# -----------------------------------------------------------------------------
# 1. Fabryki modeli
# -----------------------------------------------------------------------------
def _build_backbone(pretrained: bool = True):
    """ResNet-50 + FPN backbone z użyciem nowego API weights."""
    # Wybór weights dla nowego API
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    return resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=weights,
        trainable_layers=5
    )

def get_faster_rcnn(num_classes: int = len(CLASSES), pretrained_backbone=True):
    backbone = _build_backbone(pretrained_backbone)
    model = FasterRCNN(backbone, num_classes=num_classes)

    # predictor (torchvision >=0.15 nie robi tego automatycznie)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model

def get_mask_rcnn(num_classes: int = len(CLASSES), pretrained_backbone=True):
    backbone = _build_backbone(pretrained_backbone)
    model = MaskRCNN(backbone, num_classes=num_classes)
    # box‑predictor
    infeat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(infeat, num_classes)
    # mask‑predictor
    inmask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(inmask, 256, num_classes)
    return model

def initialize_model(model_type="faster_rcnn",
                     num_classes=len(CLASSES),
                     pretrained_backbone=True):
    if model_type == "faster_rcnn":
        return get_faster_rcnn(num_classes, pretrained_backbone)
    if model_type == "mask_rcnn":
        return get_mask_rcnn(num_classes, pretrained_backbone)
    raise ValueError(f"Unknown model_type: {model_type}")

# -----------------------------------------------------------------------------
# 2. helpery IO
# -----------------------------------------------------------------------------
def save_model(model, path: str):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved -> {path}")

def load_model(model_type, path, num_classes=len(CLASSES)):
    model = initialize_model(model_type, num_classes, pretrained_backbone=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
