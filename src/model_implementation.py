# /src/model_implementation.py
import torch
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights, ResNeXt101_32X8D_Weights

from dataset_pipeline import CLASSES

# -----------------------------------------------------------------------------
# 1. Fabryki modeli
# -----------------------------------------------------------------------------
def _build_backbone(
    name: str = "resnext101_32x8d",
    pretrained: bool = True,
    trainable_layers: int = 5
):
    weight_map = {
        "resnet50": ResNet50_Weights.IMAGENET1K_V1,
        "resnet101": ResNet101_Weights.IMAGENET1K_V2,
        "resnext101_32x8d": ResNeXt101_32X8D_Weights.IMAGENET1K_V2,
    }
    weights = weight_map[name] if pretrained else None
    return resnet_fpn_backbone(
        backbone_name=name,
        weights=weights,
        trainable_layers=trainable_layers
    )

def get_faster_rcnn(num_classes: int = len(CLASSES), name_backbone: str = 'resnext101_32x8d', layers_backbone=5, pretrained_backbone=True):
    backbone = _build_backbone(name=name_backbone, trainable_layers=layers_backbone, pretrained=pretrained_backbone)
    model = FasterRCNN(backbone, num_classes=num_classes)

    # predictor (torchvision >=0.15 nie robi tego automatycznie)
    infeat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(infeat, num_classes)
    return model

def get_mask_rcnn(num_classes: int = len(CLASSES), name_backbone: str = 'resnext101_32x8d', layers_backbone=5, pretrained_backbone=True):
    backbone = _build_backbone(name=name_backbone, trainable_layers=layers_backbone, pretrained=pretrained_backbone)
    model = MaskRCNN(backbone, num_classes=num_classes)
    # box‑predictor
    infeat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(infeat, num_classes)
    # mask‑predictor
    inmask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(inmask, 256, num_classes)
    return model

def initialize_model(model_type="mask_rcnn",
                     num_classes=len(CLASSES),
                     name_backbone='resnext101_32x8d',
                     layers_backbone=5,
                     pretrained_backbone=True):
    if model_type == "faster_rcnn":
        return get_faster_rcnn(num_classes, name_backbone, layers_backbone, pretrained_backbone)
    if model_type == "mask_rcnn":
        return get_mask_rcnn(num_classes, name_backbone, layers_backbone, pretrained_backbone)
    raise ValueError(f"Unknown model_type: {model_type}")

# -----------------------------------------------------------------------------
# 2. helpery IO
# -----------------------------------------------------------------------------
def save_model(model, path: str):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved -> {path}")

def load_model(model_type, path, num_classes=len(CLASSES)):
    model = initialize_model(model_type=model_type, num_classes=num_classes, pretrained_backbone=False)
    model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
    model.eval()
    return model
