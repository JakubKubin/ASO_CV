# /src/evaluation.py
import time
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

from dataset_pipeline import CLASSES

def calculate_ap(recall, precision):
    """
    Oblicza średnią precyzję (Average Precision) na podstawie krzywej precyzja-czułość.
    """
    # Dodaj punkt (0, 1) do początku krzywej
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Oblicz pole pod krzywą precyzja-czułość
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Oblicz indeksy gdzie recall zmienia się
    indices = np.where(recall[1:] != recall[:-1])[0]

    # Oblicz pole pod krzywą
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap

def calculate_map(predictions: list, targets: list, iou_threshold: float = 0.5) -> tuple[float, dict]:
    """
    Compute mean Average Precision (mAP) and per-class AP.
    predictions: list of dicts with keys 'boxes', 'scores', 'labels'
    targets:     list of dicts with keys 'boxes', 'labels'
    Returns mAP and dict[class_name -> AP]
    """
    n_classes = len(CLASSES)
    ap_per_class = {cls_id: [] for cls_id in range(1, n_classes)}

    # Iterate images
    for preds, targs in zip(predictions, targets):
        pred_boxes = preds['boxes'].cpu()
        pred_scores = preds['scores'].cpu()
        pred_labels = preds['labels'].cpu()
        true_boxes = targs['boxes'].cpu()
        true_labels = targs['labels'].cpu()

        # For each class (skip background at 0)
        for cls_id in range(1, n_classes):
            # Mask by class
            p_idx = (pred_labels == cls_id).nonzero(as_tuple=True)[0]
            t_idx = (true_labels == cls_id).nonzero(as_tuple=True)[0]
            if len(t_idx) == 0:
                continue
            if len(p_idx) == 0:
                ap_per_class[cls_id].append(0.0)
                continue
            # Sort preds by score
            scores = pred_scores[p_idx]
            order = torch.argsort(scores, descending=True)
            p_idx = p_idx[order]
            # True pos / false pos markers
            tp = np.zeros(len(p_idx))
            fp = np.zeros(len(p_idx))
            matched = np.zeros(len(t_idx), dtype=bool)
            # Evaluate each pred
            for i, pi in enumerate(p_idx):
                pb = pred_boxes[pi].unsqueeze(0)
                ious = calculate_iou_batch(pb, true_boxes[t_idx])  # [1, T]
                max_iou, max_j = torch.max(ious, dim=1)
                if max_iou >= iou_threshold and not matched[max_j.item()]:
                    tp[i] = 1
                    matched[max_j.item()] = True
                else:
                    fp[i] = 1
            # Cumulative
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-10)
            recalls = tp_cum / len(t_idx)
            ap = calculate_ap(recalls, precisions)
            ap_per_class[cls_id].append(ap)
    # Compute mean AP
    class_ap = {}
    for cls_id, ap_list in ap_per_class.items():
        class_ap[CLASSES[cls_id]] = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0
    mAP = float(np.mean(list(class_ap.values()))) if class_ap else 0.0
    return mAP, class_ap


def calculate_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Oblicza IoU (Intersection over Union) dla dwóch zestawów bounding boxów.
    """
    # Przekształć na tensory
    boxes1 = boxes1.clone()
    boxes2 = boxes2.clone()

    # Oblicz obszar dla każdego boxu
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Oblicz współrzędne dla części wspólnej
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom

    # Oblicz szerokość i wysokość części wspólnej
    wh = (rb - lt).clamp(min=0)  # szerokość-wysokość

    # Oblicz pole części wspólnej
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Oblicz IoU
    iou = inter / (area1[:, None] + area2 - inter)

    return iou

def evaluate_model(model: torch.nn.Module, data_loader, device: torch.device, iou_threshold: float = 0.5):
    """
    Run model on data_loader and compute metrics.
    Returns dict with mAP, class_ap, avg_inference_time (s), optionally confusion matrix.
    """
    model.eval()
    all_preds = []
    all_targs = []
    times = []

    with torch.no_grad():
        for imgs, targs in tqdm(data_loader, desc="Evaluate", leave=False):
            imgs = [img.to(device) for img in imgs]
            targs_gpu = [{k: v.to(device) for k, v in t.items()} for t in targs]
            start = time.time()
            outputs = model(imgs)
            end = time.time()
            # Collect
            times.append((end - start) / len(imgs))
            # Move to CPU
            for out in outputs:
                all_preds.append({
                    'boxes': out['boxes'].cpu(),
                    'scores': out['scores'].cpu(),
                    'labels': out['labels'].cpu()
                })
            for gt in targs:
                all_targs.append({
                    'boxes': gt['boxes'],
                    'labels': gt['labels']
                })
    # Compute detection metrics
    mAP, class_ap = calculate_map(all_preds, all_targs, iou_threshold)
    avg_inf_time = float(np.mean(times))
    # Optionally compute confusion on classification of highest scoring box per image
    # (Not typical for detection, skipped)

    return {
        'mAP': mAP,
        'class_ap': class_ap,
        'avg_inference_time': avg_inf_time
    }

def plot_confusion_matrix(cm: np.ndarray, classes: list, normalize: bool = False, title: str = 'Confusion matrix') -> plt.Figure:
    """
    Plot confusion matrix for classification tasks.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        ylabel='True label', xlabel='Predicted label', title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    return fig


def print_evaluation_results(results):
    """
    Wyświetla wyniki ewaluacji modelu.
    """
    print("\n===== WYNIKI EWALUACJI =====")
    print(f"Mean Average Precision (mAP): {results['mAP']:.4f}")

    print("\nAverage Precision per class:")
    for cls_name, ap in results['class_ap'].items():
        print(f"  {cls_name}: {ap:.4f}")

    print(f"\nŚredni czas inferencji: {results['avg_inference_time']*1000:.2f} ms")


# If this module is run directly, example usage
if __name__ == '__main__':
    import argparse
    from model_implementation import initialize_model, load_model
    from dataset_pipeline import DatasetConfig, get_dataloaders

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to saved .pth model')
    parser.add_argument('--data_cache', action='store_true')
    args = parser.parse_args()

    cfg = DatasetConfig(cache=args.data_cache)
    train_dl, val_dl = get_dataloaders(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(args.model_path, len(CLASSES)).to(device)

    # Evaluate
    results = evaluate_model(model, val_dl, device)
    print_evaluation_results(results)