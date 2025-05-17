# /src/evaluation.py
import time
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from pycocotools.cocoeval import COCOeval
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

# def evaluate_model(model: torch.nn.Module, data_loader, device: torch.device, iou_threshold: float = 0.5):
#     """
#     Run model on data_loader and compute metrics.
#     Returns dict with mAP, class_ap, avg_inference_time (s), optionally confusion matrix.
#     """
#     model.eval()
#     cpu_device = torch.device('cpu')

#     all_preds = []
#     all_targs = []
#     inference_times = []

#     with torch.no_grad():
#         for imgs, targs in tqdm(data_loader, desc="Evaluate", leave=False):
#             imgs = [img.to(device) for img in imgs]
#             targs_gpu = [{k: v.to(device) for k, v in t.items()} for t in targs]
#             start = time.time()
#             outputs = model(imgs)
#             end = time.time()
#             # Collect
#             inference_times.append((end - start) / len(imgs))
#             # Move to CPU
#             for out in outputs:
#                 all_preds.append({
#                     'boxes': out['boxes'].to(cpu_device),
#                     'scores': out['scores'].to(cpu_device),
#                     'labels': out['labels'].to(cpu_device)
#                 })
#             for gt in targs:
#                 all_targs.append({
#                     'boxes': gt['boxes'],
#                     'labels': gt['labels']
#                 })

#     mAP, class_ap = calculate_map(all_preds, all_targs, iou_threshold)
#     avg_inf_time = float(np.mean(inference_times))

#     return {
#         'mAP': mAP,
#         'class_ap': class_ap,
#         'avg_inference_time': avg_inf_time
#     }

def evaluate_model(model: torch.nn.Module,
                   data_loader,
                   device: torch.device,
                   score_threshold: float = 0.0,
                   iou_type: str = 'segm') -> dict:
    """
    Evaluate model using COCO metrics (via pycocotools).
    Filters predictions by score_threshold and computes mAP with COCOeval.
    Returns dict with mAP @[.5:.95], mAP @.50, mAP @.75, and avg inference time.
    """
    model.eval()
    cpu_device = torch.device('cpu')

    coco_gt = data_loader.dataset.coco
    # Build inverse mapping from internal class idx to original COCO category id, if available
    inv_map = None
    if hasattr(data_loader.dataset, 'coco_id_to_class_idx'):
        inv_map = {v: k for k, v in data_loader.dataset.coco_id_to_class_idx.items()}

    results = []
    image_ids = []
    inference_times = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluate", leave=False):
            images_dev = [img.to(device) for img in images]
            start = time.time()
            try:
                outputs = model(images_dev)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    model.to(cpu_device)
                    images_cpu = [img.to(cpu_device) for img in images]
                    outputs = model(images_cpu)
                    model.to(device)
                else:
                    raise
            end = time.time()
            inference_times.append((end - start) / len(images))

            for out, t in zip(outputs, targets):
                img_id = int(t['image_id'].item())
                boxes  = out['boxes'].cpu().numpy()
                scores = out['scores'].cpu().numpy()
                labels = out['labels'].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    # map internal label to COCO category_id
                    cat_id = int(label)
                    if inv_map is not None:
                        cat_id = inv_map.get(label, label)
                    results.append({
                        'image_id': img_id,
                        'category_id': cat_id,
                        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        'score': float(score)
                    })
                    print(f"Result: {results[-1]}")
                image_ids.append(img_id)

    if not results:
        print("[WARN] No detections. Returning zeros.")
        return {k: 0.0 for k in [
            'mAP_coco','mAP_50','mAP_75',
            'AP_small','AP_medium','AP_large',
            'AR_1','AR_10','AR_100',
            'AR_small','AR_medium','AR_large']
        } | {'avg_inference_time': float(np.mean(inference_times)) if inference_times else 0.0}

    # Run COCOeval
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    all_ids = (data_loader.dataset.ids if hasattr(data_loader.dataset, 'ids')
               else sorted(coco_gt.getImgIds()))
    coco_eval.params.imgIds = all_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # 12-element array
    clamp = lambda v: float(v) if v > 0 else 0.0
    results_dict = {
        'mAP_coco':   clamp(stats[0]),
        'mAP_50':     clamp(stats[1]),
        'mAP_75':     clamp(stats[2]),
        'AP_small':   clamp(stats[3]),
        'AP_medium':  clamp(stats[4]),
        'AP_large':   clamp(stats[5]),
        'AR_1':       clamp(stats[6]),
        'AR_10':      clamp(stats[7]),
        'AR_100':     clamp(stats[8]),
        'AR_small':   clamp(stats[9]),
        'AR_medium':  clamp(stats[10]),
        'AR_large':   clamp(stats[11]),
        'avg_inference_time': float(np.mean(inference_times))
    }
    return results_dict

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


def print_evaluation_results(results: dict):
    """
    Print evaluation metrics in a readable table.
    """
    # Prepare metrics
    table = [
        ("mAP@[.5:.95]", f"{results['mAP_coco']:.6f}"),
        ("mAP@0.50",       f"{results['mAP_50']:.6f}"),
        ("mAP@0.75",       f"{results['mAP_75']:.6f}"),
        ("AP (small)",     f"{results['AP_small']:.6f}"),
        ("AP (medium)",    f"{results['AP_medium']:.6f}"),
        ("AP (large)",     f"{results['AP_large']:.6f}"),
        ("AR@1",           f"{results['AR_1']:.6f}"),
        ("AR@10",          f"{results['AR_10']:.6f}"),
        ("AR@100",         f"{results['AR_100']:.6f}"),
        ("AR (small)",     f"{results['AR_small']:.6f}"),
        ("AR (medium)",    f"{results['AR_medium']:.6f}"),
        ("AR (large)",     f"{results['AR_large']:.6f}"),
        ("Avg inf time (ms)", f"{results['avg_inference_time']*1000:.2f}")
    ]
    # Column widths
    col1 = max(len(row[0]) for row in table)
    col2 = max(len(row[1]) for row in table)
    # Print header
    sep = "=" * (col1 + col2 + 5)
    print(f"\n{sep}")
    print(f"{'METRIC'.ljust(col1)} | {'VALUE'.rjust(col2)}")
    print("-" * (col1 + col2 + 5))
    # Print rows
    for name, val in table:
        print(f"{name.ljust(col1)} | {val.rjust(col2)}")
    print(f"{sep}\n")


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