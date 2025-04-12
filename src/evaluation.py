import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import time

# Importy z naszych modułów
from src.dataset_preparation import CLASSES

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

def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    Oblicza mean Average Precision (mAP) dla zestawu predykcji.
    """
    # Inicjalizacja metryk
    ap_per_class = {i: [] for i in range(1, len(CLASSES))}  # Pomijamy tło
    
    # Dla każdego obrazu
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu()
        pred_scores = pred['scores'].cpu()
        pred_labels = pred['labels'].cpu()
        
        target_boxes = target['boxes'].cpu()
        target_labels = target['labels'].cpu()
        
        # Dla każdej klasy
        for cls_id in range(1, len(CLASSES)):
            # Znajdź predykcje i cele dla tej klasy
            pred_idx = (pred_labels == cls_id).nonzero(as_tuple=True)[0]
            target_idx = (target_labels == cls_id).nonzero(as_tuple=True)[0]
            
            if len(target_idx) == 0:
                continue  # Brak obiektów tej klasy w celu
            
            if len(pred_idx) == 0:
                ap_per_class[cls_id].append(0.0)  # Brak predykcji dla tej klasy
                continue
            
            # Posortuj predykcje według malejącego pewności
            sorted_idx = torch.argsort(pred_scores[pred_idx], descending=True)
            pred_idx = pred_idx[sorted_idx]
            
            # Oblicz IoU między predykcjami a celami
            pred_boxes_cls = pred_boxes[pred_idx]
            target_boxes_cls = target_boxes[target_idx]
            
            tp = np.zeros(len(pred_idx))
            fp = np.zeros(len(pred_idx))
            
            # Przypisz predykcje do celów
            target_matched = np.zeros(len(target_idx), dtype=bool)
            
            for i, pred_box_idx in enumerate(pred_idx):
                # Oblicz IoU z wszystkimi celami
                ious = calculate_iou_batch(pred_boxes[pred_box_idx].unsqueeze(0), target_boxes_cls)
                max_iou, max_idx = torch.max(ious, dim=0)
                
                if max_iou >= iou_threshold and not target_matched[max_idx]:
                    tp[i] = 1
                    target_matched[max_idx] = True
                else:
                    fp[i] = 1
            
            # Oblicz skumulowane sumy TP i FP
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            # Oblicz precyzję i czułość
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            recall = tp_cumsum / len(target_idx)
            
            # Oblicz AP dla tej klasy
            ap = calculate_ap(recall, precision)
            ap_per_class[cls_id].append(ap)
    
    # Oblicz mAP jako średnią AP dla wszystkich klas
    mAP = np.mean([np.mean(aps) if aps else 0.0 for cls_id, aps in ap_per_class.items()])
    
    # Oblicz AP dla każdej klasy
    class_ap = {CLASSES[cls_id]: np.mean(aps) if aps else 0.0 for cls_id, aps in ap_per_class.items()}
    
    return mAP, class_ap

def calculate_iou_batch(boxes1, boxes2):
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

def evaluate_model(model, data_loader, device):
    """
    Ocenia model na zbiorze danych i oblicza różne metryki.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Ewaluacja"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Mierz czas inferencji
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            # Zapisz czas inferencji
            inference_times.extend([end_time - start_time] * len(images))
            
            # Zapisz predykcje i cele
            all_predictions.extend(outputs)
            all_targets.extend(targets)
    
    # Oblicz mAP
    mAP, class_ap = calculate_map(all_predictions, all_targets)
    
    # Oblicz średni czas inferencji
    avg_inference_time = np.mean(inference_times)
    
    # Przygotuj wyniki
    results = {
        'mAP': mAP,
        'class_ap': class_ap,
        'avg_inference_time': avg_inference_time
    }
    
    # Dodatkowo możemy obliczyć macierz pomyłek i inne metryki
    # (dla uproszczenia pomijamy to w tym przykładzie)
    
    return results

def plot_confusion_matrix(cm, classes, normalize=False, title='Macierz pomyłek', cmap=plt.cm.Blues):
    """
    Rysuje wizualizację macierzy pomyłek.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Znormalizowana macierz pomyłek")
    else:
        print('Macierz pomyłek bez normalizacji')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Etykieta prawdziwa')
    plt.xlabel('Etykieta przewidziana')
    
    return plt

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