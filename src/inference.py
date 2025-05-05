# /src/inference.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import cv2
import time

# Importy z naszych modułów
from dataset_pipeline import CLASSES
from model_implementation import load_model

def prepare_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor()
    ])
    image_tensor = transform(image).to(device)

    return image, image_tensor

def run_inference(model, image_tensor, threshold=0.155):
    model.eval()

    # Wykonaj inferecję
    with torch.no_grad():
        start_time = time.time()
        predictions = model([image_tensor])
        end_time = time.time()

    inference_time = end_time - start_time

    # Filtruj predykcje według progu pewności
    pred = predictions[0]
    keep = pred['scores'] > threshold

    filtered_predictions = {
        'boxes': pred['boxes'][keep],
        'labels': pred['labels'][keep],
        'scores': pred['scores'][keep]
    }

    if 'masks' in pred:
        filtered_predictions['masks'] = pred['masks'][keep]

    return filtered_predictions, inference_time

def visualize_predictions(image, predictions, output_path=None):
    # (bez zmian)
    img_np = np.array(image)
    img_draw = img_np.copy()

    for i, (box, label, score) in enumerate(zip(
            predictions['boxes'], predictions['labels'], predictions['scores'])):

        box = box.cpu().numpy().astype(np.int32)
        xmin, ymin, xmax, ymax = box
        class_name = CLASSES[label.item()]
        color = plt.cm.rainbow(label.item() / len(CLASSES))
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color, 10)
        text = f"{class_name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)
        cv2.rectangle(img_draw, (xmin, ymin - th - 10), (xmin + tw, ymin), color, -1)
        cv2.putText(img_draw, text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
        if 'masks' in predictions:
            mask = predictions['masks'][i, 0].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            color_mask = np.zeros_like(img_draw)
            color_mask[mask == 1] = color
            img_draw = cv2.addWeighted(img_draw, 1, color_mask, 0.5, 0)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))

    return img_draw

def process_image(model, image_path, output_path=None, threshold=0.155, show=True):
    """
    Przetwarza pojedynczy obraz i wizualizuje wyniki.

    Args:
        model: Model do inferecji
        image_path: Ścieżka do obrazu
        output_path: Ścieżka do zapisania wyniku
        threshold: Próg pewności
        show: Czy wyświetlić wynik

    Returns:
        Wyniki detekcji
    """
    # Określ urządzenie
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("[INFO] device:", device)
    model.to(device)

    # Przygotuj obraz
    image, image_tensor = prepare_image(image_path, device)

    # Wykonaj inferecję
    predictions, inference_time = run_inference(model, image_tensor, threshold)

    # Wizualizuj predykcje
    img_with_detections = visualize_predictions(image, predictions, output_path)

    # Wyświetl wyniki w terminalu
    print(f"Czas inferecji: {inference_time*1000:.2f} ms")
    print(f"Liczba wykrytych obiektów: {len(predictions['boxes'])}")

    for i, (label, score) in enumerate(zip(predictions['labels'], predictions['scores'])):
        class_name = CLASSES[label.item()]
        print(f"  {i+1}. {class_name}: {score.item():.4f}")

    # Wyświetl obraz
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(img_with_detections)
        plt.axis('off')
        plt.title(f"Detekcje ({len(predictions['boxes'])} obiektów)")
        plt.show()

    return predictions, inference_time