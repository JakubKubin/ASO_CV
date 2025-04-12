import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import cv2
import time

# Importy z naszych modułów
from src.dataset_preparation import CLASSES
from src.model_implementation import load_model

def prepare_image(image_path, device):
    """
    Przygotowuje obraz do inferecji.
    
    Args:
        image_path: Ścieżka do obrazu
        device: Urządzenie (CPU/GPU)
    
    Returns:
        Przetworzony obraz
    """
    # Wczytaj obraz
    image = Image.open(image_path).convert("RGB")
    
    # Transformacja do tensora
    transform = T.Compose([
        T.ToTensor()
    ])
    
    # Zastosuj transformację i przenieś na odpowiednie urządzenie
    image_tensor = transform(image).to(device)
    
    return image, image_tensor

def run_inference(model, image_tensor, threshold=0.5):
    """
    Wykonuje inferecję na przetworzonym obrazie.
    
    Args:
        model: Model do inferecji
        image_tensor: Tensor obrazu
        threshold: Próg pewności dla detekcji
    
    Returns:
        Predykcje modelu
    """
    # Ustaw model w trybie ewaluacji
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
    """
    Wizualizuje predykcje na obrazie.
    
    Args:
        image: Obraz PIL
        predictions: Predykcje modelu
        output_path: Ścieżka do zapisania obrazu z wizualizacją
    
    Returns:
        Obraz z naniesionymi predykcjami
    """
    # Konwertuj obraz PIL na tablicę numpy
    img_np = np.array(image)
    
    # Stwórz kopię obrazu do rysowania
    img_draw = img_np.copy()
    
    # Rysuj bounding boxy i etykiety
    for i, (box, label, score) in enumerate(zip(predictions['boxes'], predictions['labels'], predictions['scores'])):
        # Pobierz współrzędne pudełka
        box = box.cpu().numpy().astype(np.int32)
        xmin, ymin, xmax, ymax = box
        
        # Pobierz nazwę klasy
        class_name = CLASSES[label.item()]
        
        # Określ kolor na podstawie klasy (dla każdej klasy inny kolor)
        color = plt.cm.rainbow(label.item() / len(CLASSES))
        color = (color[0]*255, color[1]*255, color[2]*255)
        
        # Rysuj prostokąt
        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Przygotuj tekst z nazwą klasy i wynikiem
        text = f"{class_name}: {score.item():.2f}"
        
        # Określ rozmiar tekstu
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Rysuj tło dla tekstu
        cv2.rectangle(img_draw, (xmin, ymin - text_height - 5), (xmin + text_width, ymin), color, -1)
        
        # Rysuj tekst
        cv2.putText(img_draw, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Jeśli dostępne są maski, narysuj je
        if 'masks' in predictions:
            mask = predictions['masks'][i, 0].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            
            # Nakładamy maskę jako półprzezroczystą nakładkę
            color_mask = np.zeros_like(img_draw)
            color_mask[mask == 1] = color
            
            # Nałóż maskę z przezroczystością
            alpha = 0.3
            img_draw = cv2.addWeighted(img_draw, 1, color_mask, alpha, 0)
    
    # Zapisz obraz, jeśli podano ścieżkę
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
    
    return img_draw

def process_image(model, image_path, output_path=None, threshold=0.5, show=True):
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