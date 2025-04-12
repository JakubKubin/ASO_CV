import os
import time
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Importy klas i funkcji z poprzednich skryptów
from src.dataset_preparation import get_dataloaders, CLASSES
from src.model_implementation import initialize_model, save_model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Trenuje model przez jedną epokę.
    
    Args:
        model: Model do trenowania
        optimizer: Optymalizator
        data_loader: DataLoader z danymi treningowymi
        device: Urządzenie (CPU/GPU)
        epoch: Numer epoki
        print_freq: Częstotliwość wyświetlania informacji
    
    Returns:
        Średnie straty (losses) dla tej epoki
    """
    model.train()
    epoch_loss = 0
    
    start_time = time.time()
    
    # Iteracja po batchu danych
    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoka {epoch}")):
        # Przeniesienie danych na odpowiednie urządzenie
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Obliczenie strat
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Zerowanie gradientów
        optimizer.zero_grad()
        
        # Propagacja wsteczna
        losses.backward()
        
        # Aktualizacja parametrów
        optimizer.step()
        
        # Akumulacja strat
        epoch_loss += losses.item()
        
        # Wyświetlanie informacji
        if i % print_freq == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoka: {epoch}, Batch: {i}/{len(data_loader)}, "
                  f"Strata: {losses.item():.4f}, "
                  f"Czas: {elapsed_time:.2f}s")
            
            # Szczegóły poszczególnych strat
            loss_str = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"  Straty: {loss_str}")
    
    # Oblicz średnią stratę dla epoki
    avg_loss = epoch_loss / len(data_loader)
    print(f"Średnia strata dla epoki {epoch}: {avg_loss:.4f}")
    
    return avg_loss

def evaluate(model, data_loader, device):
    """
    Ocenia model na zbiorze walidacyjnym.
    
    Args:
        model: Model do oceny
        data_loader: DataLoader z danymi walidacyjnymi
        device: Urządzenie (CPU/GPU)
    
    Returns:
        Średnia strata na zbiorze walidacyjnym
    """
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Walidacja"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Obliczamy straty
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            val_loss += losses.item()
    
    avg_val_loss = val_loss / len(data_loader)
    print(f"Średnia strata walidacyjna: {avg_val_loss:.4f}")
    
    return avg_val_loss

def train_model(model_type, data_dir, output_dir, num_epochs=10, batch_size=2, learning_rate=0.005, weight_decay=0.0005):
    """
    Pełna procedura treningu modelu.
    
    Args:
        model_type: Typ modelu ('faster_rcnn' lub 'mask_rcnn')
        data_dir: Katalog z danymi
        output_dir: Katalog wyjściowy do zapisywania modeli
        num_epochs: Liczba epok
        batch_size: Rozmiar batcha
        learning_rate: Szybkość uczenia
        weight_decay: Współczynnik regularyzacji
    
    Returns:
        Wytrenowany model i historię uczenia
    """
    # Utwórz katalog wyjściowy
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicjalizacja urządzenia
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Używane urządzenie: {device}")
    
    # Inicjalizacja modelu
    model = initialize_model(model_type)
    model.to(device)
    
    # Przygotowanie data loaderów
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    
    # Inicjalizacja optymalizatora
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay
    )
    
    # Scheduler do zmniejszania szybkości uczenia
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    # Inicjalizacja TensorBoard dla śledzenia metryk
    log_dir = os.path.join(output_dir, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    
    # Przechowywanie historii uczenia
    history = {'train_loss': [], 'val_loss': []}
    
    # Główna pętla uczenia
    for epoch in range(num_epochs):
        # Trening
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        history['train_loss'].append(train_loss)
        
        # Aktualizacja schedulera
        lr_scheduler.step()
        
        # Ewaluacja
        val_loss = evaluate(model, val_loader, device)
        history['val_loss'].append(val_loss)
        
        # Zapisywanie metryk do TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]["lr"], epoch)
        
        # Zapisywanie modelu co kilka epok
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model_path = os.path.join(output_dir, f"{model_type}_epoch_{epoch+1}.pth")
            save_model(model, model_path)
    
    