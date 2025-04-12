import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import functional as TF   # Używamy TF, aby nie kolidowało z fiftyOne.ViewField
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F    # Używamy F do operacji na widoku zbioru danych w FiftyOne
import shutil
from pathlib import Path

# Definicja klas produktów
CLASSES = ['__background__', 'CocaCola', 'KartonMleka', 'KubekJogurtu', 'Maslo', 'CocolinoButelka']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}

# Mapowanie klas OpenImages i COCO na nasze klasy
OPENIMAGES_MAPPING = {
    # OpenImages labels -> nasze klasy
    'Tin can': 'CocaCola',
    'Soft drink': 'CocaCola',
    'Carbonated water': 'CocaCola',
    'Milk': 'KartonMleka',
    'Dairy': 'KartonMleka',
    'Yogurt': 'KubekJogurtu',
    'Butter': 'Maslo',
    'Bottle': 'CocolinoButelka',
    'Plastic bottle': 'CocolinoButelka'
}

COCO_MAPPING = {
    # COCO labels -> nasze klasy
    'bottle': 'CocolinoButelka',
    'cup': 'KubekJogurtu',
}

def download_datasets(data_dir='data'):
    """
    Pobiera i przygotowuje dane z OpenImages i COCO za pomocą biblioteki FiftyOne.
    
    Args:
        data_dir: Katalog bazowy na dane
    """
    # Utwórz katalogi na dane
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'annotations'), exist_ok=True)
    
    print("Pobieranie i przygotowywanie danych...")
    
    # 1. Pobierz dane z Open Images
    openimages_classes = list(OPENIMAGES_MAPPING.keys())
    
    print(f"Pobieranie danych z Open Images dla klas: {openimages_classes}")
    
    # Pobieramy podzbiór OpenImages z określonymi klasami
    # Limitujemy do 500 próbek na klasę, aby nie pobierać zbyt dużo danych
    openimages_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=openimages_classes,
        max_samples=500,
        seed=42
    )
    
    print(f"Pobrano {len(openimages_dataset)} obrazów z Open Images")
    
    # 2. Pobierz dane z COCO
    coco_classes = list(COCO_MAPPING.keys())
    
    print(f"Pobieranie danych z COCO dla klas: {coco_classes}")
    
    # Pobieramy podzbiór COCO z określonymi klasami
    coco_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=coco_classes,
        max_samples=300,
        seed=42
    )
    
    print(f"Pobrano {len(coco_dataset)} obrazów z COCO")
    
    # 3. Łączymy oba zestawy danych
    joint_dataset = openimages_dataset.clone()
    joint_dataset.merge_samples(coco_dataset)
    
    print(f"Łączna liczba obrazów po połączeniu: {len(joint_dataset)}")
    
    # 4. Mapujemy etykiety z oryginalnych zbiorów danych na nasze klasy
    joint_dataset_mapped = map_dataset_labels(joint_dataset)
    
    # 5. Dzielimy dane na zbiór treningowy i walidacyjny
    train_dataset, val_dataset = split_dataset(joint_dataset_mapped, val_split=0.2)
    
    # 6. Konwertujemy dane do formatu COCO
    convert_to_coco_format(train_dataset, os.path.join(data_dir, 'images', 'train'), 
                           os.path.join(data_dir, 'annotations', 'instances_train.json'))
    convert_to_coco_format(val_dataset, os.path.join(data_dir, 'images', 'val'), 
                           os.path.join(data_dir, 'annotations', 'instances_val.json'))
    
    print("Dane zostały przygotowane i zapisane w formacie COCO")
    
    return train_dataset, val_dataset

def map_dataset_labels(dataset):
    """
    Mapuje etykiety z oryginalnych zbiorów danych na nasze klasy.
    
    Args:
        dataset: Zbiór danych FiftyOne
        
    Returns:
        Zbiór danych z przemapowanymi etykietami
    """
    # Klonujemy zbiór danych, aby nie modyfikować oryginału
    mapped_dataset = dataset.clone()
    
    # Iterujemy po wszystkich próbkach
    for sample in tqdm(mapped_dataset, desc="Mapowanie etykiet"):
        # Pobierz detekcje
        detections = sample.ground_truth.detections
        
        # Utwórz nową listę detekcji
        new_detections = []
        
        # Iteruj po wszystkich detekcjach
        for detection in detections:
            label = detection.label
            
            # Sprawdź, czy etykieta jest w mapowaniu OpenImages
            if label in OPENIMAGES_MAPPING:
                mapped_label = OPENIMAGES_MAPPING[label]
                detection.label = mapped_label
                new_detections.append(detection)
            # Sprawdź, czy etykieta jest w mapowaniu COCO
            elif label in COCO_MAPPING:
                mapped_label = COCO_MAPPING[label]
                detection.label = mapped_label
                new_detections.append(detection)
        
        # Zaktualizuj detekcje w próbce lub usuń ją, jeśli brak pasujących etykiet
        if new_detections:
            sample.ground_truth.detections = new_detections
        else:
            mapped_dataset.delete_samples(sample.id)
    
    # Usuń próbki bez detekcji
    mapped_dataset = mapped_dataset.match(F("ground_truth.detections").length() > 0)
    
    print(f"Po mapowaniu etykiet pozostało {len(mapped_dataset)} obrazów")
    
    return mapped_dataset

def split_dataset(dataset, val_split=0.2):
    """
    Dzieli zbiór danych na zbiór treningowy i walidacyjny.
    
    Args:
        dataset: Zbiór danych FiftyOne
        val_split: Proporcja danych walidacyjnych
        
    Returns:
        (train_dataset, val_dataset): Podzielone zbiory danych
    """
    val_size = int(len(dataset) * val_split)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    sample_ids = dataset.values("id")
    
    train_ids = [sample_ids[i] for i in train_indices]
    train_dataset = dataset.select(train_ids)
    val_ids = [sample_ids[i] for i in val_indices]
    val_dataset = dataset.select(val_ids)
    
    print(f"Podział danych: {len(train_dataset)} próbek treningowych, {len(val_dataset)} próbek walidacyjnych")
    return train_dataset, val_dataset

def convert_to_coco_format(dataset, images_dir, json_path):
    """
    Konwertuje zbiór danych FiftyOne do formatu COCO.
    
    Args:
        dataset: Zbiór danych FiftyOne
        images_dir: Katalog docelowy na obrazy
        json_path: Ścieżka do zapisania pliku JSON z adnotacjami
    """
    os.makedirs(images_dir, exist_ok=True)
    
    coco_json = {
        "info": {"description": "Grocery Products Dataset"},
        "licenses": [{"name": "Unknown", "id": 1}],
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "grocery", "id": i, "name": cls_name}
            for i, cls_name in enumerate(CLASSES) if cls_name != "__background__"
        ]
    }
    
    class_name_to_coco_id = {cls: i for i, cls in enumerate(CLASSES) if cls != "__background__"}
    ann_id = 0
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Konwersja do formatu COCO: {json_path}")):
        src_path = sample.filepath
        filename = f"{i:06d}.jpg"
        dst_path = os.path.join(images_dir, filename)
        shutil.copy(src_path, dst_path)
        
        img = Image.open(src_path)
        width, height = img.size
        
        image_info = {
            "id": i,
            "width": width,
            "height": height,
            "file_name": filename,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        coco_json["images"].append(image_info)
        
        detections = sample.ground_truth.detections
        for det in detections:
            label = det.label
            if label not in class_name_to_coco_id:
                continue
            category_id = class_name_to_coco_id[label]
            bbox = det.bounding_box
            x, y, w, h = bbox
            x = int(x * width)
            y = int(y * height)
            w = int(w * width)
            h = int(h * height)
            
            annotation = {
                "id": ann_id,
                "image_id": i,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                "iscrowd": 0
            }
            coco_json["annotations"].append(annotation)
            ann_id += 1
    
    with open(json_path, 'w') as f:
        json.dump(coco_json, f)

class GroceryDataset(Dataset):
    """
    Klasa Dataset do wczytywania i przetwarzania zbioru danych z produktami spożywczymi.
    """

    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

        # Mapowanie ID kategorii COCO na nasze klasy
        self.coco_id_to_class_idx = {}
        for coco_cat in self.coco.cats.values():
            if coco_cat['name'] in CLASS_TO_IDX:
                self.coco_id_to_class_idx[coco_cat['id']] = CLASS_TO_IDX[coco_cat['name']]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Wczytaj obraz
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        num_objs = len(anns)
        boxes = []
        masks = []
        labels = []

        for ann in anns:
            # Pobierz bbox w formacie [x, y, width, height]
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            width = ann['bbox'][2]
            height = ann['bbox'][3]
            # Konwersja na format [xmin, ymin, xmax, ymax]
            bbox = [xmin, ymin, xmin + width, ymin + height]
            boxes.append(bbox)

            # Pobierz segmentację, jeśli dostępna
            if 'segmentation' in ann:
                mask = coco.annToMask(ann)
                masks.append(mask)

            # Mapuj ID kategorii na nasz indeks klasy
            cat_id = ann['category_id']
            if cat_id in self.coco_id_to_class_idx:
                labels.append(self.coco_id_to_class_idx[cat_id])
            else:
                labels.append(0)

        # Konwersja list na tensory z uwzględnieniem pustych przypadków
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            area = torch.empty((0,), dtype=torch.float32)
            iscrowd = torch.empty((0,), dtype=torch.int64)
            masks = torch.empty((0, img.height, img.width), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            if masks:
                masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            else:
                masks = torch.zeros((boxes.shape[0], img.height, img.width), dtype=torch.uint8)

        img_id_tensor = torch.tensor([img_id])

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': img_id_tensor,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

# Klasa do transformacji danych
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# Konwersja obrazu na tensor (używamy TF, aby nie kolidowało z FiftyOne)
class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target

def get_dataloaders(data_dir, batch_size=2):
    transforms = Compose([ToTensor()])
    
    train_dataset = GroceryDataset(
        root=os.path.join(data_dir, 'images/train'),
        annFile=os.path.join(data_dir, 'annotations/instances_train.json'),
        transform=transforms
    )
    
    val_dataset = GroceryDataset(
        root=os.path.join(data_dir, 'images/val'),
        annFile=os.path.join(data_dir, 'annotations/instances_val.json'),
        transform=transforms
    )
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def visualize_dataset_samples(dataset, num_samples=5):
    """
    Wizualizuje próbki z zbioru danych FiftyOne.
    
    Args:
        dataset: Zbiór danych FiftyOne
        num_samples: Liczba próbek do wizualizacji
    """
    samples = dataset.take(num_samples)
    # Uruchamiamy aplikację FiftyOne na porcie 5151, zgodnie z przykładem w dokumentacji
    session = fo.launch_app(samples, port=5151)
    return session

if __name__ == "__main__":
    # Pobierz dane z OpenImages i COCO
    train_dataset, val_dataset = download_datasets()
    
    # Uruchom wizualizację próbek i połącz się z lokalną sesją FiftyOne
    print("Wizualizacja przykładowych danych treningowych:")
    session = visualize_dataset_samples(train_dataset, num_samples=5)
    
    # Utwórz dataloadery
    train_loader, val_loader = get_dataloaders('data')
    
    # Sprawdź, czy dane zostały poprawnie załadowane
    print("Sprawdzanie danych...")
    for images, targets in train_loader:
        print(f"Batch obrazów: {len(images)}")
        print(f"Rozmiar pierwszego obrazu: {images[0].shape}")
        print(f"Liczba obiektów w pierwszym obrazie: {len(targets[0]['boxes'])}")
        print(f"Klasy obiektów: {targets[0]['labels']}")
        print(f"Rozmiary bounding boxów: {targets[0]['boxes']}")
        
        # Pokaż pierwszy obraz z adnotacjami
        img = images[0].permute(1, 2, 0).numpy()
        target = targets[0]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        ax = plt.gca()
        
        for box, label in zip(target['boxes'], target['labels']):
            xmin, ymin, xmax, ymax = box.numpy()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(
                xmin, ymin - 5,
                CLASSES[label.item()],
                fontsize=12, color='white',
                bbox=dict(facecolor='red', alpha=0.5)
            )
        plt.savefig('sample_annotation.png')
        plt.close()
        
        print("Zapisano wizualizację przykładowej adnotacji do pliku 'sample_annotation.png'")
        break
