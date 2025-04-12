import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import functional as TF
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import shutil
from pathlib import Path
import logging
import tempfile
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
import hashlib
import sys

# Create or get the logger
logger = logging.getLogger("grocery_dataset")
logger.setLevel(logging.INFO)

# Define a consistent formatter for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a file handler with UTF-8 encoding
file_handler = logging.FileHandler("dataset_preparation.log", encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create a stream handler for stdout with UTF-8 encoding
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Definicja klas produktów
CLASSES = ['__background__', 'CocaCola', 'KartonMleka', 'KubekJogurtu', 'Maslo', 'CocolinoButelka', 'Wine glass', 'Drink']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}

# Mapowanie klas OpenImages i COCO na nasze klasy
# OPENIMAGES_MAPPING = {
#     # OpenImages labels -> nasze klasy
#     'Tin can': 'CocaCola',
#     'Soft drink': 'CocaCola',
#     'Carbonated water': 'CocaCola',
#     'Milk': 'KartonMleka',
#     'Dairy': 'KartonMleka',
#     'Yogurt': 'KubekJogurtu',
#     'Butter': 'Maslo',
#     'Bottle': 'CocolinoButelka',
#     'Plastic bottle': 'CocolinoButelka'
# }

# COCO_MAPPING = {
#     # COCO labels -> nasze klasy
#     'bottle': 'CocolinoButelka',
#     'cup': 'KubekJogurtu',
# }

OPENIMAGES_MAPPING = {
    # OpenImages labels -> nasze klasy
    'Wine glass': 'CocaCola',
    'Drink': 'CocaCola',
    # 'Carbonated water': 'CocaCola',
    # 'Milk': 'KartonMleka',
    # 'Dairy': 'KartonMleka',
    # 'Yogurt': 'KubekJogurtu',
    # 'Butter': 'Maslo',
    # 'Bottle': 'CocolinoButelka',
    # 'Plastic bottle': 'CocolinoButelka'
}

COCO_MAPPING = {
    # COCO labels -> nasze klasy
    'bottle': 'CocaCola',
    'cup': 'CocaCola',
}

# Configuration class for easy parameter management
class DatasetConfig:
    def __init__(self, 
                 data_dir='data',
                 openimages_max_samples=500,
                 coco_max_samples=300,
                 val_split=0.2,
                 min_box_area=100,  # Minimum bounding box area in pixels
                 min_samples_per_class=20,
                 seed=42,
                 cache_data=False,
                 num_workers=4):
        self.data_dir = data_dir
        self.openimages_max_samples = openimages_max_samples
        self.coco_max_samples = coco_max_samples
        self.val_split = val_split
        self.min_box_area = min_box_area
        self.min_samples_per_class = min_samples_per_class
        self.seed = seed
        self.cache_data = cache_data
        self.num_workers = num_workers

        # Create paths
        self.train_images_dir = os.path.join(data_dir, 'images', 'train')
        self.val_images_dir = os.path.join(data_dir, 'images', 'val')
        self.annotations_dir = os.path.join(data_dir, 'annotations')
        self.train_annotations_path = os.path.join(self.annotations_dir, 'instances_train.json')
        self.val_annotations_path = os.path.join(self.annotations_dir, 'instances_val.json')
        self.cache_dir = os.path.join(data_dir, 'cache')

        # Create cache hash based on configuration
        config_str = f"{openimages_max_samples}_{coco_max_samples}_{seed}"
        self.cache_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Ensure directories exist
        for directory in [self.train_images_dir, self.val_images_dir, 
                          self.annotations_dir, self.cache_dir]:
            os.makedirs(directory, exist_ok=True)


def download_datasets(config=None):
    """
    Pobiera i przygotowuje dane z OpenImages i COCO za pomocą biblioteki FiftyOne.

    Args:
        config: Konfiguracja parametrów pobierania (DatasetConfig)

    Returns:
        (train_dataset, val_dataset): Zbiory danych FiftyOne
    """
    if config is None:
        config = DatasetConfig()

    # Check if cache exists and is valid
    cache_file = os.path.join(config.cache_dir, f"datasets_{config.cache_hash}.json")
    if config.cache_data and os.path.exists(cache_file):
        logger.info(f"Found cached dataset configuration. Checking if files exist...")
        try:
            with open(cache_file, 'r') as f:
                cache_info = json.load(f)

            # Verify annotations exist
            if (os.path.exists(config.train_annotations_path) and 
                os.path.exists(config.val_annotations_path)):

                # Check if images exist
                train_images = os.listdir(config.train_images_dir)
                val_images = os.listdir(config.val_images_dir)

                if len(train_images) >= cache_info['train_count'] and len(val_images) >= cache_info['val_count']:
                    logger.info(f"Using cached dataset with {cache_info['train_count']} training and {cache_info['val_count']} validation samples")

                    # Load datasets from cache
                    train_dataset = fo.Dataset.from_json(cache_info['train_dataset_path'])
                    val_dataset = fo.Dataset.from_json(cache_info['val_dataset_path'])

                    return train_dataset, val_dataset
        except Exception as e:
            logger.warning(f"Error loading from cache: {str(e)}. Will create new dataset.")

    logger.info(f"Creating new dataset with seed {config.seed}...")
    # Set seed for reproducibility
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create temporary dataset names
    temp_dataset_name = f"temp_grocery_{int(time.time())}"

    try:
        # 1. Pobierz dane z Open Images
        openimages_classes = list(OPENIMAGES_MAPPING.keys())

        logger.info(f"Downloading data from Open Images for classes: {openimages_classes}")

        # Pobieramy podzbiór OpenImages z określonymi klasami
        openimages_dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=openimages_classes,
            max_samples=config.openimages_max_samples,
            dataset_name=f"{temp_dataset_name}_openimages",
            seed=config.seed
        )

        logger.info(f"Downloaded {len(openimages_dataset)} images from Open Images")

        # 2. Pobierz dane z COCO
        coco_classes = list(COCO_MAPPING.keys())

        logger.info(f"Downloading data from COCO for classes: {coco_classes}")

        # Pobieramy podzbiór COCO z określonymi klasami
        coco_dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            label_types=["detections"],
            classes=coco_classes,
            max_samples=config.coco_max_samples,
            dataset_name=f"{temp_dataset_name}_coco",
            seed=config.seed
        )

        logger.info(f"Downloaded {len(coco_dataset)} images from COCO")

        # 3. Łączymy oba zestawy danych
        joint_dataset = openimages_dataset.clone(f"{temp_dataset_name}_joint")
        joint_dataset.merge_samples(coco_dataset)

        logger.info(f"Combined dataset has {len(joint_dataset)} images")

        # 4. Mapujemy etykiety z oryginalnych zbiorów danych na nasze klasy
        joint_dataset_mapped = map_dataset_labels(joint_dataset, config)

        # 4.1 Validate and filter dataset to ensure quality
        joint_dataset_filtered = validate_and_filter_dataset(joint_dataset_mapped, config)

        # 5. Dzielimy dane na zbiór treningowy i walidacyjny
        train_dataset, val_dataset = split_dataset(joint_dataset_filtered, val_split=config.val_split)

        # from pprint import pprint
        # pprint(joint_dataset_filtered)
        # pprint(train_dataset)
        # pprint(val_dataset)

        # train_dataset.persistent = True
        session = fo.launch_app(train_dataset, port=5151)

        # 6. Konwertujemy dane do formatu COCO
        convert_to_coco_format(train_dataset, config.train_images_dir, config.train_annotations_path)
        convert_to_coco_format(val_dataset, config.val_images_dir, config.val_annotations_path)

        # Save to cache if enabled
        if config.cache_data:
            # Export datasets to JSON
            train_dataset_path = os.path.join(config.cache_dir, f"train_dataset_{config.cache_hash}.json")
            val_dataset_path = os.path.join(config.cache_dir, f"val_dataset_{config.cache_hash}.json")

            train_dataset.export(export_dir=train_dataset_path, dataset_type=fo.types.FiftyOneDataset)
            val_dataset.export(export_dir=val_dataset_path, dataset_type=fo.types.FiftyOneDataset)

            # Create cache info
            cache_info = {
                "train_count": len(train_dataset),
                "val_count": len(val_dataset),
                "train_dataset_path": train_dataset_path,
                "val_dataset_path": val_dataset_path,
                "created_at": time.time(),
                "config": {
                    "openimages_max_samples": config.openimages_max_samples,
                    "coco_max_samples": config.coco_max_samples,
                    "seed": config.seed
                }
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_info, f)

            logger.info(f"Dataset cache saved to {cache_file}")

        logger.info("Data preparation complete")

        # Clean up temporary datasets
        try:
            for dataset_name in [f"{temp_dataset_name}_openimages", f"{temp_dataset_name}_coco", f"{temp_dataset_name}_joint"]:
                if fo.dataset_exists(dataset_name):
                    fo.delete_dataset(dataset_name)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary datasets: {str(e)}")

        return train_dataset, val_dataset

    except Exception as e:
        logger.error(f"Error in download_datasets: {str(e)}", exc_info=True)
        # Clean up temporary datasets on error
        try:
            for dataset_name in [f"{temp_dataset_name}_openimages", f"{temp_dataset_name}_coco", f"{temp_dataset_name}_joint"]:
                if fo.dataset_exists(dataset_name):
                    fo.delete_dataset(dataset_name)
        except:
            pass
        raise


def validate_and_filter_dataset(dataset, config):
    """
    Validate and filter dataset to ensure quality.

    Args:
        dataset: FiftyOne dataset
        config: Configuration parameters

    Returns:
        Filtered dataset
    """
    logger.info("Validating and filtering dataset...")

    # Clone dataset to avoid modifying the original
    filtered_dataset = dataset.clone()

    # Filter samples with invalid images
    invalid_samples = []
    for sample in tqdm(filtered_dataset, desc="Validating images"):
        try:
            # Check if image can be loaded
            img_path = sample.filepath
            with Image.open(img_path) as img:
                # Check if image is too small or has no detections
                width, height = img.size
                if width < 10 or height < 10:
                    invalid_samples.append(sample.id)
                    continue

                # Check if all detections are valid
                valid_detections = []
                for detection in sample.ground_truth.detections:
                    # Get pixel coordinates
                    bbox = detection.bounding_box
                    x, y, w, h = bbox
                    x_px = int(x * width)
                    y_px = int(y * height)
                    w_px = int(w * width)
                    h_px = int(h * height)

                    # Check if bbox is too small
                    if w_px * h_px < config.min_box_area:
                        continue

                    # Check if bbox is within image bounds
                    if x_px < 0 or y_px < 0 or x_px + w_px > width or y_px + h_px > height:
                        continue

                    valid_detections.append(detection)

                # Update detections or mark sample as invalid
                if valid_detections:
                    sample.ground_truth.detections = valid_detections
                else:
                    invalid_samples.append(sample.id)

        except Exception as e:
            logger.warning(f"Error validating sample {sample.id}: {str(e)}")
            invalid_samples.append(sample.id)

    # Delete invalid samples
    if invalid_samples:
        filtered_dataset.delete_samples(invalid_samples)
        logger.info(f"Removed {len(invalid_samples)} invalid samples")

    # Count samples per class
    class_counts = {}
    for sample in filtered_dataset:
        for detection in sample.ground_truth.detections:
            label = detection.label
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

    logger.info(f"Class distribution after filtering: {class_counts}")

    # Check if we have enough samples per class
    for class_name in CLASSES[1:]:  # Skip background
        if class_name not in class_counts or class_counts[class_name] < config.min_samples_per_class:
            logger.warning(f"Class {class_name} has only {class_counts.get(class_name, 0)} samples, "
                         f"which is less than the minimum {config.min_samples_per_class}")

    return filtered_dataset


def map_dataset_labels(dataset, config):
    """
    Mapuje etykiety z oryginalnych zbiorów danych na nasze klasy.

    Args:
        dataset: Zbiór danych FiftyOne
        config: Konfiguracja

    Returns:
        Zbiór danych z przemapowanymi etykietami
    """
    # Klonujemy zbiór danych, aby nie modyfikować oryginału
    mapped_dataset = dataset.clone()

    # Counters for statistics
    stats = {cls: 0 for cls in CLASSES[1:]}  # Skip background
    total_original = 0
    total_mapped = 0

    # Iterujemy po wszystkich próbkach
    for sample in tqdm(mapped_dataset, desc="Mapping labels"):
        # Pobierz detekcje
        detections = sample.ground_truth.detections
        total_original += len(detections)

        # Utwórz nową listę detekcji
        new_detections = []

        # Iteruj po wszystkich detekcjach
        for detection in detections:
            label = detection.label

            # Sprawdź, czy etykieta jest w mapowaniu OpenImages
            if label in OPENIMAGES_MAPPING:
                mapped_label = OPENIMAGES_MAPPING[label]
                # Only add if the mapped label is in our CLASSES list
                if mapped_label in CLASSES:
                    detection.label = mapped_label
                    stats[mapped_label] += 1
                    new_detections.append(detection)
            # Sprawdź, czy etykieta jest w mapowaniu COCO
            elif label in COCO_MAPPING:
                mapped_label = COCO_MAPPING[label]
                # Only add if the mapped label is in our CLASSES list
                if mapped_label in CLASSES:
                    detection.label = mapped_label
                    stats[mapped_label] += 1
                    new_detections.append(detection)

        total_mapped += len(new_detections)

        # Zaktualizuj detekcje w próbce
        if new_detections:
            sample.ground_truth.detections = new_detections

    # Usuń próbki bez detekcji
    samples_before = len(mapped_dataset)
    mapped_dataset = mapped_dataset.match(F("ground_truth.detections").length() > 0)
    samples_after = len(mapped_dataset)

    logger.info(f"Label mapping statistics:")
    logger.info(f"  - Original detections: {total_original}")
    logger.info(f"  - Mapped detections: {total_mapped}")
    logger.info(f"  - Mapping ratio: {total_mapped/total_original:.2f}")
    logger.info(f"  - Samples before filtering: {samples_before}")
    logger.info(f"  - Samples after filtering: {samples_after}")
    logger.info(f"  - Class distribution: {stats}")

    return mapped_dataset


def split_dataset(dataset, val_split=0.2):
    """
    Dzieli zbiór danych na zbiór treningowy i walidacyjny, z uwzględnieniem stratyfikacji klas.

    Args:
        dataset: Zbiór danych FiftyOne
        val_split: Proporcja danych walidacyjnych

    Returns:
        (train_dataset, val_dataset): Podzielone zbiory danych
    """
    # Count samples per class
    class_samples = {cls: [] for cls in CLASSES[1:]}  # Skip background

    # Check for unexpected labels
    unexpected_labels = set()

    for sample in dataset:
        # Get unique classes in this sample
        sample_classes = set()
        for detection in sample.ground_truth.detections:
            label = detection.label
            if label not in CLASSES:
                unexpected_labels.add(label)
                continue
            sample_classes.add(label)

        # Add sample to all its classes
        for cls in sample_classes:
            if cls in class_samples:
                class_samples[cls].append(sample.id)

    # Log unexpected labels
    if unexpected_labels:
        logger.warning(f"Found {len(unexpected_labels)} unexpected labels in dataset: {unexpected_labels}")
        logger.warning("These labels will be ignored during dataset splitting")

    # Calculate validation samples for each class
    val_ids = set()
    for cls, samples in class_samples.items():
        # Shuffle samples
        random.shuffle(samples)
        # Select validation samples
        val_count = max(1, int(len(samples) * val_split))
        val_ids.update(samples[:val_count])

    # Create train and validation datasets
    train_ids = [sample.id for sample in dataset if sample.id not in val_ids]
    train_dataset = dataset.select(train_ids)
    val_dataset = dataset.select(list(val_ids))

    logger.info(f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation samples")

    # Verify class distribution
    train_class_counts = {cls: 0 for cls in CLASSES[1:]}
    val_class_counts = {cls: 0 for cls in CLASSES[1:]}

    for sample in train_dataset:
        for detection in sample.ground_truth.detections:
            label = detection.label
            if label in train_class_counts:
                train_class_counts[label] += 1

    for sample in val_dataset:
        for detection in sample.ground_truth.detections:
            label = detection.label
            if label in val_class_counts:
                val_class_counts[label] += 1

    logger.info(f"Training class distribution: {train_class_counts}")
    logger.info(f"Validation class distribution: {val_class_counts}")

    return train_dataset, val_dataset


def copy_image_with_retries(src_path, dst_path, max_retries=3):
    """Copy image with retries in case of I/O errors"""
    for attempt in range(max_retries):
        try:
            # Try to optimize image while copying
            with Image.open(src_path) as img:
                # Convert to RGB if image is in RGBA mode
                if img.mode == 'RGBA':
                    img = img.convert('RGB')

                # Save with optimized quality
                img.save(dst_path, 'JPEG', quality=90, optimize=True)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error copying image {src_path} to {dst_path}, attempt {attempt+1}: {str(e)}")
                time.sleep(1)  # Wait before retrying
            else:
                logger.error(f"Failed to copy image after {max_retries} attempts: {str(e)}")
                # Fallback to direct copy
                try:
                    shutil.copy(src_path, dst_path)
                    return True
                except Exception as e2:
                    logger.error(f"Fallback copy failed: {str(e2)}")
                    return False


def convert_to_coco_format(dataset, images_dir, json_path):
    """
    Konwertuje zbiór danych FiftyOne do formatu COCO z wykorzystaniem wielowątkowości.

    Args:
        dataset: Zbiór danych FiftyOne
        images_dir: Katalog docelowy na obrazy
        json_path: Ścieżka do zapisania pliku JSON z adnotacjami
    """
    os.makedirs(images_dir, exist_ok=True)

    coco_json = {
        "info": {
            "description": "Grocery Products Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Script",
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [{"name": "Unknown", "id": 1}],
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "grocery", "id": i, "name": cls_name}
            for i, cls_name in enumerate(CLASSES) if cls_name != "__background__"
        ]
    }

    class_name_to_coco_id = {
        cls: i for i, cls in enumerate(CLASSES) 
        if cls != "__background__"
    }

    # Process each sample and collect image info and annotations
    image_infos = []
    annotation_infos = []
    copy_tasks = []

    for i, sample in enumerate(tqdm(dataset, desc=f"Preparing COCO conversion: {json_path}")):
        src_path = sample.filepath
        filename = f"{i:06d}.jpg"
        dst_path = os.path.join(images_dir, filename)

        # Add to copy tasks
        copy_tasks.append((src_path, dst_path))

        try:
            # Get image dimensions
            with Image.open(src_path) as img:
                width, height = img.size

            image_info = {
                "id": i,
                "width": width,
                "height": height,
                "file_name": filename,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            image_infos.append(image_info)

            # Process annotations
            detections = sample.ground_truth.detections
            for det in detections:
                label = det.label
                if label not in class_name_to_coco_id:
                    continue

                category_id = class_name_to_coco_id[label]
                bbox = det.bounding_box
                x, y, w, h = bbox
                x_px = int(x * width)
                y_px = int(y * height)
                w_px = int(w * width)
                h_px = int(h * height)

                # Skip invalid boxes
                if w_px <= 0 or h_px <= 0:
                    continue

                annotation_info = {
                    "id": len(annotation_infos),
                    "image_id": i,
                    "category_id": category_id,
                    "bbox": [x_px, y_px, w_px, h_px],
                    "area": w_px * h_px,
                    "segmentation": [[x_px, y_px, x_px + w_px, y_px, x_px + w_px, y_px + h_px, x_px, y_px + h_px]],
                    "iscrowd": 0
                }
                annotation_infos.append(annotation_info)

        except Exception as e:
            logger.warning(f"Error processing sample {i}: {str(e)}")

    # Copy images using multithreading
    logger.info(f"Copying {len(copy_tasks)} images to {images_dir}...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(lambda args: copy_image_with_retries(*args), copy_tasks),
            total=len(copy_tasks),
            desc="Copying images"
        ))

    # Add to COCO JSON
    coco_json["images"] = image_infos
    coco_json["annotations"] = annotation_infos

    # Save JSON file
    logger.info(f"Saving COCO annotations to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(coco_json, f)

    logger.info(f"Saved {len(image_infos)} images and {len(annotation_infos)} annotations to {json_path}")


class GroceryDataset(Dataset):
    """
    Klasa Dataset do wczytywania i przetwarzania zbioru danych z produktami spożywczymi.
    """

    def __init__(self, root, annFile, transform=None, augment=False):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.augment = augment

        # Mapowanie ID kategorii COCO na nasze klasy
        self.coco_id_to_class_idx = {}
        for coco_cat in self.coco.cats.values():
            if coco_cat['name'] in CLASS_TO_IDX:
                self.coco_id_to_class_idx[coco_cat['id']] = CLASS_TO_IDX[coco_cat['name']]

        # Log dataset statistics
        logger.info(f"Loaded dataset from {annFile} with {len(self.ids)} images")
        class_counts = {}
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id in self.coco_id_to_class_idx:
                    cls_idx = self.coco_id_to_class_idx[cat_id]
                    cls_name = CLASSES[cls_idx]
                    if cls_name not in class_counts:
                        class_counts[cls_name] = 0
                    class_counts[cls_name] += 1

        logger.info(f"Class distribution: {class_counts}")

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Wczytaj obraz
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return a fallback image
            img = Image.new('RGB', (100, 100), color=(128, 128, 128))
            anns = []

        num_objs = len(anns)
        boxes = []
        masks = []
        labels = []

        # Image dimensions
        width, height = img.size

        for ann in anns:
            # Pobierz bbox w formacie [x, y, width, height]
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            width_box = ann['bbox'][2]
            height_box = ann['bbox'][3]

            # Skip invalid boxes
            if width_box <= 0 or height_box <= 0:
                continue

            # Konwersja na format [xmin, ymin, xmax, ymax]
            bbox = [xmin, ymin, xmin + width_box, ymin + height_box]

            # Verify bbox is within image bounds
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(width, bbox[2])
            bbox[3] = min(height, bbox[3])

            # Skip boxes that are too small or invalid after adjustment
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

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
                labels.append(0)  # Background

        # Data augmentation if enabled
        if self.augment and len(boxes) > 0 and random.random() < 0.5:
            # Horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)
                for i in range(len(boxes)):
                    xmin, ymin, xmax, ymax = boxes[i]
                    boxes[i] = [width - xmax, ymin, width - xmin, ymax]

                if len(masks) > 0:
                    for i in range(len(masks)):
                        masks[i] = np.fliplr(masks[i])

            # Random brightness and contrast adjustment
            if random.random() < 0.3:
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                img = TF.adjust_brightness(img, brightness_factor)
                img = TF.adjust_contrast(img, contrast_factor)

        # Ensure we have boxes and labels
        if len(boxes) == 0:
            # Return empty example with just the image
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64)
            }
            if len(masks) > 0:
                target["masks"] = torch.zeros((0, height, width), dtype=torch.uint8)

            if self.transform is not None:
                img, target = self.transform(img, target)

            return img, target

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])

        # Calculate areas from boxes
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # All instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }

        # Add masks if available
        if len(masks) > 0:
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            target["masks"] = masks

        # Apply transforms
        if self.transform is not None:
            img, target = self.transform(img, target)

        sample = (img, target)
        # print("DEBUG: __getitem__ sample type:", type(sample), "length:", len(sample))
        return img, target

    def __len__(self):
        return len(self.ids)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, target):
        # Convert image to tensor
        image = TF.to_tensor(image)

        return image, target


class Compose(object):
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Normalize(object):
    """Normalize image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize(object):
    """Resize image and bounding boxes."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # Original dimensions
        orig_width, orig_height = image.size

        # Resize image
        image = TF.resize(image, (self.size, self.size))

        # Resize bounding boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            scale_x = self.size / orig_width
            scale_y = self.size / orig_height

            # Apply scaling to bounding boxes [xmin, ymin, xmax, ymax]
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y

            target["boxes"] = boxes

            # Update areas
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return image, target


def get_transform(train=True, input_size=224):
    """
    Returns transformations for training or validation.

    Args:
        train: Whether to use training transforms with augmentation
        input_size: Size to resize images to

    Returns:
        Composed transforms
    """
    transforms = []

    # Resize images
    transforms.append(Resize(input_size))

    # Convert to tensor
    transforms.append(ToTensor())

    # Normalize with ImageNet mean and std
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return Compose(transforms)


def visualize_dataset_samples(dataset, num_samples=5, figsize=(15, 10)):
    """
    Visualize random samples from the dataset with bounding boxes.

    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to show
        figsize: Figure size for the plot
    """
    indices = np.random.randint(0, len(dataset), num_samples)

    fig, axes = plt.subplots(1, num_samples, figsize=figsize)

    for i, idx in enumerate(indices):
        img, target = dataset[idx]

        # Convert tensor to numpy for visualization
        if isinstance(img, torch.Tensor):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean

            # Convert to numpy and transpose
            img = img.numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)

        # Show image
        axes[i].imshow(img)
        axes[i].set_title(f"Sample {idx}")

        # Draw bounding boxes
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            axes[i].add_patch(rect)
            axes[i].text(
                xmin, ymin - 5, CLASSES[label],
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=8, color='white'
            )

        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def collate_fn(batch):
    """
    Custom collate function for the DataLoader to handle variable-sized boxes and masks.

    Args:
        batch: List of (image, target) tuples

    Returns:
        Batched images and targets
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)

    return images, targets


def create_data_loaders(config=None, input_size=224, batch_size=8, num_workers=4):
    """
    Create train and validation data loaders.

    Args:
        config: Dataset configuration
        input_size: Input size for the model
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    if config is None:
        config = DatasetConfig()

    # Download datasets if needed
    train_dataset_fo, val_dataset_fo = download_datasets(config)

    # Create Dataset objects
    train_dataset = GroceryDataset(
        root=config.train_images_dir,
        annFile=config.train_annotations_path,
        transform=get_transform(train=True, input_size=input_size),
        augment=True
    )

    val_dataset = GroceryDataset(
        root=config.val_images_dir,
        annFile=config.val_annotations_path,
        transform=get_transform(train=False, input_size=input_size),
        augment=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    logger.info(f"Created data loaders: train={len(train_dataset)} samples, val={len(val_dataset)} samples")

    return train_loader, val_loader


def main():
    """
    Main function to prepare the dataset and test the data loading pipeline.
    """
    # Configure dataset
    config = DatasetConfig(
        openimages_max_samples=500,
        coco_max_samples=300,
        val_split=0.2,
        seed=42,
        cache_data=False,
    )

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        config=config,
        input_size=224,
        batch_size=8,
        num_workers=4
    )

    # Test data loading
    logger.info("Testing data loading...")
    train_iter = iter(train_loader)
    images, targets = next(train_iter)

    logger.info(f"Loaded batch with {len(images)} images")

    logger.info(f"Type of images[0]: {type(images[0])}")
    if isinstance(images[0], tuple):
        logger.info("images[0] is a tuple. Its content types: " + 
                    f"{type(images[0][0])}, {type(images[0][1])}")

    logger.info(f"Image shape: {images[0].shape}")
    logger.info(f"Target keys: {targets[0].keys()}")

    # Visualize some samples
    dataset = GroceryDataset(
        root=config.train_images_dir,
        annFile=config.train_annotations_path,
        transform=None,  # No transform for visualization
        augment=False
    )

    visualize_dataset_samples(dataset, num_samples=5)

    logger.info("Dataset preparation and testing complete!")


if __name__ == "__main__":
    main()