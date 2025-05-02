from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# -----------------------------------------------------------------------------
# 1. Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("grocery_dataset")
logger.setLevel(logging.INFO)

# CHANGE: use rotating file handler to avoid unbounded log size
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler("dataset_preparation.log", maxBytes=100 * 1024 ** 2, backupCount=3, encoding="utf-8")
stream_handler = logging.StreamHandler(sys.stdout)
for h in (file_handler, stream_handler):
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)

# -----------------------------------------------------------------------------
# 2. Configuration dataclass
# -----------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    data_dir             : str = "data"
    openimages_max       : int = 700
    coco_max             : int = 300
    val_split            : float = 0.2
    seed                 : int = 42
    min_box_area         : int = 100
    cache                : bool = False
    copy_workers         : int = 8
    input_size           : int = 640
    batch_size           : int = 4
    num_workers          : int = 4
    custom_ds_data_path  : str = Path(__file__).resolve().parent.parent / "own_database"
    custom_ds_label_path : str = Path(__file__).resolve().parent.parent / r"own_database\annotations\own_coco.json"
    use_downloaded       : bool = True
    use_custom           : bool = True

    train_images_dir: str = field(init=False, repr=False)
    val_images_dir: str = field(init=False, repr=False)
    annotations_dir: str = field(init=False, repr=False)
    train_annotations_path: str = field(init=False, repr=False)
    val_annotations_path: str = field(init=False, repr=False)
    cache_dir: str = field(init=False, repr=False)
    cache_hash: str = field(init=False)

    def __post_init__(self):
        self.train_images_dir = os.path.join(self.data_dir, "images", "train")
        self.val_images_dir = os.path.join(self.data_dir, "images", "val")
        self.annotations_dir = os.path.join(self.data_dir, "annotations")
        self.train_annotations_path = os.path.join(self.annotations_dir, "instances_train.json")
        self.val_annotations_path = os.path.join(self.annotations_dir, "instances_val.json")
        self.cache_dir = os.path.join(self.data_dir, "cache")

        cfg_for_hash = json.dumps({
            "oi": self.openimages_max,
            "coco": self.coco_max,
            "custom": self.use_custom,
            "seed": self.seed,
            "split": self.val_split,
            "isize": self.input_size,
        }, sort_keys=True)
        self.cache_hash = hashlib.md5(cfg_for_hash.encode()).hexdigest()

        for d in (self.train_images_dir, self.val_images_dir, self.annotations_dir, self.cache_dir):
            os.makedirs(d, exist_ok=True)


def get_dataloaders(cfg: DatasetConfig = DatasetConfig()
                   ) -> tuple[DataLoader, DataLoader]:
    """Preferred entry – returns (train_loader, val_loader)."""
    return build_loaders(cfg, batch=cfg.batch_size)

CLASSES = [
    "__background__",
    "Coca Cola",
    "Milk",
    "Yogurt",
    "Butter",
    "cocolino",
    "Wine glass",
    "Drink",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

OPENIMAGES_MAPPING = {
    "Wine glass": "Wine glass",
    "Drink": "Drink",
    "Milk": "Milk",
    "Dairy Product": "Yogurt",
}

COCO_MAPPING = {
    "bottle": "Wine glass",
    "cup": "Drink",
    "wine glass": "Wine glass",
}

def _src_to_dst(label: str) -> str | None:
    if label in OPENIMAGES_MAPPING:
        return OPENIMAGES_MAPPING[label]
    if label in COCO_MAPPING:
        return COCO_MAPPING[label]
    if label in CLASSES:
        return label
    return None


def load_custom_dataset(config: DatasetConfig = None) -> fo.Dataset | None:
    logger.info(f"Loading custom dataset from {config.custom_ds_data_path} with annotations {config.custom_ds_label_path}")

    if not os.path.isdir(config.custom_ds_data_path):
        logger.error(f"Data path {config.custom_ds_data_path} does not exist")
        return None

    if not os.path.isfile(config.custom_ds_label_path):
        logger.error(f"Labels file {config.custom_ds_label_path} does not exist")
        return None

    try:
        name = f"custom_grocery_{int(time.time())}"
        ds = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=config.custom_ds_data_path,
            labels_path=config.custom_ds_label_path,
            include_id=True,
            name=name,
        )
        logger.info(f"Loaded custom dataset with {len(ds)} samples")
        unexpected = set()
        for samp in ds:
            for det in samp.detections.detections:
                if det.label not in CLASSES:
                    unexpected.add(det.label)
        if unexpected:
            logger.warning(f"Ignoring unexpected classes: {unexpected}")
        return ds
    except Exception as e:
        logger.error(f"Error loading custom dataset: {e}", exc_info=True)
        return None


def validate_custom_dataset(dataset: fo.Dataset):

    class_counts: dict[str,int] = {}
    unexpected_classes = set()

    for sample in dataset:
        dets = getattr(sample, "detections", None)
        if dets is None or not hasattr(dets, "detections"):
            continue
        for detection in dets.detections:
            label = detection.label
            if label in CLASSES:
                class_counts[label] = class_counts.get(label, 0) + 1
            else:
                unexpected_classes.add(label)

    logger.info(f"Custom class distribution: {class_counts}")

    if unexpected_classes:
        logger.warning(f"Ignoring unexpected classes: {unexpected_classes}")

    missing_classes = set(CLASSES[1:]) - set(class_counts)
    if missing_classes:
        logger.warning(f"Custom dataset is missing expected classes: {missing_classes}")


def prepare_dataset(config: DatasetConfig) -> Tuple[fo.Dataset, fo.Dataset]:
    cache_meta = Path(config.cache_dir) / f"datasets_{config.cache_hash}.json"
    if config.cache and cache_meta.exists():
        with open(cache_meta) as fh:
            meta = json.load(fh)
        if (Path(config.train_annotations_path).exists() and len(os.listdir(config.train_images_dir)) >= meta["train_count"]
            and Path(config.val_annotations_path).exists() and len(os.listdir(config.val_images_dir)) >= meta["val_count"]):
            logger.info("Cache hit - loading datasets...")
            return fo.Dataset.from_json(meta["train_ds"]), fo.Dataset.from_json(meta["val_ds"])
        logger.warning("Cache invalid or incomplete, rebuilding...")

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    parts: List[fo.Dataset] = []
    if config.use_downloaded:
        oi_classes = list(OPENIMAGES_MAPPING.keys())
        logger.info(f"Downloading OpenImages subset: {oi_classes}")
        ds_oi = foz.load_zoo_dataset(
            "open-images-v7", split="train", classes=oi_classes,
            label_types=["detections"], max_samples=config.openimages_max,
            dataset_name=f"tmp_oi_{config.cache_hash}", seed=config.seed,
        )
        coco_classes = list(COCO_MAPPING.keys())
        logger.info(f"Downloading COCO subset: {coco_classes}")
        ds_coco = foz.load_zoo_dataset(
            "coco-2017", split="train", classes=coco_classes,
            label_types=["detections"], max_samples=config.coco_max,
            dataset_name=f"tmp_coco_{config.cache_hash}", seed=config.seed,
        )
        joint = ds_oi.clone(f"joint_tmp_{config.cache_hash}")
        joint.merge_samples(ds_coco)
        parts.append(joint)
    if config.use_custom:
        custom = load_custom_dataset(config)
        if custom:
            parts.append(custom)
    if not parts:
        raise ValueError("No datasets selected: enable use_downloaded or use_custom")

    if len(parts) > 1:
        merged = parts[0].clone(f"merged_all_{config.cache_hash}")
        for ds in parts[1:]:
            merged.merge_samples(ds)
    else:
        merged = parts[0]

    mapped = map_dataset_labels(merged)
    filtered = validate_and_filter_dataset(mapped, config)
    train_ds, val_ds = split_dataset(filtered, config.val_split)

    convert_to_coco_format(train_ds, config.train_images_dir, config.train_annotations_path, config)
    convert_to_coco_format(val_ds, config.val_images_dir, config.val_annotations_path, config)

    if config.cache:
        meta = {
            "train_count": len(train_ds),
            "val_count": len(val_ds),
            "train_ds": str(Path(config.cache_dir) / f"train_{config.cache_hash}.json"),
            "val_ds": str(Path(config.cache_dir) / f"val_{config.cache_hash}.json"),
        }
        train_ds.export(meta["train_ds"], fo.types.FiftyOneDataset)
        val_ds.export(meta["val_ds"], fo.types.FiftyOneDataset)
        with open(cache_meta, "w") as fh:
            json.dump(meta, fh)

    return train_ds, val_ds


def map_dataset_labels(dataset: fo.Dataset) -> fo.Dataset:
    mapped = dataset.clone()

    stats = {cls: 0 for cls in CLASSES[1:]}
    total_original = 0
    total_mapped   = 0

    for sample in tqdm(mapped, desc="Mapping labels"):
        dets = getattr(sample, "ground_truth", None)
        dets = dets.detections if dets else sample.detections.detections

        total_original += len(dets)

        new_dets = []
        for d in dets:
            new_lab = _src_to_dst(d.label)
            if new_lab in CLASSES:
                d.label = new_lab
                new_dets.append(d)
                stats[new_lab] += 1

        total_mapped += len(new_dets)

        if new_dets:
            sample["ground_truth"] = fo.Detections(detections=new_dets)
        elif sample.has_field("ground_truth"):
            sample.clear_field("ground_truth")

        sample.save()

    logger.info(
        f"Original dets: {total_original}, "
        f"Mapped dets: {total_mapped} "
        f"(ratio {total_mapped}/{total_original})"
    )
    logger.info(f"Samples before: {len(dataset)}, after: {len(mapped)}")
    logger.info(f"Class counts: {stats}")

    return mapped


def validate_and_filter_dataset(ds: fo.Dataset, cfg: DatasetConfig) -> fo.Dataset:
    filtered = ds.clone()
    invalid_ids: list[str] = []

    for sample in tqdm(filtered, desc="Validating images"):
        path = sample.filepath
        try:
            w, h = Image.open(path).size
        except Exception as e:
            logger.warning(f"Corrupt image {path}: {e}")
            invalid_ids.append(sample.id)
            continue

        good_dets = []
        for det in sample.ground_truth.detections:
            x, y, bw, bh = det.bounding_box
            if bw * bh * w * h < cfg.min_box_area:
                continue
            good_dets.append(det)
        if good_dets:
            sample.ground_truth.detections = good_dets
        else:
            invalid_ids.append(sample.id)

    if invalid_ids:
        filtered.delete_samples(invalid_ids)
        logger.info(f"Removed {len(invalid_ids)} bad samples → remaining {len(filtered)}")
    return filtered


def split_dataset(ds: fo.Dataset, val_split: float) -> Tuple[fo.Dataset, fo.Dataset]:
    ids = [s.id for s in ds]
    random.shuffle(ids)
    val_size = max(1, int(len(ids) * val_split))
    val_ids = set(ids[:val_size])
    return ds.select(list(set(ids) - val_ids)), ds.select(list(val_ids))


def _copy_image(src_dst: Tuple[str, str]):
    src, dst = src_dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        with Image.open(src) as img:
            if img.mode == "RGB" and src.lower().endswith(".jpg"):
                shutil.copy(src, dst)
                return True
            img.convert("RGB").save(dst, "JPEG", quality=90, optimize=True)
            return True
    except Exception as e:
        logger.error(f"Failed to copy {src} → {e}")
        return False


def convert_to_coco_format(ds: fo.Dataset, img_dir: str, json_path: str, cfg: DatasetConfig):
    os.makedirs(img_dir, exist_ok=True)
    coco: dict[str, Any] = {
        "info": {
            "description": "Grocery Dataset",
            "version": "1.0",
            "year": 2025,
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [{"id": 1, "name": "unknown"}],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": c, "supercategory": "grocery"} for i, c in enumerate(CLASSES) if c != "__background__"],
    }

    class_name_to_coco_id = {c: i for i, c in enumerate(CLASSES) if c != "__background__"}

    copy_jobs: List[Tuple[str, str]] = []

    for i, sample in enumerate(tqdm(ds, desc=f"COCO export → {json_path}")):
        src_path = sample.filepath
        filename = f"{i:06d}.jpg"
        dst_path = os.path.join(img_dir, filename)
        copy_jobs.append((src_path, dst_path))

        with Image.open(src_path) as img:
            width, height = img.size

        coco["images"].append({
            "id": i,
            "file_name": filename,
            "width": width,
            "height": height,
            "license": 1,
            "date_captured": time.strftime("%Y-%m-%d %H:%M:%S")
            })


        for det in sample.ground_truth.detections:
            label = det.label
            if label not in class_name_to_coco_id:
                continue
            x, y, bw, bh = det.bounding_box
            x_px, y_px = int(x * width), int(y * height)
            bw_px, bh_px = int(bw * width), int(bh * height)

            if bw_px <= 0 or bh_px <= 0:
                continue

            coco["annotations"].append({
                "id": len(coco["annotations"]),
                "image_id": i,
                "category_id": class_name_to_coco_id[label],
                "bbox": [x_px, y_px, bw_px, bh_px],
                "area": bw_px * bh_px,
                "segmentation": [[x_px, y_px, x_px + bw_px, y_px, x_px + bw_px, y_px + bh_px, x_px, y_px + bh_px]],
                "iscrowd": 0,
            })

    # parallel copy
    logger.info(f"Copying {len(copy_jobs)} images → {img_dir} with {cfg.copy_workers} workers ...")
    with ThreadPoolExecutor(max_workers=cfg.copy_workers) as pool:
        list(tqdm(pool.map(_copy_image, copy_jobs), total=len(copy_jobs), desc="Copy"))

    with open(json_path, "w") as fh:
        json.dump(coco, fh)
    logger.info(f"Wrote COCO file {json_path} with {len(coco['annotations'])} annotations")


class Resize:
    """Resize image and bbox to square *size* without in-place edits."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image, tgt: dict[str, Any]):
        w0, h0 = img.size
        img = TF.resize(img, (self.size, self.size))
        sx, sy = self.size / w0, self.size / h0
        if tgt.get("boxes") is not None and tgt["boxes"].numel():
            boxes = tgt["boxes"] * torch.tensor([sx, sy, sx, sy])
            tgt = {**tgt, "boxes": boxes, "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])}
        return img, tgt


class ToTensor:
    def __call__(self, img: Image.Image, tgt: dict[str, Any]):
        return TF.to_tensor(img), tgt


class Normalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img: torch.Tensor, tgt: dict[str, Any]):
        return TF.normalize(img, self.mean, self.std), tgt


class Compose:
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, img, tgt):
        for t in self.tfms:
            img, tgt = t(img, tgt)
        return img, tgt


def get_transform(train: bool, size: int) -> Compose:
    tfms = [Resize(size), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return Compose(tfms)


class GroceryDataset(Dataset):
    def __init__(self, root: str, ann_file: str, transform: Compose | None = None, augment: bool = False):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.augment = augment

        self.coco_id_to_class_idx = {cid: CLASS_TO_IDX[c["name"]] for cid, c in self.coco.cats.items() if c["name"] in CLASS_TO_IDX}

    def __getitem__(self, index: int):
        coco = self.coco
        img_id = self.ids[index]
        path = os.path.join(self.root, coco.loadImgs(img_id)[0]["file_name"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Image load failed: {e}")
            img = Image.new("RGB", (100, 100))
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes, labels = [], []
        width, height = img.size

        for ann in anns:
            xmin, ymin, width_box, height_box = ann["bbox"]

            if width_box <= 0 or height_box <= 0:
                continue

            bbox = [xmin, ymin, xmin + width_box, ymin + height_box]

            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(width, bbox[2])
            bbox[3] = min(height, bbox[3])

            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            boxes.append(bbox)

            labels.append(self.coco_id_to_class_idx.get(ann["category_id"], 0))

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        img_t = torch.tensor([img_id])
        area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0]) if boxes else torch.zeros((0,))

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": img_t,
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch: List[Tuple[torch.Tensor, dict[str, Any]]]):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)


def build_loaders(config: DatasetConfig,
                  batch: int | None = None,
                 ) -> tuple[DataLoader, DataLoader]:

    train_dataset = GroceryDataset(
        root        = config.train_images_dir,
        ann_file    = config.train_annotations_path,
        transform   = get_transform(train=True, size=config.input_size),
        augment     = True
    )

    val_dataset = GroceryDataset(
        root        = config.val_images_dir,
        ann_file    = config.val_annotations_path,
        transform   = get_transform(train=False, size=config.input_size),
        augment     = False
    )

    train_ld = DataLoader(
        dataset     = train_dataset,
        batch_size  = batch,
        shuffle     = True,
        num_workers = config.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True
    )

    val_ld = DataLoader(
        dataset     = val_dataset,
        batch_size  = batch,
        shuffle     = False,
        num_workers = config.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True
    )

    return train_ld, val_ld


def create_data_loaders(*,
                        config     : DatasetConfig | None = None,
                        input_size : int | None = None,
                        batch_size : int | None = None,
                        num_workers: int | None = None
                       ) -> tuple[DataLoader, DataLoader]:
    if config is None:
        config = DatasetConfig()

    if input_size is not None:
        config.input_size = input_size

    if batch_size is not None:
        config.batch_size = batch_size

    if num_workers is not None:
        config.num_workers = num_workers

    return build_loaders(config, batch=config.batch_size)


def main():
    cfg = DatasetConfig(cache=False)

    train_ds_fo, val_ds_fo = prepare_dataset(config=cfg)

    train_ld, val_ld = create_data_loaders(config=cfg)
    logger.info(f"Train/val sizes: {len(train_ld.dataset)}/{len(val_ld.dataset)}")

    imgs, tgts = next(iter(train_ld))
    logger.info(f"First batch shapes: img={imgs[0].shape}, targets={len(tgts)}")

if __name__ == "__main__":
    main()
