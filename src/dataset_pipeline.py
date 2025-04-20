"""
Updated grocery dataset pipeline
================================
All critical fixes requested by the user are implemented.  Search in the code for the tag
`# CHANGE:` to quickly locate each modification.
"""

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
file_handler = RotatingFileHandler("dataset_preparation.log", maxBytes=5 * 1024 ** 2, backupCount=3, encoding="utf-8")
stream_handler = logging.StreamHandler(sys.stdout)
for h in (file_handler, stream_handler):
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)

# -----------------------------------------------------------------------------
# 2. Configuration dataclass
# -----------------------------------------------------------------------------
@dataclass
class DatasetConfig:                                       # CHANGE: dataclass
    data_dir          : str = "data"
    openimages_max    : int = 700
    coco_max          : int = 300
    val_split         : float = 0.2
    seed              : int = 42
    min_box_area      : int = 100
    cache             : bool = True
    copy_workers      : int = 8
    input_size        : int = 640                          # CHANGE: większy rozmiar
    batch_size        : int = 4
    num_workers       : int = 4

    # wygeneruj hash konfiguracji do cache
    def hash(self) -> str:
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    # Those derived fields are excluded from the dataclass comparison/hash
    train_images_dir: str = field(init=False, repr=False)
    val_images_dir: str = field(init=False, repr=False)
    annotations_dir: str = field(init=False, repr=False)
    train_annotations_path: str = field(init=False, repr=False)
    val_annotations_path: str = field(init=False, repr=False)
    cache_dir: str = field(init=False, repr=False)
    cache_hash: str = field(init=False)

    def __post_init__(self):
        # folder structure
        self.train_images_dir = os.path.join(self.data_dir, "images", "train")
        self.val_images_dir = os.path.join(self.data_dir, "images", "val")
        self.annotations_dir = os.path.join(self.data_dir, "annotations")
        self.train_annotations_path = os.path.join(self.annotations_dir, "instances_train.json")
        self.val_annotations_path = os.path.join(self.annotations_dir, "instances_val.json")
        self.cache_dir = os.path.join(self.data_dir, "cache")

        # stable hash including all user‑configurable knobs
        cfg_for_hash = json.dumps({
            "oi": self.openimages_max,
            "coco": self.coco_max,
            "seed": self.seed,
            "isize": self.input_size,
        }, sort_keys=True)
        self.cache_hash = hashlib.md5(cfg_for_hash.encode()).hexdigest()

        for d in (self.train_images_dir, self.val_images_dir, self.annotations_dir, self.cache_dir):
            os.makedirs(d, exist_ok=True)


def get_dataloaders(cfg: DatasetConfig = DatasetConfig()
                   ) -> tuple[DataLoader, DataLoader]:
    """Preferred entry – returns (train_loader, val_loader)."""
    return build_loaders(cfg)

# -----------------------------------------------------------------------------
# 3. Label sets and mappings
# -----------------------------------------------------------------------------
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

# CHANGE: map "Dairy Product" → "Yogurt" which *is* in CLASSES
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

# helper for mapping
def _src_to_dst(label: str) -> str | None:
    if label in OPENIMAGES_MAPPING:
        return OPENIMAGES_MAPPING[label]
    if label in COCO_MAPPING:
        return COCO_MAPPING[label]
    return None

# -----------------------------------------------------------------------------
# 4. Utility functions
# -----------------------------------------------------------------------------


def _hash_first(items: Iterable[str], n: int = 100) -> str:
    """Return md5 hash of up to *n* strings (for quick cache integrity)."""
    m = hashlib.md5()
    for i, it in enumerate(items):
        if i >= n:
            break
        m.update(it.encode())
    return m.hexdigest()


# -----------------------------------------------------------------------------
# 5. Dataset download/merge/cache pipeline
# -----------------------------------------------------------------------------


def download_datasets(config: DatasetConfig) -> Tuple[fo.Dataset, fo.Dataset]:
    """Download OPEN‑IMAGES & COCO subsets, cache the processed splits."""

    logger.info("Checking cache …")
    cache_file = Path(config.cache_dir) / f"datasets_{config.cache_hash}.json"
    if config.cache and cache_file.exists():
        with open(cache_file) as fh:
            meta = json.load(fh)

        # quick integrity check
        train_ok = Path(config.train_annotations_path).exists() and len(os.listdir(config.train_images_dir)) >= meta["train_imgs"]
        val_ok = Path(config.val_annotations_path).exists() and len(os.listdir(config.val_images_dir)) >= meta["val_imgs"]
        if train_ok and val_ok:
            logger.info("Cache hit — loading FiftyOne datasets from JSON …")
            return (
                fo.Dataset.from_json(meta["train_ds"]),
                fo.Dataset.from_json(meta["val_ds"]),
            )
        logger.warning("Cache present but failed integrity check → rebuilding …")

    # no (valid) cache —— create from scratch
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    temp_prefix = f"tmp_grocery_{int(time.time())}"
    try:
        # Open‑Images
        oi_classes = list(OPENIMAGES_MAPPING.keys())
        logger.info(f"Downloading Open‑Images subset: {oi_classes}")
        ds_oi = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            classes=oi_classes,
            label_types=["detections"],
            max_samples=config.openimages_max,
            dataset_name=f"{temp_prefix}_oi",
            seed=config.seed,
        )

        # COCO
        coco_classes = list(COCO_MAPPING.keys())
        logger.info(f"Downloading COCO subset: {coco_classes}")
        ds_coco = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            classes=coco_classes,
            label_types=["detections"],
            max_samples=config.coco_max,
            dataset_name=f"{temp_prefix}_coco",
            seed=config.seed,
        )

        # merge
        joint = ds_oi.clone(f"{temp_prefix}_joint")
        joint.merge_samples(ds_coco)
        logger.info(f"Merged dataset size: {len(joint)}")

        mapped = map_dataset_labels(joint)
        filtered = validate_and_filter_dataset(mapped, config)
        train_ds, val_ds = split_dataset(filtered, config.val_split)

        # export to COCO + copy images
        convert_to_coco_format(train_ds, config.train_images_dir, config.train_annotations_path, config)
        convert_to_coco_format(val_ds, config.val_images_dir, config.val_annotations_path, config)

        if config.cache:
            logger.info("Saving cache metadata …")
            meta = {
                "train_imgs": len(train_ds),
                "val_imgs": len(val_ds),
                "train_ds": str(Path(config.cache_dir) / f"train_{config.cache_hash}.json"),
                "val_ds": str(Path(config.cache_dir) / f"val_{config.cache_hash}.json"),
            }
            train_ds.export(meta["train_ds"], fo.types.FiftyOneDataset)
            val_ds.export(meta["val_ds"], fo.types.FiftyOneDataset)
            with open(cache_file, "w") as fh:
                json.dump(meta, fh)

        return train_ds, val_ds

    finally:
        # cleanup tmp FO datasets (best‑effort)
        for name in fo.list_datasets():
            if name.startswith(temp_prefix):
                fo.delete_dataset(name)


# -----------------------------------------------------------------------------
# 6. Filtering, mapping, splitting helpers
# -----------------------------------------------------------------------------


def map_dataset_labels(ds: fo.Dataset) -> fo.Dataset:
    """Return clone with labels mapped to project taxonomy."""

    mapped = ds.clone()  # deep copy
    stats: dict[str, int] = {c: 0 for c in CLASSES[1:]}
    tot_original = 0
    tot_kept = 0

    for s in tqdm(mapped, desc="Mapping labels"):
        detections = s.ground_truth.detections
        tot_original += len(detections)
        kept: list[Any] = []
        for det in detections:
            new_lab = _src_to_dst(det.label)
            if new_lab and new_lab in CLASSES:
                det.label = new_lab
                kept.append(det)
                stats[new_lab] += 1
        if kept:
            s.ground_truth.detections = kept
        else:
            s.clear_field("ground_truth")  # mark empty
    mapped = mapped.match(F("ground_truth.detections").length() > 0)

    logger.info("Mapping done → kept %.2f%% of detections" % (100 * tot_kept / max(1, tot_original)))
    logger.debug(f"Distribution: {stats}")
    return mapped


def validate_and_filter_dataset(ds: fo.Dataset, cfg: DatasetConfig) -> fo.Dataset:
    filtered = ds.clone()
    invalid_ids: list[str] = []

    for s in tqdm(filtered, desc="Validating images"):
        path = s.filepath
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception as e:
            logger.warning(f"Corrupt image {path}: {e}")
            invalid_ids.append(s.id)
            continue

        if w < 10 or h < 10:
            invalid_ids.append(s.id)
            continue

        good_dets = []
        for det in s.ground_truth.detections:
            x, y, bw, bh = det.bounding_box
            if bw * bh * w * h < cfg.min_box_area:
                continue
            good_dets.append(det)
        if good_dets:
            s.ground_truth.detections = good_dets
        else:
            invalid_ids.append(s.id)

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

# -----------------------------------------------------------------------------
# 7. COCO export
# -----------------------------------------------------------------------------


def _copy_image(src_dst: Tuple[str, str]):
    src, dst = src_dst
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    try:
        # fast path — straight copy if already jpeg & RGB
        with Image.open(src) as im:
            if im.mode == "RGB" and src.lower().endswith(".jpg"):
                shutil.copy(src, dst)
                return True
            im.convert("RGB").save(dst, "JPEG", quality=90, optimize=True)
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

    class2id = {c: i for i, c in enumerate(CLASSES) if c != "__background__"}

    copy_jobs: List[Tuple[str, str]] = []

    for idx, s in enumerate(tqdm(ds, desc=f"COCO export → {json_path}")):
        src = s.filepath
        fname = f"{idx:06d}.jpg"
        dst = os.path.join(img_dir, fname)
        copy_jobs.append((src, dst))

        with Image.open(src) as im:
            w, h = im.size

        coco["images"].append({
            "id": idx,
            "file_name": fname,
            "width": w,
            "height": h,
            "license": 1,
        })

        for det in s.ground_truth.detections:
            lab = det.label
            if lab not in class2id:
                continue
            x, y, bw, bh = det.bounding_box
            x_px, y_px = int(x * w), int(y * h)
            bw_px, bh_px = int(bw * w), int(bh * h)
            if bw_px <= 0 or bh_px <= 0:
                continue
            coco["annotations"].append({
                "id": len(coco["annotations"]),
                "image_id": idx,
                "category_id": class2id[lab],
                "bbox": [x_px, y_px, bw_px, bh_px],
                "area": bw_px * bh_px,
                "iscrowd": 0,
            })

    # parallel copy
    logger.info(f"Copying {len(copy_jobs)} images → {img_dir} with {cfg.copy_workers} workers …")
    with ThreadPoolExecutor(max_workers=cfg.copy_workers) as pool:
        list(tqdm(pool.map(_copy_image, copy_jobs), total=len(copy_jobs), desc="Copy"))

    with open(json_path, "w") as fh:
        json.dump(coco, fh)
    logger.info(f"Wrote COCO file {json_path} with {len(coco['annotations'])} annotations")

# -----------------------------------------------------------------------------
# 8. Torch Dataset & transforms
# -----------------------------------------------------------------------------

class Resize:
    """Resize image and bbox to square *size* without in‑place edits."""

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

    def __getitem__(self, idx: int):
        coco = self.coco
        img_id = self.ids[idx]
        path = os.path.join(self.root, coco.loadImgs(img_id)[0]["file_name"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Image load failed: {e}")
            img = Image.new("RGB", (100, 100))
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes, labels = [], []
        w, h = img.size
        for a in anns:
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(self.coco_id_to_class_idx.get(a["category_id"], 0))

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        tgt = {
            "boxes": boxes_t,
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0]) if boxes else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        if self.transform:
            img, tgt = self.transform(img, tgt)
        return img, tgt

    def __len__(self):
        return len(self.ids)


# -----------------------------------------------------------------------------
# 9. Dataloaders
# -----------------------------------------------------------------------------

def collate_fn(batch: List[Tuple[torch.Tensor, dict[str, Any]]]):
    images, targets = zip(*batch)
    return list(images), list(targets)  # CHANGE: list – no stack


def build_loaders(cfg: DatasetConfig, batch: int | None = None
                 ) -> tuple[DataLoader, DataLoader]:
    """Main factory returning train & val DataLoaders."""
    if batch is None:
        batch = cfg.batch_size

    train_ds_fo, val_ds_fo = download_datasets(cfg)

    train_set = GroceryDataset(cfg.train_images_dir,
                               cfg.train_annotations_path,
                               get_transform(True, cfg.input_size))
    val_set   = GroceryDataset(cfg.val_images_dir,
                               cfg.val_annotations_path,
                               get_transform(False, cfg.input_size))

    train_ld = DataLoader(train_set, batch_size=batch, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate_fn,
                          pin_memory=True)
    val_ld   = DataLoader(val_set, batch_size=batch, shuffle=False,
                          num_workers=cfg.num_workers, collate_fn=collate_fn,
                          pin_memory=True)
    return train_ld, val_ld


def create_data_loaders(*,                       # keyword‑only on purpose
                        config     : DatasetConfig | None = None,
                        input_size : int | None = None,
                        batch_size : int | None = None,
                        num_workers: int | None = None
                       ) -> tuple[DataLoader, DataLoader]:
    """
    Thin wrapper kept for backward‑compat.
    Delegates to `build_loaders`.
    """
    if config is None:
        config = DatasetConfig()

    # allow runtime overrides
    if input_size is not None:
        config.input_size = input_size
    if batch_size is not None:
        config.batch_size = batch_size
    if num_workers is not None:
        config.num_workers = num_workers

    return build_loaders(config, batch=config.batch_size)

# -----------------------------------------------------------------------------
# 10. Entry point (debug)
# -----------------------------------------------------------------------------


def main():
    cfg = DatasetConfig(cache=True)  # force rebuild for demo
    train_ld, val_ld = create_data_loaders(cfg)
    logger.info(f"Train/val sizes: {len(train_ld.dataset)}/{len(val_ld.dataset)}")

    # optional FO app — guard with env var so CI/headless won’t hang
    if os.getenv("VISUALIZE") == "1":
        fo.launch_app(name="grocery_dataset", dataset=train_ld.dataset, port=5151)

    imgs, tgts = next(iter(train_ld))
    logger.info(f"First batch shapes: img={imgs[0].shape}, targets={len(tgts)}")

if __name__ == "__main__":
    main()
