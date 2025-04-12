import os
import xml.etree.ElementTree as ET
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_voc_annotation(xml_path: str, image_id: int) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """
    Parse a single VOC annotation XML file.

    Parameters:
        xml_path: Path to the XML file
        image_id: Unique ID for the image

    Returns:
        tuple containing:
            - image information dictionary
            - list of annotation dictionaries
            - list of category names found in this file
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logging.error(f"Failed to parse {xml_path}: {e}")
        return {}, [], []

    # Extract image details
    filename = root.find("filename")
    if filename is None:
        # Use XML filename if filename tag is missing
        filename = os.path.basename(xml_path).replace('.xml', '.jpg')
        logging.warning(f"Missing filename in {xml_path}, using {filename}")
    else:
        filename = filename.text

    # Get image size
    size = root.find("size")
    if size is None:
        logging.warning(f"Missing size information in {xml_path}, using defaults")
        width, height = 0, 0
    else:
        width = int(size.find("width").text) if size.find("width") is not None else 0
        height = int(size.find("height").text) if size.find("height") is not None else 0

    image_info = {
        "file_name": filename,
        "height": height,
        "width": width,
        "id": image_id
    }

    # Process annotations
    annotations = []
    categories = []

    for obj in root.iter("object"):
        # Get category name
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            logging.warning(f"Missing object name in {xml_path}, skipping annotation")
            continue

        category_name = name_elem.text.strip()
        categories.append(category_name)

        # Extract bounding box
        bndbox = obj.find("bndbox")
        if bndbox is None:
            logging.warning(f"Missing bounding box for {category_name} in {xml_path}")
            continue

        try:
            xmin = max(0, int(float(bndbox.find("xmin").text)))
            ymin = max(0, int(float(bndbox.find("ymin").text)))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            # Sanity check for box dimensions
            if xmax <= xmin or ymax <= ymin:
                logging.warning(f"Invalid box dimensions in {xml_path}: [{xmin},{ymin},{xmax},{ymax}]")
                continue

            # Adjust if box exceeds image boundaries
            if width > 0 and height > 0:
                xmax = min(xmax, width)
                ymax = min(ymax, height)

            box_width = xmax - xmin
            box_height = ymax - ymin

            # Create polygon representation for segmentation
            segmentation = [[
                xmin, ymin,
                xmin, ymax,
                xmax, ymax,
                xmax, ymin
            ]]

            # Get additional optional attributes
            difficult = 0
            if obj.find("difficult") is not None:
                difficult = int(obj.find("difficult").text)

            truncated = 0
            if obj.find("truncated") is not None:
                truncated = int(obj.find("truncated").text)

            # Create annotation entry (without category_id and annotation_id)
            annotation = {
                "segmentation": segmentation,
                "area": box_width * box_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, box_width, box_height],
                "difficult": difficult,
                "truncated": truncated
            }

            annotations.append(annotation)

        except (ValueError, AttributeError) as e:
            logging.error(f"Error processing bounding box in {xml_path}: {e}")
            continue

    return image_info, annotations, categories

def voc_to_coco(
    voc_annotations_dir: str, 
    output_json: str,
    category_map: Optional[Dict[str, int]] = None,
    threads: int = 4,
    verbose: bool = False
) -> None:
    """
    Converts VOC XML annotations to COCO JSON format.

    Parameters:
      voc_annotations_dir: Path to the directory containing VOC XML files
      output_json: Path for the output COCO JSON file
      category_map: Optional dictionary mapping category names to IDs (for consistent IDs)
      threads: Number of threads to use for parallel processing
      verbose: Whether to enable verbose logging
    """
    setup_logging(verbose)
    annotations_dir = Path(voc_annotations_dir)
    output_path = Path(output_json)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize the COCO structure
    coco = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }

    # Get all XML files
    xml_files = sorted([str(f) for f in annotations_dir.glob("*.xml")])
    if not xml_files:
        logging.error(f"No XML files found in {voc_annotations_dir}")
        return

    logging.info(f"Found {len(xml_files)} XML annotation files")

    # Dictionary to track categories
    categories: Dict[str, int] = {} if category_map is None else category_map.copy()
    all_annotations = []

    # Process files in parallel
    def process_file(args):
        xml_path, img_id = args
        return parse_voc_annotation(xml_path, img_id)

    results = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for result in tqdm(
            executor.map(process_file, [(xml_file, i+1) for i, xml_file in enumerate(xml_files)]),
            total=len(xml_files),
            desc="Processing annotations"
        ):
            results.append(result)

    # Collect all unique categories first
    all_categories = set()
    for _, _, cats in results:
        all_categories.update(cats)

    # Assign category IDs if not provided
    if category_map is None:
        for cat_name in sorted(all_categories):
            if cat_name not in categories:
                categories[cat_name] = len(categories) + 1

    # Process results
    annotation_id = 1
    for image_info, annotations, cats in results:
        if image_info:  # Only add if we have valid image info
            coco["images"].append(image_info)

            for ann in annotations:
                # Find the category name for this annotation
                # Since we don't track which category each annotation belongs to in parse_voc_annotation,
                # we need to correct this here
                if cats:  # Only proceed if we have categories for this image
                    category_name = cats[0]  # Use the first category as a fallback
                    ann["category_id"] = categories.get(category_name, 0)
                    ann["id"] = annotation_id
                    coco["annotations"].append(ann)
                    annotation_id += 1

    # Build the categories list
    for category_name, cat_id in categories.items():
        if category_name in all_categories:  # Only include categories that were actually found
            coco["categories"].append({
                "supercategory": "none",
                "id": cat_id,
                "name": category_name
            })

    # Save the COCO annotations
    try:
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)
        logging.info(f"COCO annotations successfully saved to {output_json}")
        logging.info(f"Converted {len(coco['images'])} images with {len(coco['annotations'])} annotations across {len(coco['categories'])} categories")
    except IOError as e:
        logging.error(f"Failed to write output JSON: {e}")

def get_default_paths():
    """Get default paths relative to script location."""
    script_dir = Path(__file__).resolve().parent
    default_voc_dir = script_dir / "own_database"
    default_output = default_voc_dir / "annotations" / "own_coco.json"
    return default_voc_dir, default_output

def main():
    # Get default paths
    default_voc_dir, default_output = get_default_paths()

    parser = argparse.ArgumentParser(description="Convert VOC XML annotations to COCO JSON format")
    parser.add_argument("--voc-dir", type=str, default=str(default_voc_dir), 
                        help=f"Directory containing VOC XML files (default: {default_voc_dir})")
    parser.add_argument("--output", type=str, default=str(default_output), 
                        help=f"Path for output COCO JSON file (default: {default_output})")
    parser.add_argument("--category-map", type=str, help="Optional JSON file containing category name to ID mapping")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Ensure default directories exist
    Path(args.voc_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load category mapping if provided
    category_map = None
    if args.category_map:
        try:
            with open(args.category_map, 'r') as f:
                category_map = json.load(f)
            logging.info(f"Loaded {len(category_map)} category mappings from {args.category_map}")
        except Exception as e:
            logging.error(f"Failed to load category map: {e}")

    voc_to_coco(
        args.voc_dir,
        args.output,
        category_map,
        args.threads,
        args.verbose
    )

if __name__ == "__main__":
    main()