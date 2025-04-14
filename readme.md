# Python Virtual Environment Setup and Example Script Run

## Introduction
Create and use a Python virtual environment (commonly named `.venv`). Virtual environments allow you to isolate your project's dependencies, ensuring they do not interfere with system-wide packages.

## Prerequisites
- **Python:** Ensure you have Python 3.12.6 installed. You can verify your Python version by running:
  ```bash
  python --version
## Setting Up the Virtual Environment
- Open a terminal or command prompt.
- Navigate to your project directory.
- Create the virtual environment:
    ```bash
    python -m venv .venv
## Activate virtual environment

    .venv\Scripts\activate

- When activated, your terminal prompt should display a prefix like (.venv) indicating that you're using the virtual environment.

## Installing Packages
- With the virtual environment active, you can install any required packages using pip:

    ```bash
    pip install -r requirements.txt
## Running the Python Script
While the virtual environment is active, run command to create dataset:

    python src\dataset_prep_v6.py

## Deactivating virtual environment
    deactivate
Your terminal prompt will return to its normal state once deactivated.

# General Notes

## Preparing VOCs

Download **LabelImg** following the instructions provided in [this article](https://medium.com/deepquestai/object-detection-training-preparing-your-custom-dataset-6248679f0d1d).

The tool is very straightforward to use — the **W** key is recommended for quickly creating bounding boxes.

### Label List:
- Coca Cola
- Butter
- Milk
- cocolino
- Yogurt

It is crucial to maintain **correct spelling, capitalization, and spacing** in labels to avoid unnecessary corrections later.

After annotating each image, **save the corresponding file immediately**.

A good practice is to name the image file in a way that reflects its content, which will simplify future dataset management.

**IMPORTANT:** If an image is loaded at an incorrect angle, refer to the next section. It is essential that **all images maintain a consistent orientation** — matching how they were originally captured.

## ROTATE_PHOTOS

If the image orientation in LabelImg differs from how it was originally taken, it must be corrected. A script is provided for this purpose:

```bash
python rotate_photos.py
```

You will be prompted to provide the path to the folder containing the images. Please run this in a separate folder (not your main project directory), and **verify** the output to ensure the rotation was successful.

Once verified, re-open the images in LabelImg to confirm their orientation is correct before beginning annotation.

## VOC_TO_COCO

After annotations are completed using LabelImg (which outputs in the VOC `.xml` format), they must be converted to the COCO `.json` format. Use the following script:

```bash
python voc_to_coco.py --voc-dir PATH_TO_XML_FILES
```

Replace `PATH_TO_XML_FILES` with the directory containing your XML annotation files.

---

This workflow ensures a clean, consistent dataset preparation pipeline from raw images to COCO-compatible annotations.
