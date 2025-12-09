# ======================================================================
#  generate_dataset.py
#  Synthetic ASL Dataset Generator – Image Fabrication Pipeline
#
#  Author: Andrew Bieber
#
#  Description:
#      This script generates a fully synthetic dataset of abstract shapes,
#      one visual style per letter (A–Z). Each generated image includes:
#
#          • An abstract pattern mapped deterministically to a letter
#          • A synthetic "distance" label (continuous target)
#          • Augmentations (rotation, blur, noise)
#          • A structured directory layout (asl_abstract_dataset/<letter>/)
#          • A master labels.xlsx file indexing all images
#
#      The dataset is designed to train a multi-task CNN:
#         - Task 1: 26-class classification (A–Z)
#         - Task 2: Distance regression
#
# ======================================================================

import os
import random
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from openpyxl import Workbook

from src.dataset.shapes import draw_abstract_for_letter
from src.dataset.augmentations import random_rotation, maybe_blur, add_noise, random_bg_color


# ======================================================================
# Constants
# ======================================================================

IMG_SIZE = 256
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

IMAGES_PER_LETTER = 20            # For proof of concept
# IMAGES_PER_LETTER = 4000        # For full-scale dataset

DIST_MIN = 1.0
DIST_MAX = 5.0

OUTPUT_DIR = "asl_abstract_dataset"


# ======================================================================
# generate_single_image
# ======================================================================
"""
Generate one synthetic training image for a given letter.

Args:
    letter (str):
        Alphabet character (A–Z) indicating pattern style.
    idx (int):
        Image index for filename purposes.

Returns:
    (PIL.Image, float, str):
        img        : The generated augmented image.
        distance   : A float label used for regression.
        rel_path   : Relative path where the image should be saved.

Steps:
    1. Sample distance label uniformly in [DIST_MIN, DIST_MAX].
    2. Compute scale = 1.5 / distance (inversely proportional).
    3. Create a blank RGB image with jittered background color.
    4. Draw abstract pattern mapped from the character.
    5. Apply augmentations:
            • random rotation
            • occasional blur
            • Gaussian noise
    6. Return image + metadata.
"""
def generate_single_image(letter, idx):
    # Distance value and inverse scaling rule
    distance = random.uniform(DIST_MIN, DIST_MAX)
    scale = 1.5 / distance

    # Background + drawing surface
    bg = random_bg_color()
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)

    # Draw letter-specific abstract shape
    center = (IMG_SIZE / 2, IMG_SIZE / 2)
    draw_abstract_for_letter(letter, draw, center, scale)

    # Augmentations
    img = random_rotation(img, bg)
    img = maybe_blur(img)
    img = add_noise(img, sigma=6.0)

    # Filename and relative path
    filename = f"{letter}_{idx:05d}.png"
    rel_path = os.path.join(letter, filename)

    return img, distance, rel_path


# ======================================================================
# generate_dataset
# ======================================================================
"""
Create the full synthetic dataset.

Directory Structure:
    asl_abstract_dataset/
        ├── A/
        ├── B/
        ├── ...
        ├── Z/
        └── labels.xlsx

Excel Format:
    filepath | letter | distance

Process:
    1. Create output directory & per-letter subfolders.
    2. Initialize workbook with header row.
    3. For each letter:
           - Generate IMAGES_PER_LETTER samples.
           - Save images into their folder.
           - Log metadata into labels.xlsx.
    4. Save workbook and print completion message.
"""
def generate_dataset():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ensure A–Z subfolders exist
    for letter in LETTERS:
        os.makedirs(os.path.join(OUTPUT_DIR, letter), exist_ok=True)

    # Initialize Excel workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["filepath", "letter", "distance"])

    total = len(LETTERS) * IMAGES_PER_LETTER
    count = 0

    # Generate synthetic samples
    for letter in LETTERS:
        print(f"[+] Generating for letter: {letter}")

        for i in range(IMAGES_PER_LETTER):
            img, dist, rel = generate_single_image(letter, i)

            # Save the image
            img.save(os.path.join(OUTPUT_DIR, rel))

            # Write metadata
            sheet.append([rel, letter, float(dist)])

            count += 1
            if count % 100 == 0:
                print(f"  {count}/{total} complete")

    # Write labels.xlsx
    workbook.save(os.path.join(OUTPUT_DIR, "labels.xlsx"))
    print("[✓] DONE — Dataset generated.")


# ======================================================================
# Main Execution
# ======================================================================
if __name__ == "__main__":
    generate_dataset()
