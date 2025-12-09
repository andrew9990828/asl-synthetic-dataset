"""
================================================================================
AUGMENTATION UTILITIES
Author: Andrew Bieber
Project: Synthetic Dataset Generator + Lightweight CNN Classifier
Description:
    This module provides image augmentation utilities used to introduce controlled
    variation into the synthetic dataset. These augmentations help prevent
    underfitting/overfitting by introducing stochasticity in:
        • Background colors
        • Gaussian noise
        • Optional blurring
        • Random rotations

    All functions are designed to be lightweight, deterministic when needed,
    and easily extendable for future dataset complexity.

================================================================================
PSEUDOCODE SUMMARY
--------------------------------------------------------------------------------
FUNCTION random_bg_color():
    Generate a near-white base value
    Add small jitter per RGB channel
    Clamp to [0, 255]
    RETURN (R, G, B)

FUNCTION add_noise(image, sigma):
    Convert image → numpy array
    Sample Gaussian noise with mean=0, stdev=sigma
    Add noise → clip → cast back to uint8
    Convert array back to PIL image
    RETURN noisy_image

FUNCTION maybe_blur(image):
    With 20% probability:
        Apply Gaussian blur with random radius
    Otherwise:
        Return image unchanged

FUNCTION random_rotation(image, bg_color):
    Sample random angle [0, 360)
    Rotate using bicubic interpolation
    Use bg_color to fill empty rotation areas
    RETURN rotated_image
================================================================================
"""

import random
import numpy as np
from PIL import Image, ImageFilter


# ------------------------------------------------------------------------------
# random_bg_color
# ------------------------------------------------------------------------------

def random_bg_color():
    """
    Generate a soft, near-neutral background color with mild jitter.

    Returns:
        tuple(int, int, int):
            An RGB tuple representing a lightly randomized background color.
            The values are centered near 200–255 to keep backgrounds bright
            and visually consistent across samples.

    Logic:
        - Start with a bright base value (200–255).
        - Add small random noise (±20).
        - Clamp each RGB component to remain within [0, 255].
    """
    base = random.randint(200, 255)
    jitter = 20

    return (
        max(0, min(255, base + random.randint(-jitter, jitter))),
        max(0, min(255, base + random.randint(-jitter, jitter))),
        max(0, min(255, base + random.randint(-jitter, jitter))),
    )


# ------------------------------------------------------------------------------
# add_noise
# ------------------------------------------------------------------------------

def add_noise(img, sigma=8.0):
    """
    Apply Gaussian noise to an image to increase dataset variance.

    Args:
        img (PIL.Image.Image):
            The original image to which noise will be applied.

        sigma (float):
            Standard deviation of the Gaussian noise distribution.
            Larger sigma → stronger noise.

    Returns:
        PIL.Image.Image:
            A new image with additive noise applied.

    Logic:
        - Convert PIL image → NumPy array.
        - Sample Gaussian noise (mean=0, std=sigma).
        - Add noise to the pixel array.
        - Clip values so no pixel leaves the valid [0, 255] range.
        - Convert back to uint8 and wrap into a PIL Image.
    """
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ------------------------------------------------------------------------------
# maybe_blur
# ------------------------------------------------------------------------------

def maybe_blur(img):
    """
    Apply a mild Gaussian blur with 20% probability.

    Args:
        img (PIL.Image.Image):
            Input image to (possibly) blur.

    Returns:
        PIL.Image.Image:
            Either the blurred image or the unchanged original.

    Logic:
        - With 20% probability:
            - Apply Gaussian blur with random radius (0.5 → 1.5).
        - Otherwise:
            - Return the image untouched.

    Purpose:
        Blur introduces local uncertainty and helps prevent models from
        overfitting to overly sharp synthetic strokes.
    """
    if random.random() < 0.2:
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
    return img


# ------------------------------------------------------------------------------
# random_rotation
# ------------------------------------------------------------------------------

def random_rotation(img, bg_color):
    """
    Rotate an image by a random angle while preserving the aesthetic background.

    Args:
        img (PIL.Image.Image):
            Image to rotate.

        bg_color (tuple(int, int, int)):
            Background fill color for areas revealed during rotation.

    Returns:
        PIL.Image.Image:
            A rotated version of the input image.

    Logic:
        - Choose an angle uniformly in [0, 360).
        - Apply rotation using bicubic interpolation for smoother edges.
        - Use bg_color to fill in exposed corners and minimize visual artifacts.
    """
    angle = random.uniform(0, 360)
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg_color)
