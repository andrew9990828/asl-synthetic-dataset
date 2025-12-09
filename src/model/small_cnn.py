# ======================================================================
#  small_cnn.py
#  Synthetic ASL Dataset Generator – Lightweight CNN Architecture
#
#  Author: Andrew Bieber
#
#  Description:
#      Defines a compact convolutional neural network (CNN) for:
#        • 26-class classification (A–Z)
#        • 1D regression (distance prediction)
#
#      The network is intentionally lightweight to allow fast training on
#      standard GPUs/CPUs (e.g., Rosie supercomputer). It contains a shared
#      convolutional feature extractor followed by two independent heads:
#          - class_head : outputs class logits
#          - dist_head  : outputs a single continuous value
#
# ======================================================================

import torch
import torch.nn as nn


# ======================================================================
#  SmallCNN
# ======================================================================
"""
A small dual-head CNN used for both classification and regression tasks.

Architecture Overview:
    Input: 3×128×128 RGB image

    Feature extractor:
        • Conv(3 → 32) + ReLU
        • MaxPool(2)
        • Conv(32 → 64) + ReLU
        • MaxPool(2)
        • Conv(64 → 128) + ReLU
        • MaxPool(2)
        Output shape: (128, 16, 16)

    Flatten:
        128 * 16 * 16 = 32768 features

    class_head:
        • Linear → ReLU
        • Linear → 26 class logits

    dist_head:
        • Linear → ReLU
        • Linear → 1 regression value

Forward Output:
    (class_logits, dist_value)
"""
class SmallCNN(nn.Module):

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------
    """
    Initializes convolutional layers and two output heads.

    Components:
        1. features (nn.Sequential):
            Shared convolutional backbone producing encoded image features.
        2. class_head (nn.Sequential):
            Predicts class logits for A–Z.
        3. dist_head (nn.Sequential):
            Predicts a float regression value.

    flat_dim:
        Computed as number of output features after conv layers.
        (128 channels × 16 × 16 spatial dimensions)
    """
    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------
        # Shared CNN Feature Extractor
        # --------------------------------------------------------------
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 3→32 channels, 128→64

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 64→32

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)   # 32→16
        )

        # Output dims after 3 max pools: 128 × 16 × 16
        self.flat_dim = 128 * 16 * 16

        # --------------------------------------------------------------
        # Classification Head (multiclass)
        # --------------------------------------------------------------
        self.class_head = nn.Sequential(
            nn.Linear(self.flat_dim, 256), nn.ReLU(),
            nn.Linear(256, 26)  # A–Z
        )

        # --------------------------------------------------------------
        # Regression Head (distance)
        # --------------------------------------------------------------
        self.dist_head = nn.Sequential(
            nn.Linear(self.flat_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)  # continuous distance prediction
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    """
    Forward pass for the CNN.

    Args:
        x (Tensor): Input batch of images shaped (B, 3, 128, 128).

    Returns:
        tuple:
            class_logits (Tensor): Shape (B, 26)
            dist (Tensor): Shape (B, 1)

    Forward Steps:
        1. Apply convolutional features.
        2. Flatten tensor for fully-connected layers.
        3. Pass through classification and regression heads separately.
    """
    def forward(self, x):
        x = self.features(x)              # Convolutional feature map
        x = x.view(x.size(0), -1)         # Flatten to (B, flat_dim)

        class_logits = self.class_head(x) # Predict 26 classes
        dist = self.dist_head(x)          # Predict distance

        return class_logits, dist
