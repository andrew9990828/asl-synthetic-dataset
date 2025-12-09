# ======================================================================
#  train_utils.py
#  Synthetic ASL Dataset Generator – Training Utilities
#
#  Author: Andrew Bieber
#
#  Description:
#      Contains helper functions used during the training process,
#      including multi-task loss computation for:
#         • Classification (A–Z)
#         • Distance regression (MSE)
#
#      The final loss is a weighted sum:
#           total_loss = class_loss + 0.25 * dist_loss
#
#      Weighting prevents the regression task from dominating early
#      training while still contributing meaningful gradient updates.
# ======================================================================

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# compute_loss
# ----------------------------------------------------------------------
"""
Compute total training loss for the multi-task CNN model.

Args:
    class_logits (Tensor):
        Raw classification outputs from the model (B × 26).

    class_targets (Tensor):
        Ground-truth class labels (integer indices 0–25).

    dist_pred (Tensor):
        Distance prediction output from the model (B × 1).

    dist_targets (Tensor):
        Ground-truth continuous distance values (B × 1).

Returns:
    tuple:
        total_loss (Tensor):
            Combined multi-task loss.
        class_loss (Tensor):
            Cross-entropy classification loss.
        dist_loss (Tensor):
            Mean squared error regression loss.

Loss Structure:
    classification_loss = CrossEntropy(class_logits, class_targets)
    distance_loss      = MSE(dist_pred, dist_targets)

    total_loss = classification_loss + 0.25 * distance_loss

Pseudocode:
    class_loss = cross_entropy(class_logits, class_targets)
    dist_loss  = mse(dist_pred, dist_targets)
    total_loss = class_loss + (0.25 * dist_loss)
    return total_loss, class_loss, dist_loss
"""
def compute_loss(class_logits, class_targets, dist_pred, dist_targets):
    class_loss = F.cross_entropy(class_logits, class_targets)
    dist_loss = F.mse_loss(dist_pred, dist_targets)

    # Weighted sum: classification dominates, regression is auxiliary
    return class_loss + dist_loss * 0.25, class_loss, dist_loss
