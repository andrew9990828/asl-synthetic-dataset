# ======================================================================
#  metrics.py
#  Synthetic ASL Dataset Generator – Evaluation Metrics
#
#  Author: Andrew Bieber
#
#  Description:
#      Defines reusable evaluation metrics for training and validation.
#      Includes:
#         • Classification accuracy
#         • Mean Absolute Error (MAE)
#         • Mean Squared Error (MSE)
#
#      This module ensures that training pipelines and experiments use
#      consistent, centralized metrics. It also provides a clean API for
#      logging or future metric expansion.
# ======================================================================

import torch


# ----------------------------------------------------------------------
# accuracy
# ----------------------------------------------------------------------
"""
Compute classification accuracy for predicted class logits.

Args:
    preds (Tensor):
        Model output of shape (batch_size, num_classes).
    targets (Tensor):
        Ground-truth class indices of shape (batch_size,).

Returns:
    float:
        Accuracy in range [0.0, 1.0].

Logic:
    - Take argmax over class dimension.
    - Compare predicted class vs. true label.
    - Compute ratio of correct predictions.
"""
def accuracy(preds, targets):
    predicted_classes = preds.argmax(dim=1)
    correct = (predicted_classes == targets).sum().item()
    total = targets.numel()
    return correct / total


# ----------------------------------------------------------------------
# mean_absolute_error
# ----------------------------------------------------------------------
"""
Compute Mean Absolute Error (MAE) for regression outputs.

Args:
    preds (Tensor):
        Model regression predictions, shape (batch_size, 1).
    targets (Tensor):
        Ground-truth regression values, shape (batch_size, 1).

Returns:
    float:
        Mean absolute error.

Logic:
    - abs(pred - target)
    - mean over batch dimension
"""
def mean_absolute_error(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()


# ----------------------------------------------------------------------
# mean_squared_error
# ----------------------------------------------------------------------
"""
Compute Mean Squared Error (MSE) for regression outputs.

Args:
    preds (Tensor):
        Model regression predictions, shape (batch_size, 1).
    targets (Tensor):
        Ground-truth regression values.

Returns:
    float:
        Mean squared error.

Logic:
    - (pred - target)^2
    - mean over batch dimension
"""
def mean_squared_error(preds, targets):
    return torch.mean((preds - targets) ** 2).item()


# ----------------------------------------------------------------------
# combined_metrics
# ----------------------------------------------------------------------
"""
Utility shorthand to compute all relevant metrics for a batch.

Args:
    class_preds (Tensor):
        Logits or class scores from classification head.
    class_targets (Tensor):
        Ground-truth class indices.

    dist_preds (Tensor):
        Regression predictions.
    dist_targets (Tensor):
        Ground-truth distances.

Returns:
    dict:
        {
            "accuracy": float,
            "mae": float,
            "mse": float
        }

Purpose:
    Used during training/validation loops to streamline logging.
"""
def combined_metrics(class_preds, class_targets, dist_preds, dist_targets):
    return {
        "accuracy": accuracy(class_preds, class_targets),
        "mae": mean_absolute_error(dist_preds, dist_targets),
        "mse": mean_squared_error(dist_preds, dist_targets),
    }
