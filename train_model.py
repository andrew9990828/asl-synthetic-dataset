# ======================================================================
#  train_model.py
#  Synthetic ASL Dataset Generator – Training Pipeline
#
#  Author: Andrew Bieber
#
#  Description:
#      Implements a complete supervised training routine for the SmallCNN
#      model using the synthetic ASL dataset. The model jointly learns:
#
#          • 26-class classification (letters A–Z)
#          • Regression for synthetic distance labels
#
#      Training loop features:
#          • GPU acceleration (if available)
#          • Multi-task loss (classification + regression)
#          • tqdm progress bars
#          • Model checkpoint saving
#
# ======================================================================

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ml.dataset_loader import ASLDataset
from src.model.small_cnn import SmallCNN
from src.model.train_utils import compute_loss


# ======================================================================
# Configuration
# ======================================================================

DATASET = "asl_abstract_dataset"


# ======================================================================
# train
# ======================================================================
"""
Train the SmallCNN model on the synthetic ASL dataset.

Steps:
    1. Detect GPU/CPU device.
    2. Load dataset and wrap in a DataLoader.
    3. Initialize SmallCNN and Adam optimizer.
    4. For 100 epochs:
         a. Loop over minibatches.
         b. Move data → device.
         c. Zero gradients.
         d. Forward pass: (class_logits, dist_pred).
         e. Compute multi-task loss.
         f. Backpropagate.
         g. Optimizer step.
         h. Accumulate running loss.
    5. Save model checkpoint: "asl_model.pt".

Returns:
    None
"""
def train():
    # --------------------------------------------------------------
    # Device Selection: GPU preferred if available
    # --------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------
    # Dataset + Dataloader
    # --------------------------------------------------------------
    dataset = ASLDataset(DATASET)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --------------------------------------------------------------
    # Model & Optimizer
    # --------------------------------------------------------------
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------
    for epoch in range(100):
        total_loss = 0.0

        for imgs, cls, dist in tqdm(loader):
            imgs, cls, dist = imgs.to(device), cls.to(device), dist.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            logits, dist_pred = model(imgs)

            # Multi-task loss (classification + regression)
            loss, class_loss, dist_loss = compute_loss(
                logits, cls, dist_pred, dist
            )

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")

    # --------------------------------------------------------------
    # Save Final Model Checkpoint
    # --------------------------------------------------------------
    torch.save(model.state_dict(), "asl_model.pt")
    print("Model saved: asl_model.pt")


# ======================================================================
# Main Execution
# ======================================================================
if __name__ == "__main__":
    train()
