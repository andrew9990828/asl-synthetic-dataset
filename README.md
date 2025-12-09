# ASL Abstract Synthetic Dataset + CNN Pipeline

**Author:** Andrew Bieber  
**Focus Areas:** Synthetic Data • Deep Learning • Computer Vision • Procedural Generation

---

## ⭐ Overview

This project generates a **fully synthetic ASL (American Sign Language) dataset** using procedural graphics, geometric variation, and heavy image augmentations — then trains a compact dual-head CNN to classify letters and estimate a synthetic distance scalar.

You get:

- A synthetic dataset generator  
- A PyTorch dataset loader  
- A compact dual-head CNN  
- A full training loop  
- Jupyter notebooks for dataset exploration + embedding visualization  

This project demonstrates modern dataset engineering and end-to-end ML experimentation in a clean, reproducible pipeline.

---

# 🚀 Quick Start Guide

---

## 1️⃣ Clone the repository


git clone https://github.com/andrew9990828/asl-synthetic-dataset.git
cd asl-synthetic-dataset

2️⃣ Create and activate a virtual environment
Windows:

python -m venv venv
venv\Scripts\activate

Mac/Linux:

python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Generate the synthetic dataset

python generate_dataset.py

This will create:

asl_abstract_dataset/
    A/
        A_00000.png
        A_00001.png
        ...
    B/
        ...
    labels.xlsx
Total size depends on IMAGES_PER_LETTER inside generate_dataset.py.

5️⃣ Train the CNN model

python train_model.py
A progress bar + loss per epoch will display.

The final model is saved as:
asl_model.pt

6️⃣ Explore the generated dataset (Jupyter notebook)
jupyter notebook
Then open:

notebooks/explore_dataset.ipynb

7️⃣ Visualize embeddings (TSNE)
Once the model is trained:

notebooks/visualize_embeddings.ipynb
This notebook will later be expanded to include:

TSNE projections

Distance vs. class separation plots

Embedding space debugging

📦 Project Structure
graphql

AI_DATASET/
│
├── asl_abstract_dataset/         # Generated images + labels
│
├── notebooks/
│   ├── explore_dataset.ipynb     # View sample images
│   └── visualize_embeddings.ipynb # TSNE placeholder
│
├── src/
│   ├── dataset/
│   │   ├── augmentations.py      # Rotation, blur, noise, background jitter
│   │   ├── shapes.py             # Procedural shape generation
│   │   └── dataset_loader.py     # PyTorch ASLDataset loader
│   │
│   ├── ml/
│   │   ├── metrics.py            # Custom metrics (WIP)
│   │   └── train_utils.py        # Loss function
│   │
│   ├── model/
│       └── small_cnn.py          # CNN with class + scalar regression heads
│
├── generate_dataset.py           # Synthetic dataset generator script
├── train_model.py                # Training loop
├── requirements.txt
└── README.md

🧠 How the Pipeline Works
1. Procedural Graphics
Each letter A–Z is mapped to a pattern style:

line clusters
arcs
rectangular pillars
radial spokes

Each sample is randomized by:

rotation
blur
Perlin-like noise
background jitter
scale proportional to a synthetic “distance" label

2. Dual-Head CNN
The model predicts:

letter class (26-way classifier)
distance scalar (regression)
Loss = CE(class) + 0.25*MSE(distance)

3. Fully Reproducible Training
With only 15 lines of training loop code — clean, minimal, and transparent.

📌 Notes for Researchers
This repo is ideal for:

experimenting with synthetic data generation
training lightweight CNNs
embedding-space visualization
building reproducible ML demos
teaching ML fundamentals

You can expand this pipeline into:

a vision transformer
contrastive learning
latent-space clustering
memory-augmented models

🎉 Final Notes
If you improve this dataset, add shapes, or upgrade the model — please feel free to submit a PR or open a discussion.

Happy building!