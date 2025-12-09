🚀 ASL Abstract Synthetic Dataset + CNN Pipeline

Author: Andrew Bieber
Focus Areas: Synthetic Data • Deep Learning • Computer Vision • Procedural Generation

⭐ Overview

This project generates a fully synthetic ASL (American Sign Language) dataset using procedural graphics, geometric variation, and heavy image augmentations — then trains a compact dual-head CNN to:

classify letters (A–Z)

estimate a synthetic distance scalar

You get:

✅ A synthetic dataset generator

✅ A PyTorch dataset loader

✅ A compact dual-head CNN

✅ A full end-to-end training loop

✅ Jupyter notebooks for visualization & debugging

This project is designed as a clean, reproducible ML pipeline for experimentation, learning, and research.

🚀 Quick Start Guide
1️⃣ Clone the repository
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


This creates:

asl_abstract_dataset/
    A/
        A_00000.png
        A_00001.png
    B/
        ...
labels.xlsx


Dataset size depends on IMAGES_PER_LETTER inside generate_dataset.py.

5️⃣ Train the CNN model
python train_model.py


You will see epoch loss outputs.

The final trained model is stored as:

asl_model.pt

6️⃣ Explore the dataset & model inside Jupyter

Launch Jupyter:

jupyter notebook


Open:

notebooks/explore_dataset.ipynb

7️⃣ Visualize embeddings (t-SNE, distance clusters)

Open:

notebooks/inspect_model.ipynb
notebooks/visualize_embeddings.ipynb


These notebooks include:

Feature map visualization

Filter visualization

t-SNE embeddings

Distance vs class separation plots

Debugging tools for CNN representations

📦 Project Structure
asl-synthetic-dataset/
│
├── asl_abstract_dataset/         
│
├── notebooks/
│   ├── explore_dataset.ipynb
│   ├── inspect_model.ipynb
│   └── visualize_embeddings.ipynb
│
├── src/
│   ├── dataset/
│   │   ├── augmentations.py
│   │   ├── shapes.py
│   │   └── dataset_loader.py
│   │
│   ├── ml/
│   │   ├── metrics.py
│   │   └── train_utils.py
│   │
│   ├── model/
│       └── small_cnn.py
│
├── generate_dataset.py
├── train_model.py
├── requirements.txt
└── README.md

🧠 How the Pipeline Works
1. Procedural Graphics

Each letter A–Z corresponds to a unique procedural pattern family:

line clusters

arcs

rectangular pillars

radial spokes

Each generated image includes randomization via:

rotation

blur

perlin-like noise

background jitter

geometric variation

distance scalar (controls scale & distortion)

2. Dual-Head CNN

The model outputs:

Letter class (26-way softmax)

Distance regression scalar

Loss function:

CE(class) + 0.25 * MSE(distance)

3. Fully Reproducible Training

A clean ~15-line training loop handles:

data loading

backprop

optimizer step

loss logging

model saving

Everything is deterministic when using the same seed.

📌 Notes for Researchers

This repository is ideal for:

experimenting with synthetic data

studying CNN feature extraction

embedding visualization

reproducible ML demos

curriculum teaching material

Possible extensions:

Vision Transformers

Contrastive learning

Latent-space clustering

Memory-augmented models

Variational shape priors

Procedural graphics expansion

🎉 Final Notes

If you'd like to extend the dataset, add new shape generators, or improve the CNN,
feel free to open a PR!

Happy building & exploring 👋
— Andrew