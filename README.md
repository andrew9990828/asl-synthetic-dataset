🚀 ASL Abstract Synthetic Dataset + CNN Pipeline

Author: Andrew Bieber
Focus Areas: Synthetic Data • Deep Learning • Computer Vision • Procedural Generation

⭐ Overview

This project generates a fully synthetic ASL (American Sign Language) dataset using procedural graphics, geometric variation, and heavy image augmentations — then trains a compact dual-head CNN to classify letters and estimate a synthetic distance scalar.

You get:

A synthetic dataset generator

A PyTorch dataset loader

A compact dual-head CNN

A full training loop

Jupyter notebooks for dataset exploration + embedding visualization

This project demonstrates modern dataset engineering and end-to-end ML experimentation in a clean, reproducible pipeline.

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


This will create:

asl_abstract_dataset/
    A/
        A_00000.png
        A_00001.png
    B/
        ...
    labels.xlsx


Total size depends on IMAGES_PER_LETTER in generate_dataset.py.

5️⃣ Train the CNN model
python train_model.py


You will see epoch loss values.

The final model is saved as:

asl_model.pt

6️⃣ Explore the generated dataset
jupyter notebook


Open:

notebooks/explore_dataset.ipynb

7️⃣ Visualize embeddings (TSNE)

Open:

notebooks/visualize_embeddings.ipynb


This notebook will later include:

TSNE projections

Distance vs class separation plots

Embedding space debugging tools

📦 Project Structure
asl-synthetic-dataset/
│
├── asl_abstract_dataset/         
│
├── notebooks/
│   ├── explore_dataset.ipynb     
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

Each letter A–Z corresponds to one procedural pattern style:

line clusters

arcs

rectangular pillars

radial spokes

Each image is randomized by:

rotation

blur

Perlin-like noise

background jitter

scale tied to a synthetic “distance” label

2. Dual-Head CNN

The model predicts:

Letter class (26-way classification)

Distance scalar (regression)

Loss:

CE(class) + 0.25 * MSE(distance)

3. Fully Reproducible Training

A ~15-line clean training script trains everything end-to-end.

📌 Notes for Researchers

This repo is ideal for:

experimenting with synthetic data

training small CNNs

embedding visualization

building reproducible ML demos

teaching fundamentals

Possible expansions:

Vision Transformers

Contrastive learning

Latent-space clustering

Memory-augmented models

🎉 Final Notes

If you improve this dataset, add shapes, or upgrade the model — feel free to submit a PR.

Happy building!