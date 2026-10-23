# CrossWave: Cross-Wavelength Knowledge Distillation for Reduced-Sensing Microscopy

This repository contains the code for **CrossWave**, a cross-wavelength knowledge-distillation framework for reduced-sensing microscopy.

CrossWave is designed for microscopy tasks where multiple wavelength measurements are available during model development, but only a reduced wavelength configuration is desired at deployment. A multi-wavelength fusion model is first trained as a teacher. Then, a single-wavelength student model is trained using supervised target loss together with prediction-level and feature-level distillation from the fusion teacher. At inference time, the student requires only one wavelength input.

## Overview

In the case study included in this project, the goal is to predict particle-level Cy3 fluorescence intensity from label-free iSCAT image patches collected at different illumination wavelengths.

The main workflow includes:

1. Data preprocessing and particle-level dataset construction
2. Multi-wavelength fusion teacher model training
3. CrossWave student model training through knowledge distillation
4. Model evaluation and statistical comparison
5. Latent-space visualization by intensity bins

## Repository structure

```text
CrossWave_Public/
│
├── configs/                 # Configuration files, if applicable
├── datasets/                # PyTorch dataset classes
├── experiments/             # Training, tuning, evaluation, and visualization scripts
├── models/                  # CNN regression models
├── preprocessing/           # Dataset preprocessing and construction scripts
├── training/                # Training utilities and sampling strategies
├── utils/                   # Utility functions
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
