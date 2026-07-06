# CNN Age Group Classification on UTKFace

A convolutional neural network built and evaluated from scratch to
classify face images into two age groups, using a filtered subset of
the UTKFace dataset (male, Asian faces, ages 1–60).

## Problem Statement

Given a face image, predict which of two age groups the person belongs
to:

- **Class 0 (1–29 years)**
- **Class 1 (30–60 years)**

The project focuses on a single demographic subset (male, Asian) to
keep the classification task well-defined and the dataset balanced
(capped at 500 images per class), while covering the full CNN
development workflow: data preparation, architecture experimentation,
hyperparameter tuning, final model evaluation, and interpretability.

## Dataset

- **Source:** [UTKFace — Aligned & Cropped Faces](https://susanqq.github.io/UTKFace/)
- Labels (age, gender, race) are encoded directly in each filename:
  `age_gender_race_datestamp.jpg`
- Filtered to age 1–60, gender = male, race = Asian, then balanced by
  capping each age-group class at 500 images
- Images preprocessed to 64×64 grayscale before training

> The dataset is not included in this repo. Download UTKFace from the
> link above and update the dataset path at the top of the notebook
> (originally set up for Google Colab + Google Drive) to point to your
> local copy.

## Approach

1. **Data preparation** — filename parsing to extract age/gender/race,
   demographic filtering, class balancing, 64×64 grayscale
   preprocessing, and light data augmentation (`ImageDataGenerator`)
2. **Train/validation/test split** — 70/15/15, stratified
3. **Architecture exploration** — three CNN designs compared:
   - Baseline CNN
   - Deeper CNN (three convolutional blocks)
   - Deeper CNN + Batch Normalisation + Dropout
4. **Hyperparameter tuning** — convolution kernel size (3×3, 5×5, 7×7)
   tested on the best-performing architecture
5. **Final model evaluation** — confusion matrix, classification
   report, and per-class metrics on the held-out test set
6. **Interpretability** — gradient-based saliency maps to visualise
   which regions of a face the model relies on for its predictions,
   including a look at misclassified examples

## Results Summary

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Baseline CNN | ~0.77 | 0.44 |
| Deeper CNN (Model 2) | ~0.85 | 0.37 |
| Deeper CNN + BatchNorm + Dropout (Model 3) | ~0.68 | 0.62 |
| **Final model (Deeper CNN, 3×3 kernel)** | **0.813** | **0.40** |

The deeper CNN with 3×3 kernels was selected as the final model,
offering the best balance of accuracy and generalisation across the
architecture and kernel-size experiments. Full discussion of each
result, including training/validation curves and the reasoning behind
each architectural choice, is in the notebook.

## Repository Structure

```
├── cnn_age_classification_utkface.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/cnn-age-classification-utkface.git
cd cnn-age-classification-utkface
pip install -r requirements.txt
```

The notebook was originally developed in Google Colab with the dataset
mounted from Google Drive. To run it locally:
- Download and unzip UTKFace
- Replace the Colab drive-mount cell with a local path to the
  extracted `UTKFace/` folder
- Launch with:

```bash
jupyter notebook cnn_age_classification_utkface.ipynb
```

## Tech Stack

Python · TensorFlow / Keras · OpenCV · scikit-learn · pandas · NumPy ·
matplotlib
