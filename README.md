# Depression Prediction with Neural Networks

This project implements a machine learning pipeline to predict depression in students using survey data. The workflow includes data preprocessing, feature engineering, model training with PyTorch, and evaluation with various metrics. The code is designed to leverage GPU acceleration (CUDA) if available.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Notes](#notes)
- [License](#license)

## Project Overview
This project aims to classify whether a student is likely to experience depression based on a variety of academic, demographic, and lifestyle factors. The model is a feedforward neural network implemented in PyTorch, with support for class imbalance and regularization.

## Dataset
- **File:** `student_depression_dataset.csv`
- **Description:** Contains survey responses from students, including demographic, academic, and lifestyle information, as well as a target label `Depression`.
- **Other Data:** Cleaned and split datasets (`X_train_cleaned.csv`, etc.) may be present for experimentation.

## Features
- **Numerical:** Age, Academic Pressure, CGPA, Study Satisfaction, Work/Study Hours, Financial Stress
- **Categorical:** Gender, City, Sleep Duration, Dietary Habits, Degree (one-hot encoded)
- **Binary:** Suicidal Thoughts, Family History of Mental Illness

## Preprocessing
- Missing values are handled by filling numerical columns with the median and categorical columns with the mode.
- Categorical variables are one-hot encoded.
- Numerical features are normalized using `StandardScaler`.
- Data is split into training, development (validation), and test sets.

## Model Architecture
- **Type:** Feedforward Neural Network (Multi-Layer Perceptron)
- **Layers:**
  - Input layer (size = number of features)
  - Two hidden layers (default sizes: 32 and 16 neurons)
  - Dropout for regularization
  - Output layer (size = number of classes)
- **Activation:** ReLU
- **Loss:** CrossEntropyLoss (with class weights for imbalance)
- **Optimizer:** Adam (with optional L2 regularization)

## Training
- Supports GPU acceleration (CUDA) if available.
- Batch training with configurable batch size (default: 128).
- Tracks loss and accuracy per epoch.
- Early stopping and learning rate scheduling can be added for further improvement.

## Evaluation
- Reports accuracy, precision, recall, and F1-score on training, development, and test sets.
- Handles class imbalance via weighted loss.

## Usage
1. **Install requirements:**
   ```bash
   pip install torch pandas scikit-learn numpy
   ```
2. **Place the dataset** (`student_depression_dataset.csv`) in the project directory.
3. **Run the main script:**
   ```bash
   python 478_Project.py
   ```
4. **(Optional) Use GPU:**
   If you have a CUDA-capable GPU and the correct drivers, the script will automatically use it.

## Requirements
- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- numpy

## Notes
- The model and preprocessing steps can be easily adapted for other tabular classification tasks.
- For best results, tune hyperparameters and experiment with feature engineering.
- The code is modular and well-commented for educational purposes.

## License
This project is for educational and research purposes. Please cite appropriately if used in academic work.
