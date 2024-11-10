# README for Skin Disease Classification Model Training Script

## Overview
This script trains a deep learning model for skin disease classification on a dataset of categorized skin disease images. The model is built on **MobileNetV2**, an efficient and lightweight architecture ideal for resource-limited environments. Key features include **data augmentation**, **mixed precision training**, and multiple **callbacks** to enhance performance.

You can find the model and code here:
[Kaggle - Skin Disease Model](https://www.kaggle.com/code/sanskarsri2004/skin-disease-model)

The dataset used is available at:
[Kaggle - Skin Diseases Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset/data)

## Key Components

1. **Data Augmentation**: 
   - Utilizes `ImageDataGenerator` with augmentations (shear, zoom, flip, rotation, brightness adjustment) to improve generalization and reduce overfitting.
   - Includes a validation split of 20%.

2. **MobileNetV2 Base Model**:
   - **MobileNetV2** is used for transfer learning with pretrained weights, omitting the top layer (`include_top=False`).
   - The base model is initially frozen for early training epochs.

3. **Model Architecture**:
   - Sequential architecture includes **MobileNetV2** followed by a global average pooling layer, a dense layer with 128 units, dropout, and a softmax layer for class probabilities.

4. **Training Configuration**:
   - Optimizer: **Adam** with a learning rate of 1e-4.
   - Loss function: `categorical_crossentropy`.
   - Metrics: `accuracy`.

5. **Callbacks**:
   - **EarlyStopping** to halt training if `val_loss` shows no improvement.
   - **ReduceLROnPlateau** to lower the learning rate when `val_loss` stagnates.
   - **ModelCheckpoint** to save the best-performing model based on `val_loss`.

6. **Mixed Precision Training**:
   - `mixed_float16` precision accelerates training by optimizing GPU memory usage on Kaggle.

## Usage Instructions

- Adjust dataset paths to match your Kaggle setup.
- Ensure MobileNetV2 weights are available in the specified path.
- Running the script in a Kaggle notebook will:
  - Load and preprocess data.
  - Train the model with callbacks.
  - Save the model as `skin_disease_model_mobilenet.h5` for later use.

## Expected Output

Outputs include:
- Training and validation accuracy and loss per epoch.
- Saved model file `skin_disease_model_mobilenet.h5`.
- Intermediate checkpoints saved as `best_model.keras`.

## Notes

- Modify `batch_size` and `epochs` as needed based on resources and training time.
- For higher accuracy, consider unfreezing additional layers for fine-tuning.
