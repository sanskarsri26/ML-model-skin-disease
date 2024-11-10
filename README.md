# README for Skin Disease Classification Model Training Script

## Overview
This script trains a deep learning model for skin disease classification using TensorFlow and Keras, utilizing a dataset of categorized skin disease images. The model leverages **MobileNetV2** as a lightweight and efficient base architecture, optimized to work within resource constraints on Kaggle. The script incorporates **data augmentation**, **mixed precision training**, and **callbacks** to enhance performance and optimize training.

## Key Components

1. **Data Augmentation**: 
   - The script applies various augmentations (shear, zoom, flip, rotation, brightness adjustment) using `ImageDataGenerator`, helping to improve generalization and reduce overfitting.
   - It includes a validation split (20%) from the dataset.

2. **MobileNetV2 Base Model**:
   - The **MobileNetV2** model is used for its lightweight architecture, ideal for limited-resource environments like Kaggle.
   - Pretrained weights are loaded, excluding the top layer (`include_top=False`), allowing for transfer learning.
   - The base model is initially frozen, so only the top layers train during early epochs.

3. **Model Architecture**:
   - The architecture consists of the **MobileNetV2** base model followed by a global average pooling layer, a dense layer with 128 units, a dropout layer, and a final dense layer with softmax activation.
   - The final layer uses the number of classes from `train_generator` and outputs class probabilities.

4. **Training Configuration**:
   - The model is compiled with **Adam optimizer** and a low learning rate (1e-4).
   - Loss function: `categorical_crossentropy`, suitable for multi-class classification.
   - Metric: `accuracy` to monitor model performance.

5. **Callbacks**:
   - **EarlyStopping**: Monitors `val_loss` and stops training if no improvement is seen for 5 epochs, restoring the best model.
   - **ReduceLROnPlateau**: Reduces the learning rate if `val_loss` plateaus, helping the model converge.
   - **ModelCheckpoint**: Saves the best model based on `val_loss`.

6. **Mixed Precision Training**:
   - `mixed_float16` precision is enabled to accelerate training, making efficient use of GPU resources on Kaggle.

## Usage Instructions

- Update the dataset paths to match your Kaggle dataset structure.
- Make sure the MobileNetV2 weights are available in the specified location.
- Run the script in a Kaggle notebook, which will:
  - Load and preprocess the data.
  - Initialize and train the model.
  - Save the trained model file (`skin_disease_model_mobilenet.h5`) for future inference or further fine-tuning.

## Expected Output

The script will output:
- Training and validation accuracy and loss for each epoch.
- The final trained model saved in `skin_disease_model_mobilenet.h5`.
- Intermediate model checkpoints saved with `best_model.keras` filename.

## Notes

- Adjust the `batch_size` and `epochs` if necessary based on memory constraints and desired training duration.
- For enhanced accuracy, further fine-tuning can be conducted by unfreezing some layers of the `MobileNetV2` base model.

This README serves as a guide for understanding and modifying the script for skin disease classification on a custom dataset using a Kaggle environment.
