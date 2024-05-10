# Fine-Tuning Vision Transformer for Image Classification

This project demonstrates how to fine-tune a Vision Transformer (ViT) model for image classification. The project is structured into multiple scripts for loading data, preprocessing, model training, and plotting results.

## Project Structure

- `load_data.py`: Script for loading and preprocessing the data.
- `processing.py`: Script for data augmentation and creating TensorFlow datasets.
- `finetune_model.py`: Script for defining, compiling, and training the model.
- `plot_results.py`: Script for plotting training and validation accuracy and loss.
- `main.py`: Main script to run the entire process.

## Dataset Structure

The dataset should be organized into subdirectories for each class, as shown below:
## /path/to/images/
- class_0/
- class_1/
- class_2/

## Install the required packages:
Make sure you have Python 3.7+ and pip installed. Then, install the required packages using:

- pip install -r requirements.txt

## Set the dataset path:
Modify the DATASET_PATH variable in main.py to point to your dataset directory.
Run the main script.

## Scripts Overview
1. load_data.py
Responsible for loading and preprocessing the data. It reads images from the dataset path and splits them into training and validation sets.

2. processing.py
Handles data augmentation and creates TensorFlow datasets for training and validation.

3. finetune_model.py
Defines the Vision Transformer model, compiles it, and handles the training process. It includes callbacks for learning rate reduction, early stopping, and model checkpointing.

4. plot_results.py
Plots the training and validation accuracy and loss over epochs using matplotlib.

5. main.py
Main script to run the entire process. It invokes functions from the other scripts to load data, preprocess it, train the model, and plot the results.

## Results
The training and validation accuracy and loss are plotted after the training process is completed. The best model weights are saved as model_weights.h5.
