# Federated Learning with DenseNet201 for Skin Image Classification

## Overview

This project implements a federated learning approach to train a DenseNet201 model on a dataset of skin images categorized by Fitzpatrick scale values. The dataset is split across multiple clients, and the model is trained using a federated learning framework. The code includes data preprocessing, model training, and evaluation steps.

## Requirements

To run the code, you need the following Python packages:

- `pillow`
- `Augmentor`
- `requests`
- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `opencv-python`

You can install these packages using the following commands:

```bash
pip install pillow Augmentor requests
pip install tensorflow
pip install numpy pandas scikit-learn opencv-python
```

## Dataset

The dataset used in this project is stored in a CSV file named `fitzpatrick17k.csv`. The dataset contains URLs to skin images and their corresponding Fitzpatrick scale values and labels (non-neoplastic, malignant, benign).

## Code Structure

### Data Loading and Preprocessing:

- Load the dataset from the CSV file.
- Split the data based on Fitzpatrick scale values.
- Download images from the URLs provided in the dataset.
- Resize images and split them into training, validation, and test sets.

### Model Training:

- Use the DenseNet201 model with transfer learning.
- Train the model on the preprocessed data.
- Save the trained model for each client.

### Federated Learning:

- Aggregate the weights of the last layer from each client's model.
- Compute the average weights and update the models for the next round of training.

### Evaluation:

- Evaluate the model on the test set.
- Print the test accuracy.

## Usage

### Data Preprocessing:

- Ensure the CSV file (`fitzpatrick17k.csv`) is available at the specified path.
- Run the data loading and preprocessing steps to download and split the images.

### Model Training:

- Train the DenseNet201 model on the preprocessed data.
- Save the trained model for each client.

### Federated Learning:

- Aggregate the weights from each client's model.
- Update the models with the average weights and continue training for the next round.

### Evaluation:

- Evaluate the final model on the test set and print the accuracy.
