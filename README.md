Federated Learning with USPS and MNIST Datasets

This repository contains code for implementing federated learning using USPS and MNIST datasets. Federated learning is a machine learning approach that trains an algorithm across multiple decentralized edge devices (or servers) holding local data samples, without exchanging them. This technique enables privacy preservation and reduces the need for data centralization.
Requirements

Make sure you have the following libraries installed:

    numpy
    tensorflow
    deeplake
    scikit-learn

You can install them using pip:

bash

pip install numpy tensorflow deeplake scikit-learn

Usage

    Clone the repository:

bash

git clone https://github.com/your_username/your_repository.git

    Navigate to the repository directory:

bash

cd your_repository

    Run the code:

bash

python federated_learning.py

Description

The code implements federated learning with the following steps:

    Load MNIST and USPS datasets using Deeplake.
    Preprocess the datasets by normalizing and resizing the images.
    Split the datasets into training and validation sets.
    Define a complex convolutional neural network (CNN) model with L2 regularization.
    Fine-tune the model on each dataset separately.
    Perform federated averaging to aggregate the model weights.
    Evaluate the global model on MNIST and USPS test datasets.

File Structure

    federated_learning.py: Main script for federated learning.
    README.md: This file, providing an overview of the project.

Credits

    Author: Your Name
