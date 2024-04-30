# Federated Learning with USPS and MNIST Datasets

This repository contains code for implementing federated learning using USPS and MNIST datasets. Federated learning is a machine learning approach that trains an algorithm across multiple decentralized edge devices (or servers) holding local data samples, without exchanging them. This technique enables privacy preservation and reduces the need for data centralization.
Requirements

Make sure you have the following libraries installed:

    numpy
    tensorflow
    deeplake
    scikit-learn

You can install them using pip:

bash

    pip3 install numpy tensorflow deeplake scikit-learn

Usage

    git clone https://github.com/Fa221/DataHomogeneitySolution.git

bash

    git clone https://github.com/your_username/your_repository.git
    
bash

cd DataHomogeneitySolution

    python3 DataHeterogeneityFix.py
    python3 DataHeterogeneityProblem.py


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

    DataHeterogeneityFix.py: Main script for the fix to Data Heterogeneity
    DataHeterogeneityProblem.py: Script used to gauge efficacy of the fix
    README.md: This file, providing an overview of the project.

Credits

    Author: Faraz, Sahil, Rand

Links to Data Set
https://datasets.activeloop.ai/docs/ml/datasets/mnist/
https://datasets.activeloop.ai/docs/ml/datasets/usps-dataset/
