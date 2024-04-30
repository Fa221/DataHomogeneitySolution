import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
import deeplake
from scipy.stats import skew

# Model Architecture
def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    return model

# Preprocess of the data
def preprocess(images, image_shape):
    if images.ndim == 3:
        images = images[..., np.newaxis]
    images = images.astype('float32') / 255.0
    images = np.array([tf.image.resize(img, image_shape).numpy() for img in images])
    return images

# Load the MNIST and USPS datasets
mnist_train_ds = deeplake.load("hub://activeloop/mnist-train")
mnist_train_labels = np.array(mnist_train_ds['labels'].numpy()).flatten()
usps_train_ds = deeplake.load("hub://activeloop/usps-train")
usps_train_labels = np.array(usps_train_ds['labels'].numpy()).flatten()

# Send the data to be Preprocessed
mnist_data = preprocess(mnist_train_ds['images'].numpy(), (28, 28))
usps_data = preprocess(usps_train_ds['images'].numpy(), (28, 28))

# Local client training
def train_on_client(data, labels, model, epochs=1):
    model.fit(data, labels, epochs=epochs, verbose=1)
    return model

# Calculate skewness for both datasets
mnist_skewness = skew(mnist_train_labels)
usps_skewness = skew(usps_train_labels)
client_skewness = np.array([mnist_skewness, usps_skewness])

# Train local models
num_clients = 2
client_data = [(mnist_data, mnist_train_labels), (usps_data, usps_train_labels)]
global_model = build_model()
client_models = []
for data, labels in client_data:
    client_model = clone_model(global_model)
    client_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    client_model = train_on_client(data, labels, client_model)
    client_models.append(client_model)

# Calculate client weights based skew
def calculate_client_weights(skewness):
    inverse_skewness = 1 / np.abs(skewness)
    normalized_weights = inverse_skewness / np.sum(inverse_skewness)
    return normalized_weights

client_weights = calculate_client_weights(client_skewness)

# Federated aggregation
def federated_aggregation(client_models, client_weights):
    aggregated_weights = client_models[0].get_weights()
    for i in range(len(aggregated_weights)):
        aggregated_weights[i] = np.zeros_like(aggregated_weights[i])
    for client_model, weight in zip(client_models, client_weights):
        client_weights = client_model.get_weights()
        for i, layer_weights in enumerate(client_weights):
            aggregated_weights[i] += layer_weights * weight
    return aggregated_weights

# Perform federated aggregation
aggregated_weights = federated_aggregation(client_models, client_weights)

# Update the global model weights and compile it before evaluation
global_model.set_weights(aggregated_weights)
global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#-----------------------------------------------------------------------------------------------------------
# Now that training is done begin the evaluation process!
#-----------------------------------------------------------------------------------------------------------

# Load and preprocess test datasets
mnist_test_ds = deeplake.load("hub://activeloop/mnist-test")
mnist_test_data = preprocess(mnist_test_ds['images'].numpy(), (28, 28))
mnist_test_labels = np.array(mnist_test_ds['labels'].numpy()).flatten()
usps_test_ds = deeplake.load("hub://activeloop/usps-test")
usps_test_data = preprocess(usps_test_ds['images'].numpy(), (28, 28))
usps_test_labels = np.array(usps_test_ds['labels'].numpy()).flatten()

# Evaluate
mnist_test_loss, mnist_test_accuracy = global_model.evaluate(mnist_test_data, mnist_test_labels, verbose=1)
usps_test_loss, usps_test_accuracy = global_model.evaluate(usps_test_data, usps_test_labels, verbose=1)
print("MNIST Test Accuracy:", mnist_test_accuracy)
print("USPS Test Accuracy:", usps_test_accuracy)
