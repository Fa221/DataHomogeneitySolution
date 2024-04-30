import numpy as np
import tensorflow as tf
import deeplake
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
REGULARIZATION = 1e-4

# New and improved augmentation
def advanced_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image

# Preprocess data with normalization and resizing
def enhanced_preprocess(images, image_shape):
    if images.ndim == 3:
        images = np.expand_dims(images, -1)
    images = images.astype('float32') / 255.0
    images = np.array([tf.image.resize(img, image_shape[:2]).numpy() for img in images])
    images = np.array([advanced_augmentation(img) for img in images])
    return images

# Model Architecure more complex than the other
def build_complex_model(lr=LEARNING_RATE, reg=REGULARIZATION):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(reg)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(reg)),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(reg)),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Fine Tune the model
def fine_tune_model(model, data, labels, batch_size=BATCH_SIZE, validation_data=None):
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(
        data, labels,
        batch_size=batch_size,
        epochs=EPOCHS,
        validation_data=validation_data,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )
    return model

# Calculating client weights
def calculate_client_weights(dataset_sizes):
    total_size = sum(dataset_sizes)
    return [size / total_size for size in dataset_sizes]

# Federated aggregation with dataset size-based client weights
def federated_aggregation(client_models, client_weights):
    aggregated_weights = [np.zeros_like(weights) for weights in client_models[0].get_weights()]
    for model, weight in zip(client_models, client_weights):
        client_weight = model.get_weights()
        for i in range(len(aggregated_weights)):
            aggregated_weights[i] += client_weight[i] * weight
    return aggregated_weights

# Load and split the MNIST  and USPS datasets (Training and Validation)
mnist_train_ds = deeplake.load("hub://activeloop/mnist-train")
mnist_train_labels = np.array(mnist_train_ds['labels'].numpy()).flatten()
mnist_data = enhanced_preprocess(mnist_train_ds['images'].numpy(), (28, 28, 1))
mnist_train_data, mnist_val_data, mnist_train_labels, mnist_val_labels = train_test_split(
    mnist_data, mnist_train_labels, test_size=0.2, random_state=42)
usps_train_ds = deeplake.load("hub://activeloop/usps-train")
usps_train_labels = np.array(usps_train_ds['labels'].numpy()).flatten()
usps_data = enhanced_preprocess(usps_train_ds['images'].numpy(), (28, 28, 1))
usps_train_data, usps_val_data, usps_train_labels, usps_val_labels = train_test_split(
    usps_data, usps_train_labels, test_size=0.2, random_state=42)

# Initialize and Train models for the two clients
global_model = build_complex_model()
mnist_model = fine_tune_model(clone_model(global_model), mnist_train_data, mnist_train_labels, validation_data=(mnist_val_data, mnist_val_labels))
usps_model = fine_tune_model(clone_model(global_model), usps_train_data, usps_train_labels, validation_data=(usps_val_data, usps_val_labels))

# Federated averaging
dataset_sizes = [len(mnist_train_labels), len(usps_train_labels)]
client_weights = calculate_client_weights(dataset_sizes)
client_models = [mnist_model, usps_model]
aggregated_weights = federated_aggregation(client_models, client_weights)
global_model.set_weights(aggregated_weights)

#-----------------------------------------------------------------------------------------------------------
# Now that training is done begin the evaluation process!
#-----------------------------------------------------------------------------------------------------------


# Load test datasets and evaluate the global model
mnist_test_ds = deeplake.load("hub://activeloop/mnist-test")
mnist_test_data = enhanced_preprocess(mnist_test_ds['images'].numpy(), (28, 28, 1))
mnist_test_labels = np.array(mnist_test_ds['labels'].numpy()).flatten()
usps_test_ds = deeplake.load("hub://activeloop/usps-test")
usps_test_data = enhanced_preprocess(usps_test_ds['images'].numpy(), (28, 28, 1))
usps_test_labels = np.array(usps_test_ds['labels'].numpy()).flatten()
mnist_test_loss, mnist_test_accuracy = global_model.evaluate(mnist_test_data, mnist_test_labels, verbose=1)
usps_test_loss, usps_test_accuracy = global_model.evaluate(usps_test_data, usps_test_labels, verbose=1)
print("MNIST Test Accuracy:", mnist_test_accuracy)
print("USPS Test Accuracy:", usps_test_accuracy)

