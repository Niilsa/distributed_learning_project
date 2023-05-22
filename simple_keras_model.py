import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib


import random
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    seed_everything(42)

    # Print available devices
    #print(device_lib.list_local_devices())

    # Select GPU device
    #device_name = tf.test.gpu_device_name()
    #if device_name != '/device:GPU:0':
        #print('\nGPU device not found!!!\n')

    # Ensure TensorFlow is using GPU
    #print("\n\nGPU available:", tf.test.is_gpu_available())
    print("\nGPU available, ", tf.config.list_physical_devices('GPU'))

    seed_everything(42)

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    num_classes = 10

    # Convert class labels to one-hot vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Normalize pixel values between 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Define the model architecture
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Hyperparameters --------------------------------
    batch_size = 64
    epochs = 20

    # Save the current sys.stdout
    original_stdout = sys.stdout

    # Open a file for writing
    file_stdout = open('training_log.txt', 'w')

    # Redirect sys.stdout to the file
    sys.stdout = file_stdout

    # Train the model
    start_time = time.time()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    end_time = time.time()

    # Calculate the training time
    training_time = end_time - start_time

    # Evaluate the model on the test set
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x_test, y_test, verbose=2)
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

    # Restore sys.stdout to its original value
    sys.stdout = original_stdout

    print(f"""
    accuracy = {test_accuracy:.2f},
    precision = {test_precision:.2f},
    recall = {test_recall:.2f},
    F1 score = {test_f1:.2f}
    """)
    
    # Print the training time
    print(f'training time: {training_time:.2f} seconds')

    #print(model.summary())

    # Used GPU
    #print('Used GPU: {}'.format(device_name))

    # Search for GPU device and print its name
    local_devices = device_lib.list_local_devices()
    for device in local_devices:
        if device.device_type == "GPU":
            print("GPU name:", device.physical_device_desc)
    
    print('\n\n\n\n')

main()